#!/usr/bin/env python
"""Script to index FormField full_text into ChromaDB manually."""

import argparse
import logging
import os
import sys
import time # Import time module
from typing import List, Sequence

import chromadb
# Import the specific exception for rate limiting
from langchain_google_genai._common import GoogleGenerativeAIError
from sqlalchemy.orm import Session, joinedload
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_tax_agent.settings import settings
from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import FormInstruction, FormField
# Import the shared function that returns the raw LangChain embedder
from ai_tax_agent.llm_utils import get_embedding_function

logger = logging.getLogger(__name__)

# --- Constants ---
COLLECTION_NAME = "form_instructions"
DEFAULT_BATCH_SIZE = 100
CHROMA_DATA_PATH = "chroma_data" # Ensure this path is correct relative to project root
RATE_LIMIT_DELAY = 60 # Increased delay for embedding API retries

def parse_arguments():
    parser = argparse.ArgumentParser(description="Index form instruction HTML content into ChromaDB.")
    parser.add_argument("--clear", action="store_true", help="Clear the existing collection before indexing.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of chunks to batch process before upserting to ChromaDB.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    return parser.parse_args()

def get_chroma_client() -> chromadb.Client:
    """Initializes and returns a persistent ChromaDB client."""
    # Ensure the path is absolute or correctly relative from execution location
    absolute_chroma_path = os.path.join(project_root, CHROMA_DATA_PATH)
    logger.info(f"Initializing ChromaDB client at: {absolute_chroma_path}")
    try:
        # Use the corrected path
        return chromadb.PersistentClient(path=absolute_chroma_path)
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client at {absolute_chroma_path}: {e}", exc_info=True)
        sys.exit(1)

def get_form_fields_to_index(session: Session) -> Sequence[FormField]:
    """Fetches all FormField records with non-empty full_text, joining FormInstruction."""
    logger.info("Fetching form fields and associated instructions from database...")
    form_fields = (
        session.query(FormField)
        .join(FormInstruction, FormField.instruction_id == FormInstruction.id)
        .options(joinedload(FormField.instruction)) # Eager load instruction data
        .filter(FormField.full_text != None, FormField.full_text != "")
        .order_by(FormField.id)
        .all()
    )
    logger.info(f"Found {len(form_fields)} form fields with text content to index.")
    return form_fields

def main():
    args = parse_arguments()
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger.setLevel(log_level)

    logger.info("Initializing ChromaDB client and embedding function...")
    try:
        chroma_client = get_chroma_client()
        # Get the raw LangChain embedder instance
        embedding_function_instance = get_embedding_function()
        if not embedding_function_instance:
             logger.error("Failed to initialize embedding function. Exiting.")
             sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB or embedding function: {e}", exc_info=True)
        sys.exit(1)

    if args.clear:
        logger.warning(f"Clearing existing collection: {COLLECTION_NAME}")
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
        except Exception as e:
            logger.warning(f"Could not delete collection (may not exist): {e}")

    logger.info(f"Getting or creating ChromaDB collection: {COLLECTION_NAME}")
    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None, # Tell ChromaDB we will handle embeddings
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Failed to get or create ChromaDB collection: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Getting database session...")
    db: Session | None = None
    try:
        # Use context manager for session handling if get_session supports it,
        # otherwise, manage manually with try/finally.
        # Assuming get_session provides a context manager or generator:
        session_context = get_session()
        db = next(session_context) # For generator-based session
    except Exception as e:
        logger.error(f"Failed to get database session using get_session(): {e}", exc_info=True)
        sys.exit(1)

    if not db:
        logger.error("Database session object is None or invalid.")
        sys.exit(1)

    try: # Wrap DB operations in try/finally to ensure session closure
        form_fields_to_index = get_form_fields_to_index(db)

        if not form_fields_to_index:
            logger.info("No form fields with text content found to index.")
            sys.exit(0)

        logger.info(f"Starting indexing process for {len(form_fields_to_index)} form fields...")

        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        total_items_processed = 0
        upserted_count = 0

        for form_field in tqdm(form_fields_to_index, desc="Processing Form Fields"):
            total_items_processed += 1
            if not form_field.full_text or not form_field.instruction:
                logger.warning(f"Skipping form field ID {form_field.id} due to missing text or instruction link.")
                continue

            document = form_field.full_text
            doc_id = f"field_{form_field.id}"
            metadata = {
                "field_id": form_field.id,
                "field_label": form_field.field_label or "",
                "form_number": form_field.instruction.form_number or "",
                "form_title": form_field.instruction.title or "",
                "instruction_id": form_field.instruction_id,
                "text_length": len(document)
            }

            # Simple validation for ChromaDB compatibility
            for k, v in metadata.items():
                if not isinstance(v, (str, int, float, bool)):
                    logger.debug(f"Metadata key '{k}' has incompatible type '{type(v)}' for doc_id {doc_id}. Converting to string.")
                    metadata[k] = str(v)

            batch_ids.append(doc_id)
            batch_documents.append(document)
            batch_metadatas.append(metadata)

            # Process batch when full
            if len(batch_ids) >= args.batch_size:
                # --- Manual Embedding --- #
                batch_embeddings: List[List[float]] | None = None
                embed_retries = 3
                while embed_retries > 0:
                    try:
                        batch_embeddings = embedding_function_instance.embed_documents(batch_documents)
                        logger.debug(f"Successfully embedded batch ending with ID {batch_ids[-1]}")
                        break
                    except GoogleGenerativeAIError as e:
                        if "rate limit" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                            embed_retries -= 1
                            wait_time = RATE_LIMIT_DELAY * (4 - embed_retries)
                            logger.warning(f"Rate limit hit embedding batch. Retrying in {wait_time}s... ({embed_retries} retries left)")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Non-rate-limit Google API error embedding batch: {e}", exc_info=True)
                            embed_retries = 0
                    except Exception as e:
                        logger.error(f"Unexpected error embedding batch: {e}", exc_info=True)
                        embed_retries = 0

                if not batch_embeddings:
                    logger.error(f"Failed to embed batch ending {batch_ids[-1]} after retries. Skipping upsert.")
                else:
                    # --- Upsert with Embeddings --- #
                    try:
                        # Optional: Pre-check can be added here too if needed
                        collection.upsert(
                            ids=batch_ids,
                            documents=batch_documents,
                            metadatas=batch_metadatas,
                            embeddings=batch_embeddings
                        )
                        upserted_count += len(batch_ids)
                        logger.debug(f"Upserted batch of {len(batch_ids)} chunks successfully.")
                        time.sleep(0.1) # Small delay
                    except Exception as e:
                        logger.error(f"Error upserting batch to ChromaDB: {e}", exc_info=True)
                        # Decide if retry is needed for upsert; for now, just log and continue

                # Clear batch lists after processing
                batch_ids.clear()
                batch_documents.clear()
                batch_metadatas.clear()

        # Process the final partial batch
        if batch_ids:
            # --- Manual Embedding (Final Batch) --- #
            batch_embeddings = None
            embed_retries = 3
            while embed_retries > 0:
                try:
                    batch_embeddings = embedding_function_instance.embed_documents(batch_documents)
                    logger.debug(f"Successfully embedded final batch of {len(batch_ids)} chunks.")
                    break
                except GoogleGenerativeAIError as e:
                     if "rate limit" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                        embed_retries -= 1
                        wait_time = RATE_LIMIT_DELAY * (4 - embed_retries)
                        logger.warning(f"Rate limit hit embedding final batch. Retrying in {wait_time}s... ({embed_retries} retries left)")
                        time.sleep(wait_time)
                     else:
                        logger.error(f"Non-rate-limit Google API error embedding final batch: {e}", exc_info=True)
                        embed_retries = 0
                except Exception as e:
                    logger.error(f"Unexpected error embedding final batch: {e}", exc_info=True)
                    embed_retries = 0

            if not batch_embeddings:
                logger.error("Failed to embed final batch after retries. Skipping final upsert.")
            else:
                # --- Upsert with Embeddings (Final Batch) --- #
                try:
                    collection.upsert(
                        ids=batch_ids,
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        embeddings=batch_embeddings
                    )
                    upserted_count += len(batch_ids)
                    logger.info(f"Upserted final batch of {len(batch_ids)} chunks successfully.")
                except Exception as e:
                    logger.error(f"Error upserting final batch to ChromaDB: {e}", exc_info=True)

    finally:
        # Ensure the database session is closed
        if db:
            logger.info("Closing database session.")
            db.close()

    logger.info(f"--- Indexing Complete ---")
    logger.info(f"Total form fields processed: {total_items_processed}")
    logger.info(f"Total documents successfully embedded and upserted: {upserted_count}")
    try:
        final_count = collection.count()
        logger.info(f"Final count in collection '{COLLECTION_NAME}': {final_count}")
    except Exception as e:
        logger.warning(f"Could not get final collection count: {e}")

if __name__ == "__main__":
    main() 