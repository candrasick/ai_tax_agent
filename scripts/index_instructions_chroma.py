import argparse
import logging
import os
import sys
import time # Import time module
from typing import List, Sequence

import chromadb
# Removed: from chromadb.api.types import EmbeddingFunction
# Removed: from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
# Import the shared function
from ai_tax_agent.llm_utils import get_embedding_function

logger = logging.getLogger(__name__)

# --- Removed Embedding Function Wrapper --- #
# class LangchainEmbeddingFunctionWrapper(EmbeddingFunction):
#     ... (wrapper code removed)

# --- Constants ---
COLLECTION_NAME = "form_instructions"
DEFAULT_BATCH_SIZE = 100 # Number of chunks to process before upserting
CHROMA_DATA_PATH = "chroma_data"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Index form instruction HTML content into ChromaDB.")
    parser.add_argument("--clear", action="store_true", help="Clear the existing collection before indexing.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of chunks to batch process before upserting to ChromaDB.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    return parser.parse_args()

def get_chroma_client() -> chromadb.Client:
    """Initializes and returns a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# --- Removed local get_embedding_function --- #
# def get_embedding_function(settings):
#     ... (function code removed)

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
    # Remove call to non-existent function
    # setup_logging(level=getattr(logging, args.log_level))
    
    # Add basic logging configuration
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger.setLevel(log_level) # Ensure the script's logger level is also set

    logger.info("Initializing ChromaDB client and embedding function...")
    try:
        chroma_client = get_chroma_client()
        # Get the LangChain embedder instance directly from llm_utils
        embedding_function_instance = get_embedding_function()
        if not embedding_function_instance:
             logger.error("Failed to initialize embedding function. Exiting.")
             sys.exit(1)
        # Removed wrapper creation
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB or embedding function: {e}", exc_info=True) # Added exc_info
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
            # Pass the LangChain instance directly
            embedding_function=embedding_function_instance,
            metadata={"hnsw:space": "cosine"} # Example metadata: Use cosine distance
        )
    except Exception as e:
        logger.error(f"Failed to get or create ChromaDB collection: {e}", exc_info=True) # Added exc_info
        sys.exit(1)

    logger.info("Getting database session...")
    # Call get_session() directly
    try:
        db: Session = get_session()
    except Exception as e:
        logger.error(f"Failed to get database session using get_session(): {e}", exc_info=True)
        sys.exit(1)

    if not db:
        logger.error("Database session object is None or invalid after calling get_session().")
        sys.exit(1)

    form_fields_to_index = get_form_fields_to_index(db)

    if not form_fields_to_index:
        logger.info("No form fields with text content found to index.")
        sys.exit(0)

    logger.info(f"Starting indexing process for {len(form_fields_to_index)} form fields...")

    # Batch lists
    batch_ids = []
    batch_documents = []
    batch_metadatas = []
    total_chunks_processed = 0
    upserted_count = 0 # Track successful upserts

    for form_field in tqdm(form_fields_to_index, desc="Indexing Form Fields"):
        if not form_field.full_text:
            logger.warning(f"Skipping form field ID {form_field.id} - no full_text")
            continue

        document = form_field.full_text
        if not form_field.instruction:
             logger.warning(f"Skipping form field ID {form_field.id} because related instruction data is missing.")
             continue

        doc_id = f"field_{form_field.id}"
        metadata = {
            "field_id": form_field.id,
            "field_label": form_field.field_label or "", # Ensure not None
            "form_number": form_field.instruction.form_number or "", # Ensure not None
            "form_title": form_field.instruction.title or "", # Ensure not None
            "text_length": len(document)
        }

        # Simple validation for ChromaDB compatibility
        for k, v in metadata.items():
            if not isinstance(v, (str, int, float, bool)):
                logger.warning(f"Metadata key '{k}' has incompatible type '{type(v)}' for doc_id {doc_id}. Converting to string.")
                metadata[k] = str(v)

        batch_ids.append(doc_id)
        batch_documents.append(document)
        batch_metadatas.append(metadata)
        total_chunks_processed += 1

        if len(batch_ids) >= args.batch_size:
            # Pre-check existing IDs
            try:
                existing_data = collection.get(ids=batch_ids)
                existing_ids = set(existing_data['ids'])
                new_indices = [idx for idx, batch_id in enumerate(batch_ids) if batch_id not in existing_ids]

                if new_indices:
                    ids_to_upsert = [batch_ids[i] for i in new_indices]
                    docs_to_upsert = [batch_documents[i] for i in new_indices]
                    meta_to_upsert = [batch_metadatas[i] for i in new_indices]
                    logger.debug(f"Batch check: Found {len(existing_ids)} existing chunks. Preparing to upsert {len(ids_to_upsert)} new chunks.")

                    # Upsert logic with retries
                    retries = 0
                    max_retries = 3
                    while retries < max_retries:
                        try:
                            collection.upsert(ids=ids_to_upsert, documents=docs_to_upsert, metadatas=meta_to_upsert)
                            upserted_count += len(ids_to_upsert)
                            logger.debug(f"Upserted batch of {len(ids_to_upsert)} chunks successfully.")
                            time.sleep(1) # Small delay after success
                            break
                        except GoogleGenerativeAIError as e:
                             if "Quota exceeded" in str(e) or "429" in str(e):
                                retries += 1
                                wait_time = 60 * retries # Exponential backoff
                                logger.warning(f"Rate limit hit. Attempt {retries}/{max_retries}. Waiting {wait_time}s...")
                                time.sleep(wait_time)
                             else:
                                logger.error(f"Non-rate-limit Google API error during upsert: {e}", exc_info=True)
                                break # Stop retrying for other Google errors
                        except Exception as e:
                            logger.error(f"Unexpected error upserting batch to ChromaDB: {e}", exc_info=True)
                            break # Stop retrying for other errors

                    if retries == max_retries:
                         logger.error(f"Max retries reached for batch ending with ID {ids_to_upsert[-1]}. Skipping this batch.")
                else:
                    logger.debug(f"Batch check: All {len(batch_ids)} chunks already exist. Skipping upsert.")

            except Exception as e:
                logger.error(f"Error checking/upserting batch: {e}", exc_info=True)
                logger.warning("Skipping current batch due to error.")

            # Clear batch lists after processing
            batch_ids.clear()
            batch_documents.clear()
            batch_metadatas.clear()

    # Upsert final batch (with pre-check)
    if batch_ids:
        try:
            existing_data = collection.get(ids=batch_ids)
            existing_ids = set(existing_data['ids'])
            new_indices = [idx for idx, batch_id in enumerate(batch_ids) if batch_id not in existing_ids]

            if new_indices:
                ids_to_upsert = [batch_ids[i] for i in new_indices]
                docs_to_upsert = [batch_documents[i] for i in new_indices]
                meta_to_upsert = [batch_metadatas[i] for i in new_indices]
                logger.info(f"Final Batch check: Found {len(existing_ids)} existing chunks. Preparing to upsert {len(ids_to_upsert)} new chunks.")

                # Upsert logic with retries for final batch
                retries = 0
                max_retries = 3
                while retries < max_retries:
                    try:
                        collection.upsert(ids=ids_to_upsert, documents=docs_to_upsert, metadatas=meta_to_upsert)
                        upserted_count += len(ids_to_upsert)
                        logger.info(f"Upserted final batch of {len(ids_to_upsert)} chunks successfully.")
                        break
                    except GoogleGenerativeAIError as e:
                        if "Quota exceeded" in str(e) or "429" in str(e):
                            retries += 1
                            wait_time = 60 * retries
                            logger.warning(f"Rate limit hit on final batch. Attempt {retries}/{max_retries}. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Non-rate-limit Google API error during final upsert: {e}", exc_info=True)
                            break
                    except Exception as e:
                        logger.error(f"Unexpected error upserting final batch: {e}", exc_info=True)
                        break
                if retries == max_retries:
                     logger.error("Max retries reached for final batch. Failed to upsert remaining chunks.")
            else:
                logger.info(f"Final Batch check: All {len(batch_ids)} chunks already exist. Skipping final upsert.")

        except Exception as e:
            logger.error(f"Error checking/upserting final batch: {e}", exc_info=True)

    logger.info(f"--- Indexing Complete ---")
    logger.info(f"Total form fields processed: {len(form_fields_to_index)}")
    # total_chunks_processed represents the number of fields we attempted to process
    logger.info(f"Total documents prepared for potential upsert: {total_chunks_processed}")
    logger.info(f"Total documents successfully upserted (new or updated): {upserted_count}")
    logger.info(f"Collection '{COLLECTION_NAME}' count: {collection.count()}")
    db.close()

if __name__ == "__main__":
    main() 