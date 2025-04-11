import argparse
import logging
import os
import sys
import time # Import time module
from typing import List, Sequence

import chromadb
import requests
from bs4 import BeautifulSoup
# Import only the necessary type from ChromaDB
from chromadb.api.types import EmbeddingFunction
# Import LangChain's Google embedding class
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Import the specific exception for rate limiting
from langchain_google_genai._common import GoogleGenerativeAIError
from sqlalchemy.orm import Session, joinedload
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_tax_agent.settings import settings
from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import FormInstruction
# Remove import for non-existent module
# from ai_tax_agent.log_config import setup_logging
from ai_tax_agent.text_utils import chunk_by_html_headings, extract_main_content

logger = logging.getLogger(__name__)

# --- Embedding Function Wrapper --- #

class LangchainEmbeddingFunctionWrapper(EmbeddingFunction):
    """Wraps a LangChain embedding function to ensure ChromaDB compatibility."""
    def __init__(self, langhain_embedder: GoogleGenerativeAIEmbeddings):
        self._langchain_embedder = langhain_embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embeds the input texts using the wrapped LangChain embedder."""
        # Langchain embedders usually have embed_documents method for batches
        return self._langchain_embedder.embed_documents(input)

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

def get_embedding_function(settings):
    """Initializes and returns the Langchain embedding function instance."""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    # Return the LangChain instance, not the wrapper here
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", # Ensure this is the desired model, prepended with models/
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_document"
    )

def get_instructions_to_index(session: Session) -> List[FormInstruction]:
    """Fetches all FormInstruction records that have an HTML URL."""
    logger.info("Fetching form instructions from database...")
    instructions = session.query(FormInstruction).filter(FormInstruction.html_url != None).order_by(FormInstruction.id).all()
    logger.info(f"Found {len(instructions)} instructions with HTML URLs.")
    return instructions

def fetch_and_parse_html(url: str) -> BeautifulSoup | None:
    """Fetches HTML from a URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        # Explicitly use html5lib for potentially complex/malformed HTML
        return BeautifulSoup(response.content, 'html5lib') # Use .content for bytes
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing HTML from {url} with html5lib: {e}")
        return None

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
        # Get the LangChain embedder instance
        lc_embed_fn_instance = get_embedding_function(settings)
        # Wrap it for ChromaDB compatibility
        chroma_compatible_embed_fn = LangchainEmbeddingFunctionWrapper(lc_embed_fn_instance)
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB or embedding function: {e}")
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
            # Pass the wrapper instance to ChromaDB
            embedding_function=chroma_compatible_embed_fn, 
            metadata={"hnsw:space": "cosine"} # Example metadata: Use cosine distance
        )
    except Exception as e:
        logger.error(f"Failed to get or create ChromaDB collection: {e}")
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

    instructions_to_index = get_instructions_to_index(db)

    if not instructions_to_index:
        logger.info("No instructions with HTML URLs found to index.")
        sys.exit(0)

    logger.info(f"Starting indexing process for {len(instructions_to_index)} instructions...")

    # Batch lists
    batch_ids = []
    batch_documents = []
    batch_metadatas = []
    total_chunks_processed = 0

    for instruction in tqdm(instructions_to_index, desc="Indexing Instructions"):
        if not instruction.html_url:
            logger.warning(f"Skipping instruction ID {instruction.id} - no html_url")
            continue

        logger.debug(f"Processing instruction: {instruction.form_number} ({instruction.year}) - {instruction.title}")

        soup = fetch_and_parse_html(instruction.html_url)
        if not soup:
            logger.warning(f"Could not fetch/parse HTML for instruction ID {instruction.id}. Skipping.")
            continue

        main_content_tag = extract_main_content(soup)
        if not main_content_tag:
            logger.warning(f"Could not extract main content for instruction ID {instruction.id}. Skipping.")
            continue

        # Use the robust HTML heading chunker
        # Consider adjusting max_chars based on embedding model limits / desired granularity
        chunks = chunk_by_html_headings(main_content_tag, max_chars=15000) # Example max_chars

        if not chunks:
             logger.warning(f"No text chunks generated for instruction ID {instruction.id}. Skipping.")
             continue

        for i, chunk in enumerate(chunks):
            doc_id = f"instr_{instruction.id}_chunk_{i}"
            metadata = {
                "instruction_id": instruction.id,
                "form_number": instruction.form_number,
                "title": instruction.title,
                "year": instruction.year,
                "html_url": instruction.html_url,
                "chunk_index": i,
                "chunk_char_length": len(chunk)
            }

            batch_ids.append(doc_id)
            batch_documents.append(chunk)
            batch_metadatas.append(metadata)
            total_chunks_processed += 1

            # Upsert batch if size limit reached
            if len(batch_ids) >= args.batch_size:
                # Check which IDs already exist
                ids_to_actually_upsert = []
                documents_to_actually_upsert = []
                metadatas_to_actually_upsert = []
                try:
                    existing_data = collection.get(ids=batch_ids)
                    existing_ids = set(existing_data['ids'])
                    new_indices = [idx for idx, batch_id in enumerate(batch_ids) if batch_id not in existing_ids]
                    
                    if new_indices:
                        ids_to_actually_upsert = [batch_ids[i] for i in new_indices]
                        documents_to_actually_upsert = [batch_documents[i] for i in new_indices]
                        metadatas_to_actually_upsert = [batch_metadatas[i] for i in new_indices]
                        logger.debug(f"Batch check: Found {len(batch_ids) - len(ids_to_actually_upsert)} existing IDs. Preparing to upsert {len(ids_to_actually_upsert)} new chunks.")
                    else:
                         logger.debug(f"Batch check: All {len(batch_ids)} potential chunks already exist in ChromaDB. Skipping upsert for this batch.")

                except Exception as e:
                    logger.error(f"Error checking existing IDs in ChromaDB for instruction batch: {e}", exc_info=True)
                    logger.warning("Skipping current batch due to error checking existing IDs.")
                    # Clear potentially inconsistent batch data before continuing
                    batch_ids.clear()
                    batch_documents.clear()
                    batch_metadatas.clear()
                    continue

                # Only upsert if there are genuinely new chunks
                if ids_to_actually_upsert:
                    retries = 0
                    max_retries = 3 # Limit retries
                    while retries < max_retries:
                        try:
                            logger.debug(f"Upserting batch of {len(ids_to_actually_upsert)} chunks...")
                            collection.upsert(
                                ids=ids_to_actually_upsert,
                                documents=documents_to_actually_upsert,
                                metadatas=metadatas_to_actually_upsert
                            )
                            logger.debug(f"Upserted batch successfully.")
                            
                            # Add short delay after successful upsert
                            time.sleep(1)
                            break # Success, exit retry loop
                            
                        except GoogleGenerativeAIError as e:
                            if "Quota exceeded" in str(e) or "429" in str(e):
                                retries += 1
                                wait_time = 90
                                logger.warning(
                                    f"Rate limit hit on instruction batch. Attempt {retries}/{max_retries}. "
                                    f"Waiting {wait_time}s before retry..."
                                )
                                time.sleep(wait_time)
                            else:
                                logger.error(f"Non-rate-limit Google API error during upsert: {e}", exc_info=True)
                                break # Exit retry loop on other Google errors
                                
                        except Exception as e:
                            logger.error(f"Unexpected error upserting filtered batch to ChromaDB: {e}")
                            break # Exit retry loop on other non-Google errors

                    if retries == max_retries:
                         logger.error(f"Max retries reached for instruction batch ending with ID {ids_to_actually_upsert[-1] if ids_to_actually_upsert else 'N/A'}. Skipping this batch.")
                
                # Clear original batches regardless of whether upsert happened or failed
                batch_ids.clear()
                batch_documents.clear()
                batch_metadatas.clear()

    # Upsert any remaining chunks in the last batch (with pre-check)
    if batch_ids: 
        ids_to_actually_upsert = []
        documents_to_actually_upsert = []
        metadatas_to_actually_upsert = []
        try:
            existing_data = collection.get(ids=batch_ids)
            existing_ids = set(existing_data['ids'])
            new_indices = [idx for idx, batch_id in enumerate(batch_ids) if batch_id not in existing_ids]
                    
            if new_indices:
                ids_to_actually_upsert = [batch_ids[i] for i in new_indices]
                documents_to_actually_upsert = [batch_documents[i] for i in new_indices]
                metadatas_to_actually_upsert = [batch_metadatas[i] for i in new_indices]
                logger.debug(f"Final Batch check: Found {len(batch_ids) - len(ids_to_actually_upsert)} existing IDs. Preparing to upsert {len(ids_to_actually_upsert)} new chunks.")
            else:
                logger.debug(f"Final Batch check: All {len(batch_ids)} potential chunks already exist in ChromaDB. Skipping final upsert.")
        except Exception as e:
            logger.error(f"Error checking existing IDs in ChromaDB for final instruction batch: {e}", exc_info=True)
            logger.warning("Skipping final batch due to error checking existing IDs.")
            # Clear remaining batch data
            batch_ids.clear()
            batch_documents.clear()
            batch_metadatas.clear()

        # Only upsert if there are genuinely new chunks in the final batch
        if ids_to_actually_upsert:
            retries = 0
            max_retries = 3
            while retries < max_retries:
                try:
                    logger.info(f"Upserting final filtered batch of {len(ids_to_actually_upsert)} chunks...")
                    collection.upsert(
                        ids=ids_to_actually_upsert,
                        documents=documents_to_actually_upsert,
                        metadatas=metadatas_to_actually_upsert
                    )
                    logger.info(f"Upserted final batch successfully.")
                    break # Success
                    
                except GoogleGenerativeAIError as e:
                    if "Quota exceeded" in str(e) or "429" in str(e):
                        retries += 1
                        wait_time = 90
                        logger.warning(
                            f"Rate limit hit on final instruction batch. Attempt {retries}/{max_retries}. "
                            f"Waiting {wait_time}s before retry..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Non-rate-limit Google API error during final upsert: {e}", exc_info=True)
                        break # Exit retry loop on other Google errors
                        
                except Exception as e:
                    logger.error(f"Error upserting final filtered batch to ChromaDB: {e}")
                    break # Exit retry loop on other non-Google errors
            
            if retries == max_retries:
                 logger.error("Max retries reached for final instruction batch. Failed to upsert remaining chunks.")

    logger.info(f"Indexing complete. Processed {len(instructions_to_index)} instructions, generating {total_chunks_processed} total potential chunks.") # Note: total_chunks_processed still counts all generated chunks
    db.close()

if __name__ == "__main__":
    main() 