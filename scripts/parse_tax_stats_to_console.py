#!/usr/bin/env python
"""
Script to parse tax statistics JSON files, generate descriptive text, 
index the results into ChromaDB, and print the first 10 parsed items.
"""

import os
import sys
import argparse
import json
import logging
import time # Import time module
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import chromadb
# Import only the necessary type from ChromaDB
from chromadb.api.types import EmbeddingFunction
# Import LangChain's Google embedding class
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Import the specific exception for rate limiting
from langchain_google_genai._common import GoogleGenerativeAIError
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import the parser and the Pydantic model/enum
    from ai_tax_agent.parsers.json_utils import parse_tax_stats_json, TaxStatsLineItem, TaxType 
    # Import settings for API keys etc.
    from ai_tax_agent.settings import settings
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    logger.error("Ensure ai_tax_agent.parsers.json_utils, ai_tax_agent.settings exist and are updated.")
    sys.exit(1)

# --- Embedding Function Wrapper (Copied from index_instructions_chroma.py) --- #

class LangchainEmbeddingFunctionWrapper(EmbeddingFunction):
    """Wraps a LangChain embedding function to ensure ChromaDB compatibility."""
    def __init__(self, langhain_embedder: GoogleGenerativeAIEmbeddings):
        self._langchain_embedder = langhain_embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embeds the input texts using the wrapped LangChain embedder."""
        return self._langchain_embedder.embed_documents(input)

# --- Constants ---
STATS_COLLECTION_NAME = "tax_statistics"
DEFAULT_BATCH_SIZE = 100 # Number of items to process before upserting
CHROMA_DATA_PATH = "chroma_data" # Assuming same path as other script

# --- Helper Functions (Adapted from index_instructions_chroma.py) ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse tax stats, print sample, and index into ChromaDB.")
    parser.add_argument("--stats-dir", required=True, 
                        help="Path to the directory containing tax statistics JSON files.")
    parser.add_argument("--clear", action="store_true", 
                        help="Clear the existing 'tax_statistics' collection before indexing.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                        help="Number of items to batch process before upserting to ChromaDB.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")
    return parser.parse_args()

def get_chroma_client() -> chromadb.Client:
    """Initializes and returns a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DATA_PATH)

def get_embedding_function(settings):
    """Initializes and returns the Langchain embedding function instance."""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_document" # Or semantic_similarity depending on use case
    )

def upsert_batch(collection: chromadb.Collection, batch_ids: List[str], batch_documents: List[str], batch_metadatas: List[Dict[str, Any]]) -> int:
    """Upserts a batch of data into ChromaDB, handling rate limits. Returns count successfully upserted."""
    if not batch_ids:
        return 0

    retries = 0
    max_retries = 5 # Allow more retries for potentially large batches
    upserted_count = 0

    while retries < max_retries:
        try:
            logger.debug(f"Attempting to upsert batch of {len(batch_ids)} items (Attempt {retries + 1}/{max_retries})...")
            collection.upsert(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            upserted_count = len(batch_ids) # Assume full batch success if no exception
            logger.debug(f"Upserted batch successfully.")
            
            # Optional short delay after successful upsert to help with subsequent calls
            time.sleep(1) 
            break # Success, exit retry loop
            
        except GoogleGenerativeAIError as e:
            # Check for common rate limit / quota error messages
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str or "429" in str(e):
                retries += 1
                # Exponential backoff might be better, but simple wait for now
                wait_time = 60 * (retries) # Wait 60, 120, 180 seconds...
                logger.warning(
                    f"Rate limit hit during upsert. Attempt {retries}/{max_retries}. "
                    f"Waiting {wait_time}s before retry... Error: {e}"
                )
                if retries >= max_retries:
                    logger.error(f"Max retries reached for batch ending with ID {batch_ids[-1]}. Skipping this batch.")
                    break # Exit loop after max retries
                time.sleep(wait_time)
            else:
                logger.error(f"Non-rate-limit Google API error during upsert: {e}", exc_info=True)
                # Decide if you want to retry on other errors or just break
                break # Exit retry loop on other Google errors
                
        except Exception as e:
            logger.error(f"Unexpected error upserting batch to ChromaDB: {e}", exc_info=True)
            # Decide if you want to retry on other errors or just break
            break # Exit retry loop on other non-Google errors

    return upserted_count


def main():
    args = parse_arguments()

    # Set the root logger level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level) 
    logger.info(f"Log level set to {args.log_level}")

    # --- Initialize ChromaDB and Embeddings ---
    logger.info("Initializing ChromaDB client and embedding function...")
    try:
        chroma_client = get_chroma_client()
        lc_embed_fn_instance = get_embedding_function(settings)
        chroma_compatible_embed_fn = LangchainEmbeddingFunctionWrapper(lc_embed_fn_instance)
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB or embedding function: {e}", exc_info=True)
        sys.exit(1)

    # --- Handle Collection ---
    if args.clear:
        logger.warning(f"Clearing existing collection: {STATS_COLLECTION_NAME}")
        try:
            chroma_client.delete_collection(name=STATS_COLLECTION_NAME)
            logger.info(f"Collection '{STATS_COLLECTION_NAME}' deleted.")
        except Exception as e:
            logger.warning(f"Could not delete collection '{STATS_COLLECTION_NAME}' (may not exist): {e}")

    logger.info(f"Getting or creating ChromaDB collection: {STATS_COLLECTION_NAME}")
    try:
        collection = chroma_client.get_or_create_collection(
            name=STATS_COLLECTION_NAME,
            embedding_function=chroma_compatible_embed_fn, 
            metadata={"hnsw:space": "cosine"} # Use cosine distance
        )
    except Exception as e:
        logger.error(f"Failed to get or create ChromaDB collection: {e}", exc_info=True)
        sys.exit(1)

    # --- Parse Statistics Data ---
    logger.info(f"Starting parsing of directory: {args.stats_dir}")
    all_data: List[TaxStatsLineItem] = parse_tax_stats_json(args.stats_dir)

    if not all_data:
        logger.warning("Parsing completed, but no line item data was generated. Nothing to index.")
        print("[]") # Output empty JSON list for the sample
        sys.exit(0)

    logger.info(f"Successfully parsed {len(all_data)} individual line items.")
    
    # --- Print Sample Output ---
    output_data = all_data[:10]
    logger.info(f"Printing the first {len(output_data)} parsed items as sample:")
    try:
        output_list = [item.model_dump(exclude_none=True) for item in output_data] 
        print(json.dumps(output_list, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error converting sample Pydantic objects to dictionaries: {e}")
        # Continue with indexing even if printing sample fails

    # --- Index Data into ChromaDB ---
    logger.info(f"Starting indexing process for {len(all_data)} items into collection '{STATS_COLLECTION_NAME}'...")

    batch_ids = []
    batch_documents = []
    batch_metadatas = []
    total_items_upserted = 0

    # Use tqdm for progress bar
    for i, item in enumerate(tqdm(all_data, desc="Indexing Statistics")):
        try:
            # Generate ID - simple index-based for now. Add more unique elements if needed.
            # Consider potential collisions if running multiple times without --clear
            
            # --- Fix: Defensively get tax_type string value --- 
            tax_type_str_for_id = item.tax_type.value if isinstance(item.tax_type, TaxType) else str(item.tax_type)
            # -----------------------------------------------------
            
            doc_id = f"stats_{tax_type_str_for_id}_{item.form_title.replace(' ','_')}_{item.line_item_number}_{item.amount_unit}_{i}"
            
            document = item.full_text # The text to be embedded

            # Create metadata dictionary from other fields
            metadata = {
                "form_title": item.form_title,
                "schedule_title": item.schedule_title,
                "line_item_number": item.line_item_number,
                "label": item.label,
                "amount_unit": item.amount_unit,
                "amount": item.amount,
                # Use the same defensive approach for metadata just in case
                "tax_type": tax_type_str_for_id 
            }
            # Filter out None values from metadata for cleaner storage
            metadata = {k: v for k, v in metadata.items() if v is not None}

            batch_ids.append(doc_id)
            batch_documents.append(document)
            batch_metadatas.append(metadata)

        except Exception as e:
            logger.error(f"Error processing item index {i} for ChromaDB: {item}. Error: {e}", exc_info=True)
            continue # Skip this item and proceed

        # Upsert batch if size limit reached
        if len(batch_ids) >= args.batch_size:
            upserted_in_batch = upsert_batch(collection, batch_ids, batch_documents, batch_metadatas)
            total_items_upserted += upserted_in_batch
            batch_ids.clear()
            batch_documents.clear()
            batch_metadatas.clear()

    # Upsert any remaining items in the last batch
    if batch_ids: 
        upserted_in_batch = upsert_batch(collection, batch_ids, batch_documents, batch_metadatas)
        total_items_upserted += upserted_in_batch

    logger.info(f"Indexing complete.")
    logger.info(f"Total items parsed: {len(all_data)}")
    logger.info(f"Total items successfully upserted into ChromaDB collection '{STATS_COLLECTION_NAME}': {total_items_upserted}")

if __name__ == "__main__":
    main() 