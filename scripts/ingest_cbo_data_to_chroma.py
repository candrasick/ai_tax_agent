#!/usr/bin/env python3
"""
Ingests CBO projection data from a JSON file into a ChromaDB collection.

Reads a JSON file where each entry has 'text' and 'metadata' keys,
generates embeddings using Google Generative AI, and stores them in ChromaDB.
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List

import chromadb
from tqdm import tqdm

# Add project root to path for local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ai_tax_agent.settings import settings
from ai_tax_agent.llm_utils import get_embedding_function

# --- Constants ---
DEFAULT_CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
DEFAULT_COLLECTION_NAME = "cbo_revenue_projections"
DEFAULT_INPUT_JSON = os.path.join(
    PROJECT_ROOT, "data/tax_statistics/cbo/cbo_chromadb_ingest_corrected.json"
)
DEFAULT_BATCH_SIZE = 100  # Adjust based on API limits and performance
RATE_LIMIT_DELAY = 1  # Seconds to wait between batches if rate limited

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def initialize_chroma_client(persist_directory: str) -> chromadb.PersistentClient:
    """Initializes and returns a persistent ChromaDB client."""
    logger.info(f"Initializing ChromaDB client at: {persist_directory}")
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
        sys.exit(1)


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from the specified JSON file."""
    if not os.path.exists(file_path):
        logger.error(f"Input JSON file not found: {file_path}")
        sys.exit(1)
    logger.info(f"Loading data from JSON file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list of objects.")
        logger.info(f"Successfully loaded {len(data)} records from JSON.")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file {file_path}: {e}", exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid JSON structure in {file_path}: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to read JSON file {file_path}: {e}", exc_info=True)
        sys.exit(1)


# --- Main Execution ---
def main():
    """Main function to parse arguments and run the ingestion process."""
    parser = argparse.ArgumentParser(
        description="Ingest CBO projection data from JSON into ChromaDB."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=DEFAULT_INPUT_JSON,
        help=f"Path to the input JSON file (default: {DEFAULT_INPUT_JSON})",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help=f"Path to the ChromaDB persistence directory (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Name of the ChromaDB collection (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of documents to process in each batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear (delete) the collection before ingesting if it already exists.",
    )

    args = parser.parse_args()

    # Configure logging level
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level_int)
    # Set level for root logger as well if needed, or configure specific loggers
    logging.getLogger().setLevel(log_level_int)

    logger.info("Starting CBO data ingestion script.")
    logger.info(f"Input file: {args.input_json}")
    logger.info(f"ChromaDB path: {args.chroma_path}")
    logger.info(f"Collection name: {args.collection_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Clear collection: {args.clear}")

    # 1. Load Data
    data = load_json_data(args.input_json)
    if not data:
        logger.warning("No data loaded from JSON file. Exiting.")
        return

    # 2. Initialize ChromaDB and Embedding Function
    logger.info("Initializing ChromaDB client and embedding function...")
    try:
        chroma_client = initialize_chroma_client(args.chroma_path)
        # Use the shared embedding function
        embedding_function_instance = get_embedding_function()
        if not embedding_function_instance:
             logger.error("Failed to initialize embedding function. Exiting.")
             sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB or embedding function: {e}", exc_info=True)
        sys.exit(1)

    # 3. Handle --clear flag
    if args.clear:
        try:
            logger.warning(f"Attempting to delete collection: {args.collection_name}")
            chroma_client.delete_collection(name=args.collection_name)
            logger.info(f"Collection '{args.collection_name}' deleted successfully.")
        except Exception as e:
            # Catching generic Exception as ChromaDB might raise different errors
            # if the collection doesn't exist. We can log this but continue.
            logger.warning(
                f"Could not delete collection '{args.collection_name}' (may not exist): {e}"
            )

    # 4. Get or Create Collection
    logger.info(f"Getting or creating collection: {args.collection_name}")
    try:
        collection = chroma_client.get_or_create_collection(
            name=args.collection_name,
            embedding_function=embedding_function_instance,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Failed to get or create collection '{args.collection_name}': {e}", exc_info=True)
        sys.exit(1)

    # 5. Prepare Data for Upsert
    documents = []
    metadatas = []
    ids = []
    skipped_count = 0

    for i, item in enumerate(data):
        doc_text = item.get("text")
        metadata = item.get("metadata")

        if not isinstance(doc_text, str) or not doc_text.strip():
            logger.warning(f"Skipping record {i+1} due to missing or invalid 'text' field.")
            skipped_count += 1
            continue
        if not isinstance(metadata, dict):
             logger.warning(f"Skipping record {i+1} due to missing or invalid 'metadata' field.")
             skipped_count += 1
             continue

        # Ensure metadata values are ChromaDB compatible (str, int, float, bool)
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif v is None:
                 clean_metadata[k] = "" # Or handle None as appropriate
            else:
                # Attempt to convert other types (like lists/dicts) to string
                try:
                    clean_metadata[k] = json.dumps(v)
                    logger.debug(f"Converted metadata key '{k}' to JSON string for record {i+1}.")
                except TypeError:
                    logger.warning(f"Could not serialize metadata key '{k}' for record {i+1}. Converting to string.")
                    clean_metadata[k] = str(v)

        documents.append(doc_text)
        metadatas.append(clean_metadata)
        ids.append(f"cbo_doc_{uuid.uuid4()}") # Generate unique ID

    if skipped_count > 0:
         logger.warning(f"Skipped a total of {skipped_count} records due to data issues.")

    if not documents:
        logger.warning("No valid documents found after processing JSON data. Exiting.")
        return

    total_docs = len(documents)
    logger.info(f"Prepared {total_docs} documents for ingestion.")

    # 6. Batch Upsert with Rate Limiting
    logger.info(f"Starting upsert process in batches of {args.batch_size}...")
    upserted_count = 0
    for i in tqdm(range(0, total_docs, args.batch_size), desc="Ingesting Batches"):
        batch_start = i
        batch_end = min(i + args.batch_size, total_docs)
        batch_docs = documents[batch_start:batch_end]
        batch_metadatas = metadatas[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]

        retries = 3
        while retries > 0:
            try:
                collection.upsert(
                    documents=batch_docs, metadatas=batch_metadatas, ids=batch_ids
                )
                upserted_count += len(batch_ids)
                logger.debug(f"Upserted batch {i // args.batch_size + 1}/{total_docs // args.batch_size + 1}")
                time.sleep(0.5) # Small delay
                break  # Success, exit retry loop
            except Exception as e:
                # Basic rate limit check (can be improved by importing specific exception)
                if "rate limit" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                    retries -= 1
                    wait_time = RATE_LIMIT_DELAY * (4 - retries) # Basic exponential backoff
                    logger.warning(
                        f"Rate limit likely hit on batch {i // args.batch_size + 1}. "
                        f"Retrying in {wait_time}s... ({retries} retries left)"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Error upserting batch {i // args.batch_size + 1}: {e}",
                        exc_info=True
                    )
                    retries = 0 # Stop retrying for non-rate-limit errors
            if retries == 0:
                 logger.error(f"Failed to upsert batch {i // args.batch_size + 1} after multiple retries. Skipping.")


    logger.info(f"--- Ingestion Complete ---")
    logger.info(f"Total documents processed: {len(data)} (including skipped)")
    logger.info(f"Total documents prepared for ChromaDB: {total_docs}")
    logger.info(f"Total documents successfully upserted: {upserted_count}")
    logger.info(f"Total documents skipped due to data issues: {skipped_count}")
    try:
        logger.info(f"Collection '{args.collection_name}' count: {collection.count()}")
    except Exception as e:
        logger.warning(f"Could not get collection count: {e}")
    logger.info(f"Data ingested into collection: '{args.collection_name}' at '{args.chroma_path}'")


if __name__ == "__main__":
    main() 