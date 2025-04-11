#!/usr/bin/env python
"""
Script to index U.S. Code sections into a ChromaDB vector database.

Reads UsCodeSection data from the primary database, prepares text for embedding
(title + core_text), extracts metadata, and adds it to a Chroma collection.
"""

import os
import sys
import argparse
import logging
import time # Import time module
from typing import List, Optional, Dict, Any, Sequence

import chromadb
# Import only the necessary type from ChromaDB
from chromadb.api.types import EmbeddingFunction
from sqlalchemy.orm import Session
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from tqdm import tqdm

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

# Project components
from ai_tax_agent.settings import settings
from ai_tax_agent.database.models import UsCodeSection
from ai_tax_agent.database.session import get_session

# --- Constants & Config --- #
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "us_code_sections"
DEFAULT_EMBEDDING_MODEL = "text-embedding-004" # Google's latest embedding model
DEFAULT_BATCH_SIZE = 100

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('chroma_indexer')

# --- Embedding Function Wrapper --- #

class LangchainEmbeddingFunctionWrapper(EmbeddingFunction):
    """Wraps a LangChain embedding function to ensure ChromaDB compatibility."""
    def __init__(self, langhain_embedder: GoogleGenerativeAIEmbeddings):
        self._langchain_embedder = langhain_embedder

    # Update type hints to standard Python types expected by ChromaDB 1.x+
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embeds the input texts using the wrapped LangChain embedder."""
        # Langchain embedders usually have embed_documents method for batches
        return self._langchain_embedder.embed_documents(input)

# --- Helper Functions --- #

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Index U.S. Code sections into ChromaDB.")
    parser.add_argument("--chroma-path", type=str, default=DEFAULT_CHROMA_PATH,
                        help=f"Path to the ChromaDB persistent storage directory (default: {DEFAULT_CHROMA_PATH})")
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME,
                        help=f"Name of the ChromaDB collection to use (default: {DEFAULT_COLLECTION_NAME})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of documents to add to ChromaDB in each batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--clear", action='store_true',
                        help="Clear (delete and recreate) the collection before indexing.")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                         help=f"Name of the Google embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})")
    return parser.parse_args()

def initialize_chroma(args, settings):
    """Initializes ChromaDB client, embedding function, and collection."""
    logger.info(f"Initializing ChromaDB client at path: {args.chroma_path}")
    chroma_client = chromadb.PersistentClient(path=args.chroma_path)

    logger.info(f"Using embedding model: {args.embed_model}")
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
        
    # Instantiate LangChain's embedding class
    lc_embed_fn_instance = GoogleGenerativeAIEmbeddings(
        model=f"models/{args.embed_model}", # Prepend models/ as required by LangChain
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_document" # Specify task type for potential optimization
    )
    
    # Instantiate the wrapper
    chroma_compatible_embed_fn = LangchainEmbeddingFunctionWrapper(lc_embed_fn_instance)

    if args.clear:
        logger.warning(f"Clearing existing collection: {args.collection_name}")
    collection = chroma_client.get_or_create_collection(
        name=args.collection_name,
        # Pass the wrapper instance to ChromaDB
        embedding_function=chroma_compatible_embed_fn, 
        metadata={"hnsw:space": "cosine"} # Example metadata
    )
    return chroma_client, collection

def get_sections_to_index(session: Session) -> List[UsCodeSection]:
    """Fetch all UsCodeSection records from the database."""
    logger.info("Fetching all U.S. Code sections from database...")
    try:
        sections = session.query(UsCodeSection).order_by(UsCodeSection.id).all()
        logger.info(f"Found {len(sections)} sections.")
        return sections
    except Exception as e:
        logger.error(f"Database error fetching sections: {e}", exc_info=True)
        return []

def prepare_metadata(section: UsCodeSection) -> Dict[str, Any]:
    """Prepare metadata dictionary for ChromaDB, handling None values."""
    # Metadata values must be int, float, str, or bool
    return {
        "title_number": section.title_number if section.title_number is not None else -1,
        "subtitle": section.subtitle or "",
        "chapter": section.chapter or "",
        "subchapter": section.subchapter or "",
        "part": section.part or "",
        "subpart": section.subpart or "",
        "section_number": section.section_number or "",
        "section_title": section.section_title or "",
        "db_id": section.id # Store original DB id for reference
    }

# --- Main Indexing Logic --- #

def index_sections():
    """Main function to perform the indexing."""
    args = parse_arguments()
    logger.info(f"Starting indexing process with args: {args}")

    # Initialize Chroma Collection
    chroma_client, collection = initialize_chroma(args, settings)
    if not collection:
        return # Error initializing

    # Handle clearing the collection
    if args.clear:
        logger.warning(f"Clearing collection '{args.collection_name}' before indexing.")
        try:
            # Need to delete and recreate to potentially reset metadata/embedding function
            chroma_client.delete_collection(name=args.collection_name)
            logger.info(f"Collection '{args.collection_name}' deleted.")
            chroma_client, collection = initialize_chroma(args, settings)
            if not collection:
                 logger.error("Failed to recreate collection after clearing.")
                 return
        except Exception as e:
             logger.error(f"Error clearing collection '{args.collection_name}': {e}. Maybe it didn't exist?", exc_info=True)
             # Try to proceed assuming re-initialization worked or is desired
             if not collection:
                 chroma_client, collection = initialize_chroma(args, settings)
             if not collection:
                 logger.error("Failed to initialize collection after clearing error.")
                 return

    # Initialize DB Session by calling the function directly
    try:
        db_session: Session = get_session() # Call directly, remove next()
    except Exception as e:
        logger.error(f"Failed to get database session using get_session(): {e}", exc_info=True)
        return

    # db_session: Session = get_session() # Alternative if it's not a generator

    # Check if the session object is valid
    if not db_session:
        logger.error("Failed to get database session (returned None or Falsy)." )
        return

    try:
        # Fetch sections
        sections = get_sections_to_index(db_session)
        if not sections:
            logger.info("No sections found in the database to index.")
            return

        # Prepare batches
        total_indexed = 0
        logger.info(f"Preparing to index {len(sections)} sections in batches of {args.batch_size}...")

        for i in tqdm(range(0, len(sections), args.batch_size), desc="Indexing Batches"):
            batch = sections[i : i + args.batch_size]
            
            metadatas_batch = []

            potential_ids_in_batch = []
            section_data_map = {}

            for section in batch:
                # Prepare data but don't add to final lists yet
                title = section.section_title or ""
                core_text = section.core_text or ""
                doc_content = f"{title}\n\n{core_text}".strip()

                if not core_text:
                    # logger.warning(f"Section ID {section.id} ({section.section_number}) has no core_text. Skipping pre-check.")
                    continue
                if not doc_content:
                    # logger.warning(f"Section ID {section.id} ({section.section_number}) has no combined content. Skipping pre-check.")
                    continue

                chroma_id = f"usc_{section.id}"
                potential_ids_in_batch.append(chroma_id)
                section_data_map[chroma_id] = {
                    "document": doc_content,
                    "metadata": prepare_metadata(section)
                }

            # Check which IDs already exist in ChromaDB
            ids_to_upsert = []
            if potential_ids_in_batch:
                try:
                    existing_data = collection.get(ids=potential_ids_in_batch)
                    existing_ids = set(existing_data['ids'])
                    # Determine which IDs are new
                    ids_to_upsert = [id for id in potential_ids_in_batch if id not in existing_ids]
                    if ids_to_upsert:
                         logger.debug(f"Batch check: Found {len(potential_ids_in_batch) - len(ids_to_upsert)} existing IDs. Preparing to upsert {len(ids_to_upsert)} new sections.")
                    else:
                         logger.debug(f"Batch check: All {len(potential_ids_in_batch)} potential sections already exist in ChromaDB. Skipping upsert.")
                except Exception as e:
                    logger.error(f"Error checking existing IDs in ChromaDB for batch starting at index {i}: {e}", exc_info=True)
                    # Decide how to handle error: skip batch, retry, etc. For now, assume all need upserting to be safe?
                    # Or safer: skip this batch to avoid potential errors
                    logger.warning("Skipping current batch due to error checking existing IDs.")
                    continue 

            # Prepare final lists only for IDs that need upserting
            if ids_to_upsert:
                documents_to_upsert = [section_data_map[id]["document"] for id in ids_to_upsert]
                metadatas_to_upsert = [section_data_map[id]["metadata"] for id in ids_to_upsert]

                # Upsert only the new/missing sections
                if ids_to_upsert:
                    retries = 0
                    max_retries = 3 # Limit retries to avoid infinite loops
                    while retries < max_retries:
                        try:
                            collection.upsert(
                                ids=ids_to_upsert,
                                documents=documents_to_upsert,
                                metadatas=metadatas_to_upsert
                            )
                            total_indexed += len(ids_to_upsert) # Count only newly upserted items
                            logger.debug(f"Upserted batch {i // args.batch_size + 1}, size: {len(ids_to_upsert)}")
                            
                            # Add a short delay AFTER successful batch upsert 
                            # We rely more on the retry for rate limits now, but a small delay helps
                            time.sleep(1)
                            break # Exit retry loop on success
                            
                        except GoogleGenerativeAIError as e:
                            # Check if it's specifically a rate limit error
                            if "Quota exceeded" in str(e) or "429" in str(e):
                                retries += 1
                                wait_time = 90 # 1.5 minutes
                                logger.warning(
                                    f"Rate limit hit on batch {i // args.batch_size + 1}. "
                                    f"Attempt {retries}/{max_retries}. Waiting {wait_time}s before retry..."
                                )
                                time.sleep(wait_time)
                            else:
                                # Re-raise if it's a different Google API error
                                logger.error(f"Non-rate-limit Google API error during upsert batch {i // args.batch_size + 1}: {e}", exc_info=True)
                                break # Exit retry loop on other Google errors
                                
                        except Exception as e:
                            logger.error(f"Unexpected error upserting batch {i // args.batch_size + 1} to ChromaDB: {e}", exc_info=True)
                            break # Exit retry loop on other non-Google errors
                    
                    if retries == max_retries:
                         logger.error(f"Max retries reached for batch {i // args.batch_size + 1}. Skipping this batch.")

        logger.info(f"Indexing finished. Total sections added/updated in this run: {total_indexed}")
        final_count = collection.count()
        logger.info(f"Final collection count: {final_count}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the indexing process: {e}", exc_info=True)
    finally:
        if db_session:
            db_session.close()
            logger.info("Database session closed.")

# --- Main Execution --- #

if __name__ == "__main__":
    index_sections() 