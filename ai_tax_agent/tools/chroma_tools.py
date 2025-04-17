#!/usr/bin/env python
"""Tools for interacting with ChromaDB vector stores."""

import logging
import os
import sys
from typing import List, Dict, Any

import chromadb
from chromadb import Collection

# Add project root to path for local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Go up one more level
sys.path.insert(0, PROJECT_ROOT)

from ai_tax_agent.settings import settings # Needed if settings influence path/collection
from ai_tax_agent.llm_utils import get_embedding_function # Get the SAME embedder

logger = logging.getLogger(__name__)

# --- Constants --- (Should match ingestion script)
DEFAULT_CHROMA_PATH_STR = os.path.join(PROJECT_ROOT, "chroma_data")
CBO_COLLECTION_NAME = "cbo_revenue_projections"
FORM_INSTRUCTIONS_COLLECTION_NAME = "form_instructions"

def _get_chroma_collection(collection_name: str, persist_directory: str = DEFAULT_CHROMA_PATH_STR) -> Collection | None:
    """Helper function to get a specific ChromaDB collection, associating the embedding function."""
    embedding_function = get_embedding_function() # Get the LangChain embedder
    if not embedding_function:
        logger.error("Failed to initialize embedding function for ChromaDB query.")
        return None

    try:
        # Explicitly log the path being used
        logger.info(f"Attempting to connect to ChromaDB client at path: {persist_directory}")
        client = chromadb.PersistentClient(path=persist_directory)
        logger.debug(f"Getting collection: {collection_name} with embedding function in {persist_directory}")
        # Get collection and associate the embedding function required for querying
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function # Pass the LangChain object
        )
        logger.debug(f"Successfully retrieved collection '{collection_name}' with embedding function.")
        return collection
    except ValueError as ve:
        # Handle specific case where collection might not exist
        if "Could not find collection" in str(ve):
            logger.error(f"ChromaDB collection '{collection_name}' not found in directory '{persist_directory}'. Please ensure it has been created and populated.")
            return None
        else:
             logger.error(f"ValueError getting ChromaDB collection '{collection_name}': {ve}", exc_info=True)
             return None
    except Exception as e:
        logger.error(f"Failed to get ChromaDB collection '{collection_name}': {e}", exc_info=True)
        return None

def query_cbo_projections(query_text: str, n_results: int = 3) -> str:
    """Queries the CBO Revenue Projections ChromaDB collection for relevant documents.

    Args:
        query_text: The text query to search for.
        n_results: The number of results to return.

    Returns:
        A formatted string containing the query results, or an error message.
    """
    logger.info(f"Querying CBO projections collection with: '{query_text}' (n_results={n_results})")

    collection = _get_chroma_collection(CBO_COLLECTION_NAME)
    if not collection:
        return f"Error: Could not access the ChromaDB collection '{CBO_COLLECTION_NAME}'."

    # Note: ChromaDB handles embedding the query text automatically if the collection
    # was created with an embedding function. If we used manual embedding during
    # ingestion (embedding_function=None), we might need to embed the query here
    # first using the same get_embedding_function() from llm_utils.
    # However, standard usage is to let collection.query handle it.

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        if not results or not results.get('ids') or not results['ids'][0]:
            return f"No relevant results found in CBO projections for query: '{query_text}'"

        # Format results
        output_lines = [f"Found {len(results['ids'][0])} results for '{query_text}' in CBO Projections:"]
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            doc_snippet = document[:200] + ("..." if len(document) > 200 else "")

            output_lines.append(f"\n--- Result {i+1} (ID: {doc_id}, Distance: {distance:.4f}) ---")
            output_lines.append(f"Metadata: {metadata}")
            output_lines.append(f"Document Snippet: {doc_snippet}")

        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Error querying ChromaDB collection '{CBO_COLLECTION_NAME}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred while querying CBO projections."

# --- Placeholder for Querying Form Instructions --- #

def query_form_instructions(query_text: str, n_results: int = 3) -> str:
    """Queries the Form Instructions ChromaDB collection for relevant documents.

    Args:
        query_text: The text query to search for.
        n_results: The number of results to return.

    Returns:
        A formatted string containing the query results, or an error message.
    """
    logger.info(f"Querying Form Instructions collection with: '{query_text}' (n_results={n_results})")
    # Similar implementation to query_cbo_projections, but uses FORM_INSTRUCTIONS_COLLECTION_NAME
    collection = _get_chroma_collection(FORM_INSTRUCTIONS_COLLECTION_NAME)
    if not collection:
        return f"Error: Could not access the ChromaDB collection '{FORM_INSTRUCTIONS_COLLECTION_NAME}'."

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        if not results or not results.get('ids') or not results['ids'][0]:
            return f"No relevant results found in Form Instructions for query: '{query_text}'"

        output_lines = [f"Found {len(results['ids'][0])} results for '{query_text}' in Form Instructions:"]
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]
            doc_snippet = document[:200] + ("..." if len(document) > 200 else "")

            output_lines.append(f"\n--- Result {i+1} (ID: {doc_id}, Distance: {distance:.4f}) ---")
            # Consider customizing metadata display based on what's useful (e.g., field_label, form_number)
            output_lines.append(f"Metadata: {metadata}")
            output_lines.append(f"Document Snippet: {doc_snippet}")

        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Error querying ChromaDB collection '{FORM_INSTRUCTIONS_COLLECTION_NAME}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred while querying Form Instructions."


# Example Usage (Optional)
if __name__ == '__main__':
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test_query_cbo = "individual income taxes 2025"
    print(f"--- Testing CBO Query: '{test_query_cbo}' ---")
    print(query_cbo_projections(test_query_cbo))
    print("-"*50)

    test_query_instr = "deduction for medical expenses"
    print(f"--- Testing Form Instructions Query: '{test_query_instr}' ---")
    print(query_form_instructions(test_query_instr))
    print("-"*50) 