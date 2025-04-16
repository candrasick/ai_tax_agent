# scripts/link_stats_to_code.py

import argparse
import logging
import os
import sys
import time
import json
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.api.types import EmbeddingFunction, QueryResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert # For upsert logic on SQLite
from sqlalchemy import func, select

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ai_tax_agent.settings import settings
from ai_tax_agent.database.session import get_session
# Import relevant models (ensure FormFieldUsCodeSectionLink is created)
from ai_tax_agent.database.models import FormFieldStatistics, FormFieldUsCodeSectionLink, FormField, UsCodeSection # Assuming UsCodeSection model exists

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
STATS_COLLECTION_NAME = "tax_statistics"
INSTRUCTIONS_COLLECTION_NAME = "form_instructions"
CODE_SECTIONS_COLLECTION_NAME = "us_code_sections"
CHROMA_DATA_PATH = "chroma_data"
DEFAULT_PROCESSING_LIMIT = 100 # Limit how many stats items to process per run

# --- Embedding Function Wrapper (Consistent with other scripts) ---
class LangchainEmbeddingFunctionWrapper(EmbeddingFunction):
    def __init__(self, langhain_embedder: GoogleGenerativeAIEmbeddings):
        self._langchain_embedder = langhain_embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._langchain_embedder.embed_documents(input)

# --- ChromaDB and Embeddings Setup ---
def get_chroma_client() -> chromadb.Client:
    logger.debug(f"Initializing ChromaDB client with path: {CHROMA_DATA_PATH}")
    return chromadb.PersistentClient(path=CHROMA_DATA_PATH)

def get_embedding_function(settings):
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    logger.debug("Initializing GoogleGenerativeAIEmbeddings (models/text-embedding-004)")
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_query" # Use retrieval_query for searching
    )

# --- LLM Setup ---
def get_llm(settings):
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    logger.debug("Initializing ChatGoogleGenerativeAI (gemini-1.5-flash)")
    # Using flash for potentially faster/cheaper reasoning steps
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.gemini_api_key, temperature=0.1)

# --- Helper Functions ---
def format_results_for_prompt(query_result: QueryResult, max_results: int = 5) -> str:
    """Formats ChromaDB query results for inclusion in an LLM prompt."""
    if not query_result or not query_result.get('ids') or not query_result['ids'][0]:
        return "No results found."

    formatted = []
    ids = query_result['ids'][0]
    documents = query_result['documents'][0] if query_result.get('documents') else [''] * len(ids)
    metadatas = query_result['metadatas'][0] if query_result.get('metadatas') else [{}] * len(ids)
    distances = query_result['distances'][0] if query_result.get('distances') else [None] * len(ids)

    count = 0
    for i, doc_id in enumerate(ids):
        if count >= max_results:
            break
        doc_text = documents[i] if i < len(documents) else "N/A"
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else "N/A"
        formatted.append(f"Result {count + 1}:\n  ID: {doc_id}\n  Distance: {dist}\n  Metadata: {meta}\n  Text: {doc_text[:500]}...\n---") # Truncate text for brevity
        count += 1

    return "\n".join(formatted)

# --- Core Agent Logic ---

def find_best_instruction_match(llm, stats_item: Dict[str, Any], instruction_results: QueryResult) -> Optional[str]:
    """Uses LLM to choose the best matching instruction based on similarity results."""
    if not instruction_results or not instruction_results.get('ids') or not instruction_results['ids'][0]:
        logger.warning("No instruction results found to reason upon.")
        return None

    formatted_results = format_results_for_prompt(instruction_results, max_results=5)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert tax analyst. Your task is to identify the *single best* IRS form instruction document that corresponds to the given tax statistic item.
        Analyze the tax statistic details and the provided search results from the `form_instructions` vector store.
        Focus on matching the form number, line item number, label, and overall context.
        Output *only* the ID (e.g., 'field_12345') of the best matching instruction document.
        If none of the results seem like a strong match, output 'None'.
        Do not add any explanation or preamble."""),
        ("human", """Tax Statistic Item:
        Form Title: {form_title}
        Schedule Title: {schedule_title}
        Line Item Number: {line_item_number}
        Label: {label}
        Amount: {amount} {amount_unit}
        Tax Type: {tax_type}
        Full Text: {full_text}

        Top 5 Search Results from `form_instructions`:
        {search_results}

        Based on the statistic and the search results, what is the ID of the single best matching form instruction? Output only the ID or 'None'.""")
    ])

    chain = prompt_template | llm | StrOutputParser()

    try:
        logger.debug(f"Asking LLM to find best instruction match for stats item: {stats_item.get('line_item_number')} - {stats_item.get('label')}")
        # Ensure required fields exist, provide defaults if None
        response = chain.invoke({
            "form_title": stats_item.get("form_title", "N/A"),
            "schedule_title": stats_item.get("schedule_title", "N/A"),
            "line_item_number": stats_item.get("line_item_number", "N/A"),
            "label": stats_item.get("label", "N/A"),
            "amount": stats_item.get("amount", "N/A"),
            "amount_unit": stats_item.get("amount_unit", "N/A"),
            "tax_type": stats_item.get("tax_type", "N/A"),
            "full_text": stats_item.get("full_text", "N/A"),
            "search_results": formatted_results
        })
        logger.debug(f"LLM response for best instruction: {response}")

        if response and response.strip().lower() != 'none' and response.startswith('field_'):
            chosen_id = response.strip()
            # Verify the chosen ID was actually in the results
            if chosen_id in instruction_results['ids'][0]:
                 logger.info(f"LLM selected best instruction match: {chosen_id}")
                 return chosen_id
            else:
                 logger.warning(f"LLM chose ID {chosen_id}, but it wasn't in the top 5 results. Treating as no match.")
                 return None
        else:
            logger.info("LLM indicated no single strong instruction match found.")
            return None
    except Exception as e:
        logger.error(f"Error during LLM call for finding best instruction: {e}", exc_info=True)
        return None


def find_relevant_code_sections(llm, stats_item: Dict[str, Any], chosen_instruction_doc: str, code_section_results: QueryResult) -> List[Tuple[str, str]]:
    """Uses LLM to identify relevant US code sections and generate rationale."""
    if not code_section_results or not code_section_results.get('ids') or not code_section_results['ids'][0]:
        logger.warning("No code section results found to reason upon.")
        return []

    formatted_results = format_results_for_prompt(code_section_results, max_results=10)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert tax analyst linking tax form instructions to the underlying US tax code.
        Analyze the tax statistic item, the chosen form instruction text, and the top 10 search results from the `us_code_sections` vector store.
        Identify which US code sections are directly relevant to the specific topic described by the instruction and statistic.
        For each relevant section found in the search results, provide its ID (e.g., 'usc_5678') and a concise, one-sentence rationale explaining *why* it's relevant to the instruction/statistic.
        Format your output as a JSON list of objects, where each object has 'id' (the us_code_sections ID) and 'rationale' keys.
        Example: [{{"id": "usc_123", "rationale": "This section defines the rules for capital gains mentioned in the instruction."}}, {{"id": "usc_456", "rationale": "Provides the specific deduction limits relevant to this form line."}}]
        If none of the search results are relevant, output an empty JSON list: []
        Focus only on results provided. Do not hallucinate IDs or rationales."""),
         ("human", """Tax Statistic Item:
        Form Title: {form_title}
        Line Item Number: {line_item_number}
        Label: {label}
        Full Text: {full_text}

        Chosen Form Instruction Text (Truncated):
        {instruction_text}

        Top 10 Search Results from `us_code_sections`:
        {search_results}

        Output a JSON list of relevant section IDs and rationales based *only* on the provided search results.""")
    ])

    chain = prompt_template | llm | StrOutputParser()

    try:
        logger.debug(f"Asking LLM to find relevant code sections for stats item: {stats_item.get('line_item_number')} - {stats_item.get('label')}")
        response = chain.invoke({
            "form_title": stats_item.get("form_title", "N/A"),
            "line_item_number": stats_item.get("line_item_number", "N/A"),
            "label": stats_item.get("label", "N/A"),
            "full_text": stats_item.get("full_text", "N/A"),
            "instruction_text": chosen_instruction_doc[:1000] + "..." if chosen_instruction_doc else "N/A", # Provide context from chosen instruction
            "search_results": formatted_results
        })
        logger.debug(f"LLM response for relevant code sections (raw): {response}")

        # Attempt to parse the JSON response
        try:
            # Clean potential markdown fences
            response_cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            parsed_links = json.loads(response_cleaned)
            if not isinstance(parsed_links, list):
                raise ValueError("LLM response is not a JSON list.")

            valid_links: List[Tuple[str, str]] = []
            code_result_ids = set(code_section_results['ids'][0]) # Set for quick lookup

            for link_data in parsed_links:
                if isinstance(link_data, dict) and 'id' in link_data and 'rationale' in link_data:
                    section_id = link_data['id']
                    rationale = link_data['rationale']
                    # Crucially, verify the ID came from the search results
                    if section_id in code_result_ids:
                        valid_links.append((section_id, rationale))
                    else:
                         logger.warning(f"LLM suggested code section ID {section_id} which was not in the top 10 search results. Discarding.")
                else:
                    logger.warning(f"LLM output item has invalid format: {link_data}. Skipping.")

            logger.info(f"LLM identified {len(valid_links)} relevant code sections.")
            return valid_links
        except json.JSONDecodeError:
             logger.error(f"Failed to parse LLM JSON response for code sections: {response}", exc_info=True)
             return []
        except Exception as e:
             logger.error(f"Error processing LLM response for code sections: {e}. Response: {response}", exc_info=True)
             return []

    except Exception as e:
        logger.error(f"Error during LLM call for finding code sections: {e}", exc_info=True)
        return []

def update_sql_database(db: Session, form_field_id: int, statistic: Dict[str, Any], links: List[Tuple[str, str]]):
    """Updates the form_fields_statistics and form_field_us_code_section_link tables."""
    if not form_field_id:
        logger.error("Cannot update SQL database without a valid form_field_id.")
        return

    try:
        # 1. Update form_field_us_code_section_link
        logger.debug(f"Attempting to add {len(links)} code section links for form_field_id {form_field_id}")
        added_links_count = 0
        for section_vector_id, rationale in links:
            try:
                # Extract numeric ID from 'usc_{db_id}' format
                if not section_vector_id.startswith("usc_") or section_vector_id.startswith("usc_section_"):
                    logger.warning(f"Skipping link: Invalid US Code Section vector ID format: {section_vector_id}. Expected 'usc_ID'.")
                    continue
                # Ensure we split correctly, even if multiple underscores exist (though unlikely with usc_ID)
                parts = section_vector_id.split("_")
                if len(parts) < 2:
                    logger.warning(f"Skipping link: Could not split vector ID into parts: {section_vector_id}")
                    continue
                us_code_section_db_id_str = parts[-1]
                us_code_section_db_id = int(us_code_section_db_id_str)

                # Check if the link already exists
                link_exists = db.execute(
                    select(FormFieldUsCodeSectionLink).where(
                        FormFieldUsCodeSectionLink.form_field_id == form_field_id,
                        FormFieldUsCodeSectionLink.us_code_section_id == us_code_section_db_id
                    )
                ).scalar_one_or_none()

                if not link_exists:
                    new_link = FormFieldUsCodeSectionLink(
                        form_field_id=form_field_id,
                        us_code_section_id=us_code_section_db_id,
                        rationale=rationale
                        # created_at is handled by server_default
                    )
                    db.add(new_link)
                    added_links_count += 1
                    logger.debug(f"  Added link: Form Field {form_field_id} -> Code Section {us_code_section_db_id}")
                else:
                    logger.debug(f"  Skipped link (already exists): Form Field {form_field_id} -> Code Section {us_code_section_db_id}")

            except ValueError:
                 logger.warning(f"Skipping link: Could not parse integer DB ID from vector ID: {section_vector_id}")
                 continue
            except Exception as e_inner:
                 logger.error(f"Error processing link for section {section_vector_id}: {e_inner}", exc_info=True)
                 # Continue to next link

        # 2. Update form_fields_statistics (Upsert Logic)
        amount_unit = statistic.get("amount_unit")
        amount = statistic.get("amount")

        if amount_unit and amount is not None:
            logger.debug(f"Attempting to update statistics for form_field_id {form_field_id} with unit '{amount_unit}' and amount {amount}")
            stats_column = None
            if amount_unit == "dollars":
                stats_column = FormFieldStatistics.dollars
            elif amount_unit == "forms":
                 stats_column = FormFieldStatistics.forms
            # Assuming 'individuals' maps to 'people' column
            elif amount_unit == "individuals":
                 stats_column = FormFieldStatistics.people
            else:
                logger.warning(f"Unrecognized amount unit '{amount_unit}' for form_field_id {form_field_id}. Cannot update statistics.")

            if stats_column is not None:
                # Basic check for "thousands" - adjust if needed based on actual data indication
                # This is a placeholder - a more robust check (e.g., metadata flag) is better
                if "thousands" in statistic.get("label", "").lower() or "thousands" in statistic.get("full_text", "").lower():
                     logger.info(f"Amount unit {amount_unit} for field {form_field_id} potentially in thousands. Multiplying {amount} by 1000.")
                     amount *= 1000

                # Using SQLAlchemy merge for upsert
                # Note: This requires the primary key (form_field_id) to be set correctly.
                # If FormFieldStatistics has its own auto-inc PK, this approach needs adjustment.
                # Assuming form_field_id IS the primary key here based on previous context.
                stmt = sqlite_insert(FormFieldStatistics).values(
                    form_field_id=form_field_id,
                    dollars=amount if stats_column.key == 'dollars' else None,
                    forms=amount if stats_column.key == 'forms' else None,
                    people=amount if stats_column.key == 'people' else None
                )
                # Define the columns to update on conflict
                update_dict = {
                    stats_column.key: stmt.excluded[stats_column.key] # Use the new value
                }
                # Ensure other columns are not nullified on update if they already have values
                if stats_column.key != 'dollars':
                    update_dict['dollars'] = FormFieldStatistics.dollars
                if stats_column.key != 'forms':
                    update_dict['forms'] = FormFieldStatistics.forms
                if stats_column.key != 'people':
                     update_dict['people'] = FormFieldStatistics.people

                # SQLite specific ON CONFLICT DO UPDATE
                final_stmt = stmt.on_conflict_do_update(
                    index_elements=['form_field_id'], # The constraint column
                    set_=update_dict
                )
                db.execute(final_stmt)
                logger.debug(f"Statistics upsert statement executed for form_field_id {form_field_id}")

        else:
             logger.debug(f"Skipping statistics update for form_field_id {form_field_id} due to missing amount or unit.")

        # 3. Commit Changes for this item
        db.commit()
        logger.info(f"Successfully processed and committed updates for form_field_id {form_field_id}. Added {added_links_count} links.")

    except Exception as e:
        logger.error(f"Error during SQL database update for form_field_id {form_field_id}: {e}", exc_info=True)
        db.rollback() # Rollback on error for this item

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Link Tax Statistics to Instructions and Code Sections.")
    parser.add_argument("--limit", type=int, default=DEFAULT_PROCESSING_LIMIT, help="Maximum number of statistics items to process.")
    parser.add_argument("--offset", type=int, default=0, help="Offset for fetching statistics items.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level) # Set root logger level
    # Set specific loggers if needed
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING if log_level <= logging.INFO else log_level)
    logging.getLogger('chromadb').setLevel(logging.WARNING if log_level <= logging.INFO else log_level)


    logger.info("--- Starting Tax Statistics Linking Agent ---")
    logger.info(f"Processing Limit: {args.limit}, Offset: {args.offset}")

    try:
        # Initialize components
        chroma_client = get_chroma_client()
        # Get the LangChain embedder instance first
        lc_embed_fn_instance = get_embedding_function(settings)
        # Create the wrapper for ChromaDB
        chroma_compatible_embed_fn = LangchainEmbeddingFunctionWrapper(lc_embed_fn_instance)
        
        llm = get_llm(settings)
        db: Session = get_session()

        # Get Chroma collections, explicitly providing the embedding function instance
        logger.info(f"Getting collection '{STATS_COLLECTION_NAME}'")
        stats_collection = chroma_client.get_collection(
            name=STATS_COLLECTION_NAME, 
            embedding_function=chroma_compatible_embed_fn # Explicitly provide wrapper
        )
        logger.info(f"Getting collection '{INSTRUCTIONS_COLLECTION_NAME}'")
        instructions_collection = chroma_client.get_collection(
            name=INSTRUCTIONS_COLLECTION_NAME,
            embedding_function=chroma_compatible_embed_fn # Explicitly provide wrapper
        )
        logger.info(f"Getting collection '{CODE_SECTIONS_COLLECTION_NAME}'")
        code_sections_collection = chroma_client.get_collection(
            name=CODE_SECTIONS_COLLECTION_NAME,
            embedding_function=chroma_compatible_embed_fn # Explicitly provide wrapper
        )
        logger.info("Successfully connected to ChromaDB collections and Database.")

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # Fetch statistics items to process
    logger.info(f"Fetching {args.limit} items from '{STATS_COLLECTION_NAME}' collection (offset: {args.offset})...")
    try:
        stats_results = stats_collection.get(
            limit=args.limit,
            offset=args.offset,
            include=["metadatas", "documents"] # Need document (full_text) and metadata
        )
        stats_items_to_process = []
        if stats_results and stats_results.get('ids'):
             for i, item_id in enumerate(stats_results['ids']):
                  item_data = stats_results['metadatas'][i] or {}
                  item_data['full_text'] = stats_results['documents'][i] if stats_results.get('documents') else None
                  item_data['vector_id'] = item_id # Keep track of original vector ID
                  if item_data.get('full_text'): # Only process if we have text
                       stats_items_to_process.append(item_data)
                  else:
                       logger.warning(f"Skipping stats item {item_id} due to missing document/full_text.")

        logger.info(f"Fetched {len(stats_items_to_process)} statistics items to process.")

    except Exception as e:
        logger.error(f"Failed to fetch items from '{STATS_COLLECTION_NAME}': {e}", exc_info=True)
        db.close()
        sys.exit(1)

    if not stats_items_to_process:
        logger.info("No statistics items found to process with the given limit/offset.")
        db.close()
        sys.exit(0)

    # Process each statistics item
    processed_count = 0
    skipped_count = 0
    start_time = time.time()

    for stats_item in stats_items_to_process:
        item_label = f"{stats_item.get('form_title', '')} L.{stats_item.get('line_item_number', '')} ({stats_item.get('vector_id')})"
        logger.info(f"--- Processing Statistic Item: {item_label} ---")

        try:
            query_text = stats_item['full_text'] # Use the descriptive text for searching

            # 1. Similarity Search (Form Instructions)
            logger.debug("Querying form_instructions collection...")
            instruction_results = instructions_collection.query(
                query_texts=[query_text],
                n_results=5,
                include=["metadatas", "documents", "distances"]
            )

            # 1b. Reason & Select Best Instruction
            best_instruction_vector_id = find_best_instruction_match(llm, stats_item, instruction_results)

            if not best_instruction_vector_id:
                logger.warning(f"Could not determine best instruction match for {item_label}. Skipping database updates for this item.")
                skipped_count += 1
                continue # Skip to next stats item

            # Extract form_field_id from metadata of the chosen instruction
            chosen_instruction_metadata = {}
            chosen_instruction_doc = None
            try:
                idx = instruction_results['ids'][0].index(best_instruction_vector_id)
                chosen_instruction_metadata = instruction_results['metadatas'][0][idx]
                chosen_instruction_doc = instruction_results['documents'][0][idx]
                form_field_id = chosen_instruction_metadata.get('field_id')
                if form_field_id is None:
                    logger.error(f"Chosen instruction {best_instruction_vector_id} is missing 'field_id' in metadata. Cannot proceed.")
                    skipped_count += 1
                    continue
                form_field_id = int(form_field_id) # Ensure it's an integer
            except (ValueError, IndexError, TypeError) as e:
                 logger.error(f"Error extracting field_id or document for chosen instruction {best_instruction_vector_id}: {e}", exc_info=True)
                 skipped_count += 1
                 continue


            # 2. Similarity Search (Tax Code)
            logger.debug("Querying us_code_sections collection...")
            code_section_results = code_sections_collection.query(
                query_texts=[query_text],
                n_results=10,
                include=["metadatas", "documents", "distances"]
            )

            # 2b. Reason & Select Relevant Sections + Rationale
            # Pass chosen instruction document text for better context
            relevant_links = find_relevant_code_sections(llm, stats_item, chosen_instruction_doc, code_section_results)

            # 3. Store Statistics & Links in SQL
            update_sql_database(db, form_field_id, stats_item, relevant_links)
            processed_count += 1

        except Exception as e:
            logger.error(f"Unexpected error processing item {item_label}: {e}", exc_info=True)
            skipped_count += 1
            db.rollback() # Rollback any partial changes for this item

        # Optional delay between items to help manage API rate limits if needed
        # time.sleep(1)

    # --- Wrap Up ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info("--- Agent Processing Complete ---")
    logger.info(f"Total items processed: {processed_count}")
    logger.info(f"Total items skipped: {skipped_count}")
    logger.info(f"Total duration: {duration:.2f} seconds")
    db.close()


if __name__ == "__main__":
    main() 