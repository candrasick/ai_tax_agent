#!/usr/bin/env python
"""Tools for interacting with the application's relational database."""

import logging
from decimal import Decimal, InvalidOperation
from typing import Dict, Any

from sqlalchemy import func as sql_func
from sqlalchemy.orm import Session

# Assuming project structure allows this import
from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import (
    UsCodeSection,
    FormFieldUsCodeSectionLink,
    FormFieldStatistics,
    FormField,
    FormInstruction,
)

logger = logging.getLogger(__name__)


def get_section_stats(section_identifier: str) -> str:
    """Fetches and aggregates statistics for form fields linked to a US Code section.

    Args:
        section_identifier: The identifier of the US Code section (e.g., '162' or the primary key ID).

    Returns:
        A string summarizing the aggregated statistics or an error message.
    """
    db: Session = get_session()
    if not db:
        return "Error: Could not get database session."

    section_id_int: int | None = None

    # Try converting identifier to int (if it's a PK)
    try:
        section_id_int = int(section_identifier)
        logger.debug(f"Interpreted section identifier '{section_identifier}' as primary key ID.")
        section = db.query(UsCodeSection).filter(UsCodeSection.id == section_id_int).first()
        if not section:
             logger.warning(f"No section found with ID: {section_id_int}")
             # Fallback: Try searching by section number if ID fails
             section_id_int = None # Reset to try searching by number string

    except ValueError:
        logger.debug(f"Section identifier '{section_identifier}' is not an integer ID, searching by section number.")
        # Identifier is likely a section number string (e.g., "162(a)")

    if section_id_int is None:
        # Search by section number string (case-insensitive exact match for simplicity first)
        section = db.query(UsCodeSection).filter(sql_func.lower(UsCodeSection.section_number) == section_identifier.lower()).first()
        if section:
            section_id_int = section.id
            logger.debug(f"Found section ID {section_id_int} for section number '{section_identifier}'.")
        else:
             logger.warning(f"No section found matching identifier: '{section_identifier}'.")
             db.close()
             return f"Error: Could not find a US Code section matching identifier '{section_identifier}'."

    try:
        # Find linked form field IDs
        linked_field_ids_query = (
            db.query(FormFieldUsCodeSectionLink.form_field_id)
            .filter(FormFieldUsCodeSectionLink.us_code_section_id == section_id_int)
        )
        linked_field_ids = [item[0] for item in linked_field_ids_query.all()]

        if not linked_field_ids:
            db.close()
            return f"Section '{section_identifier}' (ID: {section_id_int}) found, but no form fields are linked to it in the database."

        logger.info(f"Found {len(linked_field_ids)} form fields linked to section ID {section_id_int}.")

        # Aggregate statistics for these fields
        # Using coalesce to handle potential NULLs in sum, defaulting to 0
        stats_query = (
            db.query(
                sql_func.sum(sql_func.coalesce(FormFieldStatistics.dollars, 0)).label("total_dollars"),
                sql_func.sum(sql_func.coalesce(FormFieldStatistics.forms, 0)).label("total_forms"),
                sql_func.sum(sql_func.coalesce(FormFieldStatistics.people, 0)).label("total_people")
            )
            .filter(FormFieldStatistics.form_field_id.in_(linked_field_ids))
        )

        aggregated_stats: Dict[str, Any] | None = stats_query.first()._asdict() # type: ignore

        if not aggregated_stats:
             # This case should be unlikely if linked_field_ids is not empty, but handle it.
             logger.warning(f"Query returned no stats for linked fields: {linked_field_ids}")
             db.close()
             return f"Found {len(linked_field_ids)} linked fields for section '{section_identifier}' but could not aggregate statistics."

        # Format results
        # Ensure results are treated as Decimal or handle potential type issues
        total_dollars = aggregated_stats.get("total_dollars", Decimal(0))
        total_forms = aggregated_stats.get("total_forms", Decimal(0))
        total_people = aggregated_stats.get("total_people", Decimal(0))

        # Convert Decimals to int/float for cleaner output if they are whole numbers
        total_dollars_str = f"{total_dollars:,.0f}" if total_dollars is not None else "N/A"
        total_forms_str = f"{total_forms:,.0f}" if total_forms is not None else "N/A"
        total_people_str = f"{total_people:,.0f}" if total_people is not None else "N/A"

        result_str = (
            f"Statistics for Section '{section_identifier}' (ID: {section_id_int}):\n"
            f"- Based on {len(linked_field_ids)} linked form field(s).\n"
            f"- Total Dollars: {total_dollars_str}\n"
            f"- Total Forms: {total_forms_str}\n"
            f"- Total People: {total_people_str}"
        )

        logger.info(f"Successfully aggregated stats for section {section_id_int}.")
        return result_str

    except Exception as e:
        logger.error(f"Error fetching stats for section ID {section_id_int}: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while fetching statistics for section '{section_identifier}'."
    finally:
        db.close()


# --- New Detailed Stats Function ---

def get_section_details_and_stats(section_identifier: str) -> str:
    """Fetches detailed info and aggregated stats for form fields linked to a US Code section.

    Args:
        section_identifier: The identifier of the US Code section (e.g., '162' or the primary key ID).

    Returns:
        A string summarizing the aggregated statistics and listing details
        for each linked form field, or an error message.
    """
    db: Session = get_session()
    if not db:
        return "Error: Could not get database session."

    section_id_int: int | None = None
    section_number_str: str | None = None

    # 1. Find Section ID and Number
    try:
        section_id_int = int(section_identifier)
        logger.debug(f"Interpreted section identifier '{section_identifier}' as primary key ID.")
        section = db.query(UsCodeSection).filter(UsCodeSection.id == section_id_int).first()
        if section:
            section_number_str = section.section_number
        else:
            logger.warning(f"No section found with ID: {section_id_int}")
            section_id_int = None # Reset to try searching by number string
    except ValueError:
        logger.debug(f"Section identifier '{section_identifier}' is not an integer ID, searching by number.")

    if section_id_int is None:
        section_number_str = section_identifier # Assume input was section number
        section = db.query(UsCodeSection).filter(sql_func.lower(UsCodeSection.section_number) == section_number_str.lower()).first()
        if section:
            section_id_int = section.id
            logger.debug(f"Found section ID {section_id_int} for section number '{section_number_str}'.")
        else:
            logger.warning(f"No section found matching identifier: '{section_identifier}'.")
            db.close()
            return f"Error: Could not find a US Code section matching identifier '{section_identifier}'."

    section_display_name = section_number_str or section_identifier

    # 2. Query for Detailed Field Information and Statistics
    try:
        details_query = (
            db.query(
                FormField.id.label("field_id"),
                FormField.field_label,
                FormField.full_text,
                FormInstruction.title.label("instruction_title"),
                FormInstruction.form_number,
                FormFieldStatistics.dollars,
                FormFieldStatistics.forms,
                FormFieldStatistics.people
            )
            .select_from(FormField)
            .join(FormFieldUsCodeSectionLink, FormField.id == FormFieldUsCodeSectionLink.form_field_id)
            .filter(FormFieldUsCodeSectionLink.us_code_section_id == section_id_int)
            .outerjoin(FormInstruction, FormField.instruction_id == FormInstruction.id)
            .outerjoin(FormFieldStatistics, FormField.id == FormFieldStatistics.form_field_id)
            .order_by(FormField.id) # Consistent ordering
        )

        results = details_query.all()

        if not results:
            db.close()
            return f"Section '{section_display_name}' (ID: {section_id_int}) found, but no form fields are linked to it in the database."

        logger.info(f"Found {len(results)} linked form field details for section ID {section_id_int}.")

        # 3. Process Results: Aggregate and Itemize
        total_dollars = Decimal(0)
        total_forms = Decimal(0)
        total_people = Decimal(0)
        itemized_details = []

        for row in results:
            field_data = row._asdict()
            # Aggregate (handle Nones)
            total_dollars += field_data.get('dollars') or Decimal(0)
            total_forms += field_data.get('forms') or Decimal(0)
            total_people += field_data.get('people') or Decimal(0)

            # Prepare itemized entry
            # Truncate full_text for readability in the final output
            full_text_snippet = (field_data.get('full_text') or "")[:150] + ("..." if len(field_data.get('full_text') or "") > 150 else "")
            item = (
                f"  - Field ID: {field_data.get('field_id')}\n"
                f"    Label: {field_data.get('field_label') or 'N/A'}\n"
                f"    Instruction: {(field_data.get('instruction_title') or 'N/A')} ({field_data.get('form_number') or 'N/A'})\n"
                f"    Stats (Dollars/Forms/People): {field_data.get('dollars') or 0:,.0f} / {field_data.get('forms') or 0:,.0f} / {field_data.get('people') or 0:,.0f}\n"
                f"    Full Text Snippet: {full_text_snippet}"
            )
            itemized_details.append(item)

        # 4. Format Output
        summary_str = (
            f"Statistics Summary for Section '{section_display_name}' (ID: {section_id_int}):\n"
            f"- Based on {len(results)} linked form field(s).\n"
            f"- Total Dollars: {total_dollars:,.0f}\n"
            f"- Total Forms: {total_forms:,.0f}\n"
            f"- Total People: {total_people:,.0f}"
        )

        details_str = "\n\nLinked Form Field Details:\n" + "\n".join(itemized_details)

        logger.info(f"Successfully retrieved details and aggregated stats for section {section_id_int}.")
        return summary_str + details_str

    except Exception as e:
        logger.error(f"Error fetching detailed stats for section ID {section_id_int}: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while fetching detailed statistics for section '{section_display_name}'."
    finally:
        if db:
            db.close()


# Example Usage (Optional)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG to see identifier lookups

    # Test with an ID assumed to exist
    test_id = "6"
    print(f"--- Testing Detailed Stats with ID: {test_id} ---")
    print(get_section_details_and_stats(test_id))
    print("-"*50)

    # Test with a section number assumed to exist
    test_num = "162" # Replace with a section number in your DB
    print(f"--- Testing Detailed Stats with Section Number: {test_num} ---")
    print(get_section_details_and_stats(test_num))
    print("-"*50)

    # Test with a non-existent identifier
    test_not_found = "999999"
    print(f"--- Testing Detailed Stats with Non-existent ID: {test_not_found} ---")
    print(get_section_details_and_stats(test_not_found))
    print("-"*50) 