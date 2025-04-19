#!/usr/bin/env python
"""Tools for interacting with the application's relational database."""

import logging
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List

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
    Exemption,
    SectionImpact,
    SectionComplexity
)

logger = logging.getLogger(__name__)


def get_section_details_and_stats(section_identifier: str) -> Dict[str, Any] | str:
    """Fetches detailed info, core text, exemptions, and aggregated stats for a US Code section.

    Args:
        section_identifier: The identifier of the US Code section (e.g., '162' or the primary key ID).

    Returns:
        A dictionary containing section details, stats, core text, and exemptions,
        or an error string.
    """
    db: Session = get_session()
    if not db:
        return "Error: Could not get database session."

    section_id_int: int | None = None
    section_number_str: str | None = None
    core_text: str | None = None
    section_title: str | None = None

    # 1. Find Section ID, Number, and Core Text
    try:
        section_id_int = int(section_identifier)
        logger.debug(f"Interpreted section identifier '{section_identifier}' as primary key ID.")
        section = db.query(UsCodeSection).filter(UsCodeSection.id == section_id_int).first()
        if section:
            section_number_str = section.section_number
            core_text = section.core_text
            section_title = section.section_title
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
            core_text = section.core_text
            section_title = section.section_title
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

        field_results = details_query.all()

        # Handle case where section exists but has no linked fields
        if not field_results:
            logger.warning(f"Section '{section_display_name}' (ID: {section_id_int}) found, but no form fields are linked.")
            # Proceed to fetch exemptions even if no fields are linked

        # 3. Process Field Results: Aggregate and Itemize
        total_dollars = Decimal(0)
        total_forms = Decimal(0)
        total_people = Decimal(0)
        itemized_field_details: List[Dict[str, Any]] = [] # Store as list of dicts

        for row in field_results:
            field_data = row._asdict()
            # Aggregate (handle Nones)
            dollars_val = field_data.get('dollars') or Decimal(0)
            forms_val = field_data.get('forms') or Decimal(0)
            people_val = field_data.get('people') or Decimal(0)
            total_dollars += dollars_val
            total_forms += forms_val
            total_people += people_val

            # Prepare itemized entry
            full_text_snippet = (field_data.get('full_text') or "")[:150] + ("..." if len(field_data.get('full_text') or "") > 150 else "")
            item = {
                "field_id": field_data.get('field_id'),
                "label": field_data.get('field_label') or 'N/A',
                "instruction_title": field_data.get('instruction_title') or 'N/A',
                "form_number": field_data.get('form_number') or 'N/A',
                "stats_dollars": float(dollars_val) if dollars_val is not None else 0.0, # Convert Decimal to float for JSON
                "stats_forms": float(forms_val) if forms_val is not None else 0.0,
                "stats_people": float(people_val) if people_val is not None else 0.0,
                "full_text_snippet": full_text_snippet
            }
            itemized_field_details.append(item)

        # 4. Query for Exemptions
        exemptions_query = (
            db.query(
                Exemption.id,
                Exemption.relevant_text,
                Exemption.rationale
            )
            .filter(Exemption.section_id == section_id_int)
            .order_by(Exemption.id)
        )
        exemption_results = exemptions_query.all()
        itemized_exemptions: List[Dict[str, Any]] = [
            {
                "exemption_id": ex.id,
                "relevant_text": ex.relevant_text,
                "rationale": ex.rationale
            } for ex in exemption_results
        ]
        logger.info(f"Found {len(itemized_exemptions)} exemptions for section ID {section_id_int}.")


        # 5. Assemble Output Dictionary
        output_data = {
            "section_identifier": section_display_name,
            "section_id": section_id_int,
            "core_text": core_text,
            "section_title": section_title,
            "aggregation_summary": {
                "linked_field_count": len(field_results),
                "total_dollars": float(total_dollars) if total_dollars is not None else 0.0, # Convert Decimal to float
                "total_forms": float(total_forms) if total_forms is not None else 0.0,
                "total_people": float(total_people) if total_people is not None else 0.0,
            },
            "linked_form_fields": itemized_field_details,
            "exemptions": itemized_exemptions
        }


        logger.info(f"Successfully retrieved details and aggregated stats for section {section_id_int}.")
        return output_data # Return the dictionary

    except Exception as e:
        logger.error(f"Error fetching detailed stats for section ID {section_id_int}: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while fetching detailed statistics for section '{section_display_name}'." # Return error string
    finally:
        if db:
            db.close()


# --- New Function for Simplification Context --- 

def get_section_simplification_context(section_identifier: str) -> Dict[str, Any] | str:
    """Fetches core text, impact, complexity, and exemption details for a section."""
    db: Session = get_session()
    if not db:
        return "Error: Could not get database session."

    section_id_int: int | None = None
    section_number_str: str | None = None
    section_data = {}

    # 1. Find Section ID and Basic Info
    try:
        # Find Section ID (similar logic as get_section_details_and_stats)
        try:
            section_id_int = int(section_identifier)
            section = db.query(UsCodeSection).filter(UsCodeSection.id == section_id_int).first()
            if section: section_number_str = section.section_number
        except ValueError:
            section_number_str = section_identifier
            section = db.query(UsCodeSection).filter(sql_func.lower(UsCodeSection.section_number) == section_number_str.lower()).first()
            if section: section_id_int = section.id

        if not section:
            return f"Error: Could not find UsCodeSection matching '{section_identifier}'."

        section_data['section_id'] = section_id_int
        section_data['section_identifier'] = section_number_str or section_identifier
        section_data['core_text'] = section.core_text

        # 2. Fetch Section Impact (Handle if missing)
        impact = db.query(SectionImpact).filter(SectionImpact.section_id == section_id_int).first()
        section_data['section_revenue_impact'] = float(impact.revenue_impact) if impact and impact.revenue_impact is not None else None
        section_data['section_entity_impact'] = float(impact.entity_impact) if impact and impact.entity_impact is not None else None
        logger.debug(f"Fetched SectionImpact for {section_id_int}: Rev={section_data['section_revenue_impact']}, Ent={section_data['section_entity_impact']}")

        # 3. Fetch Section Complexity (Handle if missing)
        complexity = db.query(SectionComplexity).filter(SectionComplexity.section_id == section_id_int).first()
        section_data['complexity_score'] = complexity.complexity_score if complexity else None
        logger.debug(f"Fetched SectionComplexity for {section_id_int}: Score={section_data['complexity_score']}")

        # 4. Fetch Exemptions (with their impacts)
        exemptions_query = db.query(Exemption).filter(Exemption.section_id == section_id_int).order_by(Exemption.id)
        exemptions_list = []
        for ex in exemptions_query.all():
            exemptions_list.append({
                "exemption_id": ex.id,
                "relevant_text": ex.relevant_text,
                # Use revenue_impact_estimate and entity_impact from Exemption table
                "revenue_impact": float(ex.revenue_impact_estimate) if ex.revenue_impact_estimate is not None else None,
                "entity_impact": float(ex.entity_impact) if ex.entity_impact is not None else None,
                "rationale": ex.rationale # Keep original rationale if needed
            })
        section_data['exemptions'] = exemptions_list
        logger.info(f"Found {len(exemptions_list)} exemptions with impact data for section {section_id_int}.")

        return section_data

    except Exception as e:
        logger.error(f"Error fetching simplification context for '{section_identifier}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred while fetching context for '{section_identifier}'."
    finally:
        if db:
            db.close()


# Example Usage (Optional)
if __name__ == '__main__':
    import json # Add json for pretty printing
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG to see identifier lookups

    # Test with an ID assumed to exist
    test_id = "6"
    print(f"--- Testing Detailed Stats with ID: {test_id} ---")
    result_id = get_section_details_and_stats(test_id)
    print(json.dumps(result_id, indent=2) if isinstance(result_id, dict) else result_id) # Pretty print JSON
    print("-"*50)

    # Test with a section number assumed to exist
    test_num = "162" # Replace with a section number in your DB
    print(f"--- Testing Detailed Stats with Section Number: {test_num} ---")
    result_num = get_section_details_and_stats(test_num)
    print(json.dumps(result_num, indent=2) if isinstance(result_num, dict) else result_num) # Pretty print JSON
    print("-"*50)

    # Test with a non-existent identifier
    test_not_found = "999999"
    print(f"--- Testing Detailed Stats with Non-existent ID: {test_not_found} ---")
    result_not_found = get_section_details_and_stats(test_not_found)
    print(json.dumps(result_not_found, indent=2) if isinstance(result_not_found, dict) else result_not_found) # Pretty print JSON
    print("-"*50)

    # Test section with exemptions but maybe no fields (replace '1' with actual ID)
    test_exempt_only = "1"
    print(f"--- Testing Detailed Stats with Section ID: {test_exempt_only} (May have exemptions only) ---")
    result_exempt = get_section_details_and_stats(test_exempt_only)
    print(json.dumps(result_exempt, indent=2) if isinstance(result_exempt, dict) else result_exempt) # Pretty print JSON
    print("-"*50)

    # --- Test Block ---
    import logging
    from pprint import pprint # For readable output

    # Configure logging for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    test_identifier_str = '1' # The input the agent likely used
    test_identifier_int = 1   # Let's also test with an integer

    print(f"\n--- Testing get_section_details_and_stats with identifier: '{test_identifier_str}' ---")
    try:
        result_str = get_section_details_and_stats(test_identifier_str)
        print("--- Result (as string): ---")
        # The function returns a string (likely JSON), so just print it
        # If it's meant to return a dict/object, we'd pprint result_str directly after potential json.loads
        print(result_str)
    except Exception as e:
        logger.error(f"Error calling with '{test_identifier_str}': {e}", exc_info=True)
        print(f"Error calling with '{test_identifier_str}': {e}")

    # Optional: Test with integer if relevant to implementation
    # print(f"\n--- Testing get_section_details_and_stats with identifier: {test_identifier_int} ---")
    # try:
    #     result_int = get_section_details_and_stats(test_identifier_int)
    #     print("--- Result (as string): ---")
    #     print(result_int)
    # except Exception as e:
    #     logger.error(f"Error calling with {test_identifier_int}: {e}", exc_info=True)
    #     print(f"Error calling with {test_identifier_int}: {e}")

    print("\n--- Test Complete ---") 