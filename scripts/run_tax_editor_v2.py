# scripts/run_tax_editor.py
import logging
import argparse
import json
import os
import sys
from decimal import Decimal
from typing import Optional, Tuple
import warnings
import re # Add import for regex

# Setup project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress the specific deprecation warning from langchain_google_genai
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Convert_system_message_to_human will be deprecated!.*",
    module="langchain_google_genai.*" # Target the specific module
)

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc # For ordering

# Progress bar import
from tqdm import tqdm

# Database imports
from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import (
    UsCodeSection, 
    UsCodeSectionRevised, 
    SectionImpact, 
    SectionHistory,
    Exemption,  # Add import for Exemption model
)
from ai_tax_agent.database.versioning import determine_version_numbers

# Agent imports
from ai_tax_agent.agents import create_tax_editor_agent # Keep agent import
from ai_tax_agent.outputs import TaxEditorOutput # Import the output model from the new file
from ai_tax_agent.settings import settings # May need settings for agent config

# LangChain Parser import
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException # Import for specific error handling

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def clear_working_version_edits(db: Session, working_version: int):
    """Deletes SectionVersion records for the current working version."""
    logger.info(f"Clearing existing edits for working version {working_version}...")
    try:
        deleted_count = db.query(UsCodeSectionRevised).filter(UsCodeSectionRevised.version == working_version).delete()
        db.commit()
        logger.info(f"Successfully deleted {deleted_count} UsCodeSectionRevised records for version {working_version}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to clear edits for version {working_version}: {e}", exc_info=True)
        raise


def get_section_history(session: Session, section_id: int) -> str:
    """Get formatted history string for a section."""
    history_entries = session.query(SectionHistory).filter(
        SectionHistory.orig_section_id == section_id
    ).order_by(SectionHistory.version_changed).all()
    
    if not history_entries:
        return "No previous changes"
        
    history_lines = []
    for entry in history_entries:
        history_lines.append(
            f"Version {entry.version_changed}: {entry.action.capitalize()} - {entry.rationale}"
        )
    
    return "\n".join(history_lines)


def prepare_editor_input(
    section_id: int,
    prior_version: int,
    prior_version_text: str,
    prior_version_impact: Optional[float],
    prior_version_complexity: Optional[float],
    related_exemptions: str,
    history: str
) -> dict:
    """Prepare input for the tax editor agent."""
    return {
        "section_id": section_id,
        "prior_version": prior_version,
        "prior_version_text": prior_version_text,
        "prior_version_impact": prior_version_impact if prior_version_impact is not None else "Unknown",
        "prior_version_complexity": prior_version_complexity if prior_version_complexity is not None else "Unknown",
        "related_exemptions": related_exemptions,
        "history": history
    }


def process_sections(session: Session, version: int, limit: Optional[int] = None) -> None:
    """Process sections for the given version number."""
    try:
        # Get sections to process - exclude those marked as deleted in UsCodeSectionRevised
        sections_query = session.query(UsCodeSection).filter(
            ~UsCodeSection.id.in_(
                session.query(UsCodeSectionRevised.orig_section_id).filter(
                    UsCodeSectionRevised.deleted == True
                )
            )
        ).options(
            joinedload(UsCodeSection.impact_assessment),
            joinedload(UsCodeSection.complexity_assessment)
        )
        
        if limit:
            sections_query = sections_query.limit(limit)
            
        sections = sections_query.all()
        logger.info(f"Processing {len(sections)} sections for version {version}")
        
        for section in sections:
            try:
                # Get prior version info
                prior_version = version - 1
                
                # First check if there's a prior revision
                prior_revision = session.query(UsCodeSectionRevised).filter(
                    UsCodeSectionRevised.orig_section_id == section.id,
                    UsCodeSectionRevised.version == prior_version,
                    UsCodeSectionRevised.deleted == False
                ).first()
                
                if prior_revision:
                    prior_version_text = prior_revision.core_text if prior_revision.core_text else "No text available"
                    prior_version_impact = prior_revision.revised_financial_impact
                    prior_version_complexity = prior_revision.revised_complexity
                else:
                    # Fall back to original section and relationships
                    prior_version_text = section.core_text if section.core_text else "No text available"
                    prior_version_impact = section.impact_assessment.revenue_impact if section.impact_assessment else None
                    prior_version_complexity = section.complexity_assessment.complexity_score if section.complexity_assessment else None
                
                # Only get exemptions for version 0
                if prior_version == 0:
                    # Get related exemptions
                    exemptions = session.query(Exemption).filter_by(section_id=section.id).all()
                    related_exemptions_str = format_exemptions(exemptions)
                else:
                    related_exemptions_str = "N/A"

                # Get section history
                history = get_section_history(session, section.id)
                
                # Prepare input for editor
                editor_input = prepare_editor_input(
                    section_id=section.id,
                    prior_version=prior_version,
                    prior_version_text=prior_version_text,
                    prior_version_impact=prior_version_impact,
                    prior_version_complexity=prior_version_complexity,
                    related_exemptions=related_exemptions_str,
                    history=history
                )
                
                # Get editor decision
                editor_output = get_editor_decision(editor_input)
                
                # Process the decision
                success, message = process_editor_decision(
                    session=session,
                    section_id=section.id,
                    version=version,
                    editor_output=editor_output,
                    prior_version_text=prior_version_text,
                    prior_version_impact=prior_version_impact,
                    prior_version_complexity=prior_version_complexity,
                    prior_version_exemptions=related_exemptions_str
                )
                
                if success:
                    session.commit()
                    logger.info(f"Section {section.id}: {message}")
                else:
                    session.rollback()
                    logger.error(f"Section {section.id}: {message}")
                    
            except Exception as e:
                session.rollback()
                logger.error(f"Error processing section {section.id}: {str(e)}")
                continue
                
        if limit:
            logger.info(f"Reached processing limit of {limit}.")
            
    except Exception as e:
        logger.error(f"Error in process_sections: {str(e)}")
        raise


def process_editor_decision(
    session: Session,
    section_id: int,
    version: int,
    editor_output: dict,
    prior_version_text: str,
    prior_version_impact: float,
    prior_version_complexity: float,
    prior_version_exemptions: str,
) -> Tuple[bool, str]:
    """Process the editor's decision for a section, updating the database accordingly.
    
    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        action = editor_output["action"]
        rationale = editor_output["rationale"]
        after_text = editor_output.get("after_text")
        estimated_deleted_dollars = editor_output.get("estimated_deleted_dollars")
        estimated_kept_dollars = editor_output.get("estimated_kept_dollars")
        new_complexity_score = editor_output.get("new_complexity_score")
        merged_section_id = editor_output.get("merged_section_id")

        # Get the original section for its metadata
        original_section = session.query(UsCodeSection).filter_by(id=section_id).first()
        if not original_section:
            return False, f"Original section {section_id} not found"

        # Create history entry for the action
        history_entry = SectionHistory(
            orig_section_id=section_id,
            version_changed=version,
            action=action,
            rationale=rationale
        )
        session.add(history_entry)

        # Handle merge action
        if action == "merge":
            if not merged_section_id:
                return False, "Merge action requires merged_section_id"
                
            # Get the section being merged in
            merged_section = session.query(UsCodeSection).filter_by(id=merged_section_id).first()
            if not merged_section:
                return False, f"Section {merged_section_id} not found for merging"
                
            # Create revised section record for the deleted (merged) section
            merged_section_revised = UsCodeSectionRevised(
                orig_section_id=merged_section_id,
                version=version,
                deleted=True,
                subtitle=merged_section.subtitle,
                chapter=merged_section.chapter,
                subchapter=merged_section.subchapter,
                part=merged_section.part,
                section_number=merged_section.section_number,
                section_title=merged_section.section_title,
                core_text=None,  # Text is now in the merged section
                revised_complexity=0.0,  # Deleted sections have 0 complexity
                revised_financial_impact=0  # Impact is counted in the merged section
            )
            session.add(merged_section_revised)
            
            # Create revised section with merged content
            revised = UsCodeSectionRevised(
                orig_section_id=section_id,
                version=version,
                deleted=False,
                subtitle=original_section.subtitle,
                chapter=original_section.chapter,
                subchapter=original_section.subchapter,
                part=original_section.part,
                section_number=original_section.section_number,
                section_title=original_section.section_title,
                core_text=after_text,
                revised_complexity=new_complexity_score,
                revised_financial_impact=estimated_kept_dollars
            )
            session.add(revised)
            
            return True, f"Successfully merged section {merged_section_id} into {section_id}"

        # Handle simplify/redraft actions
        if action in ["simplify", "redraft"]:
            revised = UsCodeSectionRevised(
                orig_section_id=section_id,
                version=version,
                deleted=False,
                subtitle=original_section.subtitle,
                chapter=original_section.chapter,
                subchapter=original_section.subchapter,
                part=original_section.part,
                section_number=original_section.section_number,
                section_title=original_section.section_title,
                core_text=after_text,
                revised_complexity=new_complexity_score,
                revised_financial_impact=estimated_kept_dollars
            )
            session.add(revised)
            
            return True, f"Successfully {action}ed section {section_id}"
            
        elif action == "delete":
            # Create a revised section record marked as deleted
            revised = UsCodeSectionRevised(
                orig_section_id=section_id,
                version=version,
                deleted=True,
                subtitle=original_section.subtitle,
                chapter=original_section.chapter,
                subchapter=original_section.subchapter,
                part=original_section.part,
                section_number=original_section.section_number,
                section_title=original_section.section_title,
                core_text=None,  # Deleted sections have no text
                revised_complexity=0.0,  # Deleted sections have 0 complexity
                revised_financial_impact=estimated_deleted_dollars
            )
            session.add(revised)
            
            return True, f"Successfully deleted section {section_id}"
            
        elif action == "keep":
            # Create a revised section record for the kept section
            revised = UsCodeSectionRevised(
                orig_section_id=section_id,
                version=version,
                deleted=False,
                subtitle=original_section.subtitle,
                chapter=original_section.chapter,
                subchapter=original_section.subchapter,
                part=original_section.part,
                section_number=original_section.section_number,
                section_title=original_section.section_title,
                core_text=prior_version_text,  # Keep the same text
                revised_complexity=prior_version_complexity,  # Keep the same complexity
                revised_financial_impact=prior_version_impact  # Keep the same impact
            )
            session.add(revised)
            
            return True, f"Kept section {section_id} unchanged"
            
        else:
            return False, f"Unknown action: {action}"
            
    except Exception as e:
        return False, f"Error processing editor decision: {str(e)}"


def get_editor_decision(editor_input: dict) -> dict:
    """Get the editor agent's decision for a section.
    
    Args:
        editor_input: Dictionary containing section information
        
    Returns:
        Dictionary containing the editor's decision
    """
    try:
        # Create the editor agent
        agent_executor = create_tax_editor_agent(llm_model_name="gemini-1.5-pro", verbose=True)  # Set verbose to True
        if not agent_executor:
            raise RuntimeError("Failed to create tax editor agent")
            
        # Format input for the agent
        relevant_text = f"""
        Section ID: {editor_input['section_id']}
        Prior Version: {editor_input['prior_version']}
        Original Text: {editor_input['prior_version_text']}
        Complexity Score (Current): {editor_input['prior_version_complexity']}
        Revenue Impact ($M): {editor_input['prior_version_impact']}
        Related Exemptions: {editor_input['related_exemptions']}
        History: {editor_input['history']}
        """
        
        logger.info(f"Sending input to agent for section {editor_input['section_id']}:\n{relevant_text}")
        
        # Get agent's decision
        response = agent_executor.invoke({
            "relevant_text": relevant_text.strip(),
            "agent_scratchpad": "",
            "tools": "",  # These will be filled in by the agent framework
            "tool_names": ""  # These will be filled in by the agent framework
        })
        
        logger.info(f"Raw agent response:\n{response}")
        agent_output = response.get('output')
        
        if not agent_output:
            raise ValueError("Agent did not return 'output' key in response")
            
        logger.info(f"Agent output:\n{agent_output}")
            
        # Parse JSON from agent output
        match = re.search(r'\{.*?\}', str(agent_output), re.DOTALL)
        if not match:
            raise ValueError("No valid JSON object found in agent output")
            
        potential_json_str = match.group(0)
        logger.info(f"Extracted JSON:\n{potential_json_str}")
        
        editor_output = json.loads(potential_json_str)
        logger.info(f"Parsed editor output:\n{json.dumps(editor_output, indent=2)}")
        
        # Validate required fields
        required_fields = ["action", "rationale"]
        for field in required_fields:
            if field not in editor_output:
                raise ValueError(f"Missing required field '{field}' in editor output")
                
        return editor_output
        
    except Exception as e:
        logger.error(f"Error getting editor decision: {str(e)}")
        # Return a safe default - keep the section unchanged
        return {
            "action": "keep",
            "rationale": f"Error in editor agent processing: {str(e)}",
            "after_text": None,
            "estimated_deleted_dollars": None,
            "estimated_kept_dollars": None,
            "new_complexity_score": None,
            "merged_section_id": None
        }


def main(limit: Optional[int] = None, clear: bool = False):
    """
    Main entry point for the tax editor script.
    
    Args:
        limit: Optional limit on number of sections to process
        clear: Whether to clear existing progress for the current working version
    """
    try:
        # Initial setup
        prior_version, working_version = determine_version_numbers()
        logger.info(f"Determined versions: Prior={prior_version}, Working={working_version}")
        
        if working_version == 0:
            logger.error("Working version is 0. Cannot run editor agent without an initial version.")
            return
            
        # Clear previous edits if requested
        if clear:
            with get_session() as session:
                clear_working_version_edits(session, working_version)
                session.commit()
                
        # Process sections
        with get_session() as session:
            process_sections(session, working_version, limit)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tax code editor agent")
    parser.add_argument("--limit", type=int, help="Limit the number of sections to process")
    parser.add_argument("--clear", action="store_true", help="Clear existing progress for the current working version")
    args = parser.parse_args()
    
    main(limit=args.limit, clear=args.clear)

    logger.info("Tax Editor Agent run complete.") 