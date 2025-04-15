#!/usr/bin/env python
"""
Notebook-friendly version of the amendment analysis script.

Analyzes a few pre-defined U.S. Code sections using an LLM to estimate
amendment counts and provide reasoning.

Reads section data from a JSON file (demo_sections_data.json) in the same directory.
Outputs results directly to the console.
"""

import os
import sys
import json # Added for loading data
from typing import List, Optional, Dict, Any # Added Dict, Any

# Removed SQLAlchemy imports
from pydantic import BaseModel, Field

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Project components
# Removed settings import if only used for DB
# Removed database model and session imports
from ai_tax_agent.llm_utils import get_gemini_llm

# --- Configuration --- 
MODEL_NAME = "gemini-1.5-flash-latest"
LLM_TEMPERATURE = 0.1 
# Sections to analyze are implicitly defined by the content of the JSON file
JSON_DATA_FILENAME = os.path.join(os.path.dirname(__file__), "demo_sections_data.json")

# --- Pydantic Output Schema (Added Reasoning) --- 
class AmendmentAnalysisOutput(BaseModel):
    """Schema for the LLM output, including estimated count and reasoning."""
    amendment_count: int = Field(description="The estimated number of distinct amendments mentioned or detailed in the text.")
    reasoning: str = Field(description="A brief explanation justifying the estimated amendment count, citing relevant parts of the text if possible.")

# --- Prompt Template (Updated for Reasoning) --- 
SYSTEM_PROMPT = """You are an expert legal assistant specialized in analyzing U.S. legislative text.
Your task is to analyze the provided full text of a U.S. Code section, estimate the number of distinct amendments it describes, and explain your reasoning.

1.  Carefully read the entire text, paying attention to sections like 'Amendments', 'Historical and Statutory Notes', or notes detailing changes made by Public Laws (e.g., 'Pub. L. 115–97...').
2.  Estimate the total number of *distinct* amendments described.
3.  Provide a brief reasoning for your count, mentioning the key indicators you used (e.g., number of Pub. L. entries, specific subsections).
4.  Format your response strictly according to the provided JSON schema, including both `amendment_count` and `reasoning` fields.
"""

HUMAN_TEMPLATE = """Analyze the following U.S. Code section text:

```text
{section_text}
```

Please provide the estimated amendment count and your reasoning in the required JSON format.
{format_instructions}
"""

# --- Database Function Removed --- 
# def get_demo_sections(...) removed

# --- Main Agent Logic (Simplified for Notebook, Reads from JSON) ---
def run_amendment_analysis_notebook():
    """Runs the amendment analysis for sections defined in JSON and prints results."""
    print(f"Starting notebook demo analysis using model: {MODEL_NAME}")
    
    # --- Load data from JSON --- 
    print(f"Loading section data from {JSON_DATA_FILENAME}...")
    try:
        with open(JSON_DATA_FILENAME, 'r', encoding='utf-8') as f:
            sections_data: List[Dict[str, Any]] = json.load(f)
        if not sections_data:
             print(f"Error: {JSON_DATA_FILENAME} is empty or contains no data.")
             return
        print(f"Loaded data for {len(sections_data)} sections.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {JSON_DATA_FILENAME}")
        print("Please run 'scripts/notebook/fetch_section_data_notebook.py' first.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {JSON_DATA_FILENAME}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading data: {e}")
        return
    # -------------------------

    llm = get_gemini_llm(model_name=MODEL_NAME, temperature=LLM_TEMPERATURE)
    if not llm:
        return 

    # Setup Prompt and Parser (using the updated Pydantic model)
    parser = PydanticOutputParser(pydantic_object=AmendmentAnalysisOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE) # Template placeholders handled by LangChain
    ])
    
    chain = prompt | llm | parser
    
    # Removed session = get_session()
    # Removed call to get_demo_sections
    
    print("\n--- Analysis Results ---")
    # Iterate through data loaded from JSON
    for section_data in sections_data:
        section_number = section_data.get("section_number", "UNKNOWN")
        section_title = section_data.get("section_title", "N/A")
        full_text = section_data.get("full_text")
        
        print(f"\nAnalyzing Section: {section_number} (Title: {section_title})")
        
        if not full_text or len(full_text.strip()) == 0:
            print("  Skipping: No full text available in JSON data.")
            continue
            
        try:
            # Provide values for placeholders during invoke
            analysis_result: AmendmentAnalysisOutput = chain.invoke({
                "section_text": full_text, # Use text from JSON
                "format_instructions": parser.get_format_instructions()
            })
            
            print(f"  Estimated Amendment Count: {analysis_result.amendment_count}")
            print(f"  Reasoning: {analysis_result.reasoning}")
            
        except Exception as e:
            print(f"  Error processing section {section_number}: {e}")
            # import traceback # Uncomment for debugging
            # print(traceback.format_exc())
        
    print("\n--- Analysis Complete ---")
    # Removed session.close()

if __name__ == "__main__":
    run_amendment_analysis_notebook() 