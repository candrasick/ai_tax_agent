#!/usr/bin/env python
"""
Notebook-friendly script to compare two prompts for extracting blue amounts
from a pre-saved page image using a multimodal LLM.
"""

import os
import sys
import argparse
import logging
import json
import re
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

# Add project root to path (adjust if necessary)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LLM/LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from ai_tax_agent.llm_utils import get_gemini_llm
from ai_tax_agent.settings import settings

# Ensure Pillow is installed: pip install Pillow
try:
    from PIL import Image
except ImportError:
    logging.error("Pillow library not found. Please install it: pip install Pillow")
    sys.exit(1)

# --- Constants & Config --- 
MODEL_NAME = "gemini-1.5-flash-latest"
LLM_TEMPERATURE = 0.1

def run_blue_amount_extraction(
    llm: ChatGoogleGenerativeAI,
    image_path: str,
    prompt_text: str
) -> Optional[List[Dict[str, Any]]]:
    """Runs the blue amount extraction logic using the provided image and prompt."""
    extracted_amounts = []
    try:
        # 1. Load and encode image
        with Image.open(image_path) as img:
            buffer = BytesIO()
            # Ensure saving as PNG for consistency
            img.save(buffer, format="PNG") 
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            img_data_uri = f"data:image/png;base64,{img_base64}"

        # 2. Construct Prompt Messages
        system_prompt = (
            "You are an expert assistant specialized in analyzing PDF tax form images. "
            "Your task is to identify all text elements that appear distinctly blue, extract their text content, "
            "and estimate their bounding box coordinates."
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": img_data_uri}},
            ]
        )
        
        # 3. Call LLM
        logging.info(f"Calling multimodal LLM ({MODEL_NAME})...")
        response = llm.invoke([SystemMessage(content=system_prompt), message])
        response_content = response.content
        logging.debug(f"LLM Response received: {response_content[:500]}...")

        # 4. Parse and Filter Response (Adapted from pdf_parser_utils.py)
        parsed_data = None
        try:
            json_match = re.search(r"```json\s*\n?(\[.*?\])\n?\s*```", response_content, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_string = json_match.group(1).strip()
                if json_string:
                    try: parsed_data = json.loads(json_string)
                    except json.JSONDecodeError as e: logging.error(f"Failed to parse extracted JSON blue amount list: {e}\nString: {json_string}")
                else: logging.warning("Regex matched JSON block for blue amounts, but captured string was empty.")
            else:
                logging.warning("Could not find JSON list block ```json...``` in blue amount response. Attempting to parse entire response.")
                cleaned_response = response_content.strip()
                if cleaned_response:
                    try: parsed_data = json.loads(cleaned_response)
                    except json.JSONDecodeError as e: logging.error(f"Failed to parse entire LLM blue amount response as JSON: {e}\nResponse: {response_content}")
                else: logging.warning("LLM blue amount response was empty after stripping whitespace.")

            if isinstance(parsed_data, list):
                count_before_filter = len(parsed_data)
                for item in parsed_data:
                    if not isinstance(item, dict) or 'amount' not in item or 'position' not in item:
                        logging.warning(f"Skipping invalid item in LLM blue amount list: {item}")
                        continue
                    pos = item['position']
                    if not isinstance(pos, list) or len(pos) != 4:
                        logging.warning(f"Skipping item with invalid position: {item}")
                        continue
                    try: item['position'] = [float(p) for p in pos]
                    except (ValueError, TypeError): logging.warning(f"Skipping item with non-numeric position: {item}"); continue

                    amount_text = str(item['amount']).strip()
                    cleaned_text = amount_text.replace(",", "")
                    if cleaned_text.startswith('(') and cleaned_text.endswith(')'): cleaned_text = '-' + cleaned_text[1:-1]
                    is_numeric = False
                    try: float(cleaned_text); is_numeric = True
                    except ValueError: pass

                    if is_numeric: extracted_amounts.append({"amount": amount_text, "position": item['position']})
                    else: logging.debug(f"Skipping non-numeric blue text: '{amount_text}'")
                logging.info(f"LLM found {count_before_filter} blue elements, kept {len(extracted_amounts)} potential numeric amounts after filtering.")
            else:
                logging.error(f"Parsed LLM blue amount response was not a list. Parsed: {str(parsed_data)[:200]}...")

        except Exception as e:
            logging.error(f"Error processing LLM blue amount response content: {e}", exc_info=True)

    except FileNotFoundError:
        logging.error(f"Image file not found at: {image_path}")
        return None # Indicate failure
    except Exception as e:
        logging.error(f"An error occurred during LLM blue amount extraction: {e}", exc_info=True)
        return None # Indicate failure

    return extracted_amounts

def main():
    parser = argparse.ArgumentParser(description="Compare two prompts for extracting blue amounts from an image.")
    parser.add_argument("--image-path", required=True, help="Path to the input PNG image file (generated by save_page_image_notebook.py).")
    parser.add_argument("--prompt1-path", required=True, help="Path to the first prompt file (e.g., prompts/extract_blue_amounts.txt).")
    parser.add_argument("--prompt2-path", required=True, help="Path to the second prompt file (e.g., prompts/extract_blue_amounts_v2.txt).")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image_path):
        logging.error(f"Image file not found: {args.image_path}"); sys.exit(1)
    if not os.path.exists(args.prompt1_path):
        logging.error(f"Prompt file 1 not found: {args.prompt1_path}"); sys.exit(1)
    if not os.path.exists(args.prompt2_path):
        logging.error(f"Prompt file 2 not found: {args.prompt2_path}"); sys.exit(1)

    # Initialize LLM
    llm = get_gemini_llm(model_name=MODEL_NAME, temperature=LLM_TEMPERATURE)
    if not llm:
        logging.error("Failed to initialize LLM. Ensure API key is set."); sys.exit(1)

    # Read prompts
    try:
        with open(args.prompt1_path, 'r', encoding='utf-8') as f: prompt1_text = f.read()
        with open(args.prompt2_path, 'r', encoding='utf-8') as f: prompt2_text = f.read()
    except Exception as e:
        logging.error(f"Error reading prompt files: {e}"); sys.exit(1)

    # Run extraction for Prompt 1
    print(f"\n--- Running Extraction with Prompt 1: {os.path.basename(args.prompt1_path)} ---")
    results1 = run_blue_amount_extraction(llm, args.image_path, prompt1_text)
    print("\nResults (JSON):")
    if results1 is not None:
        print(json.dumps(results1, indent=2))
    else:
        print("[] # Extraction failed for Prompt 1")
    print("------------------------------------------------------------")

    # Run extraction for Prompt 2
    print(f"\n--- Running Extraction with Prompt 2: {os.path.basename(args.prompt2_path)} ---")
    results2 = run_blue_amount_extraction(llm, args.image_path, prompt2_text)
    print("\nResults (JSON):")
    if results2 is not None:
        print(json.dumps(results2, indent=2))
    else:
        print("[] # Extraction failed for Prompt 2")
    print("------------------------------------------------------------")

    print("\nComparison complete.")

if __name__ == "__main__":
    main() 