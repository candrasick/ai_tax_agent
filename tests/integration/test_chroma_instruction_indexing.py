import pytest
import subprocess
import sys
import os
from sqlalchemy.orm import Session, joinedload
import chromadb

# Add project root to Python path to import project modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import FormField, FormInstruction
# Assuming chroma constants are accessible or redefined here
# We might need to import directly from script or redefine if not packaged
# from scripts.index_instructions_chroma import COLLECTION_NAME, CHROMA_DATA_PATH, get_chroma_client # Ideal, but script imports might fail in test context

# --- Redefine Chroma constants or import carefully ---
# It's often better to have constants in a shared config/settings module
COLLECTION_NAME = "form_instructions"
CHROMA_DATA_PATH = "chroma_data"

# Simplified client getter for test context
def get_test_chroma_client() -> chromadb.Client:
    return chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# --- Test Case ---

def test_form_field_indexing_completeness_and_metadata():
    """
    Tests that the index_instructions_chroma script correctly indexes
    all relevant FormFields into ChromaDB with accurate metadata.
    """
    # 1. Run the indexing script with --clear
    script_path = os.path.join(project_root, "scripts", "index_instructions_chroma.py")
    # Use the Python executable running the tests to run the script
    python_executable = sys.executable
    print(f"\nRunning indexing script: {python_executable} {script_path} --clear --log-level WARNING")
    result = subprocess.run(
        [python_executable, script_path, "--clear", "--log-level", "WARNING"],
        capture_output=True,
        text=True,
        check=False # Don't raise exception on failure, check returncode instead
    )

    # Print script output for debugging if needed
    if result.returncode != 0:
        print("Indexing script stdout:")
        print(result.stdout)
        print("Indexing script stderr:")
        print(result.stderr)

    assert result.returncode == 0, f"Indexing script failed with exit code {result.returncode}"
    print("Indexing script completed successfully.")

    # 2. Query relational DB for expected data
    db: Session = next(get_session())
    expected_fields_query = (
        db.query(FormField)
        .join(FormInstruction, FormField.instruction_id == FormInstruction.id)
        .options(joinedload(FormField.instruction))
        .filter(FormField.full_text != None, FormField.full_text != "")
    )
    expected_fields_list = expected_fields_query.all()
    expected_count = len(expected_fields_list)
    # Create a dictionary for easy lookup during metadata check
    expected_fields_dict = {f"field_{field.id}": field for field in expected_fields_list}
    db.close()

    print(f"Expected count from database: {expected_count}")
    assert expected_count > 0, "Test requires some FormField data with text to be present in the DB"

    # 3. Query ChromaDB for actual data
    chroma_client = get_test_chroma_client()
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        actual_count = collection.count()
        print(f"Actual count from ChromaDB: {actual_count}")

        # 4. Assert Counts Match
        print("Asserting total counts match...")
        assert actual_count == expected_count, f"Mismatch: DB expects {expected_count} fields, ChromaDB has {actual_count} documents."
        print("Total counts match.")

        # 5. Spot Check Metadata (Check all items)
        if actual_count > 0:
            sample_size = min(10, actual_count) # Check at most 10 items
            print(f"Fetching metadata for {sample_size} sample items from ChromaDB for spot check...")
            results = collection.get(limit=sample_size, include=['metadatas']) # Fetch only a sample

            assert len(results['ids']) == sample_size, f"ChromaDB get() returned unexpected number of sample IDs (expected {sample_size})"

            missing_in_db = []
            metadata_mismatches = []

            for i, chroma_id in enumerate(results['ids']):
                chroma_metadata = results['metadatas'][i]

                # Find corresponding expected field
                expected_field = expected_fields_dict.get(chroma_id)

                if not expected_field:
                    missing_in_db.append(chroma_id)
                    continue # Cannot compare metadata if not found in DB dict

                # Compare metadata fields
                expected_metadata = {
                    "field_id": expected_field.id,
                    "field_label": expected_field.field_label,
                    "form_number": expected_field.instruction.form_number,
                    "form_title": expected_field.instruction.title,
                    "text_length": len(expected_field.full_text)
                }

                # Check if all expected keys exist and match
                mismatched_keys = []
                for key, expected_value in expected_metadata.items():
                    chroma_value = chroma_metadata.get(key)
                    if chroma_value != expected_value:
                        mismatched_keys.append(f"Key '{key}': Expected '{expected_value}', Got '{chroma_value}'")

                if mismatched_keys:
                     metadata_mismatches.append({
                         "chroma_id": chroma_id,
                         "expected": expected_metadata,
                         "actual": chroma_metadata,
                         "errors": mismatched_keys
                     })

            assert not missing_in_db, f"ChromaDB sample contains IDs not found in the expected DB query results: {missing_in_db}"
            assert not metadata_mismatches, f"Metadata mismatches found:\n{metadata_mismatches}"

            print(f"Metadata spot check passed for {sample_size} sampled items.")

    except Exception as e:
        pytest.fail(f"Error interacting with ChromaDB: {e}")

    finally:
         # Optional: Clean up Chroma collection if needed, though --clear should handle it
         # try:
         #     chroma_client.delete_collection(name=COLLECTION_NAME)
         #     print(f"Cleaned up ChromaDB collection: {COLLECTION_NAME}")
         # except:
         #     pass # Ignore errors during cleanup
         pass # Relying on --clear in the next run 