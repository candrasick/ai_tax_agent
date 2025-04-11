import pytest
import os
import sys
import chromadb
from sqlalchemy.orm import Session

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from ai_tax_agent.database.session import get_session
from ai_tax_agent.database.models import UsCodeSection
from scripts.index_sections_chroma import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION_NAME

@pytest.fixture(scope="module")
def db_session() -> Session:
    """Provides a database session for the tests."""
    session = get_session()
    yield session
    session.close()

@pytest.fixture(scope="module")
def chroma_collection() -> chromadb.Collection:
    """Provides a connection to the ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
        collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
        return collection
    except Exception as e:
        pytest.fail(f"Failed to connect to ChromaDB collection '{DEFAULT_COLLECTION_NAME}' at '{DEFAULT_CHROMA_PATH}': {e}")

def test_all_db_sections_in_chroma(db_session: Session, chroma_collection: chromadb.Collection):
    """Verify that every section in the DB has a corresponding entry in ChromaDB."""
    
    # 1. Get expected IDs from the database
    print("\nFetching expected IDs from database...")
    db_sections = db_session.query(UsCodeSection.id).filter(UsCodeSection.core_text != None, UsCodeSection.core_text != "").all() # Only check sections that should be indexed
    expected_ids_set = {f"usc_{section_id}" for (section_id,) in db_sections}
    print(f"Found {len(expected_ids_set)} expected indexable sections in DB.")
    
    if not expected_ids_set:
         pytest.fail("No indexable sections found in the database to check against ChromaDB.")

    # 2. Get actual IDs from ChromaDB
    print(f"Fetching actual IDs from ChromaDB collection '{DEFAULT_COLLECTION_NAME}'...")
    try:
        # Fetch all IDs from the collection. get() without IDs should return all.
        # We don't need embeddings or documents here, just the IDs.
        results = chroma_collection.get(include=[]) # include=[] might optimize by only fetching IDs
        actual_ids_set = set(results['ids'])
        print(f"Found {len(actual_ids_set)} actual IDs in ChromaDB.")
    except Exception as e:
        pytest.fail(f"Error fetching IDs from ChromaDB collection: {e}")

    # 3. Compare the sets
    print("Comparing database IDs and ChromaDB IDs...")
    assert expected_ids_set == actual_ids_set, \
        f"Mismatch between DB and ChromaDB IDs.\nMissing in Chroma: {len(expected_ids_set - actual_ids_set)} IDs\nExtra in Chroma: {len(actual_ids_set - expected_ids_set)} IDs"

    print("Verification successful: All indexable DB sections found in ChromaDB.") 