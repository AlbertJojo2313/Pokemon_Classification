"""
Configuration for test, only does the dataset test if the hash changes
"""
import pytest
import hashlib
import json
from pathlib import Path

# Path to the file being watched and where the hash is stored
WATCHED_FILE = Path(__file__).parent.parent / "dataset" / "dataset_creator.py"
HASH_STORE = Path(__file__).parent / ".dataset_creator_hash.json"


def _get_file_hash(path: Path) -> str:
    """Return the MD5 hash of a file's contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_stored_hash() -> str | None:
    """Load the previously stored hash, or None if it doesn't exist."""
    if HASH_STORE.exists():
        return json.loads(HASH_STORE.read_text()).get("hash")
    return None


def _save_hash(hash_value: str):
    """Persist the current hash for next run."""
    HASH_STORE.write_text(json.dumps({"hash": hash_value}))


def pytest_collection_modifyitems(config, items):
    """Skip dataset_creator tests if the file hasn't changed since last run."""
    if not WATCHED_FILE.exists():
        return

    current_hash = _get_file_hash(WATCHED_FILE)
    stored_hash = _load_stored_hash()

    if current_hash == stored_hash:
        skip_marker = pytest.mark.skip(
            reason="dataset_creator.py unchanged since last run — skipping tests"
        )
        for item in items:
            if "dataset" in item.nodeid or "dataset_creator" in item.nodeid:
                item.add_marker(skip_marker)
    else:
        # Save the new hash so next run skips if nothing changes again
        _save_hash(current_hash)