"""Change detection for documents using content hashing."""

import hashlib
import json
from pathlib import Path

from op_knowledge_base.models import SourceState


STATE_DIR = Path(__file__).parent.parent.parent / "state"


def content_hash(text: str) -> str:
    """Compute SHA256 hash of text content."""
    return hashlib.sha256(text.encode()).hexdigest()


def load_state(source_type: str) -> dict[str, SourceState]:
    """Load persisted state for a source type."""
    state_file = STATE_DIR / f"{source_type}.json"
    if not state_file.exists():
        return {}

    with open(state_file) as f:
        data = json.load(f)

    return {
        doc_id: SourceState(**entry)
        for doc_id, entry in data.items()
    }


def save_state(source_type: str, state: dict[str, SourceState]) -> None:
    """Persist state for a source type."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{source_type}.json"

    data = {
        doc_id: {
            "doc_id": s.doc_id,
            "source_type": s.source_type,
            "content_hash": s.content_hash,
            "last_updated": s.last_updated,
            "title": s.title,
        }
        for doc_id, s in state.items()
    }

    with open(state_file, "w") as f:
        json.dump(data, f, indent=2)


def has_changed(doc_id: str, new_hash: str, state: dict[str, SourceState]) -> bool:
    """Check if a document has changed since last ingestion."""
    if doc_id not in state:
        return True
    return state[doc_id].content_hash != new_hash
