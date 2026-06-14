"""Data models for the RAG system."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SourceState:
    """Tracks the last-known state of a document for change detection."""

    doc_id: str
    source_type: str  # "confluence" or "git"
    content_hash: str
    last_updated: str  # ISO format timestamp
    title: str = ""


@dataclass
class IngestionResult:
    """Summary of an ingestion run."""

    source_type: str
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_deleted: int = 0
    chunks_skipped: int = 0
    errors: list[str] = field(default_factory=list)
