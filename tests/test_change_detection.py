"""Tests for change detection module."""

import json
from pathlib import Path

from op_knowledge_base.change_detection import (
    content_hash,
    has_changed,
    load_state,
    save_state,
)
from op_knowledge_base.models import SourceState


def test_content_hash_deterministic():
    assert content_hash("hello") == content_hash("hello")


def test_content_hash_different_for_different_content():
    assert content_hash("hello") != content_hash("world")


def test_has_changed_new_document():
    state = {}
    assert has_changed("doc1", "abc123", state) is True


def test_has_changed_unchanged_document():
    state = {
        "doc1": SourceState(
            doc_id="doc1",
            source_type="git",
            content_hash="abc123",
            last_updated="2024-01-01T00:00:00",
        )
    }
    assert has_changed("doc1", "abc123", state) is False


def test_has_changed_modified_document():
    state = {
        "doc1": SourceState(
            doc_id="doc1",
            source_type="git",
            content_hash="abc123",
            last_updated="2024-01-01T00:00:00",
        )
    }
    assert has_changed("doc1", "new_hash", state) is True


def test_save_and_load_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "op_knowledge_base.change_detection.STATE_DIR", tmp_path
    )

    state = {
        "doc1": SourceState(
            doc_id="doc1",
            source_type="confluence",
            content_hash="hash1",
            last_updated="2024-01-01T00:00:00",
            title="Test Page",
        )
    }

    save_state("confluence", state)
    loaded = load_state("confluence")

    assert "doc1" in loaded
    assert loaded["doc1"].content_hash == "hash1"
    assert loaded["doc1"].title == "Test Page"
