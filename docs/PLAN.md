# RAG System for Evolving Knowledge Sources - Implementation Plan

## Choices Made

- Language: Python
- Vector DB: ChromaDB (embedded, local)
- Embeddings: OpenAI text-embedding-3-small
- Generation: Claude (Anthropic) via langchain-anthropic
- Data sources: Real Confluence + multiple Git repos
- Package manager: uv
- Framework: LangChain (where it adds value, custom code for the hard parts)

## LangChain Strategy

Use LangChain for:
- Document loaders (ConfluenceLoader, GitLoader) -- saves boilerplate
- Text splitters (RecursiveCharacterTextSplitter) -- well-tested, many strategies
- ChromaDB vector store integration -- standard and clean
- OpenAI embeddings wrapper -- consistent interface
- Anthropic Claude wrapper for generation -- consistent interface

Write yourself:
- Change detection logic -- the core engineering challenge
- Ingestion orchestration -- what to update, skip, delete
- State tracking -- knowing what was indexed and when
- Query pipeline assembly -- understand how retrieval + generation fit together

This gives you ecosystem fluency AND real understanding of the hard problem.

## Project Structure

```
op_knowledge_base/
  pyproject.toml
  config.yaml                  # Confluence URL, Git repo paths, API keys
  src/
    op_knowledge_base/
      __init__.py
      config.py                # Load and validate config.yaml
      models.py                # Dataclasses: SourceState, IngestionResult
      change_detection.py      # Hash-based and timestamp-based change detection
      sources/
        __init__.py
        confluence.py          # LangChain ConfluenceLoader + change detection
        git.py                 # LangChain GitLoader + hash-based change detection
      store.py                 # ChromaDB via LangChain + custom upsert/delete logic
      ingestion.py             # Orchestrates: detect changes -> load -> chunk -> store
      query.py                 # Query pipeline: retrieve -> format -> generate
      cli.py                   # CLI entry points (ingest, query, status)
  state/                       # Persisted state for change detection
    confluence.json
    git.json
  tests/
    test_change_detection.py
    test_store.py
    test_confluence.py
    test_git.py
    test_ingestion.py
    test_query.py
```

## Phase 1: Foundation (Steps 1-4)

### Step 1: Project scaffolding [DONE]
- Initialize uv project with pyproject.toml
- Dependencies:
  - langchain, langchain-community, langchain-openai, langchain-anthropic, langchain-chroma
  - chromadb
  - atlassian-python-api (for Confluence loader)
  - gitpython (for Git loader)
  - pyyaml, click
- Create directory structure
- Create config.yaml with placeholder values
- Create config.py to load it

### Step 2: Core data models and change detection [DONE]
- `models.py`: SourceState, IngestionResult dataclasses
- `change_detection.py`:
  - content_hash(text) -> SHA256
  - has_changed(doc_id, new_hash, state) -> bool
  - Load/save state from JSON files
  - Unit tests (6 tests passing)

### Step 3: Vector store wrapper
- `store.py`: Thin layer over LangChain's Chroma integration
  - Initialize persistent ChromaDB collection
  - add_documents(docs, metadatas): uses LangChain's Chroma.from_documents
  - query(question, top_k, filters): similarity search with metadata filtering
  - delete_by_doc_id(doc_id): remove all chunks for a document
  - list_doc_ids(): what's currently indexed
  - Unit tests

### Step 4: Verify foundation works
- Write a small script that:
  - Creates a Document manually
  - Splits it with RecursiveCharacterTextSplitter
  - Embeds and stores in ChromaDB
  - Queries it back
- This validates the LangChain + ChromaDB + OpenAI stack end to end

**Checkpoint: LangChain stack works. Can store and retrieve documents.**

## Phase 2: Confluence Source (Steps 5-6)

### Step 5: Confluence client with change detection
- `sources/confluence.py`:
  - Use LangChain's ConfluenceLoader to fetch pages
  - Add change detection on top: compare page version/lastUpdated
  - Track seen state in state/confluence.json
  - Return only changed/new documents

### Step 6: Confluence ingestion end-to-end
- `ingestion.py`: Orchestrate the pipeline
  - Detect changed docs (custom code)
  - Load content (LangChain loader)
  - Split into chunks (LangChain splitter)
  - Store (LangChain Chroma)
  - Handle deletions
  - Log what was updated/skipped/deleted
- `cli.py`: Add `ingest confluence` command
- Test against real Confluence

**Checkpoint: Can ingest Confluence pages incrementally.**

## Phase 3: Git Source (Steps 7-8)

### Step 7: Git repo scanner with change detection
- `sources/git.py`:
  - Use LangChain's GitLoader for file loading
  - Filter by file extensions (.md, .py, .txt, .yaml, etc.)
  - Change detection: SHA256 hash of file contents
  - Track in state/git.json
  - Support multiple repos from config

### Step 8: Git ingestion end-to-end
- Wire git source into ingestion.py
- `cli.py`: Add `ingest git` and `ingest all` commands
- Test with real repos

**Checkpoint: Can ingest both sources with change detection.**

## Phase 4: Query Pipeline (Steps 9-10)

### Step 9: Basic query pipeline
- `query.py`:
  - Use LangChain's Chroma retriever for similarity search
  - Build a prompt template with retrieved context + source metadata
  - Call ChatAnthropic (Claude) with the prompt
  - Return answer + source references
- `cli.py`: Add `query` command

### Step 10: Source attribution and filtering
- Include source metadata in prompt (source_type, title, last_updated)
- Instruct LLM to cite sources
- Add CLI flags to filter by source_type or recency
- Display formatted sources after answer

**Checkpoint: Working end-to-end RAG with citations.**

## Phase 5: Production Hardening (Steps 11-13)

### Step 11: Chunk-level deduplication
- Hash each chunk's content before embedding
- Skip embedding if chunk hash unchanged (saves API costs)
- Atomic document updates: delete old chunks + insert new in one operation

### Step 12: Status and observability
- `cli.py`: Add `status` command
  - Indexed document count by source
  - Last ingestion timestamp
  - Stale document detection

### Step 13: Error handling and retry
- Graceful handling of Confluence API failures
- Graceful handling of OpenAI rate limits
- Partial ingestion recovery (don't lose progress on failure)

**Checkpoint: Production-usable system.**

## Phase 6: Advanced (Future, not in initial build)

- Hybrid search (BM25 + vector) via LangChain's EnsembleRetriever
- Reranking with a cross-encoder
- Agentic routing (LangChain agents to pick the right source)
- Web UI

## Implementation Order Summary

1. ~~Project scaffolding + config~~ DONE
2. ~~Data models + change detection + tests~~ DONE
3. ChromaDB store wrapper + tests
4. Foundation verification (end-to-end smoke test)
5. Confluence loader + change detection
6. Confluence ingestion e2e
7. Git loader + change detection
8. Git ingestion e2e
9. Query pipeline
10. Source attribution + filtering
11. Chunk-level deduplication
12. Status CLI
13. Error handling

Each step is small, testable, and builds on the previous one.
