"""Smoke test: validates the LangChain + ChromaDB + OpenAI stack end to end.

Creates a document, splits it, embeds it, stores it, and queries it back.
Requires OPENAI_API_KEY to be set.
"""

import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from op_knowledge_base.config import load_config
from op_knowledge_base.store import add_documents, init_store, query

SAMPLE_TEXT = """\
Kubernetes is an open-source container orchestration platform. It automates
the deployment, scaling, and management of containerized applications.
Kubernetes groups containers into logical units called pods. A pod is the
smallest deployable unit and can contain one or more containers that share
storage and network resources.

Services in Kubernetes provide stable network endpoints for pods. A Service
abstracts access to a set of pods, enabling load balancing and service
discovery. Ingress controllers manage external access to services, typically
via HTTP routing rules.
"""


def main():
    config = load_config()

    # Use a temporary directory so we don't pollute the real store.
    with tempfile.TemporaryDirectory() as tmpdir:
        config["chroma"]["persist_directory"] = tmpdir

        # 1. Create a document.
        doc = Document(
            page_content=SAMPLE_TEXT,
            metadata={"doc_id": "test-k8s", "source_type": "manual", "title": "Kubernetes Basics"},
        )
        print(f"Created document: {doc.metadata['title']} ({len(doc.page_content)} chars)")

        # 2. Split it.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
        )
        chunks = splitter.split_documents([doc])
        print(f"Split into {len(chunks)} chunk(s)")

        # 3. Embed and store.
        store = init_store(config)
        ids = add_documents(store, chunks)
        print(f"Stored {len(ids)} chunk(s) in ChromaDB")

        # 4. Query it back.
        question = "What is a pod in Kubernetes?"
        results = query(store, question, top_k=2)
        print(f"\nQuery: {question}")
        print(f"Results: {len(results)} hit(s)\n")

        for i, (result_doc, score) in enumerate(results, 1):
            print(f"--- Result {i} (score: {score:.4f}) ---")
            print(result_doc.page_content[:200])
            print()

    print("Foundation verification passed.")


if __name__ == "__main__":
    main()
