from __future__ import annotations

from customer_agent_demo.ingest.pipeline import clean_documents, load_sources, split_documents


def test_load_clean_split_sources_have_required_payload_metadata() -> None:
    sources = load_sources()
    cleaned = clean_documents(sources)
    chunks = split_documents(cleaned, chunk_size=180, chunk_overlap=30)

    assert len(sources) >= 8
    assert len(cleaned) == len(sources)
    assert chunks
    first = chunks[0]
    assert first.metadata["source_url"].startswith("https://")
    assert first.metadata["source_title"]
    assert "source_type" not in first.metadata
    assert "language" not in first.metadata
    assert first.metadata["chunk_id"]
    assert set(first.metadata) == {"chunk_id", "source_title", "source_url", "product"}


def test_structural_chunking_strategy_keeps_section_metadata() -> None:
    cleaned = clean_documents(load_sources()[:2])

    chunks = split_documents(cleaned, chunk_size=500, chunk_overlap=60, strategy="structural")

    assert chunks
    assert all("chunk_text" not in chunk.metadata for chunk in chunks)
    assert all("split_reason" not in chunk.metadata for chunk in chunks)


def test_parent_child_chunking_strategy_embeds_child_with_parent_context() -> None:
    cleaned = clean_documents(load_sources()[:2])

    chunks = split_documents(cleaned, chunk_size=360, chunk_overlap=40, strategy="parent-child")

    assert chunks
    first = chunks[0]
    assert "chunk_level" not in first.metadata
    assert "embedded_chunk_text" not in first.metadata
    assert first.metadata.get("context_text")
    assert len(first.metadata["context_text"]) >= len(first.page_content)
    assert first.metadata["chunk_id"]
