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
    assert first.metadata["chunk_text"] == first.page_content
    assert isinstance(first.metadata["chunk_index"], int)
    assert first.metadata["chunk_id"]
    assert first.metadata["chunking_strategy"] == "recursive"


def test_structural_chunking_strategy_keeps_section_metadata() -> None:
    cleaned = clean_documents(load_sources()[:2])

    chunks = split_documents(cleaned, chunk_size=500, chunk_overlap=60, strategy="structural")

    assert chunks
    assert all(chunk.metadata["chunking_strategy"] == "structural" for chunk in chunks)
    assert all(chunk.metadata["chunk_text"] == chunk.page_content for chunk in chunks)
    assert any(chunk.metadata.get("split_reason") for chunk in chunks)


def test_parent_child_chunking_strategy_embeds_child_with_parent_context() -> None:
    cleaned = clean_documents(load_sources()[:2])

    chunks = split_documents(cleaned, chunk_size=360, chunk_overlap=40, strategy="parent-child")

    assert chunks
    first = chunks[0]
    assert first.metadata["chunking_strategy"] == "parent-child"
    assert first.metadata["chunk_level"] == "child"
    assert first.metadata["embedded_chunk_text"] == first.page_content
    assert first.metadata["context_text"]
    assert first.metadata["chunk_text"] == first.metadata["context_text"]
