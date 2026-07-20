from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from customer_agent_demo.agent.embeddings import get_embeddings
from customer_agent_demo.config import DEMO_ROOT, DemoSettings


DEFAULT_SOURCES_PATH = DEMO_ROOT / "data" / "cgm_sources.json"
@dataclass(slots=True)
class SourceDocument:
    source_title: str
    source_url: str
    product: str
    text: str


@dataclass(slots=True)
class CleanDocument:
    page_content: str
    metadata: dict[str, Any]


def load_sources(path: Path = DEFAULT_SOURCES_PATH) -> list[SourceDocument]:
    """Load manually curated, traceable CGM source records."""
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        SourceDocument(
            source_title=record["source_title"],
            source_url=record["source_url"],
            product=record["product"],
            text=record["text"],
        )
        for record in records
    ]


def clean_documents(sources: list[SourceDocument]) -> list[CleanDocument]:
    """Normalize whitespace and attach source metadata before chunking."""
    cleaned: list[CleanDocument] = []
    for source in sources:
        text = re.sub(r"\s+", " ", source.text).strip()
        if not text:
            continue
        cleaned.append(
            CleanDocument(
                page_content=text,
                metadata={
                    "source_title": source.source_title,
                    "source_url": source.source_url,
                    "product": source.product,
                },
            )
        )
    return cleaned


def split_documents(
    documents: list[CleanDocument],
    *,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    strategy: str = "recursive",
) -> list[Any]:
    """Split documents with LangChain Documents as the handoff object."""
    try:
        from langchain_core.documents import Document
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Install langchain-core before running ingest.") from exc

    normalized_strategy = (strategy or "recursive").strip().lower()
    source_docs = [
        Document(page_content=document.page_content, metadata=document.metadata)
        for document in documents
    ]
    if normalized_strategy in {"structural", "parent-child"}:
        return _split_with_app_chunking(
            source_docs,
            strategy=normalized_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", "；", "; ", "，", ", ", " "],
        )
        chunks = splitter.split_documents(source_docs)
    except ImportError:
        chunks = _fallback_split_documents(source_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    counters: dict[str, int] = {}
    enriched = []
    for chunk in chunks:
        source_url = chunk.metadata["source_url"]
        index = counters.get(source_url, 0)
        counters[source_url] = index + 1
        metadata = {
            "chunk_id": _stable_chunk_id(source_url, index, chunk.page_content),
            "source_title": chunk.metadata["source_title"],
            "source_url": source_url,
            "product": chunk.metadata.get("product"),
        }
        enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
    return enriched


def _split_with_app_chunking(
    source_docs: list[Any],
    *,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Any]:
    from langchain_core.documents import Document

    from app.ingestion.chunking import build_chunks
    from app.ingestion.types import ChunkingOptions, ParsedBlock

    parsed_blocks = [
        ParsedBlock(
            text=document.page_content,
            chunk_type="text",
            metadata=dict(document.metadata),
        )
        for document in source_docs
    ]
    options = ChunkingOptions(
        strategy=strategy,
        max_chars=chunk_size,
        overlap_chars=chunk_overlap,
        parent_max_chars=max(chunk_size * 4, 1000),
        child_max_chars=max(min(chunk_size, 800), 200),
    )
    payloads = build_chunks(parsed_blocks, options=options)
    counters: dict[str, int] = {}
    enriched = []
    for payload in payloads:
        source_url = str(payload.metadata_json["source_url"])
        index = counters.get(source_url, 0)
        counters[source_url] = index + 1
        metadata = {
            "chunk_id": _stable_chunk_id(source_url, index, payload.chunk_text),
            "source_title": payload.metadata_json["source_title"],
            "source_url": source_url,
            "product": payload.metadata_json.get("product"),
        }
        # 仅父子分块需要额外保留父块：子块用于召回，父块用于回答。
        if strategy == "parent-child" and payload.context_text:
            metadata["context_text"] = payload.context_text
        enriched.append(Document(page_content=payload.chunk_text, metadata=metadata))
    return enriched


def _fallback_split_documents(source_docs: list[Any], *, chunk_size: int, chunk_overlap: int) -> list[Any]:
    """Small local fallback for environments without langchain-text-splitters."""
    try:
        from langchain_core.documents import Document
    except ImportError as exc:  # pragma: no cover - already checked by caller
        raise RuntimeError("Install langchain-core before running ingest.") from exc

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for document in source_docs:
        text = document.page_content
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append(Document(page_content=piece, metadata=dict(document.metadata)))
            if start + chunk_size >= len(text):
                break
    return chunks


def upsert_to_qdrant(chunks: list[Any], settings: DemoSettings) -> None:
    """Delete and rebuild the demo collection so reruns are deterministic."""
    if not settings.embedding_configured:
        raise RuntimeError("Qwen embedding config is incomplete. Check .env.")

    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Install qdrant-client and langchain-qdrant from requirements.txt."
        ) from exc

    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)

    embeddings = get_embeddings(settings)
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        batch_size=10,
    )


def _stable_chunk_id(source_url: str, index: int, text: str) -> str:
    raw = f"{source_url}|{index}|{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))
