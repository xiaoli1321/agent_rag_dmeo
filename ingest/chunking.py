"""Demo-local chunking strategies used by the standalone ingestion pipeline.

This module deliberately has no dependency on the parent project's ``app``
package, so ``python -m customer_agent_demo.ingest.run`` can rebuild Qdrant
from this directory alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(slots=True)
class ParsedBlock:
    text: str
    chunk_type: str = "text"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkingOptions:
    strategy: str = "structural"
    max_chars: int = 700
    overlap_chars: int = 120
    parent_max_chars: int = 2800
    child_max_chars: int = 700


@dataclass(slots=True)
class ChunkPayload:
    chunk_text: str
    metadata_json: dict[str, object]
    context_text: str | None = None


def build_chunks(
    parsed_blocks: list[ParsedBlock], *, options: ChunkingOptions
) -> list[ChunkPayload]:
    """Build retrieval chunks for the demo's structural or parent-child modes."""
    if options.strategy == "parent-child":
        return _build_parent_child_chunks(parsed_blocks, options)
    if options.strategy != "structural":
        raise ValueError(f"Unsupported demo chunking strategy: {options.strategy}")

    chunks: list[ChunkPayload] = []
    for block in parsed_blocks:
        for section in _structural_sections(block.text):
            chunks.extend(
                ChunkPayload(part, dict(block.metadata))
                for part in _split_text(
                    section, max_chars=options.max_chars, overlap=options.overlap_chars
                )
            )
    return chunks


def _build_parent_child_chunks(
    blocks: list[ParsedBlock], options: ChunkingOptions
) -> list[ChunkPayload]:
    chunks: list[ChunkPayload] = []
    for block in blocks:
        for parent in _split_text(
            block.text,
            max_chars=options.parent_max_chars,
            overlap=options.overlap_chars,
        ):
            for child in _split_text(
                parent, max_chars=options.child_max_chars, overlap=options.overlap_chars
            ):
                chunks.append(ChunkPayload(child, dict(block.metadata), parent))
    return chunks


def _structural_sections(text: str) -> list[str]:
    """Keep Markdown-style headings with their following content when present."""
    parts = re.split(r"(?m)(?=^#{1,6}\s+)", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [text] if text else []

    step = max(1, max_chars - max(0, overlap))
    return [
        text[start : start + max_chars].strip()
        for start in range(0, len(text), step)
        if text[start : start + max_chars].strip()
    ]
