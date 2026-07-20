from __future__ import annotations

from ..config import get_settings
from .pipeline import (
    clean_documents,
    load_sources,
    split_documents,
    upsert_to_qdrant,
)


def main() -> None:
    settings = get_settings()
    sources = load_sources()
    cleaned = clean_documents(sources)
    chunks = split_documents(cleaned, strategy=settings.demo_chunking_strategy)
    upsert_to_qdrant(chunks, settings)
    print(
        f"入库完成：sources={len(sources)}, cleaned={len(cleaned)}, "
        f"chunks={len(chunks)}, strategy={settings.demo_chunking_strategy}, "
        f"collection={settings.qdrant_collection}"
    )


if __name__ == "__main__":
    main()
