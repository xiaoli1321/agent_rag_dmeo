"""Product alias loader — single source of truth for product keyword→name mappings.

All product-aliasing functions in the codebase read from ``data/product_aliases.yaml``
through this module, so adding a new product or keyword only requires one change.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import yaml

from ..config import DEMO_ROOT


_ALIASES_PATH: Final = DEMO_ROOT / "data" / "product_aliases.yaml"


@lru_cache(maxsize=1)
def load_product_aliases() -> dict[str, str]:
    """Load the canonical product keyword→name mapping from YAML."""
    payload: dict = yaml.safe_load(_ALIASES_PATH.read_text(encoding="utf-8"))
    raw: dict[str, str] = payload.get("aliases", {})
    return {keyword.strip().lower(): name for keyword, name in raw.items()}


def lookup_product(text: str) -> str | None:
    """Find the first matching product name in ``text``.

    Entries are matched in descending keyword length so that ``"gs1 pro"``
    matches before the shorter ``"gs1"``.
    """
    if not text:
        return None
    lowered = text.strip().lower()
    aliases = load_product_aliases()
    # 长关键词优先匹配，避免 "gs1 pro" 被 "gs1" 误匹配
    for keyword in sorted(aliases, key=len, reverse=True):
        if keyword in lowered:
            return aliases[keyword]
    return None


def lookup_aliases(text: str) -> list[tuple[str, str]]:
    """Return all (keyword, product_name) pairs that match in ``text``.

    Used for multi-tag scenarios where a single message may mention several
    products.
    """
    if not text:
        return []
    lowered = text.strip().lower()
    aliases = load_product_aliases()
    results: list[tuple[str, str]] = []
    for keyword in sorted(aliases, key=len, reverse=True):
        if keyword in lowered:
            results.append((keyword, aliases[keyword]))
    return results