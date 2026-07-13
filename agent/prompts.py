from __future__ import annotations

from functools import lru_cache

from customer_agent_demo.config import DEMO_ROOT


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    path = DEMO_ROOT / "prompts" / name
    return path.read_text(encoding="utf-8")

