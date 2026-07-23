from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml

from ..config import DEMO_ROOT


@dataclass(frozen=True)
class SlotDefinition:
    key: str
    entity_field: str
    question: str
    options: tuple[str, ...]
    reason: str
    vague_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentDefinition:
    name: str
    description: str
    examples: tuple[str, ...]
    handler: str
    required_slots: tuple[str, ...]
    clarification_order: tuple[str, ...]
    direct_handoff: bool
    default_actionability: str


@lru_cache(maxsize=1)
def load_slot_catalog() -> dict[str, SlotDefinition]:
    payload: dict[str, Any] = yaml.safe_load((DEMO_ROOT / "data" / "intent_catalog.yaml").read_text(encoding="utf-8"))
    slots_raw = payload.get("slots", {})
    return {
        key: SlotDefinition(
            key=key,
            entity_field=raw["entity_field"],
            question=raw["question"],
            options=tuple(raw.get("options", [])),
            reason=raw.get("reason", "missing_slot"),
            vague_values=tuple(raw.get("vague_values", [])),
        )
        for key, raw in slots_raw.items()
    }


@lru_cache(maxsize=1)
def load_intent_catalog() -> dict[str, IntentDefinition]:
    payload: dict[str, Any] = yaml.safe_load(
        (DEMO_ROOT / "data" / "intent_catalog.yaml").read_text(encoding="utf-8")
    )
    return {
        name: IntentDefinition(
            name=name,
            description=raw["description"],
            examples=tuple(raw.get("examples", [])),
            handler=raw["handler"],
            required_slots=tuple(raw.get("required_slots", [])),
            clarification_order=tuple(raw.get("clarification_order", [])),
            direct_handoff=bool(raw.get("direct_handoff", False)),
            default_actionability=raw.get("default_actionability", "ready"),
        )
        for name, raw in payload["intents"].items()
    }


def catalog_prompt_context() -> str:
    rows = []
    for definition in load_intent_catalog().values():
        examples = "；".join(definition.examples)
        rows.append(f"- {definition.name}：{definition.description} 示例：{examples}")
    return "\n".join(rows)
