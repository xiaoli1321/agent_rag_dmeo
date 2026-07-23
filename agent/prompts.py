from __future__ import annotations

from functools import lru_cache
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ..config import DEMO_ROOT

_PROMPTS_DIR = DEMO_ROOT / "prompts"

_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **context: Any) -> str:
    """使用 Jinja2 模板引擎加载并渲染指定的提示词文件。"""
    template = _env.get_template(name)
    return template.render(**context)
