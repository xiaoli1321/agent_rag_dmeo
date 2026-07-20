from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEMO_ROOT = Path(__file__).resolve().parent


class DemoSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(DEMO_ROOT / ".env", DEMO_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "cgm_customer_support"

    qwen_api_base: str | None = None
    qwen_api_key: str | None = None
    qwen_llm_model: str | None = "qwen-plus"
    qwen_embedding_model: str | None = "qwen3-vl-embedding"

    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_max_tokens: int = Field(default=4096, ge=1)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    embedding_api_base: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    embedding_vector_size: int = Field(default=1024, ge=1)

    rerank_enabled: bool = False
    rerank_api_base: str | None = None
    rerank_api_key: str | None = None
    rerank_model: str | None = None

    agent_top_k: int = Field(default=4, ge=1, le=20)
    agent_min_relevance_score: float = Field(default=0.35, ge=0.0, le=1.0)
    agent_retrieval_strategy: str = "hybrid"
    agent_fusion_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    agent_corrective_retries: int = Field(default=1, ge=0, le=2)
    agent_llm_graders_enabled: bool = True
    agent_run_log_enabled: bool = True
    agent_run_log_dir: Path = Field(default_factory=lambda: DEMO_ROOT / "data" / "runs")
    demo_chunking_strategy: str = "structural"

    @model_validator(mode="after")
    def normalize(self) -> "DemoSettings":
        self.qdrant_url = self.qdrant_url.strip().rstrip("/")
        self.qdrant_collection = (
            self.qdrant_collection.strip() or "cgm_customer_support"
        )
        self.qwen_api_base = self._clean(self.qwen_api_base)
        self.qwen_api_key = self._clean(self.qwen_api_key)
        self.qwen_llm_model = self._clean(self.qwen_llm_model)
        self.qwen_embedding_model = self._clean(self.qwen_embedding_model)
        self.llm_api_base = self._clean(self.llm_api_base) or self.qwen_api_base
        self.llm_api_key = self._clean(self.llm_api_key) or self.qwen_api_key
        self.llm_model = self._clean(self.llm_model) or self.qwen_llm_model
        self.embedding_api_base = (
            self._clean(self.embedding_api_base) or self.qwen_api_base
        )
        self.embedding_api_key = (
            self._clean(self.embedding_api_key) or self.qwen_api_key
        )
        self.embedding_model = (
            self._clean(self.embedding_model)
            or self.qwen_embedding_model
            or "qwen3-vl-embedding"
        )
        self.rerank_api_base = self._clean(self.rerank_api_base)
        self.rerank_api_key = self._clean(self.rerank_api_key)
        self.rerank_model = self._clean(self.rerank_model)
        self.agent_retrieval_strategy = (
            (self.agent_retrieval_strategy or "hybrid").strip().lower()
        )
        if self.agent_retrieval_strategy not in {"dense", "hybrid"}:
            self.agent_retrieval_strategy = "hybrid"
        self.demo_chunking_strategy = (
            (self.demo_chunking_strategy or "structural").strip().lower()
        )
        if self.demo_chunking_strategy not in {
            "recursive",
            "structural",
            "parent-child",
        }:
            self.demo_chunking_strategy = "structural"
        return self

    @staticmethod
    def _clean(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_base and self.llm_api_key and self.llm_model)

    @property
    def embedding_configured(self) -> bool:
        return bool(
            self.embedding_api_base and self.embedding_api_key and self.embedding_model
        )

    @property
    def llm_extra_body(self) -> dict[str, object] | None:
        if self.llm_model and self.llm_model.lower().startswith("qwen3"):
            return {"enable_thinking": False}
        return None


@lru_cache(maxsize=1)
def get_settings() -> DemoSettings:
    return DemoSettings()
