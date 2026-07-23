from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from time import sleep
from typing import Any

from langchain_core.embeddings import Embeddings

from ..config import DemoSettings


MULTIMODAL_EMBEDDING_PATH = (
    "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
)
EMBEDDING_RETRY_DELAYS_SECONDS = (1, 2, 4)

# 保守的 token 估算：中英文混排约 3 字符/token，512 token 上限 ≈ 1500 字符
_MAX_EMBEDDING_CHARS = 4096


def _truncate(text: str, max_chars: int = _MAX_EMBEDDING_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


class OpenAIEmbeddingAdapter(Embeddings):
    """LangChain embedding adapter for OpenAI-compatible embedding APIs (e.g., llama-server)."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            embeddings.append(self._embed_text(text))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        text = _truncate(text)
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": text,
            "model": self.model,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt, delay in enumerate((*EMBEDDING_RETRY_DELAYS_SECONDS, None), 1):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"OpenAI-compatible embedding failed: HTTP {exc.code} {body}"
                ) from exc
            except urllib.error.URLError as exc:
                if delay is None:
                    raise RuntimeError(
                        "OpenAI-compatible embedding failed after 4 attempts: "
                        f"{exc.reason}"
                    ) from exc
                sleep(delay)
        return _extract_openai_embedding(data)


def _extract_openai_embedding(data: dict[str, Any]) -> list[float]:
    entries = data.get("data", [])
    if not entries:
        raise RuntimeError(f"OpenAI-compatible embedding returned no data: {data}")
    first = sorted(entries, key=lambda item: item.get("index", 0))[0]
    embedding = first.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError(
            f"OpenAI-compatible embedding returned invalid vector: {data}"
        )
    return [float(value) for value in embedding]


def get_embeddings(settings: DemoSettings) -> Embeddings:
    base_url = settings.embedding_api_base or "https://dashscope.aliyuncs.com"
    model = settings.embedding_model or "qwen3-vl-embedding"

    if "dashscope" in base_url.lower():
        return DashScopeMultimodalEmbeddings(
            api_key=settings.embedding_api_key or "",
            base_url=base_url,
            model=model,
            dimension=settings.embedding_vector_size,
        )

    return OpenAIEmbeddingAdapter(
        api_key=settings.embedding_api_key or "",
        base_url=base_url,
        model=model,
    )


class DashScopeMultimodalEmbeddings(Embeddings):
    """LangChain embedding adapter for DashScope qwen3-vl-embedding text chunks."""

    def __init__(
        self, *, api_key: str, base_url: str, model: str, dimension: int
    ) -> None:
        self.api_key = api_key
        self.endpoint = _multimodal_endpoint(base_url)
        self.model = model
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            embeddings.append(self._embed_text(text))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        text = _truncate(text, max_chars=2000)
        payload = {
            "model": self.model,
            "input": {"contents": [{"text": text}]},
            "parameters": {"dimension": self.dimension},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt, delay in enumerate((*EMBEDDING_RETRY_DELAYS_SECONDS, None), 1):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"DashScope multimodal embedding failed: HTTP {exc.code} {body}"
                ) from exc
            except urllib.error.URLError as exc:
                if delay is None:
                    raise RuntimeError(
                        "DashScope multimodal embedding failed after 4 attempts: "
                        f"{exc.reason}"
                    ) from exc
                sleep(delay)
        return _extract_embedding(data)


def _multimodal_endpoint(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    if not parsed.scheme or not parsed.netloc:
        return "https://dashscope.aliyuncs.com" + MULTIMODAL_EMBEDDING_PATH
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, MULTIMODAL_EMBEDDING_PATH, "", "", "")
    )


def _extract_embedding(data: dict[str, Any]) -> list[float]:
    embeddings = data.get("output", {}).get("embeddings", [])
    if not embeddings:
        raise RuntimeError(
            f"DashScope multimodal embedding returned no vectors: {data}"
        )
    first = sorted(embeddings, key=lambda item: item.get("index", 0))[0]
    embedding = first.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError(
            f"DashScope multimodal embedding returned invalid vector: {data}"
        )
    return [float(value) for value in embedding]
