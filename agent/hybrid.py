from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from customer_agent_demo.agent.models import RetrievedDoc
from customer_agent_demo.ingest.pipeline import clean_documents, load_sources, split_documents


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


@dataclass(slots=True)
class DenseHit:
    chunk_id: str
    score: float
    doc: RetrievedDoc
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SparseHit:
    chunk_id: str
    score: float
    doc: RetrievedDoc
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HybridHit:
    chunk_id: str
    dense_score: float | None
    sparse_score: float | None
    fusion_score: float
    doc: RetrievedDoc
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = max(0.0, min(1.0, float(alpha)))

    def fuse(self, dense_hits: list[DenseHit], sparse_hits: list[SparseHit]) -> list[HybridHit]:
        if not dense_hits and not sparse_hits:
            return []
        if not sparse_hits:
            return [
                HybridHit(
                    chunk_id=hit.chunk_id,
                    dense_score=hit.score,
                    sparse_score=None,
                    fusion_score=hit.score,
                    doc=_with_scores(hit.doc, vector_score=hit.score, sparse_score=None, final_score=hit.score, retrieval_source="dense"),
                    metadata=hit.metadata,
                )
                for hit in dense_hits
            ]
        if not dense_hits:
            return [
                HybridHit(
                    chunk_id=hit.chunk_id,
                    dense_score=None,
                    sparse_score=hit.score,
                    fusion_score=hit.score,
                    doc=_with_scores(hit.doc, vector_score=None, sparse_score=hit.score, final_score=hit.score, retrieval_source="sparse"),
                    metadata=hit.metadata,
                )
                for hit in sparse_hits
            ]

        dense_map = {hit.chunk_id: hit for hit in dense_hits}
        sparse_map = {hit.chunk_id: hit for hit in sparse_hits}
        dense_norm = _normalize_scores({hit.chunk_id: hit.score for hit in dense_hits})
        sparse_norm = _normalize_scores({hit.chunk_id: hit.score for hit in sparse_hits})

        fused: list[HybridHit] = []
        for chunk_id in set(dense_map) | set(sparse_map):
            dense = dense_map.get(chunk_id)
            sparse = sparse_map.get(chunk_id)
            fusion_score = self.alpha * dense_norm.get(chunk_id, 0.0) + (1.0 - self.alpha) * sparse_norm.get(chunk_id, 0.0)
            source_doc = (dense.doc if dense else sparse.doc)  # type: ignore[union-attr]
            doc = _with_scores(
                source_doc,
                vector_score=dense.score if dense else None,
                sparse_score=sparse.score if sparse else None,
                final_score=round(fusion_score, 6),
                retrieval_source="hybrid",
            )
            metadata = {}
            if dense:
                metadata.update(dense.metadata)
            if sparse:
                metadata.update(sparse.metadata)
            fused.append(
                HybridHit(
                    chunk_id=chunk_id,
                    dense_score=dense.score if dense else None,
                    sparse_score=sparse.score if sparse else None,
                    fusion_score=round(fusion_score, 6),
                    doc=doc,
                    metadata=metadata,
                )
            )
        fused.sort(key=lambda item: (item.fusion_score, item.dense_score or 0.0, item.sparse_score or 0.0), reverse=True)
        return fused


class LocalSparseRetriever:
    def __init__(self, docs: list[RetrievedDoc] | None = None) -> None:
        self.docs = docs if docs is not None else _load_local_docs()
        self.doc_tokens = [_tokenize(doc.chunk_text + " " + doc.source_title + " " + (doc.product or "")) for doc in self.docs]
        self.doc_freq: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def search(self, query: str, *, top_k: int) -> list[SparseHit]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        hits: list[SparseHit] = []
        total_docs = max(1, len(self.docs))
        for doc, tokens in zip(self.docs, self.doc_tokens):
            score = 0.0
            token_count = max(1, len(tokens))
            for token in query_tokens:
                tf = tokens.count(token) / token_count
                if tf == 0:
                    continue
                idf = math.log((1 + total_docs) / (1 + self.doc_freq.get(token, 0))) + 1
                score += tf * idf
            if score > 0:
                chunk_id = f"{doc.source_url}#{doc.chunk_index}"
                hits.append(SparseHit(chunk_id=chunk_id, score=round(score, 6), doc=doc))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


def dense_docs_to_hits(docs: list[RetrievedDoc]) -> list[DenseHit]:
    return [
        DenseHit(
            chunk_id=f"{doc.source_url}#{doc.chunk_index}",
            score=doc.score,
            doc=doc,
        )
        for doc in docs
    ]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


def _with_scores(
    doc: RetrievedDoc,
    *,
    vector_score: float | None,
    sparse_score: float | None,
    final_score: float,
    retrieval_source: str,
) -> RetrievedDoc:
    return doc.model_copy(
        update={
            "score": final_score,
            "vector_score": vector_score,
            "sparse_score": sparse_score,
            "final_score": final_score,
            "retrieval_source": retrieval_source,
        }
    )


def _load_local_docs() -> list[RetrievedDoc]:
    chunks = split_documents(clean_documents(load_sources()))
    docs: list[RetrievedDoc] = []
    for chunk in chunks:
        metadata = chunk.metadata
        docs.append(
            RetrievedDoc(
                source_title=metadata["source_title"],
                source_url=metadata["source_url"],
                chunk_index=int(metadata["chunk_index"]),
                chunk_text=str(metadata.get("chunk_text") or chunk.page_content),
                score=0.0,
                product=metadata.get("product"),
                retrieval_source="sparse",
            )
        )
    return docs


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
