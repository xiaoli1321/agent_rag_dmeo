from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .models import RetrievedDoc
from ..ingest.pipeline import clean_documents, load_sources, split_documents


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

    def fuse(
        self, dense_hits: list[DenseHit], sparse_hits: list[SparseHit]
    ) -> list[HybridHit]:
        if not dense_hits and not sparse_hits:
            return []
        if not sparse_hits:
            return [
                HybridHit(
                    chunk_id=hit.chunk_id,
                    dense_score=hit.score,
                    sparse_score=None,
                    fusion_score=hit.score,
                    doc=_with_scores(
                        hit.doc,
                        vector_score=hit.score,
                        sparse_score=None,
                        final_score=hit.score,
                        retrieval_source="dense",
                    ),
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
                    doc=_with_scores(
                        hit.doc,
                        vector_score=None,
                        sparse_score=hit.score,
                        final_score=hit.score,
                        retrieval_source="sparse",
                    ),
                    metadata=hit.metadata,
                )
                for hit in sparse_hits
            ]

        dense_map = {hit.chunk_id: hit for hit in dense_hits}
        sparse_map = {hit.chunk_id: hit for hit in sparse_hits}
        dense_norm = _normalize_scores({hit.chunk_id: hit.score for hit in dense_hits})
        sparse_norm = _normalize_scores(
            {hit.chunk_id: hit.score for hit in sparse_hits}
        )

        fused: list[HybridHit] = []
        for chunk_id in set(dense_map) | set(sparse_map):
            dense = dense_map.get(chunk_id)
            sparse = sparse_map.get(chunk_id)
            fusion_score = self.alpha * dense_norm.get(chunk_id, 0.0) + (
                1.0 - self.alpha
            ) * sparse_norm.get(chunk_id, 0.0)
            source_doc = dense.doc if dense else sparse.doc  # type: ignore[union-attr]
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
        fused.sort(
            key=lambda item: (
                item.fusion_score,
                item.dense_score or 0.0,
                item.sparse_score or 0.0,
            ),
            reverse=True,
        )
        return fused


class LocalSparseRetriever:
    def __init__(self, docs: list[RetrievedDoc] | None = None) -> None:
        self.docs = docs if docs is not None else _load_local_docs()
        self._bm25: BM25Retriever | None = None

    def _get_bm25(self, top_k: int) -> BM25Retriever:
        """Lazy-initialize and return the BM25Retriever."""
        if self._bm25 is None:
            self._bm25 = BM25Retriever.from_documents(
                documents=[
                    Document(page_content=doc.chunk_text, metadata={"doc": doc})
                    for doc in self.docs
                ],
                k=top_k * 3,
                preprocess_func=_preprocess,
            )
        return self._bm25

    def search(
        self, query: str, *, top_k: int, product_tags: list[str] | None = None
    ) -> list[SparseHit]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        bm25 = self._get_bm25(top_k)
        results = bm25.invoke(query)
        if not results:
            return []

        # Get raw BM25 scores from the internal model for accurate scoring
        all_scores = bm25.vectorizer.get_scores(query_tokens)
        score_map: dict[int, float] = {}
        for i, bm25_doc in enumerate(bm25.docs):
            score_map[id(bm25_doc.metadata["doc"])] = float(all_scores[i])

        hits: list[SparseHit] = []
        for doc_result in results:
            doc: RetrievedDoc = doc_result.metadata["doc"]
            if product_tags and not set(product_tags).intersection(doc.product_tags):
                continue
            chunk_id = doc.chunk_id or doc.source_url
            score = score_map.get(id(doc), 0.0)
            hits.append(SparseHit(chunk_id=chunk_id, score=round(score, 6), doc=doc))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


def dense_docs_to_hits(docs: list[RetrievedDoc]) -> list[DenseHit]:
    return [
        DenseHit(
            chunk_id=doc.chunk_id or doc.source_url,
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
    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in scores.items()
    }


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
                chunk_id=str(metadata.get("chunk_id") or ""),
                source_title=metadata["source_title"],
                source_url=metadata["source_url"],
                chunk_text=str(metadata.get("context_text") or chunk.page_content),
                score=0.0,
                product_tags=list(metadata.get("product_tags") or []),
                retrieval_source="sparse",
            )
        )
    return docs


def _preprocess(text: str) -> list[str]:
    """Tokenize text for BM25Retriever (same logic as _tokenize)."""
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
