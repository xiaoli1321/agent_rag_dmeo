from __future__ import annotations

from ..agent.hybrid import (
    DenseHit,
    HybridRetriever,
    LocalSparseRetriever,
    SparseHit,
    _load_local_docs,
    dense_docs_to_hits,
)
from ..agent.models import RetrievedDoc
from ..agent.rag import RagService
from ..config import DemoSettings


def _doc(title: str, url: str, text: str, score: float = 0.0) -> RetrievedDoc:
    return RetrievedDoc(
        source_title=title,
        source_url=url,
        chunk_index=0,
        chunk_text=text,
        score=score,
    )


def test_hybrid_alpha_changes_fusion_order() -> None:
    dense_a = DenseHit(chunk_id="a", score=0.9, doc=_doc("A", "a", "semantic"))
    dense_b = DenseHit(chunk_id="b", score=0.1, doc=_doc("B", "b", "keyword"))
    sparse_a = SparseHit(chunk_id="a", score=0.1, doc=_doc("A", "a", "semantic"))
    sparse_b = SparseHit(chunk_id="b", score=0.9, doc=_doc("B", "b", "keyword"))

    dense_weighted = HybridRetriever(alpha=0.9).fuse(
        [dense_a, dense_b], [sparse_a, sparse_b]
    )
    sparse_weighted = HybridRetriever(alpha=0.1).fuse(
        [dense_a, dense_b], [sparse_a, sparse_b]
    )

    assert dense_weighted[0].chunk_id == "a"
    assert sparse_weighted[0].chunk_id == "b"


def test_hybrid_handles_dense_only() -> None:
    hit = DenseHit(chunk_id="a", score=0.88, doc=_doc("A", "a", "semantic"))

    fused = HybridRetriever(alpha=0.7).fuse([hit], [])

    assert fused[0].chunk_id == "a"
    assert fused[0].doc.retrieval_source == "dense"
    assert fused[0].doc.vector_score == 0.88


def test_hybrid_handles_sparse_only() -> None:
    hit = SparseHit(chunk_id="b", score=0.77, doc=_doc("B", "b", "keyword"))

    fused = HybridRetriever(alpha=0.7).fuse([], [hit])

    assert fused[0].chunk_id == "b"
    assert fused[0].doc.retrieval_source == "sparse"
    assert fused[0].doc.sparse_score == 0.77


def test_hybrid_alpha_is_clamped() -> None:
    assert HybridRetriever(alpha=-1).alpha == 0.0
    assert HybridRetriever(alpha=2).alpha == 1.0


def test_local_sparse_retriever_finds_keyword_match() -> None:
    retriever = LocalSparseRetriever(
        docs=[
            _doc("Dexcom G7 FAQ", "a", "Dexcom G7 sensor waterproof 24 hours"),
            _doc("Order FAQ", "b", "shipping order refund"),
        ]
    )

    hits = retriever.search("Dexcom G7 防水 waterproof", top_k=2)

    assert hits
    assert "Dexcom G7 FAQ" in [h.doc.source_title for h in hits]


def test_local_sparse_retriever_filters_explicit_product_tags() -> None:
    gs3 = _doc("GS3 蓝牙连接", "gs3", "蓝牙连接失败")
    gs3.product_tags = ["GS3"]
    eco = _doc("ECO 蓝牙连接", "eco", "蓝牙连接失败")
    eco.product_tags = ["ECO"]

    hits = LocalSparseRetriever(docs=[gs3, eco]).search(
        "蓝牙连接", top_k=2, product_tags=["ECO"]
    )

    assert [hit.doc.source_title for hit in hits] == ["ECO 蓝牙连接"]


def test_rag_service_hybrid_uses_sparse_when_dense_unavailable() -> None:
    settings = DemoSettings(
        qwen_api_base="",
        qwen_api_key="",
        embedding_api_base="",
        embedding_api_key="",
        agent_retrieval_strategy="hybrid",
        agent_top_k=2,
        agent_min_relevance_score=0.0,
    )
    service = RagService(settings=settings)

    docs = service.retrieve("GS3 佩戴体验")

    assert docs
    assert docs[0].retrieval_source == "sparse"


def test_dense_docs_to_hits_uses_stable_chunk_identity() -> None:
    doc = _doc("Dexcom G7 FAQ", "https://example.com", "text", score=0.9)
    doc.chunk_id = "stable-id"

    hits = dense_docs_to_hits([doc])

    assert hits[0].chunk_id == "stable-id"


def test_local_sparse_docs_use_split_chunk_identity() -> None:
    docs = _load_local_docs()

    assert docs
    assert all(doc.chunk_id for doc in docs)
    assert len({doc.chunk_id for doc in docs}) == len(docs)
