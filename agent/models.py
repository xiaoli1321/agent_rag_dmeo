from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

try:
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover - dependency guard for pure model tests
    def add_messages(left: list | None, right: list | None) -> list:
        return list(left or []) + list(right or [])


Intent = Literal["产品咨询", "使用问题", "售后诉求", "闲聊"]
Emotion = Literal["平静", "不满", "愤怒"]
ActiveAgent = Literal["product_consultant", "after_sales", "empathy_agent", "pending_clarification"]


class PerceptionResult(BaseModel):
    intent: Intent = Field(description="用户当前消息的主意图，只能从枚举中选择。")
    emotion: Emotion = Field(description="用户当前消息的情绪强度，只能从枚举中选择。")
    confidence: float = Field(ge=0.0, le=1.0, description="分类置信度，0 到 1。")
    handoff_requested: bool = Field(description="用户是否明确要求人工、客服、投诉、退款或赔偿。")
    reason: str = Field(default="模型未返回分类依据。", description="一句话解释分类依据，便于调试路由。")


class RetrievedDoc(BaseModel):
    chunk_id: str | None = None
    source_title: str
    source_url: str
    chunk_text: str
    score: float
    product_tags: list[str] = Field(default_factory=list)
    vector_score: float | None = None
    sparse_score: float | None = None
    final_score: float | None = None
    retrieval_source: Literal["dense", "sparse", "hybrid", "fallback"] = "dense"


class EvidenceDecision(BaseModel):
    status: Literal["grounded", "insufficient_evidence"]
    reason: str
    top_score: float | None = None
    has_numeric_support: bool | None = None


FailureType = Literal["knowledge_missing", "retrieval_mismatch", "hallucination", "format_unstable"]


class DocumentGrade(BaseModel):
    source_title: str
    source_url: str
    binary_score: Literal["yes", "no"]
    reason: str
    failure_type: FailureType | None = None
    score: float | None = None
    grader: Literal["llm", "heuristic"] = "heuristic"
    attempt: int = 0


class HallucinationDecision(BaseModel):
    status: Literal["grounded", "failed"]
    reason: str
    failure_type: FailureType | None = None
    risky_numbers: list[str] = Field(default_factory=list)
    grader: Literal["llm", "heuristic"] = "heuristic"
    unsupported_claims: list[str] = Field(default_factory=list)


class QueryRewrite(BaseModel):
    """A retrieval-ready query, kept separate from the user's original wording."""

    rewritten_question: str = Field(description="Standalone query that preserves the user's intent and constraints.")
    reason: str = Field(description="Short explanation of the rewrite decision.")


class RelevanceGrade(BaseModel):
    """Self-RAG/CRAG document relevance decision."""

    binary_score: Literal["yes", "no"] = Field(description="yes only when this document can help answer the question.")
    reason: str = Field(description="Short evidence-based reason.")


class GroundingGrade(BaseModel):
    """Post-generation answer grounding decision."""

    grounded: bool = Field(description="Whether every factual claim in the answer is supported by the evidence.")
    unsupported_claims: list[str] = Field(default_factory=list, description="Unsupported factual claims, if any.")
    reason: str = Field(description="Short evidence-based reason.")


class RagResult(BaseModel):
    answer: str
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    answer_status: Literal["grounded", "insufficient_evidence"]
    evidence_decision: EvidenceDecision | None = None
    debug_trace: dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    perception: PerceptionResult
    active_agent: ActiveAgent
    current_topic: str | None
    retrieved_docs: list[RetrievedDoc]
    failed_rag_count: int
    handoff_reason: str | None
    clarification_count: int
