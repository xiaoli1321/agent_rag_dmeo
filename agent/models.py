from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

try:
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover - dependency guard for pure model tests

    def add_messages(left: list | None, right: list | None) -> list:
        return list(left or []) + list(right or [])


Intent = Literal["产品咨询", "使用问题", "售后诉求", "闲聊"]
Emotion = Literal["平静", "不满", "愤怒"]
ActiveAgent = Literal[
    "product_consultant", "after_sales", "empathy_agent", "clarify", "smalltalk"
]
TurnRelation = Literal[
    "new_request", "follow_up", "clarification_answer", "correction", "other"
]
Actionability = Literal["ready", "needs_clarification", "unsupported"]
MissingSlot = Literal[
    "target_product", "user_goal", "problem_detail", "reference_target"
]
ClarificationReason = Literal[
    "missing_target",
    "missing_goal",
    "missing_detail",
    "ambiguous_reference",
    "low_confidence",
]
DialogueStatus = Literal["ready", "awaiting_clarification", "handed_off", "completed"]


class PerceptionEntities(BaseModel):
    product: str | None = Field(
        default=None, description="用户明确提到或已由会话确认的产品/型号。"
    )
    issue: str | None = Field(
        default=None, description="用户明确描述的故障、现象或咨询属性。"
    )
    requested_action: str | None = Field(
        default=None, description="用户希望查询、排障或办理的动作。"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_blank_strings(cls, value: Any) -> Any:
        """Treat provider-produced empty entity values as missing slots."""
        if not isinstance(value, dict):
            return value
        return {
            key: None if isinstance(item, str) and not item.strip() else item
            for key, item in value.items()
        }


class IntentDraft(BaseModel):
    """LLM 的语义理解结果；不包含任何路由或澄清策略。"""

    intent: Intent
    emotion: Emotion = "平静"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    handoff_requested: bool = False
    is_greeting: bool = False
    secondary_intents: list[Intent] = Field(default_factory=list)
    entities: PerceptionEntities = Field(default_factory=PerceptionEntities)
    evidence: str = Field(default="", max_length=240)

    @model_validator(mode="before")
    @classmethod
    def normalize_nullable_objects(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        # Some OpenAI-compatible providers serialize optional nested objects as null.
        if normalized.get("entities") is None:
            normalized["entities"] = {}
        if normalized.get("secondary_intents") is None:
            normalized["secondary_intents"] = []
        normalized["intent"] = {
            "product_consultation": "产品咨询",
            "product_inquiry": "产品咨询",
            "usage_issue": "使用问题",
            "troubleshooting": "使用问题",
            "after_sales": "售后诉求",
            "aftersales": "售后诉求",
            "smalltalk": "闲聊",
            "out_of_scope": "闲聊",
        }.get(normalized.get("intent"), normalized.get("intent"))
        normalized["emotion"] = {
            "calm": "平静",
            "neutral": "平静",
            "frustrated": "不满",
            "dissatisfied": "不满",
            "angry": "愤怒",
        }.get(normalized.get("emotion"), normalized.get("emotion"))
        entities = normalized.get("entities")
        if isinstance(entities, dict) and "issue" not in entities:
            normalized["entities"] = {
                **entities,
                "issue": entities.get("issue_type") or entities.get("problem"),
            }
        return normalized

    @model_validator(mode="after")
    def remove_primary_from_secondary(self) -> "IntentDraft":
        self.secondary_intents = [
            intent
            for intent in dict.fromkeys(self.secondary_intents)
            if intent != self.intent
        ]
        return self


class ClarificationDecision(BaseModel):
    needed: bool = False
    reason: ClarificationReason | None = None
    missing_slots: list[MissingSlot] = Field(default_factory=list)
    question: str | None = None
    options: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_shape(self) -> "ClarificationDecision":
        if self.needed:
            if (
                not self.reason
                or not self.missing_slots
                or not (self.question or "").strip()
            ):
                raise ValueError(
                    "needed clarification requires reason, missing_slots and question"
                )
            self.question = self.question.strip()
            self.options = [
                option.strip() for option in self.options if option.strip()
            ][:4]
        else:
            self.reason = None
            self.missing_slots = []
            self.question = None
            self.options = []
        return self


class PerceptionResult(BaseModel):
    intent: Intent = Field(description="用户当前消息的主意图，只能从枚举中选择。")
    emotion: Emotion = Field(description="用户当前消息的情绪强度，只能从枚举中选择。")
    confidence: float = Field(ge=0.0, le=1.0, description="分类置信度，0 到 1。")
    handoff_requested: bool = Field(
        description="用户是否明确要求人工、客服、投诉、退款或赔偿。"
    )
    secondary_intents: list[Intent] = Field(
        default_factory=list, description="复合诉求中的次要意图。"
    )
    turn_relation: TurnRelation = Field(
        default="new_request", description="当前消息与上一轮的关系。"
    )
    actionability: Actionability = Field(
        default="ready", description="当前信息能否安全进入下游执行。"
    )
    entities: PerceptionEntities = Field(default_factory=PerceptionEntities)
    clarification: ClarificationDecision = Field(default_factory=ClarificationDecision)
    intent_evidence: str = Field(
        default="", description="分类命中的用户表述，便于调试。"
    )
    classifier_source: Literal["llm", "fallback", "injected"] = "fallback"
    policy_reason: str = Field(default="", description="确定性路由/澄清策略的依据。")
    reason: str = Field(
        default="模型未返回分类依据。", description="一句话解释分类依据，便于调试路由。"
    )

    @model_validator(mode="after")
    def validate_actionability(self) -> "PerceptionResult":
        if (
            self.actionability == "needs_clarification"
            and not self.clarification.needed
        ):
            raise ValueError("needs_clarification requires clarification.needed=true")
        if self.actionability != "needs_clarification" and self.clarification.needed:
            self.clarification = ClarificationDecision()
        self.secondary_intents = [
            intent
            for intent in dict.fromkeys(self.secondary_intents)
            if intent != self.intent
        ]
        return self


class PendingClarification(BaseModel):
    original_request: str
    suspected_intent: Intent
    missing_slots: list[MissingSlot] = Field(default_factory=list)
    asked_slots: list[MissingSlot] = Field(default_factory=list)
    collected_entities: PerceptionEntities = Field(default_factory=PerceptionEntities)
    turn_count: int = Field(default=0, ge=0)
    last_question: str | None = None


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
    retrieval_rank: int | None = None
    retrieval_source: Literal["dense", "sparse", "hybrid", "fallback"] = "dense"


class EvidenceDecision(BaseModel):
    status: Literal["grounded", "insufficient_evidence"]
    reason: str
    top_score: float | None = None
    has_numeric_support: bool | None = None


FailureType = Literal[
    "knowledge_missing", "retrieval_mismatch", "hallucination", "format_unstable"
]


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

    rewritten_question: str = Field(
        description="Standalone query that preserves the user's intent and constraints."
    )
    reason: str = Field(description="Short explanation of the rewrite decision.")


class RelevanceGrade(BaseModel):
    """Self-RAG/CRAG document relevance decision."""

    binary_score: Literal["yes", "no"] = Field(
        description="yes only when this document can help answer the question."
    )
    reason: str = Field(description="Short evidence-based reason.")


class GroundingGrade(BaseModel):
    """Post-generation answer grounding decision."""

    grounded: bool = Field(
        description="Whether every factual claim in the answer is supported by the evidence."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Unsupported factual claims, if any."
    )
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
    intent_draft: IntentDraft
    injected_perception: PerceptionResult
    perception_trace: dict[str, Any]
    active_agent: ActiveAgent
    current_topic: str | None
    retrieved_docs: list[RetrievedDoc]
    debug_trace: dict[str, Any]
    answer: str
    answer_status: str | None
    dialogue_status: DialogueStatus
    pending_clarification: PendingClarification | None
    turn_relation: TurnRelation
    resolved_user_message: str | None
    failed_rag_count: int
    handoff_reason: str | None
    handoff_summary: str | None
