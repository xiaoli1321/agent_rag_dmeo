from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .models import (
    ClarificationDecision,
    IntentDraft,
    PendingClarification,
    PerceptionEntities,
    PerceptionResult,
)
from .intent_catalog import catalog_prompt_context, load_intent_catalog
from .prompts import load_prompt
from ..config import DemoSettings, get_settings


@dataclass(slots=True)
class PerceptionService:
    settings: DemoSettings
    temperature: float = 0.0

    def classify(
        self,
        message: str,
        history: Iterable[str] | None = None,
        *,
        current_topic: str | None = None,
        pending_clarification: PendingClarification | None = None,
    ) -> PerceptionResult:
        """Compatibility facade used by evaluation scripts.

        Graph orchestration calls ``classify_draft`` and ``decide_perception`` as
        separate nodes. Keeping this wrapper avoids breaking callers outside the
        graph while making policy deterministic in the new path.
        """
        draft, source = self.classify_draft(
            message, history, current_topic=current_topic
        )
        return decide_perception(
            draft,
            message=message,
            current_topic=current_topic,
            pending_clarification=pending_clarification,
            turn_relation="new_request",
            classifier_source=source,
        )

    def classify_draft(
        self,
        message: str,
        history: Iterable[str] | None = None,
        *,
        current_topic: str | None = None,
    ) -> tuple[IntentDraft, str]:
        if not self.settings.llm_configured:
            return heuristic_draft(message, current_topic=current_topic), "fallback"

        history_rows = list(history or [])[
            -self.settings.agent_perception_history_turns :
        ]
        history_text = "\n".join(history_rows) or "无"
        chat = ChatOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_api_base,
            model=self.settings.llm_model,
            temperature=self.temperature,
            max_tokens=min(self.settings.llm_max_tokens, 1000),
            extra_body=self.settings.llm_extra_body,
        )
        try:
            structured = chat.with_structured_output(IntentDraft, method="json_mode")
            result = structured.invoke(
                [
                    SystemMessage(
                        content=f"{load_prompt('perception.md')}\n\n当前意图目录：\n{catalog_prompt_context()}"
                    ),
                    HumanMessage(
                        content=(
                            f"当前已确认产品：{current_topic or '无'}\n"
                            f"最近会话上下文（带角色）：\n{history_text}\n\n当前用户消息：{message}"
                        )
                    ),
                ]
            )
            return (
                result
                if isinstance(result, IntentDraft)
                else IntentDraft.model_validate(result)
            ), "llm"
        except Exception:
            return heuristic_draft(message, current_topic=current_topic), "fallback"

    def generate_empathy(
        self,
        message: str,
        *,
        intent: str = "售后诉求",
        handoff_requested: bool = False,
        issue: str | None = None,
    ) -> str:
        if not self.settings.llm_configured:
            return heuristic_empathy(
                message, intent=intent, handoff_requested=handoff_requested, issue=issue
            )

        chat = ChatOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_api_base,
            model=self.settings.llm_model,
            temperature=0.3,
            max_tokens=250,
            extra_body=self.settings.llm_extra_body,
        )
        try:
            res = chat.invoke(
                [
                    SystemMessage(content=load_prompt("empathy.md")),
                    HumanMessage(
                        content=(
                            f"用户意图：{intent}\n"
                            f"已识别的具体痛点/问题：{issue or '未明'}\n"
                            f"是否明确请求人工：{'是' if handoff_requested else '否'}\n"
                            f"用户原话：{message}"
                        )
                    ),
                ]
            )
            return str(res.content).strip()
        except Exception as exc:
            raise RuntimeError("LLM empathy generation failed") from exc


def heuristic_empathy(
    message: str,
    *,
    intent: str = "售后诉求",
    handoff_requested: bool = False,
    issue: str | None = None,
) -> str:
    pain_point = issue or (
        "退换货与售后问题" if intent == "售后诉求" else "设备使用遇到的不便"
    )
    if handoff_requested or intent == "售后诉求":
        return (
            f"非常理解您遇到【{pain_point}】时的焦虑与不满，换作是我也会非常着急。"
            "请您放心，如果属于设备质量或非人为损坏，我们符合售后维权与免费补发标准。我已为您开启优先绿色通道转接人工专员。"
        )
    return (
        f"十分抱歉【{pain_point}】给您带来了不好的体验。我已为您启动优先排查，"
        "先为您提供下一步排查建议；若仍未解决，您可以随时回复“转人工”，我会立刻为您对接专员。"
    )


def heuristic_draft(message: str, *, current_topic: str | None = None) -> IntentDraft:
    """Convert the deterministic local classifier into the same semantic contract."""
    result = heuristic_perception(message, current_topic=current_topic)
    return IntentDraft(
        intent=result.intent,
        emotion=result.emotion,
        confidence=result.confidence,
        handoff_requested=result.handoff_requested,
        secondary_intents=result.secondary_intents,
        entities=result.entities.model_copy(deep=True),
        evidence=result.reason,
        is_greeting=result.intent == "闲聊" and result.actionability == "ready",
    )


def decide_perception(
    draft: IntentDraft,
    *,
    message: str,
    current_topic: str | None,
    pending_clarification: PendingClarification | None,
    turn_relation: str,
    classifier_source: str,
) -> PerceptionResult:
    """Apply business policy after semantic understanding; this is intentionally model-free."""
    if _is_medical_out_of_scope(message):
        return PerceptionResult(
            intent="闲聊",
            emotion=draft.emotion,
            confidence=1.0,
            handoff_requested=False,
            turn_relation=turn_relation,
            actionability="unsupported",
            entities=PerceptionEntities(),
            reason="医疗紧急或诊疗表达不在当前 CGM 客服 Demo 能力范围内。",
            intent_evidence=message[:120],
            classifier_source=classifier_source,
            policy_reason="医疗风险按已确认边界作为域外问题处理，不进入 RAG 或售后。",
        )
    entities = draft.entities.model_copy(deep=True)
    # “坏了/不行”等只有笼统故障词的表达不能被模型臆测为售后；除非用户明确
    # 提出退款、换货、人工等诉求，否则先收集具体故障信息再决定是否转人工。
    if _is_vague_device_issue(message) and not _has_explicit_handoff_request(message):
        draft = draft.model_copy(
            update={"intent": "使用问题", "handoff_requested": False}
        )
        entities.issue = None
    definition = load_intent_catalog()[draft.intent]
    if entities.issue in {"坏了", "不行", "有问题", "用不了", "不能用"}:
        entities.issue = None
    if not entities.product and current_topic:
        entities.product = current_topic

    if draft.handoff_requested or definition.direct_handoff:
        return _decision_from_draft(
            draft,
            entities,
            turn_relation,
            classifier_source,
            actionability="ready",
            policy_reason="意图目录声明为直接人工交接。",
        )

    if pending_clarification and turn_relation in {
        "clarification_answer",
        "correction",
    }:
        merged = pending_clarification.collected_entities.model_copy(deep=True)
        for field_name in ("product", "issue", "requested_action"):
            value = getattr(entities, field_name)
            if value:
                setattr(merged, field_name, value)
        missing = pending_clarification.missing_slots[0]
        resolved = _slot_is_resolved(missing, merged)
        if resolved:
            resolved_draft = draft.model_copy(
                update={"intent": pending_clarification.suspected_intent}
            )
            return _decision_from_draft(
                resolved_draft,
                merged,
                turn_relation,
                classifier_source,
                actionability="ready",
                policy_reason="用户已补齐上一轮唯一缺失槽位。",
            )
        return _clarification_decision(
            draft.model_copy(update={"intent": pending_clarification.suspected_intent}),
            merged,
            turn_relation,
            classifier_source,
            missing,
        )

    missing = _first_missing_slot(definition, entities)
    if missing:
        return _clarification_decision(
            draft,
            entities,
            turn_relation,
            classifier_source,
            _clarification_slot(missing),
        )
    return _decision_from_draft(
        draft,
        entities,
        turn_relation,
        classifier_source,
        actionability="ready"
        if draft.is_greeting
        else definition.default_actionability,
        policy_reason="意图目录的必填槽位已满足，按目录声明的处理器执行。",
    )


def _decision_from_draft(
    draft: IntentDraft,
    entities: PerceptionEntities,
    turn_relation: str,
    classifier_source: str,
    *,
    actionability: str,
    policy_reason: str,
) -> PerceptionResult:
    return PerceptionResult(
        intent=draft.intent,
        emotion=draft.emotion,
        confidence=draft.confidence,
        handoff_requested=draft.handoff_requested,
        secondary_intents=draft.secondary_intents,
        turn_relation=turn_relation,
        actionability=actionability,
        entities=entities,
        reason=draft.evidence or policy_reason,
        intent_evidence=draft.evidence,
        classifier_source=classifier_source,
        policy_reason=policy_reason,
    )


def _clarification_decision(
    draft: IntentDraft,
    entities: PerceptionEntities,
    turn_relation: str,
    classifier_source: str,
    slot: str,
) -> PerceptionResult:
    return PerceptionResult(
        intent=draft.intent,
        emotion=draft.emotion,
        confidence=draft.confidence,
        handoff_requested=False,
        secondary_intents=draft.secondary_intents,
        turn_relation=turn_relation,
        actionability="needs_clarification",
        entities=entities,
        clarification=ClarificationDecision(
            needed=True,
            reason=_reason_for_slot(slot),
            missing_slots=[slot],
            question=_question_for_slot(slot, entities.product),
            options=_options_for_slot(slot),
        ),
        reason="当前问题属于 CGM 范围，但缺少进入下游所需的关键信息。",
        intent_evidence=draft.evidence,
        classifier_source=classifier_source,
        policy_reason=f"缺少 {slot}，每轮只追问一个槽位。",
    )


def _first_missing_slot(definition: object, entities: PerceptionEntities) -> str | None:
    for slot in definition.clarification_order:
        if slot in definition.required_slots and not getattr(entities, slot):
            return slot
    return next(
        (slot for slot in definition.required_slots if not getattr(entities, slot)),
        None,
    )


def _clarification_slot(entity_field: str) -> str:
    return {
        "product": "target_product",
        "issue": "problem_detail",
        "requested_action": "user_goal",
    }[entity_field]


def _slot_is_resolved(slot: str, entities: PerceptionEntities) -> bool:
    entity_field = {
        "target_product": "product",
        "reference_target": "product",
        "problem_detail": "issue",
        "user_goal": "requested_action",
    }[slot]
    return bool(getattr(entities, entity_field))


def _is_medical_out_of_scope(message: str) -> bool:
    return any(
        marker in message
        for marker in ("低血糖昏迷", "昏迷", "出血", "严重过敏", "呼吸困难", "急救")
    )


def _is_vague_device_issue(message: str) -> bool:
    return any(
        marker in message.strip().lower()
        for marker in ("坏了", "不行", "有问题", "用不了", "不能用")
    )


def _has_explicit_handoff_request(message: str) -> bool:
    return any(
        marker in message
        for marker in (
            "人工",
            "客服",
            "坐席",
            "投诉",
            "退款",
            "赔偿",
            "换货",
            "退货",
            "补发",
            "保修",
        )
    )


def heuristic_perception(
    message: str,
    *,
    current_topic: str | None = None,
    pending_clarification: PendingClarification | None = None,
) -> PerceptionResult:
    """Deterministic fallback for local tests and missing model config."""
    stripped = message.strip()
    text = stripped.lower()
    handoff_words = ("人工", "客服", "坐席", "投诉", "退款", "赔偿", "换货", "退货")
    angry_words = ("太差", "垃圾", "气死", "再也不用", "无法接受", "必须马上")
    aftersales_words = (
        "订单",
        "物流",
        "发货",
        "退款",
        "换货",
        "退货",
        "保修",
        "补发",
        "投诉",
        "赔偿",
    )
    usage_words = (
        "怎么用",
        "使用",
        "连不上",
        "连接",
        "佩戴",
        "脱落",
        "读数",
        "数据不准",
        "不准",
        "告警",
        "校准",
        "坏了",
    )
    product_words = (
        "防水",
        "游泳",
        "洗澡",
        "cgm",
        "传感器",
        "dexcom",
        "libre",
        "三诺",
        "硅基",
        "血糖",
        "gs1",
        "gs3",
        "eco",
        "metatwin",
        "ks3",
    )
    greeting_words = ("你好", "您好", "嗨", "hello", "你是谁", "谢谢", "再见")
    vague_issue_words = {"坏了", "不行", "有问题", "用不了", "不能用"}
    unclear_answers = {"不知道", "不清楚", "都不是", "不确定", "没注意"}

    handoff = any(word in text for word in handoff_words)
    emotion = "愤怒" if any(word in text for word in angry_words) else "平静"
    if emotion == "平静" and any(
        word in text for word in ("没有用", "不行", "怎么回事", "烦", "服了")
    ):
        emotion = "不满"

    has_aftersales = any(word in text for word in aftersales_words)
    has_usage = any(word in text for word in usage_words)
    has_product = any(word in text for word in product_words)
    product = _extract_product(stripped) or current_topic
    issue = _extract_issue(stripped, usage_words, product_words)
    requested_action = _extract_requested_action(stripped)

    if handoff or has_aftersales:
        intent = "售后诉求"
    elif has_usage:
        intent = "使用问题"
    elif has_product:
        intent = "产品咨询"
    else:
        intent = "闲聊"

    secondary_intents = []
    if intent == "售后诉求" and (has_usage or has_product):
        secondary_intents.append("使用问题" if has_usage else "产品咨询")

    if pending_clarification and not handoff and not has_aftersales:
        return _classify_clarification_reply(
            stripped,
            pending_clarification=pending_clarification,
            current_topic=current_topic,
            emotion=emotion,
            product=product,
            issue=issue,
            requested_action=requested_action,
            unclear=stripped in unclear_answers,
        )

    if handoff or has_aftersales:
        return PerceptionResult(
            intent="售后诉求",
            secondary_intents=secondary_intents,
            emotion=emotion,
            confidence=0.94,
            handoff_requested=handoff,
            actionability="ready",
            entities=PerceptionEntities(
                product=product, issue=issue, requested_action=requested_action
            ),
            reason="识别到明确的人工、订单或售后办理诉求。",
        )

    if len(stripped) <= 12 and any(word in text for word in greeting_words):
        return PerceptionResult(
            intent="闲聊",
            emotion=emotion,
            confidence=0.95,
            handoff_requested=False,
            actionability="ready",
            entities=PerceptionEntities(),
            reason="当前消息是问候或普通闲聊。",
        )

    pronouns = ("这个", "那个", "它", "该设备", "这款")
    has_ambiguous_reference = (
        any(word in stripped for word in pronouns) and not current_topic
    )
    if has_ambiguous_reference and (has_usage or has_product or "怎么" in stripped):
        return _needs_clarification(
            intent="使用问题" if (has_usage or "怎么" in stripped) else "产品咨询",
            emotion=emotion,
            confidence=0.61,
            slot="reference_target",
            reason="ambiguous_reference",
            question="你说的“这个”具体是哪个产品或型号？",
            options=["GS3", "Dexcom G7", "硅基手表", "不清楚"],
            entities=PerceptionEntities(issue=issue, requested_action=requested_action),
        )

    if intent in {"产品咨询", "使用问题"} and not product:
        return _needs_clarification(
            intent=intent,
            emotion=emotion,
            confidence=0.62,
            slot="target_product",
            reason="missing_target",
            question="请问你咨询的是哪个 CGM 产品或设备型号？",
            options=["GS3", "Dexcom G7", "硅基动感 CGM", "不清楚"],
            entities=PerceptionEntities(issue=issue, requested_action=requested_action),
        )

    if intent == "使用问题" and (
        stripped in vague_issue_words
        or (product is not None and issue in vague_issue_words)
    ):
        return _needs_clarification(
            intent="使用问题",
            emotion=emotion,
            confidence=0.68,
            slot="problem_detail",
            reason="missing_detail",
            question=f"{product or '这个设备'}具体出现了什么问题或表现？",
            options=["无法连接", "读数异常", "传感器脱落", "其他问题"],
            entities=PerceptionEntities(product=product, requested_action="排障"),
        )

    if intent == "闲聊":
        return PerceptionResult(
            intent="闲聊",
            emotion=emotion,
            confidence=0.35,
            handoff_requested=False,
            actionability="unsupported",
            entities=PerceptionEntities(),
            reason="未识别到 CGM 产品、使用或售后相关诉求。",
        )

    return PerceptionResult(
        intent=intent,
        emotion=emotion,
        confidence=0.86,
        handoff_requested=handoff,
        secondary_intents=secondary_intents,
        actionability="ready",
        entities=PerceptionEntities(
            product=product, issue=issue, requested_action=requested_action
        ),
        reason="已识别到可直接处理的 CGM 咨询或使用诉求。",
    )


def _classify_clarification_reply(
    message: str,
    *,
    pending_clarification: PendingClarification,
    current_topic: str | None,
    emotion: str,
    product: str | None,
    issue: str | None,
    requested_action: str | None,
    unclear: bool,
) -> PerceptionResult:
    if product and any(
        marker in message for marker in ("吗", "怎么", "多少", "几", "能不能", "可以")
    ):
        fresh = heuristic_perception(
            message, current_topic=None, pending_clarification=None
        )
        fresh.turn_relation = "new_request"
        return fresh
    if any(
        marker in message for marker in ("天气", "新闻", "写代码", "讲故事", "股票")
    ):
        return PerceptionResult(
            intent="闲聊",
            emotion=emotion,  # type: ignore[arg-type]
            confidence=0.92,
            handoff_requested=False,
            turn_relation="new_request",
            actionability="unsupported",
            reason="用户已切换到与 CGM 无关的新话题。",
        )
    correction = any(marker in message for marker in ("不是", "不对", "改成", "应该是"))
    entities = pending_clarification.collected_entities.model_copy(deep=True)
    if product:
        entities.product = product
    if issue:
        entities.issue = issue
    if requested_action:
        entities.requested_action = requested_action

    missing_slot = pending_clarification.missing_slots[0]
    resolved = False
    if missing_slot in {"target_product", "reference_target"}:
        resolved = bool(product and product != current_topic) or bool(
            product and not unclear
        )
    elif missing_slot == "problem_detail":
        resolved = bool(
            issue and message not in {"坏了", "不行", "有问题", "用不了", "不能用"}
        )
    elif missing_slot == "user_goal":
        resolved = bool(requested_action)

    if resolved and not unclear:
        return PerceptionResult(
            intent=pending_clarification.suspected_intent,
            emotion=emotion,  # type: ignore[arg-type]
            confidence=0.88,
            handoff_requested=False,
            turn_relation="correction" if correction else "clarification_answer",
            actionability="ready",
            entities=entities,
            reason="当前消息补充了上一轮缺失的信息，可以继续处理。",
        )

    return _needs_clarification(
        intent=pending_clarification.suspected_intent,
        emotion=emotion,
        confidence=0.58,
        slot=missing_slot,
        reason=_reason_for_slot(missing_slot),
        question=_question_for_slot(missing_slot, entities.product),
        options=_options_for_slot(missing_slot),
        entities=entities,
        turn_relation="clarification_answer",
    )


def _needs_clarification(
    *,
    intent: str,
    emotion: str,
    confidence: float,
    slot: str,
    reason: str,
    question: str,
    options: list[str],
    entities: PerceptionEntities,
    turn_relation: str = "new_request",
) -> PerceptionResult:
    return PerceptionResult(
        intent=intent,  # type: ignore[arg-type]
        emotion=emotion,  # type: ignore[arg-type]
        confidence=confidence,
        handoff_requested=False,
        turn_relation=turn_relation,  # type: ignore[arg-type]
        actionability="needs_clarification",
        entities=entities,
        clarification=ClarificationDecision(
            needed=True,
            reason=reason,  # type: ignore[arg-type]
            missing_slots=[slot],  # type: ignore[list-item]
            question=question,
            options=options,
        ),
        reason="当前问题属于 CGM 范围，但缺少进入下游所需的关键信息。",
    )


def _extract_product(message: str) -> str | None:
    lowered = message.lower()
    products = (
        ("gs1 pro", "GS1 Pro"),
        ("gs3", "GS3"),
        ("gs1", "GS1"),
        ("eco", "ECO"),
        ("metatwin", "MetaTwin"),
        ("ks3", "KS3"),
        ("dexcom", "Dexcom G7"),
        ("g7", "Dexcom G7"),
        ("libre", "FreeStyle Libre"),
        ("硅基手表", "硅基手表"),
        ("健康app", "硅基动感健康APP"),
        ("三诺", "三诺爱看 CGM"),
        ("硅基", "硅基动感 CGM"),
        ("cgm", "CGM"),
        ("传感器", "CGM 传感器"),
    )
    for keyword, product in products:
        if keyword in lowered:
            return product
    return None


def _extract_issue(
    message: str, usage_words: tuple[str, ...], product_words: tuple[str, ...]
) -> str | None:
    for word in usage_words + product_words:
        if word in message.lower():
            return word
    return None


def _extract_requested_action(message: str) -> str | None:
    if any(word in message for word in ("怎么", "怎么办", "解决", "处理")):
        return "排障"
    if any(word in message for word in ("多少", "几", "是否", "吗", "能不能", "可以")):
        return "查询"
    if any(word in message for word in ("退款", "换货", "退货", "补发", "保修")):
        return "办理售后"
    return None


def _reason_for_slot(slot: str) -> str:
    return {
        "reference_target": "ambiguous_reference",
        "target_product": "missing_target",
        "user_goal": "missing_goal",
        "problem_detail": "missing_detail",
    }[slot]


def _question_for_slot(slot: str, product: str | None) -> str:
    return {
        "reference_target": "你说的“这个”具体是哪个产品或型号？",
        "target_product": "请问你咨询的是哪个 CGM 产品或设备型号？",
        "user_goal": "你希望查询产品信息、排查使用问题，还是办理售后？",
        "problem_detail": f"{product or '这个设备'}具体出现了什么问题或表现？",
    }[slot]


def _options_for_slot(slot: str) -> list[str]:
    return {
        "reference_target": ["GS3", "Dexcom G7", "硅基手表", "不清楚"],
        "target_product": ["GS3", "Dexcom G7", "硅基动感 CGM", "不清楚"],
        "user_goal": ["查询产品信息", "排查使用问题", "办理售后", "不清楚"],
        "problem_detail": ["无法连接", "读数异常", "传感器脱落", "其他问题"],
    }[slot]


def run_stability_experiment(
    samples: list[str] | None = None, rounds: int = 10
) -> dict[str, object]:
    settings = get_settings()
    samples = samples or [
        "Dexcom G7 可以戴着洗澡吗？",
        "我的传感器连不上手机，怎么办？",
        "你们这个太差了，我要投诉，马上转人工！",
        "我的订单为什么还没发货？",
        "这个怎么用？",
        "GS3坏了",
        "今天天气怎么样？",
    ]
    report: dict[str, object] = {}
    for temperature in (0.0, 0.7):
        service = PerceptionService(settings=settings, temperature=temperature)
        temp_rows = []
        for sample in samples:
            labels = []
            for _ in range(rounds):
                result = service.classify(sample)
                labels.append(
                    f"{result.intent}/{result.emotion}/{result.actionability}/"
                    f"{result.handoff_requested}/{result.clarification.missing_slots}"
                )
            counts = Counter(labels)
            temp_rows.append(
                {
                    "input": sample,
                    "unique_outputs": len(counts),
                    "distribution": dict(counts),
                }
            )
        report[str(temperature)] = temp_rows
    return report


def main() -> None:
    print(json.dumps(run_stability_experiment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
