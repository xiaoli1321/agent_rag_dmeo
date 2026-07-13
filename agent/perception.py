from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from customer_agent_demo.agent.models import PerceptionResult
from customer_agent_demo.agent.prompts import load_prompt
from customer_agent_demo.config import DemoSettings, get_settings


@dataclass(slots=True)
class PerceptionService:
    settings: DemoSettings
    temperature: float = 0.0

    def classify(self, message: str, history: Iterable[str] | None = None) -> PerceptionResult:
        if not self.settings.llm_configured:
            return heuristic_perception(message)

        history_text = "\n".join(history or [])
        chat = ChatOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_api_base,
            model=self.settings.llm_model,
            temperature=self.temperature,
            max_tokens=min(self.settings.llm_max_tokens, 1000),
            extra_body=self.settings.llm_extra_body,
        )
        structured = chat.with_structured_output(PerceptionResult)
        result = structured.invoke(
            [
                SystemMessage(content=load_prompt("perception.md")),
                HumanMessage(content=f"会话上下文：\n{history_text}\n\n当前用户消息：{message}"),
            ]
        )
        if isinstance(result, PerceptionResult):
            return result
        return PerceptionResult.model_validate(result)


def heuristic_perception(message: str) -> PerceptionResult:
    """Deterministic fallback for local tests and missing model config."""
    text = message.lower()
    handoff_words = ("人工", "客服", "坐席", "投诉", "退款", "赔偿", "换货", "退货")
    angry_words = ("太差", "垃圾", "投诉", "赔偿", "气死", "马上", "再也不用", "坏了")
    aftersales_words = ("订单", "物流", "发货", "退款", "换货", "退货", "保修", "补发", "投诉", "赔偿")
    usage_words = ("连不上", "连接", "佩戴", "脱落", "读数", "数据不准", "不准", "告警", "校准", "坏了")
    product_words = ("防水", "游泳", "洗澡", "cgm", "传感器", "dexcom", "libre", "三诺", "硅基", "血糖")

    handoff = any(word in text for word in handoff_words)
    emotion = "愤怒" if any(word in text for word in angry_words) else "平静"
    if emotion == "平静" and any(word in text for word in ("没有用", "不行", "怎么回事", "烦", "服了")):
        emotion = "不满"

    if handoff or any(word in text for word in aftersales_words):
        intent = "售后诉求"
    elif any(word in text for word in usage_words):
        intent = "使用问题"
    elif any(word in text for word in product_words):
        intent = "产品咨询"
    else:
        intent = "闲聊"

    return PerceptionResult(
        intent=intent,
        emotion=emotion,
        confidence=0.72 if intent != "闲聊" else 0.55,
        handoff_requested=handoff,
        reason="本地启发式分类，用于未配置 Qwen 时的开发和测试。",
    )


def run_stability_experiment(samples: list[str] | None = None, rounds: int = 10) -> dict[str, object]:
    settings = get_settings()
    samples = samples or [
        "Dexcom G7 可以戴着洗澡吗？",
        "我的传感器连不上手机，怎么办？",
        "你们这个太差了，我要投诉，马上转人工！",
        "我的订单为什么还没发货？",
    ]
    report: dict[str, object] = {}
    for temperature in (0.0, 0.7):
        service = PerceptionService(settings=settings, temperature=temperature)
        temp_rows = []
        for sample in samples:
            labels = []
            for _ in range(rounds):
                result = service.classify(sample)
                labels.append(f"{result.intent}/{result.emotion}/{result.handoff_requested}")
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
