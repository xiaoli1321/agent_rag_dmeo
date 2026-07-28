"""感知层 (Perception Layer) 独立运行与演示脚本

功能：
1. 输入用户消息 (message)
2. 通过 LangChain chat.with_structured_output(IntentDraft) + Pydantic 提取结构化数据
3. temperature=0 确保输出可预测性与高稳定性
4. 输出结构化 JSON：包含【意图】(产品咨询/使用问题/售后诉求/闲聊) + 【情绪】(平静/不满/愤怒) + 【置信度】(confidence)
"""

import json
import sys
from pathlib import Path

# 支持脚本直接执行和作为包模块执行
current_file = Path(__file__).resolve()
package_root = current_file.parent.parent
if str(package_root.parent) not in sys.path:
    sys.path.insert(0, str(package_root.parent))

from customer_agent_demo.agent.perception import PerceptionService
from customer_agent_demo.config import get_settings


def classify_message_to_json(message: str, temperature: float = 0.0) -> str:
    """输入用户消息，输出格式化的结构化 JSON 字符串"""
    settings = get_settings()
    service = PerceptionService(settings=settings, temperature=temperature)

    # 调用感知层分类（with_structured_output，temperature=0）
    draft, source = service.classify_draft(message)

    # 提取核心需求要求的 JSON 结构：意图 + 情绪 + 置信度 + 其他结构化字段
    output_dict = {
        "input_message": message,
        "intent": draft.intent,
        "emotion": draft.emotion,
        "confidence": draft.confidence,
        "handoff_requested": draft.handoff_requested,
        "secondary_intents": draft.secondary_intents,
        "entities": draft.entities.model_dump(),
        "evidence": draft.evidence,
        "classifier_source": source,
    }

    return json.dumps(output_dict, ensure_ascii=False, indent=2)


def main():
    test_messages = [
        "GS3 传感器怎么连不上手机蓝牙？",
        "你们这个垃圾传感器又不能用了，气死我了，赶紧给我退款！",
        "请问 Dexcom G7 支持 waterproof 防水吗？",
        "你好，今天天气真不错。",
    ]

    print("================ 感知层 结构化 JSON 输出演示 ================\n")
    for msg in test_messages:
        print(f"用户消息: {msg}")
        json_output = classify_message_to_json(msg, temperature=0.0)
        print("结构化 JSON 输出:")
        print(json_output)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
