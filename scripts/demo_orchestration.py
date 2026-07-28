"""编排层 (LangGraph Orchestration Layer) 多轮对话与动态路由演示脚本

演示功能：
1. 基于 StateGraph 的离散状态机编排：入口节点为感知节点，精准条件路由（产品咨询/使用问题 -> RAG，愤怒 -> 安抚，安抚无效/用户要求 -> 转人工）。
2. Checkpointer (InMemorySaver) 实现多轮对话记忆（绑定 thread_id）。
3. 演示指代继承：
   - 第一轮：问“GS3 蓝牙连不上怎么办？”（锁定 current_topic="GS3"）
   - 第二轮：问“它防水吗？”（自动将“它”理解继承为第一轮聊的“GS3”）
4. 演示愤怒安抚与转人工流转。
"""

import json
import sys
from pathlib import Path

# 支持脚本直接执行和作为包模块执行
current_file = Path(__file__).resolve()
package_root = current_file.parent.parent
if str(package_root.parent) not in sys.path:
    sys.path.insert(0, str(package_root.parent))

from customer_agent_demo.agent.graph import CustomerAgent
from customer_agent_demo.config import get_settings


def main():
    settings = get_settings()
    agent = CustomerAgent(settings=settings)

    thread_id = "demo_session_001"

    print("================ 编排层 (LangGraph) 多轮动态路由演示 ================\n")
    print(f"会话 ID (thread_id): {thread_id}\n")

    turns = [
        ("第一轮：咨询具体产品故障", "GS3 蓝牙连不上怎么办？"),
        ("第二轮：利用 Checkpointer 进行指代继承", "它支持防水吗？"),
        (
            "第三轮：触发愤怒安抚与转人工路由",
            "你们服务太差了，搞了半天还是不行，马上给我退款转人工！",
        ),
    ]

    for label, message in turns:
        print(f"👉 [{label}]")
        print(f'用户输入: "{message}"')

        # 执行图编排，传入 thread_id 保持会话 Memory Checkpoint
        state = agent.invoke(message, thread_id=thread_id)

        perception = state.get("perception")
        active_agent = state.get("active_agent")
        current_topic = state.get("current_topic")
        answer = state.get("answer", "")

        print("\n=== 编排层内部状态 (AgentState) 判定说明 ===")
        print(f"- 命中感知意图: {perception.intent if perception else '未知'}")
        print(f"- 情绪感知标签: {perception.emotion if perception else '未知'}")
        print(f"- 置信度: {perception.confidence if perception else '未知'}")
        print(f"- 跨轮继承锁定主题 (current_topic): {current_topic or '未锁定'}")
        print(f"- 激活的目标代理节点 (active_agent): {active_agent}")
        print(
            f"- 是否触发转人工 (handoff_requested): {perception.handoff_requested if perception else False}"
        )

        print("\n=== 系统最终输出给用户的回答 ===")
        print(answer)
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
