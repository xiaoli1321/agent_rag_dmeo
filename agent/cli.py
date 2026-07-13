from __future__ import annotations

from customer_agent_demo.agent.graph import CustomerAgent, new_thread_id


def main() -> None:
    agent = CustomerAgent()
    thread_id = new_thread_id()
    print("CGM 智能客服 Agent Demo")
    print("输入 exit / quit 结束。")
    print(f"thread_id={thread_id}")
    while True:
        user_message = input("\n用户> ").strip()
        if user_message.lower() in {"exit", "quit"}:
            print("再见。")
            return
        if not user_message:
            continue
        result = agent.invoke(user_message, thread_id=thread_id)
        print(f"\nAgent> {result.get('answer', '')}")
        trace = result.get("debug_trace") or {}
        if trace:
            print(
                "Debug> "
                f"strategy={trace.get('retrieval_strategy')}, "
                f"evidence={trace.get('evidence_status')}, "
                f"reason={trace.get('evidence_reason')}, "
                f"hits={len(trace.get('final_hits') or [])}"
            )


if __name__ == "__main__":
    main()
