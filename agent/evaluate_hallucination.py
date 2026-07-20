from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..config import DEMO_ROOT


DEFAULT_CASES_PATH = DEMO_ROOT / "data" / "hallucination_eval_cases.json"
DEFAULT_RUNS_DIR = DEMO_ROOT / "data" / "runs"
HIGH_RISK_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:位|天|小时|分钟|米|feet|ft|级|%)?", re.IGNORECASE
)


@dataclass(slots=True)
class EvalCase:
    id: str
    question: str
    expected_sources: list[str]
    should_answer: bool
    risk_type: str
    expected_terms: list[str]


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=record["id"],
            question=record["question"],
            expected_sources=list(record.get("expected_sources") or []),
            should_answer=bool(record["should_answer"]),
            risk_type=record["risk_type"],
            expected_terms=list(record.get("expected_terms") or []),
        )
        for record in records
    ]


def evaluate_cases(
    cases: list[EvalCase],
    *,
    answer_fn: Callable[[str], dict],
) -> list[dict]:
    return [evaluate_case(case, answer_fn(case.question)) for case in cases]


def evaluate_case(case: EvalCase, result: dict) -> dict:
    answer = str(result.get("answer") or "")
    answer_status = str(result.get("answer_status") or "")
    retrieved_docs = list(result.get("retrieved_docs") or [])
    source_titles = [
        str(doc.get("source_title") or "")
        for doc in retrieved_docs
        if isinstance(doc, dict)
    ]
    source_text = "\n".join(source_titles)
    should_refuse = not case.should_answer

    refused = (
        answer_status == "insufficient_evidence"
        or "没有在当前知识库找到足够依据" in answer
        or "已为你转人工" in answer
    )
    has_reference = "引用：" in answer
    expected_source_hit = (
        not case.expected_sources
        or any(
            expected.lower() in source_text.lower()
            for expected in case.expected_sources
        )
        or any(expected.lower() in answer.lower() for expected in case.expected_sources)
    )
    expected_terms_hit = all(
        term.lower() in answer.lower() for term in case.expected_terms
    )
    risky_numbers = _risky_numbers(answer) if should_refuse else []
    hallucination = (
        (should_refuse and not refused)
        or (should_refuse and bool(risky_numbers))
        or (case.should_answer and not refused and not expected_source_hit)
    )

    return {
        "id": case.id,
        "question": case.question,
        "risk_type": case.risk_type,
        "should_answer": case.should_answer,
        "answer_status": answer_status,
        "refused": refused,
        "has_reference": has_reference,
        "expected_source_hit": expected_source_hit,
        "expected_terms_hit": expected_terms_hit,
        "risky_numbers": risky_numbers,
        "hallucination": hallucination,
        "source_titles": source_titles,
        "answer_excerpt": answer[:300],
        "debug_trace": result.get("debug_trace") or {},
    }


def write_markdown_report(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CGM Agent 幻觉评估结果",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "| ID | 问题 | 应回答 | 状态 | 拒答 | 引用 | 来源命中 | 高危数字 | 是否幻觉 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {question} | {should_answer} | {answer_status} | {refused} | {has_reference} | "
            "{expected_source_hit} | {risky_numbers} | {hallucination} |".format(
                **{
                    **row,
                    "question": str(row["question"]).replace("|", "\\|"),
                    "risky_numbers": ", ".join(row["risky_numbers"])
                    if row["risky_numbers"]
                    else "",
                }
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_live_eval(cases: list[EvalCase]) -> list[dict]:
    from ..graph import CustomerAgent, new_thread_id

    agent = CustomerAgent()

    def answer_fn(question: str) -> dict:
        thread_id = new_thread_id()
        result = agent.invoke(question, thread_id=thread_id)
        return _state_to_result_dict(result)

    return evaluate_cases(cases, answer_fn=answer_fn)


def _state_to_result_dict(state: dict) -> dict:
    docs = state.get("retrieved_docs") or []
    return {
        "answer": state.get("answer") or "",
        "answer_status": state.get("answer_status") or "",
        "retrieved_docs": [
            doc.model_dump(mode="json") if hasattr(doc, "model_dump") else doc
            for doc in docs
        ],
        "debug_trace": state.get("debug_trace") or {},
    }


def _risky_numbers(answer: str) -> list[str]:
    return [
        match.group(0).strip() for match in HIGH_RISK_NUMBER_PATTERN.finditer(answer)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CGM Agent hallucination evaluation."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    rows = run_live_eval(cases)
    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_RUNS_DIR
        / f"hallucination_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    write_markdown_report(rows, output_path)
    print(f"幻觉评估完成：{output_path}")


if __name__ == "__main__":
    main()
