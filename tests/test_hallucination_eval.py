from __future__ import annotations

from ..agent.evaluate_hallucination import (
    evaluate_case,
    evaluate_cases,
    load_cases,
    write_markdown_report,
)


def test_hallucination_eval_cases_have_required_shape() -> None:
    cases = load_cases()

    assert len(cases) == 10
    assert sum(1 for case in cases if not case.should_answer) >= 3
    assert all(case.id and case.question and case.risk_type for case in cases)


def test_eval_marks_out_of_kb_numeric_answer_as_hallucination() -> None:
    case = next(case for case in load_cases() if case.id == "cgm-009")

    row = evaluate_case(
        case,
        {
            "answer": "连接码是 6 位数。",
            "answer_status": "grounded",
            "retrieved_docs": [],
        },
    )

    assert row["hallucination"] is True
    assert row["risky_numbers"] == ["6 位"]


def test_eval_accepts_refusal_for_out_of_kb_question() -> None:
    case = next(case for case in load_cases() if case.id == "cgm-009")

    row = evaluate_case(
        case,
        {
            "answer": "我没有在当前知识库找到足够依据。为了避免误导，建议转人工继续处理。",
            "answer_status": "insufficient_evidence",
            "retrieved_docs": [],
        },
    )

    assert row["refused"] is True
    assert row["hallucination"] is False


def test_evaluate_cases_uses_injected_answer_function() -> None:
    cases = load_cases()[:2]

    rows = evaluate_cases(
        cases,
        answer_fn=lambda question: {
            "answer": f"{question}\n引用：\n[1] Dexcom G7 - https://example.com - chunk #0",
            "answer_status": "grounded",
            "retrieved_docs": [
                {"source_title": "Dexcom G7", "source_url": "https://example.com"}
            ],
        },
    )

    assert len(rows) == 2
    assert rows[0]["has_reference"] is True


def test_write_markdown_report(tmp_path) -> None:
    output_path = tmp_path / "report.md"
    rows = [
        {
            "id": "cgm-test",
            "question": "测试问题",
            "should_answer": False,
            "answer_status": "insufficient_evidence",
            "refused": True,
            "has_reference": False,
            "expected_source_hit": True,
            "risky_numbers": [],
            "hallucination": False,
        }
    ]

    write_markdown_report(rows, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "CGM Agent 幻觉评估结果" in text
    assert "cgm-test" in text
