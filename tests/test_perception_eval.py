from __future__ import annotations

from ..agent.evaluate_perception import evaluate_cases, load_cases
from ..agent.perception import heuristic_perception


def test_perception_eval_dataset_meets_minimum_coverage() -> None:
    cases = load_cases()

    assert len(cases) >= 35
    assert {case["expected_intent"] for case in cases} == {
        "产品咨询",
        "使用问题",
        "售后诉求",
        "闲聊",
    }
    assert any(case.get("pending_clarification") for case in cases)


def test_heuristic_perception_passes_curated_eval_baseline() -> None:
    report = evaluate_cases(heuristic_perception, load_cases())

    assert report["intent_accuracy"] >= 0.90
    assert report["intent_macro_f1"] >= 0.90
    assert report["route_accuracy"] >= 0.95
    assert report["handoff_recall"] == 1.0
    assert report["clarification_precision"] >= 0.85
    assert report["unnecessary_clarification_rate"] <= 0.10
