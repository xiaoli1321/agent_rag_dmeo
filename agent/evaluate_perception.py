from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .models import PendingClarification, PerceptionResult
from .perception import PerceptionService
from ..config import DEMO_ROOT, get_settings


CASES_PATH = DEMO_ROOT / "data" / "perception_eval_cases.json"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_cases(
    classify: Callable[..., PerceptionResult],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    intent_correct = 0
    handoff_expected = 0
    handoff_correct = 0
    clarification_tp = 0
    clarification_fp = 0
    non_clarification_cases = 0
    route_correct = 0
    labels = {case["expected_intent"] for case in cases}
    per_label = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}

    for case in cases:
        pending_data = case.get("pending_clarification")
        pending = (
            PendingClarification.model_validate(pending_data) if pending_data else None
        )
        result = classify(
            case["input"],
            current_topic=case.get("current_topic"),
            pending_clarification=pending,
        )
        expected_clarification = case["expected_actionability"] == "needs_clarification"
        predicted_clarification = result.actionability == "needs_clarification"
        intent_ok = result.intent == case["expected_intent"]
        actionability_ok = result.actionability == case["expected_actionability"]
        intent_correct += int(intent_ok)
        for label in labels:
            per_label[label]["tp"] += int(
                result.intent == label and case["expected_intent"] == label
            )
            per_label[label]["fp"] += int(
                result.intent == label and case["expected_intent"] != label
            )
            per_label[label]["fn"] += int(
                result.intent != label and case["expected_intent"] == label
            )
        expected_route = case.get("expected_route") or _expected_route(case)
        route_correct += int(_route_for_result(result) == expected_route)
        if case.get("expected_handoff"):
            handoff_expected += 1
            handoff_correct += int(
                result.handoff_requested or result.intent == "售后诉求"
            )
        if expected_clarification and predicted_clarification:
            clarification_tp += 1
        if not expected_clarification:
            non_clarification_cases += 1
            clarification_fp += int(predicted_clarification)
        rows.append(
            {
                "id": case["id"],
                "intent_ok": intent_ok,
                "actionability_ok": actionability_ok,
                "actual_intent": result.intent,
                "actual_actionability": result.actionability,
            }
        )

    total = len(cases)
    predicted_clarifications = clarification_tp + clarification_fp
    f1_scores = []
    for counts in per_label.values():
        precision = (
            counts["tp"] / (counts["tp"] + counts["fp"])
            if counts["tp"] + counts["fp"]
            else 0.0
        )
        recall = (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if counts["tp"] + counts["fn"]
            else 0.0
        )
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "total": total,
        "intent_accuracy": intent_correct / total if total else 0.0,
        "intent_macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "route_accuracy": route_correct / total if total else 0.0,
        "handoff_recall": handoff_correct / handoff_expected
        if handoff_expected
        else 1.0,
        "clarification_precision": (
            clarification_tp / predicted_clarifications
            if predicted_clarifications
            else 1.0
        ),
        "unnecessary_clarification_rate": (
            clarification_fp / non_clarification_cases
            if non_clarification_cases
            else 0.0
        ),
        "passed": sum(row["intent_ok"] and row["actionability_ok"] for row in rows),
        "rows": rows,
    }


def _expected_route(case: dict[str, Any]) -> str:
    if case["expected_actionability"] == "needs_clarification":
        return "clarify"
    if case["expected_intent"] == "售后诉求":
        return "after_sales"
    if case["expected_intent"] == "闲聊":
        return "smalltalk"
    return "product_consultant"


def _route_for_result(result: PerceptionResult) -> str:
    if result.actionability == "needs_clarification":
        return "clarify"
    if result.intent == "售后诉求" or result.handoff_requested:
        return "after_sales"
    if result.intent == "闲聊" or result.actionability == "unsupported":
        return "smalltalk"
    return "product_consultant"


def main() -> None:
    service = PerceptionService(settings=get_settings(), temperature=0.0)
    report = evaluate_cases(service.classify, load_cases())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
