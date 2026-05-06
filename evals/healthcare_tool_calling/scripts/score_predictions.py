#!/usr/bin/env python3
"""Score healthcare tool-calling predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from eval_common import (
        append_jsonl,
        calls_exact_match,
        load_feature_metadata,
        load_function_catalog,
        parse_calls,
        read_jsonl,
        tool_names,
        write_json,
    )
except ModuleNotFoundError:
    from evals.healthcare_tool_calling.scripts.eval_common import (
        append_jsonl,
        calls_exact_match,
        load_feature_metadata,
        load_function_catalog,
        parse_calls,
        read_jsonl,
        tool_names,
        write_json,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a healthcare tool-calling predictions JSONL file."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=None)
    return parser.parse_args(argv)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def score_predictions(predictions_path: Path, catalog_path: Optional[Path] = None) -> Dict[str, Any]:
    function_catalog = load_function_catalog(catalog_path) if catalog_path else load_function_catalog()
    feature_metadata = load_feature_metadata()
    rows = read_jsonl(predictions_path)

    totals = Counter()
    scenario_totals: Dict[str, Counter[str]] = defaultdict(Counter)
    tool_totals: Dict[str, Counter[str]] = defaultdict(Counter)
    errors: List[Dict[str, Any]] = []

    for row in rows:
        totals["cases"] += 1
        scenario = str(row.get("scenario", "unknown"))
        expected_behavior = str(row.get("expected_behavior", ""))
        response_valid = bool(row.get("response_valid"))
        predicted_calls_raw = row.get("predicted_function_calls") or []
        expected_calls_raw = row.get("expected_function_calls") or []

        expected_calls, expected_parse_errors = parse_calls(
            expected_calls_raw,
            function_catalog,
            feature_metadata,
        )
        predicted_calls, predicted_parse_errors = parse_calls(
            predicted_calls_raw,
            function_catalog,
            feature_metadata,
        )

        expected_tools = tool_names(expected_calls)
        predicted_tools = tool_names(predicted_calls)
        no_call_expected = len(expected_calls_raw) == 0
        no_call_predicted = len(predicted_calls_raw) == 0
        intent_correct = expected_tools == predicted_tools
        joint_goal_correct = (
            response_valid
            and not expected_parse_errors
            and not predicted_parse_errors
            and calls_exact_match(expected_calls, predicted_calls)
        )

        totals["response_valid"] += int(response_valid)
        totals["predicted_calls_parse_valid"] += int(response_valid and not predicted_parse_errors)
        totals["intent_correct"] += int(response_valid and intent_correct)
        totals["joint_goal_correct"] += int(joint_goal_correct)
        totals["no_call_cases"] += int(no_call_expected)
        totals["no_call_correct"] += int(response_valid and no_call_expected and no_call_predicted)
        totals["overcall"] += int(no_call_expected and not no_call_predicted)
        totals["undercall"] += int(not no_call_expected and no_call_predicted)
        totals["hallucinated_tool"] += int(bool(predicted_parse_errors))

        if expected_calls:
            expected_arg_slots = sum(len(call.kwargs) for call in expected_calls)
            matching_arg_slots = 0
            for expected_call, predicted_call in zip(expected_calls, predicted_calls):
                if expected_call.name != predicted_call.name:
                    continue
                for arg_name, expected_value in expected_call.kwargs.items():
                    if predicted_call.kwargs.get(arg_name) == expected_value:
                        matching_arg_slots += 1
            totals["argument_slots"] += expected_arg_slots
            totals["argument_slots_correct"] += matching_arg_slots

        scenario_totals[scenario]["cases"] += 1
        scenario_totals[scenario]["response_valid"] += int(response_valid)
        scenario_totals[scenario]["intent_correct"] += int(response_valid and intent_correct)
        scenario_totals[scenario]["joint_goal_correct"] += int(joint_goal_correct)
        scenario_totals[scenario]["overcall"] += int(no_call_expected and not no_call_predicted)
        scenario_totals[scenario]["undercall"] += int(not no_call_expected and no_call_predicted)

        for tool_name in set(expected_tools) or set(row.get("target_tools", [])):
            tool_totals[tool_name]["cases"] += 1
            tool_totals[tool_name]["response_valid"] += int(response_valid)
            tool_totals[tool_name]["intent_correct"] += int(response_valid and intent_correct)
            tool_totals[tool_name]["joint_goal_correct"] += int(joint_goal_correct)
            tool_totals[tool_name]["overcall"] += int(no_call_expected and not no_call_predicted)
            tool_totals[tool_name]["undercall"] += int(not no_call_expected and no_call_predicted)

        if not joint_goal_correct:
            errors.append(
                {
                    "case_id": row.get("case_id"),
                    "scenario": scenario,
                    "expected_behavior": expected_behavior,
                    "expected_function_calls": expected_calls_raw,
                    "predicted_function_calls": predicted_calls_raw,
                    "response_valid": response_valid,
                    "error": row.get("error"),
                    "expected_parse_errors": expected_parse_errors,
                    "predicted_parse_errors": predicted_parse_errors,
                }
            )

    scores = {
        "predictions": str(predictions_path),
        "cases": totals["cases"],
        "metrics": {
            "response_schema_validity": _rate(totals["response_valid"], totals["cases"]),
            "function_call_string_validity": _rate(
                totals["predicted_calls_parse_valid"],
                totals["cases"],
            ),
            "intent_accuracy": _rate(totals["intent_correct"], totals["cases"]),
            "argument_accuracy": _rate(
                totals["argument_slots_correct"],
                totals["argument_slots"],
            ),
            "joint_goal_accuracy": _rate(totals["joint_goal_correct"], totals["cases"]),
            "no_call_accuracy": _rate(totals["no_call_correct"], totals["no_call_cases"]),
            "overcall_rate": _rate(totals["overcall"], totals["cases"]),
            "undercall_rate": _rate(totals["undercall"], totals["cases"]),
            "hallucinated_tool_rate": _rate(totals["hallucinated_tool"], totals["cases"]),
        },
        "counts": dict(totals),
        "by_scenario": _summarize_breakdowns(scenario_totals),
        "by_tool": _summarize_breakdowns(tool_totals),
    }

    scores_path = predictions_path.parent / "scores.json"
    errors_path = predictions_path.parent / "errors.jsonl"
    write_json(scores_path, scores)
    if errors_path.exists():
        errors_path.unlink()
    append_jsonl(errors_path, errors)
    return scores


def _summarize_breakdowns(breakdowns: Dict[str, Counter[str]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for name, counts in sorted(breakdowns.items()):
        cases = counts["cases"]
        summary[name] = {
            "cases": cases,
            "response_schema_validity": _rate(counts["response_valid"], cases),
            "intent_accuracy": _rate(counts["intent_correct"], cases),
            "joint_goal_accuracy": _rate(counts["joint_goal_correct"], cases),
            "overcall_rate": _rate(counts["overcall"], cases),
            "undercall_rate": _rate(counts["undercall"], cases),
        }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scores = score_predictions(args.predictions, args.catalog)
    print(f"Wrote scores to {args.predictions.parent / 'scores.json'}")
    print(f"Joint goal accuracy: {scores['metrics']['joint_goal_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
