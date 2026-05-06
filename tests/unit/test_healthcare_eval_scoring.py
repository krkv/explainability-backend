"""Tests for healthcare tool-calling scoring."""

from pathlib import Path

from evals.healthcare_tool_calling.scripts.eval_common import append_jsonl
from evals.healthcare_tool_calling.scripts.score_predictions import score_predictions


def test_score_predictions_reports_tool_call_metrics(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    append_jsonl(
        predictions,
        [
            {
                "case_id": "correct",
                "scenario": "direct_single_turn",
                "expected_behavior": "tool_call",
                "expected_function_calls": ["predict(patient_id=42)"],
                "predicted_function_calls": ["predict(patient_id=42)"],
                "response_valid": True,
                "error": None,
            },
            {
                "case_id": "wrong_arg",
                "scenario": "direct_single_turn",
                "expected_behavior": "tool_call",
                "expected_function_calls": ["predict(patient_id=42)"],
                "predicted_function_calls": ["predict(patient_id=43)"],
                "response_valid": True,
                "error": None,
            },
            {
                "case_id": "no_call",
                "scenario": "missing_required_argument",
                "expected_behavior": "no_call_clarify",
                "expected_function_calls": [],
                "target_tools": ["predict"],
                "predicted_function_calls": [],
                "response_valid": True,
                "error": None,
            },
        ],
    )

    scores = score_predictions(predictions)

    assert scores["cases"] == 3
    assert scores["metrics"]["response_schema_validity"] == 1
    assert scores["metrics"]["intent_accuracy"] == 1
    assert scores["metrics"]["joint_goal_accuracy"] == 2 / 3
    assert scores["metrics"]["argument_accuracy"] == 1 / 2
    assert scores["metrics"]["no_call_accuracy"] == 1


def test_score_predictions_counts_alias_equivalent_calls_as_correct(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    append_jsonl(
        predictions,
        [
            {
                "case_id": "feature_alias",
                "scenario": "paraphrase_or_alias",
                "expected_behavior": "tool_call",
                "expected_function_calls": ["define_feature(feature='chol')"],
                "predicted_function_calls": [
                    "define_feature(feature='Serum Cholesterol')"
                ],
                "response_valid": True,
                "error": None,
            },
            {
                "case_id": "category_alias",
                "scenario": "paraphrase_or_alias",
                "expected_behavior": "tool_call",
                "expected_function_calls": [
                    "what_if(patient_id=24, feature='cp', value_change='Typical angina')"
                ],
                "predicted_function_calls": [
                    "what_if(patient_id=24, feature='chest pain', value_change='1')"
                ],
                "response_valid": True,
                "error": None,
            },
        ],
    )

    scores = score_predictions(predictions)

    assert scores["metrics"]["intent_accuracy"] == 1
    assert scores["metrics"]["joint_goal_accuracy"] == 1
    assert scores["metrics"]["argument_accuracy"] == 1
