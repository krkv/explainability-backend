"""Tests for healthcare tool-calling eval helpers."""

from unittest.mock import patch

import pytest

from evals.healthcare_tool_calling.scripts.eval_common import (
    build_live_conversation,
    calls_exact_match,
    load_function_catalog,
    parse_calls,
    read_jsonl,
    read_model_response,
    resolve_eval_provider,
    write_jsonl,
)
from evals.healthcare_tool_calling.scripts.run_eval import (
    _remove_case_ids_from_jsonl,
    _remove_stale_score_artifacts,
    _retryable_provider_error_case_ids,
)
from src.core.constants import Model


def test_build_live_conversation_includes_function_call_history():
    case = {
        "user_input": "What about that patient?",
        "conversation_history": [
            {
                "turn": 1,
                "user_input": "Show patient 42",
                "function_calls": ["show_one(patient_id=42)"],
            },
            {
                "turn": 2,
                "user_input": "thanks",
                "function_calls": [],
            },
        ],
    }

    assert build_live_conversation(case) == [
        {"role": "user", "content": "Show patient 42"},
        {
            "role": "assistant",
            "content": "<code>show_one(patient_id=42)</code>",
            "is_function_call": True,
        },
        {"role": "user", "content": "thanks"},
        {"role": "user", "content": "What about that patient?"},
    ]


def test_read_model_response_validates_required_shape():
    result = read_model_response(
        '{"function_calls":["predict(patient_id=42)"],"freeform_response":"I will predict patient 42."}'
    )

    assert result.response_valid is True
    assert result.predicted_function_calls == ["predict(patient_id=42)"]
    assert result.predicted_freeform_response == "I will predict patient 42."


def test_read_model_response_rejects_malformed_json():
    result = read_model_response("not json")

    assert result.response_valid is False
    assert result.predicted_function_calls == []
    assert result.error.startswith("invalid_json")


def test_call_exact_match_compares_tool_and_arguments():
    catalog = load_function_catalog()
    expected, expected_errors = parse_calls(["predict(patient_id=42)"], catalog)
    predicted, predicted_errors = parse_calls(["predict(patient_id=42)"], catalog)
    wrong, wrong_errors = parse_calls(["predict(patient_id=43)"], catalog)

    assert expected_errors == []
    assert predicted_errors == []
    assert wrong_errors == []
    assert calls_exact_match(expected, predicted) is True
    assert calls_exact_match(expected, wrong) is False


def test_call_match_canonicalizes_feature_aliases():
    catalog = load_function_catalog()
    expected, expected_errors = parse_calls(
        ["define_feature(feature='chol')"],
        catalog,
    )
    predicted, predicted_errors = parse_calls(
        ["define_feature(feature='Serum Cholesterol')"],
        catalog,
    )

    assert expected_errors == []
    assert predicted_errors == []
    assert calls_exact_match(expected, predicted) is True
    assert expected[0].kwargs == {"feature": "chol"}
    assert predicted[0].kwargs == {"feature": "chol"}


def test_call_match_canonicalizes_what_if_feature_aliases():
    catalog = load_function_catalog()
    expected, expected_errors = parse_calls(
        ["what_if(patient_id=4, feature='trestbps', value_change=20)"],
        catalog,
    )
    predicted, predicted_errors = parse_calls(
        ["what_if(patient_id=4, feature='blood pressure', value_change=20)"],
        catalog,
    )

    assert expected_errors == []
    assert predicted_errors == []
    assert calls_exact_match(expected, predicted) is True


def test_call_match_canonicalizes_categorical_what_if_values():
    catalog = load_function_catalog()
    expected, expected_errors = parse_calls(
        ["what_if(patient_id=24, feature='cp', value_change='Typical angina')"],
        catalog,
    )
    predicted, predicted_errors = parse_calls(
        ["what_if(patient_id=24, feature='chest pain', value_change='1')"],
        catalog,
    )

    assert expected_errors == []
    assert predicted_errors == []
    assert calls_exact_match(expected, predicted) is True
    assert expected[0].kwargs["value_change"] == 1
    assert predicted[0].kwargs["value_change"] == 1


def test_call_match_canonicalizes_performance_metric_aliases():
    catalog = load_function_catalog()
    expected, expected_errors = parse_calls(
        ["performance_metrics(metrics=['f1_score', 'auc_roc'])"],
        catalog,
    )
    predicted, predicted_errors = parse_calls(
        ["performance_metrics(metrics=['f1-score', 'auc'])"],
        catalog,
    )

    assert expected_errors == []
    assert predicted_errors == []
    assert calls_exact_match(expected, predicted) is True
    assert predicted[0].kwargs == {"metrics": ["auc_roc", "f1_score"]}


def test_call_match_canonicalizes_performance_metric_defaults():
    catalog = load_function_catalog()
    expected, expected_errors = parse_calls(
        ["performance_metrics()"],
        catalog,
    )
    predicted, predicted_errors = parse_calls(
        [
            "performance_metrics(metrics=['recall', 'accuracy', 'auc', 'precision', 'f1'])"
        ],
        catalog,
    )

    assert expected_errors == []
    assert predicted_errors == []
    assert calls_exact_match(expected, predicted) is True


def test_call_match_canonicalizes_count_patient_aliases():
    catalog = load_function_catalog()
    expected, expected_errors = parse_calls(
        ["count_patients(count_type='positive_predicted')"],
        catalog,
    )
    predicted, predicted_errors = parse_calls(
        ["count_patients(count_type='with heart disease')"],
        catalog,
    )

    assert expected_errors == []
    assert predicted_errors == []
    assert calls_exact_match(expected, predicted) is True
    assert predicted[0].kwargs == {"count_type": "positive_predicted"}


def test_call_match_canonicalizes_count_patient_default():
    catalog = load_function_catalog()
    expected, expected_errors = parse_calls(
        ["count_patients()"],
        catalog,
    )
    predicted, predicted_errors = parse_calls(
        ["count_patients(count_type='total')"],
        catalog,
    )

    assert expected_errors == []
    assert predicted_errors == []
    assert calls_exact_match(expected, predicted) is True


def test_call_parse_rejects_unknown_feature_alias():
    catalog = load_function_catalog()
    parsed, errors = parse_calls(
        ["define_feature(feature='unsupported feature')"],
        catalog,
    )

    assert parsed == []
    assert errors


def test_resolve_eval_provider_supports_backend_model_enum():
    config = {
        "provider": "backend_model_enum",
        "model_enum": "gpt-5.4-mini",
        "status": "runnable",
    }

    class FakeProvider:
        def get_model_name(self):
            return "gpt-5.4-mini"

    with patch(
        "src.services.llm.llm_factory.get_llm_provider",
        return_value=FakeProvider(),
    ) as mock_get_provider:
        resolution = resolve_eval_provider("gpt-5.4-mini", config)

    mock_get_provider.assert_called_once_with(Model.GPT_5_4_MINI)
    assert resolution.provider_model_id == "gpt-5.4-mini"


def test_resolve_eval_provider_supports_openrouter_models():
    config = {
        "provider": "openrouter",
        "openrouter_model": "google/gemma-4-31b-it",
        "status": "runnable",
    }

    class FakeProvider:
        def get_model_name(self):
            return "google/gemma-4-31b-it"

    with patch(
        "src.services.llm.llm_factory.get_openrouter_provider",
        return_value=FakeProvider(),
    ) as mock_get_provider:
        resolution = resolve_eval_provider("gemma-4", config)

    mock_get_provider.assert_called_once_with("google/gemma-4-31b-it")
    assert resolution.provider_model_id == "google/gemma-4-31b-it"


def test_resolve_eval_provider_rejects_pending_models():
    config = {
        "provider": "openrouter",
        "openrouter_model": "google/gemma-4-31b-it",
        "status": "pending",
    }

    with pytest.raises(ValueError, match="not runnable"):
        resolve_eval_provider("gemma-4", config)


def test_resolve_eval_provider_rejects_malformed_openrouter_config():
    config = {
        "provider": "openrouter",
        "status": "runnable",
    }

    with pytest.raises(ValueError, match="openrouter_model"):
        resolve_eval_provider("gemma-4", config)


def test_retryable_provider_error_case_ids_excludes_model_format_errors(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    raw_generations_path = tmp_path / "raw_generations.jsonl"
    write_jsonl(
        predictions_path,
        [
            {"case_id": "ok", "response_valid": True, "error": None},
            {
                "case_id": "provider_error",
                "response_valid": False,
                "error": "UpstreamRateLimitException: rate limit",
            },
            {
                "case_id": "empty_provider_response",
                "response_valid": False,
                "error": "LLMProviderException: Empty response content",
            },
            {
                "case_id": "model_format_error",
                "response_valid": False,
                "error": "invalid_json: Expecting value",
            },
            {"case_id": "model_mismatch", "response_valid": True, "error": None},
        ],
    )
    write_jsonl(
        raw_generations_path,
        [
            {"case_id": "ok", "provider_error": None},
            {"case_id": "provider_error", "provider_error": "UpstreamRateLimitException"},
            {
                "case_id": "empty_provider_response",
                "provider_error": "LLMProviderException",
            },
            {"case_id": "model_format_error", "provider_error": None},
            {"case_id": "model_mismatch", "provider_error": None},
        ],
    )

    assert _retryable_provider_error_case_ids(
        predictions_path,
        raw_generations_path,
    ) == {
        "provider_error",
        "empty_provider_response",
    }


def test_remove_case_ids_from_jsonl_preserves_other_rows(tmp_path):
    path = tmp_path / "raw_generations.jsonl"
    write_jsonl(
        path,
        [
            {"case_id": "keep_1", "raw_response": "a"},
            {"case_id": "remove", "raw_response": ""},
            {"case_id": "keep_2", "raw_response": "b"},
        ],
    )

    _remove_case_ids_from_jsonl(path, {"remove"})

    assert [row["case_id"] for row in read_jsonl(path)] == ["keep_1", "keep_2"]


def test_remove_stale_score_artifacts_removes_scores_and_errors(tmp_path):
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    (output_dir / "scores.json").write_text("{}", encoding="utf-8")
    (output_dir / "errors.jsonl").write_text("", encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text("", encoding="utf-8")

    _remove_stale_score_artifacts(output_dir)

    assert not (output_dir / "scores.json").exists()
    assert not (output_dir / "errors.jsonl").exists()
    assert (output_dir / "predictions.jsonl").exists()
