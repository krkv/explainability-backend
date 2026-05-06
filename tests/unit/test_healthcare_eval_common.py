"""Tests for healthcare tool-calling eval helpers."""

from evals.healthcare_tool_calling.scripts.eval_common import (
    build_live_conversation,
    calls_exact_match,
    load_function_catalog,
    parse_calls,
    read_model_response,
)


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
