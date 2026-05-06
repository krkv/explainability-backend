#!/usr/bin/env python3
"""Shared helpers for healthcare tool-calling eval runner and scorer."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.constants import Model  # noqa: E402
from src.core.exceptions import FunctionExecutionException  # noqa: E402
from src.services.parser.ast_parser import ASTParser  # noqa: E402


EVAL_ROOT = REPO_ROOT / "evals" / "healthcare_tool_calling"
DEFAULT_MODEL_CONFIGS = EVAL_ROOT / "configs" / "model_configs.json"
DEFAULT_CATALOG = REPO_ROOT / "instances" / "heart" / "functions.json"
DEFAULT_FEATURE_METADATA = REPO_ROOT / "instances" / "heart" / "data" / "feature_metadata.json"

METRIC_ALIASES = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1_score",
    "f1_score": "f1_score",
    "f1-score": "f1_score",
    "auc": "auc_roc",
    "auc_roc": "auc_roc",
    "auc-roc": "auc_roc",
}

COUNT_TYPE_ALIASES = {
    "all": "total",
    "dataset": "total",
    "total": "total",
    "total_patients": "total",
    "positive": "positive_predicted",
    "positive_predicted": "positive_predicted",
    "predicted_positive": "positive_predicted",
    "heart_disease": "positive_predicted",
    "with_heart_disease": "positive_predicted",
    "negative": "negative_predicted",
    "negative_predicted": "negative_predicted",
    "predicted_negative": "negative_predicted",
    "without_heart_disease": "negative_predicted",
    "no_heart_disease": "negative_predicted",
}

ALL_METRICS = ["accuracy", "auc_roc", "f1_score", "precision", "recall"]


@dataclass(frozen=True)
class ParsedCall:
    raw: str
    name: str
    kwargs: Dict[str, Any]


@dataclass(frozen=True)
class ResponseReadResult:
    raw_response: str
    response_valid: bool
    predicted_function_calls: List[str]
    predicted_freeform_response: str
    error: Optional[str] = None


@dataclass(frozen=True)
class EvalProviderResolution:
    provider: Any
    provider_model_id: str


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_model_configs(path: Path = DEFAULT_MODEL_CONFIGS) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Model config file must be a JSON object keyed by model id")
    return payload


def load_function_catalog(path: Path = DEFAULT_CATALOG) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw_catalog = json.load(f)

    catalog: Dict[str, Dict[str, Any]] = {}
    for item in raw_catalog:
        function = item.get("function", {})
        name = function.get("name")
        if isinstance(name, str):
            catalog[name] = function
    return catalog


def load_feature_metadata(path: Path = DEFAULT_FEATURE_METADATA) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Feature metadata must be a JSON object")
    return payload


def build_feature_alias_lookup(
    feature_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for feature_name, metadata in feature_metadata.items():
        candidates = [feature_name, metadata.get("display_name", feature_name)]
        aliases = metadata.get("aliases", [])
        if isinstance(aliases, list):
            candidates.extend(aliases)
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                lookup[candidate.strip().lower()] = feature_name
    return lookup


def resolve_backend_model(model_id: str, config: Mapping[str, Any]) -> Model:
    if config.get("status") != "runnable":
        notes = config.get("notes") or "Provider access is not configured yet."
        raise ValueError(f"Model '{model_id}' is not runnable: {notes}")
    if config.get("provider") != "backend_model_enum":
        raise ValueError(f"Model '{model_id}' provider is not supported by this runner")
    return Model(str(config.get("model_enum") or model_id))


def resolve_eval_provider(
    model_id: str,
    config: Mapping[str, Any],
) -> EvalProviderResolution:
    if config.get("status") != "runnable":
        notes = config.get("notes") or "Provider access is not configured yet."
        raise ValueError(f"Model '{model_id}' is not runnable: {notes}")

    provider_type = config.get("provider")
    if provider_type == "backend_model_enum":
        from src.services.llm.llm_factory import get_llm_provider

        backend_model = resolve_backend_model(model_id, config)
        provider = get_llm_provider(backend_model)
        return EvalProviderResolution(
            provider=provider,
            provider_model_id=provider.get_model_name(),
        )

    if provider_type == "openrouter":
        from src.services.llm.llm_factory import get_openrouter_provider

        openrouter_model = config.get("openrouter_model") or config.get("model")
        if not isinstance(openrouter_model, str) or not openrouter_model.strip():
            raise ValueError(
                f"Model '{model_id}' OpenRouter config must include openrouter_model"
            )
        provider = get_openrouter_provider(openrouter_model)
        return EvalProviderResolution(
            provider=provider,
            provider_model_id=provider.get_model_name(),
        )

    raise ValueError(f"Model '{model_id}' provider is not supported by this runner")


def build_live_conversation(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    conversation: List[Dict[str, Any]] = []

    for turn in case.get("conversation_history", []):
        conversation.append(
            {
                "role": "user",
                "content": str(turn.get("user_input", "")),
            }
        )
        for function_call in turn.get("function_calls", []):
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"<code>{function_call}</code>",
                    "is_function_call": True,
                }
            )

    conversation.append({"role": "user", "content": str(case.get("user_input", ""))})
    return conversation


def read_model_response(raw_response: str) -> ResponseReadResult:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as e:
        return ResponseReadResult(
            raw_response=raw_response,
            response_valid=False,
            predicted_function_calls=[],
            predicted_freeform_response="",
            error=f"invalid_json: {e}",
        )

    if not isinstance(payload, dict):
        return _invalid_response(raw_response, "response_json_must_be_object")
    if "function_calls" not in payload:
        return _invalid_response(raw_response, "missing_function_calls")
    if "freeform_response" not in payload:
        return _invalid_response(raw_response, "missing_freeform_response")
    if not isinstance(payload["function_calls"], list):
        return _invalid_response(raw_response, "function_calls_must_be_list")
    if not all(isinstance(call, str) for call in payload["function_calls"]):
        return _invalid_response(raw_response, "function_calls_items_must_be_strings")
    if not isinstance(payload["freeform_response"], str):
        return _invalid_response(raw_response, "freeform_response_must_be_string")

    return ResponseReadResult(
        raw_response=raw_response,
        response_valid=True,
        predicted_function_calls=payload["function_calls"],
        predicted_freeform_response=payload["freeform_response"],
    )


def _invalid_response(raw_response: str, error: str) -> ResponseReadResult:
    return ResponseReadResult(
        raw_response=raw_response,
        response_valid=False,
        predicted_function_calls=[],
        predicted_freeform_response="",
        error=error,
    )


def parse_call(
    call: str,
    function_catalog: Mapping[str, Dict[str, Any]],
    feature_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ParsedCall:
    name, kwargs = ASTParser.parse_function_call(call)
    if name not in function_catalog:
        raise FunctionExecutionException(f"Unknown function '{name}'")

    properties = (function_catalog[name].get("parameters") or {}).get("properties") or {}
    unknown_args = set(kwargs) - set(properties)
    if unknown_args:
        raise FunctionExecutionException(
            f"Unknown args for '{name}': {sorted(unknown_args)}"
        )

    required = (function_catalog[name].get("parameters") or {}).get("required") or []
    missing_args = set(required) - set(kwargs)
    if missing_args:
        raise FunctionExecutionException(
            f"Missing required args for '{name}': {sorted(missing_args)}"
        )

    canonical_kwargs = canonicalize_kwargs(name, kwargs, feature_metadata)
    return ParsedCall(raw=call, name=name, kwargs=canonical_kwargs)


def parse_calls(
    calls: Sequence[str],
    function_catalog: Mapping[str, Dict[str, Any]],
    feature_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[List[ParsedCall], List[str]]:
    parsed_calls: List[ParsedCall] = []
    errors: List[str] = []
    for call in calls:
        try:
            parsed_calls.append(parse_call(call, function_catalog, feature_metadata))
        except FunctionExecutionException as e:
            errors.append(f"{call}: {e}")
    return parsed_calls, errors


def canonicalize_kwargs(
    function_name: str,
    kwargs: Mapping[str, Any],
    feature_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    canonical = dict(kwargs)
    feature_metadata = feature_metadata or load_feature_metadata()

    if function_name == "define_feature" and "feature" in canonical:
        canonical["feature"] = canonicalize_feature(canonical["feature"], feature_metadata)

    if function_name == "what_if":
        feature_key = None
        if "feature" in canonical:
            feature_key = canonicalize_feature(canonical["feature"], feature_metadata)
            canonical["feature"] = feature_key
        if feature_key is not None and "value_change" in canonical:
            canonical["value_change"] = canonicalize_what_if_value_change(
                feature_key,
                canonical["value_change"],
                feature_metadata,
            )

    if function_name == "performance_metrics":
        if "metrics" not in canonical:
            canonical["metrics"] = ALL_METRICS
        else:
            metrics = canonical["metrics"]
            if not isinstance(metrics, list):
                raise FunctionExecutionException("metrics must be a list")
            canonical["metrics"] = sorted(
                set(canonicalize_metric(metric) for metric in metrics)
            )

    if function_name == "count_patients":
        canonical["count_type"] = canonicalize_count_type(
            canonical.get("count_type", "total")
        )

    return canonical


def canonicalize_feature(
    feature: Any,
    feature_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    if not isinstance(feature, str):
        raise FunctionExecutionException("feature must be a string")
    normalized = feature.strip().lower()
    alias_lookup = build_feature_alias_lookup(feature_metadata)
    if normalized in alias_lookup:
        return alias_lookup[normalized]
    if normalized in feature_metadata:
        return normalized
    raise FunctionExecutionException(f"Unknown feature alias '{feature}'")


def canonicalize_metric(metric: Any) -> str:
    if not isinstance(metric, str):
        raise FunctionExecutionException("metric names must be strings")
    normalized = metric.strip().lower().replace("-", "_")
    canonical = METRIC_ALIASES.get(normalized)
    if canonical is None:
        raise FunctionExecutionException(f"Unknown metric alias '{metric}'")
    return canonical


def canonicalize_count_type(count_type: Any) -> str:
    normalized = str(count_type or "total").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    canonical = COUNT_TYPE_ALIASES.get(normalized)
    if canonical is None:
        raise FunctionExecutionException(f"Unknown count_type alias '{count_type}'")
    return canonical


def canonicalize_what_if_value_change(
    feature: str,
    value_change: Any,
    feature_metadata: Mapping[str, Mapping[str, Any]],
) -> Any:
    metadata = feature_metadata.get(feature, {})
    if metadata.get("kind", "continuous") != "categorical":
        return value_change
    if isinstance(value_change, (int, float)) and not isinstance(value_change, bool):
        return value_change

    resolved = resolve_category_value(feature, value_change, feature_metadata)
    if resolved is None:
        raise FunctionExecutionException(
            f"Unknown categorical value '{value_change}' for feature '{feature}'"
        )
    return resolved


def resolve_category_value(
    feature: str,
    value: Any,
    feature_metadata: Mapping[str, Mapping[str, Any]],
) -> Optional[Any]:
    categories = feature_metadata.get(feature, {}).get("categories", {})
    normalized_value = normalize_lookup_value(value)
    if not normalized_value:
        return None
    value_tokens = normalized_value.split()

    exact_matches: List[str] = []
    partial_matches: List[str] = []
    for code, details in sorted_categories(categories):
        label = details.get("label", code) if isinstance(details, dict) else code
        normalized_code = normalize_lookup_value(code)
        normalized_label = normalize_lookup_value(label)
        if normalized_value in {normalized_code, normalized_label}:
            exact_matches.append(str(code))
            continue
        label_tokens = normalized_label.split()
        if value_tokens and all(token in label_tokens for token in value_tokens):
            partial_matches.append(str(code))

    if len(exact_matches) == 1:
        return coerce_category_code(exact_matches[0])
    if len(partial_matches) == 1:
        return coerce_category_code(partial_matches[0])
    return None


def normalize_lookup_value(value: Any) -> str:
    normalized = str(value).strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def coerce_category_code(code: str) -> Any:
    try:
        numeric = float(code)
    except (TypeError, ValueError):
        return code
    if numeric.is_integer():
        return int(numeric)
    return numeric


def sorted_categories(categories: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    return sorted(categories.items(), key=lambda item: category_sort_key(str(item[0])))


def category_sort_key(code: str) -> Tuple[int, Any]:
    try:
        return (0, float(code))
    except (TypeError, ValueError):
        return (1, code)


def calls_exact_match(expected: Sequence[ParsedCall], predicted: Sequence[ParsedCall]) -> bool:
    if len(expected) != len(predicted):
        return False
    return all(
        expected_call.name == predicted_call.name
        and expected_call.kwargs == predicted_call.kwargs
        for expected_call, predicted_call in zip(expected, predicted)
    )


def tool_names(calls: Sequence[ParsedCall]) -> List[str]:
    return [call.name for call in calls]
