#!/usr/bin/env python3
"""Check whether the manual seed-gold dataset is ready for teacher enrichment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.exceptions import FunctionExecutionException  # noqa: E402
from src.services.parser.ast_parser import ASTParser  # noqa: E402


EVAL_ROOT = REPO_ROOT / "evals" / "healthcare_tool_calling"
DEFAULT_DATASET = EVAL_ROOT / "datasets" / "seed_gold.jsonl"
DEFAULT_CATALOG = REPO_ROOT / "instances" / "heart" / "functions.json"

SCENARIOS: Set[str] = {
    "direct_single_turn",
    "paraphrase_or_alias",
    "parameter_carryover",
    "entity_switch_or_correction",
    "missing_required_argument",
    "unsupported_intent",
    "conflicting_context",
    "multi_tool_request",
    "no_tool_needed",
}

NO_CALL_BEHAVIORS: Set[str] = {
    "no_call_clarify",
    "no_call_unsupported",
    "no_call_needed",
}

EXPECTED_BEHAVIORS: Set[str] = {"tool_call"} | NO_CALL_BEHAVIORS

REQUIRED_CASE_FIELDS: Set[str] = {
    "id",
    "usecase",
    "scenario",
    "user_input",
    "conversation_history",
    "expected_behavior",
    "expected_function_calls",
}

OPTIONAL_CASE_FIELDS: Set[str] = {
    "target_tools",
    "accepted_function_call_sets",
}

ALLOWED_CASE_FIELDS: Set[str] = REQUIRED_CASE_FIELDS | OPTIONAL_CASE_FIELDS

PATIENT_ENTITY_TOOLS: Set[str] = {
    "show_one",
    "predict",
    "prediction_outcome_patient",
    "feature_importance_patient",
    "counterfactual",
    "what_if",
}

ALIAS_TOOLS: Set[str] = {
    "define_feature",
    "what_if",
    "count_patients",
    "performance_metrics",
}


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tool_counts: Counter[str] = field(default_factory=Counter)
    scenario_counts: Counter[str] = field(default_factory=Counter)
    tool_scenario_counts: Counter[Tuple[str, str]] = field(default_factory=Counter)
    case_count: int = 0

    @property
    def ready(self) -> bool:
        return not self.errors


def load_function_catalog(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw_catalog = json.load(f)

    catalog: Dict[str, Dict[str, Any]] = {}
    for item in raw_catalog:
        function = item.get("function", {})
        name = function.get("name")
        if isinstance(name, str):
            catalog[name] = function
    return catalog


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    if not path.exists():
        return rows, [f"Dataset file does not exist: {path}"]

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_number}: invalid JSON: {e}")
                continue
            if not isinstance(row, dict):
                errors.append(f"Line {line_number}: each JSONL row must be an object")
                continue
            rows.append(row)

    if not rows and not errors:
        errors.append(f"Dataset file is empty: {path}")

    return rows, errors


def validate_dataset(
    rows: Sequence[Dict[str, Any]],
    function_catalog: Mapping[str, Dict[str, Any]],
    *,
    min_direct_per_tool: int = 2,
    min_cross_tool_cases: int = 2,
) -> ValidationResult:
    result = ValidationResult(case_count=len(rows))
    seen_ids: Set[str] = set()
    direct_counts_by_tool: Counter[str] = Counter()
    required_arg_tools = {
        name
        for name, function in function_catalog.items()
        if _required_args(function)
    }

    for index, row in enumerate(rows, start=1):
        case_label = _case_label(row, index)
        missing_fields = REQUIRED_CASE_FIELDS - row.keys()
        if missing_fields:
            result.errors.append(
                f"{case_label}: missing required fields: {sorted(missing_fields)}"
            )
            continue

        unknown_fields = set(row) - ALLOWED_CASE_FIELDS
        if unknown_fields:
            result.errors.append(
                f"{case_label}: unknown fields are not allowed: {sorted(unknown_fields)}"
            )

        case_id = row["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            result.errors.append(f"{case_label}: id must be a non-empty string")
        elif case_id in seen_ids:
            result.errors.append(f"{case_label}: duplicate id '{case_id}'")
        else:
            seen_ids.add(case_id)

        if row["usecase"] != "heart":
            result.errors.append(f"{case_label}: usecase must be 'heart'")

        scenario = row["scenario"]
        if scenario not in SCENARIOS:
            result.errors.append(
                f"{case_label}: scenario must be one of {sorted(SCENARIOS)}"
            )
        else:
            result.scenario_counts[scenario] += 1

        expected_behavior = row["expected_behavior"]
        if expected_behavior not in EXPECTED_BEHAVIORS:
            result.errors.append(
                f"{case_label}: expected_behavior must be one of {sorted(EXPECTED_BEHAVIORS)}"
            )

        has_target_tools = "target_tools" in row
        target_tools = row.get("target_tools", [])
        if target_tools is None:
            target_tools = []
        if has_target_tools and scenario not in {
            "missing_required_argument",
            "conflicting_context",
        }:
            result.errors.append(
                f"{case_label}: target_tools is only allowed for no-call cases with a supported target intent"
            )
        if not _is_call_string_list(target_tools):
            result.errors.append(f"{case_label}: target_tools must be a list of strings")
            target_tools = []
        else:
            unknown_target_tools = sorted(set(target_tools) - set(function_catalog))
            if unknown_target_tools:
                result.errors.append(
                    f"{case_label}: target_tools includes unknown tools: {unknown_target_tools}"
                )
            for target_tool in target_tools:
                if scenario in SCENARIOS and target_tool in function_catalog:
                    result.tool_scenario_counts[(target_tool, scenario)] += 1

        if not isinstance(row["user_input"], str) or not row["user_input"].strip():
            result.errors.append(f"{case_label}: user_input must be a non-empty string")

        _validate_conversation_history(
            row["conversation_history"],
            function_catalog,
            case_label,
            result,
        )

        expected_calls = row["expected_function_calls"]
        if not isinstance(expected_calls, list):
            result.errors.append(f"{case_label}: expected_function_calls must be a list")
            continue

        accepted_sets = row.get("accepted_function_call_sets")
        call_sets = [expected_calls]
        if accepted_sets is not None:
            if not _valid_accepted_call_sets(accepted_sets):
                result.errors.append(
                    f"{case_label}: accepted_function_call_sets must be a list of call-string lists"
                )
            else:
                call_sets.extend(accepted_sets)

        if expected_behavior == "tool_call" and not expected_calls:
            result.errors.append(
                f"{case_label}: tool_call cases must include expected_function_calls"
            )
        if expected_behavior in NO_CALL_BEHAVIORS and expected_calls:
            result.errors.append(
                f"{case_label}: no-call cases must have empty expected_function_calls"
            )

        if scenario == "missing_required_argument" and not target_tools:
            result.errors.append(
                f"{case_label}: missing_required_argument cases must identify target_tools"
            )

        if scenario == "multi_tool_request" and expected_behavior == "tool_call":
            if len(expected_calls) < 2:
                result.errors.append(
                    f"{case_label}: multi_tool_request cases must include at least two calls"
                )

        if scenario in {"parameter_carryover", "entity_switch_or_correction", "conflicting_context"}:
            if not row["conversation_history"]:
                result.errors.append(
                    f"{case_label}: {scenario} cases must include conversation_history"
                )

        for call_set in call_sets:
            for call in call_set:
                parsed = _validate_function_call(call, function_catalog, case_label, result)
                if parsed is None:
                    continue
                function_name, _kwargs = parsed
                result.tool_counts[function_name] += 1
                if scenario in SCENARIOS:
                    result.tool_scenario_counts[(function_name, scenario)] += 1
                    if scenario == "direct_single_turn":
                        direct_counts_by_tool[function_name] += 1

    for function_name in sorted(function_catalog):
        if direct_counts_by_tool[function_name] < min_direct_per_tool:
            result.errors.append(
                f"Coverage: tool '{function_name}' has {direct_counts_by_tool[function_name]} "
                f"direct_single_turn seed cases; required minimum is {min_direct_per_tool}"
            )

    for function_name in sorted(ALIAS_TOOLS & function_catalog.keys()):
        if result.tool_scenario_counts[(function_name, "paraphrase_or_alias")] < 1:
            result.errors.append(
                f"Coverage: tool '{function_name}' needs at least 1 paraphrase_or_alias case"
            )

    for function_name in sorted(required_arg_tools):
        if result.tool_scenario_counts[(function_name, "missing_required_argument")] < 1:
            result.errors.append(
                f"Coverage: tool '{function_name}' needs at least 1 missing_required_argument case"
            )

    for function_name in sorted(PATIENT_ENTITY_TOOLS & function_catalog.keys()):
        if result.tool_scenario_counts[(function_name, "parameter_carryover")] < 1:
            result.errors.append(
                f"Coverage: patient-level tool '{function_name}' needs at least 1 parameter_carryover case"
            )
        if result.tool_scenario_counts[(function_name, "entity_switch_or_correction")] < 1:
            result.errors.append(
                f"Coverage: patient-level tool '{function_name}' needs at least 1 entity_switch_or_correction case"
            )

    for scenario in [
        "unsupported_intent",
        "no_tool_needed",
        "multi_tool_request",
        "conflicting_context",
    ]:
        if result.scenario_counts[scenario] < min_cross_tool_cases:
            result.errors.append(
                f"Coverage: scenario '{scenario}' has {result.scenario_counts[scenario]} cases; "
                f"required minimum is {min_cross_tool_cases}"
            )

    return result


def _case_label(row: Mapping[str, Any], index: int) -> str:
    case_id = row.get("id")
    if isinstance(case_id, str) and case_id:
        return f"Case '{case_id}'"
    return f"Line item {index}"


def _validate_conversation_history(
    conversation_history: Any,
    function_catalog: Mapping[str, Dict[str, Any]],
    case_label: str,
    result: ValidationResult,
) -> None:
    if not isinstance(conversation_history, list):
        result.errors.append(f"{case_label}: conversation_history must be a list")
        return

    previous_turn = 0
    for turn_index, turn in enumerate(conversation_history):
        if not isinstance(turn, dict):
            result.errors.append(
                f"{case_label}: conversation_history[{turn_index}] must be an object"
            )
            continue

        turn_number = turn.get("turn")
        if not isinstance(turn_number, int) or isinstance(turn_number, bool):
            result.errors.append(
                f"{case_label}: conversation_history[{turn_index}].turn must be an integer"
            )
        elif turn_number <= previous_turn:
            result.errors.append(
                f"{case_label}: conversation_history[{turn_index}].turn must increase monotonically"
            )
        else:
            previous_turn = turn_number

        user_input = turn.get("user_input")
        if not isinstance(user_input, str) or not user_input.strip():
            result.errors.append(
                f"{case_label}: conversation_history[{turn_index}].user_input must be a non-empty string"
            )

        function_calls = turn.get("function_calls")
        if not _is_call_string_list(function_calls):
            result.errors.append(
                f"{case_label}: conversation_history[{turn_index}].function_calls must be a list of strings"
            )
        else:
            for call in function_calls:
                _validate_function_call(
                    call,
                    function_catalog,
                    f"{case_label} conversation_history[{turn_index}]",
                    result,
                )


def _valid_accepted_call_sets(value: Any) -> bool:
    return isinstance(value, list) and all(_is_call_string_list(item) for item in value)


def _is_call_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_function_call(
    call: Any,
    function_catalog: Mapping[str, Dict[str, Any]],
    case_label: str,
    result: ValidationResult,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not isinstance(call, str):
        result.errors.append(f"{case_label}: expected function calls must be strings")
        return None

    try:
        function_name, kwargs = ASTParser.parse_function_call(call)
    except FunctionExecutionException as e:
        result.errors.append(f"{case_label}: cannot parse function call '{call}': {e}")
        return None

    function = function_catalog.get(function_name)
    if function is None:
        result.errors.append(f"{case_label}: unknown function '{function_name}'")
        return None

    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = set(_required_args(function))
    provided = set(kwargs)

    missing_required = required - provided
    if missing_required:
        result.errors.append(
            f"{case_label}: call '{call}' is missing required args: {sorted(missing_required)}"
        )

    unknown_args = provided - set(properties)
    if unknown_args:
        result.errors.append(
            f"{case_label}: call '{call}' includes unknown args: {sorted(unknown_args)}"
        )

    for arg_name, value in kwargs.items():
        schema = properties.get(arg_name)
        if schema is not None:
            _validate_arg_value(call, arg_name, value, schema, case_label, result)

    return function_name, kwargs


def _required_args(function: Mapping[str, Any]) -> List[str]:
    parameters = function.get("parameters") or {}
    required = parameters.get("required") or []
    if isinstance(required, list):
        return [item for item in required if isinstance(item, str)]
    return []


def _validate_arg_value(
    call: str,
    arg_name: str,
    value: Any,
    schema: Mapping[str, Any],
    case_label: str,
    result: ValidationResult,
) -> None:
    if "enum" in schema and value not in schema["enum"]:
        result.errors.append(
            f"{case_label}: call '{call}' arg '{arg_name}' must be one of {schema['enum']}"
        )
        return

    if "anyOf" in schema:
        if not any(_matches_schema_type(value, option) for option in schema["anyOf"]):
            result.errors.append(
                f"{case_label}: call '{call}' arg '{arg_name}' does not match any allowed type"
            )
        return

    schema_type = schema.get("type")
    if schema_type == "array":
        if not isinstance(value, list):
            result.errors.append(
                f"{case_label}: call '{call}' arg '{arg_name}' must be an array"
            )
            return
        item_schema = schema.get("items") or {}
        for item in value:
            if not _matches_schema_type(item, item_schema):
                result.errors.append(
                    f"{case_label}: call '{call}' arg '{arg_name}' has an invalid item value"
                )
                return
        return

    if schema_type and not _matches_schema_type(value, schema):
        result.errors.append(
            f"{case_label}: call '{call}' arg '{arg_name}' must be type {schema_type}"
        )


def _matches_schema_type(value: Any, schema: Mapping[str, Any]) -> bool:
    if "enum" in schema and value not in schema["enum"]:
        return False

    schema_type = schema.get("type")
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return True


def print_report(result: ValidationResult, dataset_path: Path, catalog_path: Path) -> None:
    status = "READY" if result.ready else "NOT READY"
    print(f"Seed-gold readiness: {status}")
    print(f"Dataset: {dataset_path}")
    print(f"Function catalog: {catalog_path}")
    print(f"Cases: {result.case_count}")
    print()

    print("Scenario coverage:")
    for scenario in sorted(SCENARIOS):
        print(f"  {scenario}: {result.scenario_counts[scenario]}")
    print()

    print("Tool coverage:")
    for tool_name in sorted(result.tool_counts):
        print(f"  {tool_name}: {result.tool_counts[tool_name]}")
    if not result.tool_counts:
        print("  none")
    print()

    if result.errors:
        print("Blocking issues:")
        for error in result.errors:
            print(f"  - {error}")
        print()

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
        print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the manual healthcare seed-gold dataset is ready for teacher enrichment."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to seed-gold JSONL dataset. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Path to heart function catalog. Default: {DEFAULT_CATALOG}",
    )
    parser.add_argument(
        "--min-direct-per-tool",
        type=int,
        default=2,
        help="Minimum direct_single_turn examples required per tool.",
    )
    parser.add_argument(
        "--min-cross-tool-cases",
        type=int,
        default=2,
        help="Minimum cases required for each cross-tool stress scenario.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    catalog = load_function_catalog(args.catalog)
    rows, read_errors = read_jsonl(args.dataset)

    if read_errors:
        result = ValidationResult(errors=read_errors)
    else:
        result = validate_dataset(
            rows,
            catalog,
            min_direct_per_tool=args.min_direct_per_tool,
            min_cross_tool_cases=args.min_cross_tool_cases,
        )

    print_report(result, args.dataset, args.catalog)
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
