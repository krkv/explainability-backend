#!/usr/bin/env python3
"""Help author the manual seed-gold healthcare tool-calling dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_seed_gold import (  # noqa: E402
    ALIAS_TOOLS,
    DEFAULT_CATALOG,
    DEFAULT_DATASET,
    PATIENT_ENTITY_TOOLS,
    _required_args,
    load_function_catalog,
    read_jsonl,
    validate_dataset,
)


@dataclass(frozen=True)
class MissingNeed:
    scenario: str
    count: int
    tool: Optional[str] = None

    @property
    def label(self) -> str:
        if self.tool:
            return f"{self.scenario}:{self.tool}"
        return self.scenario


def compute_missing_needs(
    rows: Sequence[Dict[str, Any]],
    function_catalog: Mapping[str, Dict[str, Any]],
    *,
    min_direct_per_tool: int = 2,
    min_cross_tool_cases: int = 2,
) -> List[MissingNeed]:
    result = validate_dataset(
        rows,
        function_catalog,
        min_direct_per_tool=min_direct_per_tool,
        min_cross_tool_cases=min_cross_tool_cases,
    )
    needs: List[MissingNeed] = []

    for function_name in sorted(function_catalog):
        missing = min_direct_per_tool - result.tool_scenario_counts[
            (function_name, "direct_single_turn")
        ]
        if missing > 0:
            needs.append(MissingNeed("direct_single_turn", missing, function_name))

    for function_name in sorted(ALIAS_TOOLS & function_catalog.keys()):
        missing = 1 - result.tool_scenario_counts[(function_name, "paraphrase_or_alias")]
        if missing > 0:
            needs.append(MissingNeed("paraphrase_or_alias", missing, function_name))

    required_arg_tools = {
        name
        for name, function in function_catalog.items()
        if _required_args(function)
    }
    for function_name in sorted(required_arg_tools):
        missing = 1 - result.tool_scenario_counts[
            (function_name, "missing_required_argument")
        ]
        if missing > 0:
            needs.append(MissingNeed("missing_required_argument", missing, function_name))

    for function_name in sorted(PATIENT_ENTITY_TOOLS & function_catalog.keys()):
        missing = 1 - result.tool_scenario_counts[
            (function_name, "parameter_carryover")
        ]
        if missing > 0:
            needs.append(MissingNeed("parameter_carryover", missing, function_name))

        missing = 1 - result.tool_scenario_counts[
            (function_name, "entity_switch_or_correction")
        ]
        if missing > 0:
            needs.append(
                MissingNeed("entity_switch_or_correction", missing, function_name)
            )

    for scenario in [
        "unsupported_intent",
        "no_tool_needed",
        "multi_tool_request",
        "conflicting_context",
    ]:
        missing = min_cross_tool_cases - result.scenario_counts[scenario]
        if missing > 0:
            needs.append(MissingNeed(scenario, missing))

    return needs


def build_template_rows(
    needs: Sequence[MissingNeed],
    function_catalog: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for need in needs:
        for index in range(1, need.count + 1):
            rows.append(_template_for_need(need, function_catalog, index))

    return rows


def _template_for_need(
    need: MissingNeed,
    function_catalog: Mapping[str, Dict[str, Any]],
    index: int,
) -> Dict[str, Any]:
    if need.tool:
        return _tool_template(need.tool, need.scenario, function_catalog, index)
    return _cross_tool_template(need.scenario, index)


def _tool_template(
    tool_name: str,
    scenario: str,
    function_catalog: Mapping[str, Dict[str, Any]],
    index: int,
) -> Dict[str, Any]:
    call = _sample_call(tool_name, function_catalog[tool_name], patient_id=42)
    row_id = f"heart_{tool_name}_{scenario}_todo_{index:03d}"

    if scenario == "direct_single_turn":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": _direct_prompt_hint(tool_name),
            "conversation_history": [],
            "expected_behavior": "tool_call",
            "expected_function_calls": [call],
            "target_tools": [tool_name],
            "notes": "Rewrite user_input as a realistic direct request before moving to seed_gold.jsonl.",
        }

    if scenario == "paraphrase_or_alias":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": _alias_prompt_hint(tool_name),
            "conversation_history": [],
            "expected_behavior": "tool_call",
            "expected_function_calls": [call],
            "target_tools": [tool_name],
            "notes": "Use natural user wording, aliases, or display labels instead of function names.",
        }

    if scenario == "missing_required_argument":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": _missing_arg_prompt_hint(tool_name),
            "conversation_history": [],
            "expected_behavior": "no_call_clarify",
            "expected_function_calls": [],
            "target_tools": [tool_name],
            "notes": "The model should not invent missing required arguments.",
        }

    if scenario == "parameter_carryover":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": _carryover_prompt_hint(tool_name),
            "conversation_history": [
                {
                    "turn": 1,
                    "user_input": "Show me patient 42.",
                    "function_calls": ["show_one(patient_id=42)"],
                },
            ],
            "expected_behavior": "tool_call",
            "expected_function_calls": [call],
            "target_tools": [tool_name],
            "notes": "The expected call should reuse patient_id=42 from conversation history.",
        }

    if scenario == "entity_switch_or_correction":
        previous_call = _sample_call(tool_name, function_catalog[tool_name], patient_id=42)
        corrected_call = _sample_call(tool_name, function_catalog[tool_name], patient_id=51)
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": _correction_prompt_hint(tool_name),
            "conversation_history": [
                {
                    "turn": 1,
                    "user_input": _direct_prompt_hint(tool_name),
                    "function_calls": [previous_call],
                },
            ],
            "expected_behavior": "tool_call",
            "expected_function_calls": [corrected_call],
            "target_tools": [tool_name],
            "notes": "The latest user correction should override the prior patient_id=42.",
        }

    raise ValueError(f"Unsupported tool scenario: {scenario}")


def _cross_tool_template(scenario: str, index: int) -> Dict[str, Any]:
    row_id = f"heart_{scenario}_todo_{index:03d}"

    if scenario == "unsupported_intent":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "Should this patient start medication?",
            "conversation_history": [],
            "expected_behavior": "no_call_unsupported",
            "expected_function_calls": [],
            "target_tools": [],
            "notes": "Unsupported clinical advice request. Rewrite to another realistic unsupported request if needed.",
        }

    if scenario == "no_tool_needed":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "Thanks, that helps.",
            "conversation_history": [],
            "expected_behavior": "no_call_needed",
            "expected_function_calls": [],
            "target_tools": [],
            "notes": "No backend function should be called for conversational acknowledgements.",
        }

    if scenario == "multi_tool_request":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "Predict patient 42 and explain the main factors.",
            "conversation_history": [],
            "expected_behavior": "tool_call",
            "expected_function_calls": [
                "predict(patient_id=42)",
                "feature_importance_patient(patient_id=42)",
            ],
            "target_tools": ["predict", "feature_importance_patient"],
            "notes": "Use this to test whether the model emits more than one supported call.",
        }

    if scenario == "conflicting_context":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "Explain that patient's prediction.",
            "conversation_history": [
                {
                    "turn": 1,
                    "user_input": "What can you help me analyze?",
                    "function_calls": ["available_functions()"],
                },
                {
                    "turn": 2,
                    "user_input": "Thanks, that helps.",
                    "function_calls": [],
                },
                {
                    "turn": 3,
                    "user_input": "Show me patient 42.",
                    "function_calls": ["show_one(patient_id=42)"],
                },
                {
                    "turn": 4,
                    "user_input": "Now show me patient 51.",
                    "function_calls": ["show_one(patient_id=51)"],
                },
            ],
            "expected_behavior": "no_call_clarify",
            "expected_function_calls": [],
            "target_tools": ["feature_importance_patient"],
            "notes": "Ambiguous referent. Keep as no-call clarify unless the annotation policy says latest patient wins.",
        }

    raise ValueError(f"Unsupported cross-tool scenario: {scenario}")


def _sample_call(
    tool_name: str,
    function: Mapping[str, Any],
    *,
    patient_id: int,
) -> str:
    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    argument_names = set(_required_args(function))
    argument_names.update(_intentional_optional_args(tool_name))
    args = {
        arg_name: _sample_value(arg_name, schema, patient_id=patient_id)
        for arg_name, schema in properties.items()
        if arg_name in argument_names
    }

    if not args:
        return f"{tool_name}()"

    rendered_args = ", ".join(
        f"{arg_name}={_render_literal(value)}" for arg_name, value in args.items()
    )
    return f"{tool_name}({rendered_args})"


def _intentional_optional_args(tool_name: str) -> List[str]:
    optional_args_by_tool = {
        "count_patients": ["count_type"],
        "performance_metrics": ["metrics"],
    }
    return optional_args_by_tool.get(tool_name, [])


def _sample_value(arg_name: str, schema: Mapping[str, Any], *, patient_id: int) -> Any:
    if arg_name == "patient_id":
        return patient_id
    if arg_name == "feature":
        enum = schema.get("enum") or []
        if "trestbps" in enum:
            return "trestbps"
        return enum[0] if enum else "trestbps"
    if arg_name == "value_change":
        return -15
    if arg_name == "count_type":
        return "positive_predicted"
    if arg_name == "metrics":
        return ["accuracy"]

    enum = schema.get("enum") or []
    if enum:
        return enum[0]

    if "anyOf" in schema:
        return _sample_value(arg_name, schema["anyOf"][0], patient_id=patient_id)

    schema_type = schema.get("type")
    if schema_type == "integer":
        return 42
    if schema_type == "number":
        return 1.0
    if schema_type == "string":
        return "example"
    if schema_type == "array":
        item_schema = schema.get("items") or {"type": "string"}
        return [_sample_value(arg_name, item_schema, patient_id=patient_id)]
    if schema_type == "boolean":
        return True
    if schema_type == "object":
        return {}

    return "example"


def _render_literal(value: Any) -> str:
    return repr(value)


def _direct_prompt_hint(tool_name: str) -> str:
    hints = {
        "available_functions": "What can you help me analyze?",
        "age_group_performance": "How does the model perform across age groups?",
        "count_patients": "How many patients are predicted to have heart disease?",
        "show_ids": "Show me the available patient IDs.",
        "show_one": "Show me patient 42.",
        "confusion_matrix_stats": "Show the confusion matrix statistics.",
        "counterfactual": "What changes would flip the prediction for patient 42?",
        "dataset_summary": "Show a summary of the dataset.",
        "define_feature": "What does resting blood pressure mean?",
        "feature_importance_patient": "Why did the model make its prediction for patient 42?",
        "feature_importance_global": "Which features matter most overall?",
        "feature_interactions": "Which features are most correlated with each other?",
        "get_dataset_description": "Describe the heart disease dataset.",
        "get_model_description": "What kind of model is being used?",
        "get_model_parameters": "What are the model hyperparameters?",
        "misclassified_cases": "Show me the common patterns in misclassified cases.",
        "performance_metrics": "What are the model accuracy and precision?",
        "predict": "Predict the outcome for patient 42.",
        "prediction_outcome_patient": "Was the model's prediction for patient 42 correct?",
        "what_if": "What if patient 42 had blood pressure 15 points lower?",
    }
    return hints.get(tool_name, f"Ask a direct user question for {tool_name}.")


def _alias_prompt_hint(tool_name: str) -> str:
    hints = {
        "count_patients": "How many people does the model think have heart disease?",
        "define_feature": "What does blood pressure mean in this dataset?",
        "performance_metrics": "How accurate is this model?",
        "what_if": "Would patient 42 look less risky if their blood pressure was lower?",
    }
    return hints.get(tool_name, f"Ask for {tool_name} using natural language aliases.")


def _missing_arg_prompt_hint(tool_name: str) -> str:
    hints = {
        "counterfactual": "What changes would flip this patient's prediction?",
        "define_feature": "What does this feature mean?",
        "feature_importance_patient": "Why did the model make this prediction?",
        "predict": "Can you predict this patient?",
        "prediction_outcome_patient": "Was the model right for this patient?",
        "show_one": "Show me that patient.",
        "what_if": "What if their blood pressure was lower?",
    }
    return hints.get(tool_name, f"Ask for {tool_name} while omitting required details.")


def _carryover_prompt_hint(tool_name: str) -> str:
    hints = {
        "counterfactual": "What changes would flip that patient's prediction?",
        "feature_importance_patient": "Why did the model make that prediction?",
        "predict": "Can you predict that same patient?",
        "prediction_outcome_patient": "Was the model right for that same patient?",
        "show_one": "Show me that patient again.",
        "what_if": "What if that patient's blood pressure was 15 lower?",
    }
    return hints.get(tool_name, f"Ask for {tool_name} using the same patient.")


def _correction_prompt_hint(tool_name: str) -> str:
    hints = {
        "counterfactual": "Wait, not that one. Show counterfactuals for patient 51.",
        "feature_importance_patient": "Wait, explain patient 51 instead.",
        "predict": "Wait, predict patient 51 instead.",
        "prediction_outcome_patient": "Wait, check whether the prediction was correct for patient 51 instead.",
        "show_one": "Wait, show patient 51 instead.",
        "what_if": "Wait, use patient 51 for that what-if instead.",
    }
    return hints.get(tool_name, f"Correct the prior request to patient 51 for {tool_name}.")


def print_coverage(
    needs: Sequence[MissingNeed],
    rows: Sequence[Dict[str, Any]],
    read_errors: Sequence[str],
    dataset_path: Path,
) -> None:
    print(f"Dataset: {dataset_path}")
    print(f"Parsed rows: {len(rows)}")
    if read_errors:
        print()
        print("Read issues:")
        for error in read_errors:
            print(f"  - {error}")

    print()
    if not needs:
        print("Missing coverage: none")
        print("The manual seed set satisfies the readiness coverage minimums.")
        return

    print("Missing coverage:")
    grouped: Dict[str, List[MissingNeed]] = {}
    for need in needs:
        grouped.setdefault(need.scenario, []).append(need)

    for scenario in sorted(grouped):
        print(f"  {scenario}:")
        for need in grouped[scenario]:
            if need.tool:
                print(f"    - {need.tool}: need {need.count}")
            else:
                print(f"    - need {need.count}")


def print_templates(rows: Sequence[Dict[str, Any]]) -> None:
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show missing seed-gold coverage and generate JSONL authoring templates."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to current seed-gold JSONL dataset. Default: {DEFAULT_DATASET}",
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

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("coverage", help="Show missing seed-gold coverage.")
    subparsers.add_parser(
        "templates",
        help="Print JSONL template rows for missing seed-gold coverage.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    catalog = load_function_catalog(args.catalog)
    rows, read_errors = read_jsonl(args.dataset)
    needs = compute_missing_needs(
        rows,
        catalog,
        min_direct_per_tool=args.min_direct_per_tool,
        min_cross_tool_cases=args.min_cross_tool_cases,
    )

    if args.command == "coverage":
        print_coverage(needs, rows, read_errors, args.dataset)
        return 0

    if args.command == "templates":
        print_templates(build_template_rows(needs, catalog))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
