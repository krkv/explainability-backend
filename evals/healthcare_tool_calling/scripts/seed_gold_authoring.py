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
    start_index: int = 1

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
            start_index = result.tool_scenario_counts[
                (function_name, "direct_single_turn")
            ] + 1
            needs.append(
                MissingNeed("direct_single_turn", missing, function_name, start_index)
            )

    for function_name in sorted(ALIAS_TOOLS & function_catalog.keys()):
        missing = 1 - result.tool_scenario_counts[(function_name, "paraphrase_or_alias")]
        if missing > 0:
            start_index = result.tool_scenario_counts[
                (function_name, "paraphrase_or_alias")
            ] + 1
            needs.append(
                MissingNeed("paraphrase_or_alias", missing, function_name, start_index)
            )

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
            start_index = result.tool_scenario_counts[
                (function_name, "missing_required_argument")
            ] + 1
            needs.append(
                MissingNeed(
                    "missing_required_argument",
                    missing,
                    function_name,
                    start_index,
                )
            )

    for function_name in sorted(PATIENT_ENTITY_TOOLS & function_catalog.keys()):
        missing = 1 - result.tool_scenario_counts[
            (function_name, "parameter_carryover")
        ]
        if missing > 0:
            start_index = result.tool_scenario_counts[
                (function_name, "parameter_carryover")
            ] + 1
            needs.append(
                MissingNeed("parameter_carryover", missing, function_name, start_index)
            )

        missing = 1 - result.tool_scenario_counts[
            (function_name, "entity_switch_or_correction")
        ]
        if missing > 0:
            start_index = result.tool_scenario_counts[
                (function_name, "entity_switch_or_correction")
            ] + 1
            needs.append(
                MissingNeed(
                    "entity_switch_or_correction",
                    missing,
                    function_name,
                    start_index,
                )
            )

    for scenario in [
        "unsupported_intent",
        "no_tool_needed",
        "multi_tool_request",
        "conflicting_context",
        "irrelevant_context",
    ]:
        missing = min_cross_tool_cases - result.scenario_counts[scenario]
        if missing > 0:
            start_index = result.scenario_counts[scenario] + 1
            needs.append(MissingNeed(scenario, missing, start_index=start_index))

    return needs


def build_template_rows(
    needs: Sequence[MissingNeed],
    function_catalog: Mapping[str, Dict[str, Any]],
    *,
    limit: Optional[int] = 10,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for need in needs:
        for index in range(need.start_index, need.start_index + need.count):
            if limit is not None and len(rows) >= limit:
                return rows
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
    _function_catalog: Mapping[str, Dict[str, Any]],
    index: int,
) -> Dict[str, Any]:
    row_id = f"heart_{tool_name}_{scenario}_{index:03d}"

    if scenario == "direct_single_turn":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [],
            "expected_behavior": "tool_call",
            "expected_function_calls": [],
        }

    if scenario == "paraphrase_or_alias":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [],
            "expected_behavior": "tool_call",
            "expected_function_calls": [],
        }

    if scenario == "missing_required_argument":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [],
            "expected_behavior": "no_call_clarify",
            "expected_function_calls": [],
            "target_tools": [tool_name],
        }

    if scenario == "parameter_carryover":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [
                {
                    "turn": 1,
                    "user_input": "",
                    "function_calls": [],
                },
            ],
            "expected_behavior": "tool_call",
            "expected_function_calls": [],
        }

    if scenario == "entity_switch_or_correction":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [
                {
                    "turn": 1,
                    "user_input": "",
                    "function_calls": [],
                },
            ],
            "expected_behavior": "tool_call",
            "expected_function_calls": [],
        }

    raise ValueError(f"Unsupported tool scenario: {scenario}")


def _cross_tool_template(scenario: str, index: int) -> Dict[str, Any]:
    row_id = f"heart_{scenario}_{index:03d}"

    if scenario == "unsupported_intent":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [],
            "expected_behavior": "no_call_unsupported",
            "expected_function_calls": [],
        }

    if scenario == "no_tool_needed":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [],
            "expected_behavior": "no_call_needed",
            "expected_function_calls": [],
        }

    if scenario == "multi_tool_request":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [],
            "expected_behavior": "tool_call",
            "expected_function_calls": [],
        }

    if scenario == "conflicting_context":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [
                {
                    "turn": 1,
                    "user_input": "",
                    "function_calls": [],
                },
                {
                    "turn": 2,
                    "user_input": "",
                    "function_calls": [],
                },
                {
                    "turn": 3,
                    "user_input": "",
                    "function_calls": [],
                },
                {
                    "turn": 4,
                    "user_input": "",
                    "function_calls": [],
                },
            ],
            "expected_behavior": "no_call_clarify",
            "expected_function_calls": [],
            "target_tools": ["feature_importance_patient"],
        }

    if scenario == "irrelevant_context":
        return {
            "id": row_id,
            "usecase": "heart",
            "scenario": scenario,
            "user_input": "",
            "conversation_history": [
                {
                    "turn": 1,
                    "user_input": "",
                    "function_calls": [],
                },
            ],
            "expected_behavior": "tool_call",
            "expected_function_calls": [],
        }

    raise ValueError(f"Unsupported cross-tool scenario: {scenario}")


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
    templates_parser = subparsers.add_parser(
        "templates",
        help="Print JSONL template rows for missing seed-gold coverage. Defaults to 10 rows.",
    )
    templates_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of template rows to print. Use 0 for no limit.",
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
        limit = None if args.limit == 0 else args.limit
        print_templates(build_template_rows(needs, catalog, limit=limit))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
