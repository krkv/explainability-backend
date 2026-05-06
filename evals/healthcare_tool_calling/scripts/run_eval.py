#!/usr/bin/env python3
"""Run healthcare tool-calling evaluation for one configured model."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

try:
    from eval_common import (
        DEFAULT_MODEL_CONFIGS,
        append_jsonl,
        build_live_conversation,
        load_model_configs,
        read_jsonl,
        read_model_response,
        resolve_eval_provider,
        write_jsonl,
    )
except ModuleNotFoundError:
    from evals.healthcare_tool_calling.scripts.eval_common import (
        DEFAULT_MODEL_CONFIGS,
        append_jsonl,
        build_live_conversation,
        load_model_configs,
        read_jsonl,
        read_model_response,
        resolve_eval_provider,
        write_jsonl,
    )
from src.core.constants import UseCase
from src.domain.interfaces.llm_provider import AgentRole
from src.services.llm.generation_config_resolver import resolve_generation_config


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1] / "datasets" / "reviewed_gold_v1.jsonl"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one model against the healthcare tool-calling eval dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", required=True, help="Model id from model_configs.json.")
    parser.add_argument("--dataset-version", default="reviewed_gold_v1")
    parser.add_argument("--model-configs", type=Path, default=DEFAULT_MODEL_CONFIGS)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of dataset rows to run.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Optional number of dataset rows to skip before running.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing predictions file instead of resuming.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Retry only existing rows with provider/schema errors. Removes only those "
            "case IDs from raw/prediction logs before rerunning them."
        ),
    )
    return parser.parse_args(argv)


def _existing_case_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {
        str(row.get("case_id"))
        for row in read_jsonl(path)
        if row.get("case_id") is not None
    }


def _failed_response_case_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    failed_case_ids: Set[str] = set()
    for row in read_jsonl(path):
        case_id = row.get("case_id")
        if case_id is None:
            continue
        if row.get("error") or row.get("response_valid") is False:
            failed_case_ids.add(str(case_id))
    return failed_case_ids


def _remove_case_ids_from_jsonl(path: Path, case_ids: Set[str]) -> None:
    if not path.exists() or not case_ids:
        return
    rows = [
        row
        for row in read_jsonl(path)
        if str(row.get("case_id")) not in case_ids
    ]
    write_jsonl(path, rows)


def _remove_stale_score_artifacts(output_dir: Path) -> None:
    for filename in ["scores.json", "errors.jsonl"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


async def run_eval(args: argparse.Namespace) -> Path:
    if args.overwrite and args.retry_errors:
        raise ValueError("--overwrite and --retry-errors cannot be used together")

    model_configs = load_model_configs(args.model_configs)
    if args.model not in model_configs:
        raise ValueError(f"Unknown model '{args.model}'. Check {args.model_configs}")

    provider_resolution = resolve_eval_provider(args.model, model_configs[args.model])
    provider = provider_resolution.provider
    if not provider.is_available():
        raise RuntimeError(f"Provider for '{args.model}' is not available")

    output_dir = args.reports_dir / args.dataset_version / args.model
    predictions_path = output_dir / "predictions.jsonl"
    raw_generations_path = output_dir / "raw_generations.jsonl"
    if args.overwrite:
        for path in [predictions_path, raw_generations_path]:
            if path.exists():
                path.unlink()
        _remove_stale_score_artifacts(output_dir)

    retry_case_ids: Optional[Set[str]] = None
    if args.retry_errors:
        retry_case_ids = _failed_response_case_ids(predictions_path)
        if not retry_case_ids:
            return predictions_path
        _remove_case_ids_from_jsonl(predictions_path, retry_case_ids)
        _remove_case_ids_from_jsonl(raw_generations_path, retry_case_ids)
        _remove_stale_score_artifacts(output_dir)

    completed_case_ids = _existing_case_ids(predictions_path)
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")

    dataset_rows = read_jsonl(args.dataset)
    if args.offset:
        dataset_rows = dataset_rows[args.offset :]
    if args.limit is not None:
        dataset_rows = dataset_rows[: args.limit]
    if retry_case_ids is not None:
        dataset_rows = [
            row for row in read_jsonl(args.dataset) if str(row["id"]) in retry_case_ids
        ]

    for row in dataset_rows:
        case_id = str(row["id"])
        if case_id in completed_case_ids:
            continue

        conversation = build_live_conversation(row)
        _usecase_enum, generation_config = resolve_generation_config(
            conversation=conversation,
            usecase=UseCase.HEART,
            agent_role=AgentRole.ASSISTANT,
        )
        start = time.perf_counter()
        raw_response = ""
        error = None
        try:
            raw_response = await provider.generate_response(
                conversation,
                UseCase.HEART,
                generation_config=generation_config,
            )
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        latency_ms = int((time.perf_counter() - start) * 1000)
        raw_generation: Dict[str, Any] = {
            "case_id": case_id,
            "model_id": args.model,
            "provider_model_id": provider_resolution.provider_model_id,
            "dataset_version": args.dataset_version,
            "scenario": row["scenario"],
            "request": {
                "usecase": UseCase.HEART.value,
                "agent_role": AgentRole.ASSISTANT.value,
                "conversation": conversation,
                "system_prompt": generation_config.system_prompt,
                "response_schema": generation_config.response_schema,
            },
            "raw_response": raw_response,
            "provider_error": error,
            "latency_ms": latency_ms,
        }
        append_jsonl(raw_generations_path, [raw_generation])

        read_result = read_model_response(raw_response)
        prediction: Dict[str, Any] = {
            "case_id": case_id,
            "model_id": args.model,
            "provider_model_id": provider_resolution.provider_model_id,
            "dataset_version": args.dataset_version,
            "scenario": row["scenario"],
            "expected_behavior": row["expected_behavior"],
            "input": {
                "user_input": row["user_input"],
                "conversation_history": row["conversation_history"],
            },
            "raw_response": raw_response,
            "predicted_function_calls": read_result.predicted_function_calls,
            "predicted_freeform_response": read_result.predicted_freeform_response,
            "expected_function_calls": row["expected_function_calls"],
            "target_tools": row.get("target_tools", []),
            "response_valid": read_result.response_valid,
            "error": error or read_result.error,
            "latency_ms": latency_ms,
            "raw_generation_log": str(raw_generations_path),
        }
        append_jsonl(predictions_path, [prediction])

    return predictions_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        predictions_path = asyncio.run(run_eval(args))
    except Exception as e:
        print(f"Error: {e}")
        return 1
    print(f"Wrote predictions to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
