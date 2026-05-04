# Opus 4.7 Teacher Prompt: Healthcare Tool-Calling Candidate Review

You are reviewing candidate benchmark rows for a healthcare tool-calling evaluation.

The benchmark tests whether an LLM can map a user request plus compact conversation history to the correct backend function calls for a heart disease model explanation assistant.

Your job is to find problems in candidate JSONL rows before human review. You are not scoring a model. You are auditing candidate labels.

## Attached Files To Use

Use these attached files as the complete review context:

1. `functions.json`
   - The authoritative function catalog.

2. `feature_metadata.json`
   - The authoritative feature metadata, aliases, display names, and categorical labels.

3. `dataset_metadata.json`
   - The high-level dataset description.

4. `seed_gold.jsonl`
   - The reviewed manual seed dataset and annotation style reference.

5. `teacher_generated_raw_batch_001.jsonl`
   - The candidate rows to review.

Do not assume repository code or hidden variables.

## Review Criteria

Check every candidate row for:

- valid JSONL shape
- no extra fields such as `notes` or rationale
- allowed `scenario`
- allowed `expected_behavior`
- valid function names from `functions.json`
- keyword-only function-call arguments
- required arguments present
- no unsupported arguments
- feature names or aliases supported by `feature_metadata.json`
- categorical `what_if` values represented as string labels or codes
- continuous `what_if` values represented as numeric deltas
- valid patient IDs from 0 through 59
- compact `conversation_history` shape with `turn`, `user_input`, and `function_calls`
- no assistant free-form response content in history
- correct no-call behavior when the request is ambiguous, unsupported, or needs no tool
- correct carryover from prior function calls when required
- no stale-context leakage in `irrelevant_context`
- latest correction wins in `entity_switch_or_correction`
- no near-duplicates of the seed examples or other candidate rows

## Output Format

Create a file named:

```text
teacher_generated_raw_batch_001_review.json
```

Write the review result into that file as one JSON object. If your interface
does not support file creation, output only the JSON object so the user can save
it under that filename.

For later review batches, the user may change `001` to `002`, `003`, and so on
before running the prompt. Use the same batch number as the candidate JSONL file
being reviewed.

The JSON object must have exactly these top-level keys:

```json
{
  "summary": "",
  "blocking_issues": [],
  "warnings": [],
  "recommended_accept_ids": [],
  "recommended_reject_ids": []
}
```

Each `blocking_issues` item must have:

```json
{
  "id": "candidate id",
  "issue": "short precise issue",
  "suggested_fix": "specific replacement or action"
}
```

Each `warnings` item must have:

```json
{
  "id": "candidate id",
  "warning": "short precise warning",
  "suggested_fix": "specific optional improvement"
}
```

Use `blocking_issues` for rows that should not enter the reviewed benchmark without changes.

Use `warnings` for rows that are valid but weak, too broad, too easy, awkwardly worded, near-duplicate, or diagnostically low-value.

Do not rewrite the whole dataset. Do not output corrected JSONL unless a specific `suggested_fix` needs a compact replacement value.

Be strict. Candidate rows are cheap; benchmark truth is expensive.
