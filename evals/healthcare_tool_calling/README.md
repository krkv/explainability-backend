# Healthcare Tool-Calling Evaluation

This directory defines the planned evaluation structure for the healthcare
tool-calling use case. The goal is to evaluate how well different LLMs map a
user request and recent conversation history to the correct backend function
calls, including arguments.

The first benchmark target is the heart disease use case. The source of truth
for valid tools is the heart function catalog:

```text
instances/heart/functions.json
```

This scaffold includes the reviewed dataset, seed authoring helpers, a
coverage/readiness checker, and a minimal student-model runner/scorer for
tool-call evaluation. Teacher generation remains external to the repository.

## Dataset Lifecycle

Use three separate dataset stages:

```text
seed_gold -> teacher_generated_raw -> reviewed_gold_v1
```

- `seed_gold`: small, manually authored seed set with trusted labels.
- `teacher_generated_raw`: model-generated candidate samples used only to scale
  coverage.
- `reviewed_gold_v1`: manually reviewed benchmark data used for formal scoring.

Teacher-generated examples must not become benchmark truth directly. A teacher
model provides candidate labels, not ground truth. Using raw teacher output as
the benchmark risks measuring whether other models imitate the teacher's tool
selection and argument style rather than whether they follow the healthcare
tool contract correctly.

## Dataset Files

Future JSONL datasets should live in `datasets/`, for example:

```text
datasets/seed_gold.jsonl
datasets/teacher_generated_raw.jsonl
datasets/reviewed_gold_v1.jsonl
```

Future schema definitions should live in `schemas/`.

Future evaluation outputs should live in `reports/`.

## Student Model Evaluation

The frozen reviewed dataset is:

```text
datasets/reviewed_gold_v1.jsonl
```

Run one configured model at a time:

```bash
python3 evals/healthcare_tool_calling/scripts/run_eval.py \
  --dataset evals/healthcare_tool_calling/datasets/reviewed_gold_v1.jsonl \
  --dataset-version reviewed_gold_v1 \
  --model gpt-5.4-mini
```

Raw model I/O is written first, before response parsing:

```text
reports/reviewed_gold_v1/<model_id>/raw_generations.jsonl
```

Each raw row includes the live-style conversation, generated system prompt,
response schema, exact raw model response, provider error if any, and latency.

Processed predictions are then written to:

```text
reports/reviewed_gold_v1/<model_id>/predictions.jsonl
```

Score a prediction file:

```bash
python3 evals/healthcare_tool_calling/scripts/score_predictions.py \
  --predictions evals/healthcare_tool_calling/reports/reviewed_gold_v1/gpt-5.4-mini/predictions.jsonl
```

Scores are written next to predictions as `scores.json`; failed cases are
written as `errors.jsonl`.

Scoring canonicalizes backend-equivalent argument aliases before comparison.
For example, feature aliases such as `cholesterol` and `chol`, categorical
labels/codes in `what_if`, metric aliases such as `auc` and `auc_roc`, and
patient-count aliases such as `all` and `total` are compared by their canonical
backend meaning rather than by raw string equality.

The runner uses the live heart assistant prompt path and response schema. Native
provider tool calling is not used. The model must return JSON with
`function_calls` and `freeform_response`; the scorer evaluates only
`function_calls` for now and saves `freeform_response` for later evaluation.

Configured student targets live in:

```text
configs/model_configs.json
```

Gemini and GPT are runnable through the current backend providers. Gemma 4 and
Kimi K2.5 are runnable through OpenRouter when `OPENROUTER_API_KEY` is
configured:

```bash
python3 evals/healthcare_tool_calling/scripts/run_eval.py --model gemma-4 --limit 10 --overwrite
python3 evals/healthcare_tool_calling/scripts/run_eval.py --model kimi-k2.5 --limit 10 --overwrite
```

For slow or rate-limited providers, retry only failed provider/schema responses
without deleting successful rows:

```bash
python3 evals/healthcare_tool_calling/scripts/run_eval.py --model gemma-4 --retry-errors
```

After any retry or append run, rerun `score_predictions.py` because stale score
artifacts are removed when predictions change.

## Manual Authoring Workflow

Do not write every JSON object from scratch. Use the authoring helper to see
what the current manual seed set is missing:

```bash
python3 evals/healthcare_tool_calling/scripts/seed_gold_authoring.py coverage
```

Generate editable JSONL templates for the missing cases:

```bash
python3 evals/healthcare_tool_calling/scripts/seed_gold_authoring.py templates
```

The template command writes the next 10 missing JSONL skeleton rows to stdout.
It intentionally does not prefill `user_input` or `expected_function_calls`;
those fields must be manually authored so generated text does not leak into the
reviewed seed set. Copy the rows you want into a draft file such as:

```text
evals/healthcare_tool_calling/datasets/seed_gold_draft.jsonl
```

Use a different batch size when needed:

```bash
python3 evals/healthcare_tool_calling/scripts/seed_gold_authoring.py templates --limit 5
```

Use `--limit 0` to print all currently missing templates.

Then manually fill and review the rows before moving them into:

```text
evals/healthcare_tool_calling/datasets/seed_gold.jsonl
```

The most important fields to manually author are:

- `user_input`: make this a realistic user message, not a mechanical prompt.
- `conversation_history`: use compact prior turns matching the healthcare
  prompt context. Each prior turn has `turn`, `user_input`, and
  `function_calls`. Include prior turns even when no functions were called by
  setting `function_calls` to `[]`.
- `expected_function_calls`: verify the selected tools and arguments are the
  intended gold parse.
- `expected_behavior`: use `tool_call`, `no_call_clarify`,
  `no_call_unsupported`, or `no_call_needed`.
- `target_tools`: include this only for no-call cases where the intended
  supported tool cannot be recovered from `expected_function_calls`, such as
  `missing_required_argument`.

Recommended loop:

```text
authoring coverage -> authoring templates -> manual edit/review -> readiness check
```

## Seed Gold Readiness Check

Before using a teacher model to expand the dataset, run the seed-gold readiness
checker:

```bash
python3 evals/healthcare_tool_calling/scripts/check_seed_gold.py
```

By default, the checker reads:

```text
evals/healthcare_tool_calling/datasets/seed_gold.jsonl
instances/heart/functions.json
```

The checker is intentionally read-only. It reports blocking issues and exits
with a non-zero status when the manual seed set is not sufficient for teacher
enrichment.

Use custom paths when needed:

```bash
python3 evals/healthcare_tool_calling/scripts/check_seed_gold.py \
  --dataset evals/healthcare_tool_calling/datasets/seed_gold.jsonl \
  --catalog instances/heart/functions.json
```

For teacher-generated batches, use the same checker without seed coverage
requirements:

```bash
python3 evals/healthcare_tool_calling/scripts/check_seed_gold.py \
  --dataset evals/healthcare_tool_calling/datasets/teacher_generated_batch_003.jsonl \
  --skip-coverage
```

`--skip-coverage` validates JSONL structure, allowed fields, function names,
argument names, argument types, feature aliases, and conversation-history call
shape. It does not enforce or print seed-gold coverage targets, because a single
teacher batch is not expected to satisfy the full manual seed contract.

## Case Shape

Each dataset row should be a JSON object stored as one line in a JSONL file.
The readiness checker validates this shape before enrichment.

```json
{
  "id": "heart_predict_001",
  "usecase": "heart",
  "scenario": "direct_single_turn",
  "user_input": "Can you predict patient 42?",
  "conversation_history": [],
  "expected_behavior": "tool_call",
  "expected_function_calls": [
    "predict(patient_id=42)"
  ]
}
```

For context-dependent cases, `conversation_history` should match the compact
history shape used in the healthcare prompt, not the frontend chat-message
transport shape:

```json
{
  "id": "heart_prediction_outcome_patient_carryover_001",
  "usecase": "heart",
  "scenario": "parameter_carryover",
  "user_input": "Was the model right for that same patient?",
  "conversation_history": [
    {
      "turn": 1,
      "user_input": "Show me patient 42.",
      "function_calls": [
        "show_one(patient_id=42)"
      ]
    }
  ],
  "expected_behavior": "tool_call",
  "expected_function_calls": [
    "prediction_outcome_patient(patient_id=42)"
  ]
}
```

Turns where no functions were called should still be included when they are part
of the prior conversation:

```json
{
  "turn": 2,
  "user_input": "Thanks, that helps.",
  "function_calls": []
}
```

For ambiguous or unsupported cases, the expected function call list should be
empty and the expected behavior should state why no call is expected:

```json
{
  "id": "heart_ambiguity_001",
  "usecase": "heart",
  "scenario": "missing_required_argument",
  "user_input": "Can you predict this patient?",
  "conversation_history": [],
  "expected_behavior": "no_call_clarify",
  "expected_function_calls": [],
  "target_tools": [
    "predict"
  ]
}
```

When a case has more than one defensible answer, prefer representing multiple
accepted call sets in the reviewed dataset rather than forcing a single
teacher-preferred label.

```json
{
  "id": "heart_risk_001",
  "usecase": "heart",
  "scenario": "multi_tool_request",
  "user_input": "How risky is patient 42, and why?",
  "conversation_history": [],
  "expected_behavior": "tool_call",
  "expected_function_calls": [
    "predict(patient_id=42)",
    "feature_importance_patient(patient_id=42)"
  ],
  "accepted_function_call_sets": [
    [
      "predict(patient_id=42)"
    ]
  ]
}
```

Use `target_tools` only when the scenario is about a supported tool intent even
though no tool call is expected. This is required for
`missing_required_argument` cases because the checker cannot infer the intended
tool from an empty `expected_function_calls` list. Do not include
`target_tools` for ordinary `tool_call` rows; the tools are already encoded in
`expected_function_calls`.

## Scenario Taxonomy

Use scenario tags to make the benchmark diagnostic instead of only reporting one
aggregate score.

- `direct_single_turn`: basic routing and argument extraction from the latest
  user message.
- `paraphrase_or_alias`: user wording must be mapped to canonical tool
  arguments, such as "blood pressure" to the matching heart feature.
- `parameter_carryover`: the user refers to an entity or parameter established
  in prior conversation history.
- `entity_switch_or_correction`: the user corrects or replaces a previously
  mentioned entity, and the latest correction should win.
- `missing_required_argument`: the request matches a tool intent but lacks
  required information, so no tool call should be made.
- `unsupported_intent`: the user asks for a capability outside the available
  function catalog.
- `conflicting_context`: the history contains multiple possible referents or
  otherwise conflicting context.
- `irrelevant_context`: the history contains prior tool calls, but the latest
  user request is self-contained and should not inherit stale intent or
  arguments.
- `multi_tool_request`: the user asks for multiple supported operations in one
  turn.
- `no_tool_needed`: the user response does not require backend function calls.

## Recommended Sample Allocation

For each healthcare tool, manually create a small number of high-quality seed
examples before using a teacher model for expansion:

- 2 direct single-turn examples per tool.
- 1 paraphrase or alias example where the tool has natural user-facing aliases
  (`define_feature`, `what_if`, `count_patients`, `performance_metrics`).
- 1 missing-required-argument example where the tool has required parameters.
- 1 parameter-carryover example where the tool can rely on conversation
  history.
- 1 entity-switch or correction example where the tool uses patient IDs or other
  entity arguments.

Add cross-tool stress cases separately:

- unsupported healthcare or clinical advice requests.
- no-tool-needed turns.
- multi-tool requests.
- conflicting-context cases.
- irrelevant-context cases.

Not every scenario applies equally to every tool. Global tools like
`get_model_description()` or `feature_importance_global()` need fewer
context-carryover cases than patient-level tools like `predict(patient_id=...)`,
`show_one(patient_id=...)`, `feature_importance_patient(patient_id=...)`,
`counterfactual(patient_id=...)`, and `what_if(...)`.

The readiness checker formalizes these minimums:

- every catalog tool needs at least 2 `direct_single_turn` seed cases.
- alias-heavy tools need at least 1 `paraphrase_or_alias` case.
- tools with required arguments need at least 1 `missing_required_argument`
  no-call case.
- patient-level tools need at least 1 `parameter_carryover` case and at least 1
  `entity_switch_or_correction` case.
- cross-tool stress scenarios need at least 2 cases each:
  `unsupported_intent`, `no_tool_needed`, `multi_tool_request`, and
  `conflicting_context`, and `irrelevant_context`.

These are readiness minimums, not final benchmark-size targets. The purpose is
to ensure the manual seed set contains enough high-quality examples to guide
teacher-model expansion. With the current 20-tool heart catalog, these minimums
require 73 reviewed manual seed samples.

### Target Dataset Sizes

Use the following size targets for the first formal benchmark version:

| Dataset stage | Target size | Purpose |
| --- | ---: | --- |
| `seed_gold` | 73 | Manual coverage anchor for every supported tool and scenario. |
| `teacher_generated_raw` | 300-500 | Candidate pool for expansion; not benchmark truth. |
| `reviewed_gold_v1` | 250-300 | Final manually reviewed benchmark for model comparison. |
| optional hidden holdout | 50-100 | Later regression set if the benchmark will be reused often. |

For `reviewed_gold_v1`, aim for approximately 260 reviewed samples:

| Scenario | Seed target | Final v1 target |
| --- | ---: | ---: |
| `direct_single_turn` | 40 | 80 |
| `paraphrase_or_alias` | 4 | 25 |
| `parameter_carryover` | 6 | 30 |
| `entity_switch_or_correction` | 6 | 25 |
| `missing_required_argument` | 7 | 30 |
| `unsupported_intent` | 2 | 15 |
| `no_tool_needed` | 2 | 15 |
| `multi_tool_request` | 2 | 15 |
| `conflicting_context` | 2 | 12 |
| `irrelevant_context` | 2 | 13 |
| **Total** | **73** | **260** |

This size is intentionally moderate. The benchmark evaluates structured
tool-call parsing over a 20-tool catalog, so scenario coverage and annotation
quality matter more than raw volume. A final set below 200 samples makes
per-scenario scores noisy: with only a few cases in a scenario, one failure can
move the result by many percentage points. A first version above roughly 500
reviewed samples is likely to create review burden before the first model run
has shown which scenarios actually need more coverage.

The recommended v1 process is:

1. Finish the 73-row manual `seed_gold`.
2. Generate roughly 300-500 teacher candidate rows.
3. Manually review and filter down to about 260 `reviewed_gold_v1` rows.
4. Run the target models and inspect per-scenario failures.
5. Add targeted v1.1 samples only for unstable or surprising failure areas.

## Metrics

Report metrics by scenario tag and overall. The main metrics are:

- **Schema Validity**: output is valid JSON in the expected response shape, and
  each function call is parseable.
- **Intent Accuracy**: selected the correct function name or function set,
  ignoring arguments.
- **Argument Accuracy**: filled the correct arguments for correctly selected
  tools.
- **Joint Goal Accuracy**: exactly matched the expected tool call set, including
  all required arguments and no hallucinated arguments.
- **State Carryover Accuracy**: for `parameter_carryover` cases, correctly
  reused parameters from conversation history.
- **Correction Accuracy**: for `entity_switch_or_correction` cases, correctly
  applied the user's latest correction instead of stale context.
- **No-Call / Clarification Accuracy**: for ambiguity, unsupported intent, and
  no-tool-needed cases, avoided making an unwarranted tool call.
- **Hallucination Rate**: produced unsupported function names, invalid
  arguments, unsupported enum values, or invented patient IDs/features not
  grounded in the user input or history.
- **Overcall / Undercall Rate**: produced extra unwarranted calls or missed
  required calls in multi-tool and no-call cases.

Joint Goal Accuracy should be the primary strict correctness metric. The other
metrics explain why a model succeeded or failed.

## Review And Versioning Strategy

Reviewed datasets should be versioned together with the tool contract they were
created against. At minimum, record:

- dataset version, such as `healthcare_tool_calling_v1`.
- review status and review date.
- source files used to create the reviewed set.
- checksum or commit reference for `instances/heart/functions.json`.
- any accepted-answer policy for ambiguous examples.

If the heart function catalog changes, the reviewed dataset should either be
revalidated or copied into a new version. A model score is only meaningful
against the tool schema and annotation rules used when the benchmark was
created.
