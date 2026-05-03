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

This scaffold is intentionally minimal. It does not include teacher generation
scripts, student model runners, scoring scripts, or model adapters yet.

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

The template command writes JSONL rows to stdout. Copy the rows you want into a
draft file such as:

```text
evals/healthcare_tool_calling/datasets/seed_gold_draft.jsonl
```

Then manually review and rewrite the human-facing fields before moving rows
into:

```text
evals/healthcare_tool_calling/datasets/seed_gold.jsonl
```

The most important fields to manually review are:

- `user_input`: make this a realistic user message, not a mechanical prompt.
- `conversation_history`: use compact prior turns matching the healthcare
  prompt context. Each prior turn has `turn`, `user_input`, and
  `function_calls`. Include prior turns even when no functions were called by
  setting `function_calls` to `[]`.
- `expected_function_calls`: verify the selected tools and arguments are the
  intended gold parse.
- `expected_behavior`: use `tool_call`, `no_call_clarify`,
  `no_call_unsupported`, or `no_call_needed`.
- `target_tools`: identify the tool intent for no-call cases where
  `expected_function_calls` is empty.

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
  ],
  "notes": "Direct patient prediction request"
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
  ],
  "target_tools": [
    "prediction_outcome_patient"
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
  "target_tools": [
    "predict",
    "feature_importance_patient"
  ],
  "accepted_function_call_sets": [
    [
      "predict(patient_id=42)"
    ]
  ]
}
```

Use `target_tools` when the scenario is about a tool intent even though no tool
call is expected. This is required for `missing_required_argument` cases because
the checker cannot infer the intended tool from an empty
`expected_function_calls` list.

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
  `conflicting_context`.

These are readiness minimums, not final benchmark-size targets. The purpose is
to ensure the manual seed set contains enough high-quality examples to guide
teacher-model expansion. With the current 20-tool heart catalog, these minimums
require 71 reviewed manual seed samples.

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
