# Opus 4.7 Teacher Prompt: Healthcare Tool-Calling Candidate Generation

You are generating candidate benchmark rows for a healthcare tool-calling evaluation.

The benchmark tests whether an LLM can map a user request plus compact conversation history to the correct backend function calls. The target use case is a heart disease model explanation assistant.

Your output is not final benchmark truth. Your output will be manually reviewed. Generate high-quality candidate rows that follow the same schema and annotation style as the attached reviewed seed set.

## Attached Files To Use

Use these attached files as the complete task context:

1. `functions.json`
   - The authoritative function catalog.
   - Only these functions may appear in `expected_function_calls` or in `conversation_history[].function_calls`.

2. `feature_metadata.json`
   - The authoritative feature metadata.
   - Use it for display names, aliases, categorical labels, and valid feature arguments.

3. `dataset_metadata.json`
   - The high-level dataset description.

4. `seed_gold.jsonl`
   - The reviewed manual seed dataset.
   - Follow its JSONL shape, scenario taxonomy, function-call string style, and compact conversation-history style.

Do not require access to repository code. Do not assume any hidden files or variables.

## Prompt-Visible Dataset Summary

The live assistant sees summary statistics for the loaded dataset, not raw patient rows. Use this summary only for realistic wording and valid feature/value ranges:

```json
{"age":{"count":60.0,"mean":53.6333333333,"std":9.1373078902,"min":29.0,"25%":48.5,"50%":55.0,"75%":59.0,"max":70.0},"sex":{"count":60.0,"mean":0.6833333333,"std":0.4691017983,"min":0.0,"25%":0.0,"50%":1.0,"75%":1.0,"max":1.0},"cp":{"count":60.0,"mean":3.05,"std":0.9987280046,"min":1.0,"25%":2.0,"50%":3.0,"75%":4.0,"max":4.0},"trestbps":{"count":60.0,"mean":129.4166666667,"std":17.1001949307,"min":100.0,"25%":120.0,"50%":128.5,"75%":136.5,"max":178.0},"chol":{"count":60.0,"mean":239.2166666667,"std":46.6403810152,"min":131.0,"25%":208.5,"50%":238.5,"75%":270.25,"max":335.0},"fbs":{"count":60.0,"mean":0.05,"std":0.2197841777,"min":0.0,"25%":0.0,"50%":0.0,"75%":0.0,"max":1.0},"restecg":{"count":60.0,"mean":1.05,"std":0.9987280046,"min":0.0,"25%":0.0,"50%":2.0,"75%":2.0,"max":2.0},"thalch":{"count":60.0,"mean":148.6333333333,"std":25.1537419575,"min":71.0,"25%":130.0,"50%":158.0,"75%":164.25,"max":202.0},"exang":{"count":60.0,"mean":0.25,"std":0.4366668823,"min":0.0,"25%":0.0,"50%":0.0,"75%":0.25,"max":1.0},"oldpeak":{"count":60.0,"mean":1.0316666667,"std":1.120606928,"min":0.0,"25%":0.0,"50%":0.8,"75%":1.6,"max":4.2},"slope":{"count":60.0,"mean":1.6666666667,"std":0.6288721857,"min":1.0,"25%":1.0,"50%":2.0,"75%":2.0,"max":3.0},"ca":{"count":60.0,"mean":0.6666666667,"std":1.0195822785,"min":0.0,"25%":0.0,"50%":0.0,"75%":1.0,"max":3.0},"thal":{"count":60.0,"mean":4.4333333333,"std":1.9077599489,"min":3.0,"25%":3.0,"50%":3.0,"75%":7.0,"max":7.0},"num":{"count":60.0,"mean":0.4666666667,"std":0.5030977486,"min":0.0,"25%":0.0,"50%":0.0,"75%":1.0,"max":1.0}}
```

Use patient IDs from `0` through `59` when a row needs a valid patient ID.

## Output Format

Output exactly 50 JSONL rows.

Each output line must be one compact JSON object. Do not pretty-print multi-line JSON. Do not wrap output in Markdown fences. Do not include commentary before or after the JSONL.

Every row must have exactly these fields unless `target_tools` is needed:

```json
{"id":"heart_teacher_candidate_0001","usecase":"heart","scenario":"direct_single_turn","user_input":"Predict patient 42","conversation_history":[],"expected_behavior":"tool_call","expected_function_calls":["predict(patient_id=42)"]}
```

Allowed fields:

- `id`
- `usecase`
- `scenario`
- `user_input`
- `conversation_history`
- `expected_behavior`
- `expected_function_calls`
- `target_tools`
- `accepted_function_call_sets`

Do not add `notes`, explanations, rationales, comments, provenance fields, or any other metadata.

## Required Field Rules

`usecase` must always be:

```json
"heart"
```

`scenario` must be one of:

- `direct_single_turn`
- `paraphrase_or_alias`
- `parameter_carryover`
- `entity_switch_or_correction`
- `missing_required_argument`
- `unsupported_intent`
- `conflicting_context`
- `multi_tool_request`
- `no_tool_needed`
- `irrelevant_context`

`expected_behavior` must be one of:

- `tool_call`
- `no_call_clarify`
- `no_call_unsupported`
- `no_call_needed`

Use `tool_call` when one or more supported functions should be called.

Use `no_call_clarify` when a supported tool intent is present but required information is missing or context is ambiguous.

Use `no_call_unsupported` when the request asks for something outside the available function catalog, such as diagnosis, treatment planning, medication advice, clinical recommendations, or unsupported reasoning.

Use `no_call_needed` when the latest user message is conversational or does not need a backend function call.

`expected_function_calls` must be:

- a list of function-call strings for `tool_call` rows
- an empty list for no-call rows

Function-call strings must use keyword arguments only:

```text
predict(patient_id=42)
what_if(patient_id=42, feature='trestbps', value_change=-15)
performance_metrics(metrics=['accuracy'])
```

Do not use positional arguments:

```text
prediction_outcome_patient(42)
```

`conversation_history` must use the compact shape used by the live healthcare prompt:

```json
[{"turn":1,"user_input":"Show patient 42","function_calls":["show_one(patient_id=42)"]}]
```

Include turns where no function was called when relevant:

```json
[{"turn":1,"user_input":"Thanks","function_calls":[]}]
```

Do not include assistant free-form responses. Do not use chat-style `role` and `content` messages.

Use `target_tools` only when `expected_function_calls` is empty but the row still has a supported tool intent, especially `missing_required_argument` and `conflicting_context`.

## Function Argument Guidance

Only use functions from `functions.json`.

Use the feature metadata for feature aliases and categorical labels. Users may say feature aliases such as "blood pressure", "chest pain", "exercise angina", "heart rate", "cholesterol", "ST depression", or display names such as "Resting Blood Pressure". In expected calls, either canonical names or supported aliases are acceptable if they match `feature_metadata.json`.

For `what_if`:

- continuous features use a numeric delta in `value_change`, such as `-10`, `15`, or `30`
- categorical features use a category label or code as a string in `value_change`, such as `'Female'`, `'Male'`, `'Typical angina'`, or `'No'`
- do not encode categorical target values as numeric deltas

For `count_patients`:

- "how many patients total" maps to `count_patients(count_type='total')`
- "how many predicted sick / heart disease / positive" maps to `count_patients(count_type='positive_predicted')`
- "how many predicted healthy / no heart disease / negative" maps to `count_patients(count_type='negative_predicted')`

For patient-level tools, use explicit or carried-over valid patient IDs from 0 through 59.

## Scenario Guidance

Generate varied, realistic user messages. Include concise, casual, imperfect, and laconic user wording where the intent is still human-resolvable.

`direct_single_turn`:

- Current user message alone is enough.
- Use empty `conversation_history`.

`paraphrase_or_alias`:

- User wording should use aliases, shorthand, or natural descriptions rather than exact tool names.

`parameter_carryover`:

- Current user refers to a patient, feature, or previous target indirectly.
- Prior `conversation_history` must contain the argument needed for the expected call.

`entity_switch_or_correction`:

- History contains an earlier entity or parameter.
- Latest user message corrects or replaces it.
- Latest user message wins.

`missing_required_argument`:

- User intent maps to a supported tool, but required information is missing.
- Expected calls must be empty.
- Include `target_tools`.

`unsupported_intent`:

- User asks for something outside the function catalog.
- Expected calls must be empty.
- Do not include `target_tools` unless a supported tool intent is clear and ambiguity is the reason for no-call.

`conflicting_context`:

- History contains multiple possible referents or missing referents.
- Latest request cannot be grounded safely.
- Expected calls must be empty and `expected_behavior` should be `no_call_clarify`.
- Include `target_tools`.

`multi_tool_request`:

- Latest user explicitly asks for two or more supported actions.
- Include all required function calls in the expected order.

`no_tool_needed`:

- Latest user message does not need backend work.
- Expected calls must be empty.

`irrelevant_context`:

- History contains prior tool calls.
- Latest user message is self-contained and should ignore stale history.

## Desired Distribution For This 50-Row Batch

Generate a balanced batch using approximately this distribution:

- `direct_single_turn`: 10 rows
- `paraphrase_or_alias`: 5 rows
- `parameter_carryover`: 6 rows
- `entity_switch_or_correction`: 6 rows
- `missing_required_argument`: 6 rows
- `unsupported_intent`: 4 rows
- `no_tool_needed`: 4 rows
- `multi_tool_request`: 4 rows
- `conflicting_context`: 3 rows
- `irrelevant_context`: 2 rows

Avoid near-duplicates of the attached `seed_gold.jsonl`. Do not copy seed rows with only patient IDs changed. Vary wording, tools, argument combinations, and conversation-history shapes.

## Quality Bar

A row is good only if a careful human reviewer can agree that the expected function call set follows from the user input and compact history.

Do not generate intentionally invalid function calls. Do not invent functions. Do not invent unsupported feature names. Do not hallucinate unsupported patient IDs. Do not produce rows where the right answer depends on information not present in the row or attached files.

