# Healthcare Tool-Calling Evaluation: Four-Model Comparison

Date: 2026-05-07

Dataset: `reviewed_gold_v1.jsonl`

Cases: 343

Full benchmark models:

- `gemma-4`
- `gemini-3.1-flash-lite-preview`
- `gpt-5.4-mini`
- `kimi-k2.5`

Smoke-tested but excluded from the full benchmark:

- `qwen3.6-flash` (`qwen/qwen3.6-flash` through OpenRouter)

Artifacts:

- `gemma-4/raw_generations.jsonl`
- `gemma-4/predictions.jsonl`
- `gemma-4/scores.json`
- `gemma-4/errors.jsonl`
- `gemini-3.1-flash-lite-preview/raw_generations.jsonl`
- `gemini-3.1-flash-lite-preview/predictions.jsonl`
- `gemini-3.1-flash-lite-preview/scores.json`
- `gemini-3.1-flash-lite-preview/errors.jsonl`
- `gpt-5.4-mini/raw_generations.jsonl`
- `gpt-5.4-mini/predictions.jsonl`
- `gpt-5.4-mini/scores.json`
- `gpt-5.4-mini/errors.jsonl`
- `kimi-k2.5/raw_generations.jsonl`
- `kimi-k2.5/predictions.jsonl`
- `kimi-k2.5/scores.json`
- `kimi-k2.5/errors.jsonl`
- `qwen3.6-flash/raw_generations.jsonl`
- `qwen3.6-flash/predictions.jsonl`

## Summary

`gemma-4` is the strongest full-run model on this benchmark version. It has the highest joint goal accuracy, highest intent accuracy, strongest no-call accuracy among the top two models, fewer overcalls than Gemini, and no malformed function-call strings or hallucinated tools.

`gemini-3.1-flash-lite-preview` remains a close second. It has the best argument accuracy by a small margin and is especially strong on multi-tool requests, irrelevant context, feature-importance patient calls, and prediction-outcome calls. Its main weakness is overcalling in ambiguous conflicting-context cases.

`gpt-5.4-mini` is materially weaker overall than Gemma and Gemini, but it is better on some conservative/no-call scenarios, especially `conflicting_context` and `unsupported_intent`. Its main risks are lower argument accuracy, undercalling, malformed function-call strings, and hallucinated tools.

`kimi-k2.5` is not competitive as a default tool-calling model in this setup. It performs well on `no_tool_needed`, but has the lowest response schema validity, function-call string validity, intent accuracy, argument accuracy, and joint goal accuracy among the four full runs.

Overall recommendation for the current chat application: use `gemma-4` as the leading candidate from these runs, with Gemini as the closest alternative. GPT 5.4 Mini may still be useful as a comparison point for conservative no-call behavior, but it is not the best default under this exact function-calling contract. Kimi should not be selected without prompt or provider changes.

## Overall Metrics

| Metric | Gemma 4 | Gemini 3.1 Flash Lite Preview | GPT 5.4 Mini | Kimi K2.5 | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| Response schema validity | 1.000 | 1.000 | 1.000 | 0.985 | Gemma / Gemini / GPT |
| Function-call string validity | 1.000 | 1.000 | 0.980 | 0.971 | Gemma / Gemini |
| Intent accuracy | 0.898 | 0.875 | 0.802 | 0.729 | Gemma |
| Argument accuracy | 0.961 | 0.965 | 0.855 | 0.731 | Gemini |
| Joint goal accuracy | 0.895 | 0.869 | 0.790 | 0.720 | Gemma |
| No-call accuracy | 0.753 | 0.670 | 0.680 | 0.691 | Gemma |
| Overcall rate | 0.070 | 0.093 | 0.090 | 0.079 | Gemma |
| Undercall rate | 0.003 | 0.003 | 0.038 | 0.096 | Gemma / Gemini |
| Hallucinated-tool rate | 0.000 | 0.000 | 0.020 | 0.015 | Gemma / Gemini |

Counts:

| Count | Gemma 4 | Gemini 3.1 | GPT 5.4 Mini | Kimi K2.5 |
| --- | ---: | ---: | ---: | ---: |
| Cases | 343 | 343 | 343 | 343 |
| Joint-goal correct | 307 | 298 | 271 | 247 |
| Intent correct | 308 | 300 | 275 | 250 |
| Argument slots correct | 272 / 283 | 273 / 283 | 242 / 283 | 207 / 283 |
| No-call correct | 73 / 97 | 65 / 97 | 66 / 97 | 67 / 97 |
| Valid function-call strings | 343 / 343 | 343 / 343 | 336 / 343 | 333 / 343 |
| Overcalls | 24 | 32 | 31 | 27 |
| Undercalls | 1 | 1 | 13 | 33 |
| Hallucinated tools | 0 | 0 | 7 | 5 |
| Scored errors | 36 | 45 | 72 | 96 |

## Scenario Breakdown

| Scenario | Cases | Gemma 4 JGA | Gemini JGA | GPT JGA | Kimi JGA | Best |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `conflicting_context` | 15 | 0.200 | 0.133 | 0.600 | 0.400 | GPT |
| `direct_single_turn` | 98 | 0.949 | 0.939 | 0.888 | 0.704 | Gemma |
| `entity_switch_or_correction` | 38 | 0.974 | 0.921 | 0.895 | 0.737 | Gemma |
| `irrelevant_context` | 14 | 0.929 | 1.000 | 0.929 | 0.929 | Gemini |
| `missing_required_argument` | 34 | 0.912 | 0.824 | 0.500 | 0.647 | Gemma |
| `multi_tool_request` | 25 | 0.920 | 0.960 | 0.840 | 0.840 | Gemini |
| `no_tool_needed` | 23 | 0.913 | 0.826 | 0.913 | 1.000 | Kimi |
| `parameter_carryover` | 38 | 0.947 | 0.947 | 0.605 | 0.632 | Gemma / Gemini |
| `paraphrase_or_alias` | 33 | 0.970 | 0.970 | 0.818 | 0.758 | Gemma / Gemini |
| `unsupported_intent` | 25 | 0.720 | 0.640 | 0.760 | 0.640 | GPT |

Main scenario observations:

- Gemma is the best overall scenario generalist. It wins or ties on direct single-turn, entity switch/correction, missing required arguments, parameter carryover, and paraphrase/alias handling.
- Gemini is strongest on multi-tool requests and irrelevant-context cases. It is also tied with Gemma on parameter carryover and paraphrase/alias cases.
- GPT is the best model for conflicting-context and unsupported-intent cases. This points to stronger conservatism when the request should not produce a straightforward tool call.
- Kimi is best only on `no_tool_needed`. This is useful signal about conservatism, but it does not compensate for weak tool-call accuracy in supported-request scenarios.
- `conflicting_context` is the hardest scenario for the top two models. Gemma and Gemini frequently choose a plausible referent instead of clarifying.

## Tool-Level Notes

Tool-level winners:

| Tool | Cases | Best model(s) | Best JGA |
| --- | ---: | --- | ---: |
| `age_group_performance` | 6 | All models | 1.000 |
| `available_functions` | 2 | All models | 1.000 |
| `confusion_matrix_stats` | 11 | GPT | 1.000 |
| `count_patients` | 14 | Gemma / Gemini | 1.000 |
| `counterfactual` | 32 | Gemma | 0.875 |
| `dataset_summary` | 6 | Gemma / Gemini / GPT | 0.833 |
| `define_feature` | 24 | Gemma / Gemini | 1.000 |
| `feature_importance_global` | 9 | Gemma / Gemini / GPT | 1.000 |
| `feature_importance_patient` | 34 | Gemini | 0.735 |
| `feature_interactions` | 4 | Gemma / Gemini / GPT | 1.000 |
| `get_dataset_description` | 5 | Gemma | 1.000 |
| `get_model_description` | 4 | All models | 1.000 |
| `get_model_parameters` | 5 | All models | 0.800 |
| `misclassified_cases` | 5 | Gemma / Gemini / GPT | 1.000 |
| `performance_metrics` | 15 | Gemma / Gemini | 0.933 |
| `predict` | 39 | Gemma | 0.949 |
| `prediction_outcome_patient` | 14 | Gemini | 1.000 |
| `show_ids` | 7 | Gemma | 1.000 |
| `show_one` | 35 | Gemma / Gemini | 0.943 |
| `what_if` | 45 | Gemma | 0.911 |

Highest-signal tool observations:

- Gemma has the best spread across high-volume tools: `predict`, `what_if`, `show_one`, `counterfactual`, and `count_patients`.
- Gemini is strongest on explanation-adjacent patient calls, especially `feature_importance_patient` and `prediction_outcome_patient`.
- GPT has a narrow but real advantage on `confusion_matrix_stats`, and it remains perfect on several low-cardinality global tools.
- Kimi's largest gaps are on high-volume, chat-relevant tools: `predict`, `what_if`, `show_one`, `counterfactual`, and `feature_importance_patient`.
- The weakest tool for the top models is `feature_importance_patient`, suggesting the benchmark has difficult boundaries around "why", "explain", prediction lookup, and patient-specific explanation.

## Error Patterns

Gemma:

- 36 scored errors.
- 0 malformed function-call strings.
- 0 hallucinated tools.
- 24 overcalls and 1 undercall.
- Main weakness: conflicting-context ambiguity. Gemma often chooses a plausible prior referent instead of asking for clarification.
- Main strength: broad accuracy across supported tool-call scenarios with strong no-call behavior for a high-performing model.

Gemini:

- 45 scored errors.
- 0 malformed function-call strings.
- 0 hallucinated tools.
- 32 overcalls and 1 undercall.
- Main weakness: overcalling, especially in `conflicting_context`.
- Main strength: argument accuracy and multi-tool handling. Gemini is very reliable when the expected behavior is an actual supported function call.

GPT 5.4 Mini:

- 72 scored errors.
- 7 malformed function-call strings.
- 7 hallucinated-tool cases.
- 31 overcalls and 13 undercalls.
- Main weakness: argument carryover and strict call-string formation.
- Main strength: conservative behavior in ambiguous or unsupported requests.

Kimi K2.5:

- 96 scored errors.
- 10 invalid response or function-call string cases by count difference.
- 5 hallucinated-tool cases.
- 27 overcalls and 33 undercalls.
- Main weakness: supported-request execution, especially argument extraction and undercalling.
- Main strength: avoiding calls when no tool is needed.

## Case-Level Disagreement

Across the four full benchmark runs:

- 215 / 343 cases were solved by all four models.
- 14 / 343 cases were missed by all four models.
- Gemma had 7 cases where it was the only successful model.
- Gemini had 3 cases where it was the only successful model.
- GPT had 10 cases where it was the only successful model.
- Kimi had 4 cases where it was the only successful model.

The all-model-failure set is useful for benchmark review. It likely contains a mix of genuinely hard examples, ambiguous gold labels, and tool-boundary cases such as dataset summary versus dataset description, model-parameter intent, and multi-tool phrasing. These should be inspected before using the benchmark as a long-term regression suite.

## Qwen 3.6 Flash Smoke Test

`qwen3.6-flash` was added as a production-oriented chat candidate because `qwen/qwen3.6-flash` is positioned as a fast OpenRouter model. It was run only on a 10-case smoke batch and was excluded from the full benchmark.

The smoke result was poor enough that a full run is not recommended under the current configuration. Only 3 / 10 prediction rows were valid parsed assistant responses, and 7 / 10 failed with `missing_function_calls`. The dominant failure pattern was not ordinary wrong-tool selection: Qwen often returned the JSON schema itself, or a schema-shaped object with calls placed under `properties.function_calls`, instead of returning an instance like `{"function_calls": [...], "freeform_response": "..."}`. One row returned `{}`.

This is best interpreted as a response-contract failure for this chat application. Since the same OpenRouter structured-output path produced usable full benchmark runs for Gemma and Kimi, Qwen's failure to follow the required schema is part of its suitability assessment. It should be documented as excluded after smoke testing rather than included as a full-run contender.

## Interpretation

For the current healthcare chat application, the model must do more than understand the user's likely intent. It must produce a strict JSON object containing exact function-call strings. On that contract, the ranking is:

1. `gemma-4`
2. `gemini-3.1-flash-lite-preview`
3. `gpt-5.4-mini`
4. `kimi-k2.5`

Gemma is the best default candidate because it combines high joint accuracy, good no-call behavior, low overcall rate, and perfect output-contract validity. Gemini is close and may be preferable if the product prioritizes multi-tool execution and patient-explanation calls over conservative ambiguity handling. GPT is useful as a reference for safer no-call behavior but needs improvement on argument carryover and call-string syntax before it can compete as the default. Kimi is too unreliable for strict healthcare tool routing in this setup.

The strongest product-level takeaway is that "chat model quality" here is mostly about stateful tool routing, not open-ended answer quality. The highest-risk failure modes are:

- choosing a plausible referent when the context is ambiguous;
- missing arguments that should be carried over from history;
- calling a neighboring tool with similar semantics;
- returning malformed function-call strings;
- failing the required JSON response contract.

## Caveats

- This benchmark evaluates only tool-calling behavior. Free-form response quality is saved in artifacts but is not scored here.
- The benchmark is project-specific and tied to the current heart healthcare function catalog.
- Some tool boundaries are semantically close, especially `dataset_summary()` versus `get_dataset_description()` and prediction/explanation combinations.
- The Qwen result is a 10-case smoke test, not a full benchmark score. Its exclusion is based on repeated schema-contract failures in that smoke batch.
- Scores should be interpreted as relative performance under the current prompt, response schema, provider path, and scoring policy, not as general model quality.

## Recommended Next Steps

- Use `gemma-4` and Gemini as the main candidates in product-facing discussion.
- Keep GPT in the report as the conservative/no-call reference model.
- Exclude Qwen from the full table, but keep the smoke-test paragraph because it is relevant to chat-app model selection.
- Review the 14 all-model-failure cases before treating this dataset as a stable regression benchmark.
- Add a future report dimension for latency and cost if these models are being evaluated for production routing, not only correctness.
