# Healthcare Tool-Calling Evaluation: Six-Model Comparison

Date: 2026-05-07

Dataset: `reviewed_gold_v1.jsonl`

Cases: 343

Full benchmark models:

- `gemma-4-moe`
- `gemma-4`
- `gemini-3.1-flash-lite-preview`
- `deepseek-v4-flash`
- `gpt-5.4-mini`
- `kimi-k2.5`

Smoke-tested but excluded from the full benchmark:

- `qwen3.6-flash` (`qwen/qwen3.6-flash` through OpenRouter)

## Summary

`gemma-4-moe` is the new top model by strict joint goal accuracy. It narrowly beats `gemma-4` overall, has the strongest no-call accuracy by a large margin, the lowest overcall rate, and the fastest latency profile among the models with latency summarized in this report. Its main tradeoff is weaker argument accuracy than `gemma-4` and Gemini, especially on parameter carryover and paraphrase/alias cases.

`gemma-4` is effectively tied with `gemma-4-moe` on overall accuracy and remains the better model when the user request requires precise argument extraction, state carryover, aliases, or high-volume patient-level calls. It has perfect function-call string validity and zero hallucinated tools, but it is much more eager to call tools in ambiguous contexts.

`gemini-3.1-flash-lite-preview` is the third-best model. It has the best argument accuracy and remains especially strong on multi-tool requests, irrelevant context, parameter carryover, paraphrase/alias handling, and prediction-outcome calls. Its main weakness is overcalling when context is ambiguous.

`deepseek-v4-flash` lands in fourth place. It clearly beats GPT 5.4 Mini and Kimi K2.5 overall, follows the required JSON response schema on every case, and has useful median latency. Its weaknesses are missing-required-argument cases, parameter carryover, occasional malformed function-call strings, and extra neighboring calls.

`gpt-5.4-mini` remains useful as a conservative reference model because it performs well on `conflicting_context` and best on `unsupported_intent`, but it is weaker overall than both Gemma variants, Gemini, and DeepSeek.

`kimi-k2.5` is not competitive as a default tool-calling model in this setup. It performs well on `no_tool_needed`, but has the lowest response schema validity, function-call string validity, intent accuracy, argument accuracy, and joint goal accuracy among the six full runs.

Overall recommendation for the current chat application: treat `gemma-4-moe` and `gemma-4` as the two leading candidates. Prefer `gemma-4-moe` when no-call discipline, ambiguity handling, and latency matter most. Prefer `gemma-4` when conversational carryover, argument precision, and patient-level tool accuracy matter most. Gemini remains a strong high-accuracy alternative, while DeepSeek is a viable lower-cost or provider-diversity candidate if its long-tail latency is acceptable.

## Overall Metrics

| Metric | Gemma 4 MoE | Gemma 4 | Gemini 3.1 | DeepSeek V4 Flash | GPT 5.4 Mini | Kimi K2.5 | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Response schema validity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.985 | Gemma MoE / Gemma / Gemini / DeepSeek / GPT |
| Function-call string validity | 0.997 | 1.000 | 1.000 | 0.983 | 0.980 | 0.971 | Gemma / Gemini |
| Intent accuracy | 0.898 | 0.898 | 0.875 | 0.837 | 0.802 | 0.729 | Gemma MoE / Gemma |
| Argument accuracy | 0.908 | 0.961 | 0.965 | 0.898 | 0.855 | 0.731 | Gemini |
| Joint goal accuracy | 0.898 | 0.895 | 0.869 | 0.828 | 0.790 | 0.720 | Gemma MoE |
| No-call accuracy | 0.918 | 0.753 | 0.670 | 0.711 | 0.680 | 0.691 | Gemma MoE |
| Overcall rate | 0.023 | 0.070 | 0.093 | 0.082 | 0.090 | 0.079 | Gemma MoE |
| Undercall rate | 0.058 | 0.003 | 0.003 | 0.017 | 0.038 | 0.096 | Gemma / Gemini |
| Hallucinated-tool rate | 0.003 | 0.000 | 0.000 | 0.017 | 0.020 | 0.015 | Gemma / Gemini |

Counts:

| Count | Gemma 4 MoE | Gemma 4 | Gemini 3.1 | DeepSeek V4 Flash | GPT 5.4 Mini | Kimi K2.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cases | 343 | 343 | 343 | 343 | 343 | 343 |
| Joint-goal correct | 308 | 307 | 298 | 284 | 271 | 247 |
| Intent correct | 308 | 308 | 300 | 287 | 275 | 250 |
| Argument slots correct | 257 / 283 | 272 / 283 | 273 / 283 | 254 / 283 | 242 / 283 | 207 / 283 |
| No-call correct | 89 / 97 | 73 / 97 | 65 / 97 | 69 / 97 | 66 / 97 | 67 / 97 |
| Valid function-call strings | 342 / 343 | 343 / 343 | 343 / 343 | 337 / 343 | 336 / 343 | 333 / 343 |
| Overcalls | 8 | 24 | 32 | 28 | 31 | 27 |
| Undercalls | 20 | 1 | 1 | 6 | 13 | 33 |
| Hallucinated tools | 1 | 0 | 0 | 6 | 7 | 5 |
| Scored errors | 35 | 36 | 45 | 59 | 72 | 96 |

## Scenario Breakdown

| Scenario | Cases | Gemma MoE | Gemma | Gemini | DeepSeek | GPT | Kimi | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `conflicting_context` | 15 | 0.867 | 0.200 | 0.133 | 0.533 | 0.600 | 0.400 | Gemma MoE |
| `direct_single_turn` | 98 | 0.939 | 0.949 | 0.939 | 0.867 | 0.888 | 0.704 | Gemma |
| `entity_switch_or_correction` | 38 | 0.895 | 0.974 | 0.921 | 0.947 | 0.895 | 0.737 | Gemma |
| `irrelevant_context` | 14 | 1.000 | 0.929 | 1.000 | 0.929 | 0.929 | 0.929 | Gemma MoE / Gemini |
| `missing_required_argument` | 34 | 0.912 | 0.912 | 0.824 | 0.618 | 0.500 | 0.647 | Gemma MoE / Gemma |
| `multi_tool_request` | 25 | 0.960 | 0.920 | 0.960 | 0.920 | 0.840 | 0.840 | Gemma MoE / Gemini |
| `no_tool_needed` | 23 | 0.957 | 0.913 | 0.826 | 0.957 | 0.913 | 1.000 | Kimi |
| `parameter_carryover` | 38 | 0.711 | 0.947 | 0.947 | 0.737 | 0.605 | 0.632 | Gemma / Gemini |
| `paraphrase_or_alias` | 33 | 0.848 | 0.970 | 0.970 | 0.909 | 0.818 | 0.758 | Gemma / Gemini |
| `unsupported_intent` | 25 | 0.920 | 0.720 | 0.640 | 0.720 | 0.760 | 0.640 | Gemma MoE |

Main scenario observations:

- Gemma MoE is dramatically better than the earlier leaders on no-call-heavy scenarios: `conflicting_context`, `unsupported_intent`, and `no_tool_needed`.
- Gemma and Gemini remain the best models for stateful supported requests, especially `parameter_carryover` and `paraphrase_or_alias`.
- Gemma is strongest on direct single-turn and entity switch/correction.
- Gemini ties Gemma MoE on multi-tool requests and irrelevant context.
- DeepSeek is a meaningful middle candidate but does not lead any major supported-tool scenario except ties on simple/global tools.
- GPT's previous advantage on ambiguous and unsupported requests is now superseded by Gemma MoE.

## Tool-Level Notes

Tool-level winners:

| Tool | Cases | Best model(s) | Best JGA |
| --- | ---: | --- | ---: |
| `age_group_performance` | 6 | All models | 1.000 |
| `available_functions` | 2 | All models | 1.000 |
| `confusion_matrix_stats` | 11 | GPT | 1.000 |
| `count_patients` | 14 | Gemma / Gemini | 1.000 |
| `counterfactual` | 32 | Gemma MoE / Gemma | 0.875 |
| `dataset_summary` | 6 | Gemma MoE / Gemma / Gemini / GPT | 0.833 |
| `define_feature` | 24 | Gemma / Gemini | 1.000 |
| `feature_importance_global` | 9 | Gemma MoE / Gemma / Gemini / DeepSeek / GPT | 1.000 |
| `feature_importance_patient` | 34 | Gemma MoE | 0.824 |
| `feature_interactions` | 4 | Gemma MoE / Gemma / Gemini / DeepSeek / GPT | 1.000 |
| `get_dataset_description` | 5 | Gemma | 1.000 |
| `get_model_description` | 4 | Gemma MoE / Gemma / Gemini / GPT / Kimi | 1.000 |
| `get_model_parameters` | 5 | All models | 0.800 |
| `misclassified_cases` | 5 | Gemma MoE / Gemma / Gemini / DeepSeek / GPT | 1.000 |
| `performance_metrics` | 15 | Gemma MoE / Gemma / Gemini / DeepSeek | 0.933 |
| `predict` | 39 | Gemma | 0.949 |
| `prediction_outcome_patient` | 14 | Gemini | 1.000 |
| `show_ids` | 7 | Gemma MoE / Gemma | 1.000 |
| `show_one` | 35 | Gemma / Gemini | 0.943 |
| `what_if` | 45 | Gemma MoE | 0.956 |

Highest-signal tool observations:

- Gemma MoE is strongest on `what_if` and `feature_importance_patient`, two important chat-facing explanation/simulation tools.
- Gemma remains strongest on `predict`, `show_one`, `count_patients`, `define_feature`, and `get_dataset_description`.
- Gemini remains best on `prediction_outcome_patient` and is tied for several global/simple tools.
- DeepSeek is competitive on global/simple tools and ties the leaders on `performance_metrics`, `feature_importance_global`, `feature_interactions`, and `misclassified_cases`.
- GPT has a narrow advantage on `confusion_matrix_stats`.
- Kimi's largest gaps remain on high-volume, chat-relevant tools.

## Error Patterns

Gemma 4 MoE:

- 35 scored errors.
- 1 malformed function-call string case.
- 1 hallucinated-tool case.
- 8 overcalls and 20 undercalls.
- Main weakness: conservative undercalling in `parameter_carryover` and `paraphrase_or_alias` cases.
- Main strength: best no-call behavior, best ambiguity handling, best unsupported-intent behavior, excellent latency, and top overall joint goal accuracy.

Gemma 4:

- 36 scored errors.
- 0 malformed function-call strings.
- 0 hallucinated tools.
- 24 overcalls and 1 undercall.
- Main weakness: conflicting-context ambiguity. Gemma often chooses a plausible prior referent instead of asking for clarification.
- Main strength: broad accuracy across supported tool-call scenarios, excellent argument accuracy, perfect output-contract validity.

Gemini:

- 45 scored errors.
- 0 malformed function-call strings.
- 0 hallucinated tools.
- 32 overcalls and 1 undercall.
- Main weakness: overcalling, especially in `conflicting_context`.
- Main strength: best argument accuracy and strong multi-tool/stateful supported-tool behavior.

DeepSeek V4 Flash:

- 59 scored errors.
- 6 malformed function-call strings.
- 6 hallucinated-tool cases.
- 28 overcalls and 6 undercalls.
- Main weaknesses: missing-required-argument cases, parameter carryover, and adding neighboring calls around patient explanation or dataset requests.
- Main strength: strong strict-schema adherence and useful median latency.

GPT 5.4 Mini:

- 72 scored errors.
- 7 malformed function-call strings.
- 7 hallucinated-tool cases.
- 31 overcalls and 13 undercalls.
- Main weakness: argument carryover and strict call-string formation.
- Main strength: still relatively conservative, though Gemma MoE now dominates it on no-call-heavy scenarios.

Kimi K2.5:

- 96 scored errors.
- 10 invalid response or function-call string cases by count difference.
- 5 hallucinated-tool cases.
- 27 overcalls and 33 undercalls.
- Main weakness: supported-request execution, especially argument extraction and undercalling.
- Main strength: avoiding calls when no tool is needed.

## Latency Notes

Latency was summarized for the newest fast candidates:

| Latency metric | Gemma 4 MoE | DeepSeek V4 Flash |
| --- | ---: | ---: |
| Min | 0.7s | 1.2s |
| Median | 2.0s | 3.5s |
| P90 | 4.0s | 12.2s |
| P95 | 5.3s | 18.2s |
| Max | 32.7s | 134.2s |
| Average | 2.6s | 6.1s |

Gemma MoE has a substantially better latency profile than DeepSeek in this run. Its median and tail latencies are more suitable for an interactive chat application.

## Case-Level Disagreement

Across the six full benchmark runs:

- 197 / 343 cases were solved by all six models.
- 6 / 343 cases were missed by all six models.
- Gemma MoE had 4 cases where it was the only successful model.
- Gemma had 1 case where it was the only successful model.
- Gemini had 1 case where it was the only successful model.
- DeepSeek had 1 case where it was the only successful model.
- GPT had 1 case where it was the only successful model.
- Kimi had 3 cases where it was the only successful model.

The all-model-failure set is useful for benchmark review. It likely contains a mix of genuinely hard examples, ambiguous gold labels, and tool-boundary cases such as dataset summary versus dataset description, model-parameter intent, and multi-tool phrasing.

## Qwen 3.6 Flash Smoke Test

`qwen3.6-flash` was added as a production-oriented chat candidate because `qwen/qwen3.6-flash` is positioned as a fast OpenRouter model. It was run only on a 10-case smoke batch and was excluded from the full benchmark.

The smoke result was poor enough that a full run is not recommended under the current configuration. Only 3 / 10 prediction rows were valid parsed assistant responses, and 7 / 10 failed with `missing_function_calls`. The dominant failure pattern was not ordinary wrong-tool selection: Qwen often returned the JSON schema itself, or a schema-shaped object with calls placed under `properties.function_calls`, instead of returning an instance like `{"function_calls": [...], "freeform_response": "..."}`. One row returned `{}`.

This is best interpreted as a response-contract failure for this chat application. Since the same OpenRouter structured-output path produced usable full benchmark runs for Gemma, Gemma MoE, Kimi, and DeepSeek, Qwen's failure to follow the required schema is part of its suitability assessment. It should be documented as excluded after smoke testing rather than included as a full-run contender.

## Interpretation

For the current healthcare chat application, the model must do more than understand the user's likely intent. It must produce a strict JSON object containing exact function-call strings. On that contract, the ranking is:

1. `gemma-4-moe`
2. `gemma-4`
3. `gemini-3.1-flash-lite-preview`
4. `deepseek-v4-flash`
5. `gpt-5.4-mini`
6. `kimi-k2.5`

Gemma MoE is the best default candidate if the product prioritizes safe chat behavior: ambiguity handling, unsupported-intent refusal, low overcall rate, and latency. Gemma is the better candidate if the product prioritizes stateful supported-tool execution and argument precision. Gemini is still a strong high-accuracy alternative, especially for carryover and multi-tool workflows. DeepSeek is viable as a lower-ranked fast/provider-diverse option, but trails the top three on correctness. GPT and Kimi should not be selected as defaults under the current strict healthcare tool-calling contract.

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
- Gemma MoE's top overall score is driven partly by much stronger no-call behavior. It is not uniformly better than Gemma or Gemini on supported-tool execution.
- The Qwen result is a 10-case smoke test, not a full benchmark score. Its exclusion is based on repeated schema-contract failures in that smoke batch.
- Scores should be interpreted as relative performance under the current prompt, response schema, provider path, and scoring policy, not as general model quality.

## Recommended Next Steps

- Use `gemma-4-moe`, `gemma-4`, and Gemini as the main candidates in product-facing discussion.
- Treat `gemma-4-moe` as the likely best default for the chat application, but inspect carryover misses before final production selection.
- Consider `gemma-4` as the stronger option for stateful supported-tool workflows.
- Keep DeepSeek as a viable secondary candidate if provider diversity, cost, or latency constraints matter.
- Exclude Qwen from the full table, but keep the smoke-test paragraph because it is relevant to chat-app model selection.
- Review the 6 all-model-failure cases before treating this dataset as a stable regression benchmark.
- Add cost data if these models are being evaluated for production routing, not only correctness.
