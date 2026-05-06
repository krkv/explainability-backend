# Healthcare Tool-Calling Evaluation: Gemini vs GPT 5.4 Mini

Date: 2026-05-06

Dataset: `reviewed_gold_v1.jsonl`

Cases: 343

Models compared:

- `gemini-3.1-flash-lite-preview`
- `gpt-5.4-mini`

Artifacts:

- `gemini-3.1-flash-lite-preview/raw_generations.jsonl`
- `gemini-3.1-flash-lite-preview/predictions.jsonl`
- `gemini-3.1-flash-lite-preview/scores.json`
- `gemini-3.1-flash-lite-preview/errors.jsonl`
- `gpt-5.4-mini/raw_generations.jsonl`
- `gpt-5.4-mini/predictions.jsonl`
- `gpt-5.4-mini/scores.json`
- `gpt-5.4-mini/errors.jsonl`

## Summary

Both runs completed cleanly across all 343 frozen dataset cases. Neither model had provider errors or invalid structured response objects. Gemini produced stronger overall tool-calling performance, especially on argument accuracy, parameter carryover, paraphrase handling, and multi-tool requests. GPT 5.4 mini was better on conflicting-context and no-tool/unsupported-intent cases, but was weaker at carrying arguments from conversation history and made several function-call syntax errors.

Overall winner on this benchmark version: `gemini-3.1-flash-lite-preview`.

## Overall Metrics

| Metric | Gemini 3.1 Flash Lite Preview | GPT 5.4 Mini | Delta |
| --- | ---: | ---: | ---: |
| Response schema validity | 1.000 | 1.000 | 0.000 |
| Function-call string validity | 1.000 | 0.980 | +0.020 Gemini |
| Intent accuracy | 0.875 | 0.802 | +0.073 Gemini |
| Argument accuracy | 0.965 | 0.855 | +0.110 Gemini |
| Joint goal accuracy | 0.869 | 0.790 | +0.079 Gemini |
| No-call accuracy | 0.670 | 0.680 | +0.010 GPT |
| Overcall rate | 0.093 | 0.090 | +0.003 GPT |
| Undercall rate | 0.003 | 0.038 | +0.035 Gemini |
| Hallucinated-tool rate | 0.000 | 0.020 | +0.020 Gemini |

Counts:

| Count | Gemini | GPT 5.4 Mini |
| --- | ---: | ---: |
| Cases | 343 | 343 |
| Joint-goal correct | 298 | 271 |
| Intent correct | 300 | 275 |
| Argument slots correct | 273 / 283 | 242 / 283 |
| No-call correct | 65 / 97 | 66 / 97 |
| Valid response schema | 343 / 343 | 343 / 343 |
| Valid function-call strings | 343 / 343 | 336 / 343 |
| Overcalls | 32 | 31 |
| Undercalls | 1 | 13 |
| Hallucinated tools | 0 | 7 |

## Scenario Breakdown

| Scenario | Cases | Gemini JGA | GPT JGA | Better |
| --- | ---: | ---: | ---: | --- |
| `conflicting_context` | 15 | 0.133 | 0.600 | GPT |
| `direct_single_turn` | 98 | 0.939 | 0.888 | Gemini |
| `entity_switch_or_correction` | 38 | 0.921 | 0.895 | Gemini |
| `irrelevant_context` | 14 | 1.000 | 0.929 | Gemini |
| `missing_required_argument` | 34 | 0.824 | 0.500 | Gemini |
| `multi_tool_request` | 25 | 0.960 | 0.840 | Gemini |
| `no_tool_needed` | 23 | 0.826 | 0.913 | GPT |
| `parameter_carryover` | 38 | 0.947 | 0.605 | Gemini |
| `paraphrase_or_alias` | 33 | 0.970 | 0.818 | Gemini |
| `unsupported_intent` | 25 | 0.640 | 0.760 | GPT |

Main scenario observations:

- Gemini is much stronger at using conversation history for argument carryover.
- Gemini is more reliable when the user asks for supported tools using casual wording or aliases.
- GPT 5.4 mini is more conservative in some no-call scenarios, especially `no_tool_needed` and `unsupported_intent`.
- GPT 5.4 mini performs substantially better on `conflicting_context`, where Gemini frequently overcalled.
- Both models are perfect on response schema validity, so differences mostly come from tool choice, argument inference, and valid function-call string construction.

## Tool-Level Notes

Largest Gemini advantages by joint goal accuracy:

| Tool | Cases | Gemini JGA | GPT JGA | Delta |
| --- | ---: | ---: | ---: | ---: |
| `count_patients` | 14 | 1.000 | 0.714 | +0.286 |
| `prediction_outcome_patient` | 14 | 1.000 | 0.786 | +0.214 |
| `feature_importance_patient` | 34 | 0.735 | 0.559 | +0.176 |
| `define_feature` | 24 | 1.000 | 0.833 | +0.167 |
| `show_one` | 35 | 0.943 | 0.800 | +0.143 |
| `predict` | 39 | 0.897 | 0.769 | +0.128 |
| `performance_metrics` | 15 | 0.933 | 0.800 | +0.133 |
| `what_if` | 45 | 0.867 | 0.756 | +0.111 |

GPT 5.4 mini advantages:

| Tool | Cases | Gemini JGA | GPT JGA | Delta |
| --- | ---: | ---: | ---: | ---: |
| `confusion_matrix_stats` | 11 | 0.909 | 1.000 | +0.091 |

Tools with equivalent performance:

- `age_group_performance`
- `available_functions`
- `dataset_summary`
- `feature_importance_global`
- `feature_interactions`
- `get_dataset_description`
- `get_model_description`
- `get_model_parameters`
- `misclassified_cases`
- `show_ids`

## Error Patterns

Gemini:

- 45 scored errors.
- 0 malformed function-call strings.
- 0 hallucinated tools.
- Weakest scenario: `conflicting_context`, where it overcalled in 13 of 15 cases.
- Main failure mode: calling a plausible tool when the expected behavior was no call or clarification.

GPT 5.4 mini:

- 72 scored errors.
- 7 malformed function-call strings.
- 7 hallucinated-tool cases.
- Weakest scenarios: `missing_required_argument` and `parameter_carryover`.
- Main failure modes:
  - missing historical argument carryover;
  - emitting empty calls where a prior patient or feature should have been reused;
  - unquoted string arguments such as `count_patients(count_type=total)`;
  - near-neighbor tool confusion, for example `dataset_summary()` vs `get_dataset_description()`.

## Interpretation

For this healthcare tool-calling setup, Gemini is the stronger default candidate when the assistant is expected to infer arguments from conversation history and produce exact function-call strings. GPT 5.4 mini is somewhat more cautious in no-call scenarios, but this does not compensate for lower argument accuracy and higher undercall rate.

The comparison is based only on tool-calling behavior. Free-form assistant response quality is saved in the prediction artifacts but is not scored in this report.

## Caveats

- GPT 5.4 mini was run with the current backend provider defaults. No explicit reasoning or thinking-effort parameter is configured in the eval runner.
- The benchmark is frozen as `reviewed_gold_v1`, but it is still a project-specific benchmark. Scores should be interpreted as relative performance on this healthcare function-calling contract, not as general model quality.
- Some tools have overlapping semantic boundaries, especially `dataset_summary()` and `get_dataset_description()`. The benchmark intentionally scores exact tool selection against the frozen gold labels.
