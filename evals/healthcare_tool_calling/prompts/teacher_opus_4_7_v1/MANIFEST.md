# Teacher Prompt Pack: `teacher_opus_4_7_v1`

This folder is a frozen, portable prompt pack for generating healthcare
tool-calling candidate data with Opus 4.7 outside this repository.

## Contents

- `generation_prompt.md`: standalone prompt for generating candidate JSONL rows.
- `review_prompt.md`: standalone prompt for reviewing generated candidate rows.
- `context/functions.json`: frozen copy of the heart function catalog.
- `context/feature_metadata.json`: frozen copy of feature aliases, display names,
  descriptions, and categorical labels.
- `context/dataset_metadata.json`: frozen copy of the dataset description.
- `context/seed_gold.jsonl`: frozen copy of the reviewed manual seed set.

## Usage

For candidate generation, open `generation_prompt.md` in Opus 4.7 and attach:

- `context/functions.json`
- `context/feature_metadata.json`
- `context/dataset_metadata.json`
- `context/seed_gold.jsonl`

Save the model output as:

```text
evals/healthcare_tool_calling/datasets/teacher_generated_raw.jsonl
```

For candidate review, open `review_prompt.md` in Opus 4.7 and attach the same
context files plus the generated candidate file.

## Versioning Rule

Do not edit files inside this folder after using the pack for generation.
If the prompt, seed dataset, function catalog, or metadata changes, create a new
folder such as:

```text
teacher_opus_4_7_v2/
```

This keeps generated data traceable to the exact prompt and context used.

## Checksums

SHA-256 checksums for this prompt pack:

```text
511cf49c19e66cf8c128b7b04b537d65dffa3d35fd3ca96a72b1d5c6fbc34da2  generation_prompt.md
88e87c93528b49b18e3c1456b37fd7b25d872319d0ff3fe23fffec133da61693  review_prompt.md
c72fdb7bbfcb6f3b8dc432cb86abff8d37cc96c25805b32397413f14b25045a7  context/functions.json
998b7f9354ec00ef9eecf10b8da7cf933b79cba7a79100fbbf0f1ab11438db9f  context/feature_metadata.json
9e786d945083e6db8120566c631f9425b1093cc3641c22b4ec224f0f33d8885d  context/dataset_metadata.json
3b40bfbf01f70101ce4942e3a3494ad32649848f7c054f007ab10ad554573f76  context/seed_gold.jsonl
```

