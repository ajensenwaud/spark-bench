# Datasets

spark-bench's quality suite samples from two public benchmark datasets. The
files here are unmodified copies of the original test splits (MMLU converted
from the original per-subject CSV files to one JSON line per question, text
unchanged); the sampling (seed, sizes, stratification), prompts and answer
extraction are spark-bench's own.

| File | Dataset | Source | Licence |
|---|---|---|---|
| `gsm8k_test.jsonl` | GSM8K test split, 1,319 grade-school maths problems | https://github.com/openai/grade-school-math (Cobbe et al., 2021, "Training Verifiers to Solve Math Word Problems") | MIT, Copyright (c) 2021 OpenAI, full text in `LICENSE-GSM8K` |
| `mmlu_test.jsonl` | MMLU test split, 14,042 multiple-choice questions across 57 subjects | https://github.com/hendrycks/test, file https://people.eecs.berkeley.edu/~hendrycks/data.tar (Hendrycks et al., 2021, "Measuring Massive Multitask Language Understanding") | MIT, Copyright (c) 2020 Dan Hendrycks, full text in `LICENSE-MMLU` |

Both licences permit redistribution provided the copyright and permission
notice travel with the copies, which is why `LICENSE-GSM8K` and
`LICENSE-MMLU` sit next to the data. The MMLU authors note that the questions
were collected from freely available online sources.

To re-download both files from their original publishers (MMLU's `data.tar`
is about 166 MB), run `python3 scripts/fetch_data.py`. It skips files that are
already present with the expected row count.

The sparkDash-compatible mode uses its own fixed subsets
(`compat/sparkdash/vendor/server/collectors/data/`), credited in that
folder's `NOTICE.md`.
