# sparkDash-compatible mode

This folder contains code from **sparkDash** by Mia's AI Lab
(https://github.com/MiaAI-Lab/sparkDash). It is kept separate from
spark-bench's own engine and is only used when a run is started with
`--protocol sparkdash`.

## Licence

The files under `vendor/` (other than the shim noted below) are byte-identical
to sparkDash at upstream commit `5051a33` (9 October 2026, the 2.0.0 licence
change). From version 2.0.0
sparkDash is licensed under the **Apache License, Version 2.0**; code written
for versions up to 1.9.0 remains under the **MIT License**. Both texts and the
upstream NOTICE are included verbatim:

- `vendor/LICENSE` - Apache License 2.0
- `vendor/LICENSE-MIT` - MIT License (versions up to 1.9.0)
- `vendor/NOTICE` - sparkDash's NOTICE file
- `vendor/server/collectors/data/NOTICE.md` - GSM8K and MMLU sample notices
  (MIT; OpenAI and Dan Hendrycks)

## What is theirs and what was added

Theirs (verbatim, Copyright 2026 Mia's AI Lab): everything under `vendor/`
except `vendor/node_modules/undici/` - the decode, prefill and quality
benchmark modules, prompts, item generators, grading, and the GSM8K/MMLU
subsets.

Added by spark-bench (MIT, see the repository LICENSE), not part of sparkDash:

- `vendor/node_modules/undici/` - a small shim so the vendored modules'
  `import { Agent, fetch } from "undici"` resolves to Node's built-in fetch
  without an npm install. It is not the real undici package.
- `scripts/engine.mjs` - a command-line driver for the vendored modules.
- `scripts/run.py` - orchestration, idle check and summary.
- `scripts/chart.py` - chart for this mode.

No sparkDash file under `vendor/` has been changed. To check, compare git
blob hashes with upstream: `git hash-object <file>` against the `sha` from
`gh api repos/MiaAI-Lab/sparkDash/contents/<file>?ref=5051a33db3aa6ddc875dca1bf4cc71939d49af59`.

## Labelling results

Results from this mode are sparkDash-protocol numbers and should be labelled
that way when published, e.g. "measured with the sparkDash protocol (Mia's AI
Lab)". spark-bench's native results (`scripts/bench.py`, the default) use
spark-bench's own method and are not directly comparable with sparkDash
scores.
