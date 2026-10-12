# spark-bench

Benchmark any OpenAI-compatible LLM server the way you actually use it:

- **decode**: generation speed at 1 to N concurrent streams, for four workload
  types (chat, code, agent, report). Per-stream tok/s, total tok/s and time to
  first token.
- **depth**: what long prompts cost. Time to first token, prefill tok/s and
  decode tok/s at growing prompt sizes, with every prompt made unique so the
  prefix cache cannot flatter the numbers.
- **quality**: seeded samples of the GSM8K and MMLU test splits, exact-match
  scoring, Wilson 95% intervals and per-item records for paired A/B tests.

It was built for NVIDIA DGX Spark serving, but works with vLLM, SGLang,
llama.cpp server, TensorFold or anything else that speaks the OpenAI
chat-completions API with streaming. The engine is a single standard-library
Python script; charts are optional.

The repository is laid out as a Hermes Agent skill tap (`skills/spark-bench/`),
and the scripts run equally well from a plain shell.

## Quick start

### Hermes Agent

```bash
hermes skills tap add ajensenwaud/spark-bench
hermes skills install spark-bench
```

Then ask the agent, for example: "benchmark the model on
http://localhost:8000 with the quick preset, label it '1x GPU, vLLM, FP8'".
Community-tap installs are security-scanned; a skill that ships scripts and
data files may be reported as CAUTION and need `--force`, which is expected.

### Claude Code

The skill folder is a standard `SKILL.md` skill. Copy or symlink it into your
skills directory:

```bash
git clone https://github.com/ajensenwaud/spark-bench
mkdir -p ~/.claude/skills
ln -s "$PWD/spark-bench/skills/spark-bench" ~/.claude/skills/spark-bench
```

### Plain command line

```bash
git clone https://github.com/ajensenwaud/spark-bench
cd spark-bench
python3 skills/spark-bench/scripts/bench.py --base-url http://localhost:8000 \
    --preset quick --label "1x GPU, vLLM, FP8"
```

Results go to `./spark-bench-results/<timestamp>-<model>-<label>/`:
`summary.json`, `quality_items.jsonl` and, if matplotlib is available,
`chart.png`.

Compare two runs:

```bash
python3 skills/spark-bench/scripts/compare.py spark-bench-results/<A> spark-bench-results/<B>
```

## Requirements

- Python 3.9 or newer. No third-party packages for benchmarking.
- Optional: matplotlib for `chart.py`, Pillow for `xcard.py`, Node.js 20+ for
  the sparkDash-compatible mode.
- A server exposing `GET /v1/models` and streaming `POST /v1/chat/completions`
  that returns a `usage` block (`stream_options.include_usage`). Without a
  usage block, token counts fall back to counting stream chunks.

## Options

| Option / variable | Default | Meaning |
|---|---|---|
| `--base-url`, `SPARK_BENCH_BASE_URL` | `http://localhost:8000` | Server root; a trailing `/v1` is accepted |
| `--api-key`, `SPARK_BENCH_API_KEY`, `OPENAI_API_KEY` | none | Sent as a bearer token |
| `--model` | first id from `/v1/models` | Model to benchmark |
| `--label` | empty | Deployment description; chart subtitle |
| `--preset` | `quick` | `quick` or `full` (below) |
| `--suites` | all three | Any of `decode,depth,quality` |
| `--thinking` | off | Leave reasoning on (sent as `chat_template_kwargs.enable_thinking`) |
| `--seed` | 20261010 | Quality sample seed; keep it fixed for A/B |
| `--out`, `SPARK_BENCH_RESULTS` | `./spark-bench-results` | Where results go |
| `--handle`, `SPARK_BENCH_HANDLE` | none | Credit printed before "spark-bench" on charts |
| `--force` | off | Run even if the server reports busy (results are flagged) |
| `--protocol sparkdash` | native | Run the sparkDash-compatible mode instead |

Run `bench.py --help` for the full list.

## Presets

| Preset | decode | depth | quality | Typical time |
|---|---|---|---|---|
| `quick` | 1 and 4 streams x 256 tokens | 4k, 32k tokens, 2 samples | GSM8K 40 + MMLU 57 (1 per subject) | minutes |
| `full` | 1, 2, 4 streams x 512 tokens | 4k, 16k, 64k, 128k tokens, 3 samples | GSM8K 200 + MMLU 228 (4 per subject) | tens of minutes; long-context prefill dominates |

Every value can be overridden individually (`--concurrency 1,2,4,8`,
`--decode-tokens`, `--depths`, `--depth-samples`, `--gsm8k-n`,
`--mmlu-per-subject`). Depth sizes beyond the server's reported context
window are skipped.

## Method

- **decode**: three prompts per workload, rotated across streams with a random
  tag so concurrent requests differ. Each wave waits for an idle server, sends
  one warm-up request, then starts all streams together on a barrier.
  `ignore_eos` forces the full token budget. Per-stream tok/s = 1 / TPOT,
  where TPOT = (last token time - first token time) / (tokens - 1).
  `aggregate_tps` = all tokens / wave wall time; `aggregate_decode_tps`
  excludes the prefill phase. Token counts come from the server's usage block.
- **depth**: one calibration request measures tokens per word; each sample
  is then a seeded random-word document of the target size with a unique
  64-bit nonce at the start, followed by a short instruction. Medians of
  TTFT, prefill tok/s (prompt tokens / TTFT) and decode tok/s are reported.
- **quality**: GSM8K items are graded on the last `Answer: <number>`; MMLU
  items are stratified across all 57 subjects and graded on the option letter.
  Temperature 0, thinking off by default. `compare.py` matches items by id and
  reports how many items run B fixed and broke, with an exact two-sided sign
  test, so a two-point accuracy change can be told apart from noise.
- **optional server counters**: if `GET /health` returns JSON with
  `requests_total`, `drafted_total`, `accepted_total`,
  `cached_tokens_total` and `requests_running` (TensorFold does), every
  measurement window also records requests from other clients
  (`foreign_requests`), speculative-decoding acceptance and prompt-cache hits.
  Servers without these (vLLM, llama.cpp and most others) still work; the run
  logs that contention detection is off and records `contention_check: false`.
- Optional request fields the server rejects (`chat_template_kwargs`,
  `ignore_eos`, `stream_options`) are dropped automatically and listed in
  `rejected_request_fields`.

## Interpreting results

- **Per-stream tok/s** is what one user feels; **total tok/s** at higher
  concurrency is the server's throughput. A large drop in per-stream speed as
  streams increase is normal on memory-bandwidth-bound hardware.
- **Time to first token** at depth is mostly prefill. Prefill tok/s that falls
  with prompt size points to attention cost or chunked-prefill settings.
- **Quality intervals are wide at small n.** With 40 GSM8K items a 95%
  interval spans roughly +/- 12 to 15 points. Use `compare.py`'s paired test rather
  than eyeballing two accuracies.
- Check `rejected_request_fields`. If `chat_template_kwargs` was rejected on a
  reasoning model, reasoning tokens were generated and counted. If
  `ignore_eos` was rejected, answers may have stopped early; check
  `completion_tokens` in each decode row.

## Contention caveat

Anything else using the server during a run (another chat window, an agent
session, a scheduled job) steals GPU time and makes the numbers wrong, usually
without any error. spark-bench aborts if the server reports a request running
at start (override with `--force`, which flags the run), and, where the
server exposes request counters, flags every window that saw other clients
(`contended: true`). Do not publish contended runs. On servers without
counters the tool cannot tell, so make sure nothing else is using the model.

## Native and sparkDash-mode numbers are not comparable

`--protocol sparkdash` runs the benchmark modules of
[sparkDash](https://github.com/MiaAI-Lab/sparkDash) by Mia's AI Lab, vendored
unmodified under `skills/spark-bench/compat/sparkdash/vendor/`, so its numbers
match what sparkDash users see. Its prompts, token budgets, prefill sizes,
quality items and scoring differ from spark-bench's native method. Never put
native and sparkDash-mode numbers on one chart, and never call native results
sparkDash numbers. Label sparkDash-mode results "measured with the sparkDash
protocol (Mia's AI Lab)".

## Charts

```bash
python3 skills/spark-bench/scripts/chart.py --summary <run>/summary.json --out chart.png
python3 skills/spark-bench/scripts/xcard.py --summary <run>/summary.json --out card.png \
    --title "My-Model-7B" --eyebrow "1x NVIDIA DGX Spark  ·  vLLM" [--repo owner/repo] [--handle @you]
```

`bench.py` renders `chart.png` automatically when it finds an interpreter with
matplotlib (or `SPARK_BENCH_CHART_PYTHON`). `xcard.py` makes a 1600x900 card
for social posts; contended runs get a red DRAFT stamp.

## Tests

The tests use a small standard-library mock server (`tests/mock_server.py`)
on 127.0.0.1; they never contact a real model server.

```bash
python3 -m unittest discover -s tests -v
```

Chart, card and sparkDash-mode tests are skipped when matplotlib, Pillow or
Node.js are not installed.

## Repository layout

| Path | What |
|---|---|
| `skills/spark-bench/SKILL.md` | Skill instructions for agents (Hermes Agent, Claude Code) |
| `skills/spark-bench/scripts/` | `bench.py`, `compare.py`, `chart.py`, `xcard.py`, `fetch_data.py` |
| `skills/spark-bench/data/` | GSM8K and MMLU test splits with their MIT licences (`SOURCES.md`) |
| `skills/spark-bench/compat/sparkdash/` | sparkDash code (Mia's AI Lab), verbatim, with `NOTICE.md` |
| `tests/` | Mock server and tests |

## Credits and licences

spark-bench is MIT-licensed, Copyright (c) 2026 ajensenwaud (see
`LICENSE`).

- The sparkDash-compatible mode uses code from
  [sparkDash](https://github.com/MiaAI-Lab/sparkDash) by Mia's AI Lab,
  Apache License 2.0 (portions MIT). It is kept verbatim with its `LICENSE`,
  `LICENSE-MIT` and `NOTICE`; see `skills/spark-bench/compat/sparkdash/NOTICE.md`.
- GSM8K: Cobbe et al., 2021, https://github.com/openai/grade-school-math,
  MIT, Copyright (c) 2021 OpenAI.
- MMLU: Hendrycks et al., 2021, https://github.com/hendrycks/test, MIT,
  Copyright (c) 2020 Dan Hendrycks.

Dataset details are in `skills/spark-bench/data/SOURCES.md`.
