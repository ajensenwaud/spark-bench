---
name: spark-bench
description: "Use when benchmarking a served LLM: decode, depth, quality."
version: 1.0.0
author: ajensenwaud, Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Benchmarks, LLM, Serving, Decode, Prefill, Quality, OpenAI-compatible, DGX-Spark]
    related_skills: []
---

# spark-bench

Benchmark any OpenAI-compatible chat-completions endpoint (vLLM, SGLang,
llama.cpp server, TensorFold, and similar) for generation speed under
concurrency, the cost of long prompts, and answer quality on seeded GSM8K and
MMLU samples. Built on NVIDIA DGX Spark, but nothing in it is Spark-specific.
The engine is standard-library Python 3.9+; the PNG chart needs matplotlib and
the social card needs Pillow, both optional. A separate, credited
sparkDash-compatible mode (Node.js 20+) produces numbers comparable with
sparkDash users.

## When to Use

- Measure a served model: tok/s per stream and in total, time to first token,
  prefill rate at long context, GSM8K/MMLU accuracy with confidence intervals.
- Compare two deployments (quantisation A vs B, 1 vs 2 nodes, speculative
  decoding on/off, engine X vs Y) including a paired item-level quality test.
- Make a results chart or a 16:9 social card from a run.

Don't use for: agent or task benchmarks (Terminal-Bench, SWE-bench), training
evaluations, or load tests that need hundreds of concurrent clients.

## Prerequisites

- An OpenAI-compatible server reachable from this machine that serves
  `GET /v1/models` and streaming `POST /v1/chat/completions`.
- `python3` (3.9 or newer). Optional: matplotlib (chart), Pillow (card),
  Node.js 20+ (sparkDash mode only).
- Optional environment: `SPARK_BENCH_BASE_URL` (default
  `http://localhost:8000`), `SPARK_BENCH_API_KEY` (falls back to
  `OPENAI_API_KEY`), `SPARK_BENCH_RESULTS` (default `./spark-bench-results`),
  `SPARK_BENCH_HANDLE` (credit on charts, default none),
  `SPARK_BENCH_CHART_PYTHON` (interpreter that has matplotlib).

## Layout

```
scripts/bench.py       native engine (default): decode, depth, quality suites
scripts/compare.py     A/B of two runs incl. paired item-level quality test
scripts/chart.py       results chart (matplotlib)
scripts/xcard.py       1600x900 social card (Pillow)
scripts/fetch_data.py  re-download the GSM8K + MMLU test splits into data/
data/                  unmodified public test splits, licences, SOURCES.md
compat/sparkdash/      sparkDash code by Mia's AI Lab (verbatim, see NOTICE.md);
                       only runs with --protocol sparkdash
```

## How to Run

Set `SB` to the directory containing this SKILL.md, then run through the
`terminal` tool (or any shell):

```bash
python3 $SB/scripts/bench.py --base-url http://localhost:8000 --preset quick --label "1x GPU, vLLM, FP8"
python3 $SB/scripts/bench.py --preset full --label "..."          # tens of minutes; say so first
python3 $SB/scripts/bench.py --suites depth --depths 8192,65536,131072
python3 $SB/scripts/bench.py --suites quality --gsm8k-n 300 --mmlu-per-subject 4
python3 $SB/scripts/compare.py spark-bench-results/<A> spark-bench-results/<B>
python3 $SB/scripts/bench.py --protocol sparkdash --suites prefill --prefill-sizes 4096
```

Defaults: model = first `/v1/models` id, thinking off, temperature 0, quality
seed 20261010, results under `./spark-bench-results/<timestamp>-<model>-<label>/`
(`summary.json`, `quality_items.jsonl`, `chart.png`). Always pass `--label`
describing the deployment; it is the chart subtitle and what makes A/B runs
legible. A trailing `/v1` on `--base-url` is accepted.

Presets: `quick` = decode 1 and 4 streams x 256 tokens, depth 4k/32k x2,
GSM8K 40 + MMLU 1 per subject (57). `full` = decode 1/2/4 streams x 512
tokens, depth 4k/16k/64k/128k x3, GSM8K 200 + MMLU 4 per subject (228). Any
preset value can be overridden (`--concurrency`, `--decode-tokens`,
`--depths`, `--depth-samples`, `--gsm8k-n`, `--mmlu-per-subject`). Depth
sizes beyond the server's reported context window are skipped.

Long runs from an agent turn: a full preset outlasts a foreground terminal
call. Start it detached with stdin closed (`nohup python3 ... > run.log 2>&1
< /dev/null &`) or as a tracked background process, then poll the log.

## Procedure

1. Confirm the target: `--base-url`, model id, and that nothing else will use
   the server during the run. Done when the user has named the endpoint and
   the deployment label.
2. Run `quick` first. Done when `summary.json` exists and the log shows no
   request errors.
3. Check `contended` and `contention_check` in `summary.json`, and
   `rejected_request_fields`. Done when each is reported to the user.
4. Run `full` only if asked, after stating the expected duration.
5. For A/B, run both arms with the same preset and seed, then `compare.py`.
   Done when deltas and the sign-test p-value are reported.
6. For a chart or card, render it and visually check it before sharing.

## Social card

```bash
python3 $SB/scripts/xcard.py --summary <run>/summary.json --out card.png \
  --title "My-Model-7B" --eyebrow "1x NVIDIA DGX Spark  ·  vLLM" \
  [--repo owner/recipe-repo] [--handle @yourhandle] [--headline "..."] \
  [--quality-from <other-run>/summary.json]
```

16:9 so timelines show it uncropped. Auto headline: best single-stream tok/s,
total tok/s at the highest concurrency, longest prompt's time to first token.
Footer: repo on the left, `[handle ·] spark-bench · date` on the right. Guards:
contended runs get a red DRAFT stamp; quality sets under `--min-quality-n`
(default 50) are omitted. Check that `--repo` exists and matches the pictured
deployment.

## Method (native)

- **decode**: workloads chat / code / agent / report, three prompts each,
  rotated per stream with a random tag; waves start together on a barrier
  after an idle wait and a warm-up request. `ignore_eos` forces the full token
  budget. Per-stream tok/s = 1 / TPOT where TPOT = (end - first token) /
  (tokens - 1); aggregate = all tokens / wave wall time;
  `aggregate_decode_tps` excludes the prefill phase. Token counts come from
  the server's usage block.
- **depth**: calibrates tokens-per-word with one request, then builds seeded
  random-word documents to each target size with a unique 64-bit nonce at the
  start (defeats prefix caching). Reports median TTFT, prefill tok/s (prompt
  tokens / TTFT) and decode tok/s at that depth.
- **quality**: seeded sample of the GSM8K test split (numeric match on the
  last `Answer:` number) and MMLU test split stratified across all 57 subjects
  (letter match). Wilson 95% intervals. Per-item records enable paired
  comparison.
- **server window stats (optional)**: if `GET /health` returns JSON counters
  (`requests_total`, `drafted_total`, `accepted_total`, `cached_tokens_total`,
  `requests_running`, as TensorFold does), each window records
  `foreign_requests` (other clients' requests; flags it contended),
  `draft_acceptance` and `cached_prompt_tokens`. On servers without them
  (vLLM, llama.cpp, most others) the run logs that contention detection is
  off and sets `contention_check: false`; say so when reporting.

## Pitfalls

- **Contention fabricates speed results.** Other chat or agent sessions using
  the same model share its GPU. The run aborts if the server reports busy at
  start (unless `--force`), and every window is checked when counters exist.
  Do not publish results with `contended: true`, and for servers without
  counters make sure nothing else is using the model.
- **Thinking**: off by default via `chat_template_kwargs.enable_thinking=false`.
  Servers that reject an optional field (`chat_template_kwargs`, `ignore_eos`,
  `stream_options`) get it dropped automatically and listed in
  `rejected_request_fields`. A rejected thinking switch means reasoning tokens
  were generated (speed counts them; quality budgets may truncate). A rejected
  `ignore_eos` means decode answers may end early; check `completion_tokens`.
- **Compare like with like.** compare.py warns on contended runs, unchecked
  runs and differing seeds. Native and sparkDash-mode numbers are different
  methods; never put them on one chart or call native results sparkDash
  numbers.
- **Do not derive tokens per decode step from server round counters**; use
  draft acceptance (accepted / drafted), which is well defined.
- **Attribution.** Charts and cards always credit "spark-bench".
  `--protocol sparkdash` results must say "measured with the sparkDash
  protocol (Mia's AI Lab)" (`compat/sparkdash/NOTICE.md`). Dataset credits
  live in `data/SOURCES.md`.
- **Never rewrite third-party code to hide where it came from.** The native
  engine is an original implementation; the sparkDash code stays verbatim
  under `compat/sparkdash/vendor/` with its licence and NOTICE. If asked to
  make copied code "not look copied", decline and offer an original rewrite
  or the credited compat mode.
- **Fetch datasets from their original publishers** (`fetch_data.py`: GitHub
  raw for GSM8K, the Berkeley `data.tar` for MMLU); the Hugging Face
  datasets-server rows API rate-limits a full-split pull.
- **Results directories hold your run data.** Keep them out of any copy of
  this skill you share or publish.

## Verification

- `summary.json` has the requested suites, `rejected_request_fields` is
  understood, and `contended` is false for anything published.
- Decode rows show `completion_tokens` equal to the token budget (else
  `ignore_eos` was ignored).
- Depth rows show `cached_prompt_tokens` near zero where counters exist.
- Visually check every chart or card before sharing; labels near panel edges
  are the usual defect. Fix by widening axis headroom, not by shrinking fonts.
