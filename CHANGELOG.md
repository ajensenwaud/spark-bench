# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog, and the project uses semantic versioning.

## [1.0.0] - 2026-10-12

First general release.

### Added

- Native benchmark engine (`scripts/bench.py`): decode, depth and quality
  suites with `quick` and `full` presets, standard-library Python only.
- `compare.py`: A/B comparison with a paired item-level sign test for
  quality.
- `chart.py` (matplotlib) and `xcard.py` (Pillow, 1600x900 social card).
- sparkDash-compatible mode (`--protocol sparkdash`) using sparkDash's
  benchmark modules verbatim, with their Apache-2.0/MIT licences and NOTICE.
- GSM8K and MMLU test splits with their MIT licence texts, and
  `fetch_data.py` to re-download them from the original publishers.
- `--api-key` / `SPARK_BENCH_API_KEY` (falls back to `OPENAI_API_KEY`) for
  servers that require a bearer token.
- Environment defaults: `SPARK_BENCH_BASE_URL`, `SPARK_BENCH_RESULTS`,
  `SPARK_BENCH_HANDLE`, `SPARK_BENCH_CHART_PYTHON`, `SPARK_BENCH_FONT_DIR`.
- `contention_check` in `summary.json`, and a log line when the server's
  `/health` exposes no request counters.
- Standard-library mock OpenAI-compatible server and test suite under
  `tests/`.

### Changed (from the pre-release skill)

- Default `--base-url` is `http://localhost:8000` (was a private host).
- Charts and cards credit "spark-bench" with an optional `--handle`; there is
  no default personal handle.
- Results are written to `./spark-bench-results/` (or
  `$SPARK_BENCH_RESULTS`) instead of inside the skill directory.
- sparkDash-mode job state is kept in the run's results directory.
- The chart interpreter is found by probing for matplotlib instead of a fixed
  virtualenv path; `xcard.py` falls back to other font locations.
- `compare.py` prints `-` for missing values instead of failing.
- sparkDash attribution corrected: the vendored snapshot is from sparkDash
  2.0.0, which is Apache License 2.0 (portions MIT); the upstream `NOTICE`
  file is now included.

### Removed

- `scripts/overnight.sh`, which depended on a specific host, a
  TensorFold-only `/health` schema and a private scheduling setup.
