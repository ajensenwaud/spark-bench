#!/usr/bin/env python3
"""
run.py — run sparkDash-protocol model benchmarks without the dashboard.
Normally started via `bench.py --protocol sparkdash ...`.

Suites (methodology byte-identical to Mia's AI Lab sparkDash — vendored
verbatim under ../vendor, Apache-2.0 with MIT-licensed portions, see ../NOTICE.md):
  decode   streaming decode tok/s at concurrency waves, 4 prompt types
           (structured / prose / code / json), temp 0, top_p 1, thinking off
  prefill  unique-prefix context sweep (prefill tok/s + TTFT), 3 samples/size
  quality  seeded suite: qa, reason, arith, track, gsm8k, mmlu (+ follow, long)

Usage:
  python3 run.py                          # all suites, defaults, localhost:8000
  python3 run.py --suites decode,quality  # subset
  python3 run.py --base-url http://localhost:8000 --model my-model
  python3 run.py --preset quick           # decode c=1,2,4 / prefill 4k,8k / quality qa,arith
  python3 run.py --no-chart

Runs engine.mjs (Node >= 20) per suite sequentially — the managers 409 on
concurrent jobs. Writes raw job JSON + summary.json, then renders the chart.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
ENGINE = SCRIPTS / "engine.mjs"
CHART_PY = SCRIPTS / "chart.py"
DEFAULT_BASE_URL = "http://localhost:8000"


def chart_python():
    """First interpreter that can import matplotlib, or None (chart is optional)."""
    for c in (os.environ.get("SPARK_BENCH_CHART_PYTHON"), sys.executable, "/usr/bin/python3"):
        if c and Path(c).exists() and subprocess.call(
                [c, "-c", "import matplotlib"], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL) == 0:
            return c
    return None

DECODE_TYPES = ["structured", "prose", "code", "json"]
QUALITY_DEFAULT = ["qa", "reason", "arith", "track", "gsm8k", "mmlu"]

PRESETS = {
    # sparkDash defaults
    "full": {
        "decode_concurrencies": [1, 2, 4, 8],
        "decode_max_tokens": 400,
        "prefill_sizes": [4096, 8192, 16384, 32768],
        "quality_categories": QUALITY_DEFAULT,
        "quality_concurrency": 4,
    },
    # short pilot: proves the serve and gives a usable chart in ~5 min
    "quick": {
        "decode_concurrencies": [1, 2, 4],
        "decode_max_tokens": 400,
        "prefill_sizes": [4096, 8192],
        "quality_categories": ["qa", "arith"],
        "quality_concurrency": 4,
    },
}


def http_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def serve_idle(base_url):
    """Best-effort idle check. Returns (idle: bool|None, detail: str)."""
    for suffix in ("/health", "/healthz", "/metrics"):
        try:
            body = http_json(base_url + suffix, timeout=3)
        except Exception:
            continue
        text = json.dumps(body)
        m = re.search(r'"requests_running"\s*:\s*(\d+)', text)
        if m:
            n = int(m.group(1))
            return (n == 0), f"{suffix} requests_running={n}"
        m = re.search(r'num_requests_running[^0-9]*(\d+)', text)
        if m:
            n = int(m.group(1))
            return (n == 0), f"{suffix} num_requests_running={n}"
    return None, "no requests_running metric exposed"


def run_engine(suite, base_url, model, extra, out_path, timeout_s):
    cmd = ["node", str(ENGINE), "--suite", suite,
           "--host", base_url["host"], "--port", str(base_url["port"]),
           "--model", model, "--timeout", str(timeout_s)]
    for k, v in extra.items():
        cmd += [f"--{k}", str(v)]
    print(f"[{suite}] {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 60)
    elapsed = time.time() - t0
    for line in proc.stderr.splitlines():
        print(f"  {line}", flush=True)
    if not proc.stdout.strip():
        print(f"  !! {suite} produced no output (exit {proc.returncode})", flush=True)
        return None, elapsed
    try:
        job = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"  !! {suite} output not JSON (exit {proc.returncode})", flush=True)
        return None, elapsed
    out_path.write_text(json.dumps(job, indent=2))
    status = job.get("status")
    print(f"  -> {status} in {elapsed:.0f}s (exit {proc.returncode})", flush=True)
    return job, elapsed


def parse_base_url(u):
    m = re.match(r"^(https?)://([^:/]+):(\d+)", u.rstrip("/"))
    if not m:
        raise SystemExit(f"Cannot parse --base-url: {u}")
    return {"scheme": m.group(1), "host": m.group(2), "port": int(m.group(3)),
            "base": f"{m.group(1)}://{m.group(2)}:{m.group(3)}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("SPARK_BENCH_BASE_URL") or DEFAULT_BASE_URL)
    ap.add_argument("--model", default=None,
                    help="default: first model from /v1/models")
    ap.add_argument("--suites", default="decode,prefill,quality")
    ap.add_argument("--preset", choices=["full", "quick"], default="full")
    ap.add_argument("--decode-concurrencies", default=None, help="e.g. 1,2,4")
    ap.add_argument("--decode-types", default=",".join(DECODE_TYPES))
    ap.add_argument("--decode-max-tokens", type=int, default=None)
    ap.add_argument("--prefill-sizes", default=None, help="e.g. 4096,8192")
    ap.add_argument("--quality-categories", default=None)
    ap.add_argument("--quality-concurrency", type=int, default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=3600, help="per-suite seconds")
    ap.add_argument("--out", default=None,
                    help="results dir (default: $SPARK_BENCH_RESULTS or ./spark-bench-results, "
                         "then sparkdash-<ts>)")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run even if the serve reports busy (labelled contended)")
    args = ap.parse_args()

    base = parse_base_url(args.base_url)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    cfg = PRESETS[args.preset]
    if args.decode_concurrencies:
        cfg["decode_concurrencies"] = [int(x) for x in args.decode_concurrencies.split(",")]
    if args.decode_max_tokens:
        cfg["decode_max_tokens"] = args.decode_max_tokens
    if args.prefill_sizes:
        cfg["prefill_sizes"] = [int(x) for x in args.prefill_sizes.split(",")]
    if args.quality_categories:
        cfg["quality_categories"] = [x.strip() for x in args.quality_categories.split(",")]
    if args.quality_concurrency:
        cfg["quality_concurrency"] = args.quality_concurrency

    model = args.model
    if not model:
        try:
            models = http_json(base["base"] + "/v1/models", timeout=5)
            model = models["data"][0]["id"]
        except Exception as e:
            raise SystemExit(f"Cannot fetch /v1/models from {base['base']}: {e}")
    print(f"target: {base['base']}  model: {model}  preset: {args.preset}")

    idle, detail = serve_idle(base["base"])
    print(f"serve idle check: {detail}")
    if idle is False:
        msg = (f"SERVE BUSY ({detail}) — latency/throughput numbers will be "
               f"contended. Benchmarks that power this session's model also "
               f"contend with the conversation itself.")
        if not args.force:
            print(f"ABORT: {msg}\nRe-run with --force to proceed anyway (labelled contended).")
            sys.exit(3)
        print(f"WARNING: {msg} — proceeding under --force; results labelled contended.")
    contended = idle is False

    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    results_root = Path(os.environ.get("SPARK_BENCH_RESULTS") or "spark-bench-results")
    out_dir = Path(args.out) if args.out else results_root / f"sparkdash-{ts}"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generatedAt": ts,
        "target": base["base"],
        "model": model,
        "preset": args.preset,
        "config": cfg,
        "contended": contended,
        "idleDetail": detail,
        "suites": {},
        "wallSeconds": {},
    }

    # engine.mjs keeps the sparkDash managers' job state in the run dir, not the skill dir
    common = {"state-dir": str(out_dir / "state")}
    if args.api_key:
        common["api-key"] = args.api_key
    if args.context_length:
        common["context-length"] = args.context_length

    if "decode" in suites:
        types = [t.strip() for t in args.decode_types.split(",") if t.strip()]
        decode_jobs = {}
        for t in types:
            job, secs = run_engine(
                "decode", base, model,
                {**common, "types": t,
                 "concurrencies": ",".join(map(str, cfg["decode_concurrencies"])),
                 "max-tokens": cfg["decode_max_tokens"]},
                raw_dir / f"decode-{t}.json", args.timeout)
            summary["wallSeconds"][f"decode-{t}"] = round(secs, 1)
            if job:
                decode_jobs[t] = job
        summary["suites"]["decode"] = {
            t: {"status": j.get("status"),
                "waves": [{k: w.get(k) for k in
                           ("concurrency", "streamsOk", "streamsFailed",
                            "meanDecodeTps", "medianDecodeTps",
                            "aggregateDecodeTps", "meanTtftMs",
                            "totalDecodeTokens", "durationMs")}
                          for w in j.get("results", [])]}
            for t, j in decode_jobs.items()}

    if "prefill" in suites:
        job, secs = run_engine(
            "prefill", base, model,
            {**common, "sizes": ",".join(map(str, cfg["prefill_sizes"]))},
            raw_dir / "prefill.json", args.timeout)
        summary["wallSeconds"]["prefill"] = round(secs, 1)
        if job:
            summary["suites"]["prefill"] = {
                "status": job.get("status"),
                "points": [{k: r.get(k) for k in
                            ("targetTokens", "promptTokens", "prefillTps",
                             "ttftMs", "cachedTokens", "lowConfidence",
                             "samples", "error")}
                           for r in job.get("results", [])]}

    if "quality" in suites:
        job, secs = run_engine(
            "quality", base, model,
            {**common,
             "categories": ",".join(cfg["quality_categories"]),
             "concurrency": cfg["quality_concurrency"]},
            raw_dir / "quality.json", args.timeout)
        summary["wallSeconds"]["quality"] = round(secs, 1)
        if job:
            res = job.get("results") or {}
            summary["suites"]["quality"] = {
                "status": job.get("status"),
                "overallPct": res.get("overallPct"),
                "categories": res.get("categories"),
                "itemsTotal": len(res.get("items") or []),
                "suiteVersion": (job.get("config") or {}).get("suiteVersion"),
                "scoringVersion": (job.get("config") or {}).get("scoringVersion"),
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary: {out_dir / 'summary.json'}")

    chart_path = None
    if not args.no_chart:
        py = chart_python()
        if py:
            proc = subprocess.run(
                [py, str(CHART_PY),
                 "--summary", str(out_dir / "summary.json"),
                 "--out", str(out_dir / "bench-chart.png")],
                capture_output=True, text=True)
            if proc.returncode == 0:
                chart_path = out_dir / "bench-chart.png"
                print(f"chart:   {chart_path}")
            else:
                print(f"chart failed: {proc.stderr[-800:]}")
        else:
            print("chart skipped: no interpreter with matplotlib found (set SPARK_BENCH_CHART_PYTHON)")

    # compact console report
    print("\n=== RESULTS ===")
    d = summary["suites"].get("decode")
    if d:
        for t, blk in d.items():
            waves = ", ".join(
                f"c{w['concurrency']}:{w['aggregateDecodeTps']} agg"
                f" ({w['meanDecodeTps']}/stream)" for w in blk["waves"])
            print(f"decode[{t}]: {waves}")
    p = summary["suites"].get("prefill")
    if p:
        print("prefill: " + ", ".join(
            f"{pt['targetTokens']//1024}k:{pt['prefillTps']} t/s, "
            f"TTFT {pt['ttftMs']}ms" for pt in p["points"] if pt.get("prefillTps")))
    q = summary["suites"].get("quality")
    if q:
        cats = ", ".join(f"{k} {v['pct']}%" for k, v in (q["categories"] or {}).items())
        print(f"quality: overall {q['overallPct']}% — {cats}")
    if contended:
        print("NOTE: run was CONTENDED (serve busy at start).")
    return chart_path


if __name__ == "__main__":
    main()