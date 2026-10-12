#!/usr/bin/env python3
"""
spark-bench: measure an OpenAI-compatible LLM endpoint the way a local-serving
owner actually uses it (built on NVIDIA DGX Spark, works with any server).
Standard library only; matplotlib is needed only for the optional chart.

Suites
  decode   generation speed against concurrency for four workloads
           (chat, code, agent, report); per-stream and aggregate tok/s, TTFT
  depth    what long context costs: prefill rate and decode speed at growing
           context sizes, every prompt cache-busted and the busting verified
  quality  seeded samples of GSM8K and MMLU (original public test splits),
           exact-match scoring, Wilson 95% intervals, per-item records for
           paired A/B comparison

Optional server extras (used only when GET /health returns JSON counters
such as requests_total / drafted_total / accepted_total / cached_tokens_total,
as TensorFold does; on other servers they are skipped and the run says so)
  - contention detection: every wave checks that the server handled only our
    requests; anything else running at the same time flags the wave
  - speculative decoding: draft acceptance rate (accepted / drafted)
  - prompt-cache hits during the depth suite (proves the numbers are cold)

Environment
  SPARK_BENCH_BASE_URL   default for --base-url (else http://localhost:8000)
  SPARK_BENCH_API_KEY    bearer token (falls back to OPENAI_API_KEY)
  SPARK_BENCH_RESULTS    results root (else ./spark-bench-results)
  SPARK_BENCH_HANDLE     credit shown on charts next to "spark-bench"
  SPARK_BENCH_CHART_PYTHON  interpreter with matplotlib for the chart

Examples
  python3 bench.py --preset quick --label "1x GPU, vLLM, FP8"
  python3 bench.py --preset full --label "2 nodes TP2, 4-bit, spec decode"
  python3 bench.py --suites depth --depths 8192,65536,131072
  python3 bench.py --base-url http://localhost:8000 --suites quality --gsm8k-n 200
"""
import argparse, json, math, os, random, re, socket, subprocess, sys, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

VERSION = "spark-bench/1.0"
SKILL = Path(__file__).resolve().parent.parent
DATA = SKILL / "data"
DEFAULT_BASE_URL = "http://localhost:8000"


def results_root():
    return Path(os.environ.get("SPARK_BENCH_RESULTS") or "spark-bench-results")


def chart_python():
    """First interpreter that can import matplotlib, or None (chart is optional)."""
    cands = [os.environ.get("SPARK_BENCH_CHART_PYTHON"), sys.executable, "/usr/bin/python3",
             str(Path.home() / ".venv/bin/python")]
    seen = set()
    for c in cands:
        if not c or c in seen or not Path(c).exists():
            continue
        seen.add(c)
        try:
            if subprocess.call([c, "-c", "import matplotlib"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60) == 0:
                return c
        except Exception:
            continue
    return None

PRESETS = {
    "quick": dict(suites="decode,depth,quality", concurrency=[1, 4], decode_tokens=256,
                  depths=[4096, 32768], depth_samples=2, gsm8k_n=40, mmlu_per_subject=1),
    "full": dict(suites="decode,depth,quality", concurrency=[1, 2, 4], decode_tokens=512,
                 depths=[4096, 16384, 65536, 131072], depth_samples=3, gsm8k_n=200,
                 mmlu_per_subject=4),
}

# Workloads: three prompts each, rotated across streams so concurrent requests differ.
WORKLOADS = {
    "chat": [
        "Explain to a curious teenager how a heat pump moves heat from a cold place to a warm one. Use an everyday analogy and keep going until you have covered efficiency and why it drops in very cold weather.",
        "A friend is choosing between renting and buying a flat in a big city. Talk them through the trade-offs in a warm, practical way, covering cash flow, flexibility, maintenance and what happens if interest rates rise.",
        "Describe what happens, step by step, from the moment you press a key on a laptop until the letter appears on screen. Keep it conversational but accurate.",
    ],
    "code": [
        "Write a complete Python module implementing an LRU cache with a time-to-live per entry, thread safety, hit/miss statistics and a decorator for memoising functions. Include docstrings and a small self-test under __main__.",
        "Write a Go HTTP server with three endpoints for a to-do list (list, create, toggle done) backed by an in-memory store guarded by a mutex. Include request validation, JSON error responses and graceful shutdown.",
        "Write a TypeScript function that parses a cron expression with five fields, supports ranges, steps and lists, and returns the next N run times after a given Date. Include type definitions and unit-test style examples.",
    ],
    "agent": [
        "You are an operations agent with tools search_logs(query, since), restart_service(name), page_oncall(team, summary) and open_ticket(title, body, severity). A checkout service is returning 502 errors for 20% of requests. Produce your full plan as a JSON array of tool calls with arguments and a one-line rationale for each, then explain how you would verify the fix.",
        "You are a research agent with tools web_search(q), fetch(url), extract_table(url, selector) and write_file(path, content). Plan, as a JSON list of tool calls with arguments, how you would build a comparison of the five cheapest nbn 100 plans in Australia, then describe the checks you would run before trusting the numbers.",
        "You are a coding agent with tools read_file(path), grep(pattern, path), run_tests(target) and apply_patch(diff). A test named test_invoice_rounding fails after a currency library upgrade. Write your investigation as a JSON array of tool calls with arguments and expected findings, followed by the patch you would propose.",
    ],
    "report": [
        "Write a structured technical report on running large language models on local hardware: memory bandwidth limits, quantisation trade-offs, batching, speculative decoding and how to benchmark fairly. Use headings and keep writing in depth.",
        "Write a detailed post-incident review for a fictional outage in which a database failover took 40 minutes. Include timeline, root cause, contributing factors, what went well, and a prioritised list of corrective actions.",
        "Write an in-depth briefing comparing three approaches to securing AI agents that can run shell commands: containers, microVMs and policy-enforcing proxies. Cover threat model, overhead, operational cost and failure modes.",
    ],
}

FILLER_WORDS = (
    "river stone market winter signal garden ladder copper engine harbour window "
    "letter forest pattern silver morning bridge candle orbit meadow lantern "
    "journey velvet compass thunder pocket mirror island harvest feather canvas "
    "station anchor puzzle shadow marble whistle crystal saddle tunnel quarry "
    "falcon ember timber glacier violet ribbon circuit plaster beacon cobalt "
    "meadow ledger orchard pebble summit cradle domino fabric granite hollow "
    "kettle lagoon magnet nectar oyster pillar quiver rafter sapphire thistle "
    "umbra vessel walnut yonder zephyr basket cactus dune ferry gravel hazel "
    "ivory jungle kiosk lemon mango nickel olive parrot quartz radish spruce "
    "tulip urchin valley willow yarrow zinc acorn bramble cedar dahlia "
    "measures carries follows reaches gathers balances shifts settles reflects "
    "quietly slowly briefly openly nearly roughly evenly sharply softly"
).split()


# ---------------------------------------------------------------- HTTP layer

def root_url(base):
    base = base.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


API_KEY = os.environ.get("SPARK_BENCH_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


def auth_headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def get_json(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers=auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def health(base):
    h = get_json(root_url(base) + "/health")
    return h if isinstance(h, dict) else None


class Endpoint:
    """Chat-completions client that learns which optional fields the server accepts."""

    OPTIONAL = ("chat_template_kwargs", "ignore_eos", "stream_options")

    def __init__(self, base, model, timeout):
        self.url = root_url(base) + "/v1/chat/completions"
        self.model, self.timeout = model, timeout
        self.rejected = set()
        self.lock = threading.Lock()

    def _post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **auth_headers()})
        return urllib.request.urlopen(req, timeout=self.timeout)

    def stream(self, prompt, max_tokens, thinking=False, force_length=False, seed=None):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "stream": True,
                "stream_options": {"include_usage": True}}
        if seed is not None:
            body["seed"] = seed
        if not thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if force_length:
            body["ignore_eos"] = True
        for k in list(body):
            if k in self.rejected:
                body.pop(k)
        while True:
            t_send = time.perf_counter()
            try:
                resp = self._post(body)
                break
            except urllib.error.HTTPError as e:
                drop = next((k for k in self.OPTIONAL if k in body and k not in self.rejected), None)
                if e.code in (400, 422) and drop:
                    with self.lock:
                        self.rejected.add(drop)
                    body.pop(drop)
                    continue
                raise
        first = last = None
        content, reasoning, usage, finish = [], [], None, None
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    c = d.get("content") or ""
                    r = d.get("reasoning_content") or d.get("reasoning") or ""
                    if c or r:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        content.append(c)
                        reasoning.append(r)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        t_end = time.perf_counter()
        out_tokens = (usage or {}).get("completion_tokens")
        if out_tokens is None:  # no usage block: fall back to chunk count (lower bound)
            out_tokens = sum(1 for c, r in zip(content, reasoning) if c or r)
        ttft = (first - t_send) if first else None
        # Time per output token after the first (the usual TPOT definition).
        tpot = ((t_end - first) / (out_tokens - 1)) if first and out_tokens > 1 else None
        return {"t_send": t_send, "t_first": first, "t_end": t_end, "ttft_s": ttft,
                "tpot_s": tpot, "decode_tps": (1 / tpot) if tpot else None,
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": out_tokens, "finish": finish,
                "text": "".join(content), "reasoning_chars": sum(len(r) for r in reasoning)}


# ---------------------------------------------------------------- helpers

def counters(h):
    keys = ("requests_total", "drafted_total", "accepted_total",
            "completion_tokens_total", "cached_tokens_total", "prompt_tokens_total")
    return {k: h.get(k) for k in keys} if h else {}


def delta(a, b, k):
    if a.get(k) is None or b.get(k) is None:
        return None
    return b[k] - a[k]


def server_window(before, after, our_requests):
    """Contention and speculative-decoding stats for one measurement window."""
    out = {}
    reqs = delta(before, after, "requests_total")
    if reqs is not None:
        out["foreign_requests"] = max(0, reqs - our_requests)
    drafted, accepted = delta(before, after, "drafted_total"), delta(before, after, "accepted_total")
    if drafted:
        out["draft_acceptance"] = round(accepted / drafted, 4)
    cached = delta(before, after, "cached_tokens_total")
    if cached is not None:
        out["cached_prompt_tokens"] = cached
    return out


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def r(x, n=2):
    return None if x is None else round(x, n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def wait_idle(base, limit_s=120):
    """Wait for requests_running == 0 so a wave starts on an idle server."""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        h = health(base)
        if not h or h.get("requests_running", 0) == 0:
            return True
        time.sleep(2)
    return False


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- suites

def run_decode(ep, base, conc_list, max_tokens, thinking):
    results = []
    for wl, prompts in WORKLOADS.items():
        for c in conc_list:
            wait_idle(base)
            ep.stream(prompts[0][:200], 16, thinking=thinking)  # warm the path
            before = counters(health(base))
            barrier = threading.Barrier(c)

            def one(i):
                barrier.wait()
                tag = f"[run {random.getrandbits(32):08x}] "  # keeps streams distinct
                return ep.stream(tag + prompts[i % len(prompts)], max_tokens,
                                 thinking=thinking, force_length=True)

            with ThreadPoolExecutor(c) as ex:
                streams = list(ex.map(one, range(c)))
            after = counters(health(base))
            span = max(s["t_end"] for s in streams) - min(s["t_send"] for s in streams)
            firsts = [s["t_first"] for s in streams if s["t_first"]]
            dspan = (max(s["t_end"] for s in streams) - min(firsts)) if firsts else 0
            toks = sum(s["completion_tokens"] for s in streams)
            row = {"workload": wl, "concurrency": c,
                   "per_stream_tps_mean": r(sum(s["decode_tps"] or 0 for s in streams) / c),
                   "per_stream_tps_min": r(min(s["decode_tps"] or 0 for s in streams)),
                   "aggregate_tps": r(toks / span),
                   "aggregate_decode_tps": r(toks / dspan) if dspan > 0 else None,
                   "ttft_p50_s": r(pct([s["ttft_s"] for s in streams], 0.5), 3),
                   "completion_tokens": [s["completion_tokens"] for s in streams],
                   "finish": sorted({str(s["finish"]) for s in streams}),
                   **server_window(before, after, c)}
            row["contended"] = bool(row.get("foreign_requests"))
            log(f"decode {wl:<6} c={c}: {row['per_stream_tps_mean']} tok/s/stream, "
                f"{row['aggregate_tps']} agg" + ("  CONTENDED" if row["contended"] else ""))
            results.append(row)
    return results


def filler(n_words, rng):
    words, out = FILLER_WORDS, []
    while len(out) < n_words:
        k = rng.randint(8, 18)
        s = [rng.choice(words) for _ in range(k)]
        s[0] = s[0].capitalize()
        out.extend(s)
        out[-1] += "."
    return " ".join(out[:n_words])


def run_depth(ep, base, depths, samples, gen_tokens, thinking, ctx_limit):
    rng = random.Random()
    cal = ep.stream("Calibration. " + filler(2000, random.Random(1)), 1, thinking=thinking)
    tok_per_word = (cal["prompt_tokens"] or 2600) / 2000
    log(f"depth: calibrated {tok_per_word:.3f} tokens/word")
    rows = []
    for size in depths:
        if ctx_limit and size + gen_tokens + 256 > ctx_limit:
            log(f"depth {size}: skipped, over the server's {ctx_limit}-token window")
            continue
        samp = []
        for i in range(samples):
            wait_idle(base)
            before = counters(health(base))
            nonce = f"Document {rng.getrandbits(64):016x}. "
            words = int((size - 60) / tok_per_word)
            prompt = (nonce + filler(words, rng) +
                      "\n\nIn one paragraph, describe the style of the text above.")
            s = ep.stream(prompt, gen_tokens, thinking=thinking, force_length=True)
            after = counters(health(base))
            pt = s["prompt_tokens"] or size
            s.update(server_window(before, after, 1))
            s["prefill_tps"] = (pt / s["ttft_s"]) if s["ttft_s"] else None
            samp.append(s)
        row = {"target_tokens": size,
               "prompt_tokens": int(pct([s["prompt_tokens"] for s in samp], 0.5) or 0),
               "ttft_s": r(pct([s["ttft_s"] for s in samp], 0.5), 3),
               "prefill_tps": r(pct([s["prefill_tps"] for s in samp], 0.5), 1),
               "decode_tps": r(pct([s["decode_tps"] for s in samp], 0.5)),
               "samples": samples,
               "cached_prompt_tokens": [s.get("cached_prompt_tokens") for s in samp],
               "contended": any(s.get("foreign_requests") for s in samp)}
        acc = [s.get("draft_acceptance") for s in samp if s.get("draft_acceptance") is not None]
        if acc:
            row["draft_acceptance"] = r(sum(acc) / len(acc), 4)
        log(f"depth {row['prompt_tokens']:>7} tok: prefill {row['prefill_tps']} tok/s, "
            f"ttft {row['ttft_s']} s, decode {row['decode_tps']} tok/s, "
            f"cached {row['cached_prompt_tokens']}" + ("  CONTENDED" if row["contended"] else ""))
        rows.append(row)
    return rows


NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def to_num(s):
    try:
        return float(s.replace(",", "").replace("$", ""))
    except (ValueError, AttributeError):
        return None


def grade_gsm8k(text, gold):
    m = re.findall(r"answer\s*[:=]\s*\**\s*(" + NUM.pattern + ")", text, re.I)
    pred = to_num(m[-1]) if m else (to_num(NUM.findall(text)[-1]) if NUM.findall(text) else None)
    return pred is not None and abs(pred - gold) < 1e-6, pred


def grade_mmlu(text, gold):
    t = text.strip()
    m = (re.search(r"answer\s*(?:is)?\s*[:\-]?\s*\(?([ABCD])\b", t, re.I)
         or re.match(r"\(?([ABCD])\b", t) or re.search(r"\b([ABCD])\b", t))
    pred = m.group(1).upper() if m else None
    return pred == gold, pred


def sample_items(seed, gsm8k_n, mmlu_per_subject):
    rng = random.Random(seed)
    items = []
    if gsm8k_n:
        with (DATA / "gsm8k_test.jsonl").open(encoding="utf-8") as fh:
            g = [json.loads(l) for l in fh]
        for idx in sorted(rng.sample(range(len(g)), min(gsm8k_n, len(g)))):
            gold = to_num(g[idx]["answer"].split("####")[-1].strip())
            items.append({"set": "gsm8k", "id": f"gsm8k/{idx}", "gold": gold,
                          "prompt": ("Solve this problem. Work it out briefly, then end with a "
                                     "final line of the form 'Answer: <number>'.\n\n"
                                     + g[idx]["question"])})
    if mmlu_per_subject:
        by = {}
        with (DATA / "mmlu_test.jsonl").open(encoding="utf-8") as fh:
            for l in fh:
                x = json.loads(l)
                by.setdefault(x["subject"], []).append(x)
        for subj in sorted(by):
            for x in rng.sample(by[subj], min(mmlu_per_subject, len(by[subj]))):
                opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", x["choices"]))
                items.append({"set": "mmlu", "id": "mmlu/" + x["id"], "gold": x["answer"],
                              "subject": subj,
                              "prompt": (f"Multiple-choice question ({subj.replace('_', ' ')}). "
                                         "Reply with the letter of the correct option only.\n\n"
                                         f"{x['question']}\n\n{opts}")})
    return items


def run_quality(ep, base, seed, gsm8k_n, mmlu_per_subject, conc, thinking, items_path):
    items = sample_items(seed, gsm8k_n, mmlu_per_subject)
    log(f"quality: {len(items)} items, seed {seed}, concurrency {conc}")
    before = counters(health(base))

    def one(it):
        budget = (4096 if thinking else 768) if it["set"] == "gsm8k" else (2048 if thinking else 24)
        try:
            s = ep.stream(it["prompt"], budget, thinking=thinking, seed=seed)
            ok, pred = (grade_gsm8k if it["set"] == "gsm8k" else grade_mmlu)(s["text"], it["gold"])
            return {"id": it["id"], "set": it["set"], "gold": it["gold"], "pred": pred,
                    "correct": ok, "finish": s["finish"], "tokens": s["completion_tokens"]}
        except Exception as e:
            return {"id": it["id"], "set": it["set"], "gold": it["gold"], "pred": None,
                    "correct": False, "error": str(e)[:200]}

    with ThreadPoolExecutor(conc) as ex:
        recs = list(ex.map(one, items))
    after = counters(health(base))
    with items_path.open("w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    out = {"seed": seed, "thinking": thinking, "sets": {}}
    for name in ("gsm8k", "mmlu"):
        xs = [x for x in recs if x["set"] == name]
        if not xs:
            continue
        k = sum(x["correct"] for x in xs)
        out["sets"][name] = {"n": len(xs), "correct": k, "accuracy": round(k / len(xs), 4),
                             "ci95": wilson(k, len(xs)),
                             "errors": sum(1 for x in xs if x.get("error")),
                             "truncated": sum(1 for x in xs if x.get("finish") == "length")}
        a = out["sets"][name]
        log(f"quality {name}: {a['accuracy']:.1%} ({k}/{len(xs)}), 95% CI {a['ci95']}")
    out.update(server_window(before, after, len(items)))
    out["contended"] = bool(out.get("foreign_requests"))
    return out


# ---------------------------------------------------------------- main

def main():
    # Hand off to the credited sparkDash-compatible mode before parsing our own flags,
    # so its options (--decode-concurrencies, --prefill-sizes, ...) pass through untouched.
    argv = sys.argv[1:]
    if "--protocol=sparkdash" in argv or ("--protocol" in argv and
                                          argv[argv.index("--protocol") + 1:][:1] == ["sparkdash"]):
        if "--protocol=sparkdash" in argv:
            argv.remove("--protocol=sparkdash")
        else:
            i = argv.index("--protocol"); del argv[i:i + 2]
        run = SKILL / "compat/sparkdash/scripts/run.py"
        if not any(x == "--base-url" or x.startswith("--base-url=") for x in argv):
            argv += ["--base-url", os.environ.get("SPARK_BENCH_BASE_URL") or DEFAULT_BASE_URL]
        sys.exit(subprocess.call([sys.executable, str(run), *argv]))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("SPARK_BENCH_BASE_URL") or DEFAULT_BASE_URL,
                    help="OpenAI-compatible server root (default: $SPARK_BENCH_BASE_URL or "
                         f"{DEFAULT_BASE_URL}); a trailing /v1 is accepted")
    ap.add_argument("--api-key", default=None,
                    help="bearer token (default: $SPARK_BENCH_API_KEY or $OPENAI_API_KEY)")
    ap.add_argument("--model", default=None, help="default: first id from /v1/models")
    ap.add_argument("--label", default="", help="deployment label for the chart, e.g. '1x GPU, vLLM, FP8'")
    ap.add_argument("--preset", choices=PRESETS, default="quick")
    ap.add_argument("--suites")
    ap.add_argument("--concurrency", help="decode waves, e.g. 1,2,4")
    ap.add_argument("--decode-tokens", type=int)
    ap.add_argument("--depths", help="context sizes in tokens, e.g. 4096,32768,131072")
    ap.add_argument("--depth-samples", type=int)
    ap.add_argument("--depth-gen-tokens", type=int, default=128)
    ap.add_argument("--gsm8k-n", type=int)
    ap.add_argument("--mmlu-per-subject", type=int)
    ap.add_argument("--quality-concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--thinking", action="store_true", help="leave reasoning on (default off)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-request seconds")
    ap.add_argument("--force", action="store_true", help="run even if the server is busy")
    ap.add_argument("--protocol", choices=["native", "sparkdash"], default="native",
                    help="sparkdash = run the credited sparkDash-compatible mode instead "
                         "(its own flags; see compat/sparkdash/scripts/run.py --help)")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--handle", default=os.environ.get("SPARK_BENCH_HANDLE", ""),
                    help="credit printed on the chart before 'spark-bench' (default: none)")
    ap.add_argument("--out", help="results dir (default: $SPARK_BENCH_RESULTS or "
                                  "./spark-bench-results, then <timestamp>-<model>-<label>)")
    a = ap.parse_args(argv)
    if a.api_key:
        global API_KEY
        API_KEY = a.api_key

    p = dict(PRESETS[a.preset])
    for k in ("suites", "decode_tokens", "depth_samples", "gsm8k_n", "mmlu_per_subject"):
        if getattr(a, k) is not None:
            p[k] = getattr(a, k)
    if a.concurrency:
        p["concurrency"] = [int(x) for x in a.concurrency.split(",")]
    if a.depths:
        p["depths"] = [int(x) for x in a.depths.split(",")]
    suites = [s.strip() for s in p["suites"].split(",") if s.strip()]

    base = root_url(a.base_url)
    models = get_json(base + "/v1/models") or {}
    model = a.model or ((models.get("data") or [{}])[0].get("id"))
    if not model:
        sys.exit(f"no model at {base}/v1/models; pass --model")
    h0 = health(base) or {}
    has_counters = counters(h0).get("requests_total") is not None
    if not has_counters:
        log("server /health exposes no request counters: contention detection and "
            "draft acceptance are unavailable for this run; keep other clients off the server")
    busy = h0.get("requests_running", 0)
    if busy and not a.force:
        sys.exit(f"server busy (requests_running={busy}); wait, or pass --force (results flagged)")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", f"{model}-{a.label}").strip("-")[:60]
    out = Path(a.out) if a.out else results_root() / f"{ts}-{slug}"
    out.mkdir(parents=True, exist_ok=True)
    ep = Endpoint(base, model, a.timeout)
    summary = {"tool": VERSION, "started": datetime.now().isoformat(timespec="seconds"),
               "host": socket.gethostname(), "base_url": base, "model": model,
               "label": a.label, "preset": a.preset, "thinking": a.thinking,
               "server": {k: h0.get(k) for k in ("backend", "context_length", "streams")},
               "config": {**p, "suites": suites, "quality_concurrency": a.quality_concurrency,
                          "seed": a.seed, "depth_gen_tokens": a.depth_gen_tokens},
               "contention_check": has_counters,
               "busy_at_start": busy}
    log(f"{VERSION} -> {model} @ {base}  suites={suites}  out={out}")
    t0 = time.time()
    if "decode" in suites:
        summary["decode"] = run_decode(ep, base, p["concurrency"], p["decode_tokens"], a.thinking)
    if "depth" in suites:
        summary["depth"] = run_depth(ep, base, p["depths"], p["depth_samples"],
                                     a.depth_gen_tokens, a.thinking, h0.get("context_length"))
    if "quality" in suites:
        summary["quality"] = run_quality(ep, base, a.seed, p["gsm8k_n"], p["mmlu_per_subject"],
                                         a.quality_concurrency, a.thinking, out / "quality_items.jsonl")
    summary["rejected_request_fields"] = sorted(ep.rejected)
    summary["wall_s"] = round(time.time() - t0, 1)
    summary["contended"] = bool(busy) or any(
        x.get("contended") for x in summary.get("decode", []) + summary.get("depth", [])
    ) or bool((summary.get("quality") or {}).get("contended"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done in {summary['wall_s']} s -> {out/'summary.json'}"
        + ("  (some windows CONTENDED)" if summary["contended"] else ""))
    if not a.no_chart:
        py = chart_python()
        if py:
            subprocess.call([py, str(Path(__file__).parent / "chart.py"), "--handle", a.handle,
                             "--summary", str(out / "summary.json"), "--out", str(out / "chart.png")])
        else:
            log("chart skipped: no interpreter with matplotlib found "
                "(set SPARK_BENCH_CHART_PYTHON, or run with --no-chart)")


if __name__ == "__main__":
    main()
