#!/usr/bin/env python3
"""Compare two spark-bench runs (A = baseline, B = candidate).

  python3 compare.py spark-bench-results/<A>/ spark-bench-results/<B>/

Prints decode, depth and quality deltas. Quality is compared item by item on
the items both runs share (same seed => same items): it counts items B fixed,
items B broke, and gives an exact two-sided sign-test p-value on those flips,
so a 2-point accuracy change can be told apart from noise.
"""
import json, math, sys
from pathlib import Path


def load(d):
    d = Path(d)
    s = json.loads((d / "summary.json").read_text())
    items = {}
    q = d / "quality_items.jsonl"
    if q.exists():
        with q.open(encoding="utf-8") as fh:
            for line in fh:
                x = json.loads(line)
                items[x["id"]] = x["correct"]
    return s, items


def pct(a, b):
    if not a or b is None:
        return "  n/a"
    return f"{(b - a) / a * 100:+6.1f}%"


def f(v, spec):
    """Format a number, or a right-aligned '-' of the same width when it is missing."""
    if isinstance(v, (int, float)):
        return format(v, spec)
    return "-".rjust(int(spec.split(".")[0]))


def sign_test(fixed, broke):
    n = fixed + broke
    if n == 0:
        return 1.0
    k = min(fixed, broke)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    (A, ia), (B, ib) = load(sys.argv[1]), load(sys.argv[2])
    name = lambda s: f"{s['model']} [{s.get('label') or s['base_url']}]"
    print(f"A: {name(A)}  {A['started']}")
    print(f"B: {name(B)}  {B['started']}\n")
    for s, tag in ((A, "A"), (B, "B")):
        if s.get("contended"):
            print(f"WARNING: run {tag} has contended windows; treat its speeds with caution")
        if s.get("contention_check") is False:
            print(f"NOTE: run {tag}'s server exposed no request counters, so contention was not checked")

    da = {(x["workload"], x["concurrency"]): x for x in A.get("decode", [])}
    db = {(x["workload"], x["concurrency"]): x for x in B.get("decode", [])}
    keys = [k for k in da if k in db]
    if keys:
        print("\nDECODE (tok/s per stream)          A        B     change")
        for k in keys:
            a, b = da[k]["per_stream_tps_mean"], db[k]["per_stream_tps_mean"]
            print(f"  {k[0]:<7} {k[1]:>2} stream(s)        {f(a, '8.1f')} {f(b, '8.1f')}   {pct(a, b)}")

    pa = {x["target_tokens"]: x for x in A.get("depth", [])}
    pb = {x["target_tokens"]: x for x in B.get("depth", [])}
    keys = [k for k in pa if k in pb]
    if keys:
        print("\nDEPTH                    prefill A  prefill B   change |  decode A  decode B   change")
        for k in keys:
            a, b = pa[k], pb[k]
            print(f"  {k:>7} tokens       {f(a['prefill_tps'], '9.0f')}  {f(b['prefill_tps'], '9.0f')}  {pct(a['prefill_tps'], b['prefill_tps'])} |"
                  f"  {f(a['decode_tps'], '8.1f')}  {f(b['decode_tps'], '8.1f')}  {pct(a['decode_tps'], b['decode_tps'])}")

    qa = (A.get("quality") or {}).get("sets", {})
    qb = (B.get("quality") or {}).get("sets", {})
    if qa and qb:
        if A["quality"].get("seed") != B["quality"].get("seed"):
            print("\nNOTE: different quality seeds; item-level comparison covers shared items only")
        print("\nQUALITY            A         B     shared  B fixed  B broke  sign-test p")
        for name_ in qa:
            if name_ not in qb:
                continue
            shared = [i for i in ia if i in ib and i.startswith(name_ + "/")]
            fixed = sum(1 for i in shared if ib[i] and not ia[i])
            broke = sum(1 for i in shared if ia[i] and not ib[i])
            print(f"  {name_:<8} {qa[name_]['accuracy']:8.1%}  {qb[name_]['accuracy']:8.1%}  {len(shared):7d}"
                  f"  {fixed:7d}  {broke:7d}  {sign_test(fixed, broke):11.3f}")


if __name__ == "__main__":
    main()
