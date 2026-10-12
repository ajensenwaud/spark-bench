#!/usr/bin/env python3
"""Render a spark-bench summary.json as a results card (PNG).

Layout: decode speed by workload and concurrency on the left; prefill rate,
decode speed at depth and quality accuracy stacked on the right. Charcoal
background, yellow bars, values printed on the bars, source line at the foot.
Needs matplotlib (bench.py calls it with the first interpreter that has it).
"""
import argparse, json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, PANEL, GRID = "#16171a", "#1e2024", "#2c2f35"
TEXT, MUTED = "#f2f2f2", "#9aa0a8"
YELLOW, AMBER, PALE = "#ffd60a", "#ff9f1c", "#fff3b0"
CONC_COLOURS = [PALE, YELLOW, AMBER, "#ff6b35", "#e63946"]


def style(ax, title):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", loc="left", pad=10)


def hbars(ax, labels, values, colours, texts, xmax_pad=1.45):
    y = list(range(len(labels)))[::-1]
    vals = [v or 0 for v in values]
    ax.barh(y, vals, color=colours, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=10)
    mx = max(vals) if vals and max(vals) > 0 else 1
    ax.set_xlim(0, mx * xmax_pad)
    for yi, v, t in zip(y, vals, texts):
        ax.text(v + mx * 0.015, yi, t, va="center", ha="left", color=TEXT, fontsize=9.5)


def k(n):
    return f"{round(n / 1024)}k" if n and n >= 1024 else str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--handle", default=os.environ.get("SPARK_BENCH_HANDLE", ""),
                    help="credit shown before 'spark-bench' in the source line (default: none)")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())

    fig = plt.figure(figsize=(16, 9.6), dpi=150, facecolor=BG)
    gs = fig.add_gridspec(3, 2, left=0.11, right=0.97, top=0.79, bottom=0.11,
                          width_ratios=[1.15, 1], hspace=0.55, wspace=0.42)

    fig.text(0.03, 0.935, s["model"], color=YELLOW, fontsize=30, fontweight="bold",
             family="monospace")
    cfg = s.get("config", {})
    sub = s.get("label") or s.get("base_url", "")
    fig.text(0.03, 0.892, sub, color=TEXT, fontsize=14)
    strip = [f"temp 0", "thinking on" if s.get("thinking") else "thinking off"]
    if "decode" in s:
        strip.append(f"decode {cfg.get('decode_tokens')} tok/request")
    srv = s.get("server") or {}
    if srv.get("context_length"):
        strip.append(f"window {k(srv['context_length'])}")
    if (srv.get("streams") or {}).get("max"):
        strip.append(f"{srv['streams']['max']} streams max")
    fig.text(0.03, 0.858, "  ·  ".join(strip), color=MUTED, fontsize=11)

    # Decode panel (left, full height)
    if s.get("decode"):
        ax = fig.add_subplot(gs[:, 0])
        style(ax, "Decode speed (tok/s per stream)")
        rows = s["decode"]
        concs = sorted({x["concurrency"] for x in rows})
        labels, vals, cols, texts = [], [], [], []
        for x in rows:
            labels.append(f"{x['workload']} · {x['concurrency']} stream{'s' if x['concurrency'] > 1 else ''}")
            vals.append(x["per_stream_tps_mean"])
            cols.append(CONC_COLOURS[concs.index(x["concurrency"]) % len(CONC_COLOURS)])
            t = f"{x['per_stream_tps_mean']:.1f}"
            if x["concurrency"] > 1:
                t += f"   ({x['aggregate_decode_tps'] or x['aggregate_tps']:.0f} total)"
            if x.get("contended"):
                t += "  *busy"
            texts.append(t)
        hbars(ax, labels, vals, cols, texts, xmax_pad=1.6)

    right = 0
    if s.get("depth"):
        d = s["depth"]
        ax = fig.add_subplot(gs[right, 1]); right += 1
        style(ax, "Prefill rate by context size (tok/s)")
        hbars(ax, [k(x["prompt_tokens"]) + " ctx" for x in d], [x["prefill_tps"] for x in d],
              [AMBER] * len(d),
              [f"{x['prefill_tps']:,.0f}   ({x['ttft_s']:.1f} s to first token)" for x in d],
              xmax_pad=1.9)
        ax = fig.add_subplot(gs[right, 1]); right += 1
        style(ax, "Decode speed at depth (tok/s, 1 stream)")
        hbars(ax, [k(x["prompt_tokens"]) + " ctx" for x in d], [x["decode_tps"] for x in d],
              [YELLOW] * len(d), [f"{x['decode_tps']:.1f}" for x in d], xmax_pad=1.5)

    q = (s.get("quality") or {}).get("sets") or {}
    if q:
        ax = fig.add_subplot(gs[right if right < 3 else 2, 1])
        style(ax, "Quality (accuracy, 95% interval)")
        names = {"gsm8k": "GSM8K", "mmlu": "MMLU"}
        labels = [f"{names[n]} (n={q[n]['n']})" for n in q]
        vals = [q[n]["accuracy"] * 100 for n in q]
        texts = [f"{q[n]['accuracy']:.1%}   ({q[n]['ci95'][0]:.0%}–{q[n]['ci95'][1]:.0%})" for n in q]
        hbars(ax, labels, vals, [YELLOW] * len(vals), texts)
        ax.set_xlim(0, 160)
        ax.set_xticks([0, 25, 50, 75, 100])

    notes = []
    accs = [x.get("draft_acceptance") for x in s.get("decode", []) if x.get("draft_acceptance")]
    if accs:
        notes.append(f"speculative drafts accepted {sum(accs) / len(accs):.0%}")
    if s.get("depth"):
        cached = [c for x in s["depth"] for c in (x.get("cached_prompt_tokens") or []) if c is not None]
        if cached:
            notes.append(f"prompt cache hits during prefill tests: {max(cached)} tokens max")
    if s.get("contended"):
        notes.append("*busy = other requests hit the server during that measurement")
    if notes:
        fig.text(0.03, 0.055, "  ·  ".join(notes), color=MUTED, fontsize=10.5)
    credit = ", ".join(x for x in (a.handle.strip(), "spark-bench") if x)
    fig.text(0.03, 0.022, f"Source: {credit}", color=MUTED, fontsize=10.5)
    fig.text(0.97, 0.022, s.get("started", "")[:10], color=MUTED, fontsize=10.5, ha="right")
    fig.savefig(a.out, facecolor=BG)
    print(a.out)


if __name__ == "__main__":
    main()
