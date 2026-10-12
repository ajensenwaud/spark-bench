#!/usr/bin/env python3
"""
chart.py — render the benchmark summary as a dark exec-style PNG.

Conventions: horizontal bars, values printed on the bars, dark
panel background, source line underneath. Measured figures only — no
reported/derived numbers mixed in.

Layout: decode (per prompt type x concurrency) | prefill sweep + quality.
"""
import argparse, json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BG = "#0f1419"
PANEL = "#161c24"
BAR = "#3b82f6"
BAR2 = "#60a5fa"
ACCENT = "#22d3ee"
GOOD = "#34d399"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
GRID = "#243044"

TYPE_COLORS = {"structured": "#3b82f6", "prose": "#60a5fa",
               "code": "#a78bfa", "json": "#22d3ee"}


def style_axes(ax, title):
    ax.set_facecolor(PANEL)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", loc="left", pad=10)


def bar_label(ax, y, width, text, color=TEXT, size=10):
    ax.text(width, y, f"  {text}", va="center", ha="left",
            color=color, fontsize=size, fontweight="bold")


def draw_decode(ax, decode):
    """One horizontal bar per (type, concurrency): aggregate decode tok/s."""
    rows = []  # (label, agg, per_stream, color)
    for t, blk in decode.items():
        for w in blk.get("waves", []):
            if w.get("streamsOk"):
                rows.append((f"{t} · c{w['concurrency']}",
                             w["aggregateDecodeTps"], w["meanDecodeTps"],
                             TYPE_COLORS.get(t, BAR)))
    if not rows:
        ax.set_visible(False)
        return
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows],
            color=[r[3] for r in rows], height=0.62)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], color=TEXT, fontsize=10)
    ax.invert_yaxis()
    mx = max(r[1] for r in rows)
    for i, r in enumerate(rows):
        bar_label(ax, i, r[1], f"{r[1]:,.0f} agg · {r[2]:,.0f}/stream")
    ax.set_xlim(0, mx * 1.55)
    style_axes(ax, "Decode throughput by prompt type × concurrency")
    ax.set_xlabel("aggregate decode tok/s (client-measured)", color=MUTED, fontsize=10)


def draw_prefill(ax, prefill):
    pts = [p for p in prefill.get("points", []) if p.get("prefillTps")]
    if not pts:
        ax.set_visible(False)
        return
    labels = [f"{p['targetTokens'] // 1024}k" for p in pts]
    ys = range(len(pts))
    ax.barh(list(ys), [p["prefillTps"] for p in pts],
            color=ACCENT, height=0.55)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, color=TEXT, fontsize=11)
    ax.invert_yaxis()
    mx = max(p["prefillTps"] for p in pts)
    for i, p in enumerate(pts):
        bar_label(ax, i, p["prefillTps"],
                  f"{p['prefillTps']:,.0f} t/s · TTFT {p['ttftMs']:,.0f} ms")
    ax.set_xlim(0, mx * 1.65)
    style_axes(ax, "Prefill sweep (unique prefix, 3 samples/size)")
    ax.set_xlabel("prefill tok/s", color=MUTED, fontsize=10)


def draw_quality(ax, quality):
    cats = quality.get("categories") or {}
    if not cats:
        ax.set_visible(False)
        return
    labels, vals = [], []
    for k, v in cats.items():
        labels.append(k)
        vals.append(v["pct"])
    ys = range(len(vals))
    ax.barh(list(ys), vals, color=GOOD, height=0.55)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, color=TEXT, fontsize=11)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        bar_label(ax, i, v, f"{v:.1f}%")
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 25, 50, 75, 100])
    overall = quality.get("overallPct")
    style_axes(ax, f"Quality suite by category"
               + (f"  ·  overall {overall:.1f}%" if overall is not None else ""))
    ax.set_xlabel("score %", color=MUTED, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s = json.loads(Path(args.summary).read_text())
    suites = s.get("suites", {})

    fig = plt.figure(figsize=(17, 11.5), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1],
                  hspace=0.42, wspace=0.28,
                  left=0.10, right=0.97, top=0.88, bottom=0.10)

    ax_d = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1])

    if "decode" in suites:
        draw_decode(ax_d, suites["decode"])
    if "prefill" in suites:
        draw_prefill(ax_p, suites["prefill"])
    if "quality" in suites:
        draw_quality(ax_q, suites["quality"])

    model = s.get("model", "?")
    target = s.get("target", "?")
    preset = s.get("preset", "full")
    fig.suptitle(f"Model benchmark — {model}", x=0.10, ha="left",
                 color=TEXT, fontsize=20, fontweight="bold", y=0.955)
    fig.text(0.10, 0.912,
             f"sparkDash protocol (decode / prefill / quality) · preset {preset}"
             f" · target {target}",
             color=MUTED, fontsize=12)

    caveat = ""
    if s.get("contended"):
        caveat = " · CONTENDED run (serve busy at start)"
    ts = s.get("generatedAt", "")
    try:
        ts = datetime.strptime(ts, "%Y%m%d-%H%M%S").strftime("%d %b %Y %H:%M")
    except Exception:
        pass
    fig.text(0.10, 0.035,
             f"Source: measured with the sparkDash protocol (Mia's AI Lab), via spark-bench,"
             f" single run, temp 0 · {ts}{caveat}",
             color=MUTED, fontsize=10)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=BG)
    print(args.out)


if __name__ == "__main__":
    main()