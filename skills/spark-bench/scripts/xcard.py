#!/usr/bin/env python3
"""
Social results card from a spark-bench summary.json (needs Pillow only).

Built for a social timeline: 1600x900 (16:9, shown uncropped), a pixel-style
model heading, the conclusion as a headline, two tables set in one type scale
(generation on the left, long prompts on the right), a row of stat tiles
(quality, draft acceptance), one conditions strip, optional repo and handle at
the foot ("spark-bench" is always credited). No footnotes.

  python3 xcard.py --summary spark-bench-results/<run>/summary.json --out card.png \
      --title "My-Model-7B" --eyebrow "1x NVIDIA DGX Spark  ·  vLLM" \
      [--repo owner/recipe-repo] [--handle @yourhandle] \
      [--headline "..."] [--quality-from spark-bench-results/<run2>/summary.json]

Guards: a contended run gets a DRAFT stamp; quality sets with fewer than
--min-quality-n items are left off (too few to say anything).
"""
import argparse, json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
BG, TILE, RULE = (20, 20, 20), (32, 32, 30), (56, 56, 52)
YELLOW, AMBER, WHITE, GREY = (255, 224, 0), (206, 135, 51), (243, 240, 230), (170, 169, 159)
RED = (230, 57, 70)
# DejaVu fonts: the usual Linux location, then matplotlib's bundled copy, then
# SPARK_BENCH_FONT_DIR; Pillow's built-in font is the last resort.
_FONT_ENV = os.environ.get("SPARK_BENCH_FONT_DIR")
FONT_DIRS = [Path(_FONT_ENV)] if _FONT_ENV else []
FONT_DIRS += [Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu"),
              Path("/usr/local/share/fonts"), Path("/Library/Fonts")]
try:  # optional
    import matplotlib
    FONT_DIRS.append(Path(matplotlib.get_data_path()) / "fonts/ttf")
except Exception:
    pass

# one type scale for both tables
T_SECTION, T_GROUP, T_COL, T_LABEL, T_NUM = 28, 19, 19, 29, 33
ROW_H = 80


def font(size, bold=False, mono=False):
    name = "DejaVuSans" + ("Mono" if mono else "") + ("-Bold" if bold else "") + ".ttf"
    for d in FONT_DIRS:
        if (d / name).exists():
            return ImageFont.truetype(str(d / name), size)
    return ImageFont.load_default(size)  # Pillow >= 10.1


def tw(d, s, f):
    return d.textbbox((0, 0), s, font=f)[2]


def rtext(d, x_right, y, s, f, fill):
    d.text((x_right - tw(d, s, f), y), s, font=f, fill=fill)


def ctext(d, x_mid, y, s, f, fill):
    d.text((x_mid - tw(d, s, f) // 2, y), s, font=f, fill=fill)


def pixel_heading(img, text, x, y, max_w, max_h=88):
    """Low-res monospace render, thresholded, enlarged nearest-neighbour; amber offset behind."""
    f = font(20, bold=True, mono=True)
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    bw, bh = tw(probe, text, f) + 4, 28
    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).text((2, 1), text, font=f, fill=255)
    mask = mask.point(lambda v: 255 if v > 110 else 0)
    scale = max(2, min(int(max_w / bw), int(max_h / bh)))
    big = mask.resize((bw * scale, bh * scale), Image.NEAREST)
    off = max(3, scale)
    img.paste(Image.new("RGB", big.size, AMBER), (x + off, y + off), big)
    img.paste(Image.new("RGB", big.size, YELLOW), (x, y), big)
    return big.size[1]


def kfmt(n):
    return f"{round(n / 1024)}k" if n >= 1024 else str(n)


def num(v):
    return "-" if v is None else f"{v:,.0f}"


def secs(v):
    if v is None:
        return "-"
    return f"{v:.2f} s" if v < 1 else f"{v:.1f} s" if v < 10 else f"{v:.0f} s"


def table(d, x0, y, title, groups, cols, rows, bar=None):
    """
    groups: [(label, first_col, last_col, colour)] spanning numeric columns
    cols:   [(header, right_x, width, colour)] right-aligned numeric columns
    rows:   [(label, [str values], bar_fraction or None)]
    bar:    (x_start, x_end) for an optional bar after the label
    Returns the y below the last row.
    """
    d.text((x0, y), title, font=font(T_SECTION, bold=True), fill=WHITE)
    y += 46
    if groups:
        fg = font(T_GROUP, bold=True)
        for lab, a, b, col in groups:
            left, right = cols[a][1] - cols[a][2], cols[b][1]
            ctext(d, (left + right) // 2, y, lab, fg, col)
            d.line([(left + 4, y + 29), (right, y + 29)], fill=col, width=2)
        y += 38
    fc = font(T_COL)
    for head, rx, _, _ in cols:
        rtext(d, rx, y, head, fc, GREY)
    y += 36
    fl, fn = font(T_LABEL), font(T_NUM, bold=True)
    for label, vals, frac in rows:
        d.text((x0, y + (ROW_H - 36) // 2), label, font=fl, fill=WHITE)
        if bar and frac is not None:
            bx0, bx1 = bar
            d.rectangle([bx0, y + 16, bx0 + max(4, int((bx1 - bx0) * frac)), y + ROW_H - 16],
                        fill=YELLOW)
        for (head, rx, _, colr), v in zip(cols, vals):
            rtext(d, rx, y + (ROW_H - 40) // 2, v, fn, colr)
        y += ROW_H
    return y


def tile(d, x0, x1, y0, y1, big, small, colour):
    d.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=TILE)
    ctext(d, (x0 + x1) // 2, y0 + 12, big, font(46, bold=True), colour)
    ctext(d, (x0 + x1) // 2, y1 - 34, small, font(19), GREY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="", help="owner/name; printed as github.com/owner/name")
    ap.add_argument("--handle", default=os.environ.get("SPARK_BENCH_HANDLE", ""),
                    help="credit shown before 'spark-bench' in the footer (default: none)")
    ap.add_argument("--eyebrow", default="Measured with spark-bench")
    ap.add_argument("--headline", default=None)
    ap.add_argument("--title", default=None, help="heading text (default: model id)")
    ap.add_argument("--quality-from", default=None, help="take the quality block from another run")
    ap.add_argument("--min-quality-n", type=int, default=50)
    a = ap.parse_args()

    s = json.loads(Path(a.summary).read_text())
    q_src = json.loads(Path(a.quality_from).read_text()) if a.quality_from else s
    dec, depth = s.get("decode", []), s.get("depth", [])
    qsets = {n: v for n, v in ((q_src.get("quality") or {}).get("sets") or {}).items()
             if v["n"] >= a.min_quality_n}
    contended = bool(s.get("contended") or (a.quality_from and q_src.get("contended")))

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    X0, X1 = 64, W - 64

    # ---- header
    d.text((X0, 30), a.eyebrow, font=font(22, bold=True), fill=AMBER)
    y = 64 + pixel_heading(img, a.title or s["model"], X0, 64, X1 - X0) + 16

    concs = sorted({x["concurrency"] for x in dec})
    cmax = max(concs) if concs else 1
    by = {(x["workload"], x["concurrency"]): x for x in dec}
    workloads = list(dict.fromkeys(x["workload"] for x in dec))
    total = lambda x: x.get("aggregate_decode_tps") or x["aggregate_tps"]

    if a.headline:
        headline = a.headline
    else:
        parts = []
        ones = [by[(w, 1)]["per_stream_tps_mean"] for w in workloads if (w, 1) in by]
        if ones:
            parts.append(f"{max(ones):.0f} tok/s on one stream")
        tots = [total(by[(w, cmax)]) for w in workloads if (w, cmax) in by]
        if cmax > 1 and tots:
            parts.append(f"{max(tots):.0f} tok/s across {cmax} streams")
        if depth:
            dd = depth[-1]
            parts.append(f"{kfmt(dd['prompt_tokens'])} prompt read in {secs(dd['ttft_s'])}")
        headline = ", ".join(parts)
    size = 38
    while tw(d, headline, font(size, bold=True)) > X1 - X0:
        if size > 32:
            size -= 1
        elif not a.headline and ", " in headline:
            headline, size = headline.rsplit(", ", 1)[0], 38
        else:
            break
    d.text((X0, y), headline, font=font(size, bold=True), fill=WHITE)
    y += 52
    if s.get("label"):
        d.text((X0, y), s["label"], font=font(24), fill=GREY)
        y += 36
    y += 10
    d.line([(X0, y), (X1, y)], fill=RULE, width=2)
    top = y + 20
    bottom = H - 128  # body area ends here; conditions strip and footer below

    # ---- left: generation table
    LX0, LX1 = X0, 940
    if dec:
        cols = [("tok/s", LX1 - 380, 90, WHITE), ("1st token", LX1 - 225, 120, YELLOW)]
        groups = [("1 stream", 0, 1, GREY)]
        if cmax > 1:
            cols += [("each", LX1 - 115, 90, AMBER), ("total", LX1, 90, AMBER)]
            groups.append((f"{cmax} streams at once", 2, 3, AMBER))
        vmax = max([by[(w, 1)]["per_stream_tps_mean"] for w in workloads if (w, 1) in by] + [1])
        rows = []
        for w in workloads:
            o, m = by.get((w, 1)), by.get((w, cmax))
            vals = [num(o and o["per_stream_tps_mean"]), secs(o and o["ttft_p50_s"])]
            if cmax > 1:
                vals += [num(m and m["per_stream_tps_mean"]), num(m and total(m))]
            rows.append((w, vals, (o["per_stream_tps_mean"] / vmax) if o else None))
        table(d, LX0, top, "Generation, tok/s", groups, cols, rows,
              bar=(LX0 + 122, LX1 - 500))

    # ---- right: long prompts table, then tiles pinned to the body bottom
    RX0, RX1 = 1010, X1
    d.line([(RX0 - 36, top + 4), (RX0 - 36, bottom)], fill=RULE, width=2)
    if depth:
        cols = [("1st token", RX1 - 235, 120, YELLOW), ("read", RX1 - 105, 95, WHITE),
                ("write", RX1, 80, AMBER)]
        rows = [(kfmt(x["prompt_tokens"]), [secs(x["ttft_s"]), num(x["prefill_tps"]),
                                            num(x["decode_tps"])], None)
                for x in depth[-3:]]
        table(d, RX0, top, "Long prompts", [("tok/s", 1, 2, GREY)], cols, rows)

    tiles = []
    names = {"gsm8k": "GSM8K", "mmlu": "MMLU"}
    for n, v in qsets.items():
        tiles.append((f"{v['accuracy'] * 100:.1f}%", f"{names.get(n, n)} ({v['n']} q)", YELLOW))
    accs = [x["draft_acceptance"] for x in dec if x.get("draft_acceptance")]
    if accs:
        tiles.append((f"{sum(accs) / len(accs):.0%}", "drafts accepted", AMBER))
    if tiles:
        gap = 16
        twid = (RX1 - RX0 - gap * (len(tiles) - 1)) // len(tiles)
        for i, (big, small, colr) in enumerate(tiles):
            x0 = RX0 + i * (twid + gap)
            tile(d, x0, x0 + twid, bottom - 104, bottom, big, small, colr)

    # ---- conditions strip and footer
    cfg, srv = s.get("config", {}), s.get("server") or {}
    cond = ["temperature 0", "thinking on" if s.get("thinking") else "thinking off"]
    if dec:
        cond.append(f"{cfg.get('decode_tokens')}-token answers")
    if depth:
        cond.append("uncached prompts")
    if srv.get("context_length"):
        cond.append(f"{srv['context_length'] / 1000:.0f}k window")
    cond.append(f"medians of {cfg['depth_samples']} runs" if (cfg.get("depth_samples") or 1) > 1
                else "single run")
    d.text((X0, H - 106), "  ·  ".join(cond), font=font(21), fill=GREY)
    d.line([(X0, H - 70), (X1, H - 70)], fill=RULE, width=2)
    if a.repo:
        d.text((X0, H - 56), "github.com/" + a.repo, font=font(24, bold=True), fill=WHITE)
    tail = "  ·  ".join(x for x in (a.handle, "spark-bench", s.get("started", "")[:10]) if x)
    rtext(d, X1, H - 54, tail, font(22), GREY)

    if contended:
        st = "DRAFT: server was busy, do not publish"
        f = font(26, bold=True)
        d.rectangle([X1 - tw(d, st, f) - 28, 24, X1, 66], fill=RED)
        d.text((X1 - tw(d, st, f) - 14, 29), st, font=f, fill=WHITE)

    img.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
