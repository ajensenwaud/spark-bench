#!/usr/bin/env python3
"""Fetch the public quality-test datasets spark-bench samples from.

  GSM8K test split  (1,319 items)  openai/grade-school-math, MIT
  MMLU test split   (14,042 items) hendrycks/test original data.tar, MIT

Both are stored untouched under data/ and are only ever *sampled* by
bench.py with its own seed, so the item selection is spark-bench's own.
Idempotent: skips files that already exist with the expected row count.
"""
import json, sys, time, urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/test.jsonl")
MMLU_TAR = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
MMLU_TOTAL = 14042


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "spark-bench/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def count_lines(p):
    return sum(1 for _ in p.open()) if p.exists() else 0


def fetch_gsm8k():
    p = DATA / "gsm8k_test.jsonl"
    if count_lines(p) == 1319:
        print("gsm8k: present"); return
    p.write_bytes(get(GSM8K_URL))
    print("gsm8k:", count_lines(p), "items")


def fetch_mmlu():
    """Original Hendrycks release: data.tar holds data/test/<subject>_test.csv."""
    import csv, io, tarfile, tempfile
    p = DATA / "mmlu_test.jsonl"
    if count_lines(p) == MMLU_TOTAL:
        print("mmlu: present"); return
    with tempfile.TemporaryDirectory(dir=DATA) as td:
        tar_path = Path(td) / "data.tar"
        print("mmlu: downloading data.tar (~166 MB)")
        req = urllib.request.Request(MMLU_TAR, headers={"User-Agent": "spark-bench/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r, tar_path.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        rows = []
        with tarfile.open(tar_path) as tf:
            members = sorted((m for m in tf.getmembers()
                              if m.name.startswith("data/test/") and m.name.endswith("_test.csv")),
                             key=lambda m: m.name)
            for m in members:
                subject = Path(m.name).name[: -len("_test.csv")]
                text = tf.extractfile(m).read().decode("utf-8")
                for i, rec in enumerate(csv.reader(io.StringIO(text))):
                    q, a, b, c, d, ans = rec
                    rows.append({"id": f"{subject}/{i}", "subject": subject, "question": q,
                                 "choices": [a, b, c, d], "answer": ans.strip()})
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("mmlu:", len(rows), "items")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    fetch_gsm8k()
    fetch_mmlu()
