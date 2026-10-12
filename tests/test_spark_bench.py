"""
spark-bench tests: unit checks plus end-to-end runs against the local mock
server (tests/mock_server.py). Standard library only; never contacts a real
model server.

  python3 -m unittest discover -s tests -v
"""
import importlib.util, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills/spark-bench"
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_server  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = load("bench")
compare = load("compare")


def has_module(mod):
    return subprocess.call([sys.executable, "-c", f"import {mod}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


class Unit(unittest.TestCase):
    def test_root_url(self):
        self.assertEqual(bench.root_url("http://h:1/v1/"), "http://h:1")
        self.assertEqual(bench.root_url("http://h:1"), "http://h:1")

    def test_default_base_url_is_local(self):
        self.assertEqual(bench.DEFAULT_BASE_URL, "http://localhost:8000")

    def test_wilson(self):
        lo, hi = bench.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(bench.wilson(0, 0), (None, None))

    def test_pct(self):
        self.assertEqual(bench.pct([1, 2, 3, 4], 0.5), 2.5)
        self.assertIsNone(bench.pct([None], 0.5))

    def test_grade_gsm8k(self):
        self.assertEqual(bench.grade_gsm8k("so 3+4. Answer: 7", 7.0), (True, 7.0))
        self.assertEqual(bench.grade_gsm8k("Answer: $1,234.00", 1234.0), (True, 1234.0))
        self.assertEqual(bench.grade_gsm8k("first 5 then 9", 9.0), (True, 9.0))
        self.assertFalse(bench.grade_gsm8k("Answer: 8", 7.0)[0])
        self.assertEqual(bench.grade_gsm8k("no numbers", 7.0), (False, None))

    def test_grade_mmlu(self):
        self.assertEqual(bench.grade_mmlu("B", "B"), (True, "B"))
        self.assertEqual(bench.grade_mmlu("The answer is (C).", "C"), (True, "C"))
        self.assertFalse(bench.grade_mmlu("A", "D")[0])

    def test_server_window(self):
        a = {"requests_total": 10, "drafted_total": 100, "accepted_total": 40, "cached_tokens_total": 5}
        b = {"requests_total": 14, "drafted_total": 200, "accepted_total": 100, "cached_tokens_total": 9}
        self.assertEqual(bench.server_window(a, b, 3),
                         {"foreign_requests": 1, "draft_acceptance": 0.6, "cached_prompt_tokens": 4})
        self.assertEqual(bench.server_window({}, {}, 3), {})  # server without counters

    def test_sample_items_seeded_and_stratified(self):
        a = bench.sample_items(20261010, 10, 1)
        b = bench.sample_items(20261010, 10, 1)
        c = bench.sample_items(1, 10, 1)
        self.assertEqual([x["id"] for x in a], [x["id"] for x in b])
        self.assertNotEqual([x["id"] for x in a], [x["id"] for x in c])
        self.assertEqual(sum(x["set"] == "gsm8k" for x in a), 10)
        self.assertEqual(len({x["subject"] for x in a if x["set"] == "mmlu"}), 57)

    def test_filler_length(self):
        import random
        self.assertEqual(len(bench.filler(500, random.Random(0)).split()), 500)

    def test_sign_test(self):
        self.assertEqual(compare.sign_test(0, 0), 1.0)
        self.assertAlmostEqual(compare.sign_test(10, 0), 2 / 1024)
        self.assertEqual(compare.sign_test(5, 5), 1.0)

    def test_compare_format_missing(self):
        self.assertEqual(compare.f(None, "8.1f"), "       -")
        self.assertEqual(compare.f(3.14159, "8.1f"), "     3.1")


class EndToEnd(unittest.TestCase):
    """bench.py and compare.py against the mock server on 127.0.0.1."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="spark-bench-test-"))
        cls.servers = []

    @classmethod
    def tearDownClass(cls):
        for s in cls.servers:
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def serve(self, *argv):
        srv, url = mock_server.start(list(argv))
        self.servers.append(srv)
        return url

    def run_bench(self, url, out, *extra, env=None):
        cmd = [sys.executable, str(SCRIPTS / "bench.py"), "--base-url", url, "--out", str(out),
               "--no-chart", "--concurrency", "1,2", "--decode-tokens", "24", "--depths", "512,2048",
               "--depth-samples", "1", "--depth-gen-tokens", "8", "--gsm8k-n", "12",
               "--mmlu-per-subject", "1", *extra]
        e = {k: v for k, v in os.environ.items()
             if k not in ("SPARK_BENCH_API_KEY", "OPENAI_API_KEY", "SPARK_BENCH_BASE_URL")}
        e.update(env or {})
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=e)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return json.loads((out / "summary.json").read_text()), p.stdout

    def test_full_run_with_counters_and_compare(self):
        url = self.serve("--counters", "--salt", "a")
        A, log = self.run_bench(url, self.tmp / "A", "--label", "mock A")
        self.assertTrue(A["contention_check"])
        self.assertFalse(A["contended"])
        self.assertEqual({(x["workload"], x["concurrency"]) for x in A["decode"]},
                         {(w, c) for w in bench.WORKLOADS for c in (1, 2)})
        for row in A["decode"]:
            self.assertEqual(row["completion_tokens"], [24] * row["concurrency"])
            self.assertGreater(row["per_stream_tps_mean"], 0)
            self.assertEqual(row["foreign_requests"], 0)
            self.assertEqual(row["draft_acceptance"], 0.5)
        self.assertEqual([x["target_tokens"] for x in A["depth"]], [512, 2048])
        self.assertTrue(all(x["prefill_tps"] for x in A["depth"]))
        q = A["quality"]["sets"]
        self.assertEqual((q["gsm8k"]["n"], q["mmlu"]["n"]), (12, 57))
        # oracle: the mock is right on exactly the items whose salted hash < 0.7
        items = [json.loads(l) for l in (self.tmp / "A/quality_items.jsonl").read_text().splitlines()]
        self.assertEqual(sum(x["correct"] for x in items), q["gsm8k"]["correct"] + q["mmlu"]["correct"])
        self.assertTrue(0.4 < (q["gsm8k"]["correct"] + q["mmlu"]["correct"]) / 69 < 0.95)
        self.assertEqual(sum(1 for x in items if x.get("error")), 0)

        url_b = self.serve("--counters", "--salt", "b")
        B, _ = self.run_bench(url_b, self.tmp / "B", "--label", "mock B", "--suites", "decode,quality")
        p = subprocess.run([sys.executable, str(SCRIPTS / "compare.py"), str(self.tmp / "A"),
                            str(self.tmp / "B")], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("DECODE", p.stdout)
        self.assertIn("QUALITY", p.stdout)
        self.assertNotIn("DEPTH", p.stdout)  # B ran no depth suite
        ia = {x["id"]: x["correct"] for x in items}
        ib = {x["id"]: x["correct"] for x in map(json.loads, (self.tmp / "B/quality_items.jsonl").read_text().splitlines())}
        fixed = sum(1 for i in ia if i.startswith("gsm8k/") and ib[i] and not ia[i])
        broke = sum(1 for i in ia if i.startswith("gsm8k/") and ia[i] and not ib[i])
        line = next(l for l in p.stdout.splitlines() if l.strip().startswith("gsm8k"))
        self.assertEqual(line.split()[4:6], [str(fixed), str(broke)])
        print("\n" + p.stdout)

    def test_plain_server_degrades_gracefully(self):
        # no /health, rejects chat_template_kwargs and ignore_eos, requires an API key
        url = self.serve("--reject-field", "chat_template_kwargs", "--reject-field", "ignore_eos",
                         "--api-key", "sekret")
        S, log = self.run_bench(url, self.tmp / "plain", "--suites", "decode,depth",
                                env={"SPARK_BENCH_API_KEY": "sekret"})
        self.assertFalse(S["contention_check"])
        self.assertIn("no request counters", log)
        self.assertEqual(S["rejected_request_fields"], ["chat_template_kwargs", "ignore_eos"])
        self.assertTrue(all("foreign_requests" not in x for x in S["decode"]))
        self.assertTrue(all(x["per_stream_tps_mean"] > 0 for x in S["decode"]))

    def test_wrong_api_key_fails_cleanly(self):
        url = self.serve("--api-key", "sekret")
        p = subprocess.run([sys.executable, str(SCRIPTS / "bench.py"), "--base-url", url,
                            "--api-key", "wrong", "--no-chart", "--out", str(self.tmp / "x")],
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("no model at", p.stderr)

    def test_busy_server_aborts_without_force(self):
        url = self.serve("--counters", "--token-delay", "0.05")
        import threading, urllib.request
        body = json.dumps({"model": "mock-model", "max_tokens": 60, "ignore_eos": True, "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        t = threading.Thread(target=lambda: urllib.request.urlopen(urllib.request.Request(
            url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})).read())
        t.start()
        import time; time.sleep(0.3)
        p = subprocess.run([sys.executable, str(SCRIPTS / "bench.py"), "--base-url", url, "--no-chart",
                            "--out", str(self.tmp / "busy")], capture_output=True, text=True, timeout=60)
        t.join()
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("server busy", p.stderr)

    @unittest.skipUnless(has_module("PIL"), "Pillow not installed for this interpreter")
    def test_xcard_renders(self):
        url = self.serve("--counters")
        S, _ = self.run_bench(url, self.tmp / "card", "--suites", "decode,depth")
        out = self.tmp / "card.png"
        p = subprocess.run([sys.executable, str(SCRIPTS / "xcard.py"), "--summary",
                            str(self.tmp / "card/summary.json"), "--out", str(out), "--title", "mock-model",
                            "--handle", "@tester"], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        from PIL import Image
        with Image.open(out) as im:
            self.assertEqual(im.size, (1600, 900))

    @unittest.skipUnless(has_module("matplotlib"), "matplotlib not installed for this interpreter")
    def test_chart_renders(self):
        url = self.serve("--counters")
        S, _ = self.run_bench(url, self.tmp / "chart", "--suites", "decode,depth")
        out = self.tmp / "chart.png"
        p = subprocess.run([sys.executable, str(SCRIPTS / "chart.py"), "--summary",
                            str(self.tmp / "chart/summary.json"), "--out", str(out)],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(out.stat().st_size > 0)

    @unittest.skipUnless(shutil.which("node"), "Node.js not installed")
    def test_sparkdash_protocol_mode(self):
        url = self.serve("--counters")
        out = self.tmp / "sd"
        p = subprocess.run([sys.executable, str(SCRIPTS / "bench.py"), "--protocol", "sparkdash",
                            "--base-url", url, "--suites", "decode,prefill", "--decode-types", "prose",
                            "--decode-concurrencies", "1", "--decode-max-tokens", "16",
                            "--prefill-sizes", "1024", "--no-chart", "--out", str(out)],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        s = json.loads((out / "summary.json").read_text())
        self.assertEqual(s["model"], "mock-model")
        self.assertIn("decode", s["suites"])
        self.assertTrue((out / "state").is_dir())  # job state kept out of the skill dir


if __name__ == "__main__":
    unittest.main()
