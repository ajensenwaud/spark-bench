#!/usr/bin/env python3
"""
Tiny stdlib mock of an OpenAI-compatible server, for testing spark-bench
without a GPU or a real model. Listens on 127.0.0.1 only.

  GET  /v1/models            -> one model id
  POST /v1/chat/completions  -> streaming (SSE) or plain responses, with usage
  GET  /health               -> TensorFold-style JSON counters (only with --counters)

Behaviour
  - Tokens are words: prompt_tokens = word count of the messages, and the
    reply is a sequence of "tokN" words, one SSE chunk each, --token-delay
    seconds apart (so tok/s figures are finite and roughly 1/delay).
  - ignore_eos=true generates exactly max_tokens; otherwise the reply stops
    after a short answer (finish_reason "stop").
  - Quality prompts are answered with an "oracle": the mock looks the question
    up in spark-bench's data files and answers correctly for a deterministic
    fraction of items (--accuracy, keyed on --salt), so grading and paired
    comparison can be checked against known values.
  - --reject-field NAME returns HTTP 400 when a request carries NAME, to
    exercise the client's optional-field fallback.
  - --api-key KEY requires "Authorization: Bearer KEY".

  python3 tests/mock_server.py --port 18080 --counters
"""
import argparse, hashlib, json, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "skills/spark-bench/data"


def load_oracle():
    gsm, mmlu = {}, {}
    with (DATA / "gsm8k_test.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            x = json.loads(line)
            gsm[x["question"].strip()] = x["answer"].split("####")[-1].strip().replace(",", "")
    with (DATA / "mmlu_test.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            x = json.loads(line)
            mmlu[x["question"].strip()] = x["answer"]
    return gsm, mmlu


class State:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.running = 0
        self.c = {"requests_total": 0, "drafted_total": 0, "accepted_total": 0,
                  "completion_tokens_total": 0, "cached_tokens_total": 0,
                  "prompt_tokens_total": 0}
        self.gsm, self.mmlu = load_oracle()

    def correct(self, question):
        h = hashlib.sha256((self.args.salt + question).encode()).digest()[0] / 256
        return h < self.args.accuracy

    def answer(self, prompt):
        """Short reply for a quality prompt, or None for anything else."""
        if "Answer: <number>" in prompt:
            q = prompt.split("\n\n", 1)[-1].strip()
            gold = self.gsm.get(q)
            if gold is not None:
                try:
                    g = float(gold)
                except ValueError:
                    return "Answer: 0"
                v = g if self.correct(q) else g + 1
                return f"Let me work it out. Answer: {v:g}"
        if prompt.startswith("Multiple-choice question"):
            body = prompt.split("\n\n", 1)[-1]
            q = body.rsplit("\n\nA. ", 1)[0].strip()
            gold = self.mmlu.get(q)
            if gold:
                if self.correct(q):
                    return gold
                return "ABCD"[("ABCD".index(gold) + 1) % 4]
        return None


def make_handler(st):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _authorised(self):
            key = st.args.api_key
            if key and self.headers.get("Authorization") != f"Bearer {key}":
                self._json(401, {"error": {"message": "bad api key"}})
                return False
            return True

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                if self._authorised():
                    self._json(200, {"object": "list", "data": [{"id": st.args.model, "object": "model"}]})
            elif self.path == "/health" and st.args.counters:
                with st.lock:
                    self._json(200, {"ok": True, "backend": "mock", "context_length": st.args.context_length,
                                     "requests_running": st.running, **st.c})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            if not self._authorised():
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            for f in st.args.reject_field:
                if f in body:
                    return self._json(400, {"error": {"message": f"unsupported field {f}"}})
            prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
            ptoks = len(prompt.split())
            max_tokens = int(body.get("max_tokens") or 16)
            ans = st.answer(prompt)
            if body.get("ignore_eos"):
                words, finish = [f"tok{i}" for i in range(max_tokens)], "length"
            elif ans is not None:
                words = ans.split()
                finish = "stop" if len(words) <= max_tokens else "length"
                words = words[:max_tokens]
            else:
                n = min(max_tokens, 8)
                words, finish = [f"tok{i}" for i in range(n)], ("length" if n == max_tokens else "stop")
            with st.lock:
                st.running += 1
                st.c["requests_total"] += 1
                st.c["prompt_tokens_total"] += ptoks
                st.c["completion_tokens_total"] += len(words)
                if st.args.counters:  # pretend a drafter proposes 2 tokens per output token, 1 accepted
                    st.c["drafted_total"] += 2 * len(words)
                    st.c["accepted_total"] += len(words)
            try:
                usage = {"prompt_tokens": ptoks, "completion_tokens": len(words),
                         "total_tokens": ptoks + len(words)}
                if not body.get("stream"):
                    return self._json(200, {"id": "mock", "object": "chat.completion",
                                            "choices": [{"index": 0, "finish_reason": finish,
                                                         "message": {"role": "assistant",
                                                                     "content": " ".join(words)}}],
                                            "usage": usage})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def ev(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()

                time.sleep(st.args.prefill_delay_per_ktok * ptoks / 1000)
                ev({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
                for i, w in enumerate(words):
                    ev({"choices": [{"index": 0, "delta": {"content": (" " if i else "") + w}}]})
                    time.sleep(st.args.token_delay)
                ev({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    ev({"choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            finally:
                with st.lock:
                    st.running -= 1

    return H


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--counters", action="store_true", help="serve TensorFold-style /health counters")
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--token-delay", type=float, default=0.002)
    ap.add_argument("--prefill-delay-per-ktok", type=float, default=0.01)
    ap.add_argument("--accuracy", type=float, default=0.7)
    ap.add_argument("--salt", default="a")
    ap.add_argument("--reject-field", action="append", default=[])
    ap.add_argument("--api-key", default="")
    return ap.parse_args(argv)


def start(argv=None):
    """Start in a background thread; returns (server, base_url)."""
    args = parse(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(State(args)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


if __name__ == "__main__":
    srv, url = start()
    print(f"mock server on {url}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
