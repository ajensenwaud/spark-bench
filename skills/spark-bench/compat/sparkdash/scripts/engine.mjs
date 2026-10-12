#!/usr/bin/env node
/**
 * engine.mjs — run one sparkDash-protocol bench suite against any
 * OpenAI-compatible endpoint, without sparkDash installed or running.
 *
 * Uses the vendored sparkDash bench modules (MIT, Mia's AI Lab) so prompts,
 * item IDs, grading and metrics are byte-identical to the dashboard's.
 *
 * Usage:
 *   node engine.mjs --suite decode   --host H --port P --model M \
 *       [--types structured,prose,code,json] [--concurrencies 1,2,4] [--max-tokens 400] [--api-key K]
 *   node engine.mjs --suite prefill  --host H --port P --model M \
 *       [--sizes 4096,8192,16384,32768] [--context-length N] [--api-key K]
 *   node engine.mjs --suite quality  --host H --port P --model M \
 *       [--categories qa,reason,arith,track,gsm8k,mmlu,follow,long] [--concurrency 4]
 *       [--long-sizes 32768] [--long-items 2] [--context-length N] [--api-key K]
 *
 * Prints the final public job JSON on stdout. Progress goes to stderr.
 * Exit 0 = completed, 1 = failed/cancelled, 2 = timed out (partial results).
 */
import path from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import fs from "node:fs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VENDOR = path.join(HERE, "..", "vendor");
const STATE_DIR = arg("state-dir") || process.env.BENCH_STATE_DIR || path.join(HERE, ".state");
fs.mkdirSync(STATE_DIR, { recursive: true });

// Manager modules read these at import time — set BEFORE dynamic import.
process.env.BENCH_HISTORY_PATH = path.join(STATE_DIR, "decode-history.json");
process.env.BENCH_ACTIVE_PATH = path.join(STATE_DIR, "decode-active.json");
process.env.PREFILL_BENCH_HISTORY_PATH = path.join(STATE_DIR, "prefill-history.json");
process.env.PREFILL_BENCH_ACTIVE_PATH = path.join(STATE_DIR, "prefill-active.json");
process.env.QUALITY_BENCH_HISTORY_PATH = path.join(STATE_DIR, "quality-history.json");
process.env.QUALITY_BENCH_ACTIVE_PATH = path.join(STATE_DIR, "quality-active.json");

function arg(name, dflt = null) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return dflt;
  const v = process.argv[i + 1];
  if (v === undefined || v.startsWith("--")) return dflt;
  return v;
}
const list = (s, cast = (x) => x) =>
  String(s).split(",").map((x) => x.trim()).filter(Boolean).map(cast);

const suite = arg("suite");
const host = arg("host");
const port = parseInt(arg("port"), 10);
const model = arg("model");
const apiKey = arg("api-key");
const contextLength = arg("context-length") ? parseInt(arg("context-length"), 10) : null;
const timeoutMs = parseInt(arg("timeout") || "3600", 10) * 1000;

if (!suite || !host || !Number.isInteger(port) || port < 1 || port > 65535) {
  console.error("Required: --suite decode|prefill|quality --host H --port P [--model M]");
  process.exit(1);
}

const sparkId = `local-${process.pid}-${Date.now()}`;
const base = { sparkId, lanIp: host, port, modelId: model || null, apiKey };

const { decodeBenchManager } = await import(
  pathToFileURL(path.join(VENDOR, "server/collectors/DecodeBench.js")).href
);
const { prefillBenchManager } = await import(
  pathToFileURL(path.join(VENDOR, "server/collectors/PrefillBench.js")).href
);
const { qualityBenchManager } = await import(
  pathToFileURL(path.join(VENDOR, "server/collectors/QualityBench.js")).href
);

let job;
try {
  if (suite === "decode") {
    job = decodeBenchManager.start({
      ...base,
      concurrencies: list(arg("concurrencies", "1,2,4"), Number),
      maxTokens: parseInt(arg("max-tokens", "400"), 10),
      promptType: arg("types", "structured"), // one type per job; bench.py loops
    });
  } else if (suite === "prefill") {
    job = prefillBenchManager.start({
      ...base,
      contextSizes: list(arg("sizes", "4096,8192,16384,32768"), Number),
      contextLength,
    });
  } else if (suite === "quality") {
    job = qualityBenchManager.start({
      ...base,
      categories: list(arg("categories", "qa,reason,arith,track,gsm8k,mmlu")),
      concurrency: parseInt(arg("concurrency", "4"), 10),
      longSizes: arg("long-sizes") ? list(arg("long-sizes"), Number) : undefined,
      longItems: arg("long-items") ? parseInt(arg("long-items"), 10) : undefined,
      contextLength,
      label: arg("label", ""),
    });
  } else {
    console.error(`Unknown suite: ${suite}`);
    process.exit(1);
  }
} catch (err) {
  console.error(`start failed: ${err?.message || err}`);
  process.exit(1);
}

const mgr =
  suite === "decode" ? decodeBenchManager
  : suite === "prefill" ? prefillBenchManager
  : qualityBenchManager;

const benchId = job.benchId;
const t0 = Date.now();
let lastMsg = "";

while (true) {
  const j = mgr.getJob(benchId);
  if (!j) { console.error("job lost"); process.exit(1); }
  if (j.status !== "running") {
    process.stdout.write(JSON.stringify(j));
    process.exit(j.status === "completed" ? 0 : 1);
  }
  const msg = `${j.progress?.message || j.status} (${Math.round((Date.now() - t0) / 1000)}s)`;
  if (msg !== lastMsg) { console.error(msg); lastMsg = msg; }
  if (Date.now() - t0 > timeoutMs) {
    try { mgr.cancel(sparkId, benchId); } catch {}
    await new Promise((r) => setTimeout(r, 1500));
    const partial = mgr.getJob(benchId);
    process.stdout.write(JSON.stringify({ ...(partial || j), timedOut: true }));
    process.exit(2);
  }
  await new Promise((r) => setTimeout(r, 1000));
}