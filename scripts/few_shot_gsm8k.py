#!/usr/bin/env python3
"""5-shot GSM8K over all 1319 test questions against one running server: InferPilot's G2 gate.

    python3 scripts/few_shot_gsm8k.py --api-url http://127.0.0.1:30000/v1 --tag baseline --result-root RUN_DIR/accuracy

The prompt is sglang's few-shot GSM8K (`sglang.test.few_shot_gsm8k`): the first five test questions with
their answers, then `Question: <q>\\nAnswer:`; greedy, 512 new tokens, stopping at "Question", "Assistant:"
or "<|separator|>"; the last integer of the reply against the last integer of the reference answer. It
goes through the OpenAI completions endpoint, so any server that has one works. Before the set the prefix
cache is flushed and after it the speculative accept length is read from /server_info (sglang). One line
goes to <result-root>/summary.txt in eval_accuracy.sh's format, the answers to
<result-root>/<tag>_gsm8k_5shot.jsonl. A request that still fails after its retries fails the run: a
missing question would change the denominator. Standard library only.

The defaults come from the environment, as for eval_accuracy.sh: API_URL (the OpenAI endpoint), MODEL
(default: the model the server serves), TAG and RESULT_ROOT (where the results go), WAIT_SERVER (seconds
to wait for a server that is still starting). The questions are downloaded once to
$XDG_CACHE_HOME/inferpilot/gsm8k_test.jsonl (XDG_CACHE_HOME defaults to ~/.cache).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
STOP = ["Question", "Assistant:", "<|separator|>"]
INVALID = None


def answer_value(text: str):
    """The last integer in text (commas dropped), as sglang reads it; None when there is none."""
    numbers = re.findall(r"\d+", text.replace(",", ""))
    return int(numbers[-1]) if numbers else INVALID


def example(row: dict, with_answer: bool) -> str:
    return "Question: " + row["question"] + "\nAnswer:" + (" " + row["answer"] if with_answer else "")


def prompts(rows: list, shots: int) -> list:
    head = "".join(example(rows[i], True) + "\n\n" for i in range(shots))
    return [head + example(row, False) for row in rows]


def load(path: str) -> list:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=120) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def call(url: str, payload: dict | None = None, timeout: float = 600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def wait(server: str, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            call(server + "/health", timeout=5)
            return
        except Exception:
            time.sleep(15)


def accept_length(server: str):
    try:
        states = call(server + "/server_info", timeout=20).get("internal_states") or []
    except Exception:
        return [None]
    return [s.get("avg_spec_accept_length") for s in states] or [None]


def complete(api: str, model: str, prompt: str, max_tokens: int, retries: int) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stop": STOP}
    for attempt in range(retries + 1):
        try:
            res = call(api + "/completions", payload)
            return {"text": res["choices"][0]["text"], "tokens": (res.get("usage") or {}).get("completion_tokens", 0)}
        except Exception as e:  # a transient error is retried, the last one is reported
            error = "%s: %s" % (type(e).__name__, e)
            time.sleep(min(2 ** attempt, 10))
    return {"error": error}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="5-shot GSM8K against an OpenAI-compatible server.")
    ap.add_argument("--api-url", default=os.environ.get("API_URL", "http://127.0.0.1:30000/v1"))
    ap.add_argument("--model", default=os.environ.get("MODEL", ""), help="default: the model the server serves")
    ap.add_argument("--tag", default=os.environ.get("TAG", "eval"))
    ap.add_argument("--result-root", default=os.environ.get("RESULT_ROOT", "./results"))
    ap.add_argument("--data-path", default=os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "inferpilot", "gsm8k_test.jsonl"))
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--parallel", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="at most this many questions: a smoke test, not a result")
    ap.add_argument("--wait-server", type=float, default=float(os.environ.get("WAIT_SERVER", 0)))
    a = ap.parse_args(argv)
    api = a.api_url.rstrip("/")
    server = re.sub(r"/v1$", "", api)
    if a.wait_server > 0:
        wait(server, a.wait_server)
    try:
        served = call(api + "/models", timeout=20)["data"][0]["id"]
    except Exception:
        print("gsm8k_5shot: no server answers at %s" % api, file=sys.stderr)
        return 1
    model = a.model or served
    rows = load(a.data_path)
    if a.limit:
        rows = rows[:a.limit]
    labels = [answer_value(r["answer"]) for r in rows]
    try:
        call(server + "/flush_cache", {}, timeout=60)
    except Exception:
        pass
    t0 = time.time()
    with ThreadPoolExecutor(a.parallel) as pool:
        replies = list(pool.map(lambda p: complete(api, model, p, a.max_tokens, a.retries), prompts(rows, a.shots)))
    latency = time.time() - t0
    accept = accept_length(server)
    errors = [r["error"] for r in replies if "error" in r]
    preds = [INVALID if "error" in r else answer_value(r["text"]) for r in replies]
    n = len(rows)
    acc = round(sum(p is not None and p == l for p, l in zip(preds, labels)) / n, 4)
    invalid = round(sum(p is None for p in preds) / n, 4)
    tokens = sum(r.get("tokens") or 0 for r in replies)
    os.makedirs(a.result_root, exist_ok=True)
    with open(os.path.join(a.result_root, "%s_gsm8k_5shot.jsonl" % a.tag), "w", encoding="utf-8") as f:
        for row, reply, pred, label in zip(rows, replies, preds, labels):
            f.write(json.dumps({"question": row["question"], "label": label, "pred": pred, **reply}) + "\n")
    print("gsm8k_5shot: accuracy=%.4f (%d questions) invalid=%.4f errors=%d accept_length=%s latency=%.1fs "
          "output_tok_s=%.1f" % (acc, n, invalid, len(errors), accept, latency, tokens / latency if latency else 0))
    if errors:
        print("gsm8k_5shot: %d requests failed, the first: %s" % (len(errors), errors[0][:300]), file=sys.stderr)
        return 1
    with open(os.path.join(a.result_root, "summary.txt"), "a", encoding="utf-8") as f:
        f.write("%s gsm8k_5shot: accept_length=%s aggregate=%s\n"
                % (a.tag, accept, {"score": acc, "num": n, "accuracy": acc, "invalid": invalid}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
