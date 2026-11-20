#!/usr/bin/env python3
"""Baseline measurements kept for reuse: a run whose baseline was measured on the same code, model,
scripts, workload and GPU type takes those numbers instead of measuring them again.

    python3 scripts/baseline_cache.py lookup --cache DIR --commit SHA --model M --server-script F \\
        [--client-script F] [--workload TEXT] [--sets 'gsm8k ...'] --gpu ARCH [--image IMAGE] [--extra TEXT] \\
        [--max-age-days 7]
    python3 scripts/baseline_cache.py store --cache DIR <the same key> --run-dir RUN_DIR --files PATH...

The key is the framework commit, the model, the contents of the server and client scripts (an edited
script misses), the workload, the accuracy sets (a gsm8k-only baseline is no baseline for a task that also
needs gpqa_diamond and simple_qa), the contents of the accuracy harness (scripts/eval_accuracy.sh: a harness
that sends or scores otherwise measures another number), the GPU architecture (gfx950), the image, and
--extra for anything else the numbers depend on (the GPU count, the sampling). lookup prints the newest
entry no older than --max-age-days, {"hit": true, "dir", "run_dir", "stored_at", "files"}, and exits 0, or
{"hit": false} and exits 1. store copies the files and directories into <cache>/<key hash>/<time>-<random>/files/, writes
meta.json there last, and prints the entry. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_accuracy.sh")


def sha_of(path: str) -> str:
    if not path:
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def key_of(a) -> dict:
    return {"commit": a.commit, "model": a.model, "server_script": sha_of(a.server_script),
            "client_script": sha_of(a.client_script), "workload": a.workload, "sets": " ".join(sorted(a.sets.split())),
            "harness": sha_of(a.harness), "gpu": a.gpu, "image": a.image, "extra": a.extra}


def digest(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def lookup(cache: str, key: dict, max_age_days: float, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    root = os.path.join(cache, digest(key))
    best = None
    for stamp in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        try:
            with open(os.path.join(root, stamp, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        files = os.path.join(root, stamp, "files")
        if meta.get("key") != key or now - meta.get("stored", 0) > max_age_days * 86400:
            continue
        if not all(os.path.exists(os.path.join(files, n)) for n in meta.get("files", [])):
            continue
        if best is None or meta["stored"] > best[1]["stored"]:
            best = (os.path.join(root, stamp), meta)
    if not best:
        return {"hit": False}
    d, meta = best
    return {"hit": True, "dir": os.path.join(d, "files"), "run_dir": meta.get("run_dir", ""),
            "stored_at": meta.get("stored_at", ""), "files": meta.get("files", [])}


def store(cache: str, key: dict, run_dir: str, paths: list, now: float | None = None) -> dict:
    """A directory of its own per store, its meta.json written last: a lookup sees whole entries only,
    and two runs storing the same key at once never share one."""
    now = time.time() if now is None else now
    stamp = time.strftime("%Y%m%d-%H%M%SZ", time.gmtime(now))
    root = os.path.join(cache, digest(key))
    os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix=stamp + "-", dir=root)
    files = os.path.join(d, "files")
    os.makedirs(files)
    names = []
    for p in paths:
        name = os.path.basename(os.path.normpath(p))
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(files, name), dirs_exist_ok=True)
        else:
            shutil.copy2(p, os.path.join(files, name))
        names.append(name)
    meta = {"key": key, "run_dir": os.path.abspath(run_dir), "stored": now,
            "stored_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)), "files": names}
    with open(os.path.join(d, "meta.json.part"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(os.path.join(d, "meta.json.part"), os.path.join(d, "meta.json"))
    return {"stored": True, "dir": files, "files": names}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reuse a baseline measured on the same code, model, scripts and GPU type.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lookup", "store"):
        p = sub.add_parser(name)
        p.add_argument("--cache", required=True)
        p.add_argument("--commit", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--server-script", required=True)
        p.add_argument("--client-script", default="")
        p.add_argument("--workload", default="")
        p.add_argument("--sets", default="", help="the accuracy sets measured, e.g. 'gsm8k gpqa_diamond simple_qa'")
        p.add_argument("--gpu", required=True, help="the GPU architecture, e.g. gfx950")
        p.add_argument("--image", default="")
        p.add_argument("--extra", default="", help="anything else the numbers depend on")
        p.add_argument("--harness", default=HARNESS, help=argparse.SUPPRESS)  # another accuracy harness, for tests
        if name == "lookup":
            p.add_argument("--max-age-days", type=float, default=7)
        else:
            p.add_argument("--run-dir", required=True)
            p.add_argument("--files", nargs="+", required=True)
    a = ap.parse_args(argv)
    key = key_of(a)
    out = lookup(a.cache, key, a.max_age_days) if a.cmd == "lookup" else store(a.cache, key, a.run_dir, a.files)
    print(json.dumps(out, indent=1))
    return 0 if out.get("hit", True) else 1


if __name__ == "__main__":
    sys.exit(main())
