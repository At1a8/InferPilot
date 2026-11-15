#!/usr/bin/env python3
"""Summary of a lockstep_ab.sh run.

For every arm: the bench numbers (total throughput, TPOT mean and median, accept length) as the
mean over its runs and the concurrencies of each run's change against the mean of all baseline
runs, with the range of the single runs, and a paired comparison of each run against the baseline
of the same round. When the run measured accuracy: each arm's mean scores and accept lengths
against the mean of the baseline runs.

Usage: python3 ab_table.py <out_dir>   (prints markdown and writes it to <out_dir>/results.md)
"""

import ast
import json
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
KEYS = {
    "total_throughput": "total tok/s",
    "mean_tpot_ms": "TPOT mean",
    "median_tpot_ms": "TPOT median",
    "accept_length": "accept length",
}
TAG = re.compile(r"r(\d+)_(baseline|fly_w\d+_t[\d.]+)(?:_[ab])?")
SUMMARY_LINE = re.compile(r"^\S+ (\S+): accept_length=\[([\d.]+)\] aggregate=(.*)$")


def launched_tags():
    status = OUT / "status.txt"
    tags = []
    for line in status.read_text().splitlines() if status.exists() else []:
        m = re.search(r"launching (\S+) \(port", line)
        if m and m[1] not in tags:
            tags.append(m[1])
    return tags


def load_bench(tag):
    files = sorted((OUT / "bench" / tag).glob("bench_*_c*.jsonl")) or sorted(OUT.glob(f"bench*/{tag}/bench_*_c*.jsonl"))
    perf = {}
    for f in files:
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        if lines:
            r = json.loads(lines[-1])
            perf[r["max_concurrency"]] = {k: r.get(k) for k in KEYS}
    return perf


def load_acc(tag):
    out = {}
    f = OUT / "acc" / tag / "summary.txt"
    for line in f.read_text().splitlines() if f.exists() else []:
        m = SUMMARY_LINE.match(line.strip())
        if m:
            out[m[1]] = {"score": ast.literal_eval(m[3]).get("score"), "accept": float(m[2])}
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "-" if x is None else f"{x * 100:+.1f}%"


def spread(xs):
    xs = [x for x in xs if x is not None]
    return f"{pct(mean(xs))} ({min(xs) * 100:+.1f} .. {max(xs) * 100:+.1f})" if xs else "-"


def main():
    runs = []
    for tag in launched_tags():
        m = TAG.fullmatch(tag)
        if m:
            runs.append({"tag": tag, "round": int(m[1]), "arm": m[2], "perf": load_bench(tag), "acc": load_acc(tag)})
    bases = [r for r in runs if r["arm"] == "baseline"]
    arms = sorted({r["arm"] for r in runs}, key=lambda a: (a != "baseline", [float(x) for x in re.findall(r"[\d.]+", a)]))
    out = [f"# lockstep A/B summary: {OUT.name}", ""]

    perf_bases = [b for b in bases if b["perf"]]
    if perf_bases:
        conc = sorted(set.intersection(*(set(b["perf"]) for b in perf_bases)))
        ref = {c: {k: mean([b["perf"][c][k] for b in perf_bases]) for k in KEYS} for c in conc}

        def rel(perf, base, key):
            return mean([perf[c][key] / base[c][key] - 1 for c in conc if c in perf and base[c][key]])

        out += [
            f"## Bench (mean over concurrency {', '.join(map(str, conc))}; vs {len(perf_bases)} baseline runs)",
            "",
            "Each cell: the mean of the runs' changes against the baseline mean, with the range of the single runs. "
            "Paired: each run against the baseline of the same round.",
            "",
            "| arm | runs | " + " | ".join(KEYS.values()) + " | total tok/s, paired |",
            "|---|---|" + "---|" * (len(KEYS) + 1),
        ]
        for arm in arms:
            rs = [r for r in runs if r["arm"] == arm and r["perf"]]
            if not rs:
                continue
            cells = [spread([rel(r["perf"], ref, k) for r in rs]) for k in KEYS]
            paired = "-"
            if arm != "baseline":
                same_round = []
                for r in rs:
                    rb = [b["perf"] for b in perf_bases if b["round"] == r["round"]]
                    if rb:
                        base = {c: {k: mean([p[c][k] for p in rb if c in p]) for k in KEYS} for c in conc}
                        same_round.append(rel(r["perf"], base, "total_throughput"))
                paired = spread(same_round)
            out.append(f"| {arm} | {len(rs)} | " + " | ".join(cells) + f" | {paired} |")
        out += ["", "| baseline mean | " + " | ".join(f"c{c}: {ref[c]['total_throughput']:.0f} tok/s, TPOT mean {ref[c]['mean_tpot_ms']:.2f} ms" for c in conc) + " |", ""]

    acc_bases = [b for b in bases if b["acc"]]
    if acc_bases:
        datasets = list(acc_bases[0]["acc"])
        ref = {ds: {k: mean([b["acc"].get(ds, {}).get(k) for b in acc_bases]) for k in ("score", "accept")} for ds in datasets}
        out += [
            f"## Accuracy (mean of the runs; change vs {len(acc_bases)} baseline runs)",
            "",
            "| arm | runs | " + " | ".join(datasets) + " | accept length |",
            "|---|---|" + "---|" * (len(datasets) + 1),
        ]
        for arm in arms:
            rs = [r for r in runs if r["arm"] == arm and r["acc"]]
            if not rs:
                continue
            cells = []
            for ds in datasets:
                v = mean([r["acc"].get(ds, {}).get("score") for r in rs])
                delta = "" if arm == "baseline" or v is None else f" ({(v - ref[ds]['score']) * 100:+.1f}pp)"
                cells.append("-" if v is None else f"{v:.4f}{delta}")
            acc_change = mean([r["acc"][ds]["accept"] / ref[ds]["accept"] - 1 for r in rs for ds in datasets if ds in r["acc"]])
            out.append(f"| {arm} | {len(rs)} | " + " | ".join(cells) + f" | {pct(acc_change)} |")
        out += ["", "Single runs:", "", "| tag | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
        for r in runs:
            if r["acc"]:
                out.append(f"| {r['tag']} | " + " | ".join(
                    f"{r['acc'][ds]['score']:.4f} ({r['acc'][ds]['accept']:.2f})" if ds in r["acc"] else "-" for ds in datasets) + " |")
        out.append("")

    status = (OUT / "status.txt").read_text() if (OUT / "status.txt").exists() else ""
    stalls = [line for line in status.splitlines() if "compiled while serving" in line]
    if stalls:
        out += ["Runs that compiled kernels while serving (discard or repeat):", ""] + [f"- {s}" for s in stalls] + [""]

    text = "\n".join(out)
    (OUT / "results.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
