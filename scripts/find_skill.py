#!/usr/bin/env python3
"""Find feature skills by tag: the skills in knowledge/feature_skills/index.yaml whose tags, id or
feature name occur in a query (a request, a task title, an upstream PR title), best match first.

Matching is case-insensitive on whole words, with '-', '_' and '/' read as spaces, so "Speculative
Decoding" finds the tag "speculative decoding" and "hisparse_v2" the id "hisparse-v2". A skill needs at
least one tag, id or feature hit; its model architectures and platforms only rank the hits. The result
is a candidate list: the match rule of knowledge/feature_skills/README.md still decides whether a
skill is reused or only ported from.

    python3 scripts/find_skill.py "bring FLy to DSPARK speculative decoding on MI355X"
    python3 scripts/find_skill.py --json hisparse
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent / "knowledge" / "feature_skills" / "index.yaml"
STATUS_ORDER = {"validated": 0, "stale": 1, "draft": 2, "failed": 3}


def load_index(path: Path) -> list:
    """The entries under `skills:` of the index, with the keys under `match:` flattened in.

    The index is the restricted YAML the curator writes: `- id: ...` starts an entry, `key: value`
    scalars, `[a, b]` lists, `#` comments after whitespace.
    """
    skills, entry, in_skills = [], None, False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = re.split(r"\s#", raw, maxsplit=1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("skills:"):
            in_skills = True
            continue
        if not in_skills:
            continue
        item = line.strip()
        if item.startswith("- "):
            entry = {}
            skills.append(entry)
            item = item[2:]
        if entry is None or ":" not in item:
            continue
        key, _, value = item.partition(":")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
        elif value:
            value = value.strip("'\"")
        else:
            continue
        entry[key.strip()] = value
    return skills


def title_of(index_dir: Path, entry: dict) -> str:
    skill_md = index_dir / entry.get("file", "")
    if skill_md.is_file():
        for line in skill_md.read_text(encoding="utf-8").splitlines()[:20]:
            if line.startswith("title:"):
                return line.partition(":")[2].strip().strip("'\"")
    return ""


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[-_/]+", " ", text.lower()).split())


def occurs(term: str, query: str) -> bool:
    term = normalize(term)
    return bool(term) and re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", query) is not None


def as_list(value) -> list:
    return value if isinstance(value, list) else [value] if value else []


def find(query: str, index: Path = INDEX) -> list:
    """Candidate skills for `query`, best first: hits on tags, id and feature count 1 each, hits on
    model architectures and platforms 0.25 each, ties broken by validation status and id."""
    q = normalize(query)
    out = []
    for entry in load_index(index):
        terms = as_list(entry.get("tags")) + [entry.get("id", ""), entry.get("feature", "")]
        hits = sorted({t for t in terms if t and occurs(t, q)})
        if not hits:
            continue
        context = [t for t in as_list(entry.get("model_arch")) + as_list(entry.get("platforms")) if occurs(t, q)]
        out.append({
            "id": entry.get("id", ""),
            "status": entry.get("validation_status", ""),
            "file": str(index.parent / entry.get("file", "")),
            "score": len(hits) + 0.25 * len(context),
            "matched": hits + context,
            "title": title_of(index.parent, entry),
        })
    out.sort(key=lambda s: (-s["score"], STATUS_ORDER.get(s["status"], 9), s["id"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("query", nargs="+", help="words to look for: a request, a task title, an upstream PR title")
    ap.add_argument("--index", type=Path, default=INDEX, help="the feature skills index (default: %(default)s)")
    ap.add_argument("--json", action="store_true", help="print the candidates as JSON")
    args = ap.parse_args(argv)
    found = find(" ".join(args.query), args.index)
    if args.json:
        print(json.dumps(found, indent=1))
    elif not found:
        print("no feature skill matches")
    else:
        for s in found:
            print(f"{s['id']} ({s['status']}) {s['file']}\n  matched: {', '.join(s['matched'])}")
            if s["title"]:
                print(f"  {s['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
