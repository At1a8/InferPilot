#!/usr/bin/env python3
"""What a branch would make public about this machine, before a human pushes it.

    python3 scripts/leak_scan.py --repo <worktree> --base <base commit> [--body RUN_DIR/pr_body.md]

Scans the lines each commit of the branch adds to any file (a line a later commit removed is still published
by the push), the names of the files it adds, its commit messages, the author and committer of each commit,
and the PR description when given, for local paths (/home, /mnt, /data, ...: daily_report.LOCAL), this
machine's host name, the company's network (internal sites and hosts, private addresses: daily_report.leaks),
and identities such as root or an address with the host name in it. A path counts only when the base does
not carry it already (git grep at the base: AMD CI's /data2/models/huggingface is upstream's, not this
machine's), and none under /tmp or /opt counts, since those name nothing of this machine; --allow adds a
pattern for text that is public all the same. Prints one JSON object, {"leaks": [{"where", "text"}],
"identities": [...]}, and exits 1 when anything was found. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daily_report import LOCAL, leaks  # noqa: E402

# A path as a line holds it: from a / that starts a word, or follows an option (-I/x) or a shell default
# (${V:-/x}), to a space, a quote, a bracket or a colon, so each entry of PYTHONPATH=/a:/b is one. None
# starts in the // of a URL: leaks() reads its host.
PATH = re.compile(r"(?:(?<![\w./-])|(?<=(?<![\w-])-[A-Za-z])|(?<=:[-=?+]))/(?!/)[^\s'\"`<>(){}\[\],;:]+")
GENERIC = ("/tmp/", "/opt/")
# root as a person: a name with its address (Signed-off-by: root <root@node>), or a root@ address.
ROOT = re.compile(r"(?<![\w.-])root\s*<[^<>\s]+@[^<>\s]+>|(?<![\w.+-])root@[a-z0-9]", re.I)


def git(repo: str, *args: str) -> str:
    # A byte that is not UTF-8 is replaced: it must not stop the scan of the line it is in.
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, encoding="utf-8", errors="replace",
                          check=True).stdout


def added_lines(diff: str):
    """(path, line number, text) of every line a unified diff adds. File headers are read only before a
    file's first hunk: inside one, a line that adds '++ x' reads '+++ x' and is content."""
    path, n, in_hunk = "", 0, False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
            m = re.search(r"\+(\d+)", line)
            n = int(m.group(1)) if m else 0
        elif not in_hunk:
            if line.startswith("+++ "):
                path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+"):
            yield path, n, line[1:]
            n += 1
        elif line.startswith(" "):
            n += 1


def scan(text: str, host: str, allow: list, public=lambda path: False) -> bool:
    """Whether a line would publish the host name, the company's network, root as a person, or a path under a
    local directory that public(path) does not know as public."""
    for pattern in allow:
        text = re.sub(pattern, "", text)
    if (host and host.lower() in text.lower()) or ROOT.search(text) or leaks(PATH.sub(" ", text), host):
        return True
    paths = (p.rstrip(".:!?") for p in PATH.findall(text))
    return any(LOCAL.match(p) and not p.startswith(GENERIC) and not public(p) for p in paths)


def carried(repo: str, base: str):
    """public(path): whether the base's tree holds the path anywhere already, which upstream published."""
    seen = {}

    def public(path: str) -> bool:
        if path not in seen:
            seen[path] = subprocess.run(["git", "-C", repo, "grep", "-q", "-F", "-e", path, base, "--"],
                                        capture_output=True).returncode == 0
        return seen[path]
    return public


def bad_identity(ident: str, host: str) -> bool:
    name, _, email = ident.partition(" <")
    email = email.rstrip(">").lower()
    return name.strip().lower() == "root" or email.startswith("root@") or (bool(host) and host.lower() in email)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local paths, host names and identities a branch would publish.")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", required=True, help="the commit the branch starts from")
    ap.add_argument("--body", default="", help="the PR description, when written")
    ap.add_argument("--host", default=socket.gethostname(), help="this machine's name (default: the host name)")
    ap.add_argument("--allow", action="append", default=[], help="a regex for text that may stay (repeatable)")
    a = ap.parse_args(argv)
    allow = a.allow
    public = carried(a.repo, a.base)
    found = []
    # Every outgoing commit's own diff, not the branch's net one: the push publishes the history, so a line
    # a later commit removed is still public.
    for sha in git(a.repo, "rev-list", "--reverse", "%s..HEAD" % a.base).split():
        # No external diff tool or textconv driver: they could show the scan something else than the commit.
        # --text: a binary file, or one .gitattributes marks -diff, is published all the same.
        diff = git(a.repo, "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--text", "--unified=0",
                   sha + "^", sha)
        for path, n, text in added_lines(diff):
            if scan(text, a.host, allow, public):
                found.append({"where": "%s %s:%d" % (sha[:10], path, n), "text": text.strip()[:200]})
        # A file's name is published too, an empty or renamed file's included.
        for name in git(a.repo, "diff", "--name-only", "-z", "--diff-filter=ACR", sha + "^", sha).split("\0"):
            if name and scan(name, a.host, allow, public):
                found.append({"where": "commit %s file name" % sha[:10], "text": name[:200]})
    for sha in git(a.repo, "rev-list", "%s..HEAD" % a.base).split():
        for n, line in enumerate(git(a.repo, "log", "-1", "--format=%B", sha).splitlines(), 1):
            if scan(line, a.host, allow, public):
                found.append({"where": "commit %s message line %d" % (sha[:10], n), "text": line.strip()[:200]})
    if a.body:
        with open(a.body, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f.read().splitlines(), 1):
                if scan(line, a.host, allow, public):
                    found.append({"where": "%s:%d" % (a.body, n), "text": line.strip()[:200]})
    idents = sorted({i for i in git(a.repo, "log", "--format=%an <%ae>%n%cn <%ce>", "%s..HEAD" % a.base).splitlines()
                     if i and (bad_identity(i, a.host) or scan(i, a.host, allow, public))})
    print(json.dumps({"leaks": found, "identities": idents}, indent=1))
    return 1 if found or idents else 0


if __name__ == "__main__":
    sys.exit(main())
