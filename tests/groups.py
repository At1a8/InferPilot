#!/usr/bin/env python3
"""Which tests a change needs (tests/groups.json). basic runs on every pull request. A task group runs when a
change reaches its task: inferpilot.js code its entry points run or that leads to them, a role that code
calls, a file such a role names, a path the group lists, or one of its tests. A change to what all tasks
share (the runner, the dry run and its fixtures, CI, inferpilot.js code the main flow runs) runs every
group, and so does any change to inferpilot.js this cannot place.

    python3 tests/groups.py select --changed origin/main   # the task groups the changes reach, and why
    python3 tests/groups.py select --group perf,kernel
    python3 tests/groups.py select --all
    python3 tests/groups.py dry-run feature perf           # the dry run's --only: basic's and theirs
    python3 tests/groups.py unittest feature perf          # run basic's unit tests and theirs
"""
import fnmatch
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUPS = json.loads((ROOT / "tests" / "groups.json").read_text(encoding="utf-8"))
TASKS = [g for g in GROUPS if g != "basic"]
# What the roles read while a task runs: a change reaches the tasks whose code calls a role that names it.
ASSETS = ["roles/*", "knowledge/*", "templates/*", "scripts/*"]
IDENT = re.compile(r"[A-Za-z_$][\w$]*")
DECL = re.compile(r"(?:export\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)|(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=")
BEFORE_REGEX = set("(,=:[!&|?{};+-*%<>~^") | {"", "return", "typeof", "case", "do", "else", "in", "of", "new", "delete",
                                              "void", "throw", "await", "yield"}


def code_only(src):
    """src with comments, strings and regular expressions blanked, newlines kept so lines still count, and
    the ${...} of template literals kept: the text in which a name is a reference."""
    out, n = list(src), len(src)

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    def template(i):  # from just after ` or a ${...}'s }: where code resumes, and whether inside a ${
        j = i
        while j < n:
            if src[j] == "\\":
                j += 2
            elif src[j] == "`" or src.startswith("${", j):
                blank(i, j)
                return (j + 1, False) if src[j] == "`" else (j + 2, True)
            else:
                j += 1
        blank(i, n)
        return n, False

    i, depth, opened, prev = 0, 0, [], ""
    while i < n:
        c = src[i]
        word = IDENT.match(src, i)
        if c.isspace():
            i += 1
        elif src.startswith("//", i) or src.startswith("/*", i):
            j = src.find("\n" if src[i + 1] == "/" else "*/", i + 2)
            j = n if j < 0 else (j if src[i + 1] == "/" else j + 2)
            blank(i, j)
            i = j
        elif c in "'\"":
            j = i + 1
            while j < n and src[j] not in (c, "\n"):
                j += 2 if src[j] == "\\" else 1
            blank(i + 1, j)
            i, prev = j + 1, "a"
        elif c == "`" or (c == "}" and opened and opened[-1] == depth):
            if c == "}":
                opened.pop()
            i, inner = template(i + 1)
            if inner:
                opened.append(depth)
            prev = "{" if inner else "a"
        elif c == "/" and prev in BEFORE_REGEX:
            j, in_class = i + 1, False
            while j < n and src[j] != "\n" and (src[j] != "/" or in_class):
                if src[j] == "\\":
                    j += 1
                elif src[j] in "[]":
                    in_class = src[j] == "["
                j += 1
            blank(i + 1, j)
            i, prev = j + 1, "a"
            while i < n and src[i].isalpha():
                i += 1
        elif word:
            i, prev = word.end(), word.group(0)
        else:
            depth += {"{": 1, "}": -1}.get(c, 0)
            i, prev = i + 1, c
    return "".join(out)


class Script:
    """inferpilot.js as top-level statements, and which of them reach which task group."""

    def __init__(self, src):
        code, lines = code_only(src).split("\n"), src.split("\n")
        depth, top = 0, []
        for line in code:
            top.append(depth == 0)
            depth += sum(line.count(c) for c in "([{") - sum(line.count(c) for c in ")]}")
        starts = [k for k, line in enumerate(code) if top[k] and line[:1].strip() and line[0] not in ")]}."]
        firsts = []
        for s in starts:  # a statement owns the comment lines right above it
            while s > 0 and lines[s - 1].lstrip().startswith("//") and not code[s - 1].strip():
                s -= 1
            firsts.append(s)
        self.stmts = []
        for k, s in enumerate(starts):
            last = (firsts[k + 1] if k + 1 < len(starts) else len(lines)) - 1
            m = DECL.match(code[s])
            self.stmts.append({"first": firsts[k] + 1, "last": last + 1, "name": m and (m.group(1) or m.group(2)),
                               "code": "\n".join(code[s:last + 1]), "text": "\n".join(lines[firsts[k]:last + 1])})
        self.index = {s["name"]: k for k, s in enumerate(self.stmts) if s["name"]}
        self.uses = [{self.index[r] for r in set(IDENT.findall(s["code"])) if r in self.index and r != s["name"]}
                     for s in self.stmts]
        self.used_by = [set() for _ in self.stmts]
        for k, refs in enumerate(self.uses):
            for r in refs:
                self.used_by[r].add(k)
        self.entries = {g: {self.index[e] for e in GROUPS[g]["code"] if e in self.index} for g in TASKS}
        stops = set().union(*self.entries.values())
        main = [k for k, s in enumerate(self.stmts) if not s["name"]]
        self.shared = self.closure(main, self.uses, stops) - stops
        self.reach = {g: self.closure(e, self.uses) | self.closure(e, self.used_by) for g, e in self.entries.items()}

    @staticmethod
    def closure(start, edges, stops=frozenset()):
        seen, todo = set(), list(start)
        while todo:
            k = todo.pop()
            if k not in seen:
                seen.add(k)
                if k not in stops or k in start:
                    todo.extend(edges[k])
        return seen

    def at(self, line):
        return next((k for k, s in enumerate(self.stmts) if s["first"] <= line <= s["last"]), None)

    def groups_of(self, k):
        """The groups statement k reaches: all of them for the main flow, code it runs, or code nothing reaches."""
        if k is None or not self.stmts[k]["name"] or k in self.shared:
            return set(TASKS)
        return {g for g, r in self.reach.items() if k in r} or set(TASKS)

    def naming(self, pattern):
        return [k for k, s in enumerate(self.stmts) if re.search(pattern, s["text"])]


def script():
    return Script((ROOT / "inferpilot.js").read_text(encoding="utf-8"))


def matches(path, patterns):
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def reached(path, lines, js=None):
    """The task groups a change to path (at these lines of it, or None: anywhere) reaches."""
    js = js or script()
    if matches(path, GROUPS["basic"]["shared"]):
        return set(TASKS)
    if path == "inferpilot.js":
        if lines is None:
            return set(TASKS)
        return set().union(*(js.groups_of(js.at(n)) for n in lines)) if lines else set()
    out = {g for g in TASKS if matches(path, GROUPS[g]["paths"])}
    if matches(path, ASSETS):
        roles = {Path(path).stem} if re.fullmatch(r"roles/[a-z_]+\.md", path) else set()
        if not path.startswith("roles/"):
            roles |= {p.stem for p in (ROOT / "roles").glob("*.md") if path in p.read_text(encoding="utf-8")}
        for k in [k for r in roles for k in js.naming(r"\bcall\(\s*'%s'" % r)] + js.naming(re.escape(path)):
            out |= js.groups_of(k)
    test = re.fullmatch(r"tests/(test_\w+)\.py", path)
    if test:
        out |= {g for g in TASKS if any(t.split(".")[0] == test.group(1) for t in GROUPS[g]["unittest"])}
    return out


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def changes(base):
    """{path: the lines changed in it now, None for anywhere} since where HEAD left base, uncommitted and
    untracked files too."""
    out, path = {}, None
    for line in git("diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames", git("merge-base", base, "HEAD").strip()).splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[1]
            out[path] = None
        elif line.startswith("+++ b/"):
            out[path] = set()
        elif line.startswith("@@") and out.get(path) is not None:
            m = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            out[path] |= set(range(start, start + count)) if count else {start, start + 1}
    for path in git("ls-files", "--others", "--exclude-standard").splitlines():
        out[path] = None
    return out


def dry_run_prefixes(groups):
    return [p for g in ["basic", *groups] for p in GROUPS[g]["dry_run"]]


def unit_tests(groups):
    """basic's unit tests and the groups', and any test no group lists (test_groups asks to place it)."""
    listed = {t for g in TASKS for t in GROUPS[g]["unittest"]}
    want = set(GROUPS["basic"]["unittest"]) | {t for g in groups for t in GROUPS[g]["unittest"]}

    def flat(suite):
        for t in suite:
            yield from flat(t) if isinstance(t, unittest.TestSuite) else [t]

    keep = []
    for t in flat(unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT / "tests"))):
        module, cls = t.id().split(".")[:2]
        names = {module, "%s.%s" % (module, cls)}
        if names & want or not names & (listed | set(GROUPS["basic"]["unittest"])):
            keep.append(t)
    return unittest.TestSuite(keep)


def main(argv):
    if len(argv) >= 2 and argv[0] == "select":
        if argv[1] == "--all":
            print(" ".join(TASKS))
            return 0
        if argv[1] == "--group" and len(argv) == 3:
            names = [g for g in argv[2].split(",") if g]
            unknown = [g for g in names if g not in TASKS]
            if unknown:
                print("groups.py: no group %s; the groups: %s" % (", ".join(unknown), ", ".join(TASKS)), file=sys.stderr)
                return 2
            print(" ".join(g for g in TASKS if g in names))
            return 0
        if argv[1] == "--changed" and len(argv) == 3:
            try:
                changed = changes(argv[2])
            except subprocess.CalledProcessError as e:
                print("groups.py: cannot tell what changed since %s (%s); running every group"
                      % (argv[2], (e.stderr or "").strip()), file=sys.stderr)
                print(" ".join(TASKS))
                return 0
            js, out = script(), set()
            for path, lines in sorted(changed.items()):
                got = reached(path, lines, js)
                out |= got
                print("  %-44s -> %s" % (path, " ".join(g for g in TASKS if g in got) or "basic only"), file=sys.stderr)
            print(" ".join(g for g in TASKS if g in out))
            return 0
    if len(argv) >= 1 and argv[0] == "dry-run":
        print(",".join(dry_run_prefixes(argv[1:])))
        return 0
    if len(argv) >= 1 and argv[0] == "unittest":
        result = unittest.TextTestRunner(verbosity=1).run(unit_tests(argv[1:]))
        return 0 if result.wasSuccessful() else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
