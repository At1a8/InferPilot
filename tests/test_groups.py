"""The tests split in step with the code (tests/groups.py): every dry-run scenario and every unit test is in
basic or a task group, everything tests/groups.json names exists, inferpilot.js splits into whole
statements, and a change reaches the groups it should. Offline.

    python3 -m unittest discover -s tests
"""
import re
import unittest
from unittest import mock

import groups as G
from test_readme import ROOT

ALL = set(G.TASKS)


def scenario_modes():
    """The dry run's scenario prefixes, each with the modes its scenarios name (mode: '...')."""
    src = (ROOT / "tests" / "dry_run.mjs").read_text(encoding="utf-8")
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^  \['([a-z_]+): ", src, re.M)]
    out = {}
    for k, (at, prefix) in enumerate(starts):
        body = src[at:starts[k + 1][0] if k + 1 < len(starts) else len(src)]
        out.setdefault(prefix, set()).update(re.findall(r"\bmode: '([a-z]+)'", body))
    return out


def scenario_prefixes():
    return set(scenario_modes())


def test_classes():
    def flat(suite):
        for t in suite:
            yield from flat(t) if isinstance(t, unittest.TestSuite) else [t]
    loader = unittest.TestLoader()
    return {".".join(t.id().split(".")[:2]) for t in flat(loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT / "tests")))}


class PlacementTest(unittest.TestCase):
    def test_every_dry_run_scenario_is_in_exactly_one_group(self):
        placed = {}
        for g in G.GROUPS:
            for p in G.GROUPS[g]["dry_run"]:
                placed.setdefault(p, []).append(g)
        self.assertEqual({}, {p: gs for p, gs in placed.items() if len(gs) > 1}, "in more than one group")
        self.assertEqual([], sorted(scenario_prefixes() - set(placed)), "scenarios no group runs (tests/groups.json)")
        self.assertEqual([], sorted(set(placed) - scenario_prefixes()), "groups.json names scenarios there are none of")

    def test_every_unit_test_is_placed_and_every_placed_one_exists(self):
        classes = test_classes()
        modules = {c.split(".")[0] for c in classes}
        listed = {t for g in G.GROUPS for t in G.GROUPS[g]["unittest"]}
        self.assertEqual([], sorted(t for t in listed if t not in classes | modules), "groups.json names tests there are none of")
        self.assertEqual([], sorted(c for c in classes if c not in listed and c.split(".")[0] not in listed),
                         "tests no group lists: put them in basic or a task group in tests/groups.json")

    def test_every_entry_point_is_in_inferpilot_js(self):
        js = G.script()
        self.assertEqual([], ["%s: %s" % (g, e) for g in G.TASKS for e in G.GROUPS[g]["code"] if e not in js.index])

    def test_a_groups_entry_points_include_the_runner_of_every_mode_its_scenarios_run(self):
        js = G.script()
        runners = dict(re.findall(r"(\w+): (run\w+)", js.stmts[js.index["RUNNERS"]]["text"]))
        modes = scenario_modes()
        for g in G.TASKS:
            run = set().union(*(modes.get(p, set()) for p in G.GROUPS[g]["dry_run"]))
            self.assertTrue(run & set(runners), "%s: its scenarios name no mode" % g)
            self.assertEqual([], sorted(runners[m] for m in run if m in runners and runners[m] not in G.GROUPS[g]["code"]),
                             "%s's scenarios run these; add them to its code in tests/groups.json" % g)

    def test_a_script_reaches_the_groups_with_its_unit_test(self):
        js = G.script()
        for g in G.TASKS:
            for t in G.GROUPS[g]["unittest"]:
                if t.startswith("test_scripts."):
                    name = re.sub(r"(?<!^)(?=[A-Z])", "_", t.split(".")[1][:-len("Test")]).lower()
                    scripts = sorted((ROOT / "scripts").glob(name + ".*"))
                    self.assertTrue(scripts, "%s is not named after a script (scripts/%s.*)" % (t, name))
                    self.assertIn(g, G.reached("scripts/" + scripts[0].name, None, js), t)

    def test_inferpilot_js_splits_into_whole_statements(self):
        js = G.script()
        for s in js.stmts:
            for a, b in ("{}", "()", "[]"):
                self.assertEqual(s["code"].count(a), s["code"].count(b), "the statement at line %d is cut" % s["first"])
        lines = (ROOT / "inferpilot.js").read_text(encoding="utf-8").count("\n") + 1
        self.assertEqual(list(range(js.stmts[0]["first"], lines + 1)),
                         [n for s in js.stmts for n in range(s["first"], s["last"] + 1)], "a line in no statement or two")


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.js = G.script()

    def in_js(self, name):
        s = self.js.stmts[self.js.index[name]]
        return G.reached("inferpilot.js", {s["last"]}, self.js)

    def test_inferpilot_js_code_reaches_the_tasks_that_run_it(self):
        self.assertEqual(self.in_js("runScan"), {"scan"})
        self.assertEqual(self.in_js("COLLECT"), {"scan"})
        self.assertEqual(self.in_js("TRIAGE"), {"issue"})
        self.assertEqual(self.in_js("CLASSIFY"), {"ci"})
        self.assertEqual(self.in_js("SPEC"), {"feature", "perf", "kernel"}, "perf and kernel scenarios run feature tasks")
        self.assertEqual(self.in_js("implementAll"), ALL - {"scan"})
        self.assertEqual(self.in_js("deliver"), ALL - {"scan"})

    def test_what_the_main_flow_or_every_task_runs_reaches_every_group(self):
        for name in ("ROUTES", "RUNNERS", "S", "intakeStage", "meta", "finish"):
            self.assertEqual(self.in_js(name), ALL, name)
        last = (ROOT / "inferpilot.js").read_text(encoding="utf-8").count("\n")
        self.assertEqual(G.reached("inferpilot.js", {last}, self.js), ALL)
        self.assertEqual(G.reached("inferpilot.js", None, self.js), ALL)

    def test_files_reach_the_tasks_that_read_them(self):
        cases = {
            "README.md": set(), "README.zh-CN.md": set(), "docs/design.md": set(), "AGENTS.md": set(),
            "inferpilot/bootstrap.py": set(), "tests/test_bootstrap.py": set(), "tests/backend_parity.mjs": set(),
            "roles/scout.md": {"scan"}, "scripts/scan_upstream.py": {"scan"},
            "knowledge/feature_skills/some-skill/SKILL.md": {"feature"}, "knowledge/kernels/some/x.md": {"kernel"},
            "roles/intake.md": ALL, "runtime/engine.mjs": ALL, "tests/dry_run.mjs": ALL, "tests/fixtures/answers.mjs": ALL,
            ".github/workflows/tests.yml": ALL, "tests/groups.json": ALL,
        }
        for path, want in cases.items():
            self.assertEqual(G.reached(path, None, self.js), want, path)
        self.assertLessEqual({"ci", "scan"}, G.reached("tests/test_scripts.py", None, self.js))

    def test_what_changed_comes_from_git_with_its_lines(self):
        def git(*args):
            if args[0] == "merge-base":
                return "abc\n"
            if args[0] == "ls-files":
                return "new notes.txt\0"
            if "--name-only" in args:  # -z: names as they are, however odd
                return "inferpilot.js\0gone.md\0docs/说明 b/x.md\0img.png\0"
            return "\n".join(["diff --git a/inferpilot.js b/inferpilot.js", "--- a/inferpilot.js", "+++ b/inferpilot.js",
                              "@@ -10,2 +10,3 @@ x", "@@ -40 +41,0 @@ y"])
        with mock.patch.object(G, "git", git):
            got = G.changes("origin/main")
        self.assertEqual(got, {"inferpilot.js": {10, 11, 12, 41, 42}, "gone.md": None, "docs/说明 b/x.md": None,
                               "img.png": None, "new notes.txt": None})

    def test_basic_tests_always_run_and_a_groups_only_with_it(self):
        ids = lambda groups: {".".join(t.id().split(".")[:2]) for t in G.unit_tests(groups)}
        none, scan = ids([]), ids(["scan"])
        self.assertIn("test_readme.ReadmeTest", none)
        self.assertNotIn("test_scripts.ScanUpstreamTest", none)
        self.assertIn("test_scripts.ScanUpstreamTest", scan)
        self.assertNotIn("test_scripts.FindSkillTest", scan)
        self.assertEqual(G.dry_run_prefixes(["scan"]), G.GROUPS["basic"]["dry_run"] + ["scan"])


if __name__ == "__main__":
    unittest.main()
