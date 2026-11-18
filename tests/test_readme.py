"""README.md and README.zh-CN.md stay in step: each links to the other, every link to a heading lands
on one, both show the same commands, the same names in their tables, the same layout, section by section
the same headings, inline code, numbers and links, and table by table the same rows. Offline.

    python3 -m unittest discover -s tests
"""
import re
import unicodedata
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN = (ROOT / "README.md").read_text(encoding="utf-8")
ZH = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def prose(text):
    return FENCE.sub("", text)


def slug(heading):
    """GitHub's anchor for a heading (github-slugger): lower case, punctuation dropped, spaces to hyphens."""
    def keep(c):
        cat = unicodedata.category(c)
        return c in " -" or cat[0] in "LMN" or cat == "Pc"
    return "".join(c for c in heading.strip().lower() if keep(c)).replace(" ", "-")


def anchors(text):
    seen, out = {}, set()
    for heading in re.findall(r"^#{1,6} (.+)$", prose(text), re.M):
        s = slug(heading)
        n = seen.get(s, 0)
        out.add(s if n == 0 else "%s-%d" % (s, n))
        seen[s] = n + 1
    return out


def commands(text):
    """Every line of every bash block; comments are left out, the translation translates them."""
    out = []
    for block in re.findall(r"^```bash\n(.*?)^```", text, re.M | re.S):
        for line in block.splitlines():
            line = re.sub(r"\s+#.*$", "", line).rstrip()
            if line and not line.lstrip().startswith("#"):
                out.append(line)
    return out


def table_names(text):
    """Names in backticks in the first column of every table row."""
    names = set()
    for row in re.findall(r"^\|(.*)$", prose(text), re.M):
        names |= set(re.findall(r"`([^`]+)`", row.split("|")[0]))
    return names


# Inline code the translation translates: a placeholder's words (<your email>, <你的邮箱>) count as <>, and
# these spans are the same text in the two languages.
TRANSLATED = {"检查 || 动作": "check || action"}


def sections(text):
    """(heading level, body without code blocks) for the top and every heading, in order."""
    body = prose(text)
    levels = [""] + re.findall(r"^(#{1,6}) ", body, re.M)
    return list(zip(levels, re.split(r"^#{1,6} .+$", body, flags=re.M)))


def code_spans(body):
    spans = set()
    for span in re.findall(r"`([^`]+)`", body):
        span = re.sub(r"<[^<>`]*>", "<>", " ".join(span.split()))
        spans.add(TRANSLATED.get(span, span))
    return spans


def numbers(body):
    """The numbers in the prose, wherever they stand (第3步 as much as step 3)."""
    body = re.sub(r"\]\([^)]*\)", "]", re.sub(r"`[^`]*`", "", body))
    return set(re.findall(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?![A-Za-z0-9_.])", body))


def heading_numbers(text):
    """Anchor -> the heading's place in the README: anchors are translated with their headings."""
    seen, out = {}, {}
    for i, heading in enumerate(re.findall(r"^#{1,6} (.+)$", prose(text), re.M)):
        s = slug(heading)
        k = seen.get(s, 0)
        out[s if k == 0 else "%s-%d" % (s, k)] = i
        seen[s] = k + 1
    return out


def link_targets(body, places, other):
    """Link targets, with an anchor as the heading it lands on and the other language's README as one name."""
    def target(t):
        if t.startswith("#"):
            return "#heading-%s" % places.get(t[1:], t)
        return "<the other README>" if t == other else t
    return {target(t) for t in re.findall(r"\]\(([^)\s]+)\)", body)}


def cells(row):
    """A table row's cells; a | inside inline code (`a || b`) does not split one."""
    out, cur, code = [], "", False
    for ch in row.strip().strip("|"):
        code ^= ch == "`"
        if ch == "|" and not code:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def table_rows(text):
    """Every table as its rows: each row's number of cells and the inline code in it."""
    tables, rows = [], []
    for line in prose(text).splitlines() + [""]:
        if line.startswith("|"):
            row = cells(line)
            rows.append((len(row), sorted(code_spans(line))))
        elif rows:
            tables.append(rows)
            rows = []
    return tables


def layout(text):
    """The entries of the Layout block (├── inferpilot.js ...), in order."""
    return [entry for block in re.findall(r"^```[^\n]*\n(.*?)^```", text, re.M | re.S) if "├── " in block
            for entry in re.findall(r"[├└]── (\S+)", block)]


class ReadmeTest(unittest.TestCase):
    def test_each_links_to_the_other(self):
        self.assertIn("[简体中文](README.zh-CN.md)", EN)
        self.assertIn("[English](README.md)", ZH)

    def test_links_to_headings_land_on_one(self):
        for name, text in (("README.md", EN), ("README.zh-CN.md", ZH)):
            targets = set(re.findall(r"\]\(#([^)]+)\)", prose(text)))
            self.assertTrue(targets, name)
            self.assertEqual(targets - anchors(text), set(), name)

    def test_same_commands(self):
        self.assertTrue(commands(EN))
        self.assertEqual(commands(EN), commands(ZH))

    def test_same_names_in_the_tables(self):
        self.assertGreater(len(table_names(EN)), 50)
        self.assertEqual(table_names(EN) ^ table_names(ZH), set())

    def test_same_headings(self):
        en, zh = sections(EN), sections(ZH)
        self.assertGreater(len(en), 15)
        self.assertEqual([level for level, _ in en], [level for level, _ in zh])

    def test_same_inline_code_numbers_and_links_section_by_section(self):
        headings = ["(top)"] + re.findall(r"^#{1,6} (.+)$", prose(EN), re.M)
        en_places, zh_places = heading_numbers(EN), heading_numbers(ZH)
        for title, (_, en), (_, zh) in zip(headings, sections(EN), sections(ZH)):
            with self.subTest(section=title):
                self.assertEqual(code_spans(en) ^ code_spans(zh), set(), "inline code only one language has")
                self.assertEqual(numbers(en) ^ numbers(zh), set(), "numbers only one language has")
                self.assertEqual(link_targets(en, en_places, "README.zh-CN.md") ^ link_targets(zh, zh_places, "README.md"),
                                 set(), "links only one language has")

    def test_same_tables_row_by_row(self):
        en, zh = table_rows(EN), table_rows(ZH)
        self.assertGreater(len(en), 10)
        self.assertEqual(len(en), len(zh), "tables")
        for k, (a, b) in enumerate(zip(en, zh), 1):
            self.assertEqual(a, b, "table %d: rows of other sizes, or inline code in another row" % k)

    def test_same_layout(self):
        self.assertGreater(len(layout(EN)), 10)
        self.assertEqual(layout(EN), layout(ZH))


if __name__ == "__main__":
    unittest.main()
