"""Nothing private reaches the public repository: no token or private key, no internal host or site, no
private network address, no real home directory, and not the name of the machine the tests run on.
Offline; it reads the files git tracks, and new ones it does not ignore.

    python3 -m unittest discover -s tests
"""
import re
import socket
import subprocess
import unittest

from test_readme import ROOT

SECRET = re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[ousr]_[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_-]{16,}"
                    r"|\bsk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|\bhf_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
                    r"|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}")
AMD_HOST = re.compile(r"\b[a-z0-9][a-z0-9.-]*\.amd\.com\b", re.I)
# Public, or named on purpose: the LLM gateway Claude Code runs on.
PUBLIC_AMD = {"www.amd.com", "rocm.docs.amd.com", "rocm.blogs.amd.com", "instinct.docs.amd.com", "llm-api.amd.com"}
INTERNAL_SITE = re.compile(r"\b(?:confluence|jira)\.[a-z0-9-]+\.[a-z]", re.I)
PRIVATE_IP = re.compile(r"(?<![\d.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\d.])")
HOME = re.compile(r"(?<![\w.-])/home/(?![<${*])([A-Za-z0-9._-]+)")
# Names that stand for anyone.
PLACEHOLDER_USERS = {"user", "you", "me", "alice", "bob", "username"}


def tracked():
    """The files git tracks and the new ones it does not ignore; None outside a git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(set(out.decode().split("\0")) - {""})


FILES = tracked()


def files():
    for name in FILES or []:
        path = ROOT / name
        if path.is_file():
            data = path.read_bytes()
            if b"\0" not in data:
                yield name, data.decode("utf-8", "replace")


def found(pattern, keep=lambda m: True, show=lambda s: s[:60]):
    return ["%s:%d: %s" % (name, text.count("\n", 0, m.start()) + 1, show(m.group(0)))
            for name, text in files() for m in pattern.finditer(text) if keep(m)]


@unittest.skipIf(FILES is None, "not a git checkout")
class HygieneTest(unittest.TestCase):
    def test_no_token_or_private_key(self):
        # a public CI log shows what fails: the start of a token says which it is, and no more
        self.assertEqual([], found(SECRET, show=lambda s: s[:6] + "..."))

    def test_no_internal_host_or_site(self):
        self.assertEqual([], found(AMD_HOST, lambda m: m.group(0).lower() not in PUBLIC_AMD) + found(INTERNAL_SITE))

    def test_no_private_network_address(self):
        self.assertEqual([], found(PRIVATE_IP))

    def test_no_real_home_directory(self):
        self.assertEqual([], found(HOME, lambda m: m.group(1).lower() not in PLACEHOLDER_USERS))

    def test_not_the_name_of_this_machine(self):
        host = socket.gethostname().split(".")[0]
        if len(host) < 6 or host.lower() in ("localhost", "runner"):
            self.skipTest("a short or generic machine name: %s" % host)
        self.assertEqual([], found(re.compile(r"(?<![\w-])%s(?![\w-])" % re.escape(host), re.I)))


if __name__ == "__main__":
    unittest.main()
