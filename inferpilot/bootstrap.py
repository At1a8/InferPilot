"""InferPilot bootstrap: set up InferPilot where it runs; safe to run again at any time.

InferPilot runs inside a ROCm container as root: InferPilot, its runs and the code they change live in the
container's /workspace, one copy per container. Claude Code is installed in the container and its
first-start questions are answered in its config,
and your keys come in as one file: scripts/run_docker.sh mounts your ~/.inferpilot_env read-only as root's
~/.inferpilot_env, and not your home. Run this once in every new container (README step 3).

Every step checks first, keeps what is already there and installs or creates only what is missing.
Nothing is deleted, overwritten or re-cloned: a step that could only proceed by changing existing
state warns and leaves it alone. The one exception is the image's own sglang checkout, which the
framework step puts on main once, discarding only the image build's changes to its packaging files.

  checkout      clone InferPilot to $INFERPILOT_HOME, unless run inside a checkout or one is already there
  keys          (root in a container) find your keys file: INFERPILOT_KEYS, else ~/.inferpilot_env, where
                run_docker.sh mounts yours; in a container that mounts /home, the one in your home there
                (INFERPILOT_HOST_HOME, else, with a warning, the home of the user the container is named after)
  claude        install the Claude Code CLI if it is still missing; update one older than inferpilot.js needs
  path          put the CLI's directory on PATH for new shells (one marked block in ~/.bashrc)
  claude-config (root in a container) answer Claude Code's first-start questions in its config: the
                onboarding, trust in the InferPilot checkout, bypass permissions mode, and an Anthropic API
                key from the keys file
  git           (root in a container) trust checkouts on host mounts, which belong to another user
  git-identity  (root in a container) commit as you, not root@<this host>: INFERPILOT_GIT_NAME and
                INFERPILOT_GIT_EMAIL from the keys file or the environment
  shell         (root in a container) no __pycache__ in checkouts on host mounts, no Claude Code
                self-update, the keys file loaded, and with LLM_GATEWAY_KEY Claude Code on the AMD LLM
                gateway
  framework     the image's /sgl-workspace/sglang by default, which the image installs sglang from in
                editable mode: still as the image left it, it is put on main and pulled, so the current
                main is what runs and what tasks branch from; --framework-dir names a host checkout
                instead, cloned with its whole history if missing
  evalscope     (root in a container) evalscope for scripts/eval_accuracy.sh, in a venv of its own
                (/opt/evalscope, linked as /usr/local/bin/evalscope): installed into the image's Python it
                would downgrade packages sglang ships with. One already on PATH is kept
  checks        git, docker (host only), ROCm, sglang, evalscope, Node.js, API access, GitHub token
                (detect only)

  python3 -B -m inferpilot.bootstrap [--framework-dir DIR]
  python3 -B -m inferpilot.bootstrap --claude-only   # the keys, claude, path and claude-config steps
  python3 -B -m inferpilot.bootstrap --check         # the checks

Environment (a flag wins over its variable):
  INFERPILOT_HOME, INFERPILOT_REPO_URL, INFERPILOT_REF
                                the checkout (default ./inferpilot, or the one you are in)
  INFERPILOT_KEYS               (container) your keys file, when it is not ~/.inferpilot_env
  INFERPILOT_HOST_HOME          (container that mounts /home) your home there
  INFERPILOT_CONTAINER          (container) its name (run_docker.sh passes it; without it, asked of
                                Docker when its socket is mounted)
  CLAUDE_VERSION, CLAUDE_BIN_DIR
                                native-installer target (latest), its bin dir (~/.local/bin)
  CLAUDE_CONFIG_DIR             Claude Code's own: where its config lives (default ~)
  EVALSCOPE_VERSION, EVALSCOPE_VENV
                                (container) the evalscope version to install (latest), its venv
                                (/opt/evalscope)
  INFERPILOT_FRAMEWORK_DIR, INFERPILOT_FRAMEWORK_REPO
                                --framework-dir (default: the image's checkout), --framework-repo
                                (default: sglang)
  INFERPILOT_SKIP_BOOTSTRAP     set to skip everything (CI, image builds)

`pip install` runs it while building the wheel. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

CLAUDE_MIN_VERSION = "2.1.177"  # first Claude Code with the dynamic Workflow tool inferpilot.js runs in
DEFAULT_REPO_URL = "https://github.com/At1a8/inferpilot.git"
DEFAULT_FRAMEWORK_REPO = "https://github.com/sgl-project/sglang.git"
INSTALL_DOCS = "https://code.claude.com/docs/en/setup"
KEYS_FILE = ".inferpilot_env"
RC_BEGIN = "# >>> inferpilot: Claude Code CLI on PATH >>>"
RC_END = "# <<< inferpilot: Claude Code CLI on PATH <<<"
ENV_BEGIN = "# >>> inferpilot: root in the container >>>"
ENV_END = "# <<< inferpilot: root in the container <<<"
# scripts/run_docker.sh names a container after your login unless told otherwise (alice, or alice-0116
# for the next task): in a container that mounts /home without INFERPILOT_HOST_HOME, that user's home is yours.
HOMES = "/home"
NAME_SEPARATORS = "-_."
# Root in the container must not leave root-owned __pycache__ in checkouts on host mounts; Claude Code
# stays at the version installed here for the container's life (a new container gets the newest).
CONTAINER_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "DISABLE_AUTOUPDATER": "1"}
# Claude Code keeps the last 20 characters of an Anthropic API key it was told to use.
APPROVED_KEY_CHARS = 20
# The AMD LLM gateway speaks the Anthropic Messages API and authenticates on the
# Ocp-Apim-Subscription-Key header alone; the same key as a bearer token only keeps Claude Code from
# asking for a login, and the gateway ignores it.
AMD_GATEWAY_URL = "https://llm-api.amd.com/Unified"
# sglang images ship a full-history checkout of sglang and install sglang from it in editable mode; tasks
# work in it, each on its own branch. The image's build rewrites its packaging files (python/pyproject.toml
# becomes the ROCm one) and normally leaves it detached at the commit it built; some daily images leave
# the same build state on main.
IMAGE_SGLANG = "/sgl-workspace/sglang"
PACKAGING = re.compile(r"(^|/)pyproject[^/]*\.toml$")
# Some daily images leave that checkout on main instead of detached. Only these known build-time edits
# may be discarded from main; a different pyproject edit may be a user's work and must stop Setup.
IMAGE_MAIN_PACKAGING = frozenset(
    {
        "python/pyproject.toml",
        "python/sglang/kernels/aot/pyproject.toml",
        "python/sglang/kernels/aot/pyproject_rocm.toml",
    }
)
# scripts/eval_accuracy.sh reaches the server over HTTP, so evalscope needs nothing of sglang's Python, and
# installed into it, it would change it: evalscope 1.15 downgrades datasets and fsspec below the image's.
EVALSCOPE_VENV = "/opt/evalscope"
EVALSCOPE_LINK = "/usr/local/bin/evalscope"
GATEWAY_ENV = (
    'if [ -n "$LLM_GATEWAY_KEY" ] && [ -z "$ANTHROPIC_BASE_URL" ]; then\n'
    '  export ANTHROPIC_BASE_URL=%s\n'
    '  export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: $LLM_GATEWAY_KEY"\n'
    '  export ANTHROPIC_AUTH_TOKEN="$LLM_GATEWAY_KEY"\n'
    'fi' % AMD_GATEWAY_URL
)


def _styles(stream) -> tuple[str, str]:
    return ("\033[1;32m", "\033[0m") if getattr(stream, "isatty", lambda: False)() else ("", "")


def log(msg: str) -> None:
    print("[inferpilot-bootstrap] %s" % msg, flush=True)


def warn(msg: str) -> None:
    print("[inferpilot-bootstrap WARN] %s" % msg, file=sys.stderr, flush=True)


def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd, shell: bool = False):
    log(cmd if shell else " ".join(cmd))
    return subprocess.run(cmd, shell=shell)


def _out(cmd, timeout: float | None = None) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except OSError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired:
        return 124, "", "timed out after %ss" % timeout
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def _expand(path: str, env) -> str:
    if path == "~" or path.startswith("~/"):
        path = (env.get("HOME") or os.path.expanduser("~")) + path[1:]
    return os.path.abspath(path)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def in_container() -> bool:
    return os.path.exists("/.dockerenv")


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def give_back(path: str) -> None:
    """chown -R what root wrote to the owner of the closest ancestor that is not root's."""
    if not is_root():
        return
    d = os.path.dirname(os.path.abspath(path))
    while True:
        try:
            st = os.stat(d)
        except OSError:
            st = None
        if st and st.st_uid != 0:
            _run(["chown", "-R", "%d:%d" % (st.st_uid, st.st_gid), path])
            return
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def _invocation_dir(env) -> str:
    """The directory the user ran pip from. During `pip install git+...` the process cwd is a
    temporary build tree, but the shell's PWD still names the user's directory."""
    pwd = env.get("PWD", "")
    if pwd and os.path.isdir(pwd):
        return os.path.abspath(pwd)
    return os.getcwd()


def is_inferpilot_checkout(path: str) -> bool:
    return (os.path.isdir(os.path.join(path, ".git"))
            and os.path.isfile(os.path.join(path, "inferpilot.js"))
            and os.path.isdir(os.path.join(path, "roles")))


def settings(env=None, opts=None) -> dict:
    env = os.environ if env is None else env
    opts = opts or {}
    inv = _invocation_dir(env)
    home = env.get("INFERPILOT_HOME") or (inv if is_inferpilot_checkout(inv) else os.path.join(inv, "inferpilot"))
    framework_dir = opts.get("framework_dir") or env.get("INFERPILOT_FRAMEWORK_DIR") or ""
    return {
        "invocation_dir": inv,
        "home": _expand(home, env),
        "repo_url": env.get("INFERPILOT_REPO_URL") or DEFAULT_REPO_URL,
        "ref": env.get("INFERPILOT_REF") or "",
        "claude_version": env.get("CLAUDE_VERSION") or "latest",
        "claude_bin_dir": _expand(env.get("CLAUDE_BIN_DIR") or "~/.local/bin", env),
        "user_home": _expand("~", env),
        "keys": _expand(env["INFERPILOT_KEYS"], env) if env.get("INFERPILOT_KEYS") else "",
        "host_home": _expand(env["INFERPILOT_HOST_HOME"], env) if env.get("INFERPILOT_HOST_HOME") else "",
        "shell": env.get("SHELL") or "",
        "framework_dir": _expand(framework_dir, env) if framework_dir else "",
        "framework_repo": opts.get("framework_repo") or env.get("INFERPILOT_FRAMEWORK_REPO") or DEFAULT_FRAMEWORK_REPO,
        "evalscope_version": env.get("EVALSCOPE_VERSION") or "",
        "evalscope_venv": _expand(env.get("EVALSCOPE_VENV") or EVALSCOPE_VENV, env),
        "evalscope_link": EVALSCOPE_LINK,
    }


def owner_dirs(s: dict) -> list:
    """In a container that mounts /home, where your keys may be, closest first: your home there
    (INFERPILOT_HOST_HOME); without it, the ancestors of the InferPilot checkout owned by the checkout's owner,
    unless that is root, as for a checkout in /workspace."""
    if s.get("host_home"):
        return [s["host_home"]] if os.path.isdir(s["host_home"]) else []
    try:
        uid = os.stat(s["home"]).st_uid
    except OSError:
        return []
    if uid == 0:
        return []
    out, d = [], os.path.dirname(s["home"])
    while d != os.path.dirname(d):
        try:
            if os.stat(d).st_uid == uid:
                out.append(d)
        except OSError:
            pass
        d = os.path.dirname(d)
    return out


def keys_file(s: dict) -> str:
    """Your keys file, '' when there is none: INFERPILOT_KEYS; else ~/.inferpilot_env, where scripts/run_docker.sh
    mounts yours from the host; else, in a container that mounts /home, ~/.inferpilot_env in your home there."""
    if s.get("keys"):
        return s["keys"] if os.path.isfile(s["keys"]) else ""
    for d in [s["user_home"]] + owner_dirs(s):
        cand = os.path.join(d, KEYS_FILE)
        if os.path.isfile(cand):
            return cand
    return ""


def keys_missing(s: dict) -> str:
    """Why keys_file found nothing, for the warnings."""
    if s.get("keys"):
        return "INFERPILOT_KEYS=%s is not a file" % s["keys"]
    why = "no %s" % os.path.join(s["user_home"], KEYS_FILE)
    if s.get("host_home"):
        where = s["host_home"] if os.path.isdir(s["host_home"]) else "INFERPILOT_HOST_HOME=%s (not mounted)" % s["host_home"]
        return "%s, nor one in %s" % (why, where)
    if s.get("container_name"):
        return "%s, and no user in %s with one matches this container's name (%s)" % (why, HOMES, s["container_name"])
    return why


def _ver_tuple(s: str) -> tuple:
    out = []
    for part in s.split("."):
        m = re.match(r"\d+", part)
        out.append(int(m.group(0)) if m else 0)
    return tuple(out)


def ver_ge(a: str, b: str) -> bool:
    return _ver_tuple(a) >= _ver_tuple(b)


# --- checkout --------------------------------------------------------------------

def ensure_checkout(s: dict) -> tuple[str, str]:
    home = s["home"]
    if is_inferpilot_checkout(home):
        log("checkout: kept %s" % home)
        return "kept", home
    if os.path.isdir(home) and os.listdir(home):
        warn("checkout: %s exists and is not a InferPilot checkout; left untouched (set INFERPILOT_HOME)" % home)
        return "skipped", "%s is not empty" % home
    if not _has("git"):
        warn("checkout: git not found; cannot clone InferPilot to %s" % home)
        return "failed", "no git"
    os.makedirs(os.path.dirname(home) or ".", exist_ok=True)
    cmd = ["git", "clone"] + (["--branch", s["ref"]] if s["ref"] else []) + [s["repo_url"], home]
    if _run(cmd).returncode == 0:
        give_back(home)
        log("checkout: cloned to %s" % home)
        return "created", home
    warn("checkout: git clone failed; check the network and your access to %s" % s["repo_url"])
    return "failed", "git clone"


# --- your keys -------------------------------------------------------------------

def container_name(env=None) -> str:
    """This container's name: INFERPILOT_CONTAINER (scripts/run_docker.sh passes it), else Docker's answer when
    the host's socket is mounted; '' when neither tells."""
    env = os.environ if env is None else env
    if env.get("INFERPILOT_CONTAINER"):
        return env["INFERPILOT_CONTAINER"]
    m = re.search(r"/containers/([0-9a-f]{64})/", _read("/proc/self/mountinfo"))
    if not (m and _has("docker")):
        return ""
    rc, out, _ = _out(["docker", "inspect", "-f", "{{.Name}}", m.group(1)], timeout=20)
    return out.lstrip("/") if rc == 0 else ""


def home_from_container_name(name: str, homes: str | None = None) -> str:
    """The home of the user a container is named after: <user>, or <user> with a suffix after - _ or .
    (the longest such user). Only a home with a keys file in it counts (README step 1)."""
    homes = HOMES if homes is None else homes
    try:
        users = os.listdir(homes)
    except OSError:
        return ""
    best = ""
    for u in users:
        if len(u) <= len(best) or not (name == u or (name.startswith(u) and name[len(u)] in NAME_SEPARATORS)):
            continue
        if os.path.isfile(os.path.join(homes, u, KEYS_FILE)):
            best = u
    return os.path.join(homes, best) if best else ""


def find_host_home(s: dict, env=None) -> str:
    """In a container that mounts /home without INFERPILOT_HOST_HOME, the home of the user it is named after,
    kept in s["host_home"]. '' when INFERPILOT_KEYS or INFERPILOT_HOST_HOME is set (they win), the checkout names
    a home, or nothing matches."""
    if s.get("keys") or s.get("host_home") or owner_dirs(s):
        return ""
    s["container_name"] = container_name(env)
    home = home_from_container_name(s["container_name"]) if s["container_name"] else ""
    if home:
        s["host_home"] = home
    return home


def ensure_keys(s: dict) -> tuple[str, str]:
    """Without a keys file Claude Code here has no API access and asks for a login; the steps after this
    one still run, and a keys file added later is loaded by new shells."""
    keys = keys_file(s)
    if keys:
        log("keys: %s" % keys)
        return "kept", keys
    if find_host_home(s):
        keys = keys_file(s)
        warn("keys: no %s; using %s, in the home of the user this container is named after (%s). Not yours? "
             "Run this again with INFERPILOT_KEYS=<your keys file> in front"
             % (os.path.join(s["user_home"], KEYS_FILE), keys, s["container_name"]))
        return "found", keys
    warn("keys: %s; Claude Code here has no API access until there is one (see the next steps)" % keys_missing(s))
    return "skipped", "no keys file"


# --- Claude Code -----------------------------------------------------------------

def find_claude(s: dict) -> str:
    """The claude on PATH, else one the native installer left in CLAUDE_BIN_DIR (not on PATH yet)."""
    found = shutil.which("claude")
    if found:
        return found
    cand = os.path.join(s["claude_bin_dir"], "claude")
    return cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else ""


def claude_version(path: str) -> str:
    """Leading semver of `claude --version` (e.g. '2.1.206 (Claude Code)'), '' when absent."""
    if not path:
        return ""
    rc, out, _ = _out([path, "--version"])
    m = re.search(r"\d+\.\d+\.\d+", out)
    return m.group(0) if rc == 0 and m else ""


def install_claude(s: dict) -> bool:
    if _has("curl"):
        # The native installer downloads a large binary silently; minutes without output on a slow
        # link are a slow download, not a hang.
        log("claude: native installer (%s)" % s["claude_version"])
        cmd = "curl -fsSL --connect-timeout 20 https://claude.ai/install.sh | bash -s %s" % s["claude_version"]
        if _run(cmd, shell=True).returncode == 0:
            return True
        warn("claude: native installer failed")
    if _has("npm"):
        warn("claude: falling back to npm install -g @anthropic-ai/claude-code")
        if _run(["npm", "install", "-g", "@anthropic-ai/claude-code"]).returncode == 0:
            return True
    warn("claude: could not install Claude Code (need curl or npm, and network access). Manual setup: %s"
         % INSTALL_DOCS)
    return False


def ensure_claude_code(s: dict) -> tuple[str, str]:
    path = find_claude(s)
    cur = claude_version(path)
    if cur and ver_ge(cur, CLAUDE_MIN_VERSION):
        log("claude: kept %s (%s)" % (cur, path))
        return "kept", cur
    if cur:
        warn("claude: %s is older than %s; updating it" % (cur, CLAUDE_MIN_VERSION))
        _run([path, "update"])
        cur = claude_version(find_claude(s))
        if cur and ver_ge(cur, CLAUDE_MIN_VERSION):
            log("claude: updated to %s" % cur)
            return "updated", cur
    else:
        log("claude: not found; installing")
    install_claude(s)
    cur = claude_version(find_claude(s))
    if cur and ver_ge(cur, CLAUDE_MIN_VERSION):
        log("claude: installed %s" % cur)
        return "installed", cur
    if cur:
        warn("claude: %s is still older than %s; run `claude update` or set CLAUDE_VERSION"
             % (cur, CLAUDE_MIN_VERSION))
    else:
        warn("claude: not found after the install; see %s" % INSTALL_DOCS)
    return "failed", cur


# --- shell rc files --------------------------------------------------------------

def rc_file(s: dict) -> str:
    return os.path.join(s["user_home"], ".zshrc" if os.path.basename(s["shell"]) == "zsh" else ".bashrc")


def _append(rc: str, block: str) -> bool:
    try:
        with open(rc, "a", encoding="utf-8") as f:
            f.write(block)
        return True
    except OSError as e:
        warn("could not append to %s (%s)" % (rc, e))
        return False


def _on_path(d: str, path_env: str) -> bool:
    return any(os.path.abspath(p) == d for p in path_env.split(os.pathsep) if p)


def ensure_path(s: dict, path_env: str | None = None) -> tuple[str, str]:
    d = s["claude_bin_dir"]
    path_env = os.environ.get("PATH", "") if path_env is None else path_env
    if not os.path.isfile(os.path.join(d, "claude")):
        return "skipped", "claude is not in %s" % d
    if _on_path(d, path_env):
        log("path: kept (%s is on PATH)" % d)
        return "kept", d
    rc = rc_file(s)
    text = _read(rc)
    names = {d}
    if d.startswith(s["user_home"] + os.sep):
        rel = d[len(s["user_home"]):]
        names |= {"$HOME" + rel, "${HOME}" + rel, "~" + rel}
    if RC_BEGIN in text or any(n in text for n in names):
        log("path: kept (%s already puts %s on PATH)" % (rc, d))
        return "kept", rc
    block = '\n%s\ncase ":$PATH:" in *":%s:"*) ;; *) export PATH="%s:$PATH" ;; esac\n%s\n' % (RC_BEGIN, d, d, RC_END)
    if not _append(rc, block):
        return "failed", rc
    log("path: added %s to %s" % (d, rc))
    return "added", rc


def ensure_shell_env(s: dict) -> tuple[str, str]:
    rc = rc_file(s)
    if ENV_BEGIN in _read(rc):
        log("shell: kept (%s)" % rc)
        return "kept", rc
    lines = ["export %s=%s" % kv for kv in CONTAINER_ENV.items()]
    # Where run_docker.sh mounts your keys, also when there are none yet: a file put there later loads.
    keys = keys_file(s) or os.path.join(s["user_home"], KEYS_FILE)
    lines.append('[ -r "%s" ] && . "%s"' % (keys, keys))
    lines.append(GATEWAY_ENV)
    if not _append(rc, "\n%s\n%s\n%s\n" % (ENV_BEGIN, "\n".join(lines), ENV_END)):
        return "failed", rc
    log("shell: added %s and %s to %s" % (", ".join(CONTAINER_ENV), keys, rc))
    return "added", rc


def anthropic_api_key(s: dict, env=None) -> str:
    """ANTHROPIC_API_KEY from this shell, else from the keys file when it sets it to a plain value."""
    env = os.environ if env is None else env
    if env.get("ANTHROPIC_API_KEY"):
        return env["ANTHROPIC_API_KEY"]
    keys = keys_file(s)
    for line in _read(keys).splitlines() if keys else ():
        m = re.match(r"\s*(?:export\s+)?ANTHROPIC_API_KEY=(.*)$", line)
        value = m.group(1).strip().strip("'\"") if m else ""
        if value and not value.startswith("<") and "$" not in value:
            return value
    return ""


def _update_json(path: str, change) -> str:
    """change(data) on the JSON object in path, the rest of it kept: 'added' when written, 'kept' when
    nothing changed, 'skipped' when the file holds something else, 'failed' when it cannot be written."""
    text = _read(path)
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError:
        data = None
    if not isinstance(data, dict):
        warn("claude-config: %s is not a JSON object; left as it is (Claude Code asks at its first start)" % path)
        return "skipped"
    before = json.dumps(data, sort_keys=True)
    change(data)
    if json.dumps(data, sort_keys=True) == before:
        return "kept"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.chmod(path, 0o600)
    except OSError as e:
        warn("claude-config: could not write %s (%s)" % (path, e))
        return "failed"
    return "added"


def ensure_claude_config(s: dict, env=None) -> tuple[str, str]:
    """Root's Claude Code is new in every container, and its first start would ask for the onboarding,
    trust in the folder, bypass permissions mode and, for an Anthropic API key, whether to use it. Its
    config answers them ahead; anything else in it is kept. The
    bypass permissions answer lives in the user settings, the others in the global config."""
    env = os.environ if env is None else env
    config_dir = env.get("CLAUDE_CONFIG_DIR") or ""
    config = os.path.join(config_dir or s["user_home"], ".claude.json")
    user_settings = os.path.join(config_dir or os.path.join(s["user_home"], ".claude"), "settings.json")
    key = anthropic_api_key(s, env).strip()[-APPROVED_KEY_CHARS:]

    def answer(data):
        data["hasCompletedOnboarding"] = True
        data["bypassPermissionsModeAccepted"] = True
        projects = data.get("projects") if isinstance(data.get("projects"), dict) else {}
        project = projects.get(s["home"]) if isinstance(projects.get(s["home"]), dict) else {}
        project["hasTrustDialogAccepted"] = True
        projects[s["home"]] = project
        data["projects"] = projects
        if key:
            answers = data.get("customApiKeyResponses") if isinstance(data.get("customApiKeyResponses"), dict) else {}
            approved = [k for k in answers.get("approved") or [] if isinstance(k, str)]
            answers["approved"] = approved + ([] if key in approved else [key])
            answers.setdefault("rejected", [])
            data["customApiKeyResponses"] = answers

    results = {_update_json(config, answer),
               _update_json(user_settings, lambda data: data.update(skipDangerousModePermissionPrompt=True))}
    status = next(r for r in ("failed", "skipped", "added", "kept") if r in results)
    if status == "kept":
        log("claude-config: kept (%s, %s)" % (config, user_settings))
    elif status == "added":
        log("claude-config: Claude Code's first-start questions answered in %s and %s (onboarding, trust in %s, "
            "bypass permissions%s)" % (config, user_settings, s["home"], ", the API key" if key else ""))
    return status, config


# --- git, as root in a container -------------------------------------------------

def ensure_git_trust(s: dict) -> tuple[str, str]:
    """git refuses repositories owned by another user; in the container every mounted checkout is."""
    if not _has("git"):
        return "skipped", "no git"
    _, out, _ = _out(["git", "config", "--global", "--get-all", "safe.directory"])
    if "*" in out.splitlines():
        log("git: kept (safe.directory '*')")
        return "kept", "safe.directory '*'"
    if _run(["git", "config", "--global", "--add", "safe.directory", "*"]).returncode == 0:
        return "added", "safe.directory '*'"
    warn("git: could not set safe.directory")
    return "failed", "git config"


IDENTITY_VARS = ("INFERPILOT_GIT_NAME", "INFERPILOT_GIT_EMAIL")


def _keys_file_value(path: str, name: str) -> str:
    """One variable's value in the keys file. Read for the git identity only, never for a key."""
    for line in _read(path).splitlines() if path else ():
        m = re.match(r"\s*(?:export\s+)?%s=(.*)$" % re.escape(name), line)
        if m:
            value = m.group(1).strip().strip("'\"")
            return "" if value.startswith("<") else value
    return ""


def ensure_git_identity(s: dict) -> tuple[str, str]:
    """Commits made as root in the container would read root <root@<this host>>: an internal host name
    in a public history, and no author to sign them. The container carries no home, so no git config
    of yours: the identity comes from INFERPILOT_GIT_NAME and INFERPILOT_GIT_EMAIL, in the keys file or the
    environment."""
    if not _has("git"):
        return "skipped", "no git"
    have = tuple(_out(["git", "config", "--global", "--get", k])[1].strip() for k in ("user.name", "user.email"))
    keys = keys_file(s)
    want = tuple(os.environ.get(v) or _keys_file_value(keys, v) for v in IDENTITY_VARS)
    if not all(want):
        if all(have):
            log("git-identity: kept (%s <%s>)" % have)
            return "kept", "%s <%s>" % have
        warn("git-identity: no git identity for root; its commits would read root@<this host>. Add "
             "export INFERPILOT_GIT_NAME=\"<your name>\" and export INFERPILOT_GIT_EMAIL=<your email> to %s "
             "(README step 1) and run this again" % (keys or os.path.join(s["user_home"], KEYS_FILE)))
        return "skipped", "no INFERPILOT_GIT_NAME / INFERPILOT_GIT_EMAIL"
    if want == have:
        log("git-identity: kept (%s <%s>)" % have)
        return "kept", "%s <%s>" % have
    if all(_run(["git", "config", "--global", k, v]).returncode == 0 for k, v in zip(("user.name", "user.email"), want)):
        log("git-identity: commits as %s <%s>" % want)
        return ("updated" if all(have) else "added"), "%s <%s>" % want
    warn("git-identity: could not set user.name / user.email")
    return "failed", "git config"


# --- framework checkout ----------------------------------------------------------

def _repo_key(url: str) -> str:
    u = re.sub(r"\.git$", "", url.strip().rstrip("/"))
    u = re.sub(r"^(?:ssh://)?git@([^:/]+)[:/]", r"https://\1/", u)
    return u.lower()


def image_checkout() -> str:
    """The sglang image's own checkout, InferPilot's default framework source, when this is such an image."""
    return IMAGE_SGLANG if os.path.isdir(os.path.join(IMAGE_SGLANG, ".git")) else ""


def installed_commit() -> str:
    """The commit the image installed sglang at, from its version (0.5.23.dev20261105+gf3532057e1)."""
    try:
        import importlib.metadata
        m = re.search(r"\+g([0-9a-f]{7,})", importlib.metadata.version("sglang"))
    except Exception:
        return ""
    return m.group(1) if m else ""


def follow_main(d: str) -> tuple[str, str]:
    """Put the image's checkout on main once: discard the build's packaging changes, check out main, pull.
    Only a checkout as the image left it is touched: a detached checkout must be at the installed commit
    and have only packaging changes; main may discard only the exact known build-time packaging paths.
    A task branch is already past this and is left alone (Setup takes it from there)."""
    rc, branch, _ = _out(["git", "-C", d, "symbolic-ref", "-q", "--short", "HEAD"])
    _, head, _ = _out(["git", "-C", d, "log", "-1", "--format=%h %cs"])
    on_branch = rc == 0 and bool(branch)
    if on_branch and branch != "main":
        log("framework: kept %s on %s (%s); the image's checkout, which tasks branch from" % (d, branch, head))
        return "kept", "%s on %s" % (d, branch)
    sha, built = (head.split() or [""])[0], installed_commit()
    matches_image = bool(built and sha and (sha.startswith(built) or built.startswith(sha)))
    if not on_branch and built and sha and not matches_image:
        warn("framework: %s is detached at %s, not at the commit the image installed (%s); left as it is"
             % (d, head, built))
        return "skipped", "detached at %s" % head
    _, changed, _ = _out(["git", "-C", d, "diff", "--name-only", "HEAD"])
    files = [f for f in changed.splitlines() if f]
    other = (
        [f for f in files if f not in IMAGE_MAIN_PACKAGING]
        if on_branch
        else [f for f in files if not PACKAGING.search(f)]
    )
    if other:
        warn("framework: %s has changes beyond the image's packaging files (%s); left as it is, and Setup "
             "stops on it" % (d, ", ".join(other[:5])))
        return "skipped", "changed: %s" % ", ".join(other[:5])
    if files and _run(["git", "-C", d, "checkout", "HEAD", "--"] + files).returncode != 0:
        warn("framework: could not discard the image's packaging changes in %s" % d)
        return "failed", "git checkout HEAD -- <packaging files>"
    if not on_branch and _run(["git", "-C", d, "checkout", "main"]).returncode != 0:
        warn("framework: could not check out main in %s" % d)
        return "failed", "git checkout main"
    if _run(["git", "-C", d, "pull", "--ff-only"]).returncode != 0:
        warn("framework: git pull failed in %s; main stays where it was, and Setup fetches before every task" % d)
    _, head, _ = _out(["git", "-C", d, "log", "-1", "--format=%h %cs"])
    log("framework: %s now on main (%s)%s; never `pip install -e` from it: its pyproject.toml is upstream's"
        % (d, head, ", the image's changes to %s discarded" % ", ".join(files) if files else ""))
    return "updated", "%s on main (%s)" % (d, head)


def ensure_framework(s: dict) -> tuple[str, str]:
    d = s["framework_dir"] or image_checkout()
    if not d:
        return "skipped", "no --framework-dir and no image checkout"
    if os.path.exists(os.path.join(d, ".git")):
        _, head, _ = _out(["git", "-C", d, "log", "-1", "--format=%h %cs"])
        _, shallow, _ = _out(["git", "-C", d, "rev-parse", "--is-shallow-repository"])
        _, origin, _ = _out(["git", "-C", d, "remote", "get-url", "origin"])
        if shallow == "true":
            warn("framework: %s is a shallow clone; ci bisects and the upstream scan need its history. "
                 "It is left as it is (`git -C %s fetch --unshallow` would deepen it)" % (d, d))
        if origin and _repo_key(origin) != _repo_key(s["framework_repo"]):
            warn("framework: %s tracks %s, not %s; kept as it is" % (d, origin, s["framework_repo"]))
        if d == IMAGE_SGLANG:
            return follow_main(d)
        log("framework: kept %s (%s)" % (d, head or "no commits"))
        return "kept", d
    if os.path.isdir(d) and os.listdir(d):
        warn("framework: %s exists and is not a git checkout; left untouched" % d)
        return "skipped", "%s is not a git checkout" % d
    if not _has("git"):
        warn("framework: git not found; cannot clone to %s" % d)
        return "failed", "no git"
    os.makedirs(os.path.dirname(d) or ".", exist_ok=True)
    # Blobless: the whole commit history for bisects and the upstream scan, file contents on demand.
    if _run(["git", "clone", "--filter=blob:none", s["framework_repo"], d]).returncode == 0:
        give_back(d)
        log("framework: cloned %s to %s" % (s["framework_repo"], d))
        return "created", d
    warn("framework: git clone of %s failed" % s["framework_repo"])
    return "failed", "git clone"


# --- evalscope -------------------------------------------------------------------

def ensure_evalscope(s: dict) -> tuple[str, str]:
    """evalscope for scripts/eval_accuracy.sh (gate G2), in a venv of its own linked onto PATH, so the
    image's Python stays as the image built it. One already on PATH is kept, wherever it lives."""
    found = shutil.which("evalscope")
    if found:
        log("evalscope: kept (%s)" % found)
        return "kept", found
    venv, link = s["evalscope_venv"], s["evalscope_link"]
    exe = os.path.join(venv, "bin", "evalscope")
    status = "kept"
    if not os.access(exe, os.X_OK):
        spec = "evalscope==%s" % s["evalscope_version"] if s["evalscope_version"] else "evalscope"
        log("evalscope: installing %s in a venv of its own, %s (a minute or two)" % (spec, venv))
        pip = [os.path.join(venv, "bin", "python"), "-m", "pip", "install", "-q", "--disable-pip-version-check", spec]
        if _run([sys.executable, "-m", "venv", venv]).returncode != 0 or _run(pip).returncode != 0:
            warn("evalscope: could not install %s in %s; scripts/eval_accuracy.sh (accuracy, gate G2) needs it. "
                 "Check the network and run this again" % (spec, venv))
            return "failed", venv
        status = "installed"
    if not os.path.lexists(link):
        try:
            os.symlink(exe, link)
        except OSError as e:
            warn("evalscope: could not link %s (%s); scripts/eval_accuracy.sh finds it with EVALSCOPE=%s"
                 % (link, e, exe))
            return "failed", exe
        status = "added" if status == "kept" else status
    elif os.path.realpath(link) != os.path.realpath(exe):
        warn("evalscope: %s is another file; left as it is. scripts/eval_accuracy.sh finds this one with "
             "EVALSCOPE=%s" % (link, exe))
        return "skipped", exe
    if not shutil.which("evalscope"):
        warn("evalscope: %s is not on PATH; scripts/eval_accuracy.sh finds it with EVALSCOPE=%s"
             % (os.path.dirname(link), exe))
    log("evalscope: %s %s, linked as %s" % (status, exe, link))
    return status, exe


# --- checks (detect only) --------------------------------------------------------

def api_configured(env, user_home: str) -> bool:
    config_dir = env.get("CLAUDE_CONFIG_DIR") or os.path.join(user_home, ".claude")
    return bool(env.get("LLM_GATEWAY_KEY") or env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
                or env.get("CLAUDE_CODE_OAUTH_TOKEN")
                or (env.get("ANTHROPIC_BASE_URL") and env.get("ANTHROPIC_CUSTOM_HEADERS"))
                or os.path.isfile(os.path.join(config_dir, ".credentials.json")))


def _importable(module: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def keys_defined(path: str) -> set:
    """Names a keys file sets to a real value; a value still in <angle brackets> is the README's
    placeholder and does not count. Values are never read into the report."""
    names = set()
    for line in _read(path).splitlines() if path else ():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)=(.*)$", line)
        value = m.group(2).strip().strip("'\"") if m else ""
        if value and not value.startswith("<"):
            names.add(m.group(1))
    return names


def check_environment(env=None, keys_path: str = "") -> dict:
    """keys_path: ~/.inferpilot_env, which new shells load; what it sets counts before this shell has it."""
    env = dict(os.environ if env is None else env)
    for name in keys_defined(keys_path):
        env.setdefault(name, "set in %s" % keys_path)
    checks = [("git", _has("git"), "git is required: InferPilot works in a worktree of the framework checkout")]
    if not in_container():
        checks.append(("docker", _has("docker"), "docker not found: scripts/run_docker.sh starts InferPilot's container"))
    checks += [
        ("rocm", _has("rocminfo") or _has("rocm-smi"), "ROCm not detected (rocminfo / rocm-smi): GPU work needs AMD Instinct GPUs with ROCm"),
        ("sglang", _importable("sglang"), "sglang is not importable here: run InferPilot where the framework is installed (the container)"),
        ("evalscope", _has("evalscope"), "evalscope not found: scripts/eval_accuracy.sh (accuracy, gate G2) needs it; the bootstrap installs it as root in a container"),
        ("node", _has("node"), "Node.js not found: needed only for tests/dry_run.mjs and for running inferpilot.js outside Claude Code"),
        ("api", api_configured(env, _expand("~", env)), "no Claude Code API access in this shell (see the next steps)"),
        ("github_token", bool(env.get("GITHUB_TOKEN") or env.get("GH_TOKEN")),
         "no GITHUB_TOKEN: GitHub then allows 60 requests an hour and 10 searches a minute per IP, so scans "
         "crawl and can stop partway; add a read-only token to ~/.inferpilot_env (README step 1)"),
        ("repo_token", bool(env.get("INFERPILOT_REPO_TOKEN")),
         "no INFERPILOT_REPO_TOKEN (optional): the curator's knowledge changes stay in this checkout instead of a "
         "pull request to InferPilot's repository (README step 1)"),
    ]
    found = {}
    log("checks (detect only):")
    for name, ok, hint in checks:
        found[name] = ok
        if ok:
            log("  %-12s ok" % name)
        else:
            warn("  %-12s %s" % (name, hint))
    return found


# --- report ----------------------------------------------------------------------

def summary(results: list) -> str:
    rows = ["  %-13s %-10s %s" % r for r in results]
    return "summary (safe to re-run: what is already there is kept):\n" + "\n".join(rows)


def next_steps(s: dict, statuses: dict, found: dict | None = None, path_env: str | None = None,
               claude_only: bool = False) -> str:
    c_on, c_off = _styles(sys.stdout)
    hl = lambda t: "%s%s%s" % (c_on, t, c_off)  # noqa: E731
    path_env = os.environ.get("PATH", "") if path_env is None else path_env
    steps = []
    d = s["claude_bin_dir"]
    reload_rc = statuses.get("shell") == "added" or (
        os.path.isfile(os.path.join(d, "claude")) and not _on_path(d, path_env) and statuses.get("path") in ("added", "kept"))
    if reload_rc:
        steps.append(["Load the new settings into this shell (new shells have them):", hl("source %s" % rc_file(s))])
    container = in_container()
    keys = keys_file(s) if container else ""
    api = (found or {}).get("api")
    if api is None:
        env = dict(os.environ, **{name: "set in %s" % keys for name in keys_defined(keys)})
        api = api_configured(env, s["user_home"])
    if not api:
        if keys:
            where = "in %s, which new shells load:" % keys
        elif container:
            where = ("in ~/%s on the host (README step 1), which containers started after it get, or for this "
                     "container only in %s:" % (KEYS_FILE, os.path.join(s["user_home"], KEYS_FILE)))
        else:
            where = "(pick one; keep it in %s):" % rc_file(s)
        steps.append([
            "Give Claude Code API access %s" % where,
            "a. AMD LLM gateway:     " + hl("export LLM_GATEWAY_KEY=<your key>") + "   (%s)" % AMD_GATEWAY_URL,
            "b. Anthropic API:       " + hl("export ANTHROPIC_API_KEY=sk-ant-..."),
            "c. another gateway:     " + hl("export ANTHROPIC_BASE_URL=https://your-gateway"),
            "                        " + hl("export ANTHROPIC_AUTH_TOKEN=your-token"),
            "   (a gateway that wants its own header: " + hl('export ANTHROPIC_CUSTOM_HEADERS="Header-Name: <key>"') + ")",
            "d. Claude subscription: " + hl("export CLAUDE_CODE_OAUTH_TOKEN=<token>")
            + "   (from " + hl("claude setup-token") + " where a browser is)",
            "e. interactive login:   " + hl("claude") + "   then /login" + ("  (again in every new container)" if container else ""),
        ])
    if not claude_only:
        if s["framework_dir"]:
            fw = ", framework_dir=%s" % s["framework_dir"]
        else:
            fw = "" if image_checkout() else ", framework_dir=/path/to/sglang"
        start = ["Start Claude Code from the checkout, permissions auto-approved (README step 4):",
                 hl("cd %s && IS_SANDBOX=1 claude --dangerously-skip-permissions" % s["home"])]
        if container and statuses.get("claude-config") not in ("added", "kept"):
            start += ["The first start asks a few questions; answer these, the defaults would stop you:",
                      "  use the API key from your environment?  " + hl("Yes") + "   (asked only for an Anthropic API key)",
                      "  trust this folder?                      " + hl("Yes, I trust this folder"),
                      "  Bypass Permissions mode                 " + hl("Yes, I accept")]
        steps.append(start)
        elsewhere = "" if container else ", container=<container>"
        steps.append(["Give it a task, e.g. collect and rank tasks, or take one issue:",
                      "use %s/inferpilot.js with mode=scan, inferpilot_dir=%s%s" % (s["home"], s["home"], fw),
                      "use %s/inferpilot.js with mode=issue, task=<issue URL>, inferpilot_dir=%s%s, gpu_ids=0%s"
                      % (s["home"], s["home"], fw, elsewhere)])
    lines = ["Next steps:"]
    for i, (head, *body) in enumerate(steps, 1):
        lines.append("%d) %s" % (i, head))
        lines += ["     " + b for b in body]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -B -m inferpilot.bootstrap",
                                 description="Set up InferPilot where it runs (README step 3). "
                                             "Safe to re-run: what is already there is kept.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--claude-only", action="store_true", help="only the keys, Claude Code, PATH and config steps")
    g.add_argument("--check", action="store_true", help="only the environment checks")
    ap.add_argument("--framework-dir", help="framework checkout for tasks; cloned only if missing (INFERPILOT_FRAMEWORK_DIR)")
    ap.add_argument("--framework-repo", help="repository to clone it from (default: sglang; INFERPILOT_FRAMEWORK_REPO)")
    a = ap.parse_args([] if argv is None else argv)

    if os.environ.get("INFERPILOT_SKIP_BOOTSTRAP"):
        log("INFERPILOT_SKIP_BOOTSTRAP set; skipping")
        return 0
    s = settings(opts={"framework_dir": a.framework_dir, "framework_repo": a.framework_repo})
    container = in_container()
    if a.check:
        if container and is_root() and not keys_file(s):
            find_host_home(s)
        check_environment(keys_path=keys_file(s) if container else "")
        return 0
    steps = [] if a.claude_only else [("checkout", ensure_checkout)]
    if container and is_root():
        steps.append(("keys", ensure_keys))
    steps += [("claude", ensure_claude_code), ("path", ensure_path)]
    if container and is_root():
        steps.append(("claude-config", ensure_claude_config))
    if not a.claude_only:
        if container and is_root():
            steps += [("git", ensure_git_trust), ("git-identity", ensure_git_identity), ("shell", ensure_shell_env)]
        steps.append(("framework", ensure_framework))
        if container and is_root():
            steps.append(("evalscope", ensure_evalscope))
    results = [(name,) + step(s) for name, step in steps]
    statuses = {step: status for step, status, _ in results}
    keys = keys_file(s) if container else ""
    found = None if a.claude_only else check_environment(keys_path=keys)
    print("\n[inferpilot-bootstrap] %s\n\n%s" % (summary(results), next_steps(s, statuses, found, claude_only=a.claude_only)),
          flush=True)
    return 1 if "failed" in statuses.values() else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
