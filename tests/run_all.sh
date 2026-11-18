#!/usr/bin/env bash
# Everything a pull request is checked with, offline and without a GPU: the syntax of every script, then
# the dry run of inferpilot.js, the runner's self-test and the unit tests. CI runs it on every pull request
# (.github/workflows/tests.yml); run it before you push. Needs Python 3.10+ and Node.js 20+.
#
#   tests/run_all.sh
set -u
cd "$(dirname "$0")/.." || exit 2

python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null \
  || { echo "tests/run_all.sh: needs python3 3.10 or newer" >&2; exit 2; }
node -e 'process.exit(Number(process.versions.node.split(".")[0]) < 20 ? 1 : 0)' 2>/dev/null \
  || { echo "tests/run_all.sh: needs Node.js 20 or newer (bootstrap --runner installs it in a container)" >&2; exit 2; }

passed=0
failed=()
step() {  # step <name> <command...>: one line for it, and its output when it fails
  local name=$1 out rc start end
  shift
  start=$(date +%s%N)
  out=$("$@" 2>&1)
  rc=$?
  end=$(date +%s%N)
  if [ "$rc" -eq 0 ]; then
    passed=$((passed + 1))
    printf 'ok    %-38s %5.1f s  %s\n' "$name" "$(( (end - start) / 1000000 ))e-3" \
      "$(printf '%s\n' "$out" | grep -E 'checks passed|^Ran ' | tail -1)"
  else
    failed+=("$name")
    printf 'FAIL  %-38s exit %s\n' "$name" "$rc"
    printf '%s\n' "$out" | tail -60 | sed 's/^/      /'
  fi
}

syntax_js() {  # inferpilot.js as the runner compiles it; the modules and tests as Node loads them
  node --input-type=module -e "
    import { readFileSync } from 'node:fs';
    import { scriptBody } from './runtime/engine.mjs';
    const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
    new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget',
      scriptBody(readFileSync('inferpilot.js', 'utf8')));" || return 1
  local f
  for f in runtime/*.mjs tests/*.mjs; do node --check "$f" || return 1; done
}

syntax_sh() {
  local f
  for f in scripts/*.sh tests/*.sh; do bash -n "$f" || return 1; done
}

syntax_py() {  # parsed, not compiled to files: no __pycache__ in the checkout
  python3 - inferpilot/*.py scripts/*.py tests/*.py setup.py <<'EOF'
import ast, sys
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        ast.parse(f.read(), path)
EOF
}

step "syntax: inferpilot.js, runtime, tests" syntax_js
step "syntax: shell scripts" syntax_sh
step "syntax: Python" syntax_py
step "dry run (tests/dry_run.mjs)" node tests/dry_run.mjs
step "runner (tests/runtime_selftest.mjs)" node tests/runtime_selftest.mjs
step "unit tests (python3 -m unittest)" python3 -B -m unittest discover -s tests

if [ ${#failed[@]} -eq 0 ]; then
  echo "all $passed steps passed"
else
  echo "${#failed[@]} of $((passed + ${#failed[@]})) steps failed: ${failed[*]}"
  exit 1
fi
