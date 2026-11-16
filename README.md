# InferPilot

**English** | [简体中文](README.zh-CN.md)

InferPilot is the framework agent for AMD GPUs in open-source LLM inference frameworks, starting with
sglang. It turns one task — an upstream feature to bring to AMD, an AMD user's issue, a red AMD CI
job, a system-level performance gap — into a pull request a maintainer can review: **correct first**
(independent unit gates, accuracy / long-context / stress gates against a same-pattern baseline, a
controlled investigation loop), **then fast** (equal-budget A/B), within the community rules in
[`knowledge/guardrails.md`](knowledge/guardrails.md). InferPilot proposes; people review, push and merge.

Why it exists and how it maps onto the agreed design: [`docs/design.md`](docs/design.md).

## Modes

| Mode | Task | Phases | Ends with |
|---|---|---|---|
| `feature` | an upstream PR or design to bring to AMD | Setup → Spec → Recon → Design → Implement → (Kernel) → Verify → Performance → Review → PR → Report | branch + PR body + validation report |
| `issue` | an AMD user issue | Setup → Triage → Reproduce → Diagnose (skipped for an existing fix PR) → Implement → (Kernel) → Verify → (Performance) → Review → PR → Report | fix PR, or a conclusion (triaged, not reproduced, fixed on main, diagnosed); always a reply draft |
| `ci` | a red AMD CI job or Failure Tracker cluster | Setup → Classify → Bisect → Implement → (Kernel) → Verify → Review → PR → Report | fix PR, or a handover for flaky/infrastructure failures |
| `perf` | a system-level performance goal for a model and a fixed workload | Setup → Profile → Plan → Implement → (Kernel) → Verify → Performance → Review → PR → Report | PR with a measured win, or the change dropped; kernel-level items go to the kernel back end |
| `scan` | none — the day's state of the framework: the AMD CI on main, open AMD user issues, upstream changes AMD needs to follow (platform gates and host-sync leads on request) | Collect → Dispatch → Report | the daily report, a file for you to share, with the ranked tasks and the definition that starts each |

Phases in parentheses run only when they apply: Kernel when there is a kernel hand-off and
`kernel_backend` is not `none`; Performance in issue mode for issues triaged as `performance`.

A task starts from a **task definition** ([`templates/task_definition.md`](templates/task_definition.md)):
the fields `inferpilot.js` takes, the same names every role reads. It can be given as fields, or as a
request in plain words: then an Intake phase runs first and turns the request into a definition,
checking what it can without changing anything (the checkout, the model on this machine, the GPUs,
the issue). A definition that still lacks something its mode needs — the issue, the checkout, the
model for `feature` and `perf`, the fixed workload for `perf` — starts nothing and comes back as
questions, one per field, each with options or a default. A complete definition skips the Intake.

Every implementation runs one work package at a time; an independent test engineer re-runs its unit
test, checks that the test fails without the change, and runs the feature-off check; a package that
keeps failing is rolled back. Verification failures go to an investigator that runs the controls
before any instrumentation and returns a fix, a proof that the failure is pre-existing on the
baseline, or an inconclusive verdict that stops the task. Before the PR, a reviewer reads the diff and
its evidence the way a maintainer would, starting from a catalog of known reward hacks — changes that
move a number while breaking what it meant ([`knowledge/reward_hacks.md`](knowledge/reward_hacks.md));
requested changes, and a test for every confirmed hack, become work packages that go through
Implement and Verify again. Performance numbers count only on the workload fixed at Setup and on the
serving path the task targets, and a gain only if it also holds at a held-out operating point fixed
with the workload ([`knowledge/perf_evaluation.md`](knowledge/perf_evaluation.md) §3).

Kernel-level work — tuning a kernel InferPilot wrote for a feature, or a kernel-level performance
opportunity — is handed to a **kernel workflow back end** in the Kernel phase between Implement and
Verify ([`knowledge/kernel_backends.md`](knowledge/kernel_backends.md)):

| `kernel_backend` | What happens to a hand-off |
|---|---|
| `none` (default) | recorded in the report with its package; nothing runs |
| `geak` | GEAK's `kernel_workflow/kernel_lane.js` runs it from the GEAK checkout given as `kernel_backend_dir` (nested once, as GEAK's e2e_workflow does) |
| `command` | any other CLI given as `kernel_backend_command` |

A gain comes back as a work package: the implementer integrates it and InferPilot's own gates judge it. A
back end that fails or finds nothing never blocks the task. An integrated kernel that then fails
Verify goes back to the functional kernel and is handed to the back end once more, with the failing
case added to its tests; it is integrated again only if it passes.

## Roles

Each phase calls a role: an agent that reads `roles/<role>.md`, does its part and returns structured
JSON; `inferpilot.js` holds the schemas and branches only on their required fields. Four rules hold across
the roles:

- only the **implementer** edits framework source, and only on the task's branch;
- the roles that measure (test engineer, verify engineer, performance engineer), the intake and the
  reviewer never edit framework source; the measuring ones never reuse anyone else's numbers but run
  everything again;
- the **curator** edits only InferPilot's own `knowledge/`, and proposes the edits as a pull request to InferPilot's
  repository for you to review and merge (with `INFERPILOT_REPO_TOKEN`; without it they stay uncommitted in
  the checkout);
- no role pushes or posts upstream: the PR engineer returns the exact `git push`, issue replies are
  drafts, a scan's daily report is a file for you to share; the one post is the curator's knowledge
  changes, as a pull request to InferPilot's own repository that you merge, and only with a token you set
  for that.

| Role | Phases | What it does |
|---|---|---|
| `intake` | Intake | turns a request in plain words, or a definition with gaps, into the task definition: decides the mode, fills fields with read-only checks, records defaults as assumptions, asks instead of guessing (a missing source of truth, model or workload, anything that would cost something to obtain), rejects what InferPilot does not do |
| `director` | Setup, Spec, Report | creates the run directory and the task's branch, pins commit, image and GPUs and runs the baseline; writes the feature spec with measurable acceptance criteria; writes the final report for every outcome, with the next step for a human |
| `scout` | Collect | gathers candidate tasks with the scripts below |
| `dispatcher` | Dispatch | sets each candidate's gap type, level, CUDA dependencies with their AMD starting points, effort and priority, and ranks them into a queue |
| `analyst` | Recon | maps every subsystem a feature needs to a file, function and hook in the target framework, and tables which assumptions of the source design hold on AMD; reads only |
| `architect` | Design, Plan | turns the recon or the profile into work packages that can each be tested alone, with invariants, memory budget, synchronization, startup rejections and a test plan; flags designs a human must review; assigns each performance opportunity to InferPilot, the kernel back end or Hyperloom |
| `issue_triager` | Triage, Reproduce | sets the priority and the action from the issue, its comments and linked PRs; drafts the reply; writes the reproduction that becomes the issue's gate |
| `ci_guardian` | Classify, Bisect | classifies a red job (code regression, dependency bump, test bug, flaky, infrastructure) and hands the last two over; finds and confirms the culprit commit and plans the fix |
| `implementer` | Implement | one work package at a time, smallest diff, neutral with the feature off, HIP-only or behind a switch, committed lint-clean with its tests registered; functional kernels with unit tests (tuning is a hand-off); applies existing PRs and verified kernel gains; rolls back a failed package |
| `test_engineer` | Implement | re-runs each package's unit test on its commit, checks that the test can fail (on the parent commit when the change alters behaviour, on a broken copy when it keeps it) and that behaviour with the feature off is unchanged |
| `verify_engineer` | Verify | runs the correctness gates (G0, G2–G4, composability, the mode's gate) on fresh servers against a same-pattern baseline, classifying and counting every failure |
| `investigator` | Investigate, Diagnose | runs the controls first, then one-mechanism-at-a-time experiments with a hypothesis ledger; returns fix work packages, a known issue the baseline shares, an inconclusive verdict, or (for an issue) a root cause outside the AMD side; probes only behind switches |
| `perf_engineer` | Profile, Performance | checks the run took the intended path, turns a steady-state trace into fixed tables (kernels, overlap, fuse patterns, layers), metrics and opportunities with upper bounds; evaluates changes at equal budget on the fixed workload, against noise, guard workloads and a held-out operating point, checking the known measurement hacks |
| `reviewer` | Review | reads the diff and its evidence as a maintainer would, starting from the catalog of known reward hacks: shapes, dtypes, parallel layouts, graphs, fallbacks, do-no-harm, the framework's conventions, a fair benchmark; requested changes, and a test for every confirmed hack, come back as work packages; can run on another model (`review_model`) |
| `kernel_liaison` | Kernel | turns a kernel hand-off into the back end's task, runs `command` back ends, judges the result on the back end's own verification and returns a gain as a work package |
| `pr_engineer` | PR | changes no code: checks hygiene, lint and CI registration (gaps go back as work packages), arranges one commit per work package without changing the tree, trial-merges main to report conflicts, checks feature off on the final commit, writes the PR body and the push command |
| `curator` | Report | writes what the run proved into `knowledge/` (known issues, feature skills, CUDA → AMD rows, new reward hacks, method lessons), every number traceable to the run, and proposes it as a pull request to InferPilot's repository |

## Setup

InferPilot runs inside a ROCm container, as root, the way GEAK does: Claude Code, InferPilot, git and the
framework's servers and tests all run there, next to the GPUs and the framework's Python environment.
Each task gets its own container, and everything it works on lives in that container: InferPilot's
checkout and its run directories in `/workspace`, and the image's own sglang in `/sgl-workspace/sglang`,
which the task changes on its own branch. Containers do not share them, so they stay out of each
other's way, and they last as long as their container. Claude Code is installed in each container too,
its first-start questions answered ahead, as GEAK's CI and Hyperloom do. From your home on the host a
container takes one file, your keys in `~/.inferpilot_env`, mounted read-only; the rest of your home stays
out, since the agents run as root with permissions auto-approved and a home holds credentials they do
not need.

Every step can be run again: it checks first, keeps what is already there and adds only what is
missing, and run_docker.sh, run again, takes you back into the container. In the commands,
`check || action` runs the action only when the check fails: on a new machine every action runs, and
a second run skips them. Nothing is deleted, overwritten or re-cloned, apart from the image build's
packaging changes in `/sgl-workspace/sglang` (step 3).

### What the host needs

- Linux with AMD Instinct GPUs (MI300 or MI350 series) and the ROCm driver loaded (`/dev/kfd` exists).
- Docker you can run (your user in the `docker` group, or sudo), git, curl and python3.
- Network access to claude.ai, github.com and Docker Hub, and about 45 GB of disk per sglang image.
- An AMD LLM gateway key (`LLM_GATEWAY_KEY`), an Anthropic API key or another gateway; without any, a
  Claude subscription.
- A read-only GitHub token (`GITHUB_TOKEN`), strongly recommended: scans and issue runs read GitHub's
  API.
- Model weights in a directory the container mounts (`/data`, `/mnt`, `/kmd2`, ...; any other through
  `EXTRA_MOUNTS`): InferPilot never downloads weights.

### 1. Keep your keys (host)

```bash
[ -e ~/.inferpilot_env ] || (umask 077 && echo 'export LLM_GATEWAY_KEY=<your AMD LLM gateway key>' > ~/.inferpilot_env)
(umask 077 && grep -qs GITHUB_TOKEN ~/.inferpilot_env || echo 'export GITHUB_TOKEN=<your GitHub token>' >> ~/.inferpilot_env)
```

The host needs no Claude Code: step 3 installs it in the container. Put your own values in place of
`<your AMD LLM gateway key>` and `<your GitHub token>` before you run these lines: each writes its key
only when `~/.inferpilot_env` does not have it yet, so a key is changed later by editing the file.
`~/.inferpilot_env` keeps the keys on the host, readable only by you; run_docker.sh mounts it read-only into
every container it starts, so create it before the container (one started without it can get a copy in
its own `/root/.inferpilot_env`).

- **LLM key (required).** With the AMD LLM gateway key, step 3 points Claude Code at the gateway
  (`https://llm-api.amd.com/Unified`, the key in its `Ocp-Apim-Subscription-Key` header), which serves
  the Claude models under their usual names. Outside AMD, write `export ANTHROPIC_API_KEY=sk-ant-...`
  instead, or another gateway (`ANTHROPIC_BASE_URL` with `ANTHROPIC_AUTH_TOKEN` or
  `ANTHROPIC_CUSTOM_HEADERS`). With a Claude subscription, run `claude setup-token` wherever you have
  Claude Code and a browser and write `export CLAUDE_CODE_OAUTH_TOKEN=<token>`; or leave the key out and
  log in at the first start, in every new container (step 4).
- **GitHub token (strongly recommended).** Scans and issue runs read issues, pull requests and CI
  trackers through GitHub's API. Without a token GitHub allows 60 requests an hour and 10 searches a
  minute per IP address, shared with everyone behind the same address, so a scan crawls and can stop
  partway; with one, 5000 requests an hour. Read-only is enough: on github.com, Settings → Developer
  settings → Personal access tokens → Fine-grained tokens, with *Public repositories (read-only)*
  access (a classic token with no scopes also works). Every agent in the container can read what
  `~/.inferpilot_env` holds, so keep your everyday token, the one you push with, out of it.
- **InferPilot repository token (optional).** Only for the curator's knowledge changes: as a pull request
  from a `inferpilot/knowledge/<run>` branch to main, which you review and merge. That takes a token for
  this repository alone, so it can write nowhere else: a fine-grained token with *Only select
  repositories* → `At1a8/inferpilot` and *Repository permissions* → *Contents* and *Pull requests*: *Read
  and write*, added to `~/.inferpilot_env` as `export INFERPILOT_REPO_TOKEN=<token>`. InferPilot pushes only
  `inferpilot/` branches; protect main on GitHub (require a pull request) so the token cannot change it
  either. Without it the knowledge changes stay in the container's checkout, and the report gives the
  command that proposes them.
- **Git identity (recommended).** The container carries no home, so no git config of yours: commits
  InferPilot makes there would read `root@<this host>`. Add `export INFERPILOT_GIT_NAME="<your name>"` and
  `export INFERPILOT_GIT_EMAIL=<your email>`; step 3 gives them to root.

### 2. Start a container (host)

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/At1a8/inferpilot/main/scripts/run_docker.sh)    # NAME defaults to your user name
```

This fetches [`scripts/run_docker.sh`](scripts/run_docker.sh) and runs it, so the host needs no copy of
InferPilot; in a checkout, `scripts/run_docker.sh` does the same. It is the usual ROCm `run_docker.sh`: a
privileged container with the GPUs and the data directories that exist (`/data`, `/mnt`, ...) mounted at
the same paths (`EXTRA_MOUNTS` adds more), and a root shell in its `/workspace`. Unlike many such
scripts it mounts neither your home nor the host's Docker socket (`DOCKER_SOCKET=1` mounts the socket):
InferPilot's agents run as root with permissions auto-approved, a home holds credentials (git, ssh) they do
not need, and the socket would put every container on the host within their reach. From your home it
takes `~/.inferpilot_env` alone, mounted read-only as root's `~/.inferpilot_env`.

- **Image.** Tasks test the newest sglang, so it starts from the newest daily `rocm/sgl-dev` build for
  ROCm 10 and MI35x ([`scripts/latest_image.py`](scripts/latest_image.py), e.g.
  `v0.5.23-rocm10-mi35x-20261103`; the `miles` images on the same page are for reinforcement learning
  and are skipped). A new day's image is pulled at its first use, about 43 GB. On MI30x, set
  `IMAGE_VARIANT=rocm10-mi30x`; for a fixed image, `IMAGE=rocm/sgl-dev:<tag>`. A container keeps the
  image it started from: run_docker.sh says so when you come back to one whose image is no longer the
  newest daily one, and a task there stops at Setup, before it creates anything, to ask whether to run
  on it anyway (`image_ok`), unless the task names that image (`image`). A scan goes on and names its
  image in the report.
- **What is tested.** The image provides PyTorch, aiter, sgl-kernel and its own sglang checkout,
  `/sgl-workspace/sglang`, with the whole history, which it installs sglang from in editable mode: what
  runs is what is checked out there, so there is no sglang to clone or name. Step 3 puts it on the
  current main; each task then works in it on its own branch from the newest main, which Setup fetches
  first. A baseline at the base commit and a scan of the current main read worktrees of the same
  repository under the run directory (about 0.2 GB each). The run records the image, the image's sglang
  version and the task's base commit.
- **One container per task.** Run again with the same `NAME`, run_docker.sh takes you back into that
  container, started again if it was stopped, with `/workspace` as you left it: it never removes a
  container. Give the next task a new container under another name (`NAME=<name>`), from the newest
  image; `docker rm -f <name>` removes one you are done with, and its `/workspace` with it. However many
  containers there are, one InferPilot task runs on a machine at a time ([Running it](#running-it)). A
  run_docker.sh of your own works too: mount your keys as root's `~/.inferpilot_env`
  (`-v $HOME/.inferpilot_env:/root/.inferpilot_env:ro`), or name the file with `INFERPILOT_KEYS`; in a container that
  mounts `/home`, the bootstrap also finds it in your home there (step 3).

`claude` is not there yet (`claude: command not found`): step 3 installs it.

### 3. Set up InferPilot in the container (once per container)

This puts InferPilot in the container's `/workspace` and makes the container ready to run it, as the table
below lists:

```bash
[ -d /workspace/inferpilot/.git ] || git clone https://github.com/At1a8/inferpilot /workspace/inferpilot
cd /workspace/inferpilot && python3 -B -m inferpilot.bootstrap
source ~/.bashrc
claude --version    # installed in this container
```

| Step | In a new container |
|---|---|
| keys | root's `~/.inferpilot_env`, which run_docker.sh mounts from your home, or the file `INFERPILOT_KEYS` names. In a container that mounts `/home`, the one in your home there: `INFERPILOT_HOST_HOME`, else, with a warning, the home of the user the container is named after (`<you>`, `<you>-0116`). Without a keys file the rest is still set up, and the next steps say where a key goes |
| `claude` | installed in the container with Claude Code's native installer (`~/.local/bin/claude`, put on `PATH`): the newest version, a large download (minutes without output on a slow link are the download). It stays at that version for the container's life (`DISABLE_AUTOUPDATER=1`); a new container gets the newest |
| Claude Code config | root's `~/.claude.json` answers the first start's questions ahead, as GEAK's CI and Hyperloom do: the onboarding, trust in `/workspace/inferpilot`, bypass permissions mode and, for an `ANTHROPIC_API_KEY`, using it. The rest of the file is kept; like everything of root's, it goes with the container |
| git | trusts checkouts on host mounts, which belong to you, not root (`safe.directory '*'`) |
| git identity | root commits as `INFERPILOT_GIT_NAME <INFERPILOT_GIT_EMAIL>` from the keys file (or the environment), so no commit reads `root@<this host>`; with neither it says so and sets nothing |
| shell | root's `~/.bashrc` gets `PYTHONDONTWRITEBYTECODE=1` (no root-owned `__pycache__` in checkouts on host mounts) and `DISABLE_AUTOUPDATER=1`, and loads the keys file, also one put in place later; with `LLM_GATEWAY_KEY` and no `ANTHROPIC_BASE_URL`, it points Claude Code at the AMD LLM gateway |
| framework | the image's `/sgl-workspace/sglang`, still as the image left it (detached at the commit it built, its packaging files rewritten for ROCm), is put on main: the packaging changes discarded, then `git checkout main` and `git pull`. Never `pip install -e` from it afterwards: its `pyproject.toml` is then upstream's, which would pull CUDA packages. A checkout already on a branch, or with other changes, is left as it is; `--framework-dir` names a host checkout instead (used as it is, or cloned with its whole history when missing) |
| checks | GPUs, sglang, API access, `GITHUB_TOKEN`, counting what the keys file sets (detect only) |

The clone runs only when `/workspace/inferpilot` is not there yet; the bootstrap never writes keys.

### 4. Start Claude Code and give it a task (inside the container)

```bash
cd /workspace/inferpilot && IS_SANDBOX=1 claude --dangerously-skip-permissions
```

`IS_SANDBOX=1` lets Claude Code auto-approve permissions as root, which InferPilot's agents need. Step 3
answered the first start's questions in Claude Code's config, so it opens on its input box. Should one
still come up (step 3 reported `claude-config` as `skipped`), three of the defaults would stop you:

| Question | Answer |
|---|---|
| Do you want to use this API key? (asked only for an `ANTHROPIC_API_KEY`; the gateway key is not) | **Yes**; the default, No, asks you to log in instead |
| Is this a project you trust? | **Yes, I trust this folder**; the default exits |
| Bypass Permissions mode | **Yes, I accept**; the default exits |

With a key, the first start asks for no login. If it asks you to select a login method, or shows
`Paste code here if prompted >`, Claude Code has no key in this shell: enter nothing, press Ctrl-C three
times in a row to leave, and read the end of step 3's output, which names what is missing. With a Claude
subscription and no token (step 1), that login is the way in, again in every new container: pick
*Claude account with subscription*, open the URL it prints in any browser, log in, and paste the code
the page shows at that prompt.

Once it runs, the bottom line shows `bypass permissions on` for the whole session: that is the mode
`--dangerously-skip-permissions` turned on, not a question. Leave it as it is (Shift+Tab would switch to
a mode that waits for your approval before commands, and InferPilot's agents would stall) and type the task
in the input box above it:

```
use /workspace/inferpilot/inferpilot.js with mode=scan, inferpilot_dir=/workspace/inferpilot
```

The scan changes no code. It writes the day's report on the framework since the last report (else the
last 7 days): the AMD CI on main (every failing job, since when it fails and what its log says), the
open AMD user issues under P0, P1 and P2, and the upstream changes AMD needs to follow. It ranks the
tasks in them into `exp/scan_<time>/task_queue.md`, each with the task definition that starts it, and
writes the report, without anything about this machine, to `exp/scan_<time>/daily_report.md` for you
to share; InferPilot posts it nowhere. One report a day is enough: when one is less than 24 hours old, the scan asks before it starts again
(answer `rescan`), and `upstream_repos` adds repositories to follow, such as vLLM. To take one task,
stopping after the reproduction to confirm the environment, name its fields:

```
use /workspace/inferpilot/inferpilot.js with mode=issue, task=https://github.com/sgl-project/sglang/issues/41299,
inferpilot_dir=/workspace/inferpilot, model_path=<weights>, gpu_ids=0, stop_after=Reproduce
```

or say it in plain words and let the Intake phase write the definition:

```
use /workspace/inferpilot/inferpilot.js with inferpilot_dir=/workspace/inferpilot, request="DeepSeek-V4.1 decode
is slow on MI355X, make it faster"
```

When something is missing (here: which fixed workload every measurement uses), the run starts
nothing and returns `status: needs_input` with one question per field. Answer them in the chat;
Claude Code runs `inferpilot.js` again with the same arguments plus `answers={"workload": ...}`.
[Running it](#running-it) lists all arguments.

### 5. After a run (in the container)

Every run that got past Setup has its directory in `/workspace/inferpilot/exp` (ignored by git), named
`<mode>_<slug>_<time>`: the mode, a short name of the task and its start in UTC, e.g.
`issue_41299-glm-dcp_20261106-071502Z` or `scan_20261106-064327Z`. `exp/index.md` lists every run of
the container with its status. A run directory holds:

| File | What it holds |
|---|---|
| `run.json` | the task definition, the intake's assumptions and the container, which holds the task's branch |
| `validation_report.md`, `run_report.md` | the report: the first when a PR is ready, the second for any other outcome (the result names it as `report_path`) |
| `experiment_ledger.md` | the environment, the baseline and every measurement |
| `logs/` | one log per server or benchmark launch, `<tag>_<time>.log` |
| `pr_body.md`, `patches/` | the PR description and the branch's commits as patches |
| `knowledge_pr.md` | the description of the curator's pull request to InferPilot's repository |
| `daily_report.md`, `task_queue.md` | a scan's daily report, for you to share (below); the ranked tasks with local detail, each with the definition that starts it, next to what each source found (`<source>.json`) |

A scan's `daily_report.md` is written to be shared, and `scripts/daily_report.py check` has made sure
it names nothing about this machine: open a new issue in InferPilot's repository on GitHub, title it
`InferPilot daily report <date>` and paste the whole file. Its first line, a hidden marker, lets the next
scan, in any container, find it there and mark what changed since.

InferPilot never pushes. The task's branch, `inferpilot/<slug>`, is in `/sgl-workspace/sglang`, and the report
gives the exact `git -C /sgl-workspace/sglang push ...`: run it here with credentials that can push.
Run directories go with the container; to keep one, or to push from the host, where your git
credentials are, copy it out on the host (`<name>` is the container's name, `<run>` the run
directory's):

```bash
docker cp <name>:/workspace/inferpilot/exp/<run> ~/
```

The copy is yours, not root's. On the host, in your sglang checkout, `git checkout -b inferpilot/<slug> <base>`
with the base commit the report names, then `git am ~/<run>/patches/*.patch`, and push from there.

### Re-running and options

Each step of `inferpilot.bootstrap` reports `kept`, `installed`, `updated`, `added`, `created`, `skipped` or
`failed`; a summary lists them, the next steps list only what is still missing, and the exit code is 1
when a step failed. In a new container it installs Claude Code, which takes a minute or two, and writes
its config and the git and shell settings. `--claude-only` runs only the keys and Claude Code steps,
`--check` only the checks. Inside a container, `cd /workspace && pip install -v git+https://github.com/At1a8/inferpilot` runs
step 3 the GEAK way, cloning InferPilot to `/workspace/inferpilot` (pip shows the output only with `-v`; a cached wheel skips it).

| Variable | Used by | Default |
|---|---|---|
| `IMAGE` | `scripts/run_docker.sh` | the newest daily `rocm/sgl-dev` image of `IMAGE_VARIANT` (the newest local one when Docker Hub cannot be reached) |
| `IMAGE_VARIANT`, `NAME`, `EXTRA_MOUNTS` | `scripts/run_docker.sh` | `rocm10-mi35x`; your user name; more directories to mount, space-separated |
| `DOCKER_SOCKET` | `scripts/run_docker.sh` | unset; `1` also mounts the host's Docker socket and `docker` |
| `INFERPILOT_LOCK_FILE` | `scripts/run_lock.py` | `/dev/shm/inferpilot-run.lock`, the machine's one-task lock |
| `INFERPILOT_HOME` | bootstrap | `./inferpilot`, or the checkout you run from |
| `INFERPILOT_KEYS` | bootstrap | root's `~/.inferpilot_env`, where run_docker.sh mounts yours: the keys file the container's shells load |
| `INFERPILOT_HOST_HOME` | bootstrap | unset; in a container that mounts `/home`, your home there, where the bootstrap looks for the keys file (without it, the home of the user the container is named after) |
| `INFERPILOT_REPO_URL`, `INFERPILOT_REF` | bootstrap | `https://github.com/At1a8/inferpilot.git`, its default branch |
| `CLAUDE_VERSION`, `CLAUDE_BIN_DIR` | bootstrap | `latest`, `~/.local/bin` |
| `CLAUDE_CONFIG_DIR` | Claude Code, bootstrap | unset (`~`): Claude Code's own, where its config lives; the bootstrap writes the first-start answers there |
| `INFERPILOT_GIT_NAME`, `INFERPILOT_GIT_EMAIL` | `~/.inferpilot_env`, bootstrap | unset; the identity root commits with in the container |
| `INFERPILOT_REPO_TOKEN` | `~/.inferpilot_env`, `scripts/propose_knowledge.py` | unset; a fine-grained token for InferPilot's own repository alone (Contents, Pull requests: read and write), with which the curator opens its pull request |
| `LLM_GATEWAY_KEY` | `~/.inferpilot_env`, the container's shell settings | unset; set, Claude Code runs on the AMD LLM gateway unless `ANTHROPIC_BASE_URL` names another |
| `INFERPILOT_FRAMEWORK_DIR`, `INFERPILOT_FRAMEWORK_REPO` | bootstrap: `--framework-dir`, `--framework-repo` | the image's `/sgl-workspace/sglang`; sglang |
| `INFERPILOT_SKIP_BOOTSTRAP` | bootstrap | unset; set it to install the package with no side effects (CI, images) |

## Running it

InferPilot is a GEAK-style workflow: `inferpilot.js` holds the control flow, `roles/` the agent instructions,
`knowledge/` the method, `templates/` the artifacts. In Claude Code, name `inferpilot.js` and its arguments
in one sentence, as in [step 4](#4-start-claude-code-and-give-it-a-task-inside-the-container). GEAK is needed in two cases
only: the `geak` kernel back end (`kernel_backend_dir`), and running `inferpilot.js` outside Claude Code,
which uses GEAK's workflow runtime (Node 20+; also what the dry run uses):

```bash
node /path/to/GEAK/interface/runtime/engine/run_workflow.mjs inferpilot.js --agent codex \
  --args '{"mode":"scan","inferpilot_dir":"'"$PWD"'","framework_dir":"/path/to/sglang"}'
```

| Argument | Meaning |
|---|---|
| `inferpilot_dir` | absolute path of this checkout (required) |
| `request` | the task in plain words, any language; the Intake phase turns it into the fields below (`roles/intake.md`), and fields given with it win |
| `answers` | replies to a `needs_input` result, `{"<field>": <value>}`, passed with the same arguments as before |
| `mode` | `feature` \| `issue` \| `ci` \| `perf` \| `scan` (required; inferred from `request` when not given) |
| `task` | the upstream PR/design, issue URL, CI job/cluster or performance goal (required except `scan`) |
| `goal` | what done means, measurable when the mode measures; the spec and the performance plan address it |
| `framework_dir` | the framework checkout each task works in, on its own branch from the current main (for sglang the image's `/sgl-workspace/sglang` by default, which the image installs sglang from) |
| `model_path` | the model the baseline and the gates run (required for `feature` and `perf`) |
| `framework`, `platform`, `gpu_ids` | target and environment (`sglang`; empty `gpu_ids` lets Setup pick idle GPUs, which every later role then uses through `HIP_VISIBLE_DEVICES`) |
| `workload` | the benchmark every measurement uses, e.g. `{"isl":8192,"osl":1024,"conc":64}`; required for `perf`, fixed at Setup |
| `container` | only when InferPilot runs on the host: the framework's container, reached with `docker exec` |
| `exp_root` | where run directories go (default `<inferpilot_dir>/exp`, which git ignores) |
| `kernel_backend` | `none` (default) \| `geak` \| `command`: where kernel hand-offs go |
| `kernel_backend_dir` | `geak` back end: the GEAK checkout providing `kernel_workflow/kernel_lane.js` |
| `kernel_backend_command` | `command` back end: shell command with `{task_dir}`, `{out_dir}`, `{gpu_ids}` |
| `kernel_budget` | optimization rounds for the kernel back end (default 6) |
| `apply_prs` | `perf`: candidate PRs to apply and A/B |
| `sources`, `since` | `scan`: any of `ci`, `issues`, `upstream` (the default) and `hardware`, `perf`; the start of the window (default: the previous report) |
| `image`, `image_ok` | the image a task must run on (default: the newest daily one of the container's ROCm and GPU; another one stops Setup to ask); `true` runs on the container's image anyway |
| `rescan`, `upstream_repos` | `scan`: `true` scans again although a report from the last 24 hours exists; more repositories to follow upstream, as `[{"name": "vllm", "url": "https://github.com/vllm-project/vllm"}]` |
| `stop_after` | stop after the named phase of the mode (e.g. `Design`, `Reproduce`); a phase the mode does not run comes back as a question |
| `resume_from`, `run_dir` | `feature`: `implement` continues a run that stopped at Design once a human approved `design.md` |
| `review_model` | the reviewer's model, passed with its agent call (GEAK's runtime takes a model per call; elsewhere it depends on the runtime) |
| `max_fix_rounds`, `max_investigate_rounds`, `max_review_rounds`, `max_work_packages` | budgets |
| `agent_retries`, `agent_timeout_ms` | an agent call is retried after an error (the retry checks what the failed attempt left), never after a timeout: the timed-out agent may still be running, so it goes on the report's attention list. Default timeout 3 h, 6 h for the verify and performance engineers and the investigator |

Everything but `inferpilot_dir`, `request`, `answers`, `exp_root`, `review_model` and the budgets is the task
definition; the result of every run that started records the definition it used (`task_def`), where
each field came from (`provenance`) and the intake's `assumptions`.

Human checkpoints: questions from the Intake (`needs_input`); any `stop_after`; L3 tasks and designs
the architect flags stop at Design (`awaiting_design_review`); a review the reviewer marks `blocked`
stops before the PR; the PR engineer returns the exact `git push` instead of pushing; issue replies
are drafts in the run directory. A run that has its run directory ends with a report, whatever its
status; one that ends at the Intake (questions, a rejection) or before Setup created its directory
(`busy`) returns only its result.

**One task per machine.** A run takes the machine before it creates anything — a lock in `/dev/shm`,
which every container started with `--ipc=host` shares with the host and no other machine sees — and
frees it at the end of its report. While another task holds it, a new run ends at once as `busy` and
names the holder. A lock whose run finished, or wrote nothing in its run directory for twice its
longest agent call (12 h by default), is taken over by the next run. A run in another container is
never taken over for silence, since its run directory is not visible from here; once that run is
gone, the second command frees the machine, as it frees one a run that died left behind:

```bash
python3 scripts/run_lock.py status            # who holds this machine
python3 scripts/run_lock.py release --force   # free it after that run died
```

The framework's own skills (sglang's `.agents/skills/`) are referenced in place, never copied: each
role reads the ones [`knowledge/frameworks/sglang/skills.md`](knowledge/frameworks/sglang/skills.md)
lists for it from the task worktree, so they match the code it changes, with ROCm differences in
[`knowledge/platforms/rocm_skills.md`](knowledge/platforms/rocm_skills.md).

## Scripts

Standard-library Python; the GitHub ones read `GITHUB_TOKEN` or `GH_TOKEN` (step 1; without a token
GitHub allows 60 requests an hour and 10 searches a minute per IP address).

| Script | Source of tasks |
|---|---|
| `scripts/scan_issues.py` | open AMD issues with linked PRs, replies and the projected 60-day auto-close date |
| `scripts/scan_ci.py` | the AMD CI on main from its own runs and logs (each failing job: since when, last passed, failed tests, first errors), and the Failure Trackers' clusters |
| `scripts/scan_upstream.py` | merged upstream commits: new CUDA gates, CUDA-first features, new models |
| `scripts/audit_platform_gates.py` | platform gates to re-check on a new architecture (capability checks true on gfx1250, wave32 assumptions) |
| `scripts/audit_host_syncs.py` | host-device sync and copy candidates on the hot path (leads for a profile) |

`scripts/restore_owner.sh` (POSIX sh) gives what root wrote in the container on a host mount back to
its owner (in `/workspace` there is nothing to give back); every agent runs it after a git write ([`knowledge/env_hygiene.md`](knowledge/env_hygiene.md)).
`scripts/check_framework_skills.py --framework-dir <dir>` checks the skills index against a framework
checkout: skills it names that are gone, and new ones it does not name yet (the scout runs it on every
scan). `scripts/find_skill.py "<query>"` lists the feature skills whose tags, id or feature name occur
in a request or task title, best match first (the intake and the director run it). `scripts/latest_image.py [--variant rocm10-mi30x]` prints the newest daily sglang image
(`run_docker.sh` starts from it; the director compares a run's image with it). `scripts/run_lock.py`
keeps one InferPilot task per machine (`acquire`, `release`, `status`). `scripts/daily_report.py check`
tells whether a daily report can be made public (no local path, hostname or internal site) and fits in
an issue; `latest` finds the newest report shared in InferPilot's repository, which the next scan, in any
container, continues from. Both only read.
`scripts/propose_knowledge.py` turns the curator's changes under `knowledge/` into a pull request
from `inferpilot/knowledge/<run>` to main, built on the current main in a throwaway worktree, and then
restores `knowledge/` in the checkout, so later runs read only merged knowledge.

## Tests

```bash
GEAK_HOME=/path/to/GEAK node tests/dry_run.mjs   # every control path of inferpilot.js on GEAK's runtime, no GPU
python3 -m unittest discover -s tests             # the scripts' parsers and rules, the bootstrap, Claude Code's workflow rules, the two READMEs in step; offline
```

## Layout

```
inferpilot/
├── inferpilot.js            # orchestration: five modes, gates, budgets, rollback, kernel back ends
├── roles/                   # intake, director, scout, dispatcher, analyst, architect, implementer,
│                            # test_engineer, verify_engineer, investigator, perf_engineer, issue_triager,
│                            # ci_guardian, kernel_liaison, reviewer, pr_engineer, curator
├── knowledge/
│   ├── dev_methodology.md   # phases, artifacts, gates (the contract)
│   ├── validation_doctrine.md, debugging_playbook.md, perf_evaluation.md, env_hygiene.md, pr_playbook.md
│   ├── acceptance.md        # what each kind of task must show, and who confirms it
│   ├── guardrails.md        # safety constraints and community conduct
│   ├── reward_hacks.md      # changes that move a number while breaking what it meant: tell, killed by
│   ├── dispatch.md, issue_triage.md, ci_triage.md, system_perf.md, composability.md, kernel_backends.md
│   ├── frameworks/sglang/feature_dev_map.md, platforms/rocm.md, known_issues.md
│   ├── frameworks/sglang/skills.md, platforms/rocm_skills.md   # sglang's own skills: index + ROCm notes
│   ├── kernels/aklr/skills.md, playbook.md   # AKLR's own docs (index); when a kernel gain lands in AKLR
│   └── feature_skills/      # validated plans per feature (hisparse-v2 first)
├── templates/               # task_definition, feature_spec, design, experiment_ledger, validation_report,
│                            # pr_body, issue_triage, ci_triage, perf_analysis, task_queue, acceptance
├── scripts/                 # task collection, audits, run_docker.sh, latest_image.py, run_lock.py,
│                            # restore_owner.sh
├── inferpilot/bootstrap.py  # setup step 3, in the container; safe to re-run
├── setup.py, pyproject.toml # packaging; setup.py runs the bootstrap when pip builds the wheel
├── tests/                   # dry run of inferpilot.js, unit tests of the scripts, the bootstrap, Claude Code's
│                            # workflow rules and the READMEs
├── README.md, README.zh-CN.md
└── docs/design.md
```

## Relationship to GEAK and Hyperloom

- A kernel's speed is the kernel workflow's job, on whichever back end is selected (GEAK's kernel lane
  is one): InferPilot writes a functional kernel with a unit test, hands over the op spec with the reference
  as the oracle, and integrates a verified gain.
- Flag and backend searches for a workload belong to Hyperloom and GEAK `e2e_workflow`; their upstream
  candidates are InferPilot inputs, and features InferPilot lands become their options.

## Origin

The method, validation doctrine, known-issue register, templates and the HiSparse V2 feature skill come
from GEAK's `framework_workflow` (branch `fangyuan/framework_agent_init`, commit `fc857c9`), which was at
its knowledge-first phase. InferPilot adds the orchestration that was planned there, the roles, the issue /
CI / performance / scan entry points from the design, and the task-collection scripts.
