# InferPilot, for agents other than Claude Code

You drive InferPilot when the user asks for a task: "run InferPilot ...", "use inferpilot.js with ...", or
a task in InferPilot's terms (an upstream feature to port, an AMD issue, a red CI job, a performance goal,
a scan). Claude Code runs `inferpilot.js` in its Workflow tool (README step 4). You have no such tool: you
start InferPilot's own runner, `runtime/run.mjs`, and watch the run (README, *Ways to run it*).

## Where it runs

InferPilot and its runs live in the task's ROCm container, in `/workspace/inferpilot` (README steps 2-3).
Inside that container (`/.dockerenv` exists) run the commands below as they are. Anywhere else, such as
the GPU host opened over Remote-SSH, run each one through `docker exec -i <container> bash -c '...'`.
The container is the user's: `scripts/run_docker.sh` names a new one `inferpilot_<user>_<MMDD>_<random>`
and labels it, so `docker ps --filter label=inferpilot.user=<user>` lists them (older ones carry the
user's login as their name). Ask which one when there are several, or none.

## Start a task

1. Write the task definition, the arguments of README *Running it*, to
   `/workspace/inferpilot/exp/<name>.json` with `"inferpilot_dir": "/workspace/inferpilot"`. Ask the user
   for what the mode requires and they have not said, or pass their words as `"request"` and let the
   Intake ask.
2. Start the runner so it outlives your turn, every role on the Cursor CLI (or `--backend claude`, on
   Claude Code):

   ```
   cd /workspace/inferpilot && mkdir -p exp && nohup node runtime/run.mjs inferpilot.js --backend cursor \
     --args-file exp/<name>.json --result-file exp/<name>.result.json > exp/<name>.log 2>&1 &
   ```

   `--backend cursor` needs `bootstrap --cursor` and a logged-in CLI (`agent status`); `--backend claude`
   needs `bootstrap --runner` and Claude Code's API access (README step 1).
3. Tell the user where it runs (the log and the result file) and stop; look again when they ask.

## Watch it

- `exp/<name>.log` names each phase and agent call. The run directory is the newest
  `exp/<mode>_<slug>_<time>Z`.
- In the run directory, `experiment_ledger.md` holds every measurement and decision; at the end comes
  `run_report.md` or `validation_report.md`. The result file holds the final `status` and `stopped_at`.
- A run that stops for the user (`needs_input`, `awaiting_design_review`, `blocked`) says why in its
  report: relay it, and continue with the same arguments plus what the user decides (`answers`, or
  `"resume_from": "implement"` with `"run_dir"`).

## Rules

- Never edit the files of a run directory to change its task. A change of goal, workload, GPUs or gates
  comes from the user and goes in new arguments; the run's agents reject a goal changed behind them.
- One task per machine (`python3 scripts/run_lock.py status`); check the GPUs (`rocm-smi`) before a run.
- Never print or write keys. InferPilot never pushes and never posts upstream: the push command is the
  user's.
