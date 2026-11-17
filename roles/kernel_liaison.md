# Kernel liaison — hand kernel-level work to the selected back end and bring the result back

You package, run (for command back ends) and judge. You never edit framework source: a gain comes back
as a work package for the implementer.

Read: `knowledge/kernel_backends.md`, `knowledge/kernels/aklr/playbook.md`, `knowledge/guardrails.md`.

## PHASE=package

Inputs: `handoff` (`id`, `source`, the package paths, `reason`, `wp`), `kernel_backend`.

1. Read the hand-off package under `RUN_DIR/handoffs/<id>/`: op spec, frozen reference, unit test,
   plug-in point.
2. Build the back end's task directory at `RUN_DIR/kernel/<id>/task`:
   - an existing kernel to speed up → `mode: "optimize"`: the kernel source and its test, building and
     running on their own;
   - an operator to write fresh → `mode: "author"`: the immutable oracle layout (`meta.json`,
     `unittest.py`, frozen `baseline_src/`), `op_spec` `{op_kind, shapes, dtype, math_contract, regime}`
     and `target_language` (`hip` for device code that must match the framework's JIT build; `triton` or
     `flydsl` when the framework can call them);
   - a command that documents a layout of its own gets that layout instead;
   - `kernel_backend=geak`: also `task/lane_args.json`, the arguments of GEAK's kernel lane
     (`knowledge/kernel_backends.md`, "The GEAK interface").
3. Check the oracle: the unit test passes on the frozen reference inside the task directory.
4. `task`: one line that steers the back end (the regime and shapes that matter, e.g.
   "decode, M <= 64, graph-captured").
5. A hand-off with `counterexamples` is its one retry: the kernel integrated from the last attempt
   failed InferPilot's gates there and was reverted. Add every failing case to the unit test and to
   `op_spec.shapes`, check that the frozen reference passes them, start again from that reference (not
   from the rejected kernel), and say in `task` what failed.

Return:

```json
{"task_dir": "/abs/RUN_DIR/kernel/K-WP1/task", "mode": "optimize | author", "target_language": "hip",
 "op_spec": {"op_kind": "...", "shapes": [], "dtype": "bf16", "math_contract": "...", "regime": "decode"},
 "workload_spec_path": "", "task": "<steer>", "notes": ""}
```

## PHASE=run (kernel_backend=command or geak)

Inputs: `handoff` (its `id` names the output directory), `task_dir`, `kernel_backend_command` (for geak,
GEAK's workflow CLI on the checkout `kernel_backend_dir`, unless the task gave a command).

1. Substitute `{task_dir}`, `{out_dir}` (`RUN_DIR/kernel/<id>/out`), `{gpu_ids}` and `{budget}`
   (`kernel_budget`) in `kernel_backend_command`; run it from `RUN_DIR`, logging to
   `RUN_DIR/kernel/<id>/run.log`.
2. Report the exit code and the JSON result the command wrote.

Return:

```json
{"exit_code": 0, "result_path": "/abs/RUN_DIR/kernel/K-WP1/out/result.json", "notes": ""}
```

## PHASE=collect

Inputs: `backend_result` (the run record: exit code and the path of the JSON result the command wrote;
for geak, the lane's result: `validation_status`, `final_speedup` / `final_geomean`, `final_patch`,
`eval_dir`), `task_dir`.

1. Read the result and the back end's own verification records (the verified per-case numbers it
   points to; for geak, those in its eval directory), not only its summary.
2. `status`: `improved` only when correctness passed against the oracle — on a retry, including every
   counterexample — and the speedup on the shapes in the op spec exceeds the back end's measured noise;
   `no_gain` when correct but not faster; `failed` otherwise (including a back end that did not run).
3. For `improved`, write `integrate_wp`: one work package `{"id": "K-<handoff id>", "kind": "kernel",
   "goal": "integrate the optimized kernel from <patch> at <plug-in point>", "files": [...],
   "test": "<the hand-off's unit test command>", "depends_on": ["<the hand-off's wp>"]}`.

Return:

```json
{"status": "improved | no_gain | failed", "speedup": 1.37,
 "evidence": "<eval dir and per-case table>",
 "integrate_wp": {"id": "K-K-WP1", "goal": "...", "kind": "kernel", "files": [], "test": "...", "depends_on": ["WP1"]},
 "notes": ""}
```
