# Acceptance matrix — <task>

<!-- Fixed before the work starts, and not changed after results are seen (../knowledge/acceptance.md
     §5). One row per item; the profile's required items come first. The Result and Evidence columns
     are filled at the end. -->

| | |
|---|---|
| Task | <mode, title, link> |
| Profile | upstream feature / issue fix / CI fix / performance / requirement |
| Requirement owner | <name; only for a requirement-driven task> |
| Baseline | <image, commit, launch line; only the switch differs> |
| Fixed on | <date and time, with the time zone>, by <who> |

## Items

| ID | Dimension | Metric | Workload / data set | Rule | Runs / statistics | Blocking | Confirmed by | Result | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| COR-1 | correctness | G0 feature off | the baseline suites, switch off | identical to the baseline | 1 | yes | InferPilot | | |
| ACC-1 | accuracy | GSM8K, 1319 questions | <harness, sampling, seed> | within the baseline's noise | ≥3 runs per arm | yes | InferPilot | | |
| PRF-1 | performance (gain) | total tok/s | <input / output lengths, concurrency, budget> | ≥ +<x>% beyond the noise band, also at the held-out point | ≥2 timed runs per arm; lanes swapped every round | yes | InferPilot | | |
| PRF-2 | performance (guard) | total tok/s, mean TPOT | <workload where the change should be neutral> | within the noise band | as PRF-1 | yes | InferPilot | | |
| REQ-1 | requirement | <the requester's metric> | <the requester's workload or data set> | <the requester's target> | <as agreed> | yes | requirement owner | | |

## Waivers

| ID | Reason | Waived by | When |
|---|---|---|---|

## Sign-off

| Role | Name | Date | Decision |
|---|---|---|---|
| InferPilot (automatic items) | | | |
| Requirement owner | | | |
