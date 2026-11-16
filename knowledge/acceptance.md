# Acceptance — what a task must show, and to whom, before it counts as done

The gates in [`validation_doctrine.md`](validation_doctrine.md) say what "correct" means for a change,
and [`perf_evaluation.md`](perf_evaluation.md) says how a gain is proven. This document says which of
them a task needs, how its acceptance criteria are written, who confirms them and how the result is
reported. The criteria are fixed before the work starts (§5).

Status: this is the definition. The orchestration does not build or check the acceptance matrix yet;
until it does, the director and the reviewer apply this document by hand.

## 1. Dimensions

| Dimension | What it shows | Typical checks |
|---|---|---|
| Correctness | the change does what it claims, and nothing when it is off | G0 feature off, G1 unit / oracle, startup rejection of unsupported combinations |
| Accuracy | model outputs keep their quality, or lose no more than a declared budget | §1.1 |
| Performance | the claimed gain exists, and nothing else regresses | gain items and guard items, §1.2 |
| Stability | no crash, hang or leak, under pressure and over time | G4 stress; a soak (at least 1 h at the target concurrency, memory flat); cancelled requests, oversized inputs, out-of-memory fallback; a restart |
| Compatibility | other platforms, features and configurations behave as before | the CUDA path untouched by a HIP change (`pr_playbook.md`); the combinations in `composability.md`; the platform matrix the task asks for |
| Engineering | upstream can take it | lint, tests registered in CI, docs, the PR description (`pr_playbook.md`), upstream CI green, a maintainer's approval |
| Requirement | the requester's own definition of done | §1.3 |

### 1.1 Accuracy

- **Lossless changes** (a kernel swap, scheduling, parallelism, memory layout): greedy outputs match
  the baseline on a fixed prompt set, and every divergence is checked (a near-tie between the top
  logits is acceptable, a systematic difference is not); GSM8K on all 1319 questions stays within the
  baseline's noise.
- **Lossy changes** (relaxed draft verification, quantization, sparse attention, KV compression): the
  budget is declared before anything is measured — the suite, the metric and the largest acceptable
  loss. It includes at least one set that is sensitive to the loss: a recall-heavy set such as
  simple_qa for relaxed verification, a long-context set for KV compression. Run at least 3 times per
  arm, since a single run of such a set moves by about ±0.5 points (`feature_skills/fly-dspark`).
- **Capability suites**, for what the change touches: long context (needle-in-a-haystack, G3),
  reasoning (GPQA, AIME), tool calls, multimodal input.

### 1.2 Performance

- **Metrics:** total and output throughput; TTFT and TPOT, mean and P99; throughput within an SLA
  (requests per second that meet their TTFT and TPOT targets); memory and capacity (KV tokens, the
  concurrency the scheduler reached); startup time when the change touches loading or compilation.
- **Gain items:** on the fixed target workload at an equal budget, the change wins beyond the noise
  band, and still wins at the held-out operating point (`perf_evaluation.md` §1, §3).
- **Guard items:** workloads where the change should be neutral, or where it costs. The change stays
  inside the noise band (a decode-side change at 100k in / 1k out, where prefill sets the total), or its
  cost is measured and stated.

### 1.3 Requirement

A task driven by a requirement (from a customer or another team) carries the requester's definition of
done as acceptance items: the model and weights, the precision, the deployment (TP / EP / PD, GPUs),
the workload (an input / output length distribution or a trace, the concurrency), the SLA targets, the
accuracy on the requester's data set, and the delivery date. The **requirement owner**, the person who
represents the requester, confirms these items before the work starts and signs off at the end (§6).

## 2. Default profiles

Which dimensions each kind of task must show. "When" names the condition under which an item becomes
required; the task may add items, never drop a required one.

| Dimension | Upstream feature | Issue fix | CI fix | Performance | Requirement |
|---|---|---|---|---|---|
| Correctness | required | required; the reproduction becomes a regression test | required; the failing test passes on every lane it failed on | required | required |
| Accuracy | required; a lossy change declares its budget | when the issue is an accuracy loss | no regression | required, as lossless | on the requester's data set |
| Performance | when the feature claims a gain | when the issue is a performance problem: the reported number comes back | — | required: gain, guard and held-out point | the requester's SLA |
| Stability | required (G4) | when the issue is a crash or hang | a flaky test passes repeatedly (§4) | when the change touches memory or scheduling | required, with a soak |
| Compatibility | required | required | required | required | the requester's platforms |
| Engineering | required | required | required | required | when the change goes upstream |
| Requirement | — | when a requester reported it: the reporter confirms the fix | — | when a requester asked for it | required; the requirement owner signs off |

A scan changes no code; its daily report has its own checks (`scripts/daily_report.py check`).

**Strictness.** Every item is either blocking or report-only, decided when the matrix is fixed. A P0
hotfix may ship on the minimal set: the reproduction passes, G0 holds, accuracy and the guard workloads
are not worse. The rest of its profile becomes follow-up items, listed in the report.

## 3. An acceptance item

Each item is one row of the acceptance matrix (`../templates/acceptance.md`):

| Field | Content |
|---|---|
| ID | `COR-`, `ACC-`, `PRF-`, `STB-`, `CMP-`, `ENG-`, `REQ-` plus a number |
| Dimension and metric | e.g. accuracy, simple_qa score |
| Workload or data set | lengths, concurrency and budget, or the data set with its harness, sampling and seed |
| Baseline | same image, commit family and launch line, only the switch differs (`validation_doctrine.md`, same-pattern baseline) |
| Rule | an absolute threshold, a change relative to the baseline, a non-inferiority margin, or a gain beyond the noise band |
| Runs and statistics | runs per arm, how lanes are paired, how the noise band is set (§4) |
| Blocking | yes or report-only |
| Confirmed by | InferPilot (automatic), a human reviewer, or the requirement owner |
| Result and evidence | filled at the end: the measured value, the verdict, the path of the logs and tables |

Three items for FLy on DSPARK (`feature_skills/fly-dspark`), written in this form after the fact:

| ID | Item | Rule | Measured |
|---|---|---|---|
| ACC-2 | simple_qa, all 4326 questions, LLM-judged, at the defaults | at most 1 point below the mean of at least 3 baseline runs; blocking | −0.8 points (5 runs against 14 baseline runs): pass |
| PRF-1 | 8k in / 1k out at concurrency 32 / 64 / 128, two lanes swapped every round, 3 rounds | total throughput at least +1.5%, beyond the noise band; blocking | +2.3%: pass |
| PRF-2 | 100k in / 1k out, guard | total throughput and mean TPOT within ±1%; blocking | inside the band in round 1; the swapped round is still missing |

## 4. Statistics

- **Noise band:** set from repeated baseline runs (their range, or the standard error of the mean) and
  stated next to every result.
- **Runs:** failure rates at least 4 per configuration (`validation_doctrine.md`); performance at least
  2 timed runs per arm after a warm-up, more when the spread is wide (`perf_evaluation.md` §3); a lossy
  accuracy budget at least 3 runs per arm; a flaky test passes in every one of at least 10 runs.
- **Two lanes at once** (two servers on one node): swap the arms between the lanes every round, and
  report the result both pooled and paired by round.
- **Verdicts:** a gain counts when it exceeds the noise band and holds at the held-out point; a guard
  holds when the change stays inside the band. Every run is reported, with its range, never only the
  best (`reward_hacks.md` P9).

## 5. Fixed before the work

The acceptance matrix is written and confirmed before implementation: in the Spec phase for a feature,
at Setup together with the workload for a performance task, and with the requirement owner's
confirmation for a requirement. Once results are seen it is not changed. A dropped item, a loosened
rule or a blocking item made report-only is `reward_hacks.md` C1, and a changed workload is P1. An item
that has to change restarts the evaluation, and the report says why.

## 6. Who confirms

- InferPilot judges the automatic items and records their evidence. It never declares acceptance on a
  human's behalf.
- An upstream PR is accepted by the maintainers' review and the upstream CI.
- A requirement is accepted by the requirement owner, who signs the acceptance report; InferPilot
  supplies the evidence.
- A failed blocking item can be waived only by the person who confirms it, with the reason written in
  the report.

## 7. Reporting

The final report carries the acceptance matrix with a result for every item: pass, fail, waived (by
whom, and why) or not run (and why), each linked to its evidence. A requirement-driven task also gets
an acceptance report of its own, written for the requirement owner: the requester's items first, then
everything else.
