---
id: fly-dspark
title: "FLy on DSPARK speculative decoding for DeepSeek-V4.1 (sglang, ROCm gfx950): accept a rejected draft token when the target is uncertain there and the next W drafts pass"
kind: feature_skill
authors: [At1a8]
tags: [fly, entropy-gated verification, deferred verification, lossy verification, speculative decoding, speculative sampling, rejection sampling, draft acceptance, accept length, dspark]
match:
  framework: sglang
  framework_ref: 58241600c5          # PR head: the W2/0.5 defaults (4e85e85498) merged with main 55ceff1114 (Nov 2026); sweeps measured on ed3831842a (main 410e64bb84)
  feature: fly_speculative_verification
  model_arch: [deepseek_v4]          # DeepSeek-V4.1-Flash with its bundled DSPARK block drafter
  platforms: [rocm-gfx950]
depends_on: ["vllm-project/vllm#53987", "vllm-project/vllm#66593"]   # FLy in vLLM; normalized entropy gate
reference_pr: "sgl-project/sglang#50238"
expects:
  accuracy: "lossy by design: gsm8k and gpqa_diamond within noise; simple_qa loss grows with the deferral rate (default W2/0.5 -0.8pp, W2/0.7 -0.4pp, W2/0.3 -1.6pp, pooled)"
  long_context: "not claimed: FLy changes only the accept decision (G3 not run)"
  stability: "no crash or failed launch over every sweep server (G4 as a dedicated stress test not run)"
  perf: "8k/1k total tok/s +1.9..3.7% for W2 (default W2/0.5 +2.4%), accept length +6..12% on the accuracy datasets; 100k/1k unchanged (prefill-bound)"
validation:
  status: validated                  # for the gates claimed below: G0, G1, G2, G5
  last_verified: "2026-11-16"
  framework_commit: "sweeps on ed3831842a (branch fangyuan/dspark_fly merged with main 410e64bb84); defaults and CI on 4e85e85498"
  model: DeepSeek-V4.1-Flash
  gpu: "gfx950 x4 (TP4/EP4); two such lanes on one 8-GPU node"
  measured:
    unit: "123 passed + 33 subtests: scripts/test_dspark_fly.py plus 7 DSPARK accept suites, on the merged tree"
    ci_gsm8k: "0.9682 at accept length 4.45 with the defaults (W2/0.5), threshold 0.91, test_deepseek_v41_flash_fly_eval_mi35x.py; earlier at W2/0.3: 0.9674 / 0.9659"
    default_w2_t0.5: "8k/1k total tok/s +2.4% (6 runs, two campaigns), TPOT mean -2.1%; accept length +9.1% on the three accuracy sets; simple_qa -0.8pp (5 runs vs 14 baselines); gsm8k and gpqa_diamond unchanged"
    w2_t0.7: "8k/1k +1.9% (9 runs, three campaigns); accept length +5.7%; simple_qa -0.4pp (8 runs, s.e. 0.15pp)"
    perf_100k: "every configuration within +-0.6% total tok/s at 100k/1k (single runs)"
  artifact: "sglang PR #50238, branch fangyuan/dspark_fly; tables in references/results.md"
role: advisory_prior
---

## When to use

- Adding FLy (entropy-gated deferred verification, arXiv 2511.22972; vLLM #53987, entropy normalized
  in #66593) to sglang's DSPARK speculative decoding on ROCm, or extending this port.
- Porting FLy to another speculative algorithm in sglang (EAGLE, DFLASH, ...): read the assumption
  table; the pre-pass design below is the part that carries over.
- Choosing FLy's window and entropy threshold for a model: the sweep tables are in
  `references/results.md`; the recipe in Validation is how they were measured.
- FLy is **lossy**: it accepts tokens the target distribution would reject. It must stay opt-in. Its
  loss shows on a knowledge-recall set (simple_qa), not on math, and G2 runs one only when `goal` says
  the change may lose accuracy.

## Source design (vLLM #53987, #66593)

Native verification stops at the first rejected draft token: for a greedy request one that differs
from the target argmax, for a sampled one a token whose draw fails the chain rule `u * q(x) < p(x)`.
FLy defers that rejection, i.e. accepts the draft token, when all of these hold:

- the target's normalized top-k entropy at that position is at least `fly_entropy_threshold`: the
  `fly_entropy_top_k` largest processed target scores, renormalized over themselves, give
  `-sum(p log p) / log(k)` in [0, 1] (0 for k = 1);
- the target does not mask the token (finite logit, p(x) > 0) and, when sampling, the draft proposed it
  (q(x) > 0);
- the next `fly_window_size` draft tokens pass native verification and lie inside the verified drafts.

A rejection inside the window rejects at the original position; a position without a full window
after it is never deferred; several non-overlapping deferrals can happen in one step. vLLM runs this
inside its rejection sampler as a pending-rejection state machine. Settings: window an integer in
[1, K) with default `min(6, K - 1)`, threshold >= 0 (default 0.3), top-k >= 1 (default 3); the entropy
is taken on processed logits (after temperature, top-k/top-p, min-p; raw logits for greedy). vLLM
never swept the settings: its ModelRunner V2 evaluation used W2/0.3 for four models (DeepSeek-V4-Flash
with DSpark at K = 7 among them), its V1 evaluation the default window 6 at K = 15 or 25.

## Facts about the target

| Field | Value |
|---|---|
| drafter | DeepSeek-V4.1-Flash's bundled DSPARK block drafter, `dspark_block_size` 5 in the checkpoint config |
| gamma | `--speculative-dspark-block-size` = drafts per verify (5); verify window gamma + 1; a gamma other than the checkpoint's only warns |
| sampling | no `generation_config.json`: requests without sampling fields run at T = 1.0, top_p = 1.0 |
| accept paths (`dspark_verify.accept_draft_tokens`) | greedy (`AcceptGreedy`, torch / Triton); classic sampling (`AcceptSampling`, needs `target_probs`); split accept (`dspark_accept_split`, ROCm only, `_SPLIT_ACCEPT`); vocab-shard accept (`sharded_accept_hip`, ROCm only) — the default for this model at T = 1.0; the static greedy graph (`DsparkVerifyEpilogue._accept_static_greedy`) and the sampling accept graph |
| ragged verify | rows can verify fewer than gamma drafts (`cutoff_verify_lens`) |

## Assumption table (vLLM → sglang DSPARK on ROCm)

| vLLM assumption | Holds? | Consequence |
|---|---|---|
| one rejection-sampler kernel where FLy can live | No: DSPARK has five accept implementations | FLy is a **pre-pass** that rewrites the accept's input (greedy: overwrite the target argmax with the draft token; sampling: set the position's draw u to 0, which the chain rule then accepts); every accept stays as it is and takes the same decisions as vLLM's state machine |
| entropy on processed logits | Partly: greedy has raw logits, split / shard have logits and temperature, the classic path has only `target_probs` | entropy from what each path has: logits x 1/T in one pass with a running top-k (`_row_topk_entropy`), or probabilities renormalized over their top-k (`_row_topk_prob_entropy`) |
| the sampler draws u | No: the vocab-shard accept draws with Philox inside `_accept_kernel` | `_fly_kernel` recomputes the same draws (key, offset, `_ACCEPT_STEP`) and hands `_accept_kernel` a FLY_OK mask (`HAS_FLY` constexpr) |
| the target row on one device | No: the shard accept keeps per-rank statistics | p(x), q(x) merged from the ranks' statistics; every rank holds the whole target row for the entropy, so the ranks agree |
| every row verifies K drafts | No: ragged verify | the window must lie inside the row's `verify_lens - 1` drafts |
| CUDA | No: ROCm only (the request) | the hook rejects other platforms; the worker builds the config only under `is_hip()`; split and shard accepts are already HIP-only |
| K large (V1: 15 / 25) | No: gamma = 5 | vLLM's default window min(6, gamma - 1) = 4 lets only the first draft be deferred, so FLy is nearly off there; this port defaults to window min(2, gamma - 1) and threshold 0.5 |

## Design as built

1. **Flags.** `--speculative-use-fly`, `--speculative-fly-window-size` (unset: min(2, gamma - 1)),
   `--speculative-fly-entropy-threshold` (default 0.5), `--speculative-fly-entropy-top-k` (default 3)
   in `srt/arg_groups/fields/spec.py`, appended at the end of `srt/arg_groups/field_order.py`. The
   defaults are the sweep's balanced setting (W2/0.5, below), not vLLM's (min(6, K - 1), 0.3).
2. **Startup validation** (`srt/arg_groups/speculative_hook.py`): DSPARK only; ROCm only; not with
   `--speculative-use-block-verification`. `_handle_dspark` resolves the config with
   gamma = num_draft_tokens - 1, declares the resolved window, and logs a "FLy verification is lossy"
   warning that prints the `FlyConfig`.
3. **`FlyConfig` / `resolve_fly_config`** (`srt/speculative/fly_config.py`): 2 <= gamma <= 31 (one
   accept bit per draft in an int32); 1 <= window < gamma, default min(2, gamma - 1); 1 <= top-k <= 16
   (the Triton entropy unrolls the top-k); threshold >= 0, NaN rejected, no upper bound (as vLLM).
4. **Worker** (`dspark_worker_v2.py`): `self._fly` is built only when `is_hip()` and the flag is on,
   and passed to `DsparkVerifyEpilogue(fly=...)` and `TargetVerifyExecutor(fly=...)`; the sampling
   accept graph reads `epilogue.fly`.
5. **Pre-passes** (`kernels/ops/speculative/fly.py`): `FlyGreedyPrepass` (reference
   `apply_fly_greedy`, Triton `_fly_greedy_prepass_kernel`) and `FlySamplingPrepass` (reference
   `apply_fly_sampling`, Triton `_fly_sampling_prepass_kernel`). Per row: native bits, usable bits
   (unmasked, proposed), window bits; walk from draft 0; at a non-native position defer if usable,
   the window is full and inside the verified drafts, and the entropy (computed only there) reaches
   the threshold; stop at the first rejection that is not deferred.
6. **Accept integration** (`kernels/ops/speculative/dspark/`): `accept_greedy` / `accept_greedy_triton`
   and `_accept_sampling_core` call the pre-passes when `fly` is set; the split accept runs
   `_fly_prepass_kernel` on a copy of the draws; the shard accept runs `_fly_kernel` and ORs FLY_OK
   into `_accept_kernel`. Every addition sits behind `fly is not None` or a `HAS_FLY` constexpr.
7. **Entropy kernels**: one pass over the vocab keeping a running top-k (no sort), softmax over the
   top-k, divide by log(k); k = 1 returns 0, so a top-k of 1 defers only at threshold 0.

## Invariants

| # | Invariant | Enforced by / how a violation shows |
|---|---|---|
| I1 | Feature off ⇒ `fly is None` everywhere and the original kernels compile (`HAS_FLY` false) | G0: DSPARK accept suites unchanged; baseline accuracy equal to earlier baselines |
| I2 | CUDA never runs FLy code | hook rejection off ROCm; worker gate `is_hip()`; split / shard paths HIP-only |
| I3 | The pre-pass decisions equal vLLM's in-kernel state machine | reference vs Triton parity; split and shard accepts against the chain rule on the same draws |
| I4 | A deferral needs a full window inside the row's verified drafts; a rejection inside the window rejects the original position | unit tests `test_rejection_inside_window_rejects_original`, `test_window_stays_inside_verified_drafts` |
| I5 | Masked tokens (logit -inf, p = 0) and unproposed ones (q = 0) are never deferred | unit tests `test_masked_draft_is_not_deferred`, `test_unproposed_draft_is_not_deferred` |
| I6 | All TP ranks take the same decisions | shard path computes from merged statistics and recomputes the same Philox draws |
| I7 | The bonus token comes from the first rejection that is not deferred | the accepts are unchanged |

## Work packages

| WP | Goal | Files (sglang) | Closed by |
|---|---|---|---|
| 1 | settings and startup rejections | `srt/speculative/fly_config.py`, `srt/arg_groups/fields/spec.py`, `srt/arg_groups/field_order.py`, `srt/arg_groups/speculative_hook.py` | `scripts/test_fly_args.py`; `test_server_args.py` positional-slot tests |
| 2 | reference pre-passes and the entropy | `kernels/ops/speculative/fly.py` (torch) | `TestFlyGreedyAccept`, `TestFlyEntropy`, `TestFlySamplingDraws` |
| 3 | Triton pre-passes | `kernels/ops/speculative/fly.py` (Triton) | `TestFlyTritonParity`, `test_triton_prepass_matches_torch` |
| 4 | accept integration | `dspark/dspark_accept.py`, `dspark/dspark_accept_split.py`, `dspark/sharded_accept_hip.py` | `test_split_accept_matches_chain_rule`, `test_shard_accept_matches_chain_rule`; DSPARK accept suites unchanged |
| 5 | worker / verify plumbing | `dspark_worker_v2.py`, `dspark_verify.py`, `dspark_sampling_accept_graph.py`; test stubs in `test/registered/unit/spec/test_spec_registry.py`, `test/registered/unit/speculative/test_dspark_engram_commit_in_graph.py` | the server logs `FlyConfig(...)`; accept length rises; the two stub tests pass |
| 6 | CI accuracy test | `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_fly_eval_mi35x.py` (nightly suite `nightly-amd-4-gpu-mi35x-deepseek-v41-flash`) | gsm8k 0.9682 at the defaults (0.9659 / 0.9674 earlier at W2/0.3); threshold 0.91 |

## Launch configuration (as validated)

DeepSeek-V4.1-Flash, TP4/EP4 on gfx950 (`scripts/serve_dspark_fly.sh`):

```
SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank SGLANG_USE_ROCM700A=0 TORCH_BLAS_PREFER_HIPBLASLT=1 \
SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true SGLANG_HIP_FP4_INDEXER_ROWGROUP=1 SGLANG_HIP_DSV41_INDEX_K_PAGE8=1 \
sglang serve --trust-remote-code --model-path <model_dir>/DeepSeek-V4.1-Flash --tp 4 --ep-size 4 \
  --mem-fraction-static 0.85 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --enforce-shared-experts-fusion --fp8-gemm-backend aiter --enable-decoder-swa-bounded-replay \
  --cuda-graph-max-bs-decode 128 --max-running-requests 128 --cuda-graph-backend-prefill disabled \
  --reasoning-parser auto --tool-call-parser auto \
  --speculative-use-fly
```

- `--speculative-use-fly` alone runs the defaults (window 2 at this block size, threshold 0.5, top-k 3);
  add `--speculative-fly-window-size` / `--speculative-fly-entropy-threshold` for another setting.
- Startup log to check: `FLy verification is lossy ... (FlyConfig(window_size=2, entropy_threshold=0.5, entropy_top_k=3))`;
  without it the server ran standard verification.
- The CI test uses its own launch (mem fraction 0.8, 256 running requests), FLy at its defaults, and
  checks the resolved window and threshold in `/server_info`.

## Validation recipe

| Gate | Command / setup | Expected |
|---|---|---|
| G0 | DSPARK accept suites with FLy off: `kernels/ops/speculative/test_reject_sampling_split.py`, `test_block_verification.py`, `unit/spec/test_sharded_verify.py`, `test_dspark_greedy_draft_accept.py`, `test_dspark_blockwise_rejection.py`, `test_dspark_accept_tail.py`, `unit/speculative/test_dspark_sampling_accept_graph.py`; baseline servers in every round of the A/B | all pass; baseline accuracy equal to earlier baselines |
| G1 | `scripts/test_dspark_fly.py` (with the suites above) and `scripts/test_fly_args.py` | 123 passed + 33 subtests together; args tests pass on CPU |
| G2 | evalscope gsm8k (1319); gpqa_diamond (198) and simple_qa (4326, LLM judge) as well when `goal` says the change may lose accuracy (this validation ran all three); thinking on, `reasoning_effort` max, `max_tokens` 102400, the server's default sampling; `POST /flush_cache` before each dataset and `avg_spec_accept_length` from `/server_info` after it; plus the CI test | baseline (5 runs): gsm8k 0.9713, gpqa_diamond 0.9101, simple_qa 0.4452 (range 0.4434–0.4475), accept length about 3.79 / 3.15 / 3.06. CI: gsm8k >= 0.91 |
| G5 | `scripts/lockstep_ab.sh`: two TP4 lanes benching at the same time, arms swapped between lanes each round, `scripts/bench_random.sh` with 8000 / 1000 tokens at concurrency 32 / 64 / 128 (prompts 4x, warm-up 2x the concurrency, T = 1.0, cache flushed after warm-up); 3 rounds; summary `scripts/ab_table.py` | baseline (6 runs) 34,727 / 45,553 / 54,169 total tok/s; see the parameter table below |

- G3 / G4 were not run: FLy adds no memory or long-context path. A 100k/1k run (single runs, every
  configuration) moved total throughput by at most ±0.6%: at 100k input the total is ~99% prefill.
- Engagement evidence: the `FlyConfig(...)` startup line, and accept length on the accuracy datasets
  (FLy configurations raise it, W4 barely).

## Choosing the window and the threshold (gamma = 5)

Pooled over every campaign: simple_qa against all 14 baseline runs (mean 0.4443); 8k/1k as the mean
change over c32 / 64 / 128, each run against its own campaign's baselines; accept length from the
three-round sweep, the mean change over the three accuracy sets. W1 and W4: one earlier run each.

| Setting | simple_qa (runs) | accept length | 8k/1k total tok/s (runs) |
|---|---|---|---|
| W1/0.3 | −2.9pp (1) | +21.7% | +8.1% (1) |
| W1/0.5 | −2.9pp (1) | +17.3% | +7.5% (1) |
| W2/0.2 | −2.0pp (5) | +11.9% | +3.7% (6) |
| W2/0.3 (vLLM's evaluation) | −1.6pp (5) | +11.6% | +3.0% (6) |
| W2/0.4 | −1.2pp (4) | +10.3% | +3.2% (6) |
| **W2/0.5 (the default)** | **−0.8pp (5)** | **+9.1%** | **+2.4% (6)** |
| W2/0.6 | −0.6pp (4) | +7.7% | +2.5% (6) |
| W2/0.7 | −0.4pp (8) | +5.7% | +1.9% (9) |
| W3/0.3 | −0.8pp (5) | +5.6% | +1.4% (6) |
| W3/0.7 | −0.3pp (4) | +2.5% | −0.1% (6) |
| W4/0.3 (vLLM's default at gamma 5) | +0.8pp (1) | +2.1% | +0.1% (1) |

- The window decides how much FLy does; the threshold decides what it costs. Within W2, from threshold
  0.2 to 0.7 the 8k gain falls from +3.7% to +1.9% while the simple_qa loss shrinks from 2.0 to 0.4
  points. W3 is never better than W2 at an equal cost; W3 settings without a measurable loss barely
  gain.
- gsm8k stayed within ±0.2pp and gpqa_diamond within its noise for every W2 / W3 setting in the
  three-round sweep: only the recall-heavy set shows the loss.
- **The default is W2/0.5**, the balanced point: +2.4% 8k throughput, +9% accept length, simple_qa
  −0.8pp. Accuracy first: W2/0.7 (+1.9%, −0.4pp over 8 runs, standard error 0.15pp). Speed first:
  W2/0.2–0.4 (+3.0..3.7%, −1.2..−2.0pp).
- The same setting's 8k gain moved by about a point between campaigns (W2/0.7: +2.8%, +1.5%, +1.4%):
  quote gains as ranges. Full tables, W1–W4 and the three-round sweep: `references/results.md`.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| FLy on, accept length barely moves | vLLM's default window min(6, gamma - 1) = 4 at gamma 5 can defer only the first draft | the port defaults to min(2, gamma - 1); when copying vLLM settings, set W <= 2 explicitly |
| gsm8k unchanged, simple_qa 2–3pp lower | FLy is lossy and recall shows it first | only a recall set such as simple_qa shows it, and G2 runs one when `goal` says the change may lose accuracy; compare with >= 3 baseline runs |
| a threshold copied from older results behaves differently | before #66593 the entropy was raw nats over the top-k probabilities, not renormalized and not divided by log(k) | use the normalized definition and re-measure |
| FLy shows nothing at 100k/1k | total tok/s there is ~99% prefill | judge FLy on 8k/1k, TPOT mean and accept length |
| every FLy arm shows a ~4% lower TPOT median than the first-round baseline at 8k c128, even arms that do not change acceptance | that baseline compiled a Triton kernel while serving (`... took 1.45 s to compile after serving started`) | grep each server log for that line; repeat baselines later in the run (`../../perf_evaluation.md` §3) |
| `AttributeError ... speculative_use_fly` / `'DsparkVerifyEpilogue' object has no attribute 'fly'` in unit tests | hand-built stubs: a `SimpleNamespace` in `test_spec_registry.py`, `__new__` in `test_dspark_engram_commit_in_graph.py` | add the field to both stubs, as upstream did for block verification (#42297) |
| a merge conflict in `field_order.py` after every upstream update | new `ServerArgs` fields must be appended at the end (`test_builtin_fields_have_explicit_positional_slots`) | keep upstream's lines, re-append the four FLy fields (`../../pr_playbook.md`) |
| `test_no_new_undocumented_flags` fails | the four flags have no row in `docs/docs/advanced_features/server_arguments.mdx` | add the rows; the allowlist may only shrink |
| `python -m sglang.benchmark.serving` fails with `No module named ...` | a directory named `sglang` in the working directory shadows the package | `python3 -P -m ...` |
| `sglang.bench_serving` not found | moved upstream | `sglang.benchmark.serving` |

## Do-no-harm

- CUDA: the flag is rejected at startup and the worker never builds a config, so CUDA paths get
  `fly=None` and only import the new modules.
- ROCm with the flag off: every branch is skipped and `_accept_kernel` compiles without `HAS_FLY`.
- Unsupported combinations fail at startup: another algorithm than DSPARK, block verification, gamma
  above 31, top-k above 16, a window outside [1, gamma), a negative or NaN threshold.

## Limitations and open items

- DSPARK only; EAGLE and DFLASH are not wired. ROCm only.
- Gamma <= 31 and top-k <= 16 are kernel limits vLLM does not have.
- vLLM's synthetic-acceptance cost mode (`rejection_sample_method="synthetic"` with FLy) is not ported.
- No docs rows for the flags (the PR leaves `server_arguments.mdx` unchanged, so the docs coverage test
  fails until they are added).
- Settings were swept on DeepSeek-V4.1-Flash at gamma 5 and T = 1.0 only; greedy was measured only in
  early runs. Accuracy for W2/0.4, W2/0.6 and most W3 settings rests on one run each.
- G3 / G4 not run.

## Porting checklist (another speculative algorithm or platform)

- Every accept implementation the algorithm can take, and for each: does it see logits (with which
  temperature) or only probabilities; where do the draws come from (host, a sampler kernel, Philox
  inside the accept); does it support ragged verify lengths.
- How the vocabulary is sharded across TP ranks in the accept, and whether each rank can compute the
  full-row entropy.
- Gamma and the default window it implies (min(2, gamma - 1) here, min(6, gamma - 1) in vLLM); the
  int32 accept-bit limit (31).
- Graph-captured accept paths: the pre-pass must be graph-safe (no host sync, no data-dependent shapes).
- The platform gate the PR must keep (here `is_hip()`), and the tests that build verify objects by hand.

Then redo WP 4–5 for the new paths and re-run G0, G1, G2 (the sets `goal` asks for) and G5.

## Scripts

- `scripts/test_dspark_fly.py`: FLy unit tests (greedy and sampling decisions on hand-built steps, the
  entropy definition, Triton vs torch, split and shard accepts against the chain rule, settings). Run
  with the branch's sglang on `PYTHONPATH` and one GPU: `python3 -m pytest -q scripts/test_dspark_fly.py`.
- `scripts/test_fly_args.py`: the startup rejections (another algorithm, another platform, block
  verification), CPU only; the resolved window is checked in `test_dspark_fly.py`
  (`TestResolveFlyConfig`).
- `scripts/serve_dspark_fly.sh`: the launch above (`MODEL_PATH`, optional `SGLANG_TREE`, port, GPUs,
  extra flags).
- `scripts/bench_random.sh`: random-prompt throughput at fixed input / output lengths for a list of
  concurrencies, with the cache flushed after warm-up and the accept length read from the server.
- `scripts/lockstep_ab.sh`: the two-lane A/B driver (arms swapped every round, both lanes bench at the
  same time, optional accuracy command per server, compile-stall check).
- `scripts/ab_table.py`: the summary of a `lockstep_ab.sh` run — each arm against the pooled and the
  same-round baseline, with the range of the single runs.

## Sources

- vLLM PR #53987 (FLy), #66593 (normalized entropy, main commit 0f49f3a772); paper arXiv 2511.22972.
- sglang PR #50238, branch `fangyuan/dspark_fly`: e75f1cd1b7 (FLy for DSPARK), 85eaf523f0 (format),
  62afa01bdf (normalized entropy, argument checks), 6b9b6f2b3a (CI test), 4cc1810cc0 (CI threshold),
  ed3831842a / b018c58970 / 39a6484176 (merges of main), 6166b83214 (test stubs), 4e85e85498 (defaults
  W2/0.5).
- Measurements, DeepSeek-V4.1-Flash on gfx950, 2026-11-13 to 2026-11-16: a W/threshold sweep with
  8k/1k, 100k/1k and accuracy for 11 settings; three rounds of 8k/1k for 12 W2/W3 settings; a second
  accuracy run for those 12; a W2/0.7 repeat; three rounds of both 8k/1k and accuracy for the 12 W2/W3
  settings. Tables in `references/results.md`; logs stay with the runs.
