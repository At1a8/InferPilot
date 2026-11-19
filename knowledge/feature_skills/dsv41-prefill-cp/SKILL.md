---
id: dsv41-prefill-cp
title: "Prefill context parallel for DeepSeek-V4.1 on the dsv4 HIP backend (sglang, ROCm gfx950): every rank compresses the whole prompt, the indexer scores only the rank's own rows, the bounded-replay tail gets its own CP layout"
kind: feature_skill
authors: [At1a8]
tags: [prefill context parallel, prefill context parallelism, prefill cp, pcp, context parallel, context parallelism, interleave cp, round robin split, long context prefill, decoder swa bounded replay]
match:
  framework: sglang
  framework_ref: ec93a369b6          # head of PR #43465 (open) with main b085beba3d (2026-11-18) merged in; the recorded runs are on earlier heads of the branch
  feature: prefill_context_parallel
  model_arch: [deepseek_v4]          # DeepSeek-V4.1-Flash (hf model_type deepseek_v41); DeepSeek-V4 is dsv4-prefill-cp
  platforms: [rocm-gfx950]
depends_on: ["sgl-project/sglang#27928"]   # prefill CP for DeepSeek-V4 on the same HIP backend, merged
reference_pr: "sgl-project/sglang#43465"
expects:
  accuracy: "GSM8K (1319) with CP within 0.01 of TP alone"
  long_context: "not claimed: no NIAH run; GSM8K's ~1k-token prompts never reach a long prefix"
  stability: "not claimed, and known to fail: #44730 hit a GPU memory fault at concurrency 64 under mixed-length load (zero-row request groups)"
  perf: "100k input / 500 output, TP8 CP8 vs TP8: >= 1.2x total tok/s with lower TTFT and TPOT at concurrency 8-64"
validation:
  status: validated                  # for the gates claimed: G2, G5 (G0, G1, G3, G4 not claimed)
  last_verified: "2026-11-18"        # the PR head; its numbers were recorded on or before it
  framework_commit: "ec93a369b6 (PR head)"
  model: DeepSeek-V4.1-Flash
  gpu: "gfx950 x8 (TP8 CP8); the CI test gfx950 x4 (TP4 CP4)"
  measured:
    gsm8k: "0.907 at TP8 CP8 vs 0.902 at TP8 (PR description); 0.915 on 1319 questions at TP4 CP4 with DSPARK and bounded replay (the CI test; threshold 0.89, the non-CP test's)"
    perf: "100k/500, TP8 CP8 vs TP8 total tok/s: 82,251 vs 68,627 (c8, 1.20x), 89,493 vs 73,537 (c16, 1.22x), 93,275 vs 76,043 (c32, 1.23x), 95,447 vs 77,483 (c64, 1.23x); mean TTFT 1.20-1.30x lower, mean TPOT 1.17-1.23x lower"
  artifact: "sglang PR #43465; CI test test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_cp_eval_mi35x.py"
role: advisory_prior
---

## When to use

- Turning on, landing or extending prefill context parallelism (`--enable-prefill-cp --cp-strategy
  interleave`) for DeepSeek-V4.1 on ROCm, i.e. sglang PR #43465 on the dsv4 attention backend
  (`DeepseekV4HipRadixBackend`). On main 1f2d76c79d (2026-11-19) the HIP validator still rejects V4.1
  with prefill CP: the PR is not merged.
- Porting HIP prefill CP to another model on the dsv4 HIP backend whose attention has compressed
  (low-ratio) KV sources, an indexer or a bounded-replay tail. Read `dsv4-prefill-cp` first: the
  unified_kv attention under CP (gathered chunk KV, round-robin query metadata, the full SWA window on
  every rank) is its design; this plan adds the low-ratio sources, the indexer rows and the tail.
- Goal: a shorter prefill for long prompts on one node. At 100k-token prompts TTFT and TPOT both drop,
  TPOT because shorter prefills interrupt the running decodes for less time; CP does not make a decode
  step itself faster.

## Source design

Interleave prefill CP (`layers/cp/interleave.py`): the batch's extend tokens are dealt to the CP ranks
by global index (rank r holds tokens r, r + cp, r + 2cp, ...; request-major order is kept), each rank
runs the layers on its share, and attention reads KV gathered from all ranks. The CP group is the
attention TP group, `attn_cp_size = tp_size // attn_dp_size` (`validate_deepseek_v4_cp`), so there is
no CP-size flag; `can_apply` leaves a batch unsharded unless it has at least `cp_size` tokens. Decode
stays plain TP: after a CP prefill every rank must hold the request's whole KV state.

On CUDA, DeepSeek-V4.1 prefill CP was already in main (#40333, #40421, #45534 among others): the CUDA
backend (`DeepseekV4AttnBackend`) has its own `_forward_low_ratio_sources_cp` and the late-layer tail
layout `_late_layer_tail_cp_layout`, and `models/deepseek_v4.py` scatters the tail rows back before the
CP gather (`_needs_cp_tail_scatter`, CUDA only). The HIP backend borrowed the CUDA low-ratio methods
unbound, and `validate_deepseek_v4_cp` rejected DeepSeek-V4.1 and decoder SWA bounded replay on HIP.

## Facts about the target

| Field | Value |
|---|---|
| model | DeepSeek-V4.1-Flash, hf `model_type` `deepseek_v41`: compressed KV sources of ratio 1 and 2, an FP4 indexer (`index_topk` 512), a 128-token sliding window (details in `hisparse-v2`) |
| attention backend | `dsv4` for prefill and decode (required on HIP) → `DeepseekV4HipRadixBackend` |
| bounded replay | `--enable-decoder-swa-bounded-replay`: the late layers re-run only each request's tail (`LateLayerTail`) |
| speculative decoding | DSPARK block drafter, block size 5, with aux captures from the target |
| linears | FP8 through aiter on gfx950 (`--fp8-gemm-backend aiter`); `q_lora` may be an FP8 activation wrapper |
| vision | a vision tower that the CP runner does not drive: CP runs text only |

## Assumption table (CUDA V4.1 prefill CP → HIP dsv4 backend)

| Source assumption | Holds on HIP? | Consequence |
|---|---|---|
| The platform validator admits V4.1 | No: V4.1 and bounded replay were rejected on HIP | admit V4.1 with bounded replay; keep rejecting bounded replay for V4, which has no tail layout |
| The CUDA `_forward_low_ratio_sources_cp` can be borrowed like the other low-ratio methods | No: the HIP compressor store and FP4 indexer differ | a HIP `_forward_low_ratio_sources_cp`: gather the prompt, compress all of it on every rank, score only the local rows |
| The HIP low-ratio indexer sees whole requests | No: a rank holds a strided subset of each request's rows | pass `rows_per_request` (this rank's rows per request) to `low_ratio_index_topk_hip_extend`; drop the assert that rejected it |
| Activations slice with `[:n]` | No on gfx950: `q_lora` may be an FP8 activation wrapper | take rows with `_first_rows(q_lora, n)` |
| The tail can use the extend's CP metadata | No: the tail is another token set | build its own layout with the CUDA `_late_layer_tail_cp_layout` and swap it in for the tail |
| Any non-empty shard is safe for every GEMM | No: aiter's group32 MXFP8 GEMM returns NaN for `wqkv_a` (N=1792, K=5120) at M=5–8 | a per-rank token floor (16 for V4.1) below which the batch runs unsharded |
| Every request has rows on every rank | No: a one-token extend lands on one rank | zero-row request groups reach the FP4 logits kernel: a memory fault (#44730, see Pitfalls) |
| The vision tower runs under CP | No: the CP runner drives the language model directly | text only under CP |

## Design as built (#43465 at ec93a369b6)

1. **Validation** (`arg_groups/deepseek_v4_hook.py`, `validate_deepseek_v4_cp`, HIP branch): the dsv4
   backend for both phases; rejects multiple nodes, `--enable-two-batch-overlap`, and decoder SWA
   bounded replay unless the model is V4.1; for V4.1 sets `SGLANG_CP_MIN_TOKENS_PER_RANK=16` unless the
   user set it. Unchanged for every platform: strategy interleave (zigzag is NPU only), the attention-DP
   and tp ≤ 8 limits, `moe_a2a_backend` in none / deepep / megamoe / mori / ascend_ops.
2. **Token floor** (`environ.py`, `layers/cp/interleave.py`): new `SGLANG_CP_MIN_TOKENS_PER_RANK` (int,
   default 0); `can_apply` returns False when `seq_len // cp_size` is below it (tokens are dealt by
   global index, so the smallest shard holds `seq_len // cp_size`), and the batch runs unsharded on
   every rank. At CP8 with floor 16, batches under 128 extend tokens are not sharded.
3. **Low-ratio sources under CP** (`deepseek_v4_backend_hip_radix.py`, `_forward_low_ratio_sources_cp`,
   replacing the borrowed CUDA method): this rank's rows per request from
   `interleave_rows_per_request(extend_seq_lens_cpu, cp_rank, cp_size)`, or the tail's `local_lens_cpu`;
   compressor: `x_global = cp_materialize_global_token_order(x)[:total]`, then every rank writes the
   whole prompt's compressed KV and indexer K, through the non-CP fused prefill store
   (`_low_ratio_compress`) for the extend and `_low_ratio_compress_torch` for the tail; indexer:
   `_low_ratio_index_topk` on the local rows (`x[:num_local]`, `_first_rows(q_lora, num_local)`, local
   positions) with `rows_per_request`.
4. **Indexer rows** (`dsv4/low_ratio_backend_hip.py`, `low_ratio_index_topk_hip_extend`):
   `rows_per_request` replaces `extend_seq_lens_cpu` as the per-request row counts; `seq_lens_cpu` stays
   global, so each local row still scores all of its visible compressed positions.
5. **Tail layout** (`_build_late_layer_tail_metadata`): under CP, `_late_layer_tail_cp_layout(forward_batch,
   token_indices, tail_lens)` (borrowed from the CUDA backend) gives every rank the tail rows it already
   holds, padded to the largest share. `LateLayerTail` indexes the rank's rows with local token indices
   and positions, and keeps the global ones (`global_token_indices`, `req_global`, `pos_global`) for the
   SWA store and the compressor, which read the gathered tail.
6. **Tail metadata** (`init_forward_metadata_prefill(cp_metadata=...)`): the tail passes its own CP
   metadata; `swa_replay_start` is padded to the padded token count; `apply_cp_reindex(num_tokens,
   local_index=cp_metadata.local_index)`.
7. **Tail entry and exit** (`enter_late_layer_tail` / `exit_late_layer_tail`): save and restore
   `forward_batch.attn_cp_metadata` and the local DP buffer length; inside the tail they are the tail's
   CP metadata and `sum(per_rank_actual_token)`.
8. **Tests**: `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_cp_eval_mi35x.py` (TP4 CP4,
   DSPARK, bounded replay, `--language-model-only`, 8-shot GSM8K completion on 1319 questions,
   threshold 0.89, nightly suite `nightly-test-4-gpu-amd-mi35x-deepseek-v41-flash`);
   `test_dsv4_hip_bcg_metadata.py` follows the new `apply_cp_reindex(local_index=...)` call.

The PR description also lists a tail-output scatter in `models/deepseek_v4.py` and an edit of
`test_platform_prefill_cp_deprecation.py`; neither is in the diff at ec93a369b6 (main has the CUDA-only
`_needs_cp_tail_scatter`, and the deprecation test is gone). Read the diff, not the description.

## Invariants

1. After the low-ratio sources run, every rank holds the whole prompt's compressed KV and indexer K:
   decode is plain TP and reads the rank's own pools. Violation: wrong tokens right after a CP prefill,
   GSM8K drops.
2. The indexer's per-request row counts are this rank's rows, request-major, summing to its local rows;
   `seq_lens_cpu` stays global. Enforced by `interleave_rows_per_request`; violation: a page-table
   assert or a wrong top-k.
3. A batch is sharded only when `seq_len // cp_size >= SGLANG_CP_MIN_TOKENS_PER_RANK` (16 for V4.1 on
   HIP). Violation: NaN from `wqkv_a` on small batches.
4. Inside the tail, `forward_batch.attn_cp_metadata` and the DP buffer length are the tail's; after it,
   the extend's again (`exit_late_layer_tail`). Violation: shape mismatches in the CP gather after the
   tail.
5. With CP off or another model, nothing changes: the floor is 0, `rows_per_request` is None, the tail
   is built without CP.

## Work packages

To land #43465 on a newer main, or take the next step:

1. Rebase onto the target main; resolve against changes to `deepseek_v4_backend_hip_radix.py`,
   `low_ratio_backend_hip.py`, `layers/cp/interleave.py` and the hook. Closes when the CI test and
   `test_dsv4_hip_bcg_metadata.py` pass.
2. Zero-row request groups: skip a group that has no rows on this rank and publish no candidate mask
   for it, as #44730 proposed. Closes with a unit test that puts a one-token request next to a long one
   under CP (#44730's `test_dsv41_low_ratio_zero_row_hip.py` is the model) and fails without the guard,
   and with the G4 load below running clean.
3. G3: NIAH with prompts longer than `--chunked-prefill-size`, so later chunks run CP over a long
   prefix. Closes at the TP8 pass rate, concurrency 1, 8 and 16.
4. G4: mixed-length load at concurrency 64 (long prompts plus short ones and one-token extends, e.g.
   radix-cache hits). Closes with no fault and the failure taxonomy of TP8 in the same pattern.
5. Optional, from #44730: image inputs under CP (`prepare_cp_inputs`), `--enable-cp-decode-attn-tp` on
   gfx950 (narrow the scale copies with the weight).
6. Drop the token floor once aiter's group32 MXFP8 GEMM is right at small M: check `wqkv_a`
   (N=1792, K=5120) at M = 1..16 against a BF16 reference first.

## Launch configuration (as validated)

TP8 CP8 on 8 × gfx950, the PR's launch; the baseline drops `--enable-prefill-cp --cp-strategy interleave`:

```
AITER_BF16_FP8_MOE_BOUND=0 SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
SGLANG_USE_ROCM700A=0 TORCH_BLAS_PREFER_HIPBLASLT=1 \
sglang serve --trust-remote-code --model-path <model_dir>/DeepSeek-V4.1-Flash --tp 8 \
  --enable-prefill-cp --cp-strategy interleave \
  --mem-fraction-static 0.85 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --enforce-shared-experts-fusion --fp8-gemm-backend aiter --enable-decoder-swa-bounded-replay \
  --cuda-graph-max-bs-decode 128 --max-running-requests 128 --cuda-graph-backend-prefill disabled \
  --chunked-prefill-size 8192 --reasoning-parser auto --tool-call-parser auto \
  --json-model-override-args '{"vision_n_layers":0}'
```

- The CP size is `tp / attention DP`, not a flag.
- `--cuda-graph-backend-prefill disabled`: CP on HIP does not support the breakable prefill graph.
- Text only: `{"vision_n_layers":0}` here, `--language-model-only` in the CI test.
- The CI test: TP4, `--mem-fraction-static 0.8`, `AITER_FLYDSL_FORCE_REDUCE=1` (a repeatable FlyDSL MoE
  down-projection), `--model-loader-extra-config '{"enable_multithread_load": true}'`,
  `--watchdog-timeout 1200`.
- Engagement: the startup warning `Enable Context Parallel for DeepSeekV4, strategy=interleave, ...
  attn_cp_size=8`. A batch under the floor runs unsharded without a log line.

## Validation recipe

| Gate | Command / setup | Expected |
|---|---|---|
| G0 | the branch with CP off (TP8) against main: GSM8K and the 100k/500 point | not recorded; equal within noise |
| G1 | `test_dsv4_hip_bcg_metadata.py`; the zero-row unit test of WP 2 | pass; the new test must fail without the guard |
| G2 | GSM8K, 1319 questions, TP8 and TP8 CP8 (`scripts/eval_accuracy.sh`), and the CI test | PR: 0.902 → 0.907; CI 0.915 (>= 0.89) |
| G3 | NIAH (`../hisparse-v2/scripts/niah_test.py`), single and 8 needles at 30k and 100k tokens (several 8192-token chunks), concurrency 1 / 8 / 16, both arms | not run |
| G4 | the mixed-length load of WP 4, and the same pattern on TP8 | not run; #44730 faulted here |
| G5 | `INPUT_LEN=100000 OUTPUT_LEN=500 CONCURRENCY="8 16 32 64" bash ../fly-dspark/scripts/bench_random.sh <port> <out_dir>`, TP8 vs TP8 CP8 (`perf_evaluation.md`) | PR: total tok/s 1.20–1.23x, TTFT 1.20–1.30x, TPOT 1.17–1.23x |

- G2 engages CP (a batch of ~1k-token prompts is far above 8 × 16 tokens) but only with short
  prefixes; long prompts, where compressed positions and the tail run at length, are G3's.
- `acceptance.md` asks context parallelism for a length sweep up to the longest supported length; the
  PR measured 100k only.

## Pitfalls

- NaN on small batches under CP → aiter's group32 MXFP8 GEMM tile returns NaN for `wqkv_a` (N=1792,
  K=5120) at M=5–8, and a CP8 shard of a small batch has that many rows → `SGLANG_CP_MIN_TOKENS_PER_RANK`
  (16, set by the V4.1 HIP validator): a batch whose smallest shard is below it runs unsharded. Remove
  it once the GEMM is fixed (TODO in the hook).
- `q_lora[:n]` fails on gfx950 → `q_lora` can be an FP8 activation wrapper → `_first_rows(q_lora, n)`.
- GPU memory fault at concurrency 64 with mixed-length prompts → a request group with no rows on this
  rank still launches the FP4 prefill logits kernel, which reads a null page table at 0 rows (kernel
  side ROCm/aiter#6562) → skip zero-row groups (#44730, against head c55ba0cd3). `low_ratio_backend_hip.py`
  at ec93a369b6 has no guard for zero local rows.
- Wrong tokens with bounded replay under CP → the tail is another token set and needs its own CP
  layout, metadata and DP buffer length → Design 5–7.
- Image requests wrong under CP → the CP runner drives the language model directly, so image spans
  never reach the embeddings → text only, or #44730's `prepare_cp_inputs`.
- `--enable-cp-decode-attn-tp` breaks V4.1 on gfx950 → the scale copies made at load time
  (`weight_scale_mx_e8m0`, `weight_scale_inv_mx`, `weight_bf16`) are not narrowed with the weight, and
  the MXFP8-native route cannot be narrowed at all → #44730's slicing, with `--fp8-gemm-backend aiter`
  or `triton`.

## Do-no-harm

- CP off: `validate_deepseek_v4_cp` returns before the HIP checks; `rows_per_request` stays None; the
  tail is built without CP; the floor stays 0 for every other model and platform.
- DeepSeek-V4 on HIP: bounded replay with CP is still rejected and the floor is not set.
- NVIDIA does not use the HIP backend; interleave's floor is 0 unless set.
- Startup rejections on HIP: an attention backend other than dsv4 in either phase, multiple nodes,
  two-batch overlap (main's CP + TBO test is skipped on HIP for that reason), V4 with bounded replay.

## Limitations and open items

- #43465 is open without a review (2026-11-19); main rejects V4.1 prefill CP on HIP until it merges.
- Zero-row request groups (Pitfalls); G3 and G4 not run.
- Vision under CP and `--enable-cp-decode-attn-tp` on gfx950: #44730 was closed unmerged; its fixes
  can be folded into #43465.
- The 16-token floor is a work-around: batches under 16 × cp tokens run unsharded.
- No breakable prefill graph, multi-node CP or two-batch overlap with CP on HIP.
- Accuracy on GSM8K only; performance at 100k/500 only, no length sweep.

## Porting checklist (another model, or a newer main)

- Which parts of the layer's state must be whole on every rank after prefill (compressed KV, indexer
  K, sliding-window rings, a replay tail): decode is plain TP.
- Which tensors are global and which local under CP: here the compressed KV and indexer K are global
  (gathered), the queries and the indexer's rows local.
- The GEMMs that run at M = batch tokens / cp, and whether their kernels are correct at small M on the
  target (the floor exists for one such case).
- Activation wrappers (FP8 on gfx950) that do not slice like tensors.
- Speculative aux captures and the layout the CP runner gathers them from.
- The validator's gates for the model and platform, and the attention backend they require.

Then redo WP 1–4 and run G0–G5.

## Scripts

None of its own. G2: `scripts/eval_accuracy.sh` and the CI test above; G3:
`../hisparse-v2/scripts/niah_test.py`; G5: `../fly-dspark/scripts/bench_random.sh`.

## Sources

- sglang PR #43465 (open), head ec93a369b6 on main b085beba3d: description (accuracy and 100k/500
  tables), diff, CI test `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_cp_eval_mi35x.py`.
- sglang PR #27928: prefill CP for DeepSeek-V4 on the same backend (`dsv4-prefill-cp`).
- sglang PR #44730 (closed unmerged): zero-row groups, `--enable-cp-decode-attn-tp`, image inputs
  under CP; kernel issue ROCm/aiter#6562.
- CUDA V4.1 prefill CP in main: sglang #40333, #40421, #45534.
- main 1f2d76c79d (2026-11-19): `validate_deepseek_v4_cp` still rejects V4.1 on HIP;
  `_needs_cp_tail_scatter` is CUDA only.
