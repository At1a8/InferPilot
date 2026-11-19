---
id: dsv4-prefill-cp
title: "Prefill context parallel for DeepSeek-V4 unified_kv attention on the dsv4 HIP backend (sglang, ROCm gfx950): KV gathered in global token order, local queries sliced round-robin, the full SWA window in every rank's ring"
kind: feature_skill
authors: [At1a8]
tags: [prefill context parallel, prefill context parallelism, prefill cp, pcp, context parallel, context parallelism, interleave cp, round robin split, long context prefill, unified kv, unified kv attention]
match:
  framework: sglang
  framework_ref: d89bc5b574          # main of 2026-11-18, the AMD nightly that last ran the CP test; the code below as on main 7bb4884d73
  feature: prefill_context_parallel
  model_arch: [deepseek_v4]          # DeepSeek-V4 (hf model_type deepseek_v4), validated on V4-Pro; V4.1 is dsv41-prefill-cp
  platforms: [rocm-gfx950]
depends_on: ["sgl-project/sglang#33532"]   # CP V2 for DeepSeek-V4 on CUDA, whose contract the HIP port #34200 follows
reference_pr: "sgl-project/sglang#27928"   # the first landing (CP V1, removed by #38293); main runs #34200 (CP V2), #39923 (FP8 unified_kv)
expects:
  accuracy: "GSM8K (1319) with CP within 0.01 of TP alone"
  long_context: "not claimed: no NIAH run"
  stability: "not claimed"
  perf: "8K/1K at concurrency 16-48: TTFT 16-31% lower; output tok/s 1-8% lower with CP alone, 10-20% above TP8 with --enable-cp-decode-attn-tp"
validation:
  status: validated                  # for the gates claimed: G2, G5 (G0, G1, G3, G4 not claimed)
  last_verified: "2026-11-18"        # the AMD nightly on main d89bc5b574
  framework_commit: "d89bc5b574 (main, the nightly CP test); the two-arm runs on the head of #34200 (86c7cb1c36), with #39950 for decode attention TP"
  model: "DeepSeek-V4-Pro (FP4 experts in the CI test); DeepSeek-V4-Pro-0813 with DSpark-5 in #34200"
  gpu: "gfx950 x8 (TP8 CP8)"
  measured:
    gsm8k: "#34200, 1319 questions: CP8 0.956 vs TP8 0.948 (V4-Pro, EAGLE, 8-shot), CP8 0.939 vs TP8 0.945 (V4-Pro-0813, DSpark-5, 5-shot); 0.9536 with --enable-cp-decode-attn-tp (#39950); the nightly CP test on main d89bc5b574: 0.954 (threshold 0.92)"
    perf: "#34200, 8K/1K, TP8 CP8 vs TP8: mean TTFT -16..-31% at c16-48 and +26..27% at c1; mean TPOT +2..+23%; output tok/s -15..-19% at c1, -1..-8% at c16-48 (DSpark c32 +4.1%); with --enable-cp-decode-attn-tp, output tok/s +10..+20% vs TP8 at c16-48"
    cp_v1_long_context: "#27928's CP V1, since removed: TTFT 1.33x (8k) to 3.82x (1024k) lower, total tok/s 0.78x (8k, c2) to 2.97x (1024k, c2), ITL 0.74-0.81x"
  artifact: "sglang PRs #27928, #34200, #39923, #39950; AMD nightly run 44343550026 (job nightly-8-gpu-mi35x-deepseek-v4-pro, rocm10)"
role: advisory_prior
---

## When to use

- Running, re-verifying or repairing prefill context parallelism (`--enable-prefill-cp --cp-strategy
  interleave`) for DeepSeek-V4 on ROCm on current main.
- Porting HIP prefill CP to another model or attention path on the dsv4 HIP backend. This plan is the
  unified_kv (Triton) attention part; `dsv41-prefill-cp` adds V4.1's compressed sources, indexer rows
  and bounded-replay tail on top of it.
- Goal: TTFT under load and for long prompts. At 8K/1K and concurrency 16–48 TTFT drops 16–31%, and
  with `--enable-cp-decode-attn-tp` output throughput rises 10–20% over TP8. At concurrency 1 CP costs
  TTFT (+27%) and throughput.

## History on main

| Date | PR | Change |
|---|---|---|
| 2026-06-16 | #27928 | CP V1 for unified_kv on HIP: the KV gather in the model, round-robin query metadata, the full window in every ring |
| 2026-08-17 | #33480 | two-batch overlap with CP on AMD |
| 2026-09-07 | #38293 | HIP prefill CP deprecated: `--enable-prefill-cp` rejected on HIP, the ROCm model-side KV gather (#27928) and the CP + TBO path (#33480) removed, the AMD V4 CP tests disabled |
| 2026-10-03 | #34200 | CP V2 ported to the HIP backend after #33532 (CUDA): HIP prefill CP back, for single-node V4 on the dsv4 backend |
| 2026-10-03 | #39950 | AITER weight slicing for `--enable-cp-decode-attn-tp` |
| 2026-10-07 | #39923 | prefill CP with the FP8 unified_kv layout on gfx950 |

The lessons of #27928 are invariants of the port (Pitfalls); its code and its long-context numbers
are from before the rewrite.

## Source design (CP V2)

#33532's CP V2 for DeepSeek-V4: the queries stay rank-local in interleave order, the padded per-rank
layout is described by `forward_batch.attn_cp_metadata` (`per_rank_actual_token`, `local_index`), and
the KV is gathered in global logical-token order before attention and the cache write. On HIP,
DeepSeek-V4 runs on `DeepseekV4HipRadixBackend` with the unified_kv Triton attention
(`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`, HIP only through `is_unified_kv_triton()`): a
2-source prefill (the chunk's KV plus each request's SWA ring), with the ring written after attention.
Decode stays plain TP and reads each rank's own ring. The CP size is `tp_size // attn_dp_size`, not a
flag.

## Assumption table (CUDA CP V2 → HIP unified_kv)

| Source assumption | Holds on HIP? | Consequence |
|---|---|---|
| The backend builds its metadata for the CP layout | No before #34200: the HIP backend built it for the unsplit layout, giving inconsistent query metadata and cache-write locations | build the unified metadata over the padded global layout, then reindex rank-local (`apply_cp_reindex`) |
| Attention gets the whole chunk's KV | No: each rank computes its own share | gather it in global logical-token order (`cp_materialize_global_token_order`); the FP8 pair as one byte row (#39923) |
| Per-query fields are local | No: `unified.pf_*` (`state_slot`, `chunk_start`, `cu_q`, `final_pos`) cover the global layout | slice them `[cp_rank::cp_size]`; the values stay global |
| A rank's queries sit at positions 0..T-1 | No: interleave gives rank r the global positions r, r + cp, ... | `forward_batch.positions[cp_rank::cp_size]` |
| A rank writes only its own tokens into the ring | No: decode and later chunks read this rank's ring | every rank writes the full gathered window, logical rows only |
| `c4_sparse_raw_indices` exists on the HIP prefill path | No | `init_flashmla_related(is_prefill=True)` |
| Decode attention TP can narrow any RowParallel weight along dim 1 | No: AITER B-preshuffled weights (`wo_b`) are not laid out by logical K | un-shuffle, slice along logical K, re-shuffle and cache (#39950) |

## Design on main (7bb4884d73)

1. **Validation** (`validate_deepseek_v4_cp`, HIP): the dsv4 backend for both phases; rejects multiple
   nodes, V4.1 (until #43465), decoder SWA bounded replay and two-batch overlap; strategy interleave;
   attention DP 1 and tp ≤ 8; `moe_a2a_backend` in none / deepep / megamoe / mori / ascend_ops (mori
   since #35611). Under CP it turns `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` off for V4: the CP-aware
   sparse prefill chunk cache is validated on V4.1 only.
2. **Gather** (`models/deepseek_v4.py`, unified prefill branch): `kv = cp_materialize_global_token_order(kv,
   forward_batch, stream)` (`layers/cp/utils.py`). With the FP8 unified_kv layout
   (`SGLANG_DSV4_UNIFIED_KV_FP8=1`), `_materialize_cp_unified_fp8_kv` packs each rank's K into the FP8
   noPE (inline E8M0 scales) + BF16 RoPE pair, gathers it as one raw uint8 row and splits it back in
   global order (#39923). Decode and target verify under CP read the rows already in the ring and
   gather nothing.
3. **Metadata** (`init_forward_metadata_prefill`): the unified metadata is built over the padded
   global layout, then reindexed rank-local; padding rows carry an inert expansion position. CP
   draft-prefill metadata works without compression fields (#34200).
4. **Attention** (`_forward_unified_kv`): CP when `is_cp_active(forward_batch)`; `state_slot`,
   `chunk_start`, `cu_q`, `final_pos` and the positions are sliced `[attn_cp_rank::attn_cp_size]`; the
   ring write (`store_swa_into_unified`) takes the gathered KV with the full state slots and positions
   over the logical rows and drops the physical padding.
5. **Prefill buffers**: `init_flashmla_related(is_prefill=True)` allocates `c4_sparse_raw_indices`.
6. **Decode attention TP** (`--enable-cp-decode-attn-tp`, `layers/cp/cp_decode_attn_tp.py`, allowed
   for `DeepseekV4ForCausalLM`, its NextN and DSpark drafts, and the GLM-5.x DSA models): decode
   attention runs TP across the CP ranks; AITER B-preshuffled RowParallel weights are sliced along
   logical K (#39950).
7. **CI**: `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` (V4-Pro FP4, TP8 CP8, 5-shot GSM8K
   completion on 1319 questions, score > 0.92) in suite `nightly-test-8-gpu-amd-mi35x-deepseek-v4-pro`;
   `test_deepseek_v4_pro_fp4_cp_tbo.py` stays registered but skipped.

## Invariants

1. Under CP, a rank's attention sees its local queries and the whole chunk's KV in global
   logical-token order. Violation: a shape assert, or wrong outputs (the HIP backend before #34200).
2. A local query's position and per-query metadata are its global ones (round-robin), not 0..T-1.
   Violation: wrong output tokens.
3. After a CP prefill every rank's ring holds the whole window, and no physical padding row is
   written. Violation: accuracy drops in decode and in later chunks.
4. Decode and target verify under CP gather nothing; they read the ring.
5. CP off, or NVIDIA: nothing changes.

## Work packages

Re-verifying on a newer main:

1. Check the hook points of "Design on main" on the target. Closes with the CI test passing.
2. G2 with both arms; G3 with prompts longer than the chunked-prefill size; G5 on a length sweep, with
   and without `--enable-cp-decode-attn-tp`.

Porting to another unified_kv model:

1. Gather the KV in global logical-token order in the model's prepare step; an FP8 pair as one byte
   row (Design 2).
2. Build the attention metadata over the padded global layout, then reindex rank-local (Design 3).
3. Slice the per-query fields and positions round-robin (Design 4).
4. Write the full window into the ring on every rank, logical rows only (Design 4).
5. Allocate the prefill-only buffers the NV path has (Design 5).
6. For decode attention TP, add the architecture to `CP_DECODE_ATTN_TP_SUPPORTED_ARCHS` and slice its
   weight layouts along logical K (Design 6).
7. A CI test like the one above; then G0–G5.

## Launch configuration (main's CI test)

DeepSeek-V4-Pro FP4, TP8 CP8; the baseline drops `--enable-prefill-cp --cp-strategy interleave`:

```
SGLANG_DEFAULT_THINKING=1 SGLANG_DSV4_REASONING_EFFORT=max SGLANG_USE_ROCM700A=0 SGLANG_DP_USE_GATHERV=1 \
SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton AITER_BF16_FP8_MOE_BOUND=0 SGLANG_DSV4_FP4_EXPERTS=true \
sglang serve --model-path <model_dir>/DeepSeek-V4-Pro --trust-remote-code --tp 8 --dp 1 \
  --enable-prefill-cp --cp-strategy interleave \
  --attention-backend dsv4 --disable-radix-cache --page-size 256 --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.1 --chunked-prefill-size 8192 --max-running-requests 256 \
  --disable-shared-experts-fusion --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4
```

- Decode attention TP: add `--enable-cp-decode-attn-tp` (#34200's runs with it also passed
  `--attn-cp-size 8`).
- FP8 unified_kv: `SGLANG_DSV4_UNIFIED_KV_FP8=1` (#39923); the FP8 layout keeps its PD and HiCache
  restrictions.
- `SGLANG_DSV4_FP4_EXPERTS=true` is for the FP4 checkpoint (V4-Pro also detects it).
- #27928's launch of June set about 16 `SGLANG_OPT_*` variables; main's test needs none of them.

## Validation recipe

| Gate | Command / setup | Expected |
|---|---|---|
| G0 | the branch with CP off against its parent | not recorded |
| G1 | #39923's CPU tests (`TestDeepseekV4CPKVStore`, `TestUnifiedFp8QPair`, `TestPlatformPrefillCPPolicy`) and #34200's focused ROCm regressions | passed in the PRs (27 in #34200) |
| G2 | GSM8K, 1319 questions, TP8 and TP8 CP8 (`scripts/eval_accuracy.sh`); the CI test | #34200: 0.956 vs 0.948 (EAGLE), 0.939 vs 0.945 (DSpark-5); nightly 0.954 (> 0.92) |
| G3 | NIAH (`../hisparse-v2/scripts/niah_test.py`), single and 8 needles at 30k, 128k and longer (several 8192-token chunks), concurrency 1 / 8 / 16, both arms | not run |
| G4 | many long prompts at high concurrency, both arms | not run |
| G5 | `INPUT_LEN=8192 OUTPUT_LEN=1024 CONCURRENCY="1 16 32 48" bash ../fly-dspark/scripts/bench_random.sh <port> <out_dir>` for TP8, CP8 and CP8 with `--enable-cp-decode-attn-tp`; a length sweep to the longest supported length at concurrency 2 | the tables below |

#34200, DeepSeek-V4-Pro with EAGLE, 8K/1K (TP8 → CP8):

| Concurrency | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|
| 1 | 153.5 → 130.9 | 317 → 403 | 6.21 → 7.25 |
| 16 | 1,107 → 1,060 | 1,551 → 1,136 | 12.03 → 13.00 |
| 32 | 1,541 → 1,513 | 2,659 → 1,835 | 17.33 → 18.28 |
| 48 | 1,760 → 1,702 | 3,902 → 2,686 | 22.38 → 24.27 |

Output tok/s in a second run (another server config; TP8 / CP8 / CP8 with `--enable-cp-decode-attn-tp`):
219 / 165 / 207 at c1, 1,345 / 1,300 / 1,524 at c16, 1,777 / 1,732 / 1,968 at c32, 1,975 / 1,986 / 2,363
at c48.

#27928's CP V1 before the rewrite, for orientation only (CP / TP8, 1k output):

| Input | Concurrency | Total tok/s | TTFT (ms) | ITL (ms) |
|---|---|---|---|---|
| 8k | 2 | 900.7 / 1,156.4 (0.78x) | 450 / 601 (1.33x) | 19.04 / 14.47 |
| 64k | 64 | 25,945 / 16,907 (1.53x) | 45,822 / 60,363 (1.32x) | 34.02 / 27.71 |
| 1024k | 2 | 18,562 / 6,241 (2.97x) | 59,905 / 228,546 (3.82x) | 30.09 / 23.19 |

## Pitfalls

- Inconsistent query metadata and cache-write locations with CP V2 on HIP → the backend built its
  metadata for the unsplit layout → #34200's padded query-metadata and global cache-write contract.
- Wrong outputs with CP on unified_kv and no error → the unified prefill computed only the local KV
  and never gathered it (in #27928: `if unified:` short-circuited `elif use_cp:`) → gather inside the
  unified branch, on main with `cp_materialize_global_token_order`.
- Wrong output tokens → the local queries' positions taken as the first T instead of round-robin →
  slice `[cp_rank::cp_size]`.
- Accuracy errors after the first chunk and in decode → each rank wrote only its share of the window
  into its ring → the full window on every rank, without the physical padding rows (#34200).
- `c4_sparse_raw_indices` missing on HIP prefill (symptom not recorded) → the NV path allocates it and
  HIP did not → `init_flashmla_related(is_prefill=True)`.
- A wrong K shard of `wo_b` with `--enable-cp-decode-attn-tp` on HIP → AITER B-preshuffled RowParallel
  weights narrowed along the physical dim → #39950.
- HIP prefill CP can vanish in a refactor: #38293 removed it, with #33480's CP + TBO, on 2026-09-07,
  and #34200 brought it back a month later without TBO → run the CP test after every CP refactor, and
  check the TBO test before assuming CP + TBO works.
- AMD nightly failures of the V4-Pro suite from 2026-11-14 to 11-17 were not CP: once the container
  did not start, once every file of the suite, CP or not, lost its server during startup (exit -9,
  after loading stalled) → read the job log before blaming the feature.

## Do-no-harm

- NVIDIA: untouched; the HIP backend's CP code is its own.
- CP off: `_cp_active` is false; metadata and ring writes as without CP.
- Startup rejections on HIP: Design 1.

## Limitations and open items

- No two-batch overlap with CP on HIP: #33480's path was removed by #38293, and the test is skipped.
- Single node; no bounded replay on V4; V4.1 only through #43465.
- G0, G1, G3 and G4 not claimed; the validated launch disables the radix cache.
- Long-context performance (beyond 8K input) is recorded only for CP V1 (#27928), not for the CP V2
  port.
- At concurrency 1, CP costs TTFT (+27%) and output tok/s (−15..−19%).

## Porting checklist (another model or attention path)

- The attention path that serves the model on the target (unified_kv Triton, FlashMLA, another), and
  how it reads the chunk's KV and the history.
- Whether its metadata is built over the padded global layout before the rank-local reindex.
- The per-query fields that need the round-robin slice.
- Every per-rank cache that decode or later chunks read (rings, compressed pools), whether each rank
  fills it completely, and whether padding rows stay out of it.
- The prefill-only buffers the other platform allocates.
- Weight layouts that decode attention TP would narrow (pre-shuffled, packed scales).
- The CP size rule (`tp // dp`), attention DP, node count.

## Scripts

None of its own: the CI test, `scripts/eval_accuracy.sh`, `../hisparse-v2/scripts/niah_test.py`,
`../fly-dspark/scripts/bench_random.sh`.

## Sources

- sglang PR #27928 (merged 2026-06-16, a362ba9da3): CP V1, its GSM8K run and length × concurrency table.
- sglang PRs #33480 (CP + TBO on AMD), #38293 (HIP prefill CP deprecated), #33532 (CP V2 for
  DeepSeek-V4, CUDA), #34200 (CP V2 on the HIP backend: two-arm GSM8K and 8K/1K tables), #39950
  (decode attention TP weight slicing), #39923 (FP8 unified_kv), #35611 (mori).
- AMD nightly "Nightly Test (AMD)", run 44343550026 on main d89bc5b574: `test_deepseek_v4_pro_fp4_cp.py`
  scored 0.954; the failures of the V4-Pro job from 2026-11-14 to 11-17 were not CP (a container
  start, servers killed during startup in every file).
- main 7bb4884d73 (2026-11-19): the code under "Design on main".
