---
id: dsv4-prefill-cp
title: "Prefill context parallel for DeepSeek-V4 unified_kv attention on the dsv4 HIP backend (sglang, ROCm gfx950): gather the chunk's KV, slice the per-query metadata round-robin, write the full SWA window on every rank"
kind: feature_skill
authors: [At1a8]
tags: [prefill context parallel, prefill context parallelism, prefill cp, pcp, context parallel, context parallelism, interleave cp, round robin split, long context prefill, unified kv, unified kv attention]
match:
  framework: sglang
  framework_ref: a362ba9da3          # merge of PR #27928 (2026-06-16); main has renamed and moved the hook points since, hence stale
  feature: prefill_context_parallel
  model_arch: [deepseek_v4]          # DeepSeek-V4 (hf model_type deepseek_v4), validated on V4-Pro; V4.1 is dsv41-prefill-cp
  platforms: [rocm-gfx950]
depends_on: []
reference_pr: "sgl-project/sglang#27928"
expects:
  accuracy: "GSM8K (1319) with CP within 0.01 of TP alone"
  long_context: "not claimed: no NIAH run"
  stability: "not claimed"
  perf: "TTFT lower at every length and load (1.32-3.82x); total tok/s below TP8 for short prompts at low load (0.78x at 8k, concurrency 2), above it as length and load grow (2.97x at 1024k); ITL 0.74-0.81x"
validation:
  status: stale                      # validated at a362ba9da3 for G2 (CP arm only) and G5; the hook points moved since
  last_verified: "2026-06-16"        # the merge; the PR's numbers were recorded on or before it
  framework_commit: "a362ba9da3 (merge of PR #27928)"
  model: "DeepSeek-V4-Pro (the CI test: its FP4 experts)"
  gpu: "gfx950 x8 (TP8 CP8)"
  measured:
    gsm8k: "0.947 on 1319 questions at TP8 CP8 (benchmark/gsm8k/bench_sglang.py); no CP-off arm in the PR; the CI test asks > 0.92"
    perf: "TP8 CP8 vs TP8, 1k output, prompts 4x concurrency, 8k-64k input at concurrency 2-64 and 128k-1024k at 2: TTFT 1.33-1.70x lower at 8k-16k, 1.32-1.85x at 32k-64k, 1.98x (128k) to 3.82x (1024k); total tok/s 0.78x (8k, c2) to 1.53x (64k, c64) and 2.97x (1024k, c2); ITL 0.74-0.81x"
  artifact: "sglang PR #27928; CI test test/registered/amd/test_deepseek_v4_pro_fp4_cp.py"
role: advisory_prior
---

## When to use

- Re-verifying or repairing prefill context parallelism (`--enable-prefill-cp --cp-strategy
  interleave`) for DeepSeek-V4 on ROCm after upstream refactors. The design is in main, but the hook
  points have moved since a362ba9da3: see "Where it lives on main".
- Porting prefill CP to another model or attention path on the dsv4 HIP backend. This plan is the
  unified_kv (Triton) attention part; `dsv41-prefill-cp` adds the compressed sources, the indexer rows
  and the bounded-replay tail on top of it.
- Goal: long-context TTFT on one node (3.82x at 1024k) and long prefills that block running decodes
  for less time. Not for short prompts at low load (total tok/s 0.78x at 8k, concurrency 2); every
  decode step is slower with CP (ITL about 0.77x).

## Source design

The interleave strategy (`--enable-prefill-cp --cp-strategy interleave`) existed for the CUDA DSA
models: the batch's tokens are dealt round-robin to the CP ranks, and on CUDA, DeepSeek-V4 keeps the
BF16 KV of its share for a cross-rank all-gather and writes the FlashMLA cache after the gather (the
`elif use_cp:` branch of `_forward_prepare`). The CP size is `tp_size // dp_size`
(`deepseek_v4_hook.py`), not a flag. On HIP, DeepSeek-V4 runs on `DeepseekV4HipRadixBackend` with the
unified_kv Triton attention (`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`): a 2-source prefill (the
chunk's KV plus each request's SWA ring), with the ring written after attention. Decode stays plain TP
and reads each rank's own ring.

## Assumption table (CUDA DSA CP → HIP unified_kv)

| Source assumption | Holds on HIP? | Consequence |
|---|---|---|
| The model all-gathers the per-rank KV under CP | No: `if unified:` came before `elif use_cp:` in `_forward_prepare`, so the unified prefill computed only the local BF16 KV | gather it inside the unified branch |
| Per-query metadata is built for the local queries | No: `unified.pf_*` (`state_slot`, `chunk_start`, `cu_q`, `final_pos`) are built over the global layout | slice them `[cp_rank::cp_size]` to the local queries; the values stay global |
| A rank's queries sit at the first T positions | No: interleave gives rank r the global positions r, r + cp, ... | `forward_batch.positions[cp_rank::cp_size]` |
| A rank writes only its own tokens' KV | No: decode and later chunks read this rank's ring | write the full gathered window into the ring on every rank, with global positions and state slots |
| `c4_sparse_raw_indices` exists on the HIP prefill path | No | allocate it in `init_flashmla_related(is_prefill=True)`, as the NV path does |

## Design as built (#27928 at a362ba9da3)

1. **Gather** (`models/deepseek_v4.py`, `_forward_prepare`, unified prefill branch): `if use_cp and
   _is_hip: kv = cp_all_gather_rerange_output(kv.contiguous(), cp_size, forward_batch, stream)`: the
   2-source prefill needs the whole chunk's KV, for the extend source and for the ring write.
2. **Local queries** (`deepseek_v4_backend_hip_radix.py`, `_forward_unified_kv`): CP is active when the
   CP size is above 1, the round-robin split is on (`is_dsa_prefill_cp_round_robin_split`), `kv.shape[0]
   == cp_size * T` and `state_slot.shape[0] != T`. Then `state_slot`, `chunk_start`, `cu_q`,
   `final_pos` and the positions are sliced `[cp_rank::cp_size]` for attention; the full versions are
   kept for the ring.
3. **Ring write** (`store_swa_into_unified`): under CP the full gathered `kv` with the full
   `state_slot`, `final_pos` and positions; without CP the local ones, as before.
4. **Prefill buffers** (`init_flashmla_related(is_prefill)`): allocates `c4_sparse_raw_indices` like
   `c4_sparse_page_indices` for prefill.
5. Every change is behind `is_hip` or in the HIP backend; the NVIDIA path never reaches it.
6. **CI**: `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` (DeepSeek-V4-Pro FP4, TP8 CP8, 5-shot
   GSM8K completion on 1319 questions, score > 0.92, nightly suite
   `nightly-amd-8-gpu-mi35x-deepseek-v4-pro`).

## Where it lives on main (1f2d76c79d, 2026-11-19)

- The gather: `models/deepseek_v4.py`, unified prefill branch, `if use_cp: kv =
  cp_materialize_global_token_order(...)` (`layers/cp/utils.py`). The `_is_hip` guard is gone because
  `is_unified_kv_triton()` (`layers/attention/dsv4/unified_kv_gate.py`) is true only on HIP;
  `cp_all_gather_rerange_output` (`layers/utils/cp_utils.py`) has no caller left.
- `_forward_unified_kv`: CP is `is_cp_active(forward_batch)`; the round-robin slice and the full-window
  ring write are still there.
- `init_flashmla_related(is_prefill)` still allocates `c4_sparse_raw_indices`.
- `validate_deepseek_v4_cp` on HIP: the dsv4 backend for both phases; rejects multiple nodes, V4.1
  (until #43465), decoder SWA bounded replay and two-batch overlap. V4 under CP turns
  `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` off: the CP-aware sparse prefill chunk cache is validated on V4.1
  only.
- CI: the test above, and `test_deepseek_v4_pro_fp4_cp_tbo.py`, skipped on HIP (no two-batch overlap
  with CP there).

## Invariants

1. Under CP, a rank's attention sees its T local queries and the whole gathered chunk KV (`kv.shape[0]
   == cp_size * T`). Violation: a shape assert, or wrong outputs.
2. A local query's position and metadata are its global ones (round-robin), not 0..T-1. Violation:
   wrong output tokens.
3. After a CP prefill every rank's SWA ring holds the whole window. Violation: accuracy drops in decode
   and in later chunks.
4. CP off, or NVIDIA: nothing changes.

## Work packages

Re-verifying on a newer main:

1. Find the hook points of "Where it lives on main" on the target and check that each still does what
   Design 1–4 says. Closes with the CI test passing.
2. G2 with both arms, G3 with prompts longer than the chunked-prefill size, G5 on a length sweep.

Porting to another unified_kv model:

1. Gather the chunk's KV in the model's prepare step (Design 1).
2. Slice the per-query metadata and positions round-robin (Design 2).
3. Write the full window into the ring on every rank (Design 3).
4. Allocate the prefill-only buffers the NV path has (Design 4).
5. A CI test like the one above; then G0–G5.

## Launch configuration (as validated)

DeepSeek-V4-Pro, TP8 CP8, the PR's launch; the baseline drops `--enable-prefill-cp --cp-strategy interleave`:

```
SGLANG_DEFAULT_THINKING=1 SGLANG_DSV4_REASONING_EFFORT=max SGLANG_USE_AITER=1 SGLANG_USE_ROCM700A=1 \
SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton \
SGLANG_OPT_DEEPGEMM_HC_PRENORM=false SGLANG_OPT_USE_FUSED_COMPRESS=true SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true \
SGLANG_OPT_FP8_WO_A_GEMM=false SGLANG_OPT_USE_JIT_INDEXER_METADATA=false SGLANG_OPT_USE_TOPK_V2=false \
SGLANG_OPT_USE_AITER_INDEXER=true SGLANG_OPT_USE_TILELANG_INDEXER=false SGLANG_OPT_USE_TILELANG_MHC_PRE=false \
SGLANG_OPT_USE_TILELANG_MHC_POST=false SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false \
SGLANG_ROCM_USE_MULTI_STREAM=false AITER_BF16_FP8_MOE_BOUND=0 SGLANG_EAGER_INPUT_NO_COPY=false \
SGLANG_DSV4_MOE_RS_TO_NEXT_ATTN=false \
sglang serve --model-path <model_dir>/DeepSeek-V4-Pro --trust-remote-code --tp 8 --dp 1 \
  --enable-prefill-cp --cp-strategy interleave \
  --attention-backend dsv4 --disable-radix-cache --page-size 256 --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.1 --chunked-prefill-size 8192 --max-running-requests 256 \
  --disable-shared-experts-fusion --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4
```

- `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` selects the attention this plan covers.
- The CI test adds `SGLANG_DSV4_FP4_EXPERTS=true` for the FP4 checkpoint.
- The environment variables are those of June 2026: check them against the target main's
  `environ.py` before reusing the list.

## Validation recipe

| Gate | Command / setup | Expected |
|---|---|---|
| G0 | the branch with CP off against its parent | not recorded |
| G1 | none in the PR | — |
| G2 | GSM8K, 1319 questions, TP8 and TP8 CP8 (`scripts/eval_accuracy.sh`); the CI test | PR: 0.947 with CP (`bench_sglang.py`), no CP-off arm; CI > 0.92 |
| G3 | NIAH (`../hisparse-v2/scripts/niah_test.py`), single and 8 needles at 30k, 128k and longer (several 8192-token chunks), concurrency 1 / 8 / 16, both arms | not run |
| G4 | many long prompts at high concurrency, both arms | not run |
| G5 | random prompts, 1k output, 8k–64k input at concurrency 2–64 and 128k–1024k at concurrency 2, prompts 4x concurrency, TP8 vs TP8 CP8 | the PR's table; points below |

Points from the PR's table (CP / TP8):

| Input | Concurrency | Total tok/s | TTFT (ms) | ITL (ms) |
|---|---|---|---|---|
| 8k | 2 | 900.7 / 1,156.4 (0.78x) | 450 / 601 (1.33x) | 19.04 / 14.47 |
| 8k | 64 | 12,178 / 11,983 (1.02x) | 7,032 / 11,459 (1.63x) | 33.07 / 25.49 |
| 32k | 8 | 9,405 / 9,351 (1.01x) | 4,032 / 6,956 (1.72x) | 20.87 / 16.03 |
| 64k | 64 | 25,945 / 16,907 (1.53x) | 45,822 / 60,363 (1.32x) | 34.02 / 27.71 |
| 128k | 2 | 9,043 / 8,456 (1.07x) | 5,524 / 10,956 (1.98x) | 20.70 / 15.59 |
| 1024k | 2 | 18,562 / 6,241 (2.97x) | 59,905 / 228,546 (3.82x) | 30.09 / 23.19 |

## Pitfalls

- Wrong outputs with CP on unified_kv and no error → `if unified:` short-circuited `elif use_cp:`, so
  the cross-rank all-gather never ran → gather inside the unified branch.
- Wrong output tokens → the local queries' positions were taken as the first T (sequential) instead of
  round-robin → slice `[cp_rank::cp_size]`.
- Accuracy errors after the first chunk and in decode → each rank wrote only its 1/cp of the window
  into its ring → write the full gathered window on every rank.
- `c4_sparse_raw_indices` missing on HIP prefill (symptom not recorded) → the NV path allocates it and
  HIP did not → `init_flashmla_related(is_prefill=True)`.
- Upstream refactors re-break AMD prefill CP (`docs/design.md`: it had to be re-supported several
  times) → re-run the CP CI test after every refactor of `models/deepseek_v4.py` or the HIP backend;
  that drift is why this skill is stale.

## Do-no-harm

- NVIDIA: untouched (the HIP backend, and `_is_hip` on the model change).
- CP off: `_cp_active` is false; metadata and ring write as before.
- Startup rejections on HIP: as listed under "Where it lives on main".

## Limitations and open items

- No CP-off GSM8K arm in the PR; G0, G1, G3 and G4 not recorded.
- The validated launch disables the radix cache: CP over cached prefixes is not covered.
- Decode is slower with CP (ITL 0.74–0.81x); short prompts at low load lose throughput.
- No two-batch overlap with CP on HIP, no multi-node CP, no bounded replay on V4.
- Stale: re-verify on the target main first (Work packages).

## Porting checklist (another model or attention path)

- The attention path that serves the model on the target (unified_kv Triton, FlashMLA, another), and
  how it reads the chunk's KV and the history.
- The per-query fields built over the global layout that need the round-robin slice.
- Every per-rank cache that decode or later chunks read (rings, compressed pools), and whether each
  rank fills it completely.
- The prefill-only buffers the other platform allocates.
- The CP size rule (`tp // dp`), attention DP, node count.

## Scripts

None of its own: the CI test, `scripts/eval_accuracy.sh`, `../hisparse-v2/scripts/niah_test.py`,
`../fly-dspark/scripts/bench_random.sh`.

## Sources

- sglang PR #27928 (merged 2026-06-16, a362ba9da3): description (GSM8K run, length × concurrency
  table), diff, CI test `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py`.
- main 1f2d76c79d (2026-11-19): the locations under "Where it lives on main".
- `dsv41-prefill-cp` (sglang PR #43465) builds on this plan for DeepSeek-V4.1.
