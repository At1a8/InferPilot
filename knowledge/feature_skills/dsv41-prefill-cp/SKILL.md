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
  stability: "not claimed, and known to fail without Fix 1: a request group with no rows on a rank faults the FP4 prefill indexer kernel (#44730); neither main c86973c11c nor aiter (ROCm/aiter#6562, open) guards it"
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
  (`DeepseekV4HipRadixBackend`). The PR is not merged: main (c86973c11c, 2026-11-19) still rejects V4.1
  with prefill CP on HIP. "State on main" says what main already has for it, "Fixes for the open
  problems" what #43465 still needs and how.
- Porting HIP prefill CP to another model on the dsv4 HIP backend whose attention has compressed
  (low-ratio) KV sources, an indexer or a bounded-replay tail. Read `dsv4-prefill-cp` first: the
  unified_kv attention under CP (KV gathered in global token order, round-robin query metadata, the
  full SWA window on every rank) is its design; this plan adds the low-ratio sources, the indexer
  rows and the tail.
- Goal: a shorter prefill for long prompts on one node. At 100k-token prompts TTFT and TPOT both drop,
  TPOT because shorter prefills interrupt the running decodes for less time; CP does not make a decode
  step itself faster. `--enable-cp-decode-attn-tp` does, and on V4.1 it needs Fix 2.

## Source design

Interleave prefill CP (`layers/cp/interleave.py`): the batch's extend tokens are dealt to the CP ranks
by global index (rank r holds tokens r, r + cp, r + 2cp, ...; request-major order is kept), each rank
runs the layers on its share, and attention reads KV gathered from all ranks. The CP group is the
attention TP group, `attn_cp_size = tp_size // attn_dp_size` (`validate_deepseek_v4_cp`), so there is
no CP-size flag; `can_apply` leaves a batch unsharded unless it has at least `cp_size` tokens. Decode
stays plain TP: after a CP prefill every rank must hold the request's whole KV state.

DeepSeek-V4.1 prefill CP runs on CUDA in main (#40333, #40421, #45534 among others). The CUDA backend
(`DeepseekV4AttnBackend`) has its own `_forward_low_ratio_sources_cp` and the late-layer tail layout
`_late_layer_tail_cp_layout`. The shared pieces are platform-neutral: `LateLayerTail` carries the
tail's CP layout (`cp_metadata`, `local_lens_cpu`, `pad_rows`, `global_token_indices`, `req_global`,
`pos_global`), and with DSPARK `DeepseekV4ForCausalLM.postprocess_cp_output` (#47406, #50039) gathers
the compact tail outputs with `tail.cp_metadata` and orders them by `tail.global_token_indices`; the
full-shard scatter of #45534 (`_needs_cp_tail_scatter`) is CUDA only. The HIP backend borrowed the
CUDA low-ratio methods unbound, and `validate_deepseek_v4_cp` rejects DeepSeek-V4.1 and decoder SWA
bounded replay on HIP.

## Facts about the target

| Field | Value |
|---|---|
| model | DeepSeek-V4.1-Flash, hf `model_type` `deepseek_v41`, served by `DeepseekV4ForCausalLM`: compressed KV sources of ratio 1 and 2, an FP4 indexer (`index_topk` 512), a 128-token sliding window (details in `hisparse-v2`) |
| attention backend | `dsv4` for prefill and decode (required on HIP) → `DeepseekV4HipRadixBackend` |
| bounded replay | `--enable-decoder-swa-bounded-replay`: the late layers re-run only each request's tail (`LateLayerTail`) |
| speculative decoding | DSPARK block drafter, block size 5, with aux captures from the target |
| linears | FP8 through aiter on gfx950 (`--fp8-gemm-backend aiter`); `q_lora` may be an FP8 activation wrapper; the block-FP8 linears also keep scale copies made at load time (`weight_scale_inv_mx`, `weight_scale_mx_e8m0`, `weight_bf16`) |
| vision | a vision tower; with it built, the late layers route experts by token id (`late_layer_vl_routing`) |

## Assumption table (CUDA V4.1 prefill CP → HIP dsv4 backend)

| Source assumption | Holds on HIP? | Consequence |
|---|---|---|
| The platform validator admits V4.1 | No: V4.1 and bounded replay are rejected on HIP | admit V4.1 with bounded replay; keep rejecting bounded replay for V4, which has no tail layout |
| The CUDA `_forward_low_ratio_sources_cp` can be borrowed like the other low-ratio methods | No: the HIP compressor store and FP4 indexer differ | a HIP `_forward_low_ratio_sources_cp`: gather the prompt, compress all of it on every rank, score only the local rows |
| The HIP low-ratio indexer sees whole requests | No: a rank holds a strided subset of each request's rows | pass `rows_per_request` (this rank's rows per request) to `low_ratio_index_topk_hip_extend`; drop the assert that rejected it |
| Activations slice with `[:n]` | No on gfx950: `q_lora` may be an FP8 activation wrapper | take rows with `_first_rows(q_lora, n)` |
| The tail can use the extend's CP metadata | No: the tail is another token set | build its own layout with the CUDA `_late_layer_tail_cp_layout`, fill the `LateLayerTail` CP fields, swap them in for the tail |
| Any non-empty shard is safe for every GEMM | No: aiter's group32 MXFP8 GEMM returns NaN for `wqkv_a` (N=1792, K=5120) at M=5–8 | a per-rank token floor (16 for V4.1) below which the batch runs unsharded |
| Every request has rows on every rank | No: a one-token extend lands on one rank | a request group can end up with no rows and still reach the FP4 logits kernel (Fix 1) |
| Decode attention TP narrows only `weight` and `weight_scale(_inv)` | No on gfx950: the scale copies must be narrowed too | Fix 2 |
| The tail's routing ids can be taken with the tail's row indices | No: under CP `input_ids_global` is in rank-major order, the HIP tail rows are local | Fix 3 |

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
   SWA store and the compressor, which read the gathered tail, and for main's `postprocess_cp_output`.
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

## State on main (c86973c11c, 2026-11-19)

| Item | On main | With #43465 |
|---|---|---|
| V4.1 prefill CP on HIP | rejected by `validate_deepseek_v4_cp` | admitted |
| Tail CP layout and output gather | `LateLayerTail` CP fields and `postprocess_cp_output` in place (#47406, #50039); the scatter of #45534 is CUDA only | the HIP tail fills those fields |
| Prefill CUDA graph | `deepseek_v4_hook.py` rejects `--enable-decoder-swa-bounded-replay` together with prefill CP and any prefill graph: `--cuda-graph-backend-prefill disabled` is required | same |
| Zero-row request groups | not guarded: `_request_groups`, the group loop of `low_ratio_index_topk_hip_extend` and `aiter_fp4_paged_mqa_logits` have no zero-row check; unreachable, since every extend request has rows without CP | reachable: Fix 1 |
| Image inputs under CP | since #40333 the CP runner prepares image embeddings and ids on the whole sequence before sharding, and the vision tower is allowed with prefill CP when the MoE has no A2A backend (`_v41_vision_a2a_supported`) | the late layers' routing ids are wrong under bounded replay: Fix 3 |
| `--enable-cp-decode-attn-tp` | allowed for `DeepseekV4ForCausalLM`, so for V4.1; slices AITER B-preshuffled weights (#39950) but not V4.1's gfx950 scale copies | fails until Fix 2 |
| MXFP8 GEMM NaN at small M | no aiter issue or fix found (search of ROCm/aiter, 2026-11-19) | the 16-token floor stays (Fix 4) |

The PR description also lists a tail-output scatter in `models/deepseek_v4.py` and an edit of
`test_platform_prefill_cp_deprecation.py`; neither is in the diff at ec93a369b6 (main gathers the tail
through `postprocess_cp_output`, and the deprecation test is gone). Read the diff, not the description.

## Fixes for the open problems

Fixes 1–3 come from sglang PR #44730 (ChangLi811, closed unmerged), which was validated on MI355X on
top of an earlier head of #43465 (its numbers are with each fix); they can be folded into #43465.

**Fix 1: zero-row request groups (memory fault).** `_request_groups` closes a group once it holds more
than `rows_per_chunk` rows, which only one wide request can do, so a request with no rows on this rank
that follows a wide request starts a group of its own; when no request with rows joins it, the group's
row range is empty and `aiter_fp4_paged_mqa_logits` launches the FlyDSL prefill kernel on an empty page
table, which faults at address 0 (ROCm/aiter#6562); top-k cannot take an empty batch either. In the
group loop of `low_ratio_index_topk_hip_extend` (main's loop is unchanged, so the hunk applies as is):

```python
for req_lo, req_hi, tok_lo, tok_hi in groups:
    group_is_identity = is_identity[req_lo:req_hi]
    if tok_hi == tok_lo:
        if publish is not None:
            publish.extend([None] * (req_hi - req_lo))
        continue
```

Requests with no rows inside a non-empty group are already handled: `_select_topk_extend_hip` publishes
`None` for them and `_consume_candidates_per_request` skips them, so the published list keeps one entry
per request. Tests: `test/registered/unit/layers/attention/test_dsv41_low_ratio_zero_row_hip.py`
(#44730), which fails without the hunk; end to end, a ~600k-token request batched with a one-token
extend under CP4 completes (two rounds, page-8 indexer on).

**Fix 2: `--enable-cp-decode-attn-tp` on gfx950.** `CpDecodeAttnTpContext._get_linear_attrs`
(`layers/cp/cp_decode_attn_tp.py`) narrows `weight`, `weight_scale_inv` and `weight_scale` only. On
main, rebase #44730's hunk around the packed-UE8M0 check main has added since:
- at the top, raise `NotImplementedError` for a linear with `mxfp8_native_ready` set (the weight is
  stored in scaled-MFMA lane order and cannot be narrowed by rows or columns), naming
  `--fp8-gemm-backend aiter` or `triton`;
- add `weight_scale_inv_mx`, `weight_scale_mx_e8m0` and `weight_bf16` to the scale names narrowed on the
  weight's dim (`fp8_hip.process_dense_weights` lays them out over (N, K) like the weight), and keep
  only attributes that are tensors.

Tests: `test/registered/unit/layers/cp/test_cp_decode_attn_tp.py` cases (#44730). End to end (#44730):
GSM8K 0.9606 with TP4 + DSPARK + CP4 + decode attention TP against 0.9644 with TP4 + DSPARK; at 8k/1k
and concurrency 16 output tok/s +17% and TTFT −14%, at 64k/1k within −2..+8%.

**Fix 3: image requests under CP with bounded replay.** Main already prepares image inputs on the whole
sequence (#40333), so #44730's `prepare_cp_inputs` and runner change are not needed. Still needed is
its tail hunk: under CP without MoE A2A, `input_ids_global` holds the whole sequence in rank-major
order (`cp_interleave_input_ids`), the MoE gathers the tail rows rank-major too, but
`_forward_layers_hc_pre_from_prev` takes `tail.rows(input_ids_global)` with the HIP tail's local row
indices, so the late layers' vision routing reads ids of other positions. Where main computes it
(`cp_extend` exists there):

```python
input_ids = tail.rows(input_ids)
if cp_extend and get_moe_a2a_backend().is_none():
    input_ids_global = attn_cp_interleave_gather(input_ids)
elif cp_extend:
    input_ids_global = input_ids
elif not (get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none()):
    input_ids_global = tail.rows(input_ids_global)
```

Test (#44730, vision tower built, bounded replay, no DSPARK): MMMU-Pro, 200 questions, 76.5% at TP8
and 78.0% at TP8 CP8 (a 200-question sample moves a few points between runs); GSM8K 0.9674 and 0.9644.
Text-only launches (`vision_n_layers` 0 or `--language-model-only`) do not route by id in the late
layers and are not affected.

**Fix 4: the small-M MXFP8 GEMM NaN.** The floor is the work-around. Report the case to aiter with the
reproducer (gfx950, group32 MXFP8 GEMM, `wqkv_a` N=1792 K=5120, NaN at M=5–8; none was filed by
2026-11-19); once fixed, check M = 1..16 against a BF16 reference and stop setting
`SGLANG_CP_MIN_TOKENS_PER_RANK` for V4.1 in the hook.

**Fix 5: the PR description.** Drop the `models/deepseek_v4.py` tail scatter and the
`test_platform_prefill_cp_deprecation.py` edit from #43465's description, and say that the tail
outputs go through main's `postprocess_cp_output`.

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
   the extend's again (`exit_late_layer_tail`). With DSPARK, `postprocess_cp_output` needs the tail's
   `cp_metadata` and `global_token_indices` and raises `CP bounded replay output is missing its tail
   layout` without them.
5. No FP4 logits kernel launch on zero rows, and one published candidate entry per request (Fix 1).
6. In the late layers, the routing ids line up with the rows the MoE gathers (Fix 3).
7. With CP off or another model, nothing changes: the floor is 0, `rows_per_request` is None, the tail
   is built without CP.

## Work packages

To land #43465 on a newer main, or take the next step:

1. Rebase onto the target main; resolve against changes to `deepseek_v4_backend_hip_radix.py`,
   `low_ratio_backend_hip.py`, `layers/cp/interleave.py` and the hook. Closes when the CI test and
   `test_dsv4_hip_bcg_metadata.py` pass.
2. Fix 1 with its unit test. Closes with the test failing on the parent and passing with the hunk, and
   the end-to-end case running clean.
3. G3: NIAH with prompts longer than `--chunked-prefill-size`, so later chunks run CP over a long
   prefix. Closes at the TP8 pass rate, concurrency 1, 8 and 16.
4. G4: #44730's case (a ~600k-token request with a one-token extend, CP4, two rounds) and a
   mixed-length load at concurrency 64 (long prompts plus short ones and one-token extends, e.g.
   radix-cache hits). Closes with no fault and the failure taxonomy of TP8 in the same pattern.
5. Fix 2, then measure `--enable-cp-decode-attn-tp` in G2 and G5.
6. Fix 3, with an image set (MMMU-Pro) in G2 on a launch that builds the vision tower.
7. Fix 4 when aiter has a fix; Fix 5 at any time.

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
- `--cuda-graph-backend-prefill disabled` is required: with bounded replay, main rejects prefill CP
  together with any prefill graph.
- Text only (`{"vision_n_layers":0}` here, `--language-model-only` in the CI test) until Fix 3.
- No `--enable-cp-decode-attn-tp` until Fix 2; with it, keep `--fp8-gemm-backend aiter`.
- The CI test: TP4, `--mem-fraction-static 0.8`, `AITER_FLYDSL_FORCE_REDUCE=1` (a repeatable FlyDSL MoE
  down-projection), `--model-loader-extra-config '{"enable_multithread_load": true}'`,
  `--watchdog-timeout 1200`.
- Engagement: the startup warning `Enable Context Parallel for DeepSeekV4, strategy=interleave, ...
  attn_cp_size=8`. A batch under the floor runs unsharded without a log line.

## Validation recipe

| Gate | Command / setup | Expected |
|---|---|---|
| G0 | the branch with CP off (TP8) against main: GSM8K and the 100k/500 point | not recorded; equal within noise |
| G1 | `test_dsv4_hip_bcg_metadata.py`; the unit tests of Fixes 1 and 2 | pass; each new test fails without its fix |
| G2 | GSM8K, 1319 questions, TP8 and TP8 CP8 (`scripts/eval_accuracy.sh`), and the CI test; MMMU-Pro on a vision launch after Fix 3 | PR: 0.902 → 0.907; CI 0.915 (>= 0.89); #44730: MMMU-Pro 76.5% → 78.0% (200 questions) |
| G3 | NIAH (`../hisparse-v2/scripts/niah_test.py`), single and 8 needles at 30k and 100k tokens (several 8192-token chunks), concurrency 1 / 8 / 16, both arms | not run |
| G4 | WP 4's loads, and the same pattern on TP8 | not run on #43465; #44730 faulted without Fix 1 and ran clean with it |
| G5 | `INPUT_LEN=100000 OUTPUT_LEN=500 CONCURRENCY="8 16 32 64" bash ../fly-dspark/scripts/bench_random.sh <port> <out_dir>`, TP8 vs TP8 CP8 (`perf_evaluation.md`); after Fix 2 also CP8 with `--enable-cp-decode-attn-tp` | PR: total tok/s 1.20–1.23x, TTFT 1.20–1.30x, TPOT 1.17–1.23x |

- G2 engages CP (a batch of ~1k-token prompts is far above 8 × 16 tokens) but only with short
  prefixes; long prompts, where compressed positions and the tail run at length, are G3's.
- `acceptance.md` asks context parallelism for a length sweep up to the longest supported length; the
  PR measured 100k only.

## Pitfalls

- NaN on small batches under CP → aiter's group32 MXFP8 GEMM tile returns NaN for `wqkv_a` (N=1792,
  K=5120) at M=5–8, and a CP8 shard of a small batch has that many rows → the 16-token floor (Fix 4).
- `q_lora[:n]` fails on gfx950 → `q_lora` can be an FP8 activation wrapper → `_first_rows(q_lora, n)`.
- GPU memory fault at concurrency 64 with mixed-length prompts → a request group with no rows on this
  rank reaches the FP4 prefill logits kernel → Fix 1.
- Wrong tokens with bounded replay under CP → the tail is another token set and needs its own CP
  layout, metadata and DP buffer length → Design 5–7.
- `Interleave CP gather received an unexpected local token count` with DSPARK and bounded replay → the
  CP runner gathered compact tail outputs with the full chunk's CP metadata → fixed on main by #47406
  and #50039 (`postprocess_cp_output` gathers with `tail.cp_metadata`); the HIP tail must fill those
  fields.
- Image requests wrong under CP → before #40333 the image spans never reached the embeddings; with
  #40333 and bounded replay, the late layers route by ids of other positions → Fix 3.
- `--enable-cp-decode-attn-tp` breaks V4.1 on gfx950 → the gfx950 scale copies are not narrowed with
  the weight, and the MXFP8-native route cannot be narrowed at all → Fix 2.

## Do-no-harm

- CP off: `validate_deepseek_v4_cp` returns before the HIP checks; `rows_per_request` stays None; the
  tail is built without CP; the floor stays 0 for every other model and platform.
- DeepSeek-V4 on HIP: bounded replay with CP is still rejected and the floor is not set.
- NVIDIA does not use the HIP backend; interleave's floor is 0 unless set.
- Fix 1 changes nothing for a group with rows; Fix 2 narrows only attributes that exist on gfx950
  block-FP8 linears, and turns the MXFP8-native route into an error that names the fix (#44730: the
  server does not start); Fix 3 only changes the CP branch of the tail.
- Startup rejections on HIP: an attention backend other than dsv4 in either phase, multiple nodes,
  two-batch overlap (main's CP + TBO test is skipped on HIP), V4 with bounded replay, and with bounded
  replay any prefill graph.

## Limitations and open items

- #43465 is open without a review (2026-11-19); main rejects V4.1 prefill CP on HIP until it merges.
- Fixes 1–3 are not in #43465 yet; Fix 4 waits for aiter.
- G3 and G4 not run on #43465.
- The 16-token floor: batches under 16 × cp tokens run unsharded.
- No prefill graph, multi-node CP or two-batch overlap with CP on HIP.
- Accuracy on GSM8K only; performance at 100k/500 only, no length sweep.

## Porting checklist (another model, or a newer main)

- Which parts of the layer's state must be whole on every rank after prefill (compressed KV, indexer
  K, sliding-window rings, a replay tail): decode is plain TP.
- Which tensors are global and which local under CP: here the compressed KV and indexer K are global
  (gathered), the queries and the indexer's rows local.
- Every kernel that a rank may call with zero rows of a request or of a batch under CP, and whether
  it returns early (the FP4 prefill logits kernel does not).
- The GEMMs that run at M = batch tokens / cp, and whether their kernels are correct at small M on the
  target (the floor exists for one such case).
- Activation wrappers (FP8 on gfx950) that do not slice like tensors, and weight copies that decode
  attention TP must narrow.
- Every place that indexes a global-order tensor with tail or shard row indices (Fix 3).
- Speculative aux captures and the layout `postprocess_cp_output` gathers them in.
- The validator's gates for the model and platform, and the attention backend they require.

Then redo WP 1–4 and run G0–G5.

## Scripts

None of its own. G2: `scripts/eval_accuracy.sh` and the CI test above; G3:
`../hisparse-v2/scripts/niah_test.py`; G5: `../fly-dspark/scripts/bench_random.sh`.

## Sources

- sglang PR #43465 (open), head ec93a369b6 on main b085beba3d: description (accuracy and 100k/500
  tables), diff, CI test `test/registered/amd/accuracy/mi35x/test_deepseek_v41_flash_cp_eval_mi35x.py`.
- sglang PR #27928: prefill CP for DeepSeek-V4 on the same backend (`dsv4-prefill-cp`).
- sglang PR #44730 (closed unmerged; commits 9d4a5d932f for Fixes 1 and 2, 92bdcdb081 for image inputs):
  diffs, unit tests, MI355X accuracy and speed tables; ROCm/aiter#6562, open on 2026-11-19.
- In main: #40333 (V4.1 prefill CP with BCG in PD serving: whole-sequence image inputs, vision with
  CP), #40421, #45534, #47406 and #50039 (the tail gather), #39950 (AITER weight slicing for CP decode
  TP).
- main c86973c11c (2026-11-19): `validate_deepseek_v4_cp`, `low_ratio_backend_hip.py`,
  `kernels/ops/attention/dsv4/fp4_indexer_hip.py`, `layers/cp/cp_decode_attn_tp.py`,
  `layers/cp/utils.py` (`cp_interleave_input_ids`), `EagerRunner._execute_extend_cp`,
  `DeepseekV4ForCausalLM.postprocess_cp_output` and `_forward_layers_hc_pre_from_prev`.
