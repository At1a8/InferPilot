---
id: qwen35-decode-cp
title: "Decode context parallel (DCP) for Qwen3.5 on the Triton attention backend (sglang, ROCm gfx950): each rank of a DCP group keeps every dcp-th token's KV, decode gathers the query heads and merges the partial attention by LSE"
kind: feature_skill
authors: [At1a8]
tags: [decode context parallel, decode context parallelism, dcp, context parallel, context parallelism, helix parallelism, long context decode]
match:
  framework: sglang
  framework_ref: b2c8f7a22e          # merge of PR #25090 (2026-06-24); main has moved DCP into layers/dcp/ and changed its GQA KV-head layout since, hence stale
  feature: decode_context_parallel
  model_arch: [qwen3_5]              # Qwen3.5 (hf model_type qwen3_5_moe for the 397B-A17B): its full-attention layers; the Gated DeltaNet layers are untouched
  platforms: [rocm-gfx950]
depends_on: []                      # follows the design of sglang #14982 (flashinfer, closed unmerged), not its code
reference_pr: "sgl-project/sglang#25090"
expects:
  accuracy: "GSM8K (1319) with --dcp-size 2 within noise of --dcp-size 1 and of main"
  long_context: "not claimed: no NIAH run"
  stability: "not claimed"
  perf: "one request at a time, TP8 DCP2 vs TP8: ITL and total tok/s better from 32k context (ITL 113.8 -> 62.3 ms at 1M), worse up to 16k (ITL 7.49 -> 8.56 ms at 1k); TTFT within 10%"
validation:
  status: stale                      # validated at b2c8f7a22e for G0, G2, G5; main moved the code and changed the KV-head layout since
  last_verified: "2026-06-24"        # the merge; the PR's numbers were recorded on or before it
  framework_commit: "b2c8f7a22e (merge of PR #25090)"
  model: "Qwen3.5-397B-A17B-FP8 and amd/Qwen3.5-397B-A17B-MXFP4"
  gpu: "gfx950 x8 (TP8, DCP2)"
  measured:
    g0_feature_off: "--dcp-size 1 on the branch vs main, GSM8K 1319: FP8 0.942 / 0.949 / 0.947 vs 0.946 / 0.947 / 0.950; MXFP4 0.9096 +- 0.006 vs 0.9096 +- 0.002 (5 runs each)"
    gsm8k: "--dcp-size 2: FP8 0.945 / 0.941 / 0.946; MXFP4 0.9102 +- 0.006 (5 runs); the CI test asks > 0.90"
    perf: "FP8, one request, 1k output, TP8 DCP2 vs TP8 total tok/s: 230.6 vs 263.1 (1k), 986.6 vs 1,071.8 (8k), 1,754.7 vs 1,814.5 (16k), 3,018.2 vs 2,912.8 (32k), 4,801.8 vs 4,210.3 (64k), 6,613.8 vs 5,295.5 (128k), 7,467.7 vs 5,568.4 (256k), 6,631.5 vs 4,904.8 (512k), 4,581.5 vs 3,730.4 (1M)"
  artifact: "sglang PR #25090; CI test now test/registered/dcp/test_qwen3p5_triton_dcp.py"
role: advisory_prior
---

## When to use

- Bringing DCP to another model with MHA/GQA full attention on the Triton backend (Qwen3.6 is the
  L1 example in `dispatch.md`), or to another attention backend on ROCm.
- Re-verifying or repairing Qwen3.5 DCP on a newer main: the code has moved and the KV-head layout has
  changed since b2c8f7a22e (see "Where it lives on main").
- Goal: long-context decode. Each rank of a DCP group keeps and reads only the tokens it owns, so a
  decode step at long context reads less KV per rank (ITL 45% lower at 1M), at the price of collectives
  on every full-attention layer, which cost more than they save up to about 16k.

## Source design

sglang #14982 proposed DCP for GQA models on the flashinfer backend (CUDA; closed unmerged), and
#25090 follows its design; vLLM's Helix RFC (#34018) is the same idea combined with tensor parallelism
at larger scale. DCP splits the tokens of the KV cache
among the ranks of a DCP group carved out of each TP group: every rank computes attention for all the
group's query heads over its own tokens, and the partial outputs are merged with their log-sum-exp
(LSE), which gives exactly the softmax over all tokens.

## Facts about the target

| Field | Value |
|---|---|
| layers | 60: 15 full attention (`full_attention_interval` 4) and 45 Gated DeltaNet linear attention (hf config of Qwen3.5-397B-A17B) |
| full attention | 32 query heads, 2 KV heads, head_dim 256 |
| TP8 | 4 query heads per rank; each KV head on 4 consecutive ranks |
| context | 262,144 native; 1,048,576 with YaRN factor 4 (`--json-model-override-args` below) |
| attention backend | `--attention-backend triton` |

## Assumption table (#14982's flashinfer DCP on CUDA → Triton on ROCm)

| Source assumption | Holds on the target? | Consequence |
|---|---|---|
| flashinfer DCP kernels are available | No: ROCm runs the Triton backend here | DCP inside `triton_backend.py`: local KV indices, a head all-gather, an LSE merge |
| The decode kernel returns one LSE per head | No: the Triton decode kernel's `attn_lse` has a KV-split dimension | logsumexp over the splits first, then the cross-rank merge |
| The extend kernel returns its LSE | No | `STORE_LSE` output and `SKIP_PREFIX` / `SKIP_EXTEND` switches in `triton_ops/extend_attention.py` (`extend_attention_fwd_with_lse`) |
| Extend may fall back to dense attention | No at long context: a `[extend_tokens, heads, context_tokens]` score tensor runs out of memory | `_forward_extend_dcp`: current chunk, prefix shard, LSE merge |
| The KV pool writes every token on every rank | No: a rank writes only the tokens it owns | owner mask `positions % dcp == dcp_rank` at local location `out_cache_loc // dcp`, `masked_set_kv_buffer_kernel`, forwarded by `HybridLinearKVPool` (Qwen3.5 is hybrid) |
| Fused KV writes in the model are harmless | No: they bypass the masked write | off when `forward_batch.dcp_kv_mask` is set (`models/utils.py`) |
| The tree cache and the allocator share one page size | No: DCP enlarges the physical pages | paged allocation by the allocator's real page size (`mem_cache/common.py`); the invariant checker counts the slack |
| Every rank of a DCP group holds the same KV heads | Only when tp / kv_heads is a multiple of dcp: for Qwen3.5-397B TP8 with dcp 2 or 4, not TP4 with dcp 4 | as built, use only such layouts; main replicates K/V within each DCP group (#32858) |

## Design as built (#25090 at b2c8f7a22e)

1. **Arguments and groups**: `--dcp-size` / `--decode-context-parallel-size` (`server_args.py`, default
   1; above 1 only on HIP, rejected elsewhere at startup). `distributed/parallel_state.py`: `_DCP` and
   `get_dcp_group()`, each TP group split into consecutive DCP groups of `dcp_size` ranks (`tp % dcp ==
   0` required); the DCP group joins `graph_capture()`, so DCP collectives are captured with the decode
   graphs. `model_runner.py` keeps `dcp_size` / `dcp_rank` and passes `decode_context_parallel_size` to
   model-parallel init.
2. **KV ownership**: `owner(abs_pos) = abs_pos % dcp_size`. A rank writes only its own tokens:
   `ForwardBatch.dcp_kv_mask` (eager: `positions % dcp == dcp_rank`; graphs:
   `DecodeInputBuffers.dcp_kv_mask`), `MHATokenToKVPool.set_kv_buffer(..., dcp_kv_mask)` with
   `masked_set_kv_buffer_kernel`, at the local location `out_cache_loc // dcp_size` (`_kv_cache_loc`,
   `_set_kv_buffer` in the backend).
3. **Memory**: enlarged physical pages under DCP (`model_runner_kv_cache_mixin.py`); allocation by the
   allocator's page size (`mem_cache/common.py`, `allocator/paged.py`, `triton_ops/allocator.py`); the
   physical-page slack in `invariant_checker.py`.
4. **Decode metadata** (`layers/attention/utils.py`): `get_dcp_lens()` gives each request's local KV
   length (with an optional per-request start); `create_triton_kv_indices_for_dcp_triton()` builds the
   rank's `kv_indptr` / `kv_indices` from `req_to_token` (rank r reads positions r, r + dcp, ...);
   `get_num_kv_splits()` uses the local lengths.
5. **Decode**: Q reshaped to `[tokens, local_heads, head_dim]` and all-gathered along the heads in the
   DCP group (`self.num_head = local_heads * dcp`); the unchanged Triton decode kernel over the local KV
   gives a partial output and an LSE per split; logsumexp over the splits; `cp_lse_ag_out_rs()`: LSE
   all-gather, rescale, output all-reduce, slice this rank's heads.
6. **Extend** (`_forward_extend_dcp`, when dcp > 1): (a) the current chunk on the local heads with
   `skip_prefix=True`; (b) the prefix shard with the gathered heads and `skip_extend=True`; (c) the
   prefix partials merged across ranks (`cp_lse_ag_out_rs(..., return_lse=True)`), then prefix and
   current merged with `logaddexp` and `exp(part_lse - final_lse)`. An empty prefix skips (b) and (c).
   Sinks, custom masks and sliding windows are rejected.
7. MLA paths keep their own logic: the change targets MHA/GQA models.
8. **CI**: `test/registered/amd/test_qwen3p5_triton_dcp.py` (TP8 DCP2, 5-shot GSM8K completion on 1319
   questions, score > 0.90).

## Where it lives on main (1f2d76c79d, 2026-11-19)

- `dcp_size` is in `arg_groups/fields/parallel.py` (same flags), with derived `attn_dcp_size` and
  `dcp_enabled`. DCP is no longer HIP only: `parallel_state.py` accepts HIP, CUDA, NPU and XPU.
- Helpers moved to `layers/dcp/` (#29365): `layout.py` (`get_dcp_lens`, `filter_dcp_local_kv_indices`,
  the owner rule generalized to interleave blocks), `comm.py` (`cp_lse_ag_out_rs_mha`, `_mla`,
  `_mla_npu`).
- `triton_backend.py`: `_dcp_lens`, `_dcp_kv_indices`, `_forward_extend_dcp`, which still rejects
  sinks, custom masks and sliding windows.
- `forward_batch_info.py` builds `dcp_kv_mask` on HIP; `memory_pool.py` keeps
  `masked_set_kv_buffer_kernel`.
- K/V replicated within each DCP group for GQA (#32858, in `linear.py`, `qwen3_5.py`,
  `triton_backend.py` and the pool sizing).
- `--enable-kv-cache-sharding` and DCP are mutually exclusive (`arg_groups/kv_shard_hook.py`).
- CI: `test/registered/dcp/test_qwen3p5_triton_dcp.py`, now TP4 DCP4 (the layout #32858 fixed), nightly
  suite `nightly-accuracy-test-8-gpu-amd-mi35x-qwen35-triton-dcp`; `test/registered/dcp/test_dcp_layout_unit.py`.

## Invariants

1. Each token's KV is written by exactly one rank of its DCP group, its owner `pos % dcp`; enforced by
   the masked write. Violation: duplicated or missing KV, garbage after the first decode steps.
2. Rank r reads exactly the positions it owns (`get_dcp_lens`, the local indices). Violation: a wrong
   LSE merge, a quality loss that grows with the context length.
3. All ranks of a DCP group compute with the same K/V heads. Violation: silent corruption (#32858:
   GSM8K 0.880 at TP4 DCP4 against 0.990 at DCP1, 200 questions).
4. The merged output equals full attention over prefix and current chunk. Violation: drift on prompts
   with a prefix (chunked prefill, cache hits).
5. dcp 1: no DCP group, `_kv_cache_loc` returns `out_cache_loc`, non-DCP extend runs with `STORE_LSE`,
   `SKIP_PREFIX` and `SKIP_EXTEND` off.

## Work packages (DCP for another MHA/GQA model)

1. The layout: KV heads against tp and dcp. On a main with #32858, K/V replicate within each group;
   on b2c8f7a22e keep tp / kv_heads a multiple of dcp.
2. The model's write paths: every KV write goes through the pool's masked write (fused writes off
   under DCP, hybrid pools forward the mask).
3. The backend: local KV lengths and indices, the decode head gather and LSE merge, the split extend
   with LSE (in place for Triton; another backend needs its own).
4. Startup rejections for the paths not verified (sinks, sliding window, custom masks).
5. G0 (dcp 1 against main), G2, G3 at long context, G5 over lengths and concurrency.

## Launch configuration (as validated)

The PR's performance launch (Qwen3.5-397B-A17B-FP8, TP8 DCP2; the baseline is `--dcp-size 1`):

```
python3 -m sglang.launch_server --model-path <model_dir>/Qwen3.5-397B-A17B-FP8 --tp-size 8 \
  --attention-backend triton --context-length 1048576 \
  --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}' \
  --dcp-size 2
```

- The CI test adds `--trust-remote-code --disable-radix-cache` and `SGLANG_USE_AITER=1
  HSA_NO_SCRATCH_RECLAIM=1`.
- The client, one request per length: `python3 -m sglang.benchmark.serving --backend sglang
  --dataset-name random --random-input-len <L> --random-output-len 1000 --num-prompts 1
  --random-range-ratio 1` (the PR ran it as `sglang.bench_serving`, since removed).

## Validation recipe

| Gate | Command / setup | Expected |
|---|---|---|
| G0 | `--dcp-size 1` on the branch against main, GSM8K 1319, 3 runs (FP8) / 5 runs (MXFP4) | equal within noise (recorded) |
| G1 | none in the PR; main has `test/registered/dcp/test_dcp_layout_unit.py` and a four-rank GQA test (#32858) | — |
| G2 | GSM8K 1319 with `--dcp-size 2`, same runs (`benchmark/gsm8k/bench_sglang.py` in the PR; `scripts/eval_accuracy.sh` in a run) | equal within noise (recorded) |
| G3 | NIAH (`../hisparse-v2/scripts/niah_test.py --context-tokens ...`) at 64k, 256k and 1M, concurrency 1 / 8 / 16, dcp 1 against dcp 2 | not run |
| G4 | many long prompts at high concurrency, both arms | not run |
| G5 | one request per length from 1k to 1M, 1k output, TP8 against TP8 DCP2 | recorded (frontmatter); no concurrency sweep |

ITL and TTFT from the same runs (TP8 → TP8 DCP2): ITL 7.49 → 8.56 ms (1k), 8.21 → 8.94 (8k),
10.64 → 10.18 (32k), 20.18 → 15.02 (128k), 60.78 → 35.17 (512k), 113.84 → 62.26 (1M); TTFT changes by
−10% (64k: 1,580 → 1,739 ms) to +4% (512k).

## Pitfalls

- Corrupted attention with no error on some TP/DCP layouts (TP4 DCP4 on Qwen3.5: GSM8K 0.880 against
  0.990 at DCP1, on GB300) → the TP-only GQA layout puts different KV heads into one DCP group →
  replicate K/V within each DCP group, with the K/V TP layout from `attn_tp_size / dcp_size` (#32858);
  on b2c8f7a22e use only layouts with tp / kv_heads a multiple of dcp.
- KV written by every rank, or missing → a fused KV write in the model bypasses the masked pool write →
  turn it off when `dcp_kv_mask` is set.
- Out of memory in long-context extend → the dense fallback builds `[extend_tokens, heads,
  context_tokens]` scores → `_forward_extend_dcp`.
- Allocation mismatches → the tree cache's page size differs from the allocator's under DCP → allocate
  by the allocator's page size.
- False invariant-checker reports → logical tokens against physical pages under DCP → count the slack.
- DCP collectives inside captured decode graphs → the DCP group has to be in `graph_capture()`'s stack.
- Sinks, sliding windows, custom masks under DCP (not verified) → rejected explicitly, never run silently.

## Do-no-harm

- `--dcp-size 1` (the default): no DCP group; the regular Triton paths run unchanged; G0 matched main.
- At b2c8f7a22e, dcp above 1 off HIP fails at startup, and `tp % dcp != 0` fails in model-parallel init.
- MLA models keep their own path.

## Limitations and open items

- At b2c8f7a22e: the Triton backend and MHA/GQA only; no sinks, sliding window or custom masks.
- Short contexts lose (total tok/s −12% at 1k); single-request measurements only: no concurrency
  sweep, and TTFT does not improve.
- DCP shards only the full-attention layers' KV; the Gated DeltaNet state is per request and stays as
  under TP.
- The CI test disables the radix cache; G1, G3 and G4 not recorded.
- Stale: re-run G0 and G2 on the target main before reusing the plan.

## Porting checklist (another model or backend)

- KV heads, tp and dcp: does each DCP group sit inside one KV head's replica set, or does the target
  main replicate K/V per group?
- The attention kernels: can decode and extend return an LSE; is there a KV-split dimension.
- Every KV write path in the model (fused rope + store, hybrid pools, FP8 KV quantization).
- Sinks, sliding windows, custom masks or cross attention in the model.
- Page size, allocator, graph capture of the new collectives.
- The context length and rope scaling needed to reach the lengths where DCP pays (32k and up here).

## Scripts

None of its own: the CI test, `scripts/eval_accuracy.sh`, `../hisparse-v2/scripts/niah_test.py`,
`python3 -m sglang.benchmark.serving`.

## Sources

- sglang PR #25090 (merged 2026-06-24, b2c8f7a22e; co-author IzacharyI): description (design, GSM8K
  runs, single-request length sweep), diff, CI test.
- sglang PR #14982 (DCP for GQA with flashinfer, closed unmerged: the design #25090 follows); vLLM RFC
  #34018 (Helix parallelism).
- Later in sglang: #29409 (the test's own nightly job, 2026-07-01), #29365 (helpers into `layers/dcp/`,
  2026-07-03), #32858 (GQA KV-head mapping for DCP, 2026-08-08).
- Hugging Face config of Qwen/Qwen3.5-397B-A17B-FP8 for the head counts and layer types.
- main 1f2d76c79d (2026-11-19): the locations under "Where it lives on main".
