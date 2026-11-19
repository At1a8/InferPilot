---
id: qwen35-decode-cp
title: "Decode context parallel (DCP) for Qwen3.5 on the Triton attention backend (sglang, ROCm gfx950): each rank of a DCP group keeps every dcp-th token's KV, decode gathers the query heads and merges the partial attention by LSE"
kind: feature_skill
authors: [At1a8]
tags: [decode context parallel, decode context parallelism, dcp, context parallel, context parallelism, helix parallelism, long context decode]
match:
  framework: sglang
  framework_ref: b2c8f7a22e          # merge of PR #25090 (2026-06-24), where G0 / G2 / G5 were recorded; main has changed the GQA layout and moved the code since, hence stale
  feature: decode_context_parallel
  model_arch: [qwen3_5]              # Qwen3.5 (hf model_type qwen3_5_moe for the 397B-A17B): its full-attention layers; the Gated DeltaNet layers are untouched
  platforms: [rocm-gfx950]
depends_on: []                      # follows the design of sglang #14982 (flashinfer, closed unmerged), not its code
reference_pr: "sgl-project/sglang#25090"
expects:
  accuracy: "GSM8K (1319) with DCP within noise of --dcp-size 1 and of main"
  long_context: "not claimed: no NIAH run"
  stability: "not claimed"
  perf: "one request at a time, TP8 DCP2 vs TP8: ITL and total tok/s better from 32k context (ITL 113.8 -> 62.3 ms at 1M), worse up to 16k (ITL 7.49 -> 8.56 ms at 1k); TTFT within 10%"
validation:
  status: stale                      # G0, G2, G5 recorded at b2c8f7a22e (TP8 DCP2); on main only the nightly DCP arm (TP4 DCP4)
  last_verified: "2026-06-24"        # the merge; the PR's numbers were recorded on or before it
  framework_commit: "b2c8f7a22e (merge of PR #25090)"
  model: "Qwen3.5-397B-A17B-FP8 and amd/Qwen3.5-397B-A17B-MXFP4"
  gpu: "gfx950 x8 (TP8, DCP2)"
  measured:
    g0_feature_off: "--dcp-size 1 on the branch vs main, GSM8K 1319: FP8 0.942 / 0.949 / 0.947 vs 0.946 / 0.947 / 0.950; MXFP4 0.9096 +- 0.006 vs 0.9096 +- 0.002 (5 runs each)"
    gsm8k: "--dcp-size 2: FP8 0.945 / 0.941 / 0.946; MXFP4 0.9102 +- 0.006 (5 runs); the CI test asks > 0.90"
    perf: "FP8, one request, 1k output, TP8 DCP2 vs TP8 total tok/s: 230.6 vs 263.1 (1k), 986.6 vs 1,071.8 (8k), 1,754.7 vs 1,814.5 (16k), 3,018.2 vs 2,912.8 (32k), 4,801.8 vs 4,210.3 (64k), 6,613.8 vs 5,295.5 (128k), 7,467.7 vs 5,568.4 (256k), 6,631.5 vs 4,904.8 (512k), 4,581.5 vs 3,730.4 (1M)"
    nightly_on_main: "TP4 DCP4 with the radix cache, GSM8K 0.953 (threshold 0.90): AMD nightly run 44343550026 on main d89bc5b574 (2026-11-18); the job's failures from 2026-11-15 to 11-17 came before any DCP code ran"
  artifact: "sglang PR #25090; CI test test/registered/dcp/test_qwen3p5_triton_dcp.py"
role: advisory_prior
---

## When to use

- Running, re-verifying or repairing Qwen3.5 DCP on current main: "Design on main" is the code as it
  is now; #25090's original is under "As merged in #25090", and what became of its problems under
  "Fixes for #25090's problems".
- Bringing DCP to another model with MHA/GQA full attention on the Triton backend (Qwen3.6 is the L1
  example in `dispatch.md`), or to another attention backend on ROCm (the aiter backend's DCP, #34432,
  covers MLA only).
- Goal: long-context decode. Each rank of a DCP group keeps and reads only the tokens it owns, so a
  decode step at long context reads less KV per rank (ITL 45% lower at 1M), at the price of
  collectives on every full-attention layer, which cost more than they save up to about 16k.

## History on main

| Date | PR | Change |
|---|---|---|
| 2026-06-24 | #25090 | DCP for Qwen3.5 on the Triton backend, HIP only |
| 2026-07-01 | #29409 | its own nightly job |
| 2026-07-03 | #29365 | DCP helpers moved into `layers/dcp/` |
| 2026-07-11 | #30478 | DCP in the runtime parallel context (`get_parallel().attn_dcp_size`) |
| 2026-08-08 | #32858 | K/V replicated within each DCP group for GQA models: TP4 DCP4 was corrupt before |
| 2026-08-10 | #34161 | the Triton DCP prefill picks the KV heads of each rank's Q shard: a regression of #32858 |
| 2026-09-12 | #39165 | `--dcp-comm-backend` resolves to `a2a` on CUDA and ROCm (MLA reductions; the Triton MHA path ignores it) |
| 2026-09-24 | #41155 | the Qwen3.5 nightly runs with the radix cache |
| 2026-11-05 | #34212 | no QuickAllReduce for the DCP group on ROCm (it reserved 4 GiB per GPU at Kimi-K3 TP8/DCP8) |

DCP now also runs on CUDA, NPU and XPU, for MLA and DSA models besides MHA.

## Source design

sglang #14982 proposed DCP for GQA models on the flashinfer backend (CUDA; closed unmerged), and
#25090 follows its design; vLLM's Helix RFC (#34018) is the same idea combined with tensor parallelism
at larger scale. DCP splits the tokens of the KV cache among the ranks of a DCP group carved out of
each TP group: every rank computes attention for all the group's query heads over its own tokens, and
the partial outputs are merged with their log-sum-exp (LSE), which gives exactly the softmax over all
tokens.

## Facts about the target

| Field | Value |
|---|---|
| layers | 60: 15 full attention (`full_attention_interval` 4) and 45 Gated DeltaNet linear attention (hf config of Qwen3.5-397B-A17B) |
| full attention | 32 query heads, 2 KV heads, head_dim 256 |
| K/V layout on main | KV heads shard over `tp // dcp` and replicate within each DCP group (#32858). At TP4 DCP4 every rank holds both KV heads, each rank has 8 query heads, and the prefill gives ranks 0–1 KV head 0 and ranks 2–3 KV head 1 (#34161) |
| context | 262,144 native; 1,048,576 with YaRN factor 4 (`--json-model-override-args` below) |
| attention backend | `--attention-backend triton` |

## Assumption table (#14982's flashinfer DCP on CUDA → Triton on ROCm)

| Source assumption | Holds on the target? | Consequence |
|---|---|---|
| flashinfer DCP kernels are available | No: ROCm runs the Triton backend here | DCP inside `triton_backend.py`: local KV indices, a head all-gather, an LSE merge |
| The decode kernel returns one LSE per head | No: the Triton decode kernel's `attn_lse` has a KV-split dimension | logsumexp over the splits first, then the cross-rank merge |
| The extend kernel returns its LSE | No | `STORE_LSE` output and `SKIP_PREFIX` / `SKIP_EXTEND` switches in the Triton extend kernel (`extend_attention_fwd_with_lse`) |
| Extend may fall back to dense attention | No at long context: a `[extend_tokens, heads, context_tokens]` score tensor runs out of memory | `_forward_extend_dcp`: current chunk, prefix shard, LSE merge |
| The KV pool writes every token on every rank | No: a rank writes only the tokens it owns | owner mask `positions % dcp == dcp_rank`, `masked_set_kv_buffer_kernel`, forwarded by `HybridLinearKVPool` (Qwen3.5 is hybrid) |
| Fused KV writes in the model are harmless | No: they bypass the masked write | off when `forward_batch.dcp_kv_mask` is set |
| The tree cache and the allocator share one page size | No: DCP enlarges the physical pages | paged allocation by the allocator's real page size; the invariant checker counts the slack |
| Every rank of a DCP group holds the same KV heads | On main yes: #32858 replicates K/V within each DCP group, #34161 makes the Triton prefill pick each rank's heads. At b2c8f7a22e only when tp / kv_heads is a multiple of dcp (TP8 with dcp 2 or 4 here, not TP4 with dcp 4) | on main nothing to do; a new backend must pick the heads as #34161 does |

## Design on main (c86973c11c, 2026-11-19)

1. **Arguments and groups**: `dcp_size` (`--dcp-size` / `--decode-context-parallel-size`) in
   `arg_groups/fields/parallel.py`, with derived `attn_dcp_size` and `dcp_enabled`.
   `distributed/parallel_state.py` splits each TP group into consecutive DCP groups (`tp % dcp == 0`),
   accepts HIP, CUDA, NPU and XPU, and gives the DCP group no QuickAllReduce on ROCm (#34212). DCP is
   rejected with `--enable-prefill-cp` (`parallel_hook.py`) and with `--enable-kv-cache-sharding`
   (`kv_shard_hook.py`).
2. **Communication**: `--dcp-comm-backend` resolves to `a2a` on ROCm (#39165; `ag_rs` restores the old
   one), but the Triton MHA path does not read it: decode and the prefix merge call
   `cp_lse_ag_out_rs_mha` (`layers/dcp/comm.py`: LSE all-gather, rescale, FP32 all-reduce, head slice),
   as #25090 did. `a2a` / `fi_a2a` and `--dcp-replicate-q-proj` serve the MLA reductions.
3. **K/V layout** (#32858): `ModelConfig.get_num_kv_heads(tp, dcp)` shards the KV heads over
   `tp // dcp`; `models/qwen3_5.py` builds the K/V projection with `ReplicatedParallelGroup("attn_tp",
   dcp_size)`; the MHA pool and the backend metadata are sized from the local KV heads.
4. **KV ownership**: owner `abs_pos % dcp`; `forward_batch_info.py` builds `dcp_kv_mask = positions %
   attn_dcp_size == attn_dcp_rank` on HIP; `MHATokenToKVPool.set_kv_buffer(..., dcp_kv_mask)` writes
   through `masked_set_kv_buffer_kernel`; `layers/dcp/layout.py` has `get_dcp_lens` and
   `filter_dcp_local_kv_indices`, with the owner rule generalized to interleave blocks.
5. **Decode** (`triton_backend.py`): Q all-gathered along the heads in the DCP group; the Triton decode
   kernel over the rank's KV (`_dcp_kv_indices`, `_dcp_lens`); logsumexp over the KV splits;
   `cp_lse_ag_out_rs_mha`.
6. **Extend** (`_forward_extend_dcp`): (a) the current chunk on the local Q heads with the replicated
   K/V heads that match the rank's Q shard (#34161) and `skip_prefix=True`; (b) the prefix shard with
   the gathered heads and `skip_extend=True`; (c) the prefix partials merged across ranks, then prefix
   and current merged by LSE. An empty prefix skips (b) and (c). Sinks, custom masks and sliding
   windows raise `NotImplementedError`.
7. **CI**: `test/registered/dcp/test_qwen3p5_triton_dcp.py` (TP4 DCP4, Triton, the radix cache on,
   5-shot GSM8K completion on 1319 questions, score > 0.90, suite
   `nightly-accuracy-test-8-gpu-amd-mi35x-qwen35-triton-dcp`); `test/registered/dcp/test_dcp_layout_unit.py`;
   #34161's CPU test of the TP4/DCP4 head mapping.

## As merged in #25090 (b2c8f7a22e)

- `--dcp-size` in `server_args.py`, HIP only (rejected elsewhere at startup); `_DCP` / `get_dcp_group()`
  in `parallel_state.py`, joined to `graph_capture()`.
- Helpers in `layers/attention/utils.py`: `get_dcp_lens()`, `create_triton_kv_indices_for_dcp_triton()`,
  `cp_lse_ag_out_rs()`.
- KV writes at the local location `out_cache_loc // dcp_size` with the owner mask (`_kv_cache_loc`,
  `_set_kv_buffer`); enlarged physical pages (`model_runner_kv_cache_mixin.py`); allocation by the
  allocator's page size (`mem_cache/common.py`); physical-page slack in `invariant_checker.py`; fused KV
  writes off under DCP (`models/utils.py`).
- The TP-only GQA layout: K/V heads followed the TP shard, so a DCP group could mix KV heads.
- CI: `test/registered/amd/test_qwen3p5_triton_dcp.py`, TP8 DCP2 with `--disable-radix-cache`.

## Fixes for #25090's problems

| Problem of #25090 | State on main | Fix |
|---|---|---|
| The TP-only K/V layout corrupts attention when a DCP group spans KV heads (TP4 DCP4) | fixed | #32858, then #34161 |
| The DCP group builds an unused QuickAllReduce on ROCm (4 GiB per GPU at Kimi-K3 TP8/DCP8) | fixed | #34212 |
| CI only without the radix cache | fixed | #41155: the nightly runs with it (0.953) |
| Short contexts slower (total tok/s −12% at 1k, one request) | open | candidates below, unmeasured |
| The startup log names `a2a` for MHA, which never uses it | open | resolve `ag_rs` for MHA DCP |
| Performance recorded for one request at a time | open | G5 with a concurrency sweep |

**GQA layout: fixed on main.** #32858 derives a K/V TP layout from `attn_tp_size / dcp_size`, builds
the K/V projections replicated within each DCP group (no runtime K/V communication), sizes the MHA
pool and the backend metadata from the local KV heads, and adds a four-rank Q-sharding /
KV-replication test. Its GSM8K runs (GB300, Qwen3.5 NVFP4, TP4 DCP4, 200 questions): 0.880 before,
0.980 after, 0.990 at TP4 DCP1. That left the Triton current-chunk prefill passing each local Q shard
together with all the replicated KV heads, so the extend kernel inferred a local 4:1 mapping instead
of the global 16:1. #34161 selects the contiguous KV-head range of each rank's Q shard before the
kernel and adds a CPU test of the TP4/DCP4 mapping. Its GSM8K runs (Qwen3.5-397B-A17B-FP8, TP4 DCP4):
0.772 and 0.751 in two full CI runs before, 0.954 on 1,314 questions on B300 after. On ROCm, main's
nightly at TP4 DCP4 scores 0.953.

**QuickAllReduce: fixed on main.** On ROCm `GroupCoordinator` built a QuickAllReduce communicator for
every group with custom all-reduce, though DCP never dispatches to it (MHA: an all-gather and an FP32
all-reduce; a2a: all-to-all). #34212 adds `use_quick_allreduce` and passes `False` for the DCP group.

**Short contexts: candidates, unmeasured.** Per full-attention layer and decode step, the MHA path
all-gathers Q along the heads, and `cp_lse_ag_out_rs_mha` all-reduces the rescaled FP32 output of all
the group's heads before keeping this rank's slice. Two changes to measure against it at 1k–16k (G5),
each with G2:
1. reduce-scatter along the head dim instead of the all-reduce and slice:
   `cp_group.reduce_scatter_along_dim(out, dim=1)`, the call the MLA reduce makes along dim 0, since
   rank r's heads are contiguous at `[r·h, (r+1)·h)`. It moves about half the bytes of an all-reduce,
   but ROCm's custom all-reduce may still be faster for small decode batches: time both;
2. no Q all-gather: replicate the Q projection within the DCP group, as `--dcp-replicate-q-proj` does
   for MLA with the a2a backend, at `dcp` times the Q-projection work.

**The startup log.** `_dcp_comm_backend_default` resolves `a2a` on ROCm for every model (#39165), and
the Triton MHA path ignores it. Resolve `ag_rs` when the model's DCP attention is MHA, or say in the
flag's help that MHA ignores it.

**Concurrency.** The PR measured one request per length: run G5 at concurrency 1–64 at 32k and longer,
where DCP pays.

## Invariants

1. Each token's KV is written by exactly one rank of its DCP group, its owner `pos % dcp`; enforced by
   the masked write. Violation: duplicated or missing KV, garbage after the first decode steps.
2. Rank r reads exactly the positions it owns (`get_dcp_lens`, the local indices). Violation: a wrong
   LSE merge, a quality loss that grows with the context length.
3. All ranks of a DCP group compute with the same K/V heads, and each rank's Q heads meet their own KV
   head. Enforced on main by the replicated K/V projection (#32858) and the head pick in the Triton
   prefill (#34161). Violation: silent corruption (#32858: GSM8K 0.880 at TP4 DCP4 against 0.990 at
   DCP1, 200 questions; #34161: 0.75–0.77 against 0.954).
4. The merged output equals full attention over prefix and current chunk. Violation: drift on prompts
   with a prefix (chunked prefill, cache hits).
5. dcp 1: no DCP group, and the non-DCP extend runs with `STORE_LSE`, `SKIP_PREFIX` and `SKIP_EXTEND`
   off.

## Work packages (DCP for another MHA/GQA model)

1. The layout: on main, K/V replicate within each DCP group once the model builds its K/V projection
   with the replicated group (as `qwen3_5.py` does); a backend's prefill must then pick the KV heads of
   each rank's Q shard (as #34161 does in Triton).
2. The model's write paths: every KV write goes through the pool's masked write (fused writes off
   under DCP, hybrid pools forward the mask).
3. The backend: local KV lengths and indices, the decode head gather and LSE merge, the split extend
   with LSE (in place for Triton; another backend needs its own).
4. Startup rejections for the paths not verified (sinks, sliding window, custom masks).
5. G0 (dcp 1 against main), G2, G3 at long context, G5 over lengths and concurrency.

## Launch configuration

Main's CI test (Qwen3.5-397B-A17B-FP8, TP4 DCP4, radix cache on):

```
SGLANG_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1 \
sglang serve --model-path <model_dir>/Qwen3.5-397B-A17B-FP8 --trust-remote-code --tp 4 --dcp-size 4 \
  --attention-backend triton --context-length 1048576 \
  --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}'
```

- The PR's performance runs: `--tp-size 8 --dcp-size 2` with the same backend, context length and
  rope override; the baseline is `--dcp-size 1`.
- The startup log names a DCP communication backend (`a2a` by default on ROCm); the Triton MHA path
  uses the all-gather and all-reduce regardless.
- The client, one request per length: `python3 -m sglang.benchmark.serving --backend sglang
  --dataset-name random --random-input-len <L> --random-output-len 1000 --num-prompts 1
  --random-range-ratio 1` (the PR ran it as `sglang.bench_serving`, since removed).

## Validation recipe

| Gate | Command / setup | Expected |
|---|---|---|
| G0 | `--dcp-size 1` on the branch against main, GSM8K 1319, 3 runs (FP8) / 5 runs (MXFP4) | equal within noise (recorded at b2c8f7a22e) |
| G1 | `test/registered/dcp/test_dcp_layout_unit.py`; #32858's four-rank GQA test; #34161's TP4/DCP4 mapping test | pass on main |
| G2 | GSM8K 1319 with DCP against `--dcp-size 1` (`scripts/eval_accuracy.sh`); the CI test | b2c8f7a22e: equal within noise at DCP2; main: the CI test 0.953 at TP4 DCP4 (> 0.90) |
| G3 | NIAH (`../hisparse-v2/scripts/niah_test.py --context-tokens ...`) at 64k, 256k and 1M, concurrency 1 / 8 / 16, dcp 1 against dcp 2 and 4 | not run |
| G4 | many long prompts at high concurrency, both arms | not run |
| G5 | one request per length from 1k to 1M, 1k output, TP8 against TP8 DCP2; then a concurrency sweep | recorded at b2c8f7a22e (frontmatter); no concurrency sweep |

ITL and TTFT from the PR's runs (TP8 → TP8 DCP2): ITL 7.49 → 8.56 ms (1k), 8.21 → 8.94 (8k),
10.64 → 10.18 (32k), 20.18 → 15.02 (128k), 60.78 → 35.17 (512k), 113.84 → 62.26 (1M); TTFT changes by
−10% (64k: 1,580 → 1,739 ms) to +4% (512k).

## Pitfalls

- Corrupted attention with no error at TP4 DCP4 (GSM8K 0.880 against 0.990 at DCP1, GB300) → the
  TP-only GQA layout put different KV heads into one DCP group → K/V replicated within each DCP group
  (#32858, on main since 2026-08-08). On a commit before it, keep tp / kv_heads a multiple of dcp.
- Every fresh prompt's prefill corrupted after #32858 (GSM8K 0.75–0.77 at TP4 DCP4) → the Triton
  current-chunk prefill gave each local Q shard all the replicated KV heads, so the kernel inferred a
  local 4:1 mapping instead of the global 16:1 → pick the KV heads of the rank's Q shard (#34161, on
  main since 2026-08-10).
- 4 GiB per GPU gone on ROCm with DCP (Kimi-K3 TP8/DCP8) → the DCP group built a QuickAllReduce
  communicator that DCP never dispatches to → off for the DCP group (#34212, on main since 2026-11-05).
- KV written by every rank, or missing → a fused KV write in the model bypasses the masked pool write →
  turn it off when `dcp_kv_mask` is set.
- Out of memory in long-context extend → the dense fallback builds `[extend_tokens, heads,
  context_tokens]` scores → `_forward_extend_dcp`.
- Allocation mismatches → the tree cache's page size differs from the allocator's under DCP → allocate
  by the allocator's page size.
- False invariant-checker reports → logical tokens against physical pages under DCP → count the slack.
- DCP collectives inside captured decode graphs → the DCP group has to be in `graph_capture()`'s stack.
- Sinks, sliding windows, custom masks under DCP (not verified) → rejected explicitly, never run
  silently.
- AMD nightly failures of the DCP job from 2026-11-15 to 11-17 → each came before any DCP code ran:
  twice the job stopped at the VRAM cleanup (zombie KFD contexts), twice the server found about 80 of
  288 GB free when weight loading began (278 GB in the passing run) and ran out of memory building
  the MoE weights → not DCP; read the job log first.

## Do-no-harm

- `--dcp-size 1` (the default): no DCP group; the regular Triton paths run unchanged; G0 matched main
  at b2c8f7a22e.
- `tp % dcp != 0` fails at model-parallel init; DCP with prefill CP or KV-cache sharding fails at
  startup.
- MLA models keep their own path.

## Limitations and open items

- The Triton MHA path has no `a2a` reduction: it all-gathers the LSE and all-reduces the output.
- No sinks, sliding window or custom masks under DCP on Triton.
- Short contexts lose (total tok/s −12% at 1k): the candidates under "Fixes for #25090's problems" are
  unmeasured. Performance recorded only for single requests at b2c8f7a22e: no concurrency sweep, no
  TTFT gain, none on main.
- DCP shards only the full-attention layers' KV; the Gated DeltaNet state is per request and stays as
  under TP.
- G1, G3 and G4 not recorded with the PR; on main, accuracy only from the nightly's DCP arm.
- Stale: re-run G0 and G2 with both arms on the target main before reusing the numbers.

## Porting checklist (another model or backend)

- KV heads, tp and dcp, and how the model builds its K/V projection (replicated per DCP group, as on
  main for Qwen3.5).
- In the backend's prefill, which KV heads each rank's Q shard meets.
- The attention kernels: can decode and extend return an LSE; is there a KV-split dimension.
- Every KV write path in the model (fused rope + store, hybrid pools, FP8 KV quantization).
- Sinks, sliding windows, custom masks or cross attention in the model.
- Page size, allocator, graph capture of the new collectives, the communicators the DCP group builds.
- The context length and rope scaling needed to reach the lengths where DCP pays (32k and up here).

## Scripts

None of its own: the CI test, `scripts/eval_accuracy.sh`, `../hisparse-v2/scripts/niah_test.py`,
`python3 -m sglang.benchmark.serving`.

## Sources

- sglang PR #25090 (merged 2026-06-24, b2c8f7a22e; co-author IzacharyI): description (design, GSM8K
  runs, single-request length sweep), diff, CI test.
- sglang PR #14982 (DCP for GQA with flashinfer, closed unmerged: the design #25090 follows); vLLM RFC
  #34018 (Helix parallelism).
- Later in sglang: #29409, #29365, #30478, #32858 (GQA KV-head mapping), #34161 (the Triton prefill head
  pick), #39165 (comm backend default), #41155 (radix cache in the nightly), #34212 (QuickAllReduce),
  #34432 (aiter backend DCP, MLA).
- AMD nightly "Nightly Test (AMD)", run 44343550026 on main d89bc5b574: `test_qwen3p5_triton_dcp.py`
  scored 0.953; the DCP job's failures from 2026-11-15 to 11-17 came before any DCP code ran.
- Hugging Face config of Qwen/Qwen3.5-397B-A17B-FP8 for the head counts and layer types.
- main c86973c11c (2026-11-19): the code under "Design on main".
