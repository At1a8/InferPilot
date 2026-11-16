#!/bin/bash
# DeepSeek-V4.1-Flash + DSPARK on 4 gfx950 GPUs with the launch configuration FLy was
# validated with. Extra arguments go to `sglang serve` unchanged, so the same script serves
# both arms of an A/B: no FLy flags for the baseline, the FLy flags for the feature arm.
#
# Usage: MODEL_PATH=<dir> [SGLANG_TREE=<sglang checkout>] bash serve_dspark_fly.sh <port> <hip_visible_devices> [extra args...]
#   baseline: MODEL_PATH=... bash serve_dspark_fly.sh 30010 0,1,2,3
#   FLy:      MODEL_PATH=... bash serve_dspark_fly.sh 30011 4,5,6,7 \
#                 --speculative-use-fly --speculative-fly-window-size 2 --speculative-fly-entropy-threshold 0.7
# SGLANG_TREE puts <tree>/python first on PYTHONPATH, so the server runs that checkout
# (e.g. a worktree pinned at the commit under test) instead of the installed one.
PORT=${1:?usage: bash serve_dspark_fly.sh <port> <hip_visible_devices> [extra args...]}
DEVICES=${2:?usage: bash serve_dspark_fly.sh <port> <hip_visible_devices> [extra args...]}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint directory}
shift 2

if [ -n "${SGLANG_TREE:-}" ]; then
    export PYTHONPATH=${SGLANG_TREE}/python${PYTHONPATH:+:${PYTHONPATH}}
fi
export HIP_VISIBLE_DEVICES=${DEVICES}

SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
SGLANG_USE_ROCM700A=0 \
TORCH_BLAS_PREFER_HIPBLASLT=1 \
SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=true \
SGLANG_HIP_FP4_INDEXER_ROWGROUP=1 \
SGLANG_HIP_DSV41_INDEX_K_PAGE8=1 \
sglang serve \
  --trust-remote-code \
  --model-path "${MODEL_PATH}" \
  --tp 4 \
  --ep-size 4 \
  --mem-fraction-static 0.85 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enforce-shared-experts-fusion \
  --fp8-gemm-backend aiter \
  --enable-decoder-swa-bounded-replay \
  --cuda-graph-max-bs-decode 128 \
  --max-running-requests 128 \
  --cuda-graph-backend-prefill disabled \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --host 0.0.0.0 \
  --port "${PORT}" \
  "$@"
