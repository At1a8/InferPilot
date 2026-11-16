#!/bin/bash
# Serving throughput of one server at fixed lengths: random prompts of exactly INPUT_LEN input
# and OUTPUT_LEN output tokens (ignore_eos) at each max concurrency in CONCURRENCY, with
# PROMPT_MULT x concurrency prompts and WARMUP_MULT x concurrency warm-up requests. The client
# flushes the prefix cache after the warm-up and reads /server_info's avg_spec_accept_length when
# done, so the accept length covers the measured requests only. Requests are greedy unless the
# extra arguments set --temperature.
#
# Usage: bash bench_random.sh <port> <out_dir> [extra sglang.benchmark.serving args...]
#   defaults: INPUT_LEN=8000 OUTPUT_LEN=1000 CONCURRENCY="32 64 128" PROMPT_MULT=4 WARMUP_MULT=2
#   the server's default sampling of DeepSeek-V4.1-Flash: ... --temperature 1.0 --top-p 1.0
# Writes <out_dir>/bench_<in>_<out>_c<N>.{jsonl,log} and one line per concurrency to
# <out_dir>/summary.txt.
ulimit -n 65536
PORT=${1:?usage: bash bench_random.sh <port> <out_dir> [extra args...]}
OUT=${2:?usage: bash bench_random.sh <port> <out_dir> [extra args...]}
shift 2
INPUT_LEN=${INPUT_LEN:-8000}
OUTPUT_LEN=${OUTPUT_LEN:-1000}
CONCURRENCY=${CONCURRENCY:-32 64 128}
PROMPT_MULT=${PROMPT_MULT:-4}
WARMUP_MULT=${WARMUP_MULT:-2}
mkdir -p "${OUT}"

for c in ${CONCURRENCY}; do
    name=bench_${INPUT_LEN}_${OUTPUT_LEN}_c${c}
    # -P: a directory named sglang in the working directory would shadow the package.
    python3 -P -m sglang.benchmark.serving \
        --port "${PORT}" \
        --dataset-name random \
        --random-input-len "${INPUT_LEN}" \
        --random-output-len "${OUTPUT_LEN}" \
        --random-range-ratio 1 \
        --max-concurrency "${c}" \
        --num-prompts $((c * PROMPT_MULT)) \
        --flush-cache \
        --warmup-requests $((c * WARMUP_MULT)) \
        --output-file "${OUT}/${name}.jsonl" \
        "$@" > "${OUT}/${name}.log" 2>&1
    metrics=$(grep -E "Benchmark duration|Output token throughput|Total token throughput|Median TTFT|Median TPOT|Accept length" \
        "${OUT}/${name}.log" | tr -s ' ' | paste -sd ';')
    echo "c=${c}: ${metrics}" | tee -a "${OUT}/summary.txt"
done
