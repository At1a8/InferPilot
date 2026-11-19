#!/bin/bash
# Accuracy of one running server with evalscope: InferPilot's G2 harness.
#
#   bash scripts/eval_accuracy.sh                        # GSM8K, all 1319 questions, once
#   bash scripts/eval_accuracy.sh gsm8k gpqa_diamond     # several sets, one after the other
#   API_URL=http://127.0.0.1:30010/v1 TAG=baseline RESULT_ROOT=<run dir>/accuracy bash scripts/eval_accuracy.sh
#
# Sets: gsm8k (1319, rule-graded), gpqa_diamond (198), mmlu_pro (12032), simple_qa (4326, LLM judge),
# hle (the 2158 text-only questions, LLM judge), longbench_v2 (503; LONGBENCH_LOCAL_PATH for a local copy).
#
# Every key comes from the environment (InferPilot's ~/.inferpilot_env); none is written here.
#   API_URL           the server's OpenAI endpoint (default http://127.0.0.1:30000/v1)
#   MODEL             default: the model the server serves; another name makes every request fail
#   TAG, RESULT_ROOT  results go to <RESULT_ROOT>/<TAG>_<set>/ and <TAG>_<set>.log (defaults: eval,
#                     ./results); one line per set is appended to <RESULT_ROOT>/summary.txt
#   BATCH             concurrent requests (default per set). Keep it within the server's KV capacity, or
#                     sglang retracts requests and the run slows down
#   LIMIT             at most this many questions per subset: a smoke test, not a result
#   WAIT_SERVER       seconds to wait for a server that is still starting (default 0)
#   MAX_TOKENS        default 102400: a truncated reasoning trace leaves the answer empty
#   REASONING_EFFORT  default max; ENABLE_THINKING default 1 (chat_template_kwargs.thinking and
#                     .enable_thinking true; 0 sends both false). Thinking models score wrongly without
#                     them; ENABLE_THINKING=0 REASONING_EFFORT= for others
#   TOP_P, TOP_K      sampling overrides (default: the server's own)
#   DATASET_HUB       modelscope (default) or huggingface; HF_TOKEN for gated sets
#   CACHE_DIR, RERUN_REVIEW=1   resume an earlier work dir; re-run only the judging
#   EVALSCOPE         the evalscope executable (default: evalscope on PATH, which the bootstrap installs
#                     in a venv of its own)
#   Judge-graded sets: JUDGE_KEY (default LLM_GATEWAY_KEY), JUDGE_URL (default the AMD LLM gateway),
#   JUDGE_MODEL (default Claude-Opus-5; never the model under test), JUDGE_NTID (the gateway's user
#   header, which its SDK route requires)
#
# Before each set the server's prefix cache is flushed; after it, /server_info's speculative accept
# length is read, so it covers this set only. Run nothing else on the server meanwhile.
# Exit status: 0 when every set produced a report, 1 otherwise.
set -u

[ $# -eq 0 ] && set -- gsm8k

API_URL=${API_URL:-http://127.0.0.1:30000/v1}
TAG=${TAG:-eval}
RESULT_ROOT=${RESULT_ROOT:-./results}
WAIT_SERVER=${WAIT_SERVER:-0}
DATASET_HUB=${DATASET_HUB:-modelscope}
MAX_TOKENS=${MAX_TOKENS:-102400}
TOP_P=${TOP_P:-}
TOP_K=${TOP_K:-}
REASONING_EFFORT=${REASONING_EFFORT-max}
ENABLE_THINKING=${ENABLE_THINKING:-1}
EVALSCOPE=${EVALSCOPE:-evalscope}
SERVER_URL=${API_URL%/v1}
SERVER_URL=${SERVER_URL%/}
JUDGE_URL=${JUDGE_URL:-https://llm-api.amd.com/Unified/v1}
JUDGE_MODEL=${JUDGE_MODEL:-Claude-Opus-5}
JUDGE_KEY=${JUDGE_KEY:-${LLM_GATEWAY_KEY:-}}
JUDGE_NTID=${JUDGE_NTID:-}
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# DS_TOTAL checks the denominator afterwards: evalscope's --ignore-errors drops failed samples from both
# the predictions and the mean, so a missing question is invisible in the score. 0: not a fixed count.
dataset_profile() {  # <set> -> DS_BATCH, DS_ARGS, DS_TOTAL, DS_JUDGE
    DS_BATCH=256
    DS_ARGS=""
    DS_TOTAL=0
    DS_JUDGE=0
    case "$1" in
        gsm8k)        DS_TOTAL=1319 ;;
        gpqa_diamond) DS_TOTAL=198
                      [ "${DATASET_HUB}" = huggingface ] && \
                          DS_ARGS='{"gpqa_diamond":{"dataset_id":"Idavidrein/gpqa","subset_list":["gpqa_diamond"]}}' ;;
        mmlu_pro)     DS_TOTAL=12032 ;;
        simple_qa)    DS_TOTAL=4326; DS_JUDGE=1
                      [ "${DATASET_HUB}" = huggingface ] && \
                          DS_ARGS='{"simple_qa":{"dataset_id":"basicv8vc/SimpleQA"}}' ;;
        # About 14% of HLE comes with images, which a text-only server cannot see.
        hle)          DS_TOTAL=2158; DS_JUDGE=1
                      DS_ARGS='{"hle":{"extra_params":{"include_multi_modal":false}}}' ;;
        longbench_v2) DS_TOTAL=503
                      if [ -n "${LONGBENCH_LOCAL_PATH:-}" ]; then
                          DS_TOTAL=$(grep -c . "${LONGBENCH_LOCAL_PATH}/default_test.jsonl" 2>/dev/null || echo 0)
                          DS_ARGS="{\"longbench_v2\":{\"local_path\":\"${LONGBENCH_LOCAL_PATH}\",\"eval_split\":\"test\"}}"
                      elif [ "${DATASET_HUB}" = huggingface ]; then
                          DS_ARGS='{"longbench_v2":{"dataset_id":"zai-org/LongBench-v2"}}'
                      fi ;;
    esac
}

if [ "${WAIT_SERVER}" -gt 0 ]; then
    deadline=$((SECONDS + WAIT_SERVER))
    until curl -s -o /dev/null --max-time 5 "${SERVER_URL}/health" || [ "${SECONDS}" -ge "${deadline}" ]; do
        sleep 15
    done
fi

# evalscope sends --model as the request's model and names its result directory after it.
SERVED=$(curl -s --max-time 20 "${API_URL%/}/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
if [ -z "${SERVED}" ]; then
    echo "eval_accuracy: no server answers at ${API_URL} (WAIT_SERVER=<seconds> waits for one that is starting)" >&2
    exit 1
fi
MODEL=${MODEL:-${SERVED}}
if [ "${MODEL}" != "${SERVED}" ]; then
    echo "eval_accuracy: MODEL=${MODEL}, but the server serves ${SERVED}; every request would fail. Unset MODEL." >&2
    exit 1
fi

# The judge is probed only when a requested set needs one, and before any inference: a bad key then
# fails in seconds, not after an hour of generation. Three tries ride out a transient gateway error.
NEED_JUDGE=0
for ds in "$@"; do
    dataset_profile "${ds}"
    [ "${DS_JUDGE}" = 1 ] && NEED_JUDGE=1
done
JUDGE_CODE=none
JUDGE_HEADERS=(-H "Ocp-Apim-Subscription-Key: ${JUDGE_KEY}")
[ -n "${JUDGE_NTID}" ] && JUDGE_HEADERS+=(-H "user: ${JUDGE_NTID}")
if [ "${NEED_JUDGE}" = 1 ] && [ -n "${JUDGE_KEY}" ]; then
    for attempt in 1 2 3; do
        JUDGE_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "${JUDGE_HEADERS[@]}" "${JUDGE_URL%/}/models")
        [ "${JUDGE_CODE}" = 200 ] && break
        [ "${attempt}" -lt 3 ] && sleep 10
    done
fi

# A resumed work dir must not mix samples generated with other sampling settings.
fingerprint() {
    echo "model=${MODEL} max_tokens=${MAX_TOKENS} effort=${REASONING_EFFORT:-none} thinking=${ENABLE_THINKING}" \
         "top_p=${TOP_P:-server} top_k=${TOP_K:-server}"
}

check_fingerprint() {  # <work dir>
    local fp_file="$1/sampling_fingerprint.txt" current previous
    current=$(fingerprint)
    if [ -n "${CACHE_DIR:-}" ] && [ -f "${fp_file}" ]; then
        previous=$(cat "${fp_file}")
        if [ "${previous}" != "${current}" ]; then
            echo "eval_accuracy: the cached samples used other settings (${previous}); use another TAG" >&2
            return 1
        fi
    fi
    mkdir -p "$1" && echo "${current}" > "${fp_file}"
}

FAILED=0
SUMMARY=""

run_one() {  # <set>
    local ds=$1 batch work_dir log eb="" tail_cfg gen_cfg judge_args rc accept rep line think
    local -a cache_arg=() limit_arg=() args_arg=()
    dataset_profile "${ds}"
    batch=${BATCH:-${DS_BATCH}}
    work_dir="${RESULT_ROOT}/${TAG}_${ds}"
    log="${RESULT_ROOT}/${TAG}_${ds}.log"
    mkdir -p "${RESULT_ROOT}"

    if [ "${DS_JUDGE}" = 1 ] && [ "${JUDGE_CODE}" != 200 ]; then
        if [ -z "${JUDGE_KEY}" ]; then
            echo "${ds}: needs an LLM judge, and neither JUDGE_KEY nor LLM_GATEWAY_KEY is set" >&2
        else
            echo "${ds}: needs an LLM judge, and the judge did not answer (HTTP ${JUDGE_CODE}; 401: a bad key," \
                 "400: JUDGE_NTID missing)" >&2
        fi
        SUMMARY="${SUMMARY}\n  ${ds}: skipped, no judge"
        FAILED=1
        return
    fi
    if ! check_fingerprint "${work_dir}"; then
        SUMMARY="${SUMMARY}\n  ${ds}: skipped, other sampling settings in CACHE_DIR"
        FAILED=1
        return
    fi
    if [ -n "${CACHE_DIR:-}" ]; then
        cache_arg=(--use-cache "${CACHE_DIR}")
        [ "${RERUN_REVIEW:-0}" = 1 ] && cache_arg+=(--rerun-review)
    fi
    [ -n "${LIMIT:-}" ] && limit_arg=(--limit "${LIMIT}")
    [ -n "${DS_ARGS}" ] && args_arg=(--dataset-args "${DS_ARGS}")

    # extra_body is spread into the request's top level, so reasoning_effort must be a value the server
    # accepts; top_k is an sglang extension and only reaches the server through extra_body.
    [ -n "${REASONING_EFFORT}" ] && eb="\"reasoning_effort\":\"${REASONING_EFFORT}\""
    # Templates read thinking (DeepSeek) or enable_thinking (Qwen3, which thinks unless it is told not to):
    # both go, true or false, so ENABLE_THINKING=0 turns thinking off for either.
    think=false
    [ "${ENABLE_THINKING}" = 1 ] && think=true
    [ -n "${eb}" ] && eb="${eb},"
    eb="${eb}\"chat_template_kwargs\":{\"thinking\":${think},\"enable_thinking\":${think}}"
    if [ -n "${TOP_K}" ]; then
        [ -n "${eb}" ] && eb="${eb},"
        eb="${eb}\"top_k\":${TOP_K}"
    fi
    tail_cfg="\"max_tokens\":${MAX_TOKENS}"
    [ -n "${TOP_P}" ] && tail_cfg="${tail_cfg},\"top_p\":${TOP_P}"
    gen_cfg="{${tail_cfg}}"
    [ -n "${eb}" ] && gen_cfg="{\"extra_body\":{${eb}},${tail_cfg}}"

    echo "== ${ds}: model=${MODEL} batch=${batch} questions=${DS_TOTAL:-?} judge=${DS_JUDGE}; $(fingerprint); log ${log}"
    curl -s -X POST --max-time 60 "${SERVER_URL}/flush_cache" > /dev/null

    # The gateway wants its key in a header as well, which evalscope's CLI passes through model_args.
    judge_args="{\"model_id\": \"${JUDGE_MODEL}\", \"api_url\": \"${JUDGE_URL}\", \"api_key\": \"${JUDGE_KEY}\", \"model_args\": {\"default_headers\": {\"Ocp-Apim-Subscription-Key\": \"${JUDGE_KEY}\", \"user\": \"${JUDGE_NTID}\"}}}"
    "${EVALSCOPE}" eval \
        --model "${MODEL}" \
        --api-url "${API_URL}" \
        --api-key EMPTY_KEY \
        --datasets "${ds}" \
        --dataset-hub "${DATASET_HUB}" \
        "${args_arg[@]}" \
        --generation-config "${gen_cfg}" \
        --judge-strategy auto \
        --judge-model-args "${judge_args}" \
        --eval-batch-size "${batch}" \
        --stream \
        --ignore-errors \
        "${limit_arg[@]}" \
        "${cache_arg[@]}" \
        --work-dir "${work_dir}" \
        --repeats 1 > "${log}" 2>&1
    rc=$?

    accept=$(curl -s --max-time 20 "${SERVER_URL}/server_info" | python3 -c \
        "import json,sys; print([s.get('avg_spec_accept_length') for s in json.load(sys.stdin)['internal_states']])" \
        2>/dev/null)
    rep=$(find "${work_dir}" -name '*.json' -path '*/reports/*' 2>/dev/null | sort | tail -1)
    if [ -z "${rep}" ]; then
        echo "${ds}: no report (evalscope exit ${rc}); the end of ${log}:" >&2
        tail -8 "${log}" | cut -c1-200 >&2
        SUMMARY="${SUMMARY}\n  ${ds}: failed (exit ${rc})"
        FAILED=1
        return
    fi

    line=$(DS_TOTAL="${DS_TOTAL}" LOG="${log}" python3 -c "
import json, os, sys
rep = json.load(open(sys.argv[1]))
total = int(os.environ['DS_TOTAL'])
out = []
for m in rep.get('metrics', []):
    n, score = m['num'], m['score']
    name = m.get('name') or (m.get('identity') or {}).get('name') or m.get('legacy_name')
    out.append('%s=%s (%d questions)' % (name, score, n))
    if total and n < total:
        out.append('%d missing -> %.4f over all %d' % (total - n, score * n / total, total))
dropped = sum(1 for l in open(os.environ['LOG'], errors='replace') if 'Error ignored' in l)
if dropped:
    out.append('%d samples dropped by --ignore-errors' % dropped)
print('  '.join(out))
" "${rep}" 2>/dev/null)
    echo "${ds}: ${line}  accept_length=${accept:-?}"
    SUMMARY="${SUMMARY}\n  ${ds}: ${line}  accept_length=${accept:-?}"

    python3 -c "
import json, sys
rep = json.load(open(sys.argv[1]))
metrics = rep.get('metrics', [])
agg = {m.get('name') or (m.get('identity') or {}).get('name'): m['score'] for m in metrics}
if metrics:
    agg = {'score': metrics[0]['score'], 'num': metrics[0]['num'], **agg}
print('%s %s: accept_length=%s aggregate=%s' % (sys.argv[2], sys.argv[3], sys.argv[4], agg))
" "${rep}" "${TAG}" "${ds}" "${accept:-[None]}" >> "${RESULT_ROOT}/summary.txt"
}

for ds in "$@"; do
    run_one "${ds}"
done
echo -e "== summary (${TAG}):${SUMMARY}"
exit "${FAILED}"
