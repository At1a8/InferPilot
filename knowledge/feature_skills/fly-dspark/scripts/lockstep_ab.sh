#!/bin/bash
# Two-lane lockstep A/B for FLy on one 8-GPU node. Lane A is GPUs 0-3 (port 30010), lane B GPUs
# 4-7 (port 30011), each a TP4 server from serve_dspark_fly.sh. Every round starts one server per
# lane, runs bench_random.sh on both at the same time, then, when ACC_CMD is set, the accuracy
# command on both at the same time, and stops them. Swapping the arms between the lanes from round
# to round spreads any lane difference over both arms, and benching both lanes at once gives every
# measurement the same load from the other lane.
#
# Usage: MODEL_PATH=<dir> OUT=<dir> [SGLANG_TREE=<checkout>] [ACC_CMD=<command>] \
#        bash lockstep_ab.sh <laneA_arm>:<laneB_arm> [<laneA_arm>:<laneB_arm> ...]
#   an arm is "baseline" or "fly_w<W>_t<T>" (FLy window W, entropy threshold T), one pair per round:
#   bash lockstep_ab.sh baseline:fly_w2_t0.7 fly_w2_t0.7:baseline baseline:fly_w2_t0.7
# ACC_CMD runs once per server with API_URL, TAG and RESULT_ROOT in its environment and must write
# RESULT_ROOT/summary.txt, one line per dataset (scripts/eval_accuracy.sh adds " dir=<its run dir>"):
#   <tag> <dataset>: accept_length=[<x>] aggregate={'score': <s>, ...}
# The bench settings are bench_random.sh's variables (INPUT_LEN, OUTPUT_LEN, CONCURRENCY, ...).
# Lanes: GPUS_A / GPUS_B and PORT_A / PORT_B. Results, with tag r<round>_<arm> (plus _a / _b when
# both lanes run the same arm in a round): OUT/status.txt,
# OUT/server/<tag>.log, OUT/bench/<tag>/, OUT/acc/<tag>/; summary: python3 ab_table.py OUT.
# touch OUT/STOP to stop after the current round. DRY=1 only logs the schedule.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:?set OUT to the result directory}
[ -n "${DRY:-}" ] || : "${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint directory}"
[ $# -gt 0 ] || { echo "usage: bash lockstep_ab.sh <laneA_arm>:<laneB_arm> ..." >&2; exit 2; }
STATUS=${OUT}/status.txt
PORTS=("${PORT_A:-30010}" "${PORT_B:-30011}")
DEVICES=("${GPUS_A:-0,1,2,3}" "${GPUS_B:-4,5,6,7}")
SERVER_PID=(0 0)
[ -n "${SGLANG_TREE:-}" ] && export PYTHONPATH=${SGLANG_TREE}/python${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${OUT}/server"

args_of() {  # <arm>
    case "$1" in
        baseline) ;;
        fly_w*_t*)
            local w=${1#fly_w}
            echo "--speculative-use-fly --speculative-fly-window-size ${w%%_t*} --speculative-fly-entropy-threshold ${1##*_t}"
            ;;
        *) return 1 ;;
    esac
}

for pair in "$@"; do
    for arm in "${pair%%:*}" "${pair#*:}"; do
        args_of "${arm}" > /dev/null || { echo "unknown arm '${arm}' in '${pair}'" >&2; exit 2; }
    done
done

log() { echo "$(date -u '+%F %TZ') $*" | tee -a "${STATUS}"; }

wait_ready() {  # <log> <pid>: up to 20 min; a traceback fails it only if the server is not up 2 min later
    local tracebacks=0
    for _ in $(seq 1 240); do
        grep -q "fired up" "$1" && return 0
        kill -0 "$2" 2> /dev/null || return 1
        if grep -qE "Traceback|Fatal Python error" "$1"; then
            tracebacks=$((tracebacks + 1))
            [ "${tracebacks}" -ge 24 ] && return 1
        fi
        sleep 5
    done
    return 1
}

start_server() {  # <lane> <tag> <extra args>
    log "round ${round}: launching $2 (port ${PORTS[$1]}, GPUs ${DEVICES[$1]})"
    # shellcheck disable=SC2086
    bash "${HERE}/serve_dspark_fly.sh" "${PORTS[$1]}" "${DEVICES[$1]}" $3 > "${OUT}/server/$2.log" 2>&1 &
    SERVER_PID[$1]=$!
}

check_ready() {  # <lane> <tag>
    local fly
    if wait_ready "${OUT}/server/$2.log" "${SERVER_PID[$1]}"; then
        fly=$(grep -m1 -oE "FlyConfig\([^)]*\)" "${OUT}/server/$2.log")
        log "round ${round}: $2 ready, ${fly:-standard verification}"
        return 0
    fi
    log "round ${round}: $2 failed to start; see ${OUT}/server/$2.log"
    return 1
}

stop_server() {  # <port>
    pkill -TERM -f "[s]glang serve.*--port $1"
    for _ in $(seq 1 60); do
        pgrep -f "[s]glang serve.*--port $1" > /dev/null || break
        sleep 2
    done
    pkill -KILL -f "[s]glang serve.*--port $1"
    sleep 10
}

bench_lane() {  # <lane> <tag>
    local start=${SECONDS} stalls
    bash "${HERE}/bench_random.sh" "${PORTS[$1]}" "${OUT}/bench/$2" --temperature 1.0 --top-p 1.0
    log "$2: bench finished in $((SECONDS - start))s"
    stalls=$(grep -oE "kernel '[^']*' took [0-9.]+ s to compile after serving started" "${OUT}/server/$2.log" | sort -u | paste -sd ';')
    [ -n "${stalls}" ] && log "WARNING $2 compiled while serving (discard or repeat this run): ${stalls}"
}

acc_lane() {  # <lane> <tag>
    local start=${SECONDS}
    env API_URL="http://127.0.0.1:${PORTS[$1]}/v1" TAG="$2" RESULT_ROOT="${OUT}/acc/$2" \
        bash -c "${ACC_CMD}" > "${OUT}/acc_$2.out" 2>&1
    log "$2: accuracy finished in $((SECONDS - start))s: $(tr '\n' ';' < "${OUT}/acc/$2/summary.txt" 2> /dev/null | cut -c1-400)"
}

if [ -n "${DRY:-}" ]; then
    start_server() { log "round ${round}: [dry] launch $2 on port ${PORTS[$1]}, GPUs ${DEVICES[$1]}: ${3:-standard verification}"; }
    check_ready() { return 0; }
    bench_lane() { :; }
    acc_lane() { :; }
    stop_server() { :; }
fi

log "lockstep A/B: $(python3 -c 'import sglang; print(sglang.__file__)' 2> /dev/null), rounds: $*"
round=0
for pair in "$@"; do
    round=$((round + 1))
    if [ -e "${OUT}/STOP" ]; then
        log "STOP found; stopping before round ${round}"
        break
    fi
    arms=("${pair%%:*}" "${pair#*:}")
    tags=("r${round}_${arms[0]}" "r${round}_${arms[1]}")
    [ "${arms[0]}" = "${arms[1]}" ] && tags=("${tags[0]}_a" "${tags[1]}_b")
    ready=(0 0)
    start_server 0 "${tags[0]}" "$(args_of "${arms[0]}")"
    start_server 1 "${tags[1]}" "$(args_of "${arms[1]}")"
    check_ready 0 "${tags[0]}" && ready[0]=1
    check_ready 1 "${tags[1]}" && ready[1]=1
    phases=(bench_lane)
    [ -n "${ACC_CMD:-}" ] && phases+=(acc_lane)
    for phase in "${phases[@]}"; do
        pids=()
        for lane in 0 1; do
            if [ "${ready[lane]}" = 1 ]; then
                "${phase}" "${lane}" "${tags[lane]}" &
                pids+=($!)
            fi
        done
        [ "${#pids[@]}" -gt 0 ] && wait "${pids[@]}"
    done
    stop_server "${PORTS[0]}" &
    pids=($!)
    stop_server "${PORTS[1]}" &
    pids+=($!)
    wait "${pids[@]}"
    log "round ${round}: done"
done
log "LOCKSTEP_AB_DONE"
