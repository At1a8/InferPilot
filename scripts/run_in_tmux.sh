#!/usr/bin/env bash
# README step 4 where there is no terminal to type in (an agent's shell, a job scheduler): run the
# interactive Claude Code in a tmux session inside the container and give it the task from a file.
#
#   scripts/run_in_tmux.sh <task-file>                  # the line you would type, written to a file
#
#   tmux capture-pane -p -t inferpilot | tail -40       # watch it
#   tmux attach -t inferpilot                           # take over, from a terminal
#
# `claude -p "use inferpilot.js ..."` is no substitute: the Workflow tool runs inferpilot.js in the
# background, print mode answers within seconds and then ends the process, and the run dies with it.
#
# INFERPILOT_TMUX names the session (default inferpilot) and INFERPILOT_HOME the checkout (default the one
# this script is in); it looks for the prompt TRIES times (default 90), POLL_S seconds apart (default 2).
#
# Exit 0 = task handed over, 2 = usage or no tmux, 3 = a first-start question is up (the bootstrap's
# claude-config step did not answer it), 4 = Claude Code asks to log in (no key), 5 = no prompt in time.
set -u
TASK=${1:-}
[ -n "$TASK" ] && [ -r "$TASK" ] || { echo "usage: run_in_tmux.sh <task-file>" >&2; exit 2; }
command -v tmux >/dev/null || { echo "run_in_tmux.sh: tmux not found (apt-get install -y tmux)" >&2; exit 2; }
S=${INFERPILOT_TMUX:-inferpilot}
DIR=${INFERPILOT_HOME:-$(cd "$(dirname "$0")/.." && pwd)}
TRIES=${TRIES:-90}
POLL=${POLL_S:-2}

screen() { tmux capture-pane -p -t "$S"; }
show() { screen | grep -v '^[[:space:]]*$' | tail -20 >&2; }

if ! tmux has-session -t "$S" 2>/dev/null; then
  # an interactive shell: root's ~/.bashrc loads the keys after its interactive-only guard
  tmux new-session -d -s "$S" -x 200 -y 50 "bash -i"
  tmux send-keys -t "$S" "cd $DIR && IS_SANDBOX=1 claude --dangerously-skip-permissions" Enter
fi

ready=0
for _ in $(seq 1 "$TRIES"); do
  sleep "$POLL"
  out=$(screen)
  case "$out" in
    *"bypass permissions on"*) ready=1; break ;;
    *"login method"*|*"Paste code here if prompted"*)
      show; echo "run_in_tmux.sh: Claude Code asks to log in: its shell has no key (README step 3)" >&2; exit 4 ;;
    *"Choose the text style"*|*"Press Enter to continue"*|*"trust this folder"*|*"Bypass Permissions mode"*|*"use this API key"*)
      show
      echo "run_in_tmux.sh: a first-start question is up, which claude-config should have answered: run" \
        "python3 -B -m inferpilot.bootstrap --claude-only, or answer it as README step 4 says (tmux attach -t $S)" >&2
      exit 3 ;;
  esac
done
[ $ready = 1 ] || { show; echo "run_in_tmux.sh: no Claude Code prompt after $TRIES tries" >&2; exit 5; }

# no trailing newline: the Enter below submits it once
printf '%s' "$(cat "$TASK")" | tmux load-buffer -b inferpilot-task -
tmux paste-buffer -p -d -b inferpilot-task -t "$S"
sleep "$POLL"
tmux send-keys -t "$S" Enter
echo "run_in_tmux.sh: task given to Claude Code in tmux session $S; watch: tmux capture-pane -p -t $S | tail -40" >&2
