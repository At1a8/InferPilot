#!/usr/bin/env bash
# Start a ROCm container for InferPilot, or get back into it, and open a root shell in its /workspace
# (README step 2).
#
#   scripts/run_docker.sh                                # a new container from the newest daily for the GPU
#   NAME=<name> scripts/run_docker.sh                    # back into container <name>, or a new one of that name
#   IMAGE_VARIANT=rocm724-mi30x scripts/run_docker.sh    # exactly this GPU and ROCm version
#   IMAGE=rocm/sgl-dev:<tag> scripts/run_docker.sh       # a new container from this image
#   EXTRA_MOUNTS="/home/<you>/models" scripts/run_docker.sh   # more host directories, at the same paths
#   DOCKER_SOCKET=1 scripts/run_docker.sh                # also mount the host's Docker socket
#   MOUNT_HOME=0 scripts/run_docker.sh                   # leave /home out (a clean-room task)
#   DETACH=1 scripts/run_docker.sh                       # start it in the background, open no shell
#   PRIVILEGED=0 scripts/run_docker.sh                   # where the Docker daemon refuses --privileged
#   bash <(curl -fsSL https://raw.githubusercontent.com/At1a8/inferpilot/main/scripts/run_docker.sh)   # no checkout
#
# Each container keeps its own /workspace, where InferPilot and its runs live, and its own copy of the image's
# /sgl-workspace/sglang, the code tasks change: apart from every other container's, for as long as the
# container exists. Only a new container from the daily image has that code changed, and only once:
#
#                      no IMAGE                                   IMAGE given
#   no NAME, or no     new, from the newest daily image, pulled   new, from IMAGE (INFERPILOT_IMAGE_PINNED=1);
#   container NAME     at its first use; then in its sglang:      its sglang as the image has it
#                      git checkout -- . && git checkout main
#                      && git pull
#   container NAME     entered, its code as it was left; warns    entered, its code as it was left; warns that
#                      that it is not a new container             it is not a new container, and when it runs
#                                                                 another image than IMAGE, that too
#
# Without NAME a new container is named inferpilot_<you>_<MMDD>_<4 random letters or digits> (the UTC date), a name
# no container has, and
# labelled inferpilot.user=<you>; a new container's name, the way back into it and your other such containers
# are printed. A container is never removed: run again with its NAME, this starts it if it was stopped and opens
# a shell in it, /workspace as it was left, and says so when its image is no longer the newest daily one (tasks
# there stop at Setup to ask). The shell is a `docker exec` into the container's own bash, so leaving it leaves
# the container running. A new task gets a new container; `docker rm -f <name>` removes one you are done with,
# and its /workspace with it. Run README step 3 once in every new container: it installs Claude Code there.
#
# /home and the data directories that exist (/data, /mnt, ...) are mounted at the same paths. InferPilot's
# agents run as root with permissions auto-approved, so they can read every home on the machine and the
# credentials in it (git, ssh): MOUNT_HOME=0 leaves /home out, for a task that must not see what is there
# (a clean-room re-implementation). Your keys also come in as root's ~/.inferpilot_env, mounted read-only.
# The host's Docker socket is not mounted by default: it would let the agents stop or start any container
# on the host, and InferPilot inside the container does not need it.
#
# `docker run -it` needs a terminal. Without one (srun, CI, an agent's shell) the container starts in the
# background and the script prints the `docker exec` that enters it; DETACH=0 or 1 overrides the guess.
VARIANT=${IMAGE_VARIANT:-}
PINNED=$([ -n "$IMAGE" ] && echo 1 || echo 0)
ME=${SUDO_USER:-$USER}
ME=${ME//[^a-zA-Z0-9_.-]/_}
[[ $ME =~ ^[a-zA-Z0-9] ]] || ME="u$ME"
HOST_HOME=$HOME
[ -n "$SUDO_USER" ] && HOST_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
SUDO=$([ -w /var/run/docker.sock ] || echo sudo)
RAW=https://raw.githubusercontent.com/At1a8/inferpilot/main/scripts
SELF=$0  # how to run this again, for the hints
case "$SELF" in /dev/fd/*|/proc/*) SELF="bash <(curl -fsSL $RAW/run_docker.sh)" ;; esac
if [ -z "$DETACH" ]; then
  if [ -t 0 ]; then DETACH=0; else DETACH=1; fi
fi

entered() {  # what to run when this script opened no shell
  echo "run_docker.sh: container $NAME is running; enter it with:" \
    "${SUDO:+sudo }docker exec -it -w /workspace -e INFERPILOT_CONTAINER=$NAME $NAME bash" >&2
}

new_name() {  # inferpilot_<you>_<MMDD, UTC>_<4 random letters or digits>, a name no container has
  local name i
  for i in 1 2 3 4 5 6 7 8; do
    name="inferpilot_${ME}_$(date -u +%m%d)_$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 4)"
    $SUDO docker inspect --type container "$name" >/dev/null 2>&1 || { echo "$name"; return 0; }
  done
  return 1
}

way_back() {  # after a new container: its name, the way back into it, and your other ones
  local n img age st others=""
  echo "run_docker.sh: this is container $NAME; come back to it with: NAME=$NAME $SELF" >&2
  while IFS='|' read -r n img age st; do
    [ -n "$n" ] && [ "$n" != "$NAME" ] && others+="  $n  $img  (created $age; $st)"$'\n'
  done < <($SUDO docker ps -a --filter "label=inferpilot.user=$ME" \
             --format '{{.Names}}|{{.Image}}|{{.RunningFor}}|{{.Status}}' 2>/dev/null)
  [ -z "$others" ] || printf 'run_docker.sh: your other InferPilot containers (`%sdocker rm -f <name>` removes one you are done with, and its /workspace):\n%s' \
    "${SUDO:+sudo }" "$others" >&2
}

warn() {  # in bold yellow on a terminal
  if [ -t 2 ]; then
    printf '\033[1;33mrun_docker.sh: WARNING: %s\033[0m\n' "$*" >&2
  else
    echo "run_docker.sh: WARNING: $*" >&2
  fi
}

follow_main() {  # a new container from the daily image: its sglang on the newest main
  local out
  echo "run_docker.sh: in /sgl-workspace/sglang: git checkout -- . && git checkout main && git pull" >&2
  if out=$($SUDO docker exec -w /sgl-workspace/sglang "$NAME" bash -c \
      'git checkout -- . && git checkout -q main && git pull -q && git log -1 --format="%h %cs"' 2>&1); then
    echo "run_docker.sh: /sgl-workspace/sglang is on main at ${out##*$'\n'}; never \`pip install -e\` from it:" \
      "its pyproject.toml is upstream's" >&2
  else
    warn "could not put /sgl-workspace/sglang on the newest main: ${out##*$'\n'}"
  fi
}

latest_image() {  # scripts/latest_image.py next to this script, else the one on GitHub
  if [ -f "$(dirname "$0")/latest_image.py" ]; then
    python3 "$(dirname "$0")/latest_image.py" "$@"
  else
    curl -fsSL "$RAW/latest_image.py" | python3 - "$@"
  fi
}

detect_gpu() {  # the host GPU's part of the image variant: mi30x on gfx942, else mi35x
  local gfx
  gfx=$(rocm-smi --showproductname 2>/dev/null | awk '/GFX Version:/ { print $NF; exit }')
  [ -n "$gfx" ] || gfx=$(rocminfo 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx[0-9]+$/ { print $2; exit }')
  if [ "$gfx" = gfx942 ]; then
    echo mi30x
  else
    [ "$gfx" = gfx950 ] || echo "run_docker.sh: cannot map GPU architecture ${gfx:-unknown} to an image" \
      "variant; defaulting to mi35x. Set IMAGE_VARIANT or IMAGE to override." >&2
    echo mi35x
  fi
}

newest_local() {  # the newest local daily image of a variant
  $SUDO docker images --format '{{.Repository}}:{{.Tag}}' rocm/sgl-dev |
    grep -E ":v[0-9][0-9a-z.]*-$1-[0-9]{8}\$" | awk -F- '{print $NF, $0}' | sort | tail -n 1 | cut -d' ' -f2
}

if [ -z "$NAME" ]; then
  NAME=$(new_name) || { echo "run_docker.sh: found no free container name; set NAME=<name>" >&2; exit 1; }
fi

if OLD=$($SUDO docker inspect -f '{{.Config.Image}}' "$NAME" 2>/dev/null); then
  if [ -z "$IMAGE" ]; then
    warn "container $NAME already exists ($OLD): you enter it, not a new container, so its" \
      "/sgl-workspace/sglang is as it was left, not put on the newest main. For a new one, run without NAME:" \
      "$SELF"
  elif [ "$IMAGE" != "$OLD" ]; then
    warn "container $NAME already exists: you enter it, not a new container, and it runs $OLD, not $IMAGE:" \
      "another image version than the one you asked for. Its code is as it was left. For a new one from" \
      "$IMAGE, run without NAME: IMAGE=$IMAGE $SELF"
  else
    warn "container $NAME already exists ($OLD): you enter it, not a new container; its code is as it was" \
      "left. For a new one from $IMAGE, run without NAME: IMAGE=$IMAGE $SELF"
  fi
  echo "run_docker.sh: back into container $NAME ($OLD), /workspace as it was left" >&2
  CMP=$(latest_image --compare "$OLD" 2>/dev/null)
  case "$CMP" in
    older\ *) echo "run_docker.sh: $OLD is not the newest daily image any more; that is ${CMP#older }. Tasks in" \
                "this container stop at Setup to ask. For the newest sglang, start a new container: $SELF" >&2 ;;
  esac
  $SUDO docker start "$NAME" >/dev/null || exit 1
  [ "$DETACH" = 1 ] && { entered; exit 0; }
  exec $SUDO docker exec -it -w /workspace -e INFERPILOT_CONTAINER="$NAME" "$NAME" bash
fi

if [ -z "$IMAGE" ]; then
  # IMAGE_VARIANT is taken as given; a detected GPU falls back to its older ROCm daily builds.
  VARIANTS=$VARIANT
  if [ -z "$VARIANTS" ]; then GPU=$(detect_gpu); VARIANTS="rocm10-$GPU rocm724-$GPU rocm720-$GPU"; fi
  for VARIANT in $VARIANTS; do
    IMAGE=$(latest_image --variant "$VARIANT")
    [ -n "$IMAGE" ] && break
    IMAGE=$(newest_local "$VARIANT")
    [ -n "$IMAGE" ] && { echo "run_docker.sh: Docker Hub not reachable; using the newest local image" >&2; break; }
  done
  [ -n "$IMAGE" ] || { echo "run_docker.sh: no image found for $VARIANTS; set IMAGE=rocm/sgl-dev:<tag>" >&2; exit 1; }
  [ "$VARIANT" = "${VARIANTS%% *}" ] || echo "run_docker.sh: no ${VARIANTS%% *} image; using $VARIANT" >&2
fi
echo "run_docker.sh: new container $NAME from $IMAGE" >&2

DIRS="/data /data2 /apps /mnt /shared /nfs_shared1 /kmd2 $EXTRA_MOUNTS"
HOME_ENV=()
if [ "${MOUNT_HOME:-1}" != 0 ]; then
  DIRS="/home $DIRS"
  HOME_ENV=(-e INFERPILOT_HOST_HOME="$HOST_HOME")
fi
MOUNTS=()
for d in $DIRS; do
  [ -d "$d" ] && MOUNTS+=(-v "$d:$d")
done
# Only a file that exists: docker would create a directory in place of a missing one.
if [ -f "$HOST_HOME/.inferpilot_env" ]; then
  MOUNTS+=(-v "$HOST_HOME/.inferpilot_env:/root/.inferpilot_env:ro")
else
  echo "run_docker.sh: no $HOST_HOME/.inferpilot_env, so Claude Code in this container has no key (README step 1)." \
    "Create it, then start a new container; or write /root/.inferpilot_env in this one" >&2
fi
[ "$DOCKER_SOCKET" = 1 ] && MOUNTS+=(-v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker:ro)

# Without --privileged the GPUs still come through the devices below, but the container cannot write
# /proc/sys (kernel.numa_balancing has to be set on the host).
PRIV=(--privileged)
[ "$PRIVILEGED" = 0 ] && PRIV=()
OPTS=("${PRIV[@]}" --ipc=host --network=host --device=/dev/kfd --device=/dev/dri --group-add video
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -w /workspace
  "${MOUNTS[@]}"
  -e PYTHONDONTWRITEBYTECODE=1 -e DISABLE_AUTOUPDATER=1 -e INFERPILOT_IMAGE="$IMAGE"
  -e INFERPILOT_IMAGE_PINNED="$PINNED" -e INFERPILOT_CONTAINER="$NAME"
  "${HOME_ENV[@]}"
  --shm-size=512G --ulimit memlock=-1 --ulimit stack=67108864
  --label "inferpilot.user=$ME" --entrypoint /bin/bash --name "$NAME" "$IMAGE")

# Pulled here, in sight: `docker run -dit` below keeps its output, and a new day's image is about 43 GB.
$SUDO docker image inspect "$IMAGE" >/dev/null 2>&1 || $SUDO docker pull "$IMAGE" || exit 1
# -dit: the same bash as an interactive run, kept alive by its open stdin, so `docker start` works the same; the
# shell you get is a `docker exec` into it, as when you come back, after the daily image's sglang is on main.
if ! OUT=$($SUDO docker run -dit "${OPTS[@]}" 2>&1); then
  echo "$OUT" >&2
  case "$OUT" in
    *authorization*--privileged*) echo "run_docker.sh: this Docker daemon refuses --privileged; run again with PRIVILEGED=0" >&2 ;;
    *authorization*/home:/home*) echo "run_docker.sh: this Docker daemon refuses a mount of all of /home; run again with" \
                                   "MOUNT_HOME=0 (EXTRA_MOUNTS=$HOST_HOME brings back only your home)" >&2 ;;
  esac
  exit 1
fi
[ "$PINNED" = 1 ] || follow_main
way_back
[ "$DETACH" = 1 ] && { entered; exit 0; }
exec $SUDO docker exec -it -w /workspace -e INFERPILOT_CONTAINER="$NAME" "$NAME" bash
