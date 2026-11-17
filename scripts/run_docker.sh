#!/usr/bin/env bash
# Start a ROCm container for InferPilot, or get back into it, and open a root shell in its /workspace
# (README step 2).
#
#   scripts/run_docker.sh                                # newest ROCm 10 daily for the detected gfx942/gfx950 GPU
#   IMAGE_VARIANT=rocm724-mi30x scripts/run_docker.sh    # override the detected GPU or ROCm version
#   IMAGE=rocm/sgl-dev:<tag> NAME=<name> scripts/run_docker.sh
#   EXTRA_MOUNTS="/home/<you>/models" scripts/run_docker.sh   # more host directories, at the same paths
#   DOCKER_SOCKET=1 scripts/run_docker.sh                # also mount the host's Docker socket
#   MOUNT_HOME=0 scripts/run_docker.sh                   # leave /home out (a clean-room task)
#   DETACH=1 scripts/run_docker.sh                       # start it in the background, open no shell
#   PRIVILEGED=0 scripts/run_docker.sh                   # where the Docker daemon refuses --privileged
#   bash <(curl -fsSL https://raw.githubusercontent.com/At1a8/inferpilot/main/scripts/run_docker.sh)   # no checkout
#
# Each container keeps its own /workspace, where InferPilot and its runs live, and its own copy of the image's
# /sgl-workspace/sglang, the code tasks change: apart from every other container's, for as long as the
# container exists. A new container starts from the newest daily image, pulled at its first use. A
# container named NAME is never removed: run again, this starts it if it was stopped and opens a shell in
# it, /workspace as it was left, and says so when its image is no longer the newest daily one (tasks there
# stop at Setup to ask). A new task gets a new container under another NAME; `docker rm -f <name>`
# removes one you are done with, and its /workspace with it. Run README step 3 once in every new
# container: it installs Claude Code there.
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
NAME=${NAME:-${SUDO_USER:-$USER}}
HOST_HOME=$HOME
[ -n "$SUDO_USER" ] && HOST_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
SUDO=$([ -w /var/run/docker.sock ] || echo sudo)
RAW=https://raw.githubusercontent.com/At1a8/inferpilot/main/scripts
if [ -z "$DETACH" ]; then
  if [ -t 0 ]; then DETACH=0; else DETACH=1; fi
fi

entered() {  # what to run when this script opened no shell
  echo "run_docker.sh: container $NAME is running; enter it with:" \
    "${SUDO:+sudo }docker exec -it -w /workspace -e INFERPILOT_CONTAINER=$NAME $NAME bash" >&2
}

latest_image() {  # scripts/latest_image.py next to this script, else the one on GitHub
  if [ -f "$(dirname "$0")/latest_image.py" ]; then
    python3 "$(dirname "$0")/latest_image.py" "$@"
  else
    curl -fsSL "$RAW/latest_image.py" | python3 - "$@"
  fi
}

detect_variant() {
  local info="" gfx=""
  if command -v rocm-smi >/dev/null 2>&1; then
    info=$(rocm-smi --showproductname 2>/dev/null || true)
    gfx=$(printf '%s\n' "$info" | awk '/GFX Version:/ { print $NF; exit }')
  fi
  if [ -z "$gfx" ] && command -v rocminfo >/dev/null 2>&1; then
    info=$(rocminfo 2>/dev/null || true)
    gfx=$(printf '%s\n' "$info" | awk '$1 == "Name:" && $2 ~ /^gfx[0-9]+$/ { print $2; exit }')
  fi
  if [ "$gfx" = gfx942 ]; then
    echo rocm10-mi30x
  elif [ "$gfx" = gfx950 ]; then
    echo rocm10-mi35x
  else
    echo "run_docker.sh: cannot map GPU architecture ${gfx:-unknown} to an image variant;" \
      "defaulting to rocm10-mi35x. Set IMAGE_VARIANT or IMAGE to override." >&2
    echo rocm10-mi35x
  fi
}

if OLD=$($SUDO docker inspect -f '{{.Config.Image}}' "$NAME" 2>/dev/null); then
  if [ -n "$IMAGE" ] && [ "$IMAGE" != "$OLD" ]; then
    echo "run_docker.sh: container $NAME runs $OLD, not $IMAGE. Pick another NAME, or remove it first" \
      "(\`${SUDO:+sudo }docker rm -f $NAME\`, which deletes its /workspace)" >&2
    exit 1
  fi
  echo "run_docker.sh: back into container $NAME ($OLD), /workspace as it was left" >&2
  CMP=$(latest_image --compare "$OLD" 2>/dev/null)
  case "$CMP" in
    older\ *) echo "run_docker.sh: $OLD is not the newest daily image any more; that is ${CMP#older }. Tasks in" \
                "this container stop at Setup to ask. For the newest sglang, start a new container:" \
                "NAME=$NAME-$(date +%m%d) $0 (or remove this one first: \`${SUDO:+sudo }docker rm -f $NAME\`," \
                "which deletes its /workspace)" >&2 ;;
  esac
  $SUDO docker start "$NAME" >/dev/null || exit 1
  [ "$DETACH" = 1 ] && { entered; exit 0; }
  exec $SUDO docker exec -it -w /workspace -e INFERPILOT_CONTAINER="$NAME" "$NAME" bash
fi

if [ -z "$IMAGE" ]; then
  [ -n "$VARIANT" ] || VARIANT=$(detect_variant)
  IMAGE=$(latest_image --variant "$VARIANT")
  if [ -z "$IMAGE" ]; then
    IMAGE=$($SUDO docker images --format '{{.Repository}}:{{.Tag}}' rocm/sgl-dev |
      grep -E ":v[0-9][0-9a-z.]*-$VARIANT-[0-9]{8}\$" | awk -F- '{print $NF, $0}' | sort | tail -n 1 | cut -d' ' -f2)
    [ -n "$IMAGE" ] && echo "run_docker.sh: Docker Hub not reachable; using the newest local image" >&2
  fi
  [ -n "$IMAGE" ] || { echo "run_docker.sh: no $VARIANT image found; set IMAGE=rocm/sgl-dev:<tag>" >&2; exit 1; }
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
  -e PYTHONDONTWRITEBYTECODE=1 -e DISABLE_AUTOUPDATER=1 -e INFERPILOT_IMAGE="$IMAGE" -e INFERPILOT_CONTAINER="$NAME"
  "${HOME_ENV[@]}"
  --shm-size=512G --ulimit memlock=-1 --ulimit stack=67108864
  --entrypoint /bin/bash --name "$NAME" "$IMAGE")

[ "$DETACH" = 1 ] || exec $SUDO docker run -it "${OPTS[@]}"
# -dit: the same bash as an interactive run, kept alive by its open stdin, so `docker start` works the same
if ! OUT=$($SUDO docker run -dit "${OPTS[@]}" 2>&1); then
  echo "$OUT" >&2
  case "$OUT" in
    *authorization*--privileged*) echo "run_docker.sh: this Docker daemon refuses --privileged; run again with PRIVILEGED=0" >&2 ;;
    *authorization*/home:/home*) echo "run_docker.sh: this Docker daemon refuses a mount of all of /home; run again with" \
                                   "MOUNT_HOME=0 (EXTRA_MOUNTS=$HOST_HOME brings back only your home)" >&2 ;;
  esac
  exit 1
fi
entered
