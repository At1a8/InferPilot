// The processes a run started, found by the tag each of them inherits (INFERPILOT_RUN_TAG): an agent's
// process group misses what it started with setsid or nohup, the environment does not. A process
// started with `docker exec` carries the tag only when it was given (-e INFERPILOT_RUN_TAG). Linux only:
// without /proc nothing is found.
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync, readlinkSync } from 'node:fs';

export const TAG = 'INFERPILOT_RUN_TAG';

// A process's start time in clock ticks since boot (field 22 of /proc/<pid>/stat), or null when it is gone;
// a zombie, which has exited and waits only to be reaped, is gone.
export function startOf(pid) {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8');
    const fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
    return fields[0] === 'Z' || fields[0] === 'X' ? null : fields[19] || null;
  } catch {
    return null;
  }
}

// The pid namespace this process sees /proc in, by its inode number.
const nsOf = () => {
  try {
    return (readlinkSync('/proc/self/ns/pid').match(/\d+/) || [''])[0];
  } catch {
    return '';
  }
};
export const PID_NS = nsOf();

// <pid>-<start>-<pid namespace>: a pid alone could be reused once the runner is gone, and names another
// process in another namespace (a host's runner seen from a container, or the other way round).
export const runTag = (pid = process.pid) => `${pid}-${startOf(pid) || 0}-${PID_NS}`;

// False only when the runner is known to be gone: one in another pid namespace cannot be looked up here.
export function runnerAlive(tag) {
  const [pid, start, ns] = String(tag).split('-');
  if (ns && ns !== PID_NS) return true;
  const now = startOf(pid);
  return now !== null && (!start || start === '0' || now === start);
}

// Every process whose environment holds a run tag, this one excepted: [{pid, tag, cmd}].
export function tagged() {
  let pids;
  try {
    pids = readdirSync('/proc').filter((d) => /^\d+$/.test(d) && Number(d) !== process.pid);
  } catch {
    return [];
  }
  const out = [];
  for (const pid of pids) {
    let kv;
    try {
      kv = readFileSync(`/proc/${pid}/environ`, 'utf8').split('\0').find((e) => e.startsWith(`${TAG}=`));
    } catch {
      continue;
    }
    if (!kv) continue;
    let cmd = '';
    try {
      cmd = readFileSync(`/proc/${pid}/cmdline`, 'utf8').replace(/\0+/g, ' ').trim();
    } catch { /* gone meanwhile */ }
    out.push({ pid: Number(pid), tag: kv.slice(TAG.length + 1), cmd });
  }
  return out;
}

// What runs that are no longer running left behind.
export const orphans = () => tagged().filter((p) => !runnerAlive(p.tag));

// With InferPilot on the host, servers run in the task's container through `docker exec -e
// INFERPILOT_RUN_TAG`: their environments are root's there and unreadable from the host, and stopping the
// local docker client leaves them running. The container's root finds and stops them instead (SIGTERM,
// SIGKILL after graceS). Resolves to the pids it signalled, or null when docker could not be run.
export function stopInContainer(container, tag, graceS = 10) {
  const script = 't="INFERPILOT_RUN_TAG=$1"; pids=""; '
    + 'for d in /proc/[0-9]*; do tr "\\0" "\\n" < "$d/environ" 2>/dev/null | grep -qx "$t" && pids="$pids ${d#/proc/}"; done; '
    + '[ -z "$pids" ] && exit 0; kill -TERM $pids 2>/dev/null; i=0; '
    + 'while [ "$i" -lt "$2" ]; do alive=""; for p in $pids; do kill -0 "$p" 2>/dev/null && alive="$alive $p"; done; '
    + '[ -z "$alive" ] && break; sleep 1; i=$((i + 1)); done; kill -KILL $pids 2>/dev/null; echo $pids';
  const r = spawnSync('docker', ['exec', container, 'sh', '-c', script, 'sh', tag, String(graceS)],
    { encoding: 'utf8', timeout: (graceS + 30) * 1000 });
  return r.status === 0 ? String(r.stdout || '').trim().split(/\s+/).filter(Boolean) : null;
}

export function signal(pids, sig) {
  for (const pid of pids) {
    try {
      process.kill(pid, sig);
    } catch { /* already gone */ }
  }
}

// Whether pid is still the process that started at start, and not a zombie.
function same(pid, start) {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8');
    const fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
    return fields[0] !== 'Z' && fields[19] === start;
  } catch {
    return false;
  }
}

// SIGTERM to every process find() returns, looking several times: one scan can miss a process in the
// middle of an exec, whose environment reads empty then. SIGKILL for what is left after graceMs.
// Resolves to every process it signalled.
export async function stopPids(find, graceMs = 10000) {
  const seen = new Map();
  const look = () => {
    for (const p of find()) {
      if (seen.has(p.pid)) continue;
      seen.set(p.pid, { ...p, start: startOf(p.pid) });
      signal([p.pid], 'SIGTERM');
    }
  };
  const alive = () => [...seen.values()].filter((p) => same(p.pid, p.start));
  look();
  const deadline = Date.now() + graceMs;
  for (let i = 0; Date.now() < deadline && (i < 3 || alive().length); i++) {
    await new Promise((r) => setTimeout(r, 150));
    look();
  }
  signal(alive().map((p) => p.pid), 'SIGKILL');
  return [...seen.values()];
}
