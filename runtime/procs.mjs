// The processes a run started, found by the tag each of them inherits (INFERPILOT_RUN_TAG): an agent's
// process group misses what it started with setsid or nohup, the environment does not. A process
// started with `docker exec` carries the tag only when it was given (-e INFERPILOT_RUN_TAG). Linux only:
// without /proc nothing is found.
import { readdirSync, readFileSync } from 'node:fs';

export const TAG = 'INFERPILOT_RUN_TAG';

// A process's start time in clock ticks since boot (field 22 of /proc/<pid>/stat), or null when it is gone.
export function startOf(pid) {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8');
    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19] || null;
  } catch {
    return null;
  }
}

// <pid>-<start>: a pid alone could be reused by another process once the runner is gone.
export const runTag = (pid = process.pid) => `${pid}-${startOf(pid) || 0}`;

export function runnerAlive(tag) {
  const [pid, start] = String(tag).split('-');
  const now = startOf(pid);
  return now !== null && (start === '0' || now === start);
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

export function signal(pids, sig) {
  for (const pid of pids) {
    try {
      process.kill(pid, sig);
    } catch { /* already gone */ }
  }
}

// SIGTERM, then SIGKILL for what is still there after graceMs; resolves to the pids that needed SIGKILL.
export async function stopPids(find, graceMs = 10000) {
  signal(find().map((p) => p.pid), 'SIGTERM');
  const deadline = Date.now() + graceMs;
  while (find().length && Date.now() < deadline) await new Promise((r) => setTimeout(r, 200));
  const left = find().map((p) => p.pid);
  signal(left, 'SIGKILL');
  return left;
}
