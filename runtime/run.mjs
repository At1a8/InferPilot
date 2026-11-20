#!/usr/bin/env node
// Run inferpilot.js without Claude Code's Workflow tool: every role is one run of a command-line agent.
//
//   node runtime/run.mjs inferpilot.js --backend cursor --args '{"mode":"scan","inferpilot_dir":"/workspace/inferpilot"}'
//
//   --backend claude | cursor     the agent CLI each role runs on (claude -p, or the Cursor CLI `agent -p`)
//   --model <id>                  its model (default: the CLI's own); the reviewer's comes from review_model
//   --args <json> | --args-file <path>
//   --result-file <path>          also write the run's result there (it always goes to stdout)
//   --concurrency <n>             agent calls at once (default 4)
//   --agent-timeout-ms <ms>       kill an agent run after this long (default 0: inferpilot.js's own timeouts)
//   --schema-retries <n>          asks again for an answer that does not match its schema (default 2)
//   --kill-orphans                first stop what runs that are gone left running (agents, servers)
//   --control-file <path>         read at every phase boundary (default with --args-file: <name>.control.json
//                                 next to it): {"stop_after": "<phase>"} moves the stop, "now" stops
//                                 after the phase that is ending, "" runs to the end; a file already
//                                 there when the run starts counts only once it changes
//   --agent-log-dir <dir>         each agent run's prompt, whole output stream and stderr (default with
//                                 --result-file: <name>.agents/ next to it); a run the log's NEXT names
//                                 keeps its own, next to its result file
//   --show-text                   the log also shows what the agents write, not only their tool calls
//   --help                        this usage
//
// With --result-file, <name>.status.json next to it is rewritten every 30 s while the runner lives: its
// state, phase, the agent calls under way (each with its last tool call or text, --show-text or not, and
// when that came: last_at) and when it was updated. An update more than two minutes old means the runner
// is gone.
//
// The log ends with what the result's next says comes after the run (=== NEXT); with --args-file, the
// arguments of each run it names are written next to that file (<name>.again.json, <name>.start-<rank>.json).
// INFERPILOT_CLAUDE_BIN and INFERPILOT_CURSOR_BIN name the CLIs when they are not `claude` and `agent` on PATH.
//
// Every agent, and whatever it starts, inherits INFERPILOT_RUN_TAG, which names this runner. On SIGINT or
// SIGTERM the runner stops them all (servers started with setsid or nohup too, and, for a task with a
// container, those started there with docker exec -e INFERPILOT_RUN_TAG), frees the machine lock
// (scripts/run_lock.py) and writes a result with status "interrupted" (exit 130 or 143). A second signal
// kills them at once instead of waiting for them, and the lock and the result still follow; a third exits
// at once. At its end it stops what its agents left running; a signal then only hurries that. Its first log
// line gives its pid; then it lists what runs that are gone left running, those in the task's container too.
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BACKENDS, cliBackend } from './backends.mjs';
import { createRuntime } from './engine.mjs';
import { TAG, orphans, orphansIn, runTag, signal, stopInContainer, stopPids, tagged } from './procs.mjs';

function parse(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const [k, v] = a.slice(2).split(/=(.*)/s, 2);
      out[k] = v !== undefined ? v : (argv[i + 1] !== undefined && !argv[i + 1].startsWith('--') ? argv[++i] : true);
    } else out._.push(a);
  }
  return out;
}

const FLAGS = ['backend', 'model', 'args', 'args-file', 'result-file', 'concurrency', 'agent-timeout-ms', 'schema-retries', 'kill-orphans', 'control-file', 'agent-log-dir', 'show-text', 'help'];
const usage = (code = 2) => {
  (code ? process.stderr : process.stdout).write('usage: node runtime/run.mjs <script.js> --backend claude|cursor [--model <id>] '
    + '(--args <json> | --args-file <path>) [--result-file <path>] [--concurrency <n>] '
    + '[--agent-timeout-ms <ms>] [--schema-retries <n>] [--kill-orphans] [--control-file <path>] '
    + '[--agent-log-dir <dir>] [--show-text] [--help]\n');
  process.exit(code);
};

const opts = parse(process.argv.slice(2));
if (opts.help) usage(0);
const script = opts._[0];
const unknown = Object.keys(opts).filter((k) => k !== '_' && !FLAGS.includes(k));
if (unknown.length) process.stderr.write(`unknown ${unknown.map((k) => `--${k}`).join(', ')}\n`);
if (unknown.length || !script || !opts.backend || !BACKENDS[opts.backend]) usage();
let args = {};
try {
  if (opts['args-file']) args = JSON.parse(readFileSync(opts['args-file'], 'utf8'));
  else if (typeof opts.args === 'string') args = JSON.parse(opts.args);
} catch (e) {
  process.stderr.write(`invalid arguments JSON: ${e.message}\n`);
  process.exit(2);
}
const int = (v, d) => (v !== undefined && v !== true && Number.isFinite(Number(v)) ? Number(v) : d);
const resultStem = typeof opts['result-file'] === 'string' ? opts['result-file'].replace(/(\.result)?\.json$/, '') : '';
const agentLogDir = typeof opts['agent-log-dir'] === 'string' ? opts['agent-log-dir'] : resultStem ? `${resultStem}.agents` : '';
const backend = cliBackend(opts.backend, { model: typeof opts.model === 'string' ? opts.model : '', logDir: agentLogDir });
const showText = opts['show-text'] === true || opts['show-text'] === 'true';
const controlFile = typeof opts['control-file'] === 'string' ? opts['control-file']
  : typeof opts['args-file'] === 'string' ? `${opts['args-file'].replace(/\.json$/, '')}.control.json` : '';
const startedMs = Date.now();
// A file this run found at its start was written for an earlier one, which a "now" in it already
// stopped: it counts once it changes. Its state, not the clock, tells: file times lag the clock a little.
const fileState = (f) => {
  try {
    const s = statSync(f);
    return `${s.mtimeMs}:${s.size}:${s.ino}`;
  } catch {
    return '';
  }
};
const staleControl = controlFile ? fileState(controlFile) : '';
// A file that holds no JSON object changes nothing; the log names it once for each state it is in.
let badControl = '';
const control = () => {
  try {
    if (!controlFile || !existsSync(controlFile) || (staleControl && fileState(controlFile) === staleControl)) return {};
    const c = JSON.parse(readFileSync(controlFile, 'utf8'));
    if (c && typeof c === 'object' && !Array.isArray(c)) return c;
    throw new Error('not a JSON object');
  } catch (e) {
    const state = fileState(controlFile);
    if (state && state !== badControl) {
      badControl = state;
      rt.log(`!! control file ${controlFile}: ${e.message}; the stop stays as it was`);
    }
    return {};
  }
};
const rt = createRuntime({
  backend,
  concurrency: int(opts.concurrency, 4),
  schemaRetries: int(opts['schema-retries'], 2),
  agentTimeoutMs: int(opts['agent-timeout-ms'], 0),
  onContainer: (box) => lookIn(box),
  control,
  showText,
});
const shq = (s) => (/^[\w./:=@,+-]+$/.test(String(s)) ? String(s) : `'${String(s).replace(/'/g, "'\\''")}'`);

// The steps a result's next names, as the log's last lines: what to ask, each run to start (its arguments in
// a file next to --args-file, with the command that starts it like this one) and what only the user does.
function nextLines(next) {
  const lines = [`=== NEXT: ${next.say} ===`];
  for (const q of next.ask || []) lines.push(`  ask ${q.field}: ${q.question}`);
  const runs = [];
  if (next.rerun_with && Object.keys(next.rerun_with).length) {
    const again = { ...args, ...next.rerun_with };
    if (next.rerun_with.answers) again.answers = { ...(args.answers || {}), ...next.rerun_with.answers };
    runs.push(['again', next.ask ? 'run again with the answers filled in' : 'run again', again]);
  } else if (next.rerun_with) {
    lines.push('  run again: the same command');
  }
  for (const s of next.start || []) runs.push([`start-${s.rank}`, `start #${s.rank} (${s.task})`, s.args]);
  // Each run named gets its own result file and agent logs: this run's are not carried.
  const carried = FLAGS.filter((k) => !['args', 'args-file', 'result-file', 'agent-log-dir', 'show-text', 'help'].includes(k)
    && typeof opts[k] === 'string').map((k) => ` --${k} ${shq(opts[k])}`).join('') + (showText ? ' --show-text' : '');
  const here = relative(process.cwd(), process.argv[1]);
  const runner = `node ${here && !here.startsWith('..') ? here : process.argv[1]} ${script}${carried}`;
  for (const [tag, what, a] of runs) {
    if (typeof opts['args-file'] === 'string') {
      // a run of <name>.again.json that asks again writes <name>.again.json anew, not <name>.again.again.json
      const stem = `${opts['args-file'].replace(/\.json$/, '').replace(/\.again$/, '')}.${tag}`;
      writeFileSync(`${stem}.json`, `${JSON.stringify(a, null, 2)}\n`);
      lines.push(`  ${what}: ${stem}.json`, `    nohup ${runner} --args-file ${stem}.json --result-file ${stem}.result.json < /dev/null > ${stem}.log 2>&1 &`);
    } else {
      lines.push(`  ${what}: ${runner} --args ${shq(JSON.stringify(a))}`);
    }
  }
  for (const u of next.user || []) lines.push(`  you: ${u}`);
  return lines;
}

const tag = runTag();
process.env[TAG] = tag;
rt.log(`runner pid ${process.pid}: kill ${process.pid} stops this run and what it started`);
if (controlFile) rt.log(`control file: ${controlFile} (read at every phase boundary)`);
if (backend.logDir) rt.log(`agent streams: ${backend.logDir}/`);
const mine = () => tagged().filter((p) => p.tag === tag);
const listed = (ps) => ps.slice(0, 20).map((p) => `\n     ${p.pid} ${p.cmd.slice(0, 160)}`).join('');
const releaseLock = () => spawnSync('python3', [join(dirname(fileURLToPath(import.meta.url)), '..', 'scripts', 'run_lock.py'),
  'release', '--run-tag', tag], { stdio: 'ignore', timeout: 30000 });

const statusFile = resultStem ? `${resultStem}.status.json` : '';
let statusBeat = null;
// Written beside it and renamed over it: who reads it never gets half of it. Once the run's end is in it,
// the 30 s rewrite stops, and cannot say running again.
const writeStatus = (state) => {
  if (!statusFile) return;
  if (state !== 'running') clearInterval(statusBeat);
  try {
    writeFileSync(`${statusFile}.tmp`, `${JSON.stringify({ state, pid: process.pid, run_tag: tag, started: new Date(startedMs).toISOString(),
      updated: new Date().toISOString(), phase: rt.current(), agents: rt.live(), summary: rt.summary() }, null, 2)}\n`);
    renameSync(`${statusFile}.tmp`, statusFile);
  } catch { /* a full disk does not stop the run */ }
};
writeStatus('running');
if (statusFile) statusBeat = setInterval(() => writeStatus('running'), 30000).unref();

// The container the task's servers run in (InferPilot on the host), however the script resolved it.
const container = () => (rt.state.container || (typeof args.container === 'string' ? args.container : '')).trim();

// What runs that are gone left running, here or (where) in the task's container: listed, or stopped
// with --kill-orphans.
function reportOrphans(left, where = '') {
  if (left.length && opts['kill-orphans']) {
    rt.log(`stopped ${left.length} process(es) left${where} by runs that are gone:${listed(left)}`);
  } else if (left.length) {
    rt.log(`!! ${left.length} process(es) left${where} by runs that are gone, which may hold GPUs or ports; `
      + `--kill-orphans stops them:${listed(left)}`);
  }
}
reportOrphans(opts['kill-orphans'] ? await stopPids(orphans) : orphans());

// In the task's container their environments are root's, unreadable from here: the container's root looks,
// once per container, at the start for one given in the arguments, else when the script names it, and
// the next agent waits for that.
const looked = new Map();
function lookIn(box) {
  if (!looked.has(box)) {
    looked.set(box, (async () => {
      const left = await orphansIn(box);
      if (left === null) {
        rt.log(`!! could not look in container ${box} for what runs that are gone left running`);
        return;
      }
      if (opts['kill-orphans'] && left.length) {
        const stopped = await Promise.all([...new Set(left.map((p) => p.tag))].map((t) => stopInContainer(box, t)));
        if (stopped.includes(null)) {
          rt.log(`!! could not stop what runs that are gone left running in container ${box}:${listed(left)}`);
          return;
        }
      }
      reportOrphans(left, ` in container ${box}`);
    })());
  }
  return looked.get(box);
}
if (container()) await lookIn(container());

let stopping = false;
let hurried = false;
let ended = false; // the script is done: a signal only hurries what the run's end stops
let hurry = () => {}; // cuts short a sweep of the task's container that waits out its grace
async function interrupted(sig, code) {
  if (hurried) process.exit(code);
  if (stopping || ended) {
    hurried = true;
    rt.log(stopping
      ? `=== ${sig} again: killing them at once, then freeing the lock; once more to exit without that ===`
      : `=== ${sig} after the run's end: killing what its agents left running at once, then freeing the lock; `
        + 'once more to exit without that ===');
    backend.stopAll('SIGKILL');
    signal(mine().map((p) => p.pid), 'SIGKILL');
    hurry();
    return;
  }
  stopping = true;
  const at = rt.current();
  rt.log(`=== ${sig}: stopping the agents (${backend.running}) and what they started; once more to kill at once ===`);
  backend.close();
  const groups = (async () => {
    const deadline = Date.now() + 10000;
    while (backend.running && !hurried && Date.now() < deadline) await new Promise((r) => setTimeout(r, 200));
    backend.stopAll('SIGKILL');
  })();
  await Promise.all([groups, stopPids(mine)]);
  await inContainer();
  releaseLock();
  const result = { status: 'interrupted', signal: sig, stopped_at: at, summary: rt.summary(),
    next: { say: `Interrupted by ${sig}${at ? ` in ${at}` : ''}; its agents and what they started were stopped and the `
      + 'machine lock freed. Run again with the same arguments to start over.' } };
  if (typeof opts['result-file'] === 'string') writeFileSync(opts['result-file'], `${JSON.stringify(result, null, 2)}\n`);
  writeStatus('interrupted');
  rt.log(`=== END (interrupted by ${sig}${at ? ` in ${at}` : ''}): ${rt.summary()} ===`);
  process.exit(code);
}
process.on('SIGINT', () => interrupted('SIGINT', 130));
process.on('SIGTERM', () => interrupted('SIGTERM', 143));

// What the agents left running when the run ends: servers, benchmarks.
async function tidyUp() {
  const still = await stopPids(mine);
  if (still.length) rt.log(`stopped ${still.length} process(es) the run's agents left running:${listed(still)}`);
  await inContainer();
  releaseLock();
}

// A task with a container (InferPilot on the host) started its servers there: stop those too. Hurried,
// at once: a second signal drops a sweep that waits out its grace for one that kills.
async function inContainer() {
  const box = container();
  if (!box) return;
  const wait = new AbortController();
  hurry = () => wait.abort();
  let got = await stopInContainer(box, tag, hurried ? 0 : 10, wait.signal);
  hurry = () => {};
  if (wait.signal.aborted) got = await stopInContainer(box, tag, 0);
  if (got === null) rt.log(`!! could not look for this run's processes in container ${box}: check it before reusing its GPUs`);
  else if (got.length) rt.log(`stopped ${got.length} process(es) of this run in container ${box}: ${got.join(' ')}`);
}

try {
  const result = await rt.runScript(resolve(script), args);
  ended = true;
  if (stopping) await new Promise(() => {});
  await tidyUp();
  rt.log(`=== END: ${rt.summary()} ===`);
  const text = JSON.stringify(result, null, 2);
  if (typeof opts['result-file'] === 'string') writeFileSync(opts['result-file'], `${text}\n`);
  process.stdout.write(`${text}\n`);
  if (result && result.next && result.next.say) nextLines(result.next).forEach((line) => rt.log(line));
  // Last: who sees the run ended finds its result and the arguments of the runs it names.
  writeStatus('ended');
} catch (e) {
  ended = true;
  if (stopping) await new Promise(() => {});
  await tidyUp();
  writeStatus('failed');
  rt.log(`=== END (failed): ${rt.summary()} ===`);
  process.stderr.write(`run failed: ${e && e.stack ? e.stack : e}\n`);
  process.exit(1);
}
