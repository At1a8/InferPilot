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
//                                 after the phase that is ending, "" runs to the end; a file older than
//                                 the run is ignored
//   --help                        this usage
//
// The log ends with what the result's next says comes after the run (=== NEXT); with --args-file, the
// arguments of each run it names are written next to that file (<name>.again.json, <name>.start-<rank>.json).
// INFERPILOT_CLAUDE_BIN and INFERPILOT_CURSOR_BIN name the CLIs when they are not `claude` and `agent` on PATH.
//
// Every agent, and whatever it starts, inherits INFERPILOT_RUN_TAG, which names this runner. On SIGINT or
// SIGTERM the runner stops them all (servers started with setsid or nohup too), frees the machine lock
// (scripts/run_lock.py) and writes a result with status "interrupted" (exit 130 or 143); at its end it
// stops what its agents left running. At the start it lists what runs that are gone left running.
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BACKENDS, cliBackend } from './backends.mjs';
import { createRuntime } from './engine.mjs';
import { TAG, orphans, runTag, signal, stopPids, tagged } from './procs.mjs';

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

const FLAGS = ['backend', 'model', 'args', 'args-file', 'result-file', 'concurrency', 'agent-timeout-ms', 'schema-retries', 'kill-orphans', 'control-file', 'help'];
const usage = (code = 2) => {
  (code ? process.stderr : process.stdout).write('usage: node runtime/run.mjs <script.js> --backend claude|cursor [--model <id>] '
    + '(--args <json> | --args-file <path>) [--result-file <path>] [--concurrency <n>] '
    + '[--agent-timeout-ms <ms>] [--schema-retries <n>] [--kill-orphans] [--control-file <path>] [--help]\n');
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
const backend = cliBackend(opts.backend, { model: typeof opts.model === 'string' ? opts.model : '' });
const controlFile = typeof opts['control-file'] === 'string' ? opts['control-file']
  : typeof opts['args-file'] === 'string' ? `${opts['args-file'].replace(/\.json$/, '')}.control.json` : '';
const startedMs = Date.now();
// A file older than this run was written for an earlier one, which a "now" in it already stopped.
const control = () => {
  try {
    if (!controlFile || !existsSync(controlFile) || statSync(controlFile).mtimeMs < startedMs) return {};
    const c = JSON.parse(readFileSync(controlFile, 'utf8'));
    return c && typeof c === 'object' ? c : {};
  } catch {
    return {};
  }
};
const rt = createRuntime({
  backend,
  concurrency: int(opts.concurrency, 4),
  schemaRetries: int(opts['schema-retries'], 2),
  agentTimeoutMs: int(opts['agent-timeout-ms'], 0),
  control,
});
if (controlFile) rt.log(`control file: ${controlFile} (read at every phase boundary)`);
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
  const carried = FLAGS.filter((k) => !['args', 'args-file', 'result-file', 'help'].includes(k) && typeof opts[k] === 'string')
    .map((k) => ` --${k} ${shq(opts[k])}`).join('');
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
const mine = () => tagged().filter((p) => p.tag === tag);
const listed = (ps) => ps.slice(0, 20).map((p) => `\n     ${p.pid} ${p.cmd.slice(0, 160)}`).join('');
const releaseLock = () => spawnSync('python3', [join(dirname(fileURLToPath(import.meta.url)), '..', 'scripts', 'run_lock.py'),
  'release', '--run-tag', tag], { stdio: 'ignore', timeout: 30000 });

const left = orphans();
if (left.length && opts['kill-orphans']) {
  await stopPids(orphans);
  rt.log(`stopped ${left.length} process(es) left by runs that are gone:${listed(left)}`);
} else if (left.length) {
  rt.log(`!! ${left.length} process(es) left by runs that are gone, which may hold GPUs or ports; `
    + `--kill-orphans stops them:${listed(left)}`);
}

let stopping = false;
async function interrupted(sig, code) {
  if (stopping) {
    backend.stopAll('SIGKILL');
    signal(mine().map((p) => p.pid), 'SIGKILL');
    process.exit(code);
  }
  stopping = true;
  const at = rt.current();
  rt.log(`=== ${sig}: stopping the agents (${backend.running}) and what they started; once more to kill at once ===`);
  backend.close();
  signal(mine().map((p) => p.pid), 'SIGTERM');
  const deadline = Date.now() + 10000;
  while ((backend.running || mine().length) && Date.now() < deadline) await new Promise((r) => setTimeout(r, 200));
  backend.stopAll('SIGKILL');
  signal(mine().map((p) => p.pid), 'SIGKILL');
  releaseLock();
  const result = { status: 'interrupted', signal: sig, at, summary: rt.summary(),
    next: { say: `Interrupted by ${sig}${at ? ` in ${at}` : ''}; its agents and what they started were stopped and the `
      + 'machine lock freed. Run again with the same arguments to start over.' } };
  if (typeof opts['result-file'] === 'string') writeFileSync(opts['result-file'], `${JSON.stringify(result, null, 2)}\n`);
  rt.log(`=== END (interrupted by ${sig}${at ? ` in ${at}` : ''}): ${rt.summary()} ===`);
  process.exit(code);
}
process.on('SIGINT', () => interrupted('SIGINT', 130));
process.on('SIGTERM', () => interrupted('SIGTERM', 143));

// What the agents left running when the run ends: servers, benchmarks.
async function tidyUp() {
  const still = mine();
  if (still.length) {
    await stopPids(mine);
    rt.log(`stopped ${still.length} process(es) the run's agents left running:${listed(still)}`);
  }
  releaseLock();
}

try {
  const result = await rt.runScript(resolve(script), args);
  if (stopping) await new Promise(() => {});
  await tidyUp();
  rt.log(`=== END: ${rt.summary()} ===`);
  const text = JSON.stringify(result, null, 2);
  if (typeof opts['result-file'] === 'string') writeFileSync(opts['result-file'], `${text}\n`);
  process.stdout.write(`${text}\n`);
  if (result && result.next && result.next.say) nextLines(result.next).forEach((line) => rt.log(line));
} catch (e) {
  if (stopping) await new Promise(() => {});
  await tidyUp();
  rt.log(`=== END (failed): ${rt.summary()} ===`);
  process.stderr.write(`run failed: ${e && e.stack ? e.stack : e}\n`);
  process.exit(1);
}
