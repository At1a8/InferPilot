// Self-test of runtime/: the schema checks, the engine's globals, the command-line backends (on fake
// CLIs written to a temporary directory) and runtime/run.mjs end to end. No network, no model.
//
//   node tests/runtime_selftest.mjs
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { extractJson, validate } from '../runtime/schema.mjs';
import { createRuntime, scriptBody } from '../runtime/engine.mjs';
import { BACKENDS, cliBackend } from '../runtime/backends.mjs';

const INFERPILOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TMP = mkdtempSync(path.join(tmpdir(), 'inferpilot-runtime-'));
const quiet = () => {};

let failed = 0;
let passed = 0;
function check(cond, msg) {
  if (cond) passed++;
  else { failed++; console.error(`  FAIL: ${msg}`); }
}
async function rejects(p, re, msg) {
  try {
    await p;
    check(false, `${msg}: did not throw`);
  } catch (e) {
    check(re.test(String(e && e.message)), `${msg}: ${e && e.message}`);
  }
}
function fakeCli(name, body) {
  const file = path.join(TMP, name);
  writeFileSync(file, `#!/bin/bash\n${body}\n`);
  chmodSync(file, 0o755);
  return file;
}
const preset = (bin, promptVia, extra = {}) => ({ bin, args: [], promptVia, modelFlag: '--model', env: {}, ...extra });
const OK_JSON = 'echo "working"; printf \'```json\\n{"ok": true}\\n```\\n\'';
const SCHEMA = { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] };

const scenarios = [
  ['schema: the last JSON object of a reply, fenced or bare', async () => {
    check(extractJson('text\n```json\n{"a": 1}\n```\nmore\n```json\n{"a": 2}\n```').a === 2, 'the last fenced block wins');
    check(extractJson('```json\nnot json\n```\n```json\n{"a": 3}\n```').a === 3, 'a fenced block that fails to parse is skipped');
    check(extractJson('done: {"a": {"b": [1, 2]}} bye').a.b[1] === 2, 'a bare object with nesting');
    let threw = false;
    try {
      extractJson('no object here');
    } catch {
      threw = true;
    }
    check(threw, 'no object throws');
  }],
  ['schema: type, enum, required, properties and items are checked', async () => {
    const s = {
      type: 'object', required: ['status', 'list'],
      properties: { status: { type: 'string', enum: ['ok', 'no'] }, n: { type: 'number' }, x: { type: ['string', 'null'] },
        list: { type: 'array', items: { type: 'object', required: ['id'], properties: { id: { type: 'string' } } } } },
    };
    check(validate({ status: 'ok', n: 3, x: null, list: [{ id: 'a' }], extra: 1 }, s).length === 0, 'a valid value, extra fields allowed, integer as number');
    const errs = validate({ status: 'maybe', n: '3', list: [{}] }, s);
    check(errs.some((e) => e.startsWith('status:')) && errs.some((e) => e.startsWith('n:')) && errs.some((e) => e.includes('[0].id: missing')),
      `errors: ${errs.join('; ')}`);
    check(validate('x', { type: 'object' }).length === 1 && validate({}, { type: 'object', required: ['a'] })[0] === 'a: missing', 'top level');
  }],
  ['engine: a script gets the Workflow globals; a bad structured answer is asked for again', async () => {
    const script = path.join(TMP, 'wf.js');
    writeFileSync(script, `export const meta = { name: 'wf', description: 'test', phases: [] };
phase('One');
log('hello');
const a = await agent('first', { label: 'a', schema: ${JSON.stringify(SCHEMA)} });
const b = await agent('plain', { label: 'b', model: 'm2' });
const par = await parallel([() => agent('x', { label: 'p1' }), () => { throw new Error('boom'); }]);
let nested = '';
try { await workflow({ scriptPath: '/x.js' }, {}); } catch (e) { nested = e.message; }
return { a, b, par, nested, n: args.n, budget: budget.remaining() };`);
    const logs = [];
    const calls = [];
    const backend = {
      name: 'fake',
      async runAgent({ prompt, label, model }) {
        calls.push({ label, model, prompt });
        if (label === 'a') return { text: calls.filter((c) => c.label === 'a').length === 1 ? 'no json' : '```json\n{"ok": true}\n```' };
        return { text: `text-${label}` };
      },
    };
    const rt = createRuntime({ backend, log: (m) => logs.push(m) });
    const r = await rt.runScript(script, { n: 7 });
    check(r.a.ok === true && calls.filter((c) => c.label === 'a').length === 2, 'the schema answer, after one retry');
    check(calls[0].prompt.includes('## Output contract') && calls[0].prompt.includes('"required"'), 'the schema reaches the prompt');
    check(r.b === 'text-b' && calls.find((c) => c.label === 'b').model === 'm2', 'a plain call returns text; the model reaches the backend');
    check(r.par[0] === 'text-p1' && r.par[1] === null, 'parallel: a failing thunk is null');
    check(/not supported/.test(r.nested) && r.n === 7 && r.budget === Infinity, 'workflow() refuses; args and budget');
    check(logs.includes('=== PHASE: One ===') && logs.includes('hello'), `phase and log lines: ${logs.slice(0, 3)}`);
    check(scriptBody('export const meta = {};\n  export async function f() {}\nconst s = "export x";').startsWith('const meta'), 'export is stripped');
  }],
  ['engine: retries run out; concurrency is bounded', async () => {
    const rt = createRuntime({ backend: { name: 'fake', runAgent: async () => ({ text: 'never json' }) }, schemaRetries: 1, log: quiet });
    await rejects(rt.agent('q', { schema: SCHEMA }), /no JSON object/, 'an answer that never parses throws');
    let now = 0;
    let most = 0;
    const slow = { name: 'slow', async runAgent() {
      most = Math.max(most, ++now);
      await new Promise((r) => setTimeout(r, 40));
      now--;
      return { text: 'x' };
    } };
    const rt2 = createRuntime({ backend: slow, concurrency: 2, log: quiet });
    await Promise.all(Array.from({ length: 6 }, (_, i) => rt2.agent('p', { label: `c${i}` })));
    check(most === 2, `at most 2 at once: ${most}`);
  }],
  ['backends: the prompt on stdin, or as the last argument; the model flag', async () => {
    const stdinCli = fakeCli('stdin.sh', `prompt=$(cat); echo "args=$*"; echo "len=\${#prompt}"; ${OK_JSON}`);
    const out = await cliBackend('fake', { preset: preset(stdinCli, 'stdin'), model: 'm1' }).runAgent({ prompt: 'hello world', label: 'x' });
    check(out.text.includes('len=11') && out.text.includes('args=--model m1'), `stdin: ${out.text}`);
    const argCli = fakeCli('arg.sh', 'for a; do last=$a; done; timeout 5 cat >/dev/null && echo stdin-closed; echo "last=$last"');
    const b = cliBackend('fake', { preset: preset(argCli, 'arg') });
    const short = await b.runAgent({ prompt: 'do it', label: 'y' });
    check(short.text.includes('stdin-closed') && short.text.includes('last=do it'), `arg: ${short.text}`);
    const long = await b.runAgent({ prompt: 'z'.repeat(150 * 1024), label: 'big/role' });
    const file = (/in (\S+)\. Read/.exec(long.text) || [])[1];
    check(file && existsSync(file) && readFileSync(file, 'utf8').length === 150 * 1024, `a long prompt goes in a file: ${long.text.slice(0, 160)}`);
    const rt = createRuntime({ backend: cliBackend('fake', { preset: preset(stdinCli, 'stdin') }), log: quiet });
    check((await rt.agent('q', { schema: SCHEMA })).ok === true, 'a structured answer through a real process');
  }],
  ['backends: a failure, a missing CLI and a timeout that stops the process group', async () => {
    const fail = cliBackend('fake', { preset: preset(fakeCli('fail.sh', 'echo oops >&2; exit 3'), 'stdin') });
    await rejects(fail.runAgent({ prompt: 'p', label: 'f' }), /exited 3: oops/, 'non-zero exit');
    const missing = cliBackend('fake', { preset: preset(path.join(TMP, 'no-such-cli'), 'stdin') });
    await rejects(missing.runAgent({ prompt: 'p' }), /cannot run/, 'missing CLI');
    const sleeper = cliBackend('fake', { preset: preset(fakeCli('sleep.sh', 'sleep 37.25 & wait'), 'arg') });
    await rejects(sleeper.runAgent({ prompt: 'p', label: 's', timeoutMs: 300 }), /timed out/, 'timeout');
    await new Promise((r) => setTimeout(r, 300));
    const left = spawnSync('pgrep', ['-f', 'sleep 37.25']).status === 0;
    check(!left, 'the agent\'s own child process is stopped too');
    check(BACKENDS.cursor.args.includes('--force') && BACKENDS.cursor.promptVia === 'arg' && BACKENDS.claude.promptVia === 'stdin',
      'the presets');
  }],
  ['run.mjs: a workflow end to end on a CLI backend, the result on stdout and in a file', async () => {
    const script = path.join(TMP, 'e2e.js');
    writeFileSync(script, `export const meta = { name: 'e2e', description: 'test', phases: [] };
phase('Only');
const r = await agent('go', { label: 'only', schema: ${JSON.stringify(SCHEMA)} });
return { status: r.ok ? 'done' : 'no', n: args.n };`);
    const resultFile = path.join(TMP, 'result.json');
    const out = execFileSync(process.execPath, [path.join(INFERPILOT, 'runtime', 'run.mjs'), script, '--backend', 'cursor',
      '--args', '{"n": 5}', '--result-file', resultFile],
    { env: { ...process.env, INFERPILOT_CURSOR_BIN: fakeCli('cursor.sh', OK_JSON) }, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    const r = JSON.parse(out);
    check(r.status === 'done' && r.n === 5 && JSON.parse(readFileSync(resultFile, 'utf8')).n === 5, `result ${out}`);
    const bad = spawnSync(process.execPath, [path.join(INFERPILOT, 'runtime', 'run.mjs'), script, '--backend', 'nope'], { encoding: 'utf8' });
    check(bad.status === 2 && bad.stderr.includes('usage:'), 'an unknown backend prints the usage');
  }],
];

for (const [name, fn] of scenarios) {
  const before = failed;
  try {
    await fn();
  } catch (e) {
    failed++;
    console.error(`  FAIL: threw ${e && e.stack ? e.stack : e}`);
  }
  console.log(`${failed === before ? 'ok  ' : 'FAIL'} ${name}`);
}
console.log(`\n${passed} checks passed, ${failed} failed.`);
process.exit(failed ? 1 : 0);
