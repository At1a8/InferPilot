// Self-test of runtime/: the schema checks, the engine's globals, the command-line backends (on fake
// CLIs written to a temporary directory) and runtime/run.mjs end to end. No network, no model.
//
//   node tests/runtime_selftest.mjs
import path from 'node:path';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdtempSync, readFileSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { extractJson, validate } from '../runtime/schema.mjs';
import { createRuntime, fmtTokens, scriptBody } from '../runtime/engine.mjs';
import { BACKENDS, cliBackend } from '../runtime/backends.mjs';

const INFERPILOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TMP = mkdtempSync(path.join(tmpdir(), 'inferpilot-runtime-'));
const quiet = () => {};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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
    // 128 KiB: one more byte than Linux takes in a single argument.
    const fileCli = fakeCli('file.sh', 'for a; do last=$a; done; f=${last#*instructions are in }; f=${f%%. Read*}; echo "file=$f size=$(wc -c < "$f")"');
    const long = await cliBackend('fake', { preset: preset(fileCli, 'arg') }).runAgent({ prompt: 'z'.repeat(128 * 1024), label: 'big/role' });
    const file = (/file=(\S+)/.exec(long.text) || [])[1];
    check(file && long.text.includes(`size=${128 * 1024}`), `a long prompt goes in a file the CLI reads: ${long.text.slice(0, 160)}`);
    check(file && !existsSync(file.replace(/\/[^/]+$/, '')), `and its directory is gone once the CLI is done: ${file}`);
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
  ['backends: stream-json shows each tool call as it starts; the answer comes from the result event', async () => {
    const answer = JSON.stringify({ type: 'result', subtype: 'success', is_error: false, result: '```json\n{"ok": true}\n```' });
    const claudeCli = fakeCli('claude-stream.sh', `cat >/dev/null
cat <<'EOF'
{"type":"system","subtype":"init"}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"python3  scripts/scan_ci.py\\n --out ci.json"}}]}}
EOF
sleep 0.4
cat <<'EOF'
{"type":"user","message":{"content":[]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{"file_path":"/x/ci.md"}},{"type":"text","text":"almost"}]}}
${answer}
EOF`);
    const seen = [];
    const t0 = Date.now();
    const out = await cliBackend('fake', { preset: preset(claudeCli, 'stdin', { events: BACKENDS.claude.events }) })
      .runAgent({ prompt: 'p', label: 'c', onProgress: (m) => seen.push([m, Date.now() - t0]) });
    const done = Date.now() - t0;
    check(seen.map((s) => s[0]).join('|') === 'Bash: python3 scripts/scan_ci.py --out ci.json|Read: /x/ci.md', `claude tool calls: ${seen}`);
    check(seen[0][1] < done - 250, `the first tool call is shown while the agent still runs (${seen[0][1]} ms of ${done} ms)`);
    check(extractJson(out.text).ok === true, `claude answer: ${out.text}`);

    const cursorCli = fakeCli('cursor-stream.sh', `cat <<'EOF'
{"type":"system","subtype":"init","model":"x"}
{"type":"thinking","subtype":"delta","text":"hmm"}
{"type":"tool_call","subtype":"started","tool_call":{"shellToolCall":{"args":{"command":"echo hi"},"description":"Run echo hi"}}}
{"type":"tool_call","subtype":"completed","tool_call":{"shellToolCall":{"args":{"command":"echo hi"},"result":{}}}}
{"type":"tool_call","subtype":"started","tool_call":{"readToolCall":{"args":{"path":"/etc/hostname"}}}}
{"type":"tool_call","subtype":"started","tool_call":{"editToolCall":{"args":{"path":"/tmp/a.txt","streamContent":"hi"}}}}
${JSON.stringify({ type: 'result', subtype: 'success', is_error: false, result: 'x'.repeat(300000) })}
EOF`);
    const lines = [];
    const big = await cliBackend('fake', { preset: preset(cursorCli, 'arg', { events: BACKENDS.cursor.events }) })
      .runAgent({ prompt: 'p', label: 'k', onProgress: (m) => lines.push(m) });
    check(lines.join('|') === 'shell: echo hi|read: /etc/hostname|edit: /tmp/a.txt', `cursor tool calls: ${lines}`);
    check(big.text.length === 300000, `a result line longer than a pipe read: ${big.text.length}`);

    const plain = await cliBackend('fake', { preset: preset(fakeCli('plain.sh', OK_JSON), 'arg', { events: BACKENDS.cursor.events }) })
      .runAgent({ prompt: 'p', label: 'plain' });
    check(extractJson(plain.text).ok === true && plain.text.includes('working'), 'no result event: the plain text is the answer');
    const erred = fakeCli('erred.sh', `echo '${JSON.stringify({ type: 'result', subtype: 'error', is_error: true, result: 'quota exceeded' })}'`);
    await rejects(cliBackend('fake', { preset: preset(erred, 'arg', { events: BACKENDS.cursor.events }) }).runAgent({ prompt: 'p', label: 'e' }),
      /ended in an error: quota exceeded/, 'a result that is an error');
  }],
  ['backends: the tokens of a result event, under either CLI\'s names', async () => {
    const claudeUsage = { type: 'result', is_error: false, result: 'a',
      usage: { input_tokens: 10, output_tokens: 200, cache_read_input_tokens: 3000, cache_creation_input_tokens: 400 } };
    const cursorUsage = { type: 'result', is_error: false, result: 'b',
      usage: { inputTokens: 6, outputTokens: 295, cacheReadTokens: 21041, cacheWriteTokens: 4746 } };
    const c = await cliBackend('fake', { preset: preset(fakeCli('cu.sh', `echo '${JSON.stringify(claudeUsage)}'`), 'arg',
      { events: BACKENDS.claude.events }) }).runAgent({ prompt: 'p' });
    check(JSON.stringify(c.usage) === JSON.stringify({ input: 10, output: 200, cache: 3400, total: 3610 }), `claude usage ${JSON.stringify(c.usage)}`);
    const k = await cliBackend('fake', { preset: preset(fakeCli('ku.sh', `echo '${JSON.stringify(cursorUsage)}'`), 'arg',
      { events: BACKENDS.cursor.events }) }).runAgent({ prompt: 'p' });
    check(k.usage.total === 26088 && k.usage.output === 295, `cursor usage ${JSON.stringify(k.usage)}`);
    const none = await cliBackend('fake', { preset: preset(fakeCli('nu.sh', OK_JSON), 'arg', { events: BACKENDS.cursor.events }) })
      .runAgent({ prompt: 'p' });
    check(none.usage === null, 'plain text: no tokens to report');
  }],
  ['engine: phases numbered along the route the script logs, tokens per call and for the run', async () => {
    const logs = [];
    let n = 0;
    const backend = { name: 'fake', async runAgent() {
      n++;
      return { text: '```json\n{"ok": true}\n```', usage: { input: 0, output: 0, cache: 0, total: n === 1 ? 410000 : 1250000 } };
    } };
    const rt = createRuntime({ backend, quietMs: 0, log: (m) => logs.push(m) });
    rt.log('[inferpilot] Route: Collect → Dispatch → Report');
    rt.phase('Collect');
    await rt.agent('q', { label: 'scout:collect', schema: SCHEMA });
    rt.phase('Dispatch');
    await rt.agent('q', { label: 'dispatcher:dispatch', schema: SCHEMA });
    rt.phase('Bisect');
    check(logs.includes('=== PHASE 1/3 [#--]: Collect ==='), `the first phase: ${logs}`);
    check(logs.includes('  <- [Collect] scout:collect  done in 0 min, 410k tokens (run total 410k)'), `tokens of a call: ${logs}`);
    check(logs.includes('=== PHASE 2/3 [##-]: Dispatch ===  so far: 1 agent call, 410k tokens, 0 min'), `so far: ${logs}`);
    check(logs.includes('  <- [Dispatch] dispatcher:dispatch  done in 0 min, 1.3M tokens (run total 1.7M)'), `the run total: ${logs}`);
    check(logs.some((l) => l.startsWith('=== PHASE (off the route): Bisect ===')), `a phase off the route: ${logs}`);
    check(rt.summary() === '2/3 phases of the route, 2 agent calls, 1.7M tokens, 0 min', `summary: ${rt.summary()}`);
    const plain = createRuntime({ backend: { name: 'fake', runAgent: async () => ({ text: 'x' }) }, quietMs: 0, log: quiet });
    plain.phase('One');
    await plain.agent('q', { label: 'a' });
    check(plain.summary() === '1 phase, 1 agent call, 0 min', `no route, no tokens: ${plain.summary()}`);
    const loose = [];
    const rt3 = createRuntime({ backend, quietMs: 0, log: (m) => loose.push(m) });
    rt3.log('Route: A → B');
    rt3.log('[inferpilot] intake: ready — the user wrote "Route: X → Y" in the request');
    rt3.phase('A');
    check(loose.includes('=== PHASE 1/2 [#-]: A ==='), `only a whole Route line sets the route: ${loose}`);
    check([[999, '999'], [1000, '1k'], [1049, '1k'], [26088, '26.1k'], [999949, '999.9k'], [999950, '1M'], [1700000, '1.7M']]
      .every(([n, s]) => fmtTokens(n) === s), `token counts: ${[999, 1000, 999949, 999950].map(fmtTokens)}`);
  }],
  ['engine: the tool calls, a still-working line after a quiet spell, and the end of each call', async () => {
    const logs = [];
    const backend = { name: 'fake', async runAgent({ onProgress }) {
      onProgress('shell: python3 scripts/scan_ci.py');
      await new Promise((r) => setTimeout(r, 200));
      return { text: '```json\n{"ok": true}\n```' };
    } };
    const rt = createRuntime({ backend, quietMs: 50, log: (m) => logs.push(m) });
    rt.phase('Collect');
    check((await rt.agent('q', { label: 'scout:collect', schema: SCHEMA })).ok === true, 'the answer');
    check(logs.includes('     · [Collect] scout:collect  shell: python3 scripts/scan_ci.py'), `a tool call: ${logs}`);
    check(logs.some((l) => /^ {5}· \[Collect\] scout:collect {2}still working, 0 min$/.test(l)), `still working: ${logs}`);
    check(logs[logs.length - 1] === '  <- [Collect] scout:collect  done in 0 min', `the end: ${logs[logs.length - 1]}`);
    const quietLogs = [];
    const rt2 = createRuntime({ backend, quietMs: 0, log: (m) => quietLogs.push(m) });
    await rt2.agent('q', { label: 'x' });
    check(!quietLogs.some((l) => l.includes('still working')), 'quietMs 0: no still-working line');
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
  ['run.mjs: the control file next to --args-file is read when the script asks; one older than the run is ignored', async () => {
    const script = path.join(TMP, 'control.js');
    writeFileSync(script, `const before = control();\nawait agent('go', { label: 'writes' });\nreturn { before, after: control() };`);
    const argsFile = path.join(TMP, 'ctl.json');
    writeFileSync(argsFile, '{}');
    const controlFile = path.join(TMP, 'ctl.control.json');
    writeFileSync(controlFile, '{"stop_after": "now"}');
    utimesSync(controlFile, new Date(Date.now() - 3600e3), new Date(Date.now() - 3600e3));
    const out = execFileSync(process.execPath, [path.join(INFERPILOT, 'runtime', 'run.mjs'), script, '--backend', 'cursor',
      '--args-file', argsFile], { env: { ...process.env, INFERPILOT_CURSOR_BIN: fakeCli('ctl.sh',
      `echo '{"stop_after": "Verify"}' > ${controlFile}; echo done`) }, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    const r = JSON.parse(out);
    check(JSON.stringify(r.before) === '{}', `an earlier run's file is ignored: ${JSON.stringify(r.before)}`);
    check(r.after.stop_after === 'Verify', `one written during the run is read: ${JSON.stringify(r.after)}`);
  }],
  ['run.mjs: SIGTERM stops the agents and what they started with setsid, frees the lock, writes an interrupted result', async () => {
    if (!existsSync('/proc/self/environ')) return;
    const script = path.join(TMP, 'stop.js');
    writeFileSync(script, `phase('Long');\nawait agent('go', { label: 'long' });\nreturn { status: 'done' };`);
    const lock = path.join(TMP, 'stop.lock');
    const runDir = path.join(TMP, 'stop-run');
    const cli = fakeCli('long.sh', `python3 ${path.join(INFERPILOT, 'scripts', 'run_lock.py')} acquire --run-dir ${runDir} >/dev/null
setsid sleep 41.25 > /dev/null 2>&1 < /dev/null &
sleep 42.25 & wait`);
    const resultFile = path.join(TMP, 'stop.result.json');
    const runner = spawn(process.execPath, [path.join(INFERPILOT, 'runtime', 'run.mjs'), script, '--backend', 'cursor',
      '--args', '{}', '--result-file', resultFile], { env: { ...process.env, INFERPILOT_CURSOR_BIN: cli, INFERPILOT_LOCK_FILE: lock },
      stdio: ['ignore', 'ignore', 'pipe'] });
    let log = '';
    runner.stderr.on('data', (d) => { log += d; });
    const running = (what) => spawnSync('pgrep', ['-f', what]).status === 0;
    for (let i = 0; i < 100 && !(running('sleep 41.25') && running('sleep 42.25') && existsSync(lock)); i++) await sleep(100);
    check(running('sleep 41.25') && running('sleep 42.25'), 'the agent and its setsid child started');
    runner.kill('SIGTERM');
    const code = await new Promise((r) => runner.on('exit', (c) => r(c)));
    check(code === 143, `exit ${code}`);
    check(!running('sleep 41.25'), 'the setsid child of the agent was stopped');
    check(!running('sleep 42.25'), 'the agent was stopped');
    const result = JSON.parse(readFileSync(resultFile, 'utf8'));
    check(result.status === 'interrupted' && result.signal === 'SIGTERM' && result.at === 'Long', `result ${JSON.stringify(result)}`);
    const st = JSON.parse(execFileSync('python3', [path.join(INFERPILOT, 'scripts', 'run_lock.py'), '--lock', lock, 'status'], { encoding: 'utf8' }));
    check(st.status === 'free', `the lock is ${st.status}`);
    check(/=== END \(interrupted by SIGTERM in Long\)/.test(log), 'the log ends with the interruption');
  }],
  ['run.mjs: what the agents left running is stopped at the end; orphans are listed, and stopped with --kill-orphans', async () => {
    if (!existsSync('/proc/self/environ')) return;
    const script = path.join(TMP, 'tidy.js');
    writeFileSync(script, `phase('Only');\nreturn { status: (await agent('go', { label: 'only', schema: ${JSON.stringify(SCHEMA)} })).ok };`);
    const run = (...extra) => spawnSync(process.execPath, [path.join(INFERPILOT, 'runtime', 'run.mjs'), script, '--backend', 'cursor',
      '--args', '{}', ...extra], { env: { ...process.env, INFERPILOT_CURSOR_BIN: fakeCli('tidy.sh',
      `setsid sleep 43.25 > /dev/null 2>&1 < /dev/null &\n${OK_JSON}`), INFERPILOT_LOCK_FILE: path.join(TMP, 'tidy.lock') }, encoding: 'utf8' });
    const running = (what) => spawnSync('pgrep', ['-f', what]).status === 0;
    const r = run();
    check(r.status === 0 && /stopped 1 process\(es\) the run's agents left running/.test(r.stderr), `end ${r.stderr.slice(-300)}`);
    check(!running('sleep 43.25'), 'the leftover was stopped');
    const orphan = spawn('sleep', ['44.25'], { env: { ...process.env, INFERPILOT_RUN_TAG: '999999999-1' }, detached: true, stdio: 'ignore' });
    orphan.unref();
    await sleep(200);
    const warned = run();
    check(/1 process\(es\) left by runs that are gone/.test(warned.stderr) && running('sleep 44.25'), 'an orphan is listed and left alone');
    const killed = run('--kill-orphans');
    check(/stopped 1 process\(es\) left by runs that are gone/.test(killed.stderr) && !running('sleep 44.25'), 'and stopped on request');
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
