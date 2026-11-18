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
//
// INFERPILOT_CLAUDE_BIN and INFERPILOT_CURSOR_BIN name the CLIs when they are not `claude` and `agent` on PATH.
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { BACKENDS, cliBackend } from './backends.mjs';
import { createRuntime } from './engine.mjs';

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

const usage = () => {
  process.stderr.write('usage: node runtime/run.mjs <script.js> --backend claude|cursor [--model <id>] '
    + '(--args <json> | --args-file <path>) [--result-file <path>] [--concurrency <n>] '
    + '[--agent-timeout-ms <ms>] [--schema-retries <n>]\n');
  process.exit(2);
};

const opts = parse(process.argv.slice(2));
const script = opts._[0];
if (!script || !opts.backend || !BACKENDS[opts.backend]) usage();
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
const rt = createRuntime({
  backend,
  concurrency: int(opts.concurrency, 4),
  schemaRetries: int(opts['schema-retries'], 2),
  agentTimeoutMs: int(opts['agent-timeout-ms'], 0),
});
try {
  const result = await rt.runScript(resolve(script), args);
  rt.log(`=== END: ${rt.summary()} ===`);
  const text = JSON.stringify(result, null, 2);
  if (typeof opts['result-file'] === 'string') writeFileSync(opts['result-file'], `${text}\n`);
  process.stdout.write(`${text}\n`);
} catch (e) {
  rt.log(`=== END (failed): ${rt.summary()} ===`);
  process.stderr.write(`run failed: ${e && e.stack ? e.stack : e}\n`);
  process.exit(1);
}
