// InferPilot's workflow runner: runs inferpilot.js outside Claude Code, with the globals Claude Code's
// Workflow tool gives a script (agent, parallel, pipeline, phase, log, workflow, args, budget). Every
// agent() call goes to a backend: { name, async runAgent({prompt, label, cwd, env, model, timeoutMs}) -> {text} }.
import { readFile } from 'node:fs/promises';
import { extractJson, outputContract, validate } from './schema.mjs';

const MAX_AGENTS = 1000; // a backstop for a script that loops on agent()

class Slots {
  constructor(n) {
    this.free = Math.max(1, n);
    this.waiting = [];
  }

  async take() {
    if (this.free > 0) {
      this.free--;
      return;
    }
    await new Promise((resolve) => this.waiting.push(resolve));
  }

  give() {
    const next = this.waiting.shift();
    if (next) next();
    else this.free++;
  }
}

// The script's `export` keywords go: its body runs as the body of an async function, whose return
// value is the workflow's result.
export function scriptBody(src) {
  return String(src)
    .replace(/^\uFEFF/, '')
    .replace(/^[ \t]*export\s+(?=(?:const|let|var|function|class|async|default)\b)/gm, '');
}

const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;

export function createRuntime({
  backend,
  concurrency = 4,
  schemaRetries = 2,
  agentTimeoutMs = 0,
  log = (msg) => process.stderr.write(`[${new Date().toISOString().slice(11, 19)}] ${msg}\n`),
} = {}) {
  if (!backend || typeof backend.runAgent !== 'function') throw new Error('createRuntime: backend.runAgent is required');
  const slots = new Slots(concurrency);
  const state = { agents: 0, schemaFailures: 0 };
  let current = '';

  const phase = (title) => {
    current = title;
    log(`=== PHASE: ${title} ===`);
  };

  // With a schema, the answer is the validated JSON object; a reply that does not parse or validate is
  // asked for again, and after the last attempt the error is thrown to the script.
  async function agent(prompt, opts = {}) {
    if (++state.agents > MAX_AGENTS) throw new Error(`more than ${MAX_AGENTS} agent calls`);
    const label = opts.label || 'agent';
    const where = opts.phase || current;
    const full = opts.schema ? prompt + outputContract(opts.schema) : prompt;
    await slots.take();
    try {
      log(`  -> [${where}] ${label} (${backend.name})`);
      let last;
      for (let attempt = 1; attempt <= (opts.schema ? schemaRetries + 1 : 1); attempt++) {
        const { text } = await backend.runAgent({
          prompt: full, label, cwd: opts.cwd, env: opts.env, model: opts.model, timeoutMs: agentTimeoutMs,
        });
        if (!opts.schema) return text;
        try {
          const value = extractJson(text);
          const errors = validate(value, opts.schema);
          if (errors.length) throw new Error(`schema: ${errors.slice(0, 3).join('; ')}`);
          return value;
        } catch (e) {
          last = e;
          state.schemaFailures++;
          log(`  !! [${where}] ${label}: unusable answer (attempt ${attempt}): ${e.message}`);
        }
      }
      throw last;
    } finally {
      slots.give();
    }
  }

  // A thunk that throws resolves to null, as in the Workflow tool.
  const parallel = (thunks) => Promise.all((thunks || []).map((t) => Promise.resolve().then(t).catch((e) => {
    log(`  xx parallel task failed: ${String(e && e.message ? e.message : e).slice(0, 200)}`);
    return null;
  })));

  const pipeline = (items, ...stages) => Promise.all((items || []).map(async (item, i) => {
    let value = item;
    for (const stage of stages) {
      try {
        value = await stage(value, item, i);
      } catch (e) {
        log(`  xx pipeline item ${i} failed: ${String(e && e.message ? e.message : e).slice(0, 200)}`);
        return null;
      }
    }
    return value;
  }));

  const workflow = async () => {
    throw new Error('workflow(): nested workflows are not supported by this runner');
  };

  const budget = { total: null, spent: () => 0, remaining: () => Infinity };

  async function runScript(path, args = {}) {
    let fn;
    try {
      fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget',
        scriptBody(await readFile(path, 'utf8')));
    } catch (e) {
      throw new Error(`cannot compile ${path}: ${e.message}`);
    }
    return fn(agent, parallel, pipeline, phase, log, workflow, args, budget);
  }

  return { agent, parallel, pipeline, phase, log, runScript, state };
}
