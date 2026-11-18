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

export const fmtTokens = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : `${n}`)
  .replace(/\.0(?=[kM])/, '');
export const fmtMinutes = (ms) => {
  const m = Math.round(ms / 60000);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`;
};

// quietMs: an agent call that showed nothing for that long gets a still-working line (0: never).
export function createRuntime({
  backend,
  concurrency = 4,
  schemaRetries = 2,
  agentTimeoutMs = 0,
  quietMs = 5 * 60 * 1000,
  log = (msg) => process.stderr.write(`[${new Date().toISOString().slice(11, 19)}] ${msg}\n`),
} = {}) {
  if (!backend || typeof backend.runAgent !== 'function') throw new Error('createRuntime: backend.runAgent is required');
  const slots = new Slots(concurrency);
  const state = { agents: 0, schemaFailures: 0, done: 0, tokens: 0, tokensKnown: false };
  const runStarted = Date.now();
  let current = '';
  // The phases the script says the run passes ("Route: A → B → C" in a log line), for x/y in each phase line.
  let route = [];
  const entered = new Set();

  const soFar = () => `${state.done} agent call${state.done === 1 ? '' : 's'}`
    + `${state.tokensKnown ? `, ${fmtTokens(state.tokens)} tokens` : ''}, ${fmtMinutes(Date.now() - runStarted)}`;

  const phase = (title) => {
    current = title;
    entered.add(title);
    const at = route.indexOf(title);
    const head = at >= 0
      ? `PHASE ${at + 1}/${route.length} [${'#'.repeat(at + 1)}${'-'.repeat(route.length - at - 1)}]: ${title}`
      : `PHASE${route.length ? ' (off the route)' : ''}: ${title}`;
    log(`=== ${head} ===${state.done ? `  so far: ${soFar()}` : ''}`);
  };

  const scriptLog = (msg) => {
    const m = /\bRoute: (.+)$/.exec(String(msg));
    if (m) route = m[1].split('→').map((s) => s.trim()).filter(Boolean);
    log(msg);
  };

  // One line for the end of a run: how far along its route it got, and what it took.
  const summary = () => {
    const phases = route.length ? `${route.filter((t) => entered.has(t)).length}/${route.length} phases of the route`
      : `${entered.size} phase${entered.size === 1 ? '' : 's'}`;
    return `${phases}, ${soFar()}`;
  };

  // With a schema, the answer is the validated JSON object; a reply that does not parse or validate is
  // asked for again, and after the last attempt the error is thrown to the script.
  async function agent(prompt, opts = {}) {
    if (++state.agents > MAX_AGENTS) throw new Error(`more than ${MAX_AGENTS} agent calls`);
    const label = opts.label || 'agent';
    const where = opts.phase || current;
    const full = opts.schema ? prompt + outputContract(opts.schema) : prompt;
    await slots.take();
    const started = Date.now();
    const minutes = () => Math.round((Date.now() - started) / 60000);
    let shown = started;
    const show = (msg) => {
      shown = Date.now();
      log(`     · [${where}] ${label}  ${msg}`);
    };
    const beat = quietMs > 0 ? setInterval(() => {
      if (Date.now() - shown >= quietMs) show(`still working, ${minutes()} min`);
    }, Math.min(quietMs, 60000)) : null;
    let tokens = null;
    const done = () => {
      state.done++;
      log(`  <- [${where}] ${label}  done in ${minutes()} min`
        + `${tokens !== null ? `, ${fmtTokens(tokens)} tokens (run total ${fmtTokens(state.tokens)})` : ''}`);
    };
    try {
      log(`  -> [${where}] ${label} (${backend.name})`);
      let last;
      for (let attempt = 1; attempt <= (opts.schema ? schemaRetries + 1 : 1); attempt++) {
        const { text, usage } = await backend.runAgent({
          prompt: full, label, cwd: opts.cwd, env: opts.env, model: opts.model, timeoutMs: agentTimeoutMs, onProgress: show,
        });
        if (usage && Number.isFinite(usage.total)) {
          tokens = (tokens || 0) + usage.total;
          state.tokens += usage.total;
          state.tokensKnown = true;
        }
        if (!opts.schema) {
          done();
          return text;
        }
        try {
          const value = extractJson(text);
          const errors = validate(value, opts.schema);
          if (errors.length) throw new Error(`schema: ${errors.slice(0, 3).join('; ')}`);
          done();
          return value;
        } catch (e) {
          last = e;
          state.schemaFailures++;
          log(`  !! [${where}] ${label}: unusable answer (attempt ${attempt}): ${e.message}`);
        }
      }
      throw last;
    } finally {
      if (beat) clearInterval(beat);
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
    return fn(agent, parallel, pipeline, phase, scriptLog, workflow, args, budget);
  }

  return { agent, parallel, pipeline, phase, log: scriptLog, runScript, state, summary };
}
