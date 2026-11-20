// Agent backends for runtime/engine.mjs: one command-line agent run per agent() call. The agent's
// tools (shell, files) run in the agent's own process, in the directory it was started from.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Linux limits one argument to 128 KiB; a longer prompt goes in a file the agent is told to read.
const ARG_PROMPT_MAX = 100 * 1024;
const KILL_GRACE_MS = 10000;
// A stream's answer comes in its result event; of the rest, a tail is kept for a CLI that printed plain text.
const STREAM_TAIL = 4 * 1024 * 1024;

// A tool call in a few words: the tool, and the command, file or pattern it was given.
function toolLine(tool, args, description) {
  const a = args || {};
  const what = a.command || a.file_path || a.path || a.pattern || a.glob_pattern || a.url || a.query || description || '';
  return `${tool}${what ? `: ${String(what).replace(/\s+/g, ' ').trim().slice(0, 160)}` : ''}`;
}

// The tokens a result event reports, under either CLI's names; cache covers reads and writes.
function usageOf(u) {
  if (!u || typeof u !== 'object') return null;
  const n = (...keys) => keys.reduce((sum, k) => sum + (Number(u[k]) || 0), 0);
  const input = n('input_tokens', 'inputTokens');
  const output = n('output_tokens', 'outputTokens');
  const cache = n('cache_read_input_tokens', 'cacheReadTokens', 'cache_creation_input_tokens', 'cacheWriteTokens');
  return { input, output, cache, total: input + output + cache };
}

const resultOf = (e) => ({ result: String(e.result ?? ''), error: Boolean(e.is_error), usage: usageOf(e.usage) });

// What one event of `claude -p --output-format stream-json --verbose` tells: the tool calls of an
// assistant message, or the final answer.
function claudeEvent(e) {
  if (e.type === 'assistant') {
    const content = (e.message && e.message.content) || [];
    return { tools: content.filter((c) => c && c.type === 'tool_use').map((c) => toolLine(c.name, c.input)) };
  }
  if (e.type === 'result') return resultOf(e);
  return null;
}

// The same for the Cursor CLI's stream-json, where a tool call starts as a tool_call event holding one
// <kind>ToolCall (shellToolCall, readToolCall, editToolCall, grepToolCall, ...).
function cursorEvent(e) {
  if (e.type === 'tool_call' && e.subtype === 'started') {
    const call = e.tool_call || {};
    const kind = Object.keys(call).find((k) => k.endsWith('ToolCall'));
    return kind ? { tools: [toolLine(kind.slice(0, -'ToolCall'.length), (call[kind] || {}).args, (call[kind] || {}).description)] } : null;
  }
  if (e.type === 'result') return resultOf(e);
  return null;
}

export const BACKENDS = {
  // Claude Code, one `claude -p` per role, permissions auto-approved as in the container setup.
  claude: {
    bin: 'claude', binEnv: 'INFERPILOT_CLAUDE_BIN', modelFlag: '--model', promptVia: 'stdin',
    args: ['-p', '--dangerously-skip-permissions', '--output-format', 'stream-json', '--verbose'], env: { IS_SANDBOX: '1' },
    events: claudeEvent,
  },
  // The Cursor CLI on the account it is logged in with (`agent login`) or CURSOR_API_KEY. Its Linux
  // sandbox needs kernel 6.2 or newer, and roles need GPUs, /dev/shm and the network: it is off.
  cursor: {
    bin: 'agent', binEnv: 'INFERPILOT_CURSOR_BIN', modelFlag: '--model', promptVia: 'arg',
    args: ['-p', '--force', '--trust', '--sandbox', 'disabled', '--output-format', 'stream-json'], env: {},
    events: cursorEvent,
  },
};

// The prompt as the CLI's last argument; one too long for an argument goes in a file in dir instead,
// which is removed once the CLI is done.
function promptArg(prompt, label) {
  if (Buffer.byteLength(prompt) <= ARG_PROMPT_MAX) return { arg: prompt, dir: null };
  const dir = mkdtempSync(join(tmpdir(), 'inferpilot-prompt-'));
  const file = join(dir, `${String(label).replace(/[^\w.-]+/g, '_')}.md`);
  writeFileSync(file, prompt);
  return { arg: `Your complete instructions are in ${file}. Read the whole file first, then follow it exactly.`, dir };
}

export function cliBackend(name, { model: defaultModel = '', preset = BACKENDS[name] } = {}) {
  if (!preset) throw new Error(`unknown backend ${name}; one of ${Object.keys(BACKENDS).join(', ')}`);
  const bin = (preset.binEnv && process.env[preset.binEnv]) || preset.bin;
  const live = new Set(); // the stop function of every agent run still going
  let closed = false;
  return {
    name,
    bin,
    // Signals the process group of every agent run still going; returns how many there were.
    stopAll(signal = 'SIGTERM') {
      for (const stop of live) stop(signal);
      return live.size;
    },
    // No agent run starts after this, so a script that retries a stopped call starts nothing new.
    close() {
      closed = true;
      return this.stopAll('SIGTERM');
    },
    get running() {
      return live.size;
    },
    // onProgress(line) hears each tool call as it starts, when the CLI streams events; the answer comes
    // with the tokens its result event reports (usage: {input, output, cache, total}, or null).
    runAgent({ prompt, label = 'agent', cwd, env, model, timeoutMs = 0, onProgress = () => {} }) {
      if (closed) return Promise.reject(new Error(`${name} [${label}] not started: the runner is stopping`));
      const argv = [...preset.args];
      const m = model || defaultModel;
      if (m) argv.push(preset.modelFlag, m);
      const viaArg = preset.promptVia === 'arg' ? promptArg(prompt, label) : { dir: null };
      if (preset.promptVia === 'arg') argv.push(viaArg.arg);
      const tidy = () => {
        if (viaArg.dir) rmSync(viaArg.dir, { recursive: true, force: true });
      };
      return new Promise((resolve, reject) => {
        // Its own process group, so a timeout also stops what the agent started.
        const child = spawn(bin, argv, {
          cwd, env: { ...process.env, ...preset.env, ...(env || {}) }, stdio: ['pipe', 'pipe', 'pipe'], detached: true,
        });
        let out = '';
        let err = '';
        let partial = '';
        let answer = null;
        let answerIsError = false;
        let usage = null;
        let timer = null;
        let killed = false;
        const stop = (signal) => {
          try {
            process.kill(-child.pid, signal);
          } catch { /* already gone */ }
        };
        if (child.pid) live.add(stop);
        const take = (raw) => {
          let e;
          try {
            e = JSON.parse(raw);
          } catch {
            return;
          }
          const said = e && typeof e === 'object' ? preset.events(e) : null;
          if (!said) return;
          for (const t of said.tools || []) {
            try {
              onProgress(t);
            } catch { /* a listener's mistake is not the agent's */ }
          }
          if (said.result !== undefined) {
            answer = said.result;
            answerIsError = said.error;
            usage = said.usage || null;
          }
        };
        child.stdout.setEncoding('utf8');
        child.stderr.setEncoding('utf8');
        child.stdout.on('data', (d) => {
          if (!preset.events) {
            out += d;
            return;
          }
          out = (out + d).slice(-STREAM_TAIL);
          const lines = (partial + d).split('\n');
          partial = lines.pop();
          lines.forEach(take);
        });
        child.stderr.on('data', (d) => { err = (err + d).slice(-65536); });
        child.on('error', (e) => {
          live.delete(stop);
          if (timer) clearTimeout(timer);
          tidy();
          reject(new Error(`${name}: cannot run ${bin}: ${e.message}`));
        });
        child.on('close', (code, signal) => {
          live.delete(stop);
          if (timer) clearTimeout(timer);
          tidy();
          if (preset.events && partial) take(partial);
          if (killed) reject(new Error(`${name} [${label}] timed out after ${Math.round(timeoutMs / 1000)} s`));
          else if (code !== 0) reject(new Error(`${name} [${label}] exited ${code ?? signal}: ${err.slice(-600)}`));
          else if (answer !== null && answerIsError) reject(new Error(`${name} [${label}] ended in an error: ${answer.slice(-600)}`));
          // No result event: a CLI that printed its answer as plain text.
          else resolve({ text: answer !== null ? answer : out, usage });
        });
        if (timeoutMs > 0) {
          timer = setTimeout(() => {
            killed = true;
            stop('SIGTERM');
            setTimeout(() => stop('SIGKILL'), KILL_GRACE_MS).unref();
          }, timeoutMs);
        }
        // An open stdin keeps some CLIs waiting: it is closed at once when the prompt is an argument.
        child.stdin.on('error', () => {});
        child.stdin.end(preset.promptVia === 'stdin' ? prompt : '');
      });
    },
  };
}
