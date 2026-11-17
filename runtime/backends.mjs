// Agent backends for runtime/engine.mjs: one command-line agent run per agent() call. The agent's
// tools (shell, files) run in the agent's own process, in the directory it was started from.
import { spawn } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Linux limits one argument to 128 KiB; a longer prompt goes in a file the agent is told to read.
const ARG_PROMPT_MAX = 100 * 1024;
const KILL_GRACE_MS = 10000;

export const BACKENDS = {
  // Claude Code, one `claude -p` per role, permissions auto-approved as in the container setup.
  claude: {
    bin: 'claude', binEnv: 'INFERPILOT_CLAUDE_BIN', modelFlag: '--model', promptVia: 'stdin',
    args: ['-p', '--dangerously-skip-permissions', '--output-format', 'text'], env: { IS_SANDBOX: '1' },
  },
  // The Cursor CLI on the account it is logged in with (`agent login`) or CURSOR_API_KEY. Its Linux
  // sandbox needs kernel 6.2 or newer, and roles need GPUs, /dev/shm and the network: it is off.
  cursor: {
    bin: 'agent', binEnv: 'INFERPILOT_CURSOR_BIN', modelFlag: '--model', promptVia: 'arg',
    args: ['-p', '--force', '--trust', '--sandbox', 'disabled', '--output-format', 'text'], env: {},
  },
};

function promptArg(prompt, label) {
  if (Buffer.byteLength(prompt) <= ARG_PROMPT_MAX) return prompt;
  const file = join(mkdtempSync(join(tmpdir(), 'inferpilot-prompt-')), `${String(label).replace(/[^\w.-]+/g, '_')}.md`);
  writeFileSync(file, prompt);
  return `Your complete instructions are in ${file}. Read the whole file first, then follow it exactly.`;
}

export function cliBackend(name, { model: defaultModel = '', preset = BACKENDS[name] } = {}) {
  if (!preset) throw new Error(`unknown backend ${name}; one of ${Object.keys(BACKENDS).join(', ')}`);
  const bin = (preset.binEnv && process.env[preset.binEnv]) || preset.bin;
  return {
    name,
    bin,
    runAgent({ prompt, label = 'agent', cwd, env, model, timeoutMs = 0 }) {
      const argv = [...preset.args];
      const m = model || defaultModel;
      if (m) argv.push(preset.modelFlag, m);
      if (preset.promptVia === 'arg') argv.push(promptArg(prompt, label));
      return new Promise((resolve, reject) => {
        // Its own process group, so a timeout also stops what the agent started.
        const child = spawn(bin, argv, {
          cwd, env: { ...process.env, ...preset.env, ...(env || {}) }, stdio: ['pipe', 'pipe', 'pipe'], detached: true,
        });
        let out = '';
        let err = '';
        let timer = null;
        let killed = false;
        const stop = (signal) => {
          try {
            process.kill(-child.pid, signal);
          } catch { /* already gone */ }
        };
        child.stdout.on('data', (d) => { out += d; });
        child.stderr.on('data', (d) => { err = (err + d).slice(-65536); });
        child.on('error', (e) => {
          if (timer) clearTimeout(timer);
          reject(new Error(`${name}: cannot run ${bin}: ${e.message}`));
        });
        child.on('close', (code, signal) => {
          if (timer) clearTimeout(timer);
          if (killed) reject(new Error(`${name} [${label}] timed out after ${Math.round(timeoutMs / 1000)} s`));
          else if (code !== 0) reject(new Error(`${name} [${label}] exited ${code ?? signal}: ${err.slice(-600)}`));
          else resolve({ text: out });
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
