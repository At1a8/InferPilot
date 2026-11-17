// Dry run of inferpilot.js on InferPilot's own workflow runner (runtime/engine.mjs) with a scripted
// fake backend: no GPU, no network, no model. Each scenario scripts what the roles answer and
// asserts the phases InferPilot walks through, the roles it calls and the final status.
//
//   node tests/dry_run.mjs
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { createRuntime } from '../runtime/engine.mjs';

const INFERPILOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SCRIPT = path.join(INFERPILOT, 'inferpilot.js');
const RUN = '/tmp/inferpilot_dry_run';
// A kernel back end the kernel liaison runs; a hand-off whose reason is FAIL makes it fail.
const CMD = { kernel_backend: 'command', kernel_backend_command: 'my-kernel-agent --task {task_dir} --out {out_dir} --gpus {gpu_ids}' };

const WP = (id, extra) => ({ id, goal: `goal of ${id}`, kind: 'framework', files: ['x.py'], test: `test ${id}`, depends_on: [], ...(extra || {}) });

// Answers every role gives unless a scenario overrides them.
const DEFAULTS = {
  'intake:define': (ins) => ({ status: 'ready', task_def: ins.given || {}, provenance: {}, assumptions: [], questions: [], summary: 'as given' }),
  'director:setup': () => ({ lock: 'acquired', run_dir: RUN, baseline_status: 'pass', baseline: '3/3 runs within spread', branch: 'inferpilot/task', worktree: `${RUN}/wt`, gpu_ids: '0,1' }),
  'director:spec': () => ({ spec_path: `${RUN}/feature_spec.md`, gates_measurable: true, perf_claim: '+20% output tok/s at equal HBM' }),
  'director:report': () => ({ report_path: `${RUN}/run_report.md`, summary: 'done' }),
  'analyst:recon': () => ({ design_path: `${RUN}/design.md`, subsystems: [{ name: 'scheduler', files: ['a.py'], hook: 'h' }], unmapped: [], gap_type: 'framework-only', level: 'L1' }),
  'architect:design': () => ({ design_path: `${RUN}/design.md`, work_packages: [WP('WP1'), WP('WP2', { depends_on: ['WP1'] })], gate_ok: true, needs_human_review: false, perf_claim: '+20% output tok/s at equal HBM' }),
  'architect:perf_plan': () => ({ work_packages: [WP('WP1')], handoffs: [{ id: 'O2', owner: 'kernel_workflow', spec_path: `${RUN}/op_spec.md` }], perf_claim: '-5% TPOT', gate_ok: true }),
  'implementer:implement': (ins) => ({ wp: ins.work_package.id, status: 'done', commit: `c_${ins.work_package.id}`, unit_test_cmd: 'pytest -q t.py', files_changed: ['x.py'], kernel_handoff: null }),
  'implementer:rollback': () => ({ wp: 'x', reset_to: 'base', clean: true }),
  'test_engineer:unit': (ins) => ({ wp: ins.work_package.id, unit_pass: true, feature_off_unchanged: true, fails_without_change: 'parent', evidence: 'ok' }),
  'verify_engineer:verify': () => ({ all_pass: true, gates: { G0: 'pass', G2: 'pass' }, failures: [], ledger_path: `${RUN}/experiment_ledger.md` }),
  'investigator:investigate': () => ({ verdict: 'fix_needed', work_packages: [WP('FIX1')], summary: 'race in hook' }),
  'investigator:diagnose': () => ({ verdict: 'fix_needed', root_cause: 'stale import', work_packages: [WP('WP1')] }),
  'perf_engineer:perf_eval': () => ({ claim_reproduced: true, held_out_ok: true, guards_ok: true, workload_unchanged: true, path_ok: true, results: [], summary: '+22%' }),
  'perf_engineer:profile': () => ({ analysis_path: `${RUN}/perf_analysis.md`, path_ok: true, path_evidence: 'aiter attention, graphs on', metrics: { gpu_busy: 0.71 }, opportunities: [
    { id: 'O1', kind: 'host_sync', where: 'aiter_backend.py', evidence: 'trace', upper_bound_pct: 4, owner: 'inferpilot' },
    { id: 'O2', kind: 'kernel', where: 'moe', evidence: 'trace', upper_bound_pct: 9, owner: 'kernel_workflow' }] }),
  'reviewer:review': () => ({ verdict: 'approve', findings: [{ severity: 'minor', where: 'x.py:1', issue: 'a name' }], review_path: `${RUN}/review.md`, summary: 'one minor' }),
  'pr_engineer:pr': () => ({ branch: 'inferpilot/task', title: '[Fix] [AMD] scope: thing', commits: ['c_WP1'], lint_pass: true, feature_off_final: true, content_unchanged: true, merges_cleanly: true, pr_body_path: `${RUN}/pr_body.md`, push_command: 'git push origin inferpilot/task' }),
  'curator:learn': () => ({ updates: [{ file: 'knowledge/known_issues.md', change: 'none' }], skill_status: 'draft' }),
  'issue_triager:triage': () => ({ priority: 'P0', category: 'crash', action: 'reproduce', reason: 'default config crash', existing_prs: [], triage_path: `${RUN}/issue_triage.md`, reply_draft_path: `${RUN}/reply.md` }),
  'issue_triager:reproduce': () => ({ reproduced: true, repro_cmd: 'bash repro.sh', evidence: 'segfault 3/3' }),
  'ci_guardian:classify': () => ({ failure_class: 'code_regression', failing_test_cmd: 'pytest test_x.py', window: 'a..b', evidence: 'new regression since 10-21', triage_path: `${RUN}/ci_triage.md` }),
  'ci_guardian:bisect': () => ({ culprit: 'deadbeef', confirmed: true, work_packages: [WP('WP1')], evidence: 'fails at culprit, passes at parent' }),
  'ci_guardian:bisect_fix': () => ({ fix_commit: 'feedface', confirmed: true, evidence: 'fails at parent, passes at fix' }),
  'scout:collect': () => ({ lock: 'acquired', run_dir: RUN, candidates_path: `${RUN}/candidates.json`, counts: { upstream: 3, issues: 32, ci: 76, hardware: 5 } }),
  'dispatcher:dispatch': () => ({ queue_path: `${RUN}/task_queue.md`, top: [{ rank: 1, task: 'GLM-5.3 DCP crash', type: 'bug', level: 'L1', priority: 'P0', source: 'issues' }], matrix_changes: 2 }),
  'kernel_liaison:package': (ins) => ({ task_dir: `${RUN}/kernel/${ins.handoff.id}/task`, mode: 'optimize', target_language: 'hip',
    op_spec: { op_kind: 'swap_in' }, task: ins.handoff.reason === 'FAIL' ? 'FAIL on purpose' : 'decode, M <= 64' }),
  'kernel_liaison:run': (ins) => (ins.handoff.reason === 'FAIL'
    ? { exit_code: 1, result_path: '', error: 'the kernel agent failed' }
    : { exit_code: 0, result_path: `${RUN}/kernel/out/result.json` }),
  'kernel_liaison:collect': (ins) => ((ins.backend_result || {}).error
    ? { status: 'failed', notes: ins.backend_result.error }
    : { status: 'improved', speedup: 1.37, evidence: 'eval dir', integrate_wp: WP(`K-${ins.handoff.id}`, { kind: 'kernel' }) }),
};

// The implementer hands WP1's kernel over (reason FAIL makes the fake back end throw).
const withKernelHandoff = (reason) => ({
  'implementer:implement': (ins) => ({ wp: ins.work_package.id, status: 'done', commit: `c_${ins.work_package.id}`, unit_test_cmd: 'pytest -q t.py',
    kernel_handoff: ins.work_package.id === 'WP1'
      ? { op_spec_path: `${RUN}/handoffs/K-WP1/op_spec.md`, reference_path: `${RUN}/handoffs/K-WP1/ref.py`, test_cmd: 'pytest k.py', plug_in: 'x.py:f', reason }
      : null }),
});

const ROLE_RE = /You are the ([a-z_]+)\. PHASE=([a-z_]+)\./;

function parseInputs(prompt) {
  const out = {};
  const block = prompt.split('## Inputs\n')[1] || '';
  for (const line of block.split('\n')) {
    const m = /^- ([a-z_]+): (.*)$/.exec(line);
    if (!m) continue;
    try { out[m[1]] = JSON.parse(m[2]); } catch { out[m[1]] = m[2]; }
  }
  return out;
}

function makeBackend(overrides) {
  const calls = [];
  const counts = {};
  const b = { name: 'fake' };
  b.runAgent = async ({ prompt, model }) => {
    const m = ROLE_RE.exec(prompt);
    if (!m) throw new Error('prompt without role header');
    const key = `${m[1]}:${m[2]}`;
    const n = (counts[key] = (counts[key] || 0) + 1);
    const ins = parseInputs(prompt);
    calls.push({ key, n, ins, model, prompt });
    const f = (overrides && overrides[key]) || DEFAULTS[key];
    if (!f) throw new Error(`no scripted answer for ${key}`);
    const ans = await f(ins, n);
    return { text: JSON.stringify(ans) };
  };
  return { backend: b, calls };
}

async function run(args, overrides) {
  const phases = [];
  const { backend, calls } = makeBackend(overrides);
  const rt = createRuntime({
    backend,
    concurrency: 2,
    schemaRetries: 0,
    log: (msg) => {
      const m = /=== PHASE: (.*) ===/.exec(msg);
      if (m) phases.push(m[1]);
    },
  });
  const modeDefaults = {
    feature: { model_path: '/models/m', server_script: '/s/server.sh' },
    perf: { model_path: '/models/m', workload: '{"isl":8192,"osl":1024,"conc":64}', server_script: '/s/server.sh', client_script: '/s/client.sh' },
  }[args.mode] || {};
  const result = await rt.runScript(SCRIPT, { inferpilot_dir: INFERPILOT, framework_dir: '/tmp/sglang', agent_retries: 1, ...modeDefaults, ...args }, 0);
  return { result, phases, calls, keys: calls.map((c) => c.key) };
}

let failed = 0;
let passed = 0;
function check(cond, msg) {
  if (cond) passed++;
  else { failed++; console.error(`  FAIL: ${msg}`); }
}
const eqList = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// The run stopped at Intake with a question about `field`, before anything was set up.
const asks = (r, field) => r.result.status === 'needs_input' && r.result.stopped_at === 'Intake'
  && (r.result.questions || []).some((q) => q.field === field) && !r.keys.includes('director:setup');

const scenarios = [
  ['feature: straight path to a reviewable PR', async () => {
    const r = await run({ mode: 'feature', task: 'port HiSparse V2 to DeepSeek-V4.1' });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases, ['Setup', 'Spec', 'Recon', 'Design', 'Implement', 'Verify', 'Performance', 'Review', 'PR', 'Report']), `phases ${r.phases}`);
    check(r.result.done.map((d) => d.wp).join() === 'WP1,WP2', `done ${JSON.stringify(r.result.done)}`);
    check(r.result.pr && r.result.pr.push_command === 'git push origin inferpilot/task', 'push command surfaced, not executed');
    check(r.keys.filter((k) => k === 'curator:learn').length === 1, 'curator runs once');
  }],
  ['feature: unit gate fails once, feedback reaches the second round', async () => {
    const r = await run({ mode: 'feature', task: 't' }, {
      'test_engineer:unit': (ins, n) => (n === 1
        ? { wp: ins.work_package.id, unit_pass: false, feature_off_unchanged: true, fails_without_change: 'parent', failure_summary: 'case 7 mismatches the reference' }
        : { wp: ins.work_package.id, unit_pass: true, feature_off_unchanged: true, fails_without_change: 'parent' }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    const second = r.calls.filter((c) => c.key === 'implementer:implement')[1];
    check(second && second.ins.feedback === 'case 7 mismatches the reference', 'feedback passed to round 1');
  }],
  ['implement: a unit test that passes without the change is a failed round', async () => {
    const r = await run({ mode: 'feature', task: 't' }, {
      'test_engineer:unit': (ins, n) => ({ wp: ins.work_package.id, unit_pass: true, feature_off_unchanged: true,
        fails_without_change: n === 1 ? 'no' : 'mutant', failure_summary: n === 1 ? 'passes on the parent: it never calls the new path' : '' }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    const rounds = r.calls.filter((c) => c.key === 'implementer:implement');
    check(rounds.length === 3 && rounds[1].ins.feedback.includes('cannot fail')
      && rounds[1].ins.feedback.includes('never calls the new path'), `feedback: ${rounds[1] && rounds[1].ins.feedback}`);
  }],
  ['implement: only a docs package may have nothing to fail', async () => {
    const asNa = (ins) => ({ wp: ins.work_package.id, unit_pass: true, feature_off_unchanged: true, fails_without_change: 'n/a' });
    const docs = await run({ mode: 'feature', task: 't' }, {
      'architect:design': () => ({ design_path: `${RUN}/design.md`, work_packages: [WP('D1', { kind: 'docs' })], gate_ok: true, perf_claim: '' }),
      'test_engineer:unit': asNa,
    });
    check(docs.result.status === 'ready_for_review', `docs: ${docs.result.status}`);
    const code = await run({ mode: 'feature', task: 't', max_fix_rounds: 0 }, { 'test_engineer:unit': asNa });
    check(code.result.status === 'blocked' && code.result.stopped_at === 'Implement', `framework with n/a: ${code.result.status}`);
    check(code.keys.includes('implementer:rollback'), 'rolled back');
  }],
  ['feature: verify fails, investigation finds a fix, re-verify passes', async () => {
    const r = await run({ mode: 'feature', task: 't' }, {
      'verify_engineer:verify': (ins, n) => (n === 1
        ? { all_pass: false, failures: [{ gate: 'G4', failure_class: 'catastrophic collapse', counts: '3/4 runs' }] }
        : { all_pass: true, failures: [] }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases.slice(4, 9), ['Implement', 'Verify', 'Investigate', 'Implement', 'Verify']), `phases ${r.phases}`);
    check(r.result.done.map((d) => d.wp).join() === 'WP1,WP2,FIX1', `done ${JSON.stringify(r.result.done)}`);
  }],
  ['feature: baseline bug is recorded and the gate is re-judged', async () => {
    const r = await run({ mode: 'feature', task: 't' }, {
      'verify_engineer:verify': (ins, n) => (n === 1
        ? { all_pass: false, failures: [{ gate: 'G4', failure_class: 'catastrophic collapse' }] }
        : { all_pass: (ins.known_issues || []).length === 1, failures: [] }),
      'investigator:investigate': () => ({ verdict: 'baseline_bug', known_issue: { id: 'KI-002', title: 'whole-batch collapse' }, summary: 'same-pattern baseline fails the same way' }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(r.result.known_issues.length === 1 && r.result.known_issues[0].id === 'KI-002', 'known issue recorded and passed to verify');
  }],
  ['feature: investigation budget is bounded', async () => {
    const r = await run({ mode: 'feature', task: 't', max_investigate_rounds: 2 }, {
      'verify_engineer:verify': () => ({ all_pass: false, failures: [{ gate: 'G3' }] }),
      'investigator:investigate': () => ({ verdict: 'baseline_bug', known_issue: { id: 'KI-x' } }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Verify', `status ${r.result.status} at ${r.result.stopped_at}`);
    check(r.keys.filter((k) => k === 'verify_engineer:verify').length === 3, 'verify runs max_investigate_rounds + 1 times');
  }],
  ['feature: with auto_approve_design false, L3 waits for a human-approved design', async () => {
    const r = await run({ mode: 'feature', task: 't', auto_approve_design: false }, {
      'analyst:recon': () => ({ design_path: `${RUN}/design.md`, unmapped: [], level: 'L3', gap_type: 'framework-only' }),
    });
    check(r.result.status === 'awaiting_design_review' && r.result.stopped_at === 'Design', `status ${r.result.status}`);
    check(!r.keys.includes('implementer:implement'), 'no implementation before review');
  }],
  ['feature: stop_after Design', async () => {
    const r = await run({ mode: 'feature', task: 't', stop_after: 'Design' });
    check(r.result.status === 'stopped' && r.result.stopped_at === 'Design', `status ${r.result.status}`);
  }],
  ['feature: a flagged design goes on by default; auto_approve_design false makes it wait', async () => {
    const flagged = {
      'architect:design': () => ({ design_path: `${RUN}/design.md`, work_packages: [WP('WP1')], gate_ok: true,
        needs_human_review: true, perf_claim: '+20% output tok/s at equal HBM', summary: 'S4 path decision' }),
    };
    const r = await run({ mode: 'feature', task: 't' }, flagged);
    check(r.result.status === 'ready_for_review' && r.keys.includes('implementer:implement'), `status ${r.result.status}`);
    const d = r.result.phases.find((p) => p.phase === 'Design' && p.verdict === 'auto_approved');
    check(d && d.summary.includes('the architect asked for a human review') && d.summary.includes('S4 path decision'),
      `the auto-approval is on record: ${JSON.stringify(d)}`);
    for (const off of [false, 'false']) {
      const waits = await run({ mode: 'feature', task: 't', auto_approve_design: off }, flagged);
      check(waits.result.status === 'awaiting_design_review' && !waits.keys.includes('implementer:implement'),
        `auto_approve_design=${JSON.stringify(off)}: a flagged design waits: ${waits.result.status}`);
    }
    const l3 = await run({ mode: 'feature', task: 't' }, {
      'analyst:recon': () => ({ design_path: `${RUN}/design.md`, unmapped: [], level: 'L3', gap_type: 'framework-only' }),
    });
    check(l3.result.status === 'ready_for_review'
      && l3.result.phases.some((p) => p.verdict === 'auto_approved' && p.summary.includes('L3 work')), `L3: ${l3.result.status}`);
    const stop = await run({ mode: 'feature', task: 't', stop_after: 'Design' }, flagged);
    check(stop.result.status === 'stopped' && stop.result.stopped_at === 'Design', `stop_after still stops: ${stop.result.status}`);
    const plain = await run({ mode: 'feature', task: 't' });
    check(!plain.result.phases.some((p) => p.verdict === 'auto_approved'), 'a design that needs no review is not marked');
  }],
  ['feature: resume_from implement after review', async () => {
    const r = await run({ mode: 'feature', task: 't', resume_from: 'implement', run_dir: RUN });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(!r.keys.includes('director:spec') && !r.keys.includes('analyst:recon'), 'spec and recon skipped');
    const d = r.calls.find((c) => c.key === 'architect:design');
    check(d && d.ins.approved_design === true, 'architect told the design is approved');
    const pe = r.calls.find((c) => c.key === 'perf_engineer:perf_eval');
    check(pe && pe.ins.claim === '+20% output tok/s at equal HBM', `the resumed run measures the claim: ${pe && pe.ins.claim}`);
    const noClaim = await run({ mode: 'feature', task: 't', resume_from: 'implement', run_dir: RUN }, {
      'architect:design': () => ({ design_path: `${RUN}/design.md`, work_packages: [WP('WP1')], gate_ok: true }),
    });
    check(noClaim.result.status === 'blocked' && noClaim.result.stopped_at === 'Design', `a design that does not state its claim: ${noClaim.result.status}`);
  }],
  ['agents: a timed-out call is not retried and goes on the attention list', async () => {
    const r = await run({ mode: 'feature', task: 't', agent_retries: 3, agent_timeout_ms: 100 }, {
      'verify_engineer:verify': async () => {
        await new Promise((resolve) => setTimeout(resolve, 400));
        return { all_pass: true, failures: [] };
      },
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Verify', `status ${r.result.status}`);
    check(r.keys.filter((k) => k === 'verify_engineer:verify').length === 1, 'a timed-out call is not run twice');
    check(r.result.attention.some((a) => String(a.call).startsWith('verify_engineer') && a.issue.includes('may still be running')),
      `attention: ${JSON.stringify(r.result.attention)}`);
  }],
  ['agents: a timed-out implement or unit call starts no further round and no rollback', async () => {
    const slow = (ans) => async (ins) => {
      await new Promise((resolve) => setTimeout(resolve, 400));
      return ans(ins);
    };
    for (const role of ['implementer:implement', 'test_engineer:unit']) {
      const r = await run({ mode: 'feature', task: 't', agent_timeout_ms: 100 }, {
        [role]: slow(role === 'implementer:implement'
          ? (ins) => ({ wp: ins.work_package.id, status: 'done', commit: `c_${ins.work_package.id}`, unit_test_cmd: 'pytest -q t.py' })
          : () => ({ unit_pass: true, feature_off_unchanged: true, fails_without_change: 'parent' })),
      });
      check(r.result.status === 'blocked' && r.result.stopped_at === 'Implement', `${role}: status ${r.result.status}`);
      check(r.keys.filter((k) => k === 'implementer:implement').length === 1, `${role}: the work package gets no second round`);
      check(!r.keys.includes('implementer:rollback'), `${role}: no rollback while the timed-out agent may still run`);
      check(r.result.phases.some((p) => p.phase === 'Implement' && p.summary.includes('timed out')), `${role}: the stop says why`);
      check(r.result.attention.some((a) => String(a.call).startsWith(role.split(':')[0]) && a.issue.includes('may still be running')),
        `${role}: attention ${JSON.stringify(r.result.attention)}`);
    }
  }],
  ['agents: a failed call is retried, and the retry checks what the first one left', async () => {
    const r = await run({ mode: 'feature', task: 't', agent_retries: 2 }, {
      'verify_engineer:verify': (ins, n) => (n === 1 ? { verdict: 'not the schema' } : { all_pass: true, failures: [] }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    const calls = r.calls.filter((c) => c.key === 'verify_engineer:verify');
    check(calls.length === 2 && !calls[0].prompt.includes('This is attempt') && calls[1].prompt.includes('This is attempt 2'),
      'the second attempt is told to check what the first left');
  }],
  ['setup: the GPUs Setup pins reach every later role; the user\'s GPUs win', async () => {
    const picked = await run({ mode: 'feature', task: 't' }, {
      'director:setup': () => ({ lock: 'acquired', run_dir: RUN, baseline_status: 'pass', worktree: `${RUN}/wt`, gpu_ids: '4,5' }),
    });
    const later = picked.calls.filter((c) => !['director:setup', 'intake:define'].includes(c.key));
    check(later.length > 5 && later.every((c) => c.ins.gpu_ids === '4,5'), 'every role after Setup sees gpu_ids=4,5');
    check(later[0].prompt.includes('HIP_VISIBLE_DEVICES'), 'roles are told to export HIP_VISIBLE_DEVICES');
    const given = await run({ mode: 'feature', task: 't', gpu_ids: '2' }, {
      'director:setup': () => ({ lock: 'acquired', run_dir: RUN, baseline_status: 'pass', gpu_ids: '6,7' }),
    });
    check(String(given.calls.find((c) => c.key === 'verify_engineer:verify').ins.gpu_ids) === '2', 'the user\'s GPUs win');
    const none = await run({ mode: 'issue', task: 'i' }, {
      'director:setup': () => ({ lock: 'acquired', run_dir: RUN, baseline_status: 'skip', gpu_ids: '' }),
    });
    check(none.result.status === 'blocked' && none.result.stopped_at === 'Setup', `no GPUs pinned: ${none.result.status}`);
  }],
  ['lock: while another task holds the machine, a run ends busy and creates nothing', async () => {
    const busy = { lock: 'busy', run_dir: '', baseline_status: 'skip', gpu_ids: '',
      notes: 'held by /x/exp/perf_dsv41 (container fangyuan) since 09:12; release: scripts/run_lock.py release --force' };
    for (const mode of ['feature', 'issue', 'ci', 'perf']) {
      const r = await run({ mode, task: 't' }, { 'director:setup': () => busy });
      check(r.result.status === 'busy' && r.result.stopped_at === 'Setup' && eqList(r.keys, ['director:setup']),
        `${mode}: ${r.result.status}, calls ${r.keys}`);
      check(r.result.run_dir === '' && r.result.phases[0].summary.includes('release --force'), `${mode}: the holder is reported`);
    }
    const scan = await run({ mode: 'scan' }, {
      'scout:collect': () => ({ lock: 'busy', run_dir: '', candidates_path: '', notes: 'held by /x/exp/issue_41299' }),
    });
    check(scan.result.status === 'busy' && eqList(scan.keys, ['scout:collect']), `scan: ${scan.result.status}`);
    const setup = (await run({ mode: 'issue', task: 'i' })).calls.find((c) => c.key === 'director:setup');
    check(String(setup.ins.lock_idle_hours) === '12', `idle hours for the lock: ${setup.ins.lock_idle_hours}`);
  }],
  ['image: a task on an image other than the newest daily one stops at Setup to ask; image_ok=true goes on', async () => {
    const older = (ins) => (ins.image_ok ? undefined : {
      lock: 'skipped', run_dir: '', baseline_status: 'skip', gpu_ids: '',
      image: { current: 'rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261105', wanted: 'rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261109', verdict: 'older' } });
    const ok = { lock: 'acquired', run_dir: RUN, baseline_status: 'pass', baseline: '3/3 runs within spread', branch: 'inferpilot/task',
      worktree: '/sgl-workspace/sglang', gpu_ids: '0,1' };
    for (const mode of ['feature', 'issue', 'ci', 'perf']) {
      const asked = await run({ mode, task: 't' }, { 'director:setup': older });
      check(asked.result.status === 'needs_input' && asked.result.stopped_at === 'Setup' && eqList(asked.keys, ['director:setup']),
        `${mode}: asks before anything is created: ${asked.result.status}, calls ${asked.keys}`);
      check(asked.result.questions[0].field === 'image_ok' && asked.result.questions[0].question.includes('20261109'),
        `${mode}: the question names the newest image`);
    }
    const goes = await run({ mode: 'issue', task: 't', answers: { image_ok: true } },
      { 'director:setup': (ins) => (ins.image_ok ? ok : older(ins)) });
    check(goes.result.status !== 'needs_input' && goes.calls.find((c) => c.key === 'director:setup').ins.image_ok === true,
      `answers={"image_ok": true} runs on: ${goes.result.status}`);
    const named = await run({ mode: 'issue', task: 't', image: 'rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261021' });
    check(named.calls.find((c) => c.key === 'director:setup').ins.image === 'rocm/sgl-dev:v0.5.22-rocm10-mi35x-20261021',
      'a named image reaches Setup');
  }],
  ['pr: what the PR engineer finds missing goes back through implement, verify and review', async () => {
    const r = await run({ mode: 'ci', task: 'c' }, {
      'pr_engineer:pr': (ins, n) => (n === 1
        ? { branch: 'inferpilot/task', lint_pass: false, feature_off_final: true, content_unchanged: true, pr_body_path: `${RUN}/pr_body.md`,
          work_packages: [WP('H1', { goal: 'register the new test in test/registered' })] }
        : { branch: 'inferpilot/task', lint_pass: true, feature_off_final: true, content_unchanged: true, pr_body_path: `${RUN}/pr_body.md` }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    const afterPr = r.phases.slice(r.phases.indexOf('PR'));
    check(eqList(afterPr, ['PR', 'Implement', 'Verify', 'Review', 'PR', 'Report']), `phases after the first PR: ${afterPr}`);
    check(r.result.done.some((d) => d.wp === 'H1'), 'the hygiene package was verified');
    const changed = await run({ mode: 'ci', task: 'c' }, {
      'pr_engineer:pr': () => ({ branch: 'inferpilot/task', lint_pass: true, feature_off_final: true, content_unchanged: false,
        pr_body_path: `${RUN}/pr_body.md`, work_packages: [WP('H2')] }),
    });
    check(changed.result.status === 'blocked' && changed.result.stopped_at === 'PR', `a branch that changed after review: ${changed.result.status}`);
    check(changed.keys.filter((k) => k === 'implementer:implement').length === 1, 'no loop on a branch that changed after review');
  }],
  ['stop_after: every mode stops after Setup; a phase the mode lacks is asked about', async () => {
    for (const mode of ['issue', 'ci', 'perf']) {
      const r = await run({ mode, task: 't', stop_after: 'Setup' });
      check(r.result.status === 'stopped' && r.result.stopped_at === 'Setup' && r.keys.at(-2) === 'director:report',
        `${mode}: ${r.result.status} at ${r.result.stopped_at}`);
    }
    check(asks(await run({ mode: 'feature', task: 't', stop_after: 'Reproduce' }), 'stop_after'), 'Reproduce is not a feature phase');
    check(asks(await run({ mode: 'ci', task: 't', stop_after: 'Performance' }), 'stop_after'), 'ci never runs Performance');
    const ok = await run({ mode: 'issue', task: 'i', stop_after: 'reproduce' });
    check(ok.result.status === 'stopped' && ok.result.stopped_at === 'Reproduce', `case does not matter: ${ok.result.status}`);
  }],
  ['feature: a work package that never passes is rolled back', async () => {
    const r = await run({ mode: 'feature', task: 't', max_fix_rounds: 1 }, {
      'test_engineer:unit': (ins) => ({ wp: ins.work_package.id, unit_pass: false, feature_off_unchanged: true, fails_without_change: 'parent', failure_summary: 'still wrong' }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Implement', `status ${r.result.status}`);
    check(r.keys.filter((k) => k === 'implementer:implement').length === 2, 'max_fix_rounds + 1 attempts');
    check(r.keys.includes('implementer:rollback'), 'rollback called');
    check(!r.keys.includes('verify_engineer:verify'), 'no verification of a failed package');
  }],
  ['feature: baseline that does not reproduce blocks at Setup', async () => {
    const r = await run({ mode: 'feature', task: 't' }, {
      'director:setup': () => ({ lock: 'acquired', run_dir: RUN, baseline_status: 'fail', baseline: 'spread 9% > 2%', gpu_ids: '0' }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Setup', `status ${r.result.status}`);
    check(r.keys.includes('director:report'), 'blocked runs still get a report');
  }],
  ['setup: a skipped baseline is enough only where none is required', async () => {
    const skip = { 'director:setup': () => ({ lock: 'acquired', run_dir: RUN, baseline_status: 'skip', baseline: 'not needed yet', gpu_ids: '0' }) };
    const feature = await run({ mode: 'feature', task: 't' }, skip);
    check(feature.result.status === 'blocked' && feature.result.stopped_at === 'Setup', `feature with skip: ${feature.result.status}`);
    const issue = await run({ mode: 'issue', task: 'i' }, skip);
    check(issue.result.status === 'ready_for_review', `issue with skip: ${issue.result.status}`);
    const prose = await run({ mode: 'issue', task: 'i' }, { 'director:setup': () => ({ run_dir: RUN, baseline: 'PASS - 3/3' }) });
    check(prose.result.status === 'blocked' && prose.result.stopped_at === 'Setup', 'a baseline in prose without baseline_status is rejected');
  }],
  ['implement: done without a commit is a failed round', async () => {
    const r = await run({ mode: 'feature', task: 't', max_fix_rounds: 1 }, {
      'implementer:implement': (ins) => ({ wp: ins.work_package.id, status: 'done', commit: '', unit_test_cmd: 'pytest -q t.py' }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Implement', `status ${r.result.status}`);
    check(!r.keys.includes('test_engineer:unit'), 'nothing to test without a commit');
    const imp = r.result.phases.find((p) => p.phase === 'Implement');
    check(imp && imp.summary.includes('without a commit'), `reason recorded: ${imp && imp.summary}`);
  }],
  ['rollback: an unclean rollback goes on the attention list of the report', async () => {
    const r = await run({ mode: 'feature', task: 't', max_fix_rounds: 0 }, {
      'test_engineer:unit': (ins) => ({ wp: ins.work_package.id, unit_pass: false, feature_off_unchanged: true, fails_without_change: 'parent', failure_summary: 'wrong' }),
      'implementer:rollback': () => ({ wp: 'WP1', reset_to: 'base', clean: false }),
    });
    check(r.result.status === 'blocked' && r.result.attention.length === 1 && r.result.attention[0].wp === 'WP1',
      `attention ${JSON.stringify(r.result.attention)}`);
    const rep = r.calls.find((c) => c.key === 'director:report');
    check(rep && Array.isArray(rep.ins.attention) && rep.ins.attention.length === 1, 'the report sees what a human must clean up');
    const clean = await run({ mode: 'feature', task: 't', max_fix_rounds: 0 }, {
      'test_engineer:unit': (ins) => ({ wp: ins.work_package.id, unit_pass: false, feature_off_unchanged: true, fails_without_change: 'parent', failure_summary: 'wrong' }),
    });
    check(clean.result.attention.length === 0, 'a clean rollback needs no attention');
  }],
  ['issue: reproduce, diagnose, fix, verify with the reproduction as a gate', async () => {
    const r = await run({ mode: 'issue', task: 'https://github.com/sgl-project/sglang/issues/38849' });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases, ['Setup', 'Triage', 'Reproduce', 'Diagnose', 'Implement', 'Verify', 'Review', 'PR', 'Report']), `phases ${r.phases}`);
    const v = r.calls.find((c) => c.key === 'verify_engineer:verify');
    check(v && JSON.stringify(v.ins.extra_gates).includes('issue_repro'), 'issue_repro gate passed to verify');
    check(r.result.priority === 'P0', 'priority recorded');
  }],
  ['issue: needs_info ends at triage with a reply draft', async () => {
    const r = await run({ mode: 'issue', task: 'i' }, {
      'issue_triager:triage': () => ({ priority: 'P1', category: 'crash', action: 'needs_info', triage_path: `${RUN}/t.md`, reply_draft_path: `${RUN}/reply.md` }),
    });
    check(r.result.status === 'triaged' && r.result.stopped_at === 'Triage', `status ${r.result.status}`);
  }],
  ['issue: close_fixed that no longer reproduces', async () => {
    const r = await run({ mode: 'issue', task: 'i' }, {
      'issue_triager:triage': () => ({ priority: 'P2', category: 'crash', action: 'close_fixed', triage_path: `${RUN}/t.md` }),
      'issue_triager:reproduce': (ins) => ({ reproduced: false, repro_cmd: 'bash repro.sh', evidence: `expect_fixed=${ins.expect_fixed}; clean 3/3` }),
    });
    check(r.result.status === 'fixed_on_main', `status ${r.result.status}`);
  }],
  ['issue: the reporter\'s code fails and main passes: fixed on main, not "not reproduced"', async () => {
    const onReporter = { 'issue_triager:reproduce': () => ({ reproduced: false, reproduced_on_reporter: true, repro_cmd: 'bash repro.sh',
      evidence: 'reporter commit 3/3 crash; main 6/6 pass' }) };
    const r = await run({ mode: 'issue', task: 'i' }, onReporter);
    check(r.result.status === 'fixed_on_main' && r.result.stopped_at === 'Reproduce', `status ${r.result.status} at ${r.result.stopped_at}`);
    check(!r.keys.includes('investigator:diagnose') && !r.keys.includes('ci_guardian:bisect_fix'), 'no diagnosis, no bisect unless asked');
    const neither = await run({ mode: 'issue', task: 'i' }, {
      'issue_triager:reproduce': () => ({ reproduced: false, repro_cmd: 'bash repro.sh', evidence: 'main 6/6 pass; reporter image not run' }),
    });
    check(neither.result.status === 'not_reproduced', `without a reporter result: ${neither.result.status}`);

    const bis = await run({ mode: 'issue', task: 'i', bisect_fixed: true }, {
      ...onReporter,
      'ci_guardian:bisect_fix': (ins) => ({ fix_commit: 'abc1234 (#44302)', confirmed: ins.repro_cmd === 'bash repro.sh',
        evidence: 'fails at parent 3/3, passes at fix 3/3' }),
    });
    check(bis.result.status === 'fixed_on_main' && bis.result.fixed_by === 'abc1234 (#44302)', `bisect: ${bis.result.status} ${bis.result.fixed_by}`);
    check(bis.phases.indexOf('Bisect') === bis.phases.indexOf('Reproduce') + 1 && !bis.phases.includes('Diagnose')
      && bis.phases[bis.phases.length - 1] === 'Report', `phases ${bis.phases}`);
    const rep = bis.calls.find((c) => c.key === 'director:report');
    check(rep && rep.ins.fixed_by === 'abc1234 (#44302)', 'the report names the fix');
    const unconfirmed = await run({ mode: 'issue', task: 'i', bisect_fixed: 'true' }, {
      ...onReporter, 'ci_guardian:bisect_fix': () => ({ fix_commit: 'abc1234', confirmed: false }),
    });
    check(unconfirmed.result.status === 'fixed_on_main' && !unconfirmed.result.fixed_by
      && unconfirmed.result.phases.some((p) => p.phase === 'Bisect' && p.verdict === 'fail'), 'an unconfirmed bisect names nothing');
    const stopped = await run({ mode: 'issue', task: 'i', bisect_fixed: true, stop_after: 'Reproduce' }, onReporter);
    check(stopped.result.status === 'fixed_on_main' && !stopped.keys.includes('ci_guardian:bisect_fix'),
      `stop_after=Reproduce runs no bisect: ${stopped.result.status}`);
  }],
  ['issue: an existing fix PR is reproduced on main first and gated on that reproduction', async () => {
    const r = await run({ mode: 'issue', task: 'i' }, {
      'issue_triager:triage': () => ({ priority: 'P1', category: 'crash', action: 'push_existing_pr', existing_prs: ['#40872'], triage_path: `${RUN}/t.md` }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases, ['Setup', 'Triage', 'Reproduce', 'Implement', 'Verify', 'Review', 'PR', 'Report']), `phases ${r.phases}`);
    const im = r.calls.find((c) => c.key === 'implementer:implement');
    check(im && im.ins.work_package.kind === 'apply_pr' && im.ins.work_package.goal.includes('#40872'), 'apply_pr work package');
    const v = r.calls.find((c) => c.key === 'verify_engineer:verify');
    check(v && v.ins.extra_gates[0].cmd === 'bash repro.sh', `issue_repro gate carries the reproduction: ${JSON.stringify(v && v.ins.extra_gates)}`);
  }],
  ['issue: a reproduction without a command stops before any change', async () => {
    const r = await run({ mode: 'issue', task: 'i' }, {
      'issue_triager:reproduce': () => ({ reproduced: true, repro_cmd: '  ', evidence: 'segfault 3/3' }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Reproduce', `status ${r.result.status}`);
    check(!r.keys.includes('investigator:diagnose'), 'no diagnosis without a reproduction command');
  }],
  ['issue: only a category from the list is accepted, and performance runs the evaluation', async () => {
    const odd = await run({ mode: 'issue', task: 'i' }, {
      'issue_triager:triage': () => ({ priority: 'P1', category: 'Performance', action: 'reproduce', triage_path: `${RUN}/t.md` }),
    });
    check(odd.result.status === 'blocked' && odd.result.stopped_at === 'Triage', `category Performance: ${odd.result.status}`);
    const perfTriage = { 'issue_triager:triage': () => ({ priority: 'P1', category: 'performance', action: 'reproduce', triage_path: `${RUN}/t.md` }) };
    const perf = await run({ mode: 'issue', task: 'i' }, {
      ...perfTriage,
      'issue_triager:reproduce': () => ({ reproduced: true, repro_cmd: 'bash bench.sh', perf_claim: 'decode tok/s back to 2400 at c32 (v0.5.21)', evidence: '1710 tok/s 3/3' }),
    });
    check(perf.phases.includes('Performance') && perf.result.status === 'ready_for_review', `performance issue: ${perf.phases}`);
    const pe = perf.calls.find((c) => c.key === 'perf_engineer:perf_eval');
    check(pe && pe.ins.claim === 'decode tok/s back to 2400 at c32 (v0.5.21)', `the issue's number is the claim: ${pe && pe.ins.claim}`);
    const noNumber = await run({ mode: 'issue', task: 'i' }, perfTriage);
    check(noNumber.result.status === 'blocked' && noNumber.result.stopped_at === 'Reproduce', `no number to restore: ${noNumber.result.status}`);
  }],
  ['ci: infrastructure is handed over', async () => {
    const r = await run({ mode: 'ci', task: 'tracker #45847 cluster NIXL' }, {
      'ci_guardian:classify': () => ({ failure_class: 'infrastructure', failing_test_cmd: '', evidence: 'UCX interface errno 19', triage_path: `${RUN}/ci.md` }),
    });
    check(r.result.status === 'handed_off', `status ${r.result.status}`);
    check(!r.keys.includes('ci_guardian:bisect'), 'no bisect for infrastructure');
  }],
  ['ci: code regression to a fix PR', async () => {
    const r = await run({ mode: 'ci', task: 'tracker cluster _check ImportError' });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases, ['Setup', 'Classify', 'Bisect', 'Implement', 'Verify', 'Review', 'PR', 'Report']), `phases ${r.phases}`);
    const v = r.calls.find((c) => c.key === 'verify_engineer:verify');
    check(v && JSON.stringify(v.ins.extra_gates).includes('failing_test'), 'failing_test gate passed to verify');
  }],
  ['ci: unconfirmed culprit blocks', async () => {
    const r = await run({ mode: 'ci', task: 'c' }, {
      'ci_guardian:bisect': () => ({ culprit: 'maybe', confirmed: false, work_packages: [WP('WP1')] }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Bisect', `status ${r.result.status}`);
  }],
  ['ci: a regression without a failing test command stops before the bisect', async () => {
    const r = await run({ mode: 'ci', task: 'c' }, {
      'ci_guardian:classify': () => ({ failure_class: 'code_regression', failing_test_cmd: '', evidence: 'red since 10-21', triage_path: `${RUN}/ci.md` }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Classify', `status ${r.result.status}`);
    check(!r.keys.includes('ci_guardian:bisect'), 'no bisect without the failing test');
  }],
  ['perf: a hand-off owner outside the list is rejected', async () => {
    const r = await run({ mode: 'perf', task: 'p' }, {
      'architect:perf_plan': () => ({ work_packages: [WP('WP1')], handoffs: [{ id: 'O2', owner: 'kernel-workflow' }], gate_ok: true }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Plan', `status ${r.result.status}`);
  }],
  ['perf: no measured win drops the change', async () => {
    const r = await run({ mode: 'perf', task: 'DeepSeek-V4.1 decode on MI355X' }, {
      'perf_engineer:perf_eval': () => ({ claim_reproduced: false, held_out_ok: true, guards_ok: true, workload_unchanged: true, path_ok: true, summary: '-0.3% within noise' }),
    });
    check(r.result.status === 'dropped' && r.result.stopped_at === 'Performance', `status ${r.result.status}`);
    check(r.keys.includes('implementer:rollback'), 'change rolled back');
    check(r.result.handoffs.some((h) => h.owner === 'kernel_workflow'), 'kernel opportunity handed off');
  }],
  ['perf: with held_out on, a gain that vanishes at the held-out operating point is dropped', async () => {
    const r = await run({ mode: 'perf', task: 'p', held_out: true }, {
      'perf_engineer:perf_eval': () => ({ claim_reproduced: true, held_out_ok: false, guards_ok: true, workload_unchanged: true, path_ok: true,
        summary: '+9% at c64, -0.2% at the held-out c32 / seed 2' }),
    });
    check(r.result.status === 'dropped' && r.result.stopped_at === 'Performance', `status ${r.result.status}`);
    check(r.keys.includes('implementer:rollback'), 'the fitted change is rolled back');
    check(r.result.phases.some((p) => p.phase === 'Performance' && p.summary.includes('held-out')), 'the report says why');
    const feature = await run({ mode: 'feature', task: 't', held_out: true }, {
      'perf_engineer:perf_eval': () => ({ claim_reproduced: true, held_out_ok: false, guards_ok: true, workload_unchanged: true, path_ok: true }),
    });
    check(feature.result.status === 'blocked' && feature.result.stopped_at === 'Performance', `feature: ${feature.result.status}`);
    const noClaim = await run({ mode: 'feature', task: 't' }, {
      'director:spec': () => ({ spec_path: `${RUN}/feature_spec.md`, gates_measurable: true, perf_claim: '' }),
      'perf_engineer:perf_eval': () => ({ claim_reproduced: true, held_out_ok: false, guards_ok: true, workload_unchanged: true, path_ok: true }),
    });
    check(noClaim.result.status === 'ready_for_review' && noClaim.keys.includes('perf_engineer:perf_eval'),
      `no claim, only guards: ${noClaim.result.status}`);
  }],
  ['perf: held_out, guards and sweep are off by default, and every role sees them', async () => {
    const vanish = {
      'perf_engineer:perf_eval': () => ({ claim_reproduced: true, held_out_ok: false, guards_ok: true, workload_unchanged: true, path_ok: true }),
    };
    const perf = await run({ mode: 'perf', task: 'p' }, vanish);
    check(perf.result.status === 'ready_for_review', `perf, held_out off: ${perf.result.status}`);
    const feature = await run({ mode: 'feature', task: 't' }, vanish);
    check(feature.result.status === 'ready_for_review', `feature, held_out off: ${feature.result.status}`);
    const pe = feature.calls.find((c) => c.key === 'perf_engineer:perf_eval');
    check(pe && pe.ins.held_out === false && pe.ins.guards === false && pe.ins.sweep === false,
      `off for the perf engineer: ${JSON.stringify(pe && [pe.ins.held_out, pe.ins.guards, pe.ins.sweep])}`);
    const asked = await run({ mode: 'feature', task: 't', held_out: 'true', guards: true, sweep: true });
    const sp = asked.calls.find((c) => c.key === 'director:spec');
    check(sp && sp.ins.held_out === true && sp.ins.guards === true && sp.ins.sweep === true, 'on for the spec when the task sets them');
  }],
  ['perf: measured win to a PR', async () => {
    const r = await run({ mode: 'perf', task: 'p', apply_prs: ['#46329'] });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases, ['Setup', 'Profile', 'Plan', 'Implement', 'Verify', 'Performance', 'Review', 'PR', 'Report']), `phases ${r.phases}`);
    const p = r.calls.find((c) => c.key === 'architect:perf_plan');
    check(p && eqList(p.ins.apply_prs, ['#46329']), 'apply_prs reach the planner');
  }],
  ['perf: only kernel-level opportunities are handed off', async () => {
    const r = await run({ mode: 'perf', task: 'p' }, {
      'architect:perf_plan': () => ({ work_packages: [], handoffs: [{ id: 'O2', owner: 'kernel_workflow' }], gate_ok: true }),
    });
    check(r.result.status === 'handed_off' && r.result.stopped_at === 'Plan', `status ${r.result.status}`);
  }],
  ['scan: collect and dispatch without code changes', async () => {
    const r = await run({ mode: 'scan', framework_dir: '' });
    check(r.result.status === 'queued', `status ${r.result.status}`);
    check(eqList(r.phases, ['Collect', 'Dispatch', 'Report']), `phases ${r.phases}`);
    check(!r.keys.some((k) => k.startsWith('implementer')), 'no implementer in scan');
    check(r.result.queue && r.result.queue[0].priority === 'P0', 'queue surfaced');
    const collect = r.calls.find((c) => c.key === 'scout:collect').ins;
    check(eqList(collect.sources, ['ci', 'issues', 'upstream']) && collect.rescan === false,
      `the daily report's sources by default, no rescan: ${collect.sources} ${collect.rescan}`);
  }],
  ['scan: a report from the last 24 hours makes the run ask first; rescan=true scans again', async () => {
    const recent = (ins) => (ins.rescan
      ? { lock: 'acquired', run_dir: RUN, candidates_path: `${RUN}/candidates.json`, counts: { ci: 5, issues: 30, upstream: 40 } }
      : { lock: 'skipped', run_dir: '', candidates_path: '',
        recent: { at: '2026-11-09T01:12:00Z', url: 'https://github.com/At1a8/inferpilot/issues/3', run_dir: '' } });
    const asked = await run({ mode: 'scan' }, { 'scout:collect': recent });
    check(asked.result.status === 'needs_input' && asked.result.stopped_at === 'Collect' && eqList(asked.keys, ['scout:collect']),
      `asks before scanning again: ${asked.result.status}, calls ${asked.keys}`);
    check(asked.result.questions[0].field === 'rescan' && asked.result.questions[0].question.includes('issues/3'),
      'the question names the recent report');
    const again = await run({ mode: 'scan', answers: { rescan: true } }, { 'scout:collect': recent });
    check(again.result.status === 'queued' && again.calls.find((c) => c.key === 'scout:collect').ins.rescan === true,
      `answers={"rescan": true} scans again: ${again.result.status}`);
    const vllm = [{ name: 'vllm', url: 'https://github.com/vllm-project/vllm' }];
    const more = await run({ mode: 'scan', upstream_repos: vllm });
    check(more.calls.find((c) => c.key === 'scout:collect').ins.upstream_repos[0].name === 'vllm', 'upstream_repos reach the scout');
  }],
  ['kernel: the default back end records hand-offs and runs nothing', async () => {
    const r = await run({ mode: 'feature', task: 't' }, withKernelHandoff('decode hot path'));
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(r.result.handoffs.length === 1 && r.result.handoffs[0].status === 'recorded', `handoffs ${JSON.stringify(r.result.handoffs)}`);
    check(!r.keys.some((k) => k.startsWith('kernel_liaison')) && !r.phases.includes('Kernel'), 'no liaison, no Kernel phase');
    check(r.calls.filter((c) => !c.key.startsWith('intake')).every((c) => c.ins.kernel_backend === 'none'), 'every role sees the back end');
  }],
  ['kernel: a command back end runs the hand-off through the liaison and the gain is integrated', async () => {
    const r = await run({ mode: 'feature', task: 't', ...CMD, gpu_ids: '3' }, withKernelHandoff('decode hot path'));
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases.slice(4, 8), ['Implement', 'Kernel', 'Implement', 'Verify']), `phases ${r.phases}`);
    const h = r.result.handoffs[0];
    check(h && h.id === 'K-WP1' && h.status === 'integrated', `handoff ${JSON.stringify(h)}`);
    const runCall = r.calls.find((c) => c.key === 'kernel_liaison:run');
    check(runCall && runCall.ins.task_dir === `${RUN}/kernel/K-WP1/task` && runCall.ins.kernel_backend_command.includes('{task_dir}')
      && String(runCall.ins.gpu_ids) === '3', `the liaison gets the task, the command and the GPUs: ${runCall && JSON.stringify(runCall.ins)}`);
    const col = r.calls.find((c) => c.key === 'kernel_liaison:collect');
    check(col && col.ins.backend_result.exit_code === 0, 'the run record reaches collect');
    check(r.result.done.map((d) => d.wp).includes('K-K-WP1'), 'integration package verified by the test engineer');
  }],
  ['kernel: the geak interface runs a GEAK checkout as an outside program through the liaison', async () => {
    const r = await run({ mode: 'feature', task: 't', kernel_backend: 'geak', kernel_backend_dir: '/opt/GEAK/', gpu_ids: '3' },
      withKernelHandoff('decode hot path'));
    check(r.result.status === 'ready_for_review' && r.result.handoffs[0].status === 'integrated', `status ${r.result.status}`);
    const pkg = r.calls.find((c) => c.key === 'kernel_liaison:package');
    check(pkg && pkg.ins.kernel_backend === 'geak' && pkg.ins.kernel_backend_dir === '/opt/GEAK', 'the package phase writes GEAK lane arguments');
    const runCall = r.calls.find((c) => c.key === 'kernel_liaison:run');
    check(runCall && runCall.ins.kernel_backend_command === 'node /opt/GEAK/interface/runtime/engine/run_workflow.mjs '
      + '/opt/GEAK/kernel_workflow/kernel_lane.js --args-file {task_dir}/lane_args.json --result-file {out_dir}/result.json',
    `the default GEAK command: ${runCall && runCall.ins.kernel_backend_command}`);
    const own = await run({ mode: 'feature', task: 't', kernel_backend: 'geak', kernel_backend_dir: '/opt/GEAK', kernel_backend_command: 'geak-lane {task_dir} {out_dir}' },
      withKernelHandoff('decode hot path'));
    check(own.calls.find((c) => c.key === 'kernel_liaison:run').ins.kernel_backend_command === 'geak-lane {task_dir} {out_dir}', 'a given command replaces it');
    const plain = await run({ mode: 'feature', task: 't', ...CMD }, withKernelHandoff('decode hot path'));
    check(plain.calls.every((c) => !c.ins.kernel_backend_dir), 'no GEAK path reaches a role unless geak is chosen');
  }],
  ['kernel: an integrated kernel that fails Verify goes back to the functional kernel and is retried once with the failing case', async () => {
    const r = await run({ mode: 'feature', task: 't', ...CMD, gpu_ids: '3' }, {
      ...withKernelHandoff('decode hot path'),
      'verify_engineer:verify': (ins, n) => (n === 1 ? { all_pass: false, failures: [{ gate: 'G3' }] } : { all_pass: true, failures: [] }),
      'investigator:investigate': () => ({ verdict: 'kernel_regression', kernel_handoffs: ['K-WP1'],
        counterexample: 'G3 needle at 128k: wrong token at step 7, bf16, TP4', summary: 'goes away with K-K-WP1 reverted' }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    const inv = r.calls.find((c) => c.key === 'investigator:investigate');
    const ik = inv ? inv.ins.integrated_kernels : [];
    check(ik.length === 1 && ik[0].handoff === 'K-WP1' && ik[0].commit === 'c_K-K-WP1', `the investigator sees the integrated kernel: ${JSON.stringify(ik)}`);
    const revert = r.calls.find((c) => c.key === 'implementer:rollback');
    check(revert && revert.ins.revert === 'c_K-K-WP1', `only the kernel's commit is reverted: ${revert && revert.ins.revert}`);
    const packages = r.calls.filter((c) => c.key === 'kernel_liaison:package');
    check(packages.length === 2 && String(packages[1].ins.handoff.counterexamples).includes('needle'), 'the retry carries the failing case');
    const h = r.result.handoffs[0];
    check(h.status === 'integrated' && h.retries === 1, `handoff ${JSON.stringify(h)}`);
    check(r.keys.filter((k) => k === 'verify_engineer:verify').length === 2, 'verified again after the retry');
  }],
  ['kernel: a kernel that fails Verify again after its retry is rejected; the functional kernel stays', async () => {
    const r = await run({ mode: 'feature', task: 't', ...CMD, gpu_ids: '3' }, {
      ...withKernelHandoff('decode hot path'),
      'verify_engineer:verify': (ins, n) => (n <= 2 ? { all_pass: false, failures: [{ gate: 'G3' }] } : { all_pass: true, failures: [] }),
      'investigator:investigate': () => ({ verdict: 'kernel_regression', kernel_handoffs: ['K-WP1'], counterexample: 'G3 needle at 128k' }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    const h = r.result.handoffs[0];
    check(h.status === 'rejected' && h.counterexamples.length === 2, `handoff ${JSON.stringify(h)}`);
    check(r.keys.filter((k) => k === 'kernel_liaison:package').length === 2, 'one retry only');
    check(r.keys.filter((k) => k === 'implementer:rollback').length === 2 && !r.result.done.some((d) => d.wp === 'K-K-WP1'),
      'both integrations reverted; the functional kernel stays');
  }],
  ['kernel: a kernel_regression that names no integrated kernel stops', async () => {
    const r = await run({ mode: 'feature', task: 't' }, {
      'verify_engineer:verify': () => ({ all_pass: false, failures: [{ gate: 'G2' }] }),
      'investigator:investigate': () => ({ verdict: 'kernel_regression', kernel_handoffs: ['K-X'], counterexample: 'c' }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Verify', `status ${r.result.status}`);
    check(!r.keys.includes('implementer:rollback'), 'nothing reverted');
  }],
  ['kernel: a hand-off raised by a fix reaches the Kernel phase', async () => {
    const r = await run({ mode: 'feature', task: 't' }, {
      'verify_engineer:verify': (ins, n) => (n === 1 ? { all_pass: false, failures: [{ gate: 'G2' }] } : { all_pass: true, failures: [] }),
      'investigator:investigate': () => ({ verdict: 'fix_needed', work_packages: [WP('FIX1', { kind: 'kernel' })] }),
      'implementer:implement': (ins) => ({ wp: ins.work_package.id, status: 'done', commit: `c_${ins.work_package.id}`, unit_test_cmd: 'pytest -q t.py',
        kernel_handoff: ins.work_package.id === 'FIX1' ? { op_spec_path: `${RUN}/handoffs/K-FIX1/op_spec.md`, reason: 'the fixed kernel is slow' } : null }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    const h = r.result.handoffs.find((x) => x.id === 'K-FIX1');
    check(h && h.status === 'recorded', `the fix's hand-off reached the Kernel phase: ${JSON.stringify(h)}`);
  }],
  ['kernel: a failing back end does not block the task', async () => {
    const r = await run({ mode: 'feature', task: 't', ...CMD }, withKernelHandoff('FAIL'));
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(r.result.handoffs[0].status === 'failed', `handoff ${JSON.stringify(r.result.handoffs[0])}`);
    check(!r.result.done.some((d) => d.wp.startsWith('K-')), 'nothing integrated');
    const silent = await run({ mode: 'feature', task: 't', ...CMD }, { ...withKernelHandoff('decode hot path'), 'kernel_liaison:run': () => null });
    const col = silent.calls.find((c) => c.key === 'kernel_liaison:collect');
    check(silent.result.status === 'ready_for_review' && col && /did not run/.test(col.ins.backend_result.error),
      `a run that returns nothing is a back end that did not run: ${silent.result.status}`);
  }],
  ['kernel: a perf plan with only kernel hand-offs runs on the selected back end', async () => {
    const plan = { 'architect:perf_plan': () => ({ work_packages: [], handoffs: [{ id: 'O2', owner: 'kernel_workflow', spec_path: `${RUN}/handoffs/O2/op_spec.md` },
      { id: 'O3', owner: 'hyperloom' }], perf_claim: '-6% TPOT', gate_ok: true }) };
    const r = await run({ mode: 'perf', task: 'p', ...CMD }, plan);
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases, ['Setup', 'Profile', 'Plan', 'Kernel', 'Implement', 'Verify', 'Performance', 'Review', 'PR', 'Report']), `phases ${r.phases}`);
    const byId = Object.fromEntries(r.result.handoffs.map((h) => [h.id, h.status]));
    check(byId.O2 === 'integrated' && byId.O3 === 'recorded', `statuses ${JSON.stringify(byId)}`);
    const none = await run({ mode: 'perf', task: 'p', ...CMD },
      { ...plan, 'kernel_liaison:collect': () => ({ status: 'no_gain', speedup: 1.01, integrate_wp: null }) });
    check(none.result.status === 'no_gain' && none.result.stopped_at === 'Kernel', `no gain: ${none.result.status}`);
  }],
  ['kernel: back-end arguments are checked before anything starts', async () => {
    check(asks(await run({ mode: 'feature', task: 't', kernel_backend: 'triton-magic' }), 'kernel_backend'), 'unknown back end asked');
    check(asks(await run({ mode: 'feature', task: 't', kernel_backend: 'geak' }), 'kernel_backend_dir'), 'geak without its checkout asked');
    check(asks(await run({ mode: 'feature', task: 't', kernel_backend: 'command' }), 'kernel_backend_command'), 'command without a command asked');
  }],
  ['review: requested changes are implemented, verified, measured and reviewed again', async () => {
    const r = await run({ mode: 'feature', task: 't' }, {
      'reviewer:review': (ins, n) => (n === 1
        ? { verdict: 'changes_requested', findings: [{ severity: 'major', where: 'x.py:9', issue: 'breaks TP=8 with DP attention' },
          { severity: 'blocker', where: 'x.py:40', issue: 'fast path only for batch 64', hack: 'P7' }],
          work_packages: [WP('R1')], review_path: `${RUN}/review.md` }
        : { verdict: 'approve', findings: [], review_path: `${RUN}/review.md` }),
    });
    check(r.result.status === 'ready_for_review', `status ${r.result.status}`);
    check(eqList(r.phases.slice(6), ['Performance', 'Review', 'Implement', 'Verify', 'Performance', 'Review', 'PR', 'Report']), `phases ${r.phases}`);
    check(r.result.done.map((d) => d.wp).includes('R1'), 'the review fix passed the test engineer');
    const second = r.calls.filter((c) => c.key === 'reviewer:review')[1];
    check(second && second.ins.round === 1, 'the second review knows its round');
  }],
  ['review: a blocked review, or an approval with a blocker, stops before the PR', async () => {
    const blocked = await run({ mode: 'ci', task: 'c' }, {
      'reviewer:review': () => ({ verdict: 'blocked', findings: [{ severity: 'blocker', issue: 'changes a shared CUDA path' }], review_path: `${RUN}/review.md` }),
    });
    check(blocked.result.status === 'blocked' && blocked.result.stopped_at === 'Review', `blocked: ${blocked.result.status} at ${blocked.result.stopped_at}`);
    check(!blocked.keys.includes('pr_engineer:pr'), 'no PR after a blocked review');
    const odd = await run({ mode: 'ci', task: 'c' }, {
      'reviewer:review': () => ({ verdict: 'approve', findings: [{ severity: 'blocker', issue: 'x' }], review_path: `${RUN}/review.md` }),
    });
    check(odd.result.status === 'blocked' && odd.result.stopped_at === 'Review', 'an approval with a blocker is not an approval');
  }],
  ['review: the review rounds are bounded', async () => {
    const r = await run({ mode: 'ci', task: 'c', max_review_rounds: 1 }, {
      'reviewer:review': (ins, n) => ({ verdict: 'changes_requested', findings: [{ severity: 'major', issue: 'again' }],
        work_packages: [WP(`R${n}`)], review_path: `${RUN}/review.md` }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Review', `status ${r.result.status}`);
    check(r.keys.filter((k) => k === 'reviewer:review').length === 2, 'max_review_rounds + 1 reviews');
  }],
  ['review: review_model reaches the reviewer only', async () => {
    const r = await run({ mode: 'ci', task: 'c', review_model: 'another-model' });
    const reviews = r.calls.filter((c) => c.key === 'reviewer:review');
    check(reviews.length === 1 && reviews.every((c) => c.model === 'another-model'), `reviewer model: ${reviews.map((c) => c.model)}`);
    check(r.calls.filter((c) => c.key !== 'reviewer:review').every((c) => !c.model), 'the other roles keep the session model');
  }],
  ['perf: a workload and a model are required', async () => {
    check(asks(await run({ mode: 'perf', task: 'p', workload: '' }), 'workload'), 'perf without a workload asked');
    check(asks(await run({ mode: 'perf', task: 'p', model_path: '' }), 'model_path'), 'perf without a model asked');
    check(asks(await run({ mode: 'feature', task: 't', model_path: '' }), 'model_path'), 'feature without a model asked');
    check(asks(await run({ mode: 'feature', task: 't', server_script: '' }), 'server_script'), 'feature without a server script asked');
    check(asks(await run({ mode: 'perf', task: 'p', client_script: '' }), 'client_script'), 'perf without a client script asked');
    check(asks(await run({ mode: 'feature', task: 't', workload: '{"isl":8000}' }), 'client_script'), 'a feature with a workload but no client script asked');
    const plain = await run({ mode: 'feature', task: 't' });
    check(!asks(plain, 'client_script'), 'a feature without a workload needs no client script');
    const setupIns = plain.calls.find((c) => c.key === 'director:setup');
    check(setupIns && setupIns.ins.server_script === '/s/server.sh', 'the server script reaches the roles');
  }],
  ['perf: a profile of the wrong path stops before any plan', async () => {
    const r = await run({ mode: 'perf', task: 'p' }, {
      'perf_engineer:profile': () => ({ analysis_path: `${RUN}/perf_analysis.md`, path_ok: false,
        path_evidence: 'intended aiter attention, the log shows the triton fallback', opportunities: [] }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Profile', `status ${r.result.status}`);
    check(!r.keys.includes('architect:perf_plan'), 'no plan from an untrusted profile');
  }],
  ['perf: a measurement off the fixed workload is not evidence and drops nothing', async () => {
    const r = await run({ mode: 'perf', task: 'p' }, {
      'perf_engineer:perf_eval': () => ({ claim_reproduced: false, held_out_ok: true, guards_ok: true, workload_unchanged: false, path_ok: true, summary: 'c64 dropped' }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Performance', `status ${r.result.status}`);
    check(!r.keys.includes('implementer:rollback'), 'an untrusted measurement rolls nothing back');
  }],
  ['roles: every role and PHASE inferpilot.js calls is documented', async () => {
    const src = readFileSync(SCRIPT, 'utf8');
    const sites = [...src.matchAll(/call\('([a-z_]+)', '([a-z_]+)'/g)].map((m) => [m[1], m[2]]);
    check(sites.length >= 20, `found ${sites.length} call sites`);
    for (const [role, ph] of sites) {
      const f = path.join(INFERPILOT, 'roles', `${role}.md`);
      const doc = existsSync(f) ? readFileSync(f, 'utf8') : '';
      check(doc.includes(`## PHASE=${ph}`), `roles/${role}.md documents PHASE=${ph}`);
      check(DEFAULTS[`${role}:${ph}`] !== undefined, `dry run scripts an answer for ${role}:${ph}`);
    }
  }],
  ['roles: every knowledge file, template and script a role names exists', async () => {
    const refs = new Set();
    for (const f of readdirSync(path.join(INFERPILOT, 'roles'))) {
      const text = readFileSync(path.join(INFERPILOT, 'roles', f), 'utf8');
      for (const m of text.matchAll(/`((?:knowledge|templates|scripts)\/[A-Za-z0-9_./-]+\.(?:md|py|sh))`/g)) refs.add(m[1]);
    }
    check(refs.size >= 20, `found ${refs.size} references`);
    for (const ref of refs) check(existsSync(path.join(INFERPILOT, ref)), `${ref} exists`);
  }],
  ['skills: every role in the framework-skills index reads it', async () => {
    const index = readFileSync(path.join(INFERPILOT, 'knowledge', 'frameworks', 'sglang', 'skills.md'), 'utf8');
    const roles = [...new Set([...index.matchAll(/^\| ([a-z_]+) \| [a-z_, ]+ \|/gm)].map((m) => m[1]))].filter((r) => r !== 'role');
    check(roles.length >= 10, `index rows for ${roles.length} roles`);
    for (const role of roles) {
      const f = path.join(INFERPILOT, 'roles', `${role}.md`);
      check(existsSync(f) && readFileSync(f, 'utf8').includes('frameworks/<framework>/skills.md'), `roles/${role}.md points to the index`);
    }
  }],
  ['schemas: a malformed answer is rejected and the run stops cleanly', async () => {
    const r = await run({ mode: 'issue', task: 'i' }, {
      'issue_triager:triage': () => ({ priority: 'P5', action: 'maybe', triage_path: `${RUN}/t.md` }),
    });
    check(r.result.status === 'blocked' && r.result.stopped_at === 'Triage', `status ${r.result.status}`);
  }],
  ['arguments: an invalid or incomplete definition comes back as questions', async () => {
    check(asks(await run({ mode: 'magic', task: 't' }), 'mode'), 'unknown mode asked');
    check(asks(await run({ mode: 'issue' }), 'task'), 'missing task asked');
    check(asks(await run({ mode: 'issue', task: 't', framework: 'vllm', framework_dir: '' }), 'framework_dir'), 'missing checkout asked');
    const image = await run({ mode: 'issue', task: 't', framework_dir: '' });
    const setupIns = image.calls.find((c) => c.key === 'director:setup').ins;
    check(setupIns.framework_dir === '/sgl-workspace/sglang', `sglang defaults to the image's checkout: ${setupIns.framework_dir}`);
    check(asks(await run({ mode: 'issue', task: 't', resume_from: 'implement', run_dir: RUN }), 'resume_from'), 'resume outside feature asked');
    let msg = '';
    try { await run({ mode: 'issue', task: 't', inferpilot_dir: '' }); } catch (e) { msg = e.message; }
    check(msg.includes('args.inferpilot_dir is required'), `no InferPilot checkout is still an error: ${msg}`);
  }],
  ['intake: a complete definition starts without it', async () => {
    const r = await run({ mode: 'issue', task: 'i' });
    check(!r.keys.includes('intake:define') && r.phases[0] === 'Setup', `phases ${r.phases}`);
    check(r.result.task_def.mode === 'issue' && r.result.task_def.task === 'i', `task_def ${JSON.stringify(r.result.task_def)}`);
  }],
  ['intake: a request in plain words becomes a definition and the task runs', async () => {
    const r = await run({ request: 'DeepSeek-V4.1 在 MI355X 上 decode 太慢，8k 输入 1k 输出 64 并发，目标提升 5%', framework_dir: '' }, {
      'intake:define': () => ({ status: 'ready', summary: 'perf task',
        task_def: { mode: 'perf', task: 'DeepSeek-V4.1 decode throughput on MI355X', goal: '+5% output tok/s at c64',
          framework_dir: '/tmp/sglang', model_path: '/models/dsv41', workload: { isl: 8192, osl: 1024, conc: 64 },
          server_script: '/s/server.sh', client_script: '/s/client.sh' },
        provenance: { mode: 'inferred', workload: 'inferred', gpu_ids: 'default' }, assumptions: ['Setup picks free GPUs'], questions: [] }),
    });
    check(r.result.status === 'ready_for_review' && r.phases[0] === 'Intake' && r.phases[1] === 'Setup', `${r.result.status}: ${r.phases}`);
    const setup = r.calls.find((c) => c.key === 'director:setup');
    check(setup && setup.ins.mode === 'perf' && setup.ins.goal === '+5% output tok/s at c64'
      && setup.ins.workload.conc === 64, `setup sees the definition: ${JSON.stringify(setup && setup.ins)}`);
    check(setup && setup.ins.assumptions.length === 1, 'assumptions reach the roles');
    const intakeCall = r.calls.find((c) => c.key === 'intake:define');
    check(intakeCall.prompt.includes('MODE=undecided') && intakeCall.prompt.includes('change nothing')
      && !intakeCall.prompt.includes('restore_owner'), 'the intake is told it has no run directory and changes nothing');
    check(setup.prompt.includes('MODE=perf') && setup.prompt.includes('restore_owner.sh'), 'later roles keep the ownership rule');
    check(r.result.task_def.mode === 'perf' && r.result.assumptions.length === 1, 'the run records the definition it used');
  }],
  ['intake: missing information comes back as questions, and answers continue the task', async () => {
    const intake = { 'intake:define': (ins) => (ins.answers && ins.answers.workload
      ? { status: 'ready', task_def: { mode: 'perf', task: 'p', framework_dir: '/tmp/sglang', model_path: '/models/m', workload: ins.answers.workload, server_script: '/s/server.sh', client_script: '/s/client.sh' }, questions: [] }
      : { status: 'needs_input', task_def: { mode: 'perf', task: 'p', framework_dir: '/tmp/sglang', model_path: '/models/m', server_script: '/s/server.sh', client_script: '/s/client.sh' },
        questions: [{ field: 'workload', question: 'Which workload?', why: 'fixed before profiling', default: '{"isl":8192}' }] }) };
    const first = await run({ request: 'make it faster' }, intake);
    check(asks(first, 'workload') && first.result.next.includes('answers='), `first: ${JSON.stringify(first.result.questions)}`);
    check(first.result.run_dir === '' && !first.keys.includes('director:report'), 'nothing was created');
    const again = await run({ request: 'make it faster', answers: { workload: { isl: 8192, osl: 1024, conc: 64 } } }, intake);
    check(again.result.status === 'ready_for_review', `with the answer: ${again.result.status}`);
    const asText = await run({ request: 'make it faster', answers: '{"workload": {"isl": 8192, "osl": 1024, "conc": 64}}' }, intake);
    check(asText.result.status === 'ready_for_review', `answers as a JSON string: ${asText.result.status}`);
  }],
  ['intake: the code still checks a definition the intake calls ready', async () => {
    const r = await run({ request: 'speed up decode' }, {
      'intake:define': () => ({ status: 'ready', task_def: { mode: 'perf', task: 'p', framework_dir: '/tmp/sglang', model_path: '/models/m' }, questions: [] }),
    });
    check(asks(r, 'workload'), `a ready definition without a workload: ${r.result.status}`);
  }],
  ['intake: fields the caller set win over what the intake inferred', async () => {
    const r = await run({ request: 'look at 38849', mode: 'issue', task: 'https://github.com/sgl-project/sglang/issues/38849' }, {
      'intake:define': () => ({ status: 'ready', task_def: { mode: 'feature', task: 'something else', framework_dir: '/tmp/sglang', model_path: '/m' }, questions: [] }),
    });
    check(r.result.mode === 'issue' && r.result.task_def.task.endsWith('/38849'), `mode ${r.result.mode}, task ${r.result.task_def.task}`);
  }],
  ['intake: a request outside InferPilot is rejected with what to do instead', async () => {
    const r = await run({ request: 'push my branch and comment on the PR' }, {
      'intake:define': () => ({ status: 'rejected', task_def: {}, questions: [], reason: 'InferPilot never pushes or posts; run the push yourself' }),
    });
    check(r.result.status === 'rejected' && r.result.reason.includes('never pushes'), `status ${r.result.status}`);
    check(!r.keys.includes('director:setup'), 'nothing started');
  }],
  ['intake: an intake that returns nothing falls back to the given fields', async () => {
    const r = await run({ request: 'this issue', mode: 'issue', task: 'i' }, { 'intake:define': () => ({ status: 'maybe' }) });
    check(r.result.status === 'ready_for_review' && r.result.phases[0].summary.includes('returned nothing'), `status ${r.result.status}`);
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
