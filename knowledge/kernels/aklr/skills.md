# AKLR's own documents and skills — which role reads which, from an AKLR checkout

[AKLR](https://github.com/AMD-AGI/AKLR) is AMD's kernel library for ROCm. Its kernels reach SGLang and
vLLM through runtime plugins: a route replaces one framework function and dispatches to an AKLR op when
the call's workload has a qualifying kernel, and falls back to the framework's own code otherwise.
AKLR is maintained by its own team, and it keeps its conventions in its repository: tutorials, the
acceptance criteria, the PR template and agent skills under `harness/skills/`. They change with AKLR
(AKLR #153 set one 5% kernel bar for admission and dispatch), so Petra **references them in place and
never copies them**: a role reads the file in a checkout of AKLR's latest `main`, and takes every
threshold from there. Index checked against AKLR main `78496c03` (2026-11-16).

What Petra adds on top — when a kernel gain belongs in AKLR at all, and what AKLR's reviewers have asked
of GEAK kernels — is in [`playbook.md`](playbook.md).

## Precedence

1. `../../guardrails.md` and Petra's gates come first. AKLR's `aklr-open-pr` pushes, labels and opens
   PRs: Petra reads it for what the PR must carry and stops before those steps; a human pushes and
   opens the PR (as AKLR's own `harness/rules.md` rule 8 also says).
2. Within those limits, AKLR's documents decide how a thing is done in AKLR: layout, names, thresholds,
   the PR body. Follow the version on AKLR's latest `main`.
3. `playbook.md` adds what AKLR's documents do not say. Where it disagrees with them, AKLR wins and the
   curator corrects the playbook.
4. A document named here that is missing on AKLR's `main`: skip it and say so in your answer.

## Index

Paths are relative to the AKLR checkout.

| Role | Phase | AKLR files | When to read |
|---|---|---|---|
| architect | design, perf_plan | `.github/README.md` (Development), `docs/ops_tutorial.md`, `docs/plugin_tutorial.md`, `docs/acceptance_criteria.md` | a kernel-level opportunity may land in AKLR: op or plugin, the bars it must clear |
| kernel_liaison | kernel | `docs/acceptance_criteria.md`, `harness/skills/geak-campaign/SKILL.md` | an `improved` result may land in AKLR (`../../kernel_backends.md` §2) |
| implementer | implement | `aklr/ops/README.md`, `aklr/plugins/README.md`, `harness/skills/aklr-port-generated-kernels/SKILL.md`, `CONTRIBUTING.md` | porting a kernel into AKLR: schema fields, kernel interface, plugin layout, route rules, provenance, pre-commit, the AITER baseline |
| verify_engineer, perf_engineer | verify, perf_eval | `docs/acceptance_criteria.md`, `harness/gates.md` | the kernel bar (`python -m aklr validate`) and the end-to-end off/on run |
| reviewer | review | `harness/skills/aklr-review-kernel-pr/SKILL.md`, `harness/skills/aklr-bad-cases/SKILL.md`, `harness/rules.md` | reviewing the AKLR branch as AKLR's maintainers would |
| pr_engineer | pr | `.github/pull_request_template.md`, `harness/skills/aklr-open-pr/SKILL.md` | the PR body, title and the evidence each kind of change needs |

Not used by Petra: `harness/skills/aklr-build-runtime-image`, `aklr-monitor-upstreams`,
`aklr-qualify-downstream`, `aklr-read-validation`, `aklr-kb`, `forge-loop-campaign`, `remote-gpu*`,
`report-progress` (AKLR's own release and campaign work).
