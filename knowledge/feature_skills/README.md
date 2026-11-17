# feature_skills — validated development plans for one feature

> **Contract.** A feature skill is an **advisory prior**: a validated, reusable plan for landing one
> feature (`feature × framework × model family × platform`). It shortens Recon and Design and lists the
> pitfalls already paid for. It never replaces the gates in `../validation_doctrine.md`: a run that uses
> a skill still runs every gate, and when the target code or a measurement disagrees with the skill, the
> code and the measurement win and the skill is marked `stale`.

## How it differs from other knowledge

A feature skill is a development plan for a feature on a stack, judged by InferPilot's gates (G0–G5):
not an optimization recipe for one operator in one scenario, judged by a speed A/B, and not a single
distilled principle.

## Layout

```
feature_skills/
├── README.md              # this contract
├── index.yaml             # selector registry (one entry per skill)
├── _template/SKILL_TEMPLATE.md
└── <id>/
    ├── SKILL.md           # REQUIRED: frontmatter selector + the plan
    ├── scripts/           # optional: unit tests, eval clients the validation recipe uses
    └── references/        # optional: design notes, diagrams, captured configs
```

## Selection

First the candidates: `python3 scripts/find_skill.py "<task title, request, goal>"` lists the skills
one of whose `tags`, id or feature name occurs in the text (case-insensitive, whole words, `-` and `_`
read as spaces), best match first. Tags only widen the net — a task that says "sparse attention" or
"KV cache offload" reaches the skill whose feature name is `hisparse_v2`. Then a run keeps the
candidates whose `match` fits the task:

```
match.framework == task.framework
AND match.feature == task.feature            (or the task is "port <feature> to <target>")
AND task.model_arch ∈ match.model_arch       (or the skill is the closest source to port from)
AND task.platform ∈ match.platforms          (a different platform is a port: read, do not reuse blindly)
AND validation_status ∈ {validated, stale}   (stale = read as reference, re-verify every step)
```

There is no ranking. A matching skill is read in full; a near match (same feature, different model or
platform) is the starting point of a port — its *assumption table* tells you what to re-check.

## Lifecycle

`draft` → (all gates pass with recorded numbers and artifacts) → `validated` → (the framework moved past
`framework_ref` and a hook point named in the skill changed, or a gate result contradicts the skill) →
`stale` → (re-validated) → `validated`. A skill whose plan fails its gates is `failed` and is kept only
as a reference.

## Required sections (see the template)

When to use · Source design · Assumption table (source → target) · Design as built · Invariants ·
Work packages (ordered, with files) · Launch configuration · Validation recipe (gates, commands,
expected numbers) · Pitfalls (symptom → cause → fix) · Do-no-harm · Limitations and open items ·
Porting checklist · Scripts · Sources.

## Adding a skill

1. Copy `_template/SKILL_TEMPLATE.md` to `<id>/SKILL.md` and fill every section; put the scripts the
   validation recipe runs under `<id>/scripts/`.
2. Add an entry to `index.yaml` with `validation_status: draft` and the same `tags` as the skill's
   frontmatter: English words a task about the feature may use (its name and aliases, the algorithm
   or subsystem it belongs to, the techniques it is built on), not the model or platform. Check that
   `scripts/find_skill.py` finds it from a few phrasings of such a task.
3. Run the gates; record the measured numbers, the framework commit, model and GPU in the frontmatter's
   `validation` block, with pointers to the artifacts (PR, branch, result files).
4. Set `validated` only when every gate the skill claims has a recorded result.
