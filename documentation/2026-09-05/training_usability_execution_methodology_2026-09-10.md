# Side Plan — Execution Methodology (Recipe Guide)

**Date:** 2026-09-10
**Companion to:** `training_usability_execution_plan_2026-09-10.md` (the plan; 15 phases, 5 milestones)
**What this is:** a management layer, a side system for *running* the plan. It defines roles, a per-phase protocol, artifacts, gates, and decision rules. It is **not** the plan and never overrides it. Where this document and the plan disagree on *what to build*, the plan wins; this document governs only *how the work is driven*.

**Constraint honored:** methodology only. No code, no orchestrator.

---

## 1. Roles

| Role | Model (harness) | Job | Never does |
|---|---|---|---|
| **S** — Supervisor | GPT 5.6 Terra (hosted, via opencode) + user as gate | Decides the next protocol step; writes every prompt; audits results against the plan's acceptance criteria; revises specifications/intents when blocked; manages the review budget; stops/starts the local LLM service (Gate G0); launches live-test subagents | Writes implementation code. Deep code reading (delegated to R). Starts a review without recording why |
| **R** — Researcher | Muse Glimmer 30B (local) | Produces the pre-context bundle for a fresh executor session: which files/lines matter, what the code actually does, gotchas, expected behavior. A guide, not an implementer | Edits any file. Runs anything on the GPU |
| **E** — Executor | Qwen3.8 27B (local) | Executes exactly one phase: writes a subplan, implements, runs the CPU test suite, writes the result file | Runs live/GPU steps. Touches the LLM service (it would be killing itself). Starts the next phase |
| **RV** — Reviewer | Claude Sonnet 5 (hosted) | Blind review: fresh session, judges the phase's work against the plan's spec + intent | Sees the executor's session history. Re-implements anything |
| **L** — Live-test subagent | Any hosted model of S's choice | Executes a batch of live/GPU steps while the local service is stopped | Runs while the local service is up (that is the point). Decides pass/fail beyond the written criteria |
| **User (gate)** | — | Approvals only: review #6+, spec revisions, live-batch start/end, final model choice, anything destructive | Nothing else. The flow is designed so the user's involvement is a handful of session starts plus a few approvals per phase |

Notes:
- S is the only role that *operates the environment* (service stop/start, git, file bookkeeping). S's writes are limited to: prompts, ledger, supervisor notes, live-test prompts. S reads code only to make decisions; detailed reading is R's job.
- Every model session is **fresh** (no memory). The only continuity is the files on disk. Everything a role needs to know must be inside a prompt file.

## 2. The core constraint: local models vs live GPU work

R and E are served by the local LLM service (`llama-server` holding all 3x RTX 3090). A live GPU test needs the GPUs exclusively, which requires stopping the service, which would kill R/E mid-inference. Consequences (non-negotiable):

1. **E and R never run live tests and never touch the service.** Their phases end with live steps marked `LIVE-PENDING` in the result file (exact command, config, and pass criterion written down for later).
2. **Live work is done by L**, a *hosted* subagent (local models are unavailable while the service is down), launched by S.
3. **Live work is batched at the end of each milestone** (Section 6) to minimize stop/start cycles, except when a later phase genuinely cannot proceed without a live result (then S may run an early single-step live batch; that is a recorded deviation, and the user gate approves it).
4. **Safety ordering around every live batch:** S writes the live prompt, user approves (gate: GPU stop + time commitment), S stops the service, user starts the hosted L session, L runs and writes its result file, **S restarts the service before anything else**, S folds results back into the phase records and the ledger. S never leaves the service stopped.

## 3. Per-phase protocol

For each phase, S drives the steps below. The user's only actions are starting fresh sessions and pointing them at a file (plus gate approvals where marked).

- **[0] Decide.** S reads the plan's phase section + the ledger + (if any) the previous phase's result file. S decides: is a research pass needed? (Section 8.) S records the decision + one-line rationale in the phase's `00_supervisor.md`.
- **[1] Research (if required).** S writes `01_research_prompt.md` (self-contained: goal, files to read, questions to answer, output format, "research only, no file edits"). User starts a fresh **R** session, points it at the prompt file. R's answer is saved as `02_research_response.md`.
- **[2] Executor prompt.** S writes `03_executor_prompt.md` = the phase spec from the plan (verbatim Goal/Steps/Acceptance) + intent + R's bundle (if any) + relevant prior-phase outcomes + the standing constraints (Section 4) + artifact instructions (write subplan first, then result file).
- **[3] Execute.** User starts a fresh **E** session, points it at the prompt file. E first writes `04_subplan.md` (decomposition, risks, test plan). *Optional user pre-flight:* the user may skim the subplan before the long run; a scope deviation is cheap to catch here. E then implements, runs the full CPU test suite, and writes `05_executor_result.md`.
- **[4] Audit.** S reads `05` **and the evidence**: the actual diff, the test output (S does not trust "tests passed" without seeing output), the measured acceptance values. S decides: **pass** to [7]; **issues** re-issue [3] with a corrective note (max 2 re-issues, then user gate); **pass but high-risk/complex** send for review (budget permitting). Decision + rationale recorded in `00_supervisor.md`.
- **[5] Review (only when S decides to).** S writes `06_review_prompt.md`, the blind bundle: phase spec (verbatim), intent, the diff, the test output, the result file, and the instruction that the executor's narrative is context, not evidence. User starts a fresh **RV** session, points it at the prompt file. Output saved as `07_review.md`. Spends one of the 5-review budget (Section 5).
- **[6] Account for review.** S forwards `07` to E (short prompt: account for every finding, there is always at least a minor one; if E disagrees with a finding, say so with evidence, S decides). E updates `05` and writes `08_review_accounting.md`. S verifies the accounting closes the findings (or records the disagreement).
- **[7] Close.** S updates the ledger (status, review spent?, live steps `LIVE-PENDING`?), commits the phase (Section 4), and points to the next phase.

If a phase has live steps, its status at [7] is `done (live-pending)`, not `done`.

## 4. Standing constraints (pasted into every executor prompt)

1. Workdir: `/home/obenomar/Trade/BinanceAlgo/BinanceDataTraining`.
2. No GPU/live steps: mark them `LIVE-PENDING` in the result file with the exact command, config, and pass criterion. No stopping/starting any service.
3. Full CPU test suite must pass before the phase is reported done: `CI=true python -m unittest discover -s tests -v`. Paste the summary + any failures into the result file.
4. Result file must contain: files changed (with `file:line`), the diff, test output, each acceptance bullet answered with evidence, deviations, open issues.
5. Stay inside the phase's Write scope from the plan. A scope change is a deviation to flag, not a silent expansion.
6. Commit when done (repo is under git): message `phase N: <short description>`. One commit per phase keeps diffs clean for S's audit and for the reviewer. Before Phase 1 starts, the user should commit or stash the pre-existing dirty files (`config/observability.yaml`, `main.py`, `observability/run_state.py`, `tests/test_observability_sqlite_backend.py`) so Phase 1's diff is clean.

## 5. Review budget

- **5 blind reviews** may be spent without explicit user approval. Review #6+ requires the user gate.
- S may **reallocate** within the 5 at any time (recorded in `00_supervisor.md`).
- Recommended allocation (highest expected value first):
  1. **Phase 4** — aux vector branch: architecture change across model + dataset + distributed guard.
  2. **Phase 6** — checkpoint/resume: if broken, the multi-day final fits are at risk; recovery-state concurrency is subtle.
  3. **Phase 10** — input-pipeline parallelization: races are hard to see in a diff.
  4. **Phase 11** — label/evaluation alignment: semantic correctness with no live batch to catch it (CPU-only phase).
  5. **Phase 7** — run planner: new user-facing contract; cheap to get wrong, expensive to discover later.
- Phases 13/14/15 are protected by their live results + the plan's built-in evaluation gate; Phase 12 by its matrix design. If S's audit on any of those cannot resolve an issue, that is a gate question, not a silent 6th review.

## 6. Live-test protocol (batches)

Each plan phase marked "Gate G0 required" defers its live steps (Section 2). S batches them at milestone end. Mapping:

| Batch | When | Covers (plan phases) | Character |
|---|---|---|---|
| L1 | end of M1 | Phase 2 (one-epoch preflight) | Short; validates the frozen profile's throughput/VRAM band |
| L2 | end of M2 | Phase 4 (forward-pass/throughput), Phase 6 (interrupt/resume) | Medium; the Phase 6 test must include a kill mid-epoch |
| L3 | end of M3 | Phase 10 (before/after throughput benchmark) | Short-medium |
| L4 | end of M4 | Phase 12 (matrix arms), Phase 13 (HPO trials + 3 seed fits + eval gate) | Long; S should split L4 into two batches (12, then 13) since 13's profile depends on 12's results |
| L5 | M5 | Phase 14 (full-range fit), Phase 15a (XLA/compiled-LSTM benchmark) | Multi-day; Gate G0 step 8 applies (stated end time, daily re-confirmation, checkpoint-based pause) |

Rules:
- L's prompt (written by S, saved in the batch dir) contains: exact steps, commands, pass criteria with numeric bands, time limits, and the failure protocol (on OOM or crash: stop, record, exit; do not retry beyond the stated limit).
- L is hosted, because the local service is down. S picks the model per batch (cheapest capable model that follows a mechanical script is fine for L1/L3; something stronger for L4/L5 where judgment is needed).
- L's result file is folded back: S appends the measured values to the relevant phase result files, flips `LIVE-PENDING` to `LIVE-DONE`, and updates the ledger.
- A phase is only `done` when its live batch has closed (until then it is `done (live-pending)`).

## 7. State and artifacts

**Root:** `documentation/2026-09-05/execution/`

- `EXECUTION_LEDGER.md` — the single source of truth for *position*. A fresh S session needs only this methodology + the plan + the ledger to resume cold. Top line: `NEXT: phase N, step [k]`. Then a table: phase | status | R used? | re-issues | review spent? | live batch | open items | updated (date).
- `phase-01/` ... `phase-15/` — one dir per phase, fixed file names:
  - `00_supervisor.md` — S's decisions (research? review? spec revisions) + rationale
  - `01_research_prompt.md`, `02_research_response.md` (only if R used)
  - `03_executor_prompt.md`, `04_subplan.md`, `05_executor_result.md`
  - `06_review_prompt.md`, `07_review.md`, `08_review_accounting.md` (only if reviewed)
- `live/L1/` ... `live/L5/` — `prompt.md`, `result.md` per batch.

**Status values:** `new` to `researching` to `prompt-ready` to `executing` to `audit` to `reviewing` to `accounting` to `done`; plus `done (live-pending)` and `blocked (reason)`.

**Self-contained prompt rule.** Every prompt file must be complete for a fresh session with zero prior context: goal, exact file paths, what to read first, acceptance criteria, constraints, and where to write output. The user's gesture is always identical: start a fresh session of the right model, point it at the file. If a prompt needs "remember X from earlier", the prompt is defective; the needed content belongs in the file.

## 8. S's decision rules

**Research (step [0]):** required for phases touching unfamiliar multi-file internals or semantic correctness: **3, 4, 5, 6, 7, 10, 11**. Optional (S decides at [0]): **13** (eval-gate plumbing), **15** (pruning sub-item only). Not needed (the plan already cites everything; mechanical): **1, 2, 8, 9, 12, 14**. R is cheap (local); when in doubt, research — the cost is one session start, and a good bundle reduces E re-issues, which are far costlier.

**Review (step [4]):** spend when (a) S's audit finds an issue it cannot resolve by re-issuing, or (b) the phase is on the Section 5 allocation list and the work is substantial. Otherwise S's audit + CPU tests suffice.

**Re-issues:** max 2 corrective re-issues of the executor per phase; then user gate.

**Escalation to the user (gate), mandatory for:** review #6+; any spec/intent revision; starting or deviating from a live batch (incl. any mid-run pause, per Gate G0 step 8); a subplan that deviates in scope (pre-flight); Phase 15 final model choice; anything destructive (deletion, re-snapshotting); 2 failed re-issues.

**Spec/intent revision protocol:** when a blocker makes a plan step infeasible or suboptimal, S may revise. Record in `00_supervisor.md`: original text, revised text, why. If the revision touches a plan **acceptance criterion**, it is a gate question before E proceeds. The plan file itself is edited only for accepted revisions (S edits it, records it).

## 9. Definition of done (per phase)

All of the following, checked by S at [4]/[7]:
1. Every acceptance bullet of the plan's phase answered in the result file with evidence (measurement, output, or diff).
2. Full CPU test suite green, output in the artifact.
3. Live steps either `LIVE-DONE` (batch closed) or `LIVE-PENDING` (exact command + pass criterion written).
4. Ledger updated; phase committed to git with the standard message.

## 10. What the user actually does (manual flow)

Per typical phase: start R (if S says so), start E, optionally skim the subplan, start RV (if S says so). Per milestone end: approve the live batch, start the hosted L session, approve the close. Approvals only where Section 8 says gate. Nothing else. The user never has to carry state in their head; the ledger says where things stand.

Expected session starts across the whole plan: R for 7-9 phases, E for 15 phases plus a few re-issues, 5 for RV, 5 for L. S is the user's ongoing hosted session.

## 11. Optional semi-automation (deferred; description only, no code)

The user runs sessions manually; there is no orchestrator. If friction proves real after M1, a minimal helper is spec'd here (to be built then; single small CLI, a few hours of LLM work, no API calls, no model orchestration):

- `next` — prints the ledger's current position and which file to point the next session at.
- `render <phase> <step>` — places S's prompt into the phase dir with the fixed name (S still writes the content; this only files it).
- `log <phase> <status>` — appends a timestamped line to the ledger.

Honest assessment: it saves about 10 minutes of bookkeeping per phase; every model session stays manual. Fixed names + the ledger already make the manual flow mechanical, so the recommendation is to **skip it for M1** and revisit only if the overhead is felt.

## 12. Additions beyond the specified protocol (and why)

The user's protocol (roles, steps [0]-[7], 5-review budget, local-LLM deferral) is kept intact. Additions:

1. **The execution ledger** — fresh S sessions must resume cold; the ledger is the single resumption artifact. Highest-value addition.
2. **Fixed per-phase artifact layout** — makes the manual flow mechanical and auditable; every prompt/response/result has exactly one home, so "point the session at this file" always means the same thing.
3. **Self-contained prompt rule** — the fresh-session reality made explicit; also defines the one uniform user gesture for all sessions.
4. **Optional subplan pre-flight** — E writes the subplan before the long run; a 2-minute user skim catches scope drift before it costs an hour of local inference.
5. **`LIVE-PENDING` deferral + milestone live batches (L1-L5)** — the consequence of the local-LLM constraint, made operational: E never touches the GPU, L is always hosted, and the service is stopped the minimum number of times.
6. **Safety rule: local models never stop the LLM service** — they would be killing the process serving their own inference; only S (hosted) performs stop/start, and always restarts before continuing.
7. **Blind-review contract** — reviewer gets spec + intent + evidence (diff, test output, result file) and is told the executor's narrative is context, not evidence; this is what makes "blind" operational rather than aspirational.
8. **Review budget allocation recommendation** (4, 6, 10, 11, 7) + reallocation rule — the 5 budget is a real constraint; spending it on the highest-risk phases maximizes value.
9. **Retry policy (max 2) + explicit escalation list** — bounds the worst case of a weak local executor and defines exactly when the user must be asked.
10. **S never writes implementation code** — keeps S's context clean for decisions; implementation stays with E, whose subplan makes its reasoning inspectable.
11. **Definition of done + one git commit per phase** — gives S's audit and the blind reviewer a clean, bounded diff; also gives the user a per-phase rollback point.
12. **S audits evidence, not claims** — the result file must contain actual test output and measurements; "done" without evidence is not done.
