# CS336 Assignment 3 Scaling Handoff

Upstream source: <https://github.com/stanford-cs336/assignment3-scaling>

Use this doc when turning CS336 A3 into BrowserGrad/craftingattention lab
slices. Keep the assignment profile generic and route native server/JAX work as
capability choices, not hardcoded runtime branches.

## Current Upstream Facts

- Spring 2026 assignment repo title is "CS336 Spring 2026 Assignment 3:
  Scaling".
- Student path sets `A3_API_KEY` to an 8-digit student ID.
- Hosted training API base URL is `http://hyperturing.stanford.edu:8000`.
- Client methods call `/budget`, `/submit`, `/experiments`,
  `/experiment/{id}`, `/final_submission`, and `/dashboard`.
- Non-student/local path can run the server stack with extra deps:
  FastAPI/API, dispatcher, Postgres URLs, Modal tokenized-data download, and
  direct JAX training.
- Portable upstream tests cover budget accounting, duplicate training-config
  rejection, experiment listing/getting, final submission, and scheduler
  selection fairness.

## BrowserGrad Slice

Start from `docs/internal/cs336-assignment3-scaling.profile.json`.

Use `@unlocalhosted/browsergrad-primitives` as the first browser-safe reference:

- `createHostedTrainingApiFixture()` for fixture-scale hosted API behavior.
- `selectExperimentsForDispatch()` for scheduler fairness.
- `fitPowerLawScalingLaw()` for scaling-law projection rubrics.

Declare these capabilities:

- `hosted-api-mock`
- `server-fixture`
- `scheduler-simulator`
- `scaling-law-oracle`
- `http-client`

Keep these as external or future runner paths:

- `jax-external`
- `postgres-external`
- `modal-external`
- `wandb-external`

## Portable Tests

Portable as JS/TS or Pyodide rubrics:

- `test_dashboard_serves_html`
- `test_budget`
- `test_submit_jobs`
- `test_submit_rejects_duplicate_training_config`
- `test_final_submission_accepts_training_config_and_predicted_loss`
- `test_select_experiments_for_dispatch_orders_by_running_count_then_queue_time`
- `test_select_experiments_for_dispatch_counts_running_jobs_without_capacity`

Browser-safe replacements:

- Replace live `requests` calls with `createHostedTrainingApiFixture()`.
- Replace SQLAlchemy/Postgres scheduler setup with plain JSON experiment
  fixtures passed to `selectExperimentsForDispatch()`.
- Replace live training runs with deterministic loss fixtures and
  `fitPowerLawScalingLaw()`.

External-only:

- Full JAX training.
- Modal data download.
- Real hosted server/dispatcher/Postgres deployment.
- W&B logging.

## Platform Handoff

craftingattention should:

1. Import `@unlocalhosted/browsergrad-primitives` in the assignment-profile E2E.
   This is now proven: CraftingAttention loads the real
   `cs336-assignment3-scaling.profile.json`, selects the Pyodide route with a
   simulated scheduler capability, and exercises hosted API, scheduler, and
   scaling-law primitives through the platform e2e suite.
2. Register the scaling reference through `runAssignmentJavascriptRubric()` or
   Pyodide JS module bridge.
3. Mount small JSON fixtures for training configs, queued/running experiments,
   and expected scaling-law outputs.
4. Render preflight rows for browser, simulated, and external routes:
   - browser/simulated: hosted API mock + scheduler simulator.
   - external: hosted server/JAX/Postgres runner.
5. Show duplicate-config, budget-exceeded, and non-positive scaling samples as
   rubric failures with clear student-facing messages.

## Acceptance

- BrowserGrad package tests pass:
  `pnpm --filter @unlocalhosted/browsergrad-primitives test`.
- BrowserGrad runtime integration proves the real Python-rubric path can call a
  primitive-facade JS oracle from Pyodide:
  `pnpm --filter @unlocalhosted/browsergrad-runtime test:integration -- assignment-runner`.
- Root profile matrix tests keep A3 expected status `simulated`.
- Platform E2E consumes the package as one declared oracle before adding a full
  A3 content unit.
