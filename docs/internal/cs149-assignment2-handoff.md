# CS149 Assignment 2 Handoff

Source: <https://github.com/stanford-cs149/asst2>

Use this doc when turning Stanford CS149 Assignment 2 into BrowserGrad and
craftingattention lab slices. Keep it as a benchmark profile record, not
BrowserGrad package identity.

## Upstream Shape

- The assignment teaches a task execution library: synchronous bulk launch,
  asynchronous launch, and task graphs with dependencies.
- Native upstream assumptions include C++ builds, thread pools, timing
  comparisons, and scheduler behavior.
- The browser first slice should preserve dependency and scheduling semantics
  before trying to mimic native pthread performance.

## Browser-Safe First Slice

- Start from `docs/internal/cs149-assignment2.profile.json`.
- Register `_bg_task_graph` as the profile-local JS module backed by
  `@unlocalhosted/browsergrad-primitives` `simulation.createTaskGraphSimulator()`.
- Use deterministic task graphs to verify:
  - tasks do not start before dependencies finish.
  - ready tasks are assigned to available workers.
  - makespan and completion order match the expected schedule.
  - dependency cycles fail clearly.
- Keep `performance-rubric` as a concept gate, not as a native timing clone.

## Non-Portable Upstream Assumptions

- C++ compilation in the browser runtime.
- Native pthread scheduling and host timing parity.
- Exact upstream wall-clock speedups.

## Platform Work

- Mount rubric/starter files under `/assignments/cs149-assignment2`.
- Provide `_bg_task_graph` to JavaScript rubrics.
- Keep the platform proof wired through
  `runVerifiedAssignmentJavascriptProfile()` against the real
  `cs149-assignment2.profile.json`, not a hand-written local fixture. The
  CraftingAttention e2e suite should prove:
  - `simple_test_sync`: single task completes with makespan `1`.
  - `mandelbrot_chunked`: independent chunk tasks fill a fixed worker pool.
  - `ping_pong_equal`: equal-duration ready tasks start together on separate
    workers.
  - `super_light_async`: dependent async batches preserve completion order.
  - `task_graph_dependencies`: root/left/right dependency graph has makespan
    `6`, deterministic start/finish events, and cycle rejection.
- Render failures as task-graph feedback, for example:
  - `dependent task started before parent finished`
  - `task graph makespan mismatch`
  - `ready task was never scheduled`
  - `dependency cycle was not rejected`

## Later Slices

- Add Worker-backed task execution after deterministic simulator semantics are
  stable.
- Add native/external runner handoff for C++ task-library implementations.
- Add small visual traces for worker assignment and critical-path timing.
