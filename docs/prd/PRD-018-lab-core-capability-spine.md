# PRD-018 — Lab Core Capability Spine

## Problem Statement

BrowserGrad needs to support many guided labs from many courses without turning
each assignment into custom platform code. The immediate benchmark set includes
Stanford CS336 assignments 2 through 5, CS149 systems assignments, GPU Puzzles,
and CS149GPT. These labs span Python, PyTorch, JAX, hosted APIs, text/data
pipelines, CUDA/Triton-style kernels, C++/ISPC/OpenMP, distributed training,
vLLM, and snapshot-heavy RL rubrics.

The current runtime already has lab manifests, assignment profiles, JS oracle
registration, and tokenizer oracles, but it does not yet expose a systematic
capability spine that lets a platform answer: can this lab run here, what needs
to be mounted, which gates are browser-native replacements, and which upstream
tests are portable, replaced, future-gated, or intentionally external?

Without that spine, every course port risks becoming a one-off. BrowserGrad
would either overfit to Pyodide-only labs or prematurely give up on native-style
systems work that could be approximated through WebGPU, Worker meshes, custom
oracles, simulator layers, and future native runners.

## Solution

Create a lab-core capability spine in `@unlocalhosted/browsergrad-runtime` and
the surrounding platform docs. The first implementation should be a small,
tested public API for extracting and evaluating assignment-profile capability
requirements. Later slices will use the same vocabulary to drive file mounts,
rubric execution, worker watchdogs, JS/TS oracle registration, WebGPU kernel
labs, distributed simulators, and craftingattention platform handoffs.

The capability spine should stay assignment-agnostic. CS336 and CS149 are
benchmark probes: they should pressure-test the vocabulary, not become hardcoded
runtime branches.

## User Stories

1. As a platform author, I want to load any assignment profile and see missing capabilities, so that I can route the learner to the right runtime or explain why a lab is unavailable.
2. As a platform author, I want capability gates to support alternatives, so that a lab can accept WebGPU, native Dawn, or a future CUDA-like subset without rewriting the profile.
3. As a rubric author, I want browser-safe gates to be declared separately from native-only upstream checks, so that failures teach the concept rather than browser internals.
4. As a course-porting agent, I want each benchmark assignment mapped to capability families, so that implementation work can be sliced by reusable substrate.
5. As a systems-lab author, I want CUDA/Triton/ISPC/OpenMP assumptions represented as explicit compatibility targets, so that BrowserGrad can progressively approximate them instead of abandoning them.
6. As a data-lab author, I want datasets, network access, classifier dependencies, and large-file behavior modeled in profiles, so that Common Crawl-style labs can run on small fixtures first.
7. As an alignment-lab author, I want snapshot fixtures, tokenizer/model oracles, and GPU-only inference gates represented, so that RLHF/GRPO labs can separate math correctness from heavy training.
8. As a learner, I want clear preflight messages, so that I know whether a lab is runnable, partially simulated, or waiting for a stronger runtime.
9. As a future agent, I want a handoff doc for craftingattention, so that platform UI work can consume BrowserGrad capabilities without reading runtime internals.

## Research Dossier

- Repo exploration: `packages/browsergrad-runtime/src/assignment.ts` already
  defines assignment profiles, runtime packages, file paths, timeouts, allowed
  tests, oracles, datasets, and generic gates. `docs/platform/assignment-authoring.md`
  and `docs/platform/curriculum-platform-architecture.md` already describe
  assignment profiles, browser-safe gates, fixtures, and JS oracles. This PRD
  deepens that seam instead of creating a parallel lab framework.
- Course source: https://cs336.stanford.edu/ describes CS336 as implementation
  heavy, with assignments covering systems optimization, scaling laws, data
  processing, and alignment/RL. The course page explicitly points at GPU and
  multi-machine work for systems optimization.
- Benchmark repo: https://github.com/stanford-cs336/assignment2-systems uses
  Python 3.12+, PyTorch, tests for FlashAttention forward/backward, Triton
  variants, DDP, FSDP, and sharded optimizer behavior.
- Benchmark repo: https://github.com/stanford-cs336/assignment3-scaling uses a
  hosted training API plus optional server stack with FastAPI, Postgres, JAX,
  Modal, W&B, tokenized-data download, and scheduler tests.
- Benchmark repo: https://github.com/stanford-cs336/assignment4-data uses raw
  data processing fixtures plus dependencies such as WARC readers, fastText,
  tiktoken, transformers, psutil, Modal, and PyTorch. Tests cover language ID,
  quality filters, toxicity, HTML extraction, deduplication, and PII masking.
- Benchmark repo: https://github.com/stanford-cs336/assignment5-alignment uses
  PyTorch/transformers, optional GPU dependencies including flash-attn and vLLM,
  GRPO/DPO/SFT tests, and `.npz` snapshot fixtures. Its optional safety/RLHF
  supplement is part of the benchmark context.
- Benchmark repo: https://github.com/srush/gpu-puzzles teaches CUDA-shaped
  kernels through interactive puzzles. It is useful as a browser-native kernel
  lab reference because the exercises are small, visual, and concept-focused.
- Benchmark repos: https://github.com/stanford-cs149/asst1,
  https://github.com/stanford-cs149/asst2, and
  https://github.com/stanford-cs149/asst3 cover CPU threads, SIMD intrinsics,
  ISPC, task systems, CUDA SAXPY, scan, and rendering. They require a capability
  model that can express native build/runtime assumptions and browser
  substitutions separately.
- Benchmark repo: https://github.com/stanford-cs149/cs149gpt uses C++/ISPC
  custom PyTorch modules for attention optimization, making it a bridge between
  systems labs and ML-model labs.
- Browser/runtime implication: Pyodide remains useful for Python rubrics, but
  the benchmark set requires TS/JS oracles, WebGPU/WGSL kernels, Worker mesh
  simulators, dataset fixtures, hosted API mocks, snapshot comparison, native
  build gates, and future native runner hooks.

## Grill Decisions

1. Question: Should the lab core start with a full assignment runner or a
   smaller capability API?
   Recommended answer: Start with a capability API that profiles and platform
   UIs can call before any runner launches.
   Decision: Ship the capability spine first, then layer runner execution on
   top of stable preflight output.
2. Question: Should capabilities be hardcoded around CS336 and CS149?
   Recommended answer: Use CS336/CS149 only as benchmark inputs and keep the
   public API course-agnostic.
   Decision: Capabilities are string identifiers plus evaluated gates; benchmark
   docs may recommend names, but runtime code stays generic.
3. Question: Should a missing CUDA/Triton capability make a lab impossible?
   Recommended answer: No. It should mark the exact gate missing while allowing
   browser-native replacements or future runner alternatives to satisfy the
   same profile.
   Decision: Capability gates support alternatives so WebGPU, native Dawn,
   CUDA-like subsets, or external runners can be modeled without rewrites.
4. Question: Should platform handoffs live in craftingattention only?
   Recommended answer: No. BrowserGrad should keep source-of-truth handoff docs,
   then log craftingattention issues that point to those docs.
   Decision: Add BrowserGrad docs under `docs/platform/` and mirror actionable
   platform work into `unlocalhosted/craftingattention` issues.

## Novelty Reach

- Novel idea considered: treat course assignments as benchmark probes for a
  general browser-lab capability graph rather than as one-off ports.
- Why selected or rejected: selected because it lets BrowserGrad learn from
  CS336/CS149 without letting either course define core architecture.
- Novel idea considered: define progressive compatibility gates where a native
  requirement can be satisfied by WebGPU/WGSL, Worker mesh simulation, a pure
  TS oracle, a native Dawn runner, or a future CUDA-like educational subset.
- Why selected or rejected: selected because it preserves ambition while still
  allowing small browser-native slices to ship.
- Novel idea considered: make craftingattention consume BrowserGrad handoffs
  through GitHub issues for every lab handoff and implementation slice.
- Why selected or rejected: selected because it gives platform work a durable
  queue without polluting BrowserGrad runtime code with UI assumptions.

## Implementation Decisions

- Add public runtime helpers for capability preflight:
  - `requiredAssignmentCapabilities(profile)` returns de-duplicated capability
    identifiers implied by profile gates.
  - `evaluateAssignmentCapabilities(profile, environment)` returns whether
    required capability gates are satisfied and explains missing requirements.
- Extend capability gate options conservatively:
  - `requires`: all listed capabilities must be present.
  - `any_of`: at least one listed group of capabilities must be present.
  - `message`: optional platform-facing explanation.
- Keep non-capability gates such as streaming, timeout, and forbidden-read as
  separate rubric/runtime gates. The capability API should report capability
  readiness, not execute behavioral gates.
- Keep identifiers generic and string-based. Recommended names can be documented
  in platform docs, but runtime code should not contain a closed enum for every
  course or backend.
- Use the existing `AssignmentProfile` type rather than adding another manifest
  format.
- First stable guarantee: a platform can evaluate a parsed profile against an
  available capability list and get deterministic missing-capability output.
- Next stable guarantee: a platform can label available capabilities as
  `browser`, `simulated`, or `external` and call
  `assignmentRunReadiness(plan)` to render a deterministic learner-facing state:
  `runnable`, `simulated`, `external-only`, or `blocked`.
- Environment-builder guarantee: `createAssignmentCapabilityEnvironment()`
  converts browser, simulated, and external capability groups into one
  deterministic environment with mode labels, so platforms do not duplicate
  capability-map construction.
- Platform preflight guarantee: `createAssignmentPreflightReport(profile,
  environment)` returns the run plan, readiness, rubric kind, required
  capabilities, runner route, mount plan, and dataset cache plan as one readonly
  handoff object, and benchmark profile tests assert readiness for the
  CS336/GPU Puzzles/CS149 probe set.
- Benchmark-matrix guarantee:
  `createAssignmentBenchmarkPreflightMatrix(profiles, environment, contents?)`
  turns many profile reports into platform-ready rows with readiness, runner,
  capability, content-gap, cache-strategy, and external-runner fields. This is
  the handoff shape for platform dashboards and cross-course smoke tests.
- Runner-route guarantee: `assignmentRunnerRoute(plan)` maps preflight output
  to `pyodide`, `javascript`, `external`, `unsupported`, or `blocked`, so
  platform launch controls do not duplicate BrowserGrad's rubric/readiness
  branching.
- External-runner guarantee: `createAssignmentExternalRunnerRequest(plan)`
  packages external-only plans into a deterministic native/hosted runner
  handoff with selected capabilities, resolved files, timeouts, mount/cache
  metadata, and behavioral gates; external preflight reports include this
  request directly.
- Gate-route guarantee: each capability gate evaluation exposes `status`,
  `selectedAnyOf`, and `selectedCapabilities`, so platform preflight rows can
  show the exact browser/simulated/external path selected by BrowserGrad.
- Behavioral-gate schema guarantee: profile parsing validates `streaming`,
  `forbidden-read`, and `timeout` gate option shapes before rubrics consume
  those declarations.
- Watchdog guarantee: Python rubric exec requests use the shortest declared
  runtime watchdog among `test_ms` and `worker_ms`, and honor `worker_ms` when
  no `test_ms` is declared.
- Streaming-gate guarantee: Python rubrics can use `browsergrad.streaming_gate`
  to enforce `max_chunks_before_first_yield` from behavioral gates and fail eager
  iterable consumers with assignment-specific messages; JS/TS rubrics can use
  `createStreamingGate().wrapInput()` and `.wrapOutput()` for the same behavior.
- Forbidden-read guarantee: Python rubrics can use
  `browsergrad.forbidden_read_gate` to permit incremental text consumption while
  rejecting eager `read()` / `readlines()` calls declared by `forbidden-read`
  gates.
- Mount-readiness guarantee: `evaluateAssignmentMountContents(mountPlan,
  contents)` dry-runs platform-provided files and datasets before any
  `Session.fs` write, and benchmark tests assert every profile reports missing
  rubric/dataset contents deterministically; `createAssignmentMountPreflightReport`
  bundles content readiness and hash checks for platform preflight UI.
- Fixture-hash guarantee: `verifyAssignmentMountContentHashes(mountPlan,
  contents)` validates dataset `sha256:<64 hex>` hashes for text and binary
  mount contents before execution.
- Dataset-cache guarantee: `createAssignmentDatasetCachePlan(mountPlan)` gives
  platform fetch/cache layers deterministic cache keys and paths, using valid
  SHA-256 declarations as content addresses while surfacing malformed or
  unsupported hashes as explicit preflight states.
- Binary fixture guarantee: assignment mount contents and `Session.fs.write`
  accept `Uint8Array` bytes so small `.pt`, `.npz`, and snapshot fixtures can
  be mounted without text/base64 loss; hosts can verify worker bytes with
  `Session.fs.readBytes(path)`, and JS rubrics can inspect mounted bytes with
  `ctx.readBytes(path)`.
- Kernel-rubric guarantee: `@unlocalhosted/browsergrad-kernels` exports
  `createKernelRubric()` so JS/WebGPU labs can collect pass/fail assertions and
  compare tensor outputs against CPU oracles without requiring Pyodide or a
  real GPU.
- Kernel-runtime bridge guarantee: `createBrowsergradKernelRubric(ctx)` adapts
  kernel tensor checks to `runAssignmentJavascriptRubric()` contexts, and a
  cross-package integration test proves pass/fail assertions survive through
  BrowserGrad's JS rubric runner.
- Later implementation slices:
  - Assignment runner plan: packages, mounts, JS oracles, timeout/watchdog, and
    allowed tests.
  - Fixture registry: small checked-in fixtures, large external datasets, hashes,
    OPFS caching, and snapshot comparison.
  - Browser kernel lab core: WGSL puzzle runner, CUDA-like educational subset,
    and performance/correctness rubrics.
  - Distributed simulator: Worker mesh and deterministic event traces for DDP,
    FSDP, sharding, and task scheduling.
  - External/native runner bridge: explicit gates for vLLM, flash-attn, CUDA,
    ISPC, OpenMP, and C++ build steps.
- Benchmark capability families:
  - CS336 A2: `pyodide`, `torch-compat`, `webgpu`, `triton-compatible`,
    `worker-mesh`, `distributed-simulator`, `snapshot-oracle`.
  - CS336 A3: `http-client`, `hosted-api-mock`, `server-fixture`, `jax-external`,
    `postgres-external`, `scheduler-simulator`.
  - CS336 A4: `dataset-fixture`, `large-file-streaming`, `warc-reader`,
    `classifier-oracle`, `dedupe-oracle`, `pii-oracle`, `network-gated`.
  - CS336 A5: `torch-compat`, `transformers-compatible`, `snapshot-oracle`,
    `tokenizer-oracle`, `rl-loss-oracle`, `vllm-external`, `flash-attn-external`.
  - GPU Puzzles / CS149 A3: `webgpu`, `wgsl-kernel`, `cuda-compatible-subset`,
    `kernel-visualizer`, `performance-rubric`.
  - CS149 A1/A2/CS149GPT: `native-cpp-external`, `pthreads-simulator`,
    `simd-simulator`, `ispc-external`, `openmp-external`, `attention-oracle`.

## Testing Decisions

- Follow tracer-bullet TDD. Do not write a broad imagined suite first.
- First RED test: capability preflight accepts all-of and any-of gates and
  reports missing alternatives through the public runtime API.
- Second RED test: malformed capability gate options produce profile parse
  errors instead of silently passing unusable gate specs.
- Third RED test: required capabilities are de-duplicated deterministically and
  ignore non-capability behavioral gates.
- Later RED test: capability environment construction de-duplicates browser,
  simulated, and external groups, sorts them deterministically, and gives
  browser-native support precedence when the same capability appears in multiple
  groups.
- Later RED test: runner-route construction maps Python rubrics to Pyodide, JS
  rubrics to the browser-native runner, external-only readiness to external
  launch, failed readiness to blocked, and unknown rubric kinds to unsupported.
- Later benchmark test: CS336 Assignment 5 and CS149GPT profile drafts produce
  `externalRunnerRequest` objects under external capability environments.
- Later RED test: batch benchmark matrix helper returns one flattened row per
  profile and marks the matrix not-ok when required content or datasets are
  missing, without duplicating preflight semantics in platform code.
- Run focused package tests:
  - `pnpm --filter @unlocalhosted/browsergrad-runtime test -- assignment`
  - `pnpm --filter @unlocalhosted/browsergrad-runtime typecheck`
- Run `pnpm validate:prd docs/prd/PRD-018-lab-core-capability-spine.md`.
- Acceptance criteria:
  - PRD validates under the research-gated linter.
  - Runtime public API exports the capability helpers and types.
  - Runtime tests exercise public behavior rather than private parser helpers.
  - Platform handoff doc names how craftingattention should consume the output.
  - A corresponding craftingattention issue exists for the platform work.

## Out of Scope

- Building the full assignment runner in this first slice.
- Executing CUDA, Triton, ISPC, OpenMP, vLLM, or flash-attn inside the browser.
- Porting all CS336/CS149 assignments end-to-end.
- Building a platform UI in BrowserGrad.
- Creating large benchmark fixtures or downloading course datasets.
- Replacing Pyodide; Pyodide remains one runtime backend.

## Further Notes

This is a meta PRD because it defines the spine that later lab PRDs hang from.
It should be judged by whether it makes the next assignment ports more
systematic: every future lab handoff should name required capabilities, fixture
strategy, rubric strategy, browser-native substitutions, and the craftingattention
issue that consumes the handoff.
