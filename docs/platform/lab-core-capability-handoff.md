# Lab Core Capability Handoff

This handoff is for platform consumers such as `craftingattention`. BrowserGrad
owns the runtime/library primitives; the platform owns UI, authoring workflows,
learner navigation, and issue-level lab rollout.

## Source Of Truth

- Meta PRD: `docs/prd/PRD-018-lab-core-capability-spine.md`
- Runtime API: `@unlocalhosted/browsergrad-runtime`
- Assignment profile parser: `packages/browsergrad-runtime/src/assignment.ts`
- Authoring guide: `docs/platform/assignment-authoring.md`
- Architecture guide: `docs/platform/curriculum-platform-architecture.md`
- Kernel guide: `docs/platform/kernel-lab-foundation.md`

## Platform Contract

For every lab profile, the platform should:

1. Parse the assignment profile with `parseAssignmentProfile`.
2. Convert oracle specs with `profileOracleJsModules`.
3. Build a substrate-neutral run plan with `createAssignmentRunPlan`.
4. Classify the rubric with `assignmentRubricKind`.
5. Build the environment with `createAssignmentCapabilityEnvironment()` from
   detected browser, simulated, and external capability groups.
6. For benchmark dashboards or multi-assignment smoke tests, call
   `createAssignmentBenchmarkPreflightMatrix(profiles, environment, contents?)`
   and render its rows instead of hand-flattening individual reports.
7. Call `assignmentRunReadiness(plan)` before launching the lab, or call
   `createAssignmentPreflightReport(profile, environment)` when the platform
   wants the run plan, readiness, rubric kind, required capabilities, and mount
   plan/cache plan together.
   Use `assignmentRunnerRoute(plan)` or `report.runnerRoute` for the final
   launch branch: `pyodide`, `javascript`, `external`, `unsupported`, or
   `blocked`.
8. Show `runnable`, `simulated`, `external-only`, or `blocked` as preflight
   status, not as runtime crashes.
9. Build a file/dataset mount plan with `createAssignmentMountPlan`.
10. Build dataset cache metadata with `createAssignmentDatasetCachePlan` before
   fetching, caching, or mounting dataset contents.
11. Dry-run platform-provided contents with `evaluateAssignmentMountContents`.
12. Materialize provided file and dataset contents with
   `materializeAssignmentMountPlan`, use `runAssignmentRubric` for the common
   Pyodide mount-and-execute path, or use `runAssignmentJavascriptRubric` for
   browser-native JS rubrics.
   Contents may be strings or `Uint8Array` bytes for compact binary fixtures
   such as `.pt`, `.npz`, or `.safetensors`.
   Use `createAssignmentMountPreflightReport` when rendering one platform
   preflight result for missing content plus hash verification.
   Verify dataset `sha256:<64 hex>` declarations with
   `verifyAssignmentMountContentHashes` before writing to runtime FS.
   Use `Session.fs.readBytes(path)` when the platform needs to verify mounted
   worker bytes against cache, hash, or snapshot metadata.
13. Route runnable labs to the right substrate: Pyodide, TS/JS oracle, WebGPU,
   Worker mesh, external/native runner, or future custom compiler.
14. For simulator-backed systems labs, use
   `@unlocalhosted/browsergrad-simulators` `createDeterministicMesh()` inside
   JS rubrics or platform oracles to produce deterministic rank/collective
   traces before adding real Worker execution. Use `createTaskGraphSimulator()`
   for dependency-constrained task scheduling traces.
   Use `simulateDdpGradientSynchronization()`,
    `simulateFsdpParameterSharding()`, `simulateFsdpGradientReduceScatter()`,
    and `simulateShardedAdamWStep()` for CS336 A2-style DDP/FSDP/sharded
    optimizer fixtures before adding native distributed runners.
    CS149 A1 CPU/SIMD labs can use `simulateCs149ClampedExpVector()`,
    `simulateCs149ArraySumVector()`, and `partitionStaticWork()` for
    browser-safe lane-mask, vector-reduction, and static work-decomposition
    rubrics before C++/ISPC/native timing runners exist.
15. For external/native labs, call `createAssignmentExternalRunnerRequest(plan)`
   and hand that object to platform-owned native, hosted, or CI runners.
   If using `createAssignmentPreflightReport()`, read
   `report.externalRunnerRequest` for external routes.
16. For Pyodide-backed labs, create the rubric execution request with
   `createAssignmentRubricExecRequest`.
   The request uses the shorter runtime watchdog from `test_ms` and `worker_ms`;
   keep `setup_ms` for package preload/cache UI.
17. For JavaScript-backed labs, pass the imported rubric function, declared
    oracle objects, and browser substrates such as WebGPU devices to
    `runAssignmentJavascriptRubric`.
    Prefer `runAssignmentJavascriptProfile()` when the platform has a full
    profile; it owns preflight, route validation, mount collection,
    oracle/substrate wiring, and rubric execution as one e2e path. CS149 A1 is
    the first benchmark profile covered by this route.
    JS/TS streaming checks can import `createStreamingGate` and use
    `gate.wrapInput` plus `gate.wrapOutput`.
    FlashAttention labs can use `@unlocalhosted/browsergrad-kernels`
    `referenceFlashAttention()` and `referenceFlashAttentionBackward()` for
    browser-safe output/LSE/backward checks before Triton/CUDA execution is
    available.
    GPU Puzzles and CS149 A3 CUDA concept labs can use
    `simulateCuda1DGrid()`, `referenceSaxpy()`, and
    `referenceExclusiveScan()` for browser-safe map/guard/SAXPY/scan checks
    before native CUDA runners exist.
    Snapshot-backed labs can use `@unlocalhosted/browsergrad-snapshots`
    `createSnapshotOracle()` to compare small JSON/numeric fixtures and emit
    deterministic mismatch paths.
    Data-cleaning labs can use `@unlocalhosted/browsergrad-data`
    `maskPii()`, `exactLineDeduplicate()`, `minhashDeduplicateDocuments()`,
    `evaluateGopherQuality()`, `gopherQualityFilter()`, and
    `extractVisibleTextFromHtml()` for fixture-scale CS336 A4 checks before
    external classifiers or WARC tooling are available.
    Scaling-law labs can use `@unlocalhosted/browsergrad-scaling`
    `createHostedScalingApiMock()`, `selectExperimentsForDispatch()`, and
    `fitPowerLawScalingLaw()` for CS336 A3-style hosted API, scheduler, and
    scaling-law checks before external FastAPI/Postgres/JAX/Modal runners are
    attached.
    Alignment labs can use `@unlocalhosted/browsergrad-alignment`
    `computePerInstanceDpoLoss()`, `parseMmluResponse()`,
    `parseGsm8kResponse()`, `computeRolloutRewards()`,
    `computeGroupNormalizedRewards()`, `computePolicyGradientLoss()`, and
    `aggregateLossAcrossMicrobatch()` for fixture-scale CS336 A5 DPO/GRPO
    checks before vLLM, flash-attn, or full model training enter the path.
18. In Python rubrics, call profile-registered JS oracles with
    `browsergrad.oracle("<module-name>")`.
19. In Python rubrics, read root, fixture, allowed-test, and behavioral-gate
    context with `browsergrad.assignment_context()`.
20. In Python rubrics, enforce streaming gates with
    `browsergrad.streaming_gate(name, iterable)` plus
    `gate.wrap_output(student_output)` so eager consumers fail before launchers
    need Linux RSS behavior.
21. In Python rubrics, enforce forbidden-read gates with
    `browsergrad.forbidden_read_gate(name, text)` so eager `read()` or
    `readlines()` calls fail while incremental line reads still work.
22. Log one `unlocalhosted/craftingattention` issue for each platform handoff or
    implementation slice.

## Capability Vocabulary

Capability names are strings. Keep them descriptive and reusable:

| Capability | Meaning |
| --- | --- |
| `pyodide` | Python execution in browser Worker. |
| `torch-compat` | BrowserGrad PyTorch-shaped teaching surface is sufficient. |
| `webgpu` | Browser WebGPU adapter is available. |
| `wgsl-kernel` | Lab can run WGSL kernels directly. |
| `flash-attention-oracle` | FlashAttention output, log-sum-exp, and gradients are checked by a deterministic oracle. |
| `cuda-compatible-subset` | Lab targets a BrowserGrad CUDA-like educational subset. |
| `worker-mesh` | Multiple Workers can simulate distributed participants. |
| `distributed-simulator` | Deterministic simulator for DDP/FSDP/task-system behavior. |
| `pthreads-simulator` | Static thread/work decomposition can be checked without native pthreads. |
| `task-graph-simulator` | Dependency-constrained task readiness and scheduling traces are simulated. |
| `simd-simulator` | SIMD lane-mask behavior, utilization, and vector reductions are simulated. |
| `ispc-simulator` | ISPC-style program instance/task concepts are modeled without the native compiler. |
| `ddp-simulator` | DDP gradient averaging and parameter-sync behavior is simulated deterministically. |
| `fsdp-simulator` | FSDP parameter sharding, all-gather, and reduce-scatter behavior is simulated deterministically. |
| `sharded-optimizer-simulator` | Optimizer-state ownership and sharded update equivalence are simulated deterministically. |
| `dataset-fixture` | Small checked-in fixture replaces a large external dataset. |
| `large-file-streaming` | Lab can stream large files instead of loading whole corpora. |
| `snapshot-oracle` | Expected outputs are stored as JSON/NPZ/safetensors snapshots. |
| `tokenizer-oracle` | JS/TS tokenizer oracle is available to rubrics. |
| `near-dedupe-oracle` | Near-duplicate document decisions are checked by a deterministic oracle. |
| `quality-rule-oracle` | Rule-based data quality filters are checked by a deterministic oracle. |
| `response-parser-oracle` | Metric response parsers are checked by deterministic text oracles. |
| `rl-loss-oracle` | Alignment/RL losses have independent reference checks. |
| `hosted-api-mock` | Hosted API behavior is reproduced by a deterministic local mock. |
| `scaling-law-oracle` | Scaling-law projections are checked by a deterministic browser-safe oracle. |
| `native-cpp-external` | Lab requires external native C++ build/run support. |
| `ispc-external` | Lab requires external ISPC support or a simulator. |
| `openmp-external` | Lab requires external OpenMP support or a simulator. |
| `vllm-external` | Lab requires an external vLLM service. |
| `flash-attn-external` | Lab requires external flash-attn/CUDA support. |

This table is a living convention, not a runtime enum. Add names when a new
assignment family needs them, but prefer reusing names across courses.
The first reusable simulator substrate is
`@unlocalhosted/browsergrad-simulators`: it provides deterministic mesh event
traces, simple collectives, and task-graph ready/start/finish traces for labs
that choose simulated `worker-mesh`, `distributed-simulator`, or
`task-graph-simulator` paths. It also provides DDP gradient synchronization,
FSDP sharding/reduce-scatter, and sharded AdamW update simulators for labs that
choose `ddp-simulator`, `fsdp-simulator`, or `sharded-optimizer-simulator`
paths.
The first reusable snapshot substrate is
`@unlocalhosted/browsergrad-snapshots`: it provides JSON/numeric snapshot
comparison for labs that choose `snapshot-oracle` paths before heavier `.npz`,
PyTorch, or model-runtime fixtures are needed.
The first reusable data substrate is `@unlocalhosted/browsergrad-data`: it
provides fixture-scale PII masking, exact line dedupe, near-duplicate document
dedupe, Gopher quality rules, and HTML text extraction for labs that choose
`pii-oracle`, `dedupe-oracle`, `near-dedupe-oracle`, or `quality-rule-oracle`
paths.
The first reusable scaling substrate is `@unlocalhosted/browsergrad-scaling`:
it provides deterministic CS336 A3-style hosted API behavior, scheduler
selection, and log-space power-law fitting for labs that choose
`hosted-api-mock`, `scheduler-simulator`, or `scaling-law-oracle` paths.
The first reusable alignment substrate is
`@unlocalhosted/browsergrad-alignment`: it provides deterministic DPO loss,
response parsing, rollout reward, group-normalized advantage, policy-gradient,
and masked aggregation helpers for labs that choose `rl-loss-oracle` or
`response-parser-oracle` paths.
The first reusable CUDA-concept substrate lives in
`@unlocalhosted/browsergrad-kernels`: it provides `simulateCuda1DGrid()`,
`referenceSaxpy()`, and `referenceExclusiveScan()` for labs that choose
`cuda-compatible-subset`, `wgsl-kernel`, or `performance-rubric` paths.
The first reusable CS149 CPU/SIMD substrate also lives in
`@unlocalhosted/browsergrad-simulators`: it provides clamped-exp lane-mask
simulation, vector array-sum reduction traces, and static contiguous/cyclic work
partitioning for labs that choose `simd-simulator`, `pthreads-simulator`,
`ispc-simulator`, or `performance-rubric` paths.

## Readiness Modes

Profiles declare capability names; platforms declare what those capabilities
mean in the current environment:

- `browser`: direct in-browser execution such as Pyodide, WebGPU, or JS oracle
  execution.
- `simulated`: deterministic Worker/oracle/fixture substitutes that preserve the
  learning objective without native infrastructure.
- `external`: native or hosted runner paths such as CUDA, ISPC, vLLM, Modal, or
  external servers.

Pass these labels through `createAssignmentCapabilityEnvironment()` when calling
`createAssignmentRunPlan`, then use `assignmentRunReadiness(plan)`.
The helper de-duplicates and sorts capabilities, and direct `browser` support
wins over `simulated` or `external` labels for duplicate capability names.
For overall readiness status, selected `external` capabilities produce
`external-only`, selected `simulated` capabilities produce `simulated`, and
failed capability preflight becomes `blocked`.
When several `any_of` groups are available, BrowserGrad selects the strongest
group by mode: direct `browser` path first, then `simulated`, then `external`.
This prevents a teaching simulator from hiding a real browser-native path such
as WGSL.
Render each gate from `plan.capabilityEvaluation.gates`: `status` is the
gate-level route state, `selectedAnyOf` is the chosen alternative group, and
`selectedCapabilities` is the complete selected path including required caps.

## Profile Gate Shape

Use a capability gate when availability can be checked before execution:

```json
{
  "name": "flash_attention_kernel_path",
  "kind": "capability",
  "options": {
    "requires": ["torch-compat"],
    "any_of": [["webgpu", "wgsl-kernel"], ["triton-compatible"], ["native-cuda-external"]],
    "message": "FlashAttention labs need a browser kernel path or an external native path."
  }
}
```

- `requires` means every listed capability must be present.
- `any_of` means at least one group must be fully present.
- `message` is platform-facing explanatory text.
- Non-capability gates such as streaming and timeout should remain separate
  because they are checked by rubrics or watchdogs during execution.
- BrowserGrad validates option shapes for `streaming`, `forbidden-read`, and
  `timeout` gates at profile parse time, so platform authoring should surface
  those errors before fetching fixtures.

## Benchmark Pressure Matrix

Machine-readable profile drafts live in `docs/internal/*.profile.json`. They are
parsed by `packages/browsergrad-runtime/tests/benchmark-profiles.test.ts` so the
handoff matrix cannot drift silently from runtime profile validation.
That test also builds a browser-teaching environment with
`createAssignmentCapabilityEnvironment()`, runs `createAssignmentPreflightReport`
for every benchmark profile, and checks expected readiness states. It dry-runs
empty mount contents for every profile with
`evaluateAssignmentMountContents` so missing rubric files and datasets stay
visible before filesystem writes. Runtime integration tests also mount binary
fixture bytes through the Pyodide path.
Use `createAssignmentBenchmarkPreflightMatrix()` when the platform wants the
same benchmark pressure as one consumable object. Its rows are intentionally
flat: `readinessStatus`, `runnerTarget`, `rubricKind`, capability lists,
`contentOk`, `missingRequiredFiles`, `missingDatasets`, `cacheStrategies`, and
`externalRunnerRequired`. Each row includes a `gates` array with per-gate
`status`, selected alternatives, selected capabilities, missing requirements,
missing alternatives, and optional author messages, so dashboards can show the
exact BrowserGrad-selected route without rebuilding profile reports.
When files and datasets have been fetched, call
`createVerifiedAssignmentBenchmarkPreflightMatrix()` instead. It preserves the
same rows, adds `hashOk` and `hashChecks`, and keeps row `ok` false until
declared dataset SHA-256 hashes match; platforms should run it before any
Pyodide/JS filesystem mount.
The same benchmark test pressure-checks external-runner handoffs for
CS336 Assignment 5 and CS149GPT, proving their profile drafts can produce
`externalRunnerRequest` objects from real capability environments.

| Benchmark | First platform slice | Core capabilities |
| --- | --- | --- |
| CS336 A2 Systems | FlashAttention oracle + DDP/FSDP/sharded optimizer simulator preflight | `torch-compat`, `flash-attention-oracle`, `webgpu`, `worker-mesh`, `distributed-simulator`, `ddp-simulator`, `fsdp-simulator`, `sharded-optimizer-simulator` |
| CS336 A3 Scaling | Hosted API mock + scheduler tests via `@unlocalhosted/browsergrad-scaling` | `http-client`, `hosted-api-mock`, `server-fixture`, `scheduler-simulator`, `scaling-law-oracle` |
| CS336 A4 Data | Small Common Crawl fixtures + `browsergrad-data` PII/dedupe/quality/HTML rubrics | `dataset-fixture`, `large-file-streaming`, `classifier-oracle`, `pii-oracle`, `near-dedupe-oracle`, `quality-rule-oracle` |
| CS336 A5 Alignment | GRPO/DPO math snapshot labs via `@unlocalhosted/browsergrad-alignment` + snapshots | `torch-compat`, `transformers-compatible`, `snapshot-oracle`, `rl-loss-oracle`, `response-parser-oracle` |
| GPU Puzzles | WGSL puzzle runner | `webgpu`, `wgsl-kernel`, `kernel-visualizer` |
| CS149 A1/A2 | Thread/SIMD/task-system simulator with deterministic lane and task traces | `pthreads-simulator`, `simd-simulator`, `task-graph-simulator`, `performance-rubric` |
| CS149 A3 | CUDA scan/SAXPY/render concepts via 1D grid and reference oracles | `webgpu`, `cuda-compatible-subset`, `performance-rubric` |
| CS149GPT | CPU attention optimization oracle | `native-cpp-external`, `attention-oracle`, `simd-simulator` |

## Platform Issue Convention

For each handoff or implementation slice, create a craftingattention issue with:

- BrowserGrad source doc or PRD link.
- Lab benchmark family.
- Required capabilities.
- Platform UI states: runnable, simulated, external-only, blocked.
- Fixture/mount expectations.
- Rubric/oracle expectations.
- Acceptance checks the platform can run.

Use the issue title pattern:

```text
BrowserGrad handoff: <lab or capability slice>
```

## Next Platform Slice

After PRD-018 lands, craftingattention should add a preflight panel that:

1. Reads one or more assignment profiles.
2. Builds a BrowserGrad run plan for a single profile, or builds a batch
   dashboard with
   `createAssignmentBenchmarkPreflightMatrix(profiles, environment, contents?)`
   when showing many benchmark assignments together.
3. Classifies rubric kind with `assignmentRubricKind`.
4. Calls BrowserGrad capability evaluation from the run plan.
5. Calls `assignmentRunReadiness(plan)` and renders its status, selected
   capabilities, and missing capabilities.
6. Or uses `createAssignmentPreflightReport(profile, environment)` to get all
   preflight fields, including `datasetCachePlan`, in one object.
   Read `report.runnerRoute.target` to choose launch controls.
7. Renders `plan.capabilityEvaluation.gates` as preflight rows using each gate's
   `status`, `selectedAnyOf`, `selectedCapabilities`, and missing fields.
8. Builds the BrowserGrad mount plan for runnable or inspectable labs.
9. Builds dataset cache metadata with `createAssignmentDatasetCachePlan`; valid
   hashes become content-addressed cache paths, missing hashes become
   source-addressed URL cache paths, and malformed hashes remain preflight
   failures.
10. Fetches or provides assignment file/dataset contents, then calls
    `evaluateAssignmentMountContents` to show missing files/datasets.
11. For batch dashboards with fetched contents, calls
    `createVerifiedAssignmentBenchmarkPreflightMatrix` so `hashOk` and
    `hashChecks` block stale or wrong datasets before mount.
12. Materializes validated contents into `Session.fs`.
13. Shows packages, oracle modules, rubric kind, file mounts, and
    satisfied/missing capability groups.
14. For external-only labs, calls `createAssignmentExternalRunnerRequest(plan)`
    and queues platform-owned native/hosted execution with the returned files,
    timeouts, selected external capabilities, mount plan, and dataset cache plan.
15. For runnable Pyodide labs, uses `runAssignmentRubric` to mount contents and
    launch the rubric through `Session.exec`, or uses
    `createAssignmentRubricExecRequest` when the platform needs manual staging.
    Binary fixtures can be verified after staging with `Session.fs.readBytes`.
    Dataset hashes should be verified before staging with
    `verifyAssignmentMountContentHashes`.
16. For runnable JavaScript labs, import the rubric module and call
    `runAssignmentJavascriptProfile()` for full profile-driven runs or
    `runAssignmentJavascriptRubric()` when the platform already owns preflight.
    JS rubrics read binary fixtures with `ctx.readBytes(path)`. Kernel labs can use
    `@unlocalhosted/browsergrad-kernels` `createBrowsergradKernelRubric(ctx)` to
    compare WGSL outputs against CPU references and emit BrowserGrad assertions.
    CS336 A2 FlashAttention labs can use `referenceFlashAttention()` and
    `referenceFlashAttentionBackward()` for output, log-sum-exp, and Q/K/V
    gradient fixtures.
    GPU Puzzles and CS149 A3 CUDA concept labs can use
    `simulateCuda1DGrid()`, `referenceSaxpy()`, and
    `referenceExclusiveScan()` for map/guard/SAXPY/scan fixtures and
    out-of-bounds guard diagnostics.
    Simulator-backed labs can use `@unlocalhosted/browsergrad-simulators`
    `createDeterministicMesh()` or `createTaskGraphSimulator()` for event-trace
    rubrics before real Worker execution exists. CS336 A2 systems labs can also
    use the DDP/FSDP/sharded-optimizer simulator helpers for gradient averaging,
    all-gather/reduce-scatter, and AdamW state-sharding checks.
    CS149 A1 CPU/SIMD labs can use the same package's
    `simulateCs149ClampedExpVector()`, `simulateCs149ArraySumVector()`, and
    `partitionStaticWork()` to check clamped exponentiation, vector sums,
    active-lane utilization, tails, and static work partitioning.
    Snapshot-backed labs can use `@unlocalhosted/browsergrad-snapshots`
    `compareSnapshot()` for JSON/numeric fixture checks.
    CS336 A4 data labs can use `@unlocalhosted/browsergrad-data` for
    browser-safe PII, exact/near dedupe, Gopher quality, and HTML extraction
    checks.
    CS336 A3 scaling labs can use `@unlocalhosted/browsergrad-scaling` for
    hosted API mock, scheduler fairness, and scaling-law fixture checks.
    CS336 A5 alignment labs can use
    `@unlocalhosted/browsergrad-alignment` for DPO, parser, reward,
    group-normalized advantage, policy-gradient, and masked aggregation checks.
17. Offers the learner a runnable browser path, simulated path, or external-runner
   note depending on the profile result.
