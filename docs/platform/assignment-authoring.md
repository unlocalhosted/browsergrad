# Assignment Authoring

BrowserGrad assignments are platform profiles around reusable runtime
capabilities. Keep assignment facts in profiles, fixtures, and rubrics; keep
package code assignment-agnostic.
Primitive packages should expose domain names, not course names. Put
course-specific wording in the profile/handoff layer. See
`docs/platform/primitive-package-architecture.md` for the naming rule.

## Profile Shape

An assignment profile should name:

- `id`: stable kebab-case assignment identifier.
- `version`: semver version for the profile.
- `requires_browsergrad`: runtime semver range.
- `metadata`: optional title, course, source URL, lecture links, and tags for
  guided-lab discovery.
- `runtime_packages`: Pyodide packages loaded before rubrics run.
- `files`: starter, rubric, reference, and fixture mount paths.
- `timeouts`: wall-clock limits for setup, tests, and long-running student code.
- `oracles`: JS or Python reference helpers available to rubrics.
- `gates`: browser-safe behavioral checks and capability declarations such as
  streaming, timeout, forbidden eager APIs, CUDA/Triton requirements, or
  Worker-mesh requirements.

Profiles should not encode course-specific assumptions in `runtime`, `grad`,
`jit`, or `kernels`. If a behavior is broadly reusable, expose it as a small
library API and let the profile opt into it.

Use `parseAssignmentProfile()` to validate this shape,
`requiredAssignmentCapabilities()` to list the profile's preflight needs,
`evaluateAssignmentCapabilities()` to compare those needs against the current
platform substrate, `profileOracleJsModules()` to convert oracle declarations
into `createSession({ jsModules })` registrations, and
`createAssignmentRunPlan()` to hand the platform a complete launch recipe.
Use `createAssignmentCapabilityCatalog(profiles)` for cross-course planning. It
returns sorted capability entries with the profiles/gates that require each
capability and the alternative groups where each capability appears.
Use `createAssignmentPreflightReport(profile, environment)` when the platform
needs one readonly object containing the run plan, rubric kind, readiness,
required capabilities, mount plan, and dataset cache plan.
Use `createAssignmentPlatformHandoff(profile, report, contents?)` when UI needs
one launch-panel object. Its `nextAction` is one of `install-capabilities`,
`mount-content`, `run-pyodide`, `run-javascript`, `request-external-runner`, or
`unsupported`, and its `messages`/missing fields are safe to render directly.
Use `createAssignmentBenchmarkPreflightMatrix(profiles, environment, contents?)`
when a platform needs one dashboard/checklist row per benchmark assignment. It
returns flattened readiness, route, capability, content-gap, cache-strategy, and
external-runner fields without asking the platform to duplicate BrowserGrad's
preflight logic. Each row also includes capability `gates` with selected
alternatives and missing alternatives so UI can explain why a lab is runnable,
simulated, external-only, or blocked per gate.
Use `createVerifiedAssignmentBenchmarkPreflightMatrix(profiles, environment,
contents)` once dataset contents are available. It adds `hashOk` and
`hashChecks`, so platforms can reject wrong or stale fixtures before writing
anything into Pyodide or a JS rubric context.
Use `createAssignmentCapabilityEnvironment()` to build that environment from
`browserCapabilities`, `simulatedCapabilities`, and `externalCapabilities`
instead of handcrafting `capabilityModes`. It de-duplicates names, sorts them
deterministically, and labels each selected capability as `browser`,
`simulated`, or `external`. Then call `assignmentRunReadiness(plan)` to get the
learner-facing preflight state: `runnable`, `simulated`, `external-only`, or
`blocked`.
Use `assignmentRubricKind()` to route Python, JavaScript, and unknown rubric
paths to the right execution substrate.
Use `assignmentRunnerRoute(plan)` when the platform wants one branch key:
`pyodide`, `javascript`, `external`, `unsupported`, or `blocked`.
When that branch is `external`, use `createAssignmentExternalRunnerRequest(plan)`
to hand native/hosted infrastructure a stable object containing selected
external capabilities, timeouts, mount/cache metadata, behavioral gates, and a
ready-to-inject `environment` map of `BROWSERGRAD_*` variables.
`createAssignmentPreflightReport()` also includes `externalRunnerRequest` when
the active route is external.
Use `createAssignmentMountPlan()` to derive the files and dataset fixtures that
must exist in the runtime filesystem.
Use `createAssignmentDatasetCachePlan(mountPlan)` before fetch/cache work so
datasets get deterministic platform cache keys and explicit hash-status
strategies.
Use `evaluateAssignmentMountContents(mountPlan, contents)` before writing files
to report missing rubric files, missing datasets, optional skips, and writable
paths without touching `Session.fs`.
Use `materializeAssignmentMountPlan()` to write provided file and dataset
contents into `Session.fs`.
Use `createAssignmentRubricExecRequest()` when the platform is ready to run the
profile's rubric through `Session.exec`. It refuses run plans whose capability
preflight failed.
Use `runAssignmentRubric()` for the common Pyodide path when the platform wants
BrowserGrad to derive mounts, materialize contents, and execute the rubric as
one operation.
Use `runAssignmentJavascriptRubric()` for browser-native JS rubrics that need
assignment context, mounted text, oracles, and structured assertion/artifact
helpers without Pyodide. Pass browser resources such as WebGPU adapters/devices
through `substrates` and read them with `ctx.substrate(name)`.
Use `runAssignmentJavascriptProfile()` when the platform has a full profile and
wants one profile-driven call that builds preflight, validates the JavaScript
route, mounts declared contents, preflights declared JS oracles, wires
oracles/substrates, and runs the rubric. CS149 A1 and GPU Puzzles have e2e
tests proving this route with simulator/kernel oracles.
Kernel-style JS rubrics can use `createKernelRubric()` from
`@unlocalhosted/browsergrad-kernels` to compare tensors against CPU references
and forward pass/fail assertions into the BrowserGrad JS rubric context. Use
`kernelRubricFailureToAssertionDetails()` when forwarding failures to
`ctx.assertFail()` so previews and mismatch details survive as expected/actual
strings.
Prefer `createBrowsergradKernelRubric(ctx)` inside `runAssignmentJavascriptRubric`
callbacks; it wires those callbacks for you without making kernel packages
depend on the runtime package.
CS336 A2-style FlashAttention rubrics can use
`referenceFlashAttention()` and `referenceFlashAttentionBackward()` from
`@unlocalhosted/browsergrad-kernels` for `flash-attention-oracle` fixture
checks that include output, saved log-sum-exp, and Q/K/V gradients before native
Triton/CUDA paths are available. Register a profile-local module such as
`_bg_attention_math` when Python rubrics need JSON-string bridge methods.
GPU Puzzles and CS149 A3-style CUDA concept rubrics can use the same package's
`simulateCuda1DGrid()`, `referenceSaxpy()`, `referenceExclusiveScan()`, and
`referenceFindRepeats()` for `cuda-compatible-subset` fixture checks. Use
`referenceOrderedCircleRender()` for renderer-ordering fixtures where circle
input order determines alpha blending. Use `defineCuda1DProgram()`,
`simulateCuda1DProgram()`, `emitCuda1DProgramWgsl()`, and
`runCuda1DProgramWebGpu()` for author-once CUDA-shaped 1D kernels that need
scalar params, input reads, output reads, WGSL source generation, and browser
WebGPU dispatch. The grid simulator records thread/block ids, global
reads/writes, and out-of-bounds accesses so missing guards fail as teaching
feedback before native CUDA runners exist.
Distributed or systems-style JS rubrics can use `simulation.createDeterministicMesh()`
from `@unlocalhosted/browsergrad-primitives` to model worker ranks, barriers,
broadcasts, point-to-point messages, and `allReduce` results as deterministic
event traces. Use `simulation.createTaskGraphSimulator()` from the same package for
dependency-constrained task-system traces with deterministic ready/start/finish
events and worker assignment. Register profile glue such as `_bg_task_graph`
for task-graph rubrics. These APIs back `worker-mesh`,
`distributed-simulator`, `pthreads-simulator`, or `task-graph-simulator`
profile paths when the teaching goal is ordering and participation rather than
native throughput.
CS336 A2-style distributed-training rubrics can use the same package's
`simulateDdpGradientSynchronization()`, `simulateFsdpParameterSharding()`,
`simulateFsdpGradientReduceScatter()`, and `simulateShardedAdamWStep()` for
`ddp-simulator`, `fsdp-simulator`, and `sharded-optimizer-simulator` profile
paths before using native `torch.distributed`, multiprocessing, or CUDA.
Register profile glue such as `_bg_distributed_training` for Python rubrics.
CS149 A1-style JavaScript rubrics can use
`simulateVectorizedClampedExp()`, `simulateVectorizedArraySum()`, and
`partitionStaticWork()` for `simd-simulator`, `pthreads-simulator`, and
`performance-rubric` fixture checks. These helpers verify outputs plus lane
utilization, vector-instruction traces, tail masks, horizontal reductions, and
static contiguous/cyclic work decomposition without depending on native C++,
AVX2, ISPC, or host timing.
Fixture-backed JS rubrics can use `compareSnapshot()` or
`createSnapshotComparator()` from `@unlocalhosted/browsergrad-primitives` for
`snapshot-oracle` profile paths. Prefer it for small JSON/numeric outputs such
as losses, logits, masks, event traces, dedupe decisions, or alignment math
fixtures; it reports deterministic mismatch paths and numeric tolerances
without native `.npz` or PyTorch dependencies.
Data-cleaning rubrics can use `@unlocalhosted/browsergrad-primitives` for
`pii-oracle`, `dedupe-oracle`, `near-dedupe-oracle`, and `quality-rule-oracle`
fixture checks. Register a profile-local module such as `_bg_data_cleaning`
when Python needs snake_case or JSON-string bridge methods; keep that wrapper
outside the primitive package surface. The primitive `data` helpers cover `maskPii()`,
`exactLineDeduplicate()`, `minhashDeduplicateDocuments()`,
`evaluateGopherQuality()`, `gopherQualityFilter()`, and
`extractVisibleTextFromHtml()` so CS336 A4-style
tests can validate browser-safe behavior before WARC readers, fastText,
transformers, or full Common Crawl data enter the loop.
Scaling-law and hosted API rubrics can use
`@unlocalhosted/browsergrad-primitives` for `hosted-api-mock`,
`scheduler-simulator`, and `scaling-law-oracle` fixture checks. Its helpers
cover hosted training API fixture behavior, duplicate training-config
rejection, dispatch fairness, and log-space power-law fits before a real
FastAPI/Postgres/JAX/Modal stack enters the loop.
Alignment rubrics can use `@unlocalhosted/browsergrad-primitives` for
`rl-loss-oracle` and `response-parser-oracle` fixture checks. Its helpers cover
CS336 A5-style DPO loss, MMLU/GSM8K parsing, rollout reward metadata,
group-normalized rewards, policy-gradient token losses, and masked microbatch
aggregation before full vLLM, flash-attn, Qwen inference, or native training
loops enter the loop. Register a profile-local module such as `_bg_rl_math`
when Python rubrics need JSON-string bridge methods.

## Files And Fixtures

Mount assignment files under a dedicated root such as `/assignments/<id>/`.
Use deterministic fixture names and keep generated expected outputs checked in
when they are small enough to review. Large datasets should be declared as
datasets with hashes and mounted by the host before execution.
Fixture contents can be strings or `Uint8Array` bytes. Use bytes for compact
binary fixtures and snapshots such as `.pt`, `.npz`, and `.safetensors`.
After materializing a binary fixture, platforms can call
`Session.fs.readBytes(path)` to verify the worker-visible bytes match cache or
snapshot expectations.

Dataset mount paths default under `<fixturesPath>/datasets/<filename>`.
Profiles can still point at large external URLs; BrowserGrad records the mount
intent and leaves fetching/caching policy to the platform.
`createAssignmentDatasetCachePlan()` adds cache metadata without fetching: valid
SHA-256 declarations are `content-addressed`, missing hashes are
`source-addressed`, and malformed/unsupported hashes are explicit preflight
statuses.
Use `evaluateAssignmentMountContents()` after fetching/cache lookup and before
`materializeAssignmentMountPlan()` so missing fixture inputs surface as
preflight status, not partial filesystem writes.
When dataset declarations include `sha256:<64 hex>`, call
`verifyAssignmentMountContentHashes()` before materializing. Treat `mismatch`,
`invalid`, and `unsupported` as platform preflight failures, not rubric failures.
Use `createAssignmentMountPreflightReport()` when the UI wants content readiness
and hash verification in one object.
`materializeAssignmentMountPlan()` expects dataset contents keyed by dataset
name, so platforms can fetch/cache however they want before writing to Pyodide.
JavaScript rubrics should use `ctx.readBytes(path)` for binary mounts and
`ctx.readText(path)` for UTF-8 text.
JS/TS rubrics that need streaming checks can import `createStreamingGate()` from
`@unlocalhosted/browsergrad-primitives`, then use `gate.wrapInput(chunks)` and
`gate.wrapOutput(studentOutput)` to mirror the Python streaming-gate contract.
JS/TS rubrics that need hosted API fixtures can import
`createHostedTrainingApiFixture()` from `@unlocalhosted/browsergrad-primitives`,
then
exercise student HTTP-client logic against deterministic `/budget`, `/submit`,
`/experiments`, `/experiment/{id}`, and `/final_submission` behavior. Keep live
hosted servers and JAX training behind explicit external capability gates.

Rubrics should prefer exact fixtures for correctness and calibrated benchmark
fixtures for performance. Do not depend on host OS paths, subprocesses, POSIX
signals, Linux `/proc`, or process RSS behavior inside Pyodide.

## JS Oracles From Pyodide

The platform may register JS modules into Pyodide with `registerJsModule` via
`createSession({ jsModules })`.
Python rubrics should access those modules through `browsergrad.oracle(name)`
and call small deterministic helpers:

```py
import browsergrad as bg

tokenizers = bg.oracle("_bg_tokenizers")
model = tokenizers.train_byte_bpe(corpus, vocab_size, special_tokens)
```

Use this pattern when:

- A browser-safe JS implementation is the source of truth.
- Native Python dependencies are unavailable in Pyodide.
- The oracle result is small enough to serialize between JS and Python.

Keep the bridge narrow. Return plain JSON-compatible values or byte arrays
encoded in an explicit representation. If `bg.oracle(name)` cannot find a
registered module, the rubric should fail as a platform wiring error, not a
student-code failure.

## Browser-Safe Gates

Resource tests should verify behavior rather than emulate Linux:

- Use worker `timeoutMs` for runaway code.
- For Python rubrics, BrowserGrad chooses the shorter declared runtime watchdog
  from `test_ms` and `worker_ms`; `setup_ms` remains a platform preload/cache
  concern.
- Use iterables that fail when consumers call forbidden eager APIs.
- Verify streaming by requiring first output before the whole input is consumed.
- Use browser memory telemetry only for diagnostics, not hard pass/fail rules.

Failure messages should describe the assignment contract, not the browser
implementation detail.

Browser-safe gate options are validated at profile parse time:

- `streaming`: required `max_chunks_before_first_yield`, optional `chunk_count`.
- `forbidden-read`: required string-array `methods`, usually `read` and
  `readlines`.
- `timeout`: required `timeout_ms`.

`createAssignmentRubricExecRequest()` exposes non-capability gates to Python
rubrics through `BROWSERGRAD_BEHAVIORAL_GATES_JSON`, alongside
`BROWSERGRAD_ASSIGNMENT_ROOT`, optional `BROWSERGRAD_FIXTURES_PATH`, and
`BROWSERGRAD_ALLOWED_TESTS_JSON`. Rubrics should use
`browsergrad.assignment_context()` to parse those values and enforce only the
tests/gates declared by the active profile.
For Python iterable streaming checks, wrap rubric input with
`browsergrad.streaming_gate(name, iterable)` and wrap the returned student output
with `gate.wrap_output(output)`. The helper reads
`max_chunks_before_first_yield` from the matching streaming gate and raises
`StreamingGateViolation` on eager consumption.
For Python file-style checks, wrap fixture text with
`browsergrad.forbidden_read_gate(name, text)`. The helper reads forbidden
`methods` from the matching `forbidden-read` gate, allows incremental iteration
and `readline()`, and raises `ForbiddenReadViolation` for eager `read()` or
`readlines()` calls.

Capability gates should make upstream-native requirements explicit. For
example, a CUDA/Triton test can be declared as a capability gate and replaced by
a WebGPU oracle or skipped with a clear browser-edition reason.
Evaluate these gates before fetching large fixtures or creating rubric exec
requests.

Capability gate options use:

- `requires`: every listed capability must be available.
- `any_of`: at least one listed capability group must be available.
- `message`: optional platform-facing explanation.

Capability modes are supplied by the platform environment, not the profile.
This lets the same profile run as a direct browser lab on one platform, a
simulator-backed lab on another, or an external-runner lab when native tooling
is attached. Prefer `createAssignmentCapabilityEnvironment()` so duplicate
capabilities are normalized consistently; direct browser support wins over
simulated and external mode labels for the same capability. If multiple
`any_of` capability groups are available, BrowserGrad chooses the strongest
group by mode: `browser` before `simulated` before `external`.
Each capability gate evaluation includes gate-level `status`, `selectedAnyOf`,
and `selectedCapabilities`; use those fields for preflight rows instead of
duplicating route-choice logic in the platform.
Use `assignmentRunnerRoute(plan)` or `report.runnerRoute` for the final launch
branch. A simulated-but-runnable Python lab still routes to `pyodide`; an
external-only plan routes to `external`; unknown rubric extensions route to
`unsupported`.
External plans should not call Pyodide or JS rubric runners. Generate an
external runner request and pass it to platform-owned native, hosted, or CI
execution infrastructure. Inject `request.environment` into that runner so
Python, shell, or hosted adapters see the same assignment id, root, fixtures,
allowed tests, selected capabilities, and behavioral gates as BrowserGrad
rubrics.

## New Assignment Checklist

1. Add a profile with runtime packages, mounts, timeouts, oracles, and gates.
2. Keep reusable helpers in a package and assignment-specific wiring in docs or
   profile code.
3. Classify the rubric kind and route non-Python rubrics to JS/WebGPU/native
   substrates instead of Pyodide with `assignmentRunnerRoute()`.
4. Port upstream tests only when their assumptions are browser-safe.
5. Replace native OS resource checks with behavior gates.
6. Add unit tests for profile validation and at least one platform integration
   test using `runAssignmentRubric()` or `runAssignmentJavascriptRubric()` that
   mounts files, runs the rubric, calls declared oracles, and reports clear
   failures.
7. If using `runAssignmentJavascriptProfile()`, provide every profile-declared
   oracle in `options.oracles`; missing names should fail before the rubric runs.
