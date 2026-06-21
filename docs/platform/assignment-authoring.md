# Assignment Authoring

BrowserGrad assignments are platform profiles around reusable runtime
capabilities. Keep assignment facts in profiles, fixtures, and rubrics; keep
package code assignment-agnostic.

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
Use `createAssignmentPreflightReport(profile, environment)` when the platform
needs one readonly object containing the run plan, rubric kind, readiness,
required capabilities, and mount plan.
Pass `capabilityModes` with the platform environment when a capability should
be labeled as `browser`, `simulated`, or `external`, then call
`assignmentRunReadiness(plan)` to get the learner-facing preflight state:
`runnable`, `simulated`, `external-only`, or `blocked`.
Use `assignmentRubricKind()` to route Python, JavaScript, and unknown rubric
paths to the right execution substrate.
Use `createAssignmentMountPlan()` to derive the files and dataset fixtures that
must exist in the runtime filesystem.
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
model = tokenizers.train_cs336_bpe(corpus, vocab_size, special_tokens)
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

`createAssignmentRubricExecRequest()` exposes non-capability gates to Python
rubrics through `BROWSERGRAD_BEHAVIORAL_GATES_JSON`, alongside
`BROWSERGRAD_ASSIGNMENT_ROOT`, optional `BROWSERGRAD_FIXTURES_PATH`, and
`BROWSERGRAD_ALLOWED_TESTS_JSON`. Rubrics should use
`browsergrad.assignment_context()` to parse those values and enforce only the
tests/gates declared by the active profile.

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
is attached. If multiple `any_of` capability groups are available, BrowserGrad
chooses the strongest group by mode: `browser` before `simulated` before
`external`.
Each capability gate evaluation includes gate-level `status`, `selectedAnyOf`,
and `selectedCapabilities`; use those fields for preflight rows instead of
duplicating route-choice logic in the platform.

## New Assignment Checklist

1. Add a profile with runtime packages, mounts, timeouts, oracles, and gates.
2. Keep reusable helpers in a package and assignment-specific wiring in docs or
   profile code.
3. Classify the rubric kind and route non-Python rubrics to JS/WebGPU/native
   substrates instead of Pyodide.
4. Port upstream tests only when their assumptions are browser-safe.
5. Replace native OS resource checks with behavior gates.
6. Add unit tests for profile validation and at least one platform integration
   test using `runAssignmentRubric()` or `runAssignmentJavascriptRubric()` that
   mounts files, runs the rubric, calls declared oracles, and reports clear
   failures.
