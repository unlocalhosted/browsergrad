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
Use `createAssignmentMountPlan()` to derive the files and dataset fixtures that
must exist in the runtime filesystem.
Use `materializeAssignmentMountPlan()` to write provided file and dataset
contents into `Session.fs`.
Use `createAssignmentRubricExecRequest()` when the platform is ready to run the
profile's rubric through `Session.exec`. It refuses run plans whose capability
preflight failed.

## Files And Fixtures

Mount assignment files under a dedicated root such as `/assignments/<id>/`.
Use deterministic fixture names and keep generated expected outputs checked in
when they are small enough to review. Large datasets should be declared as
datasets with hashes and mounted by the host before execution.

Dataset mount paths default under `<fixturesPath>/datasets/<filename>`.
Profiles can still point at large external URLs; BrowserGrad records the mount
intent and leaves fetching/caching policy to the platform.
`materializeAssignmentMountPlan()` expects dataset contents keyed by dataset
name, so platforms can fetch/cache however they want before writing to Pyodide.

Rubrics should prefer exact fixtures for correctness and calibrated benchmark
fixtures for performance. Do not depend on host OS paths, subprocesses, POSIX
signals, Linux `/proc`, or process RSS behavior inside Pyodide.

## JS Oracles From Pyodide

The platform may register JS modules into Pyodide with `registerJsModule` via
`createSession({ jsModules })`.
Python rubrics can import those modules through Pyodide's JS bridge and call
small deterministic helpers.

Use this pattern when:

- A browser-safe JS implementation is the source of truth.
- Native Python dependencies are unavailable in Pyodide.
- The oracle result is small enough to serialize between JS and Python.

Keep the bridge narrow. Return plain JSON-compatible values or byte arrays
encoded in an explicit representation.

## Browser-Safe Gates

Resource tests should verify behavior rather than emulate Linux:

- Use worker `timeoutMs` for runaway code.
- Use iterables that fail when consumers call forbidden eager APIs.
- Verify streaming by requiring first output before the whole input is consumed.
- Use browser memory telemetry only for diagnostics, not hard pass/fail rules.

Failure messages should describe the assignment contract, not the browser
implementation detail.

Capability gates should make upstream-native requirements explicit. For
example, a CUDA/Triton test can be declared as a capability gate and replaced by
a WebGPU oracle or skipped with a clear browser-edition reason.
Evaluate these gates before fetching large fixtures or creating rubric exec
requests.

Capability gate options use:

- `requires`: every listed capability must be available.
- `any_of`: at least one listed capability group must be available.
- `message`: optional platform-facing explanation.

## New Assignment Checklist

1. Add a profile with runtime packages, mounts, timeouts, oracles, and gates.
2. Keep reusable helpers in a package and assignment-specific wiring in docs or
   profile code.
3. Port upstream tests only when their assumptions are browser-safe.
4. Replace native OS resource checks with behavior gates.
5. Add unit tests for profile validation and at least one platform integration
   test that mounts files, runs the rubric, and reports clear failures.
