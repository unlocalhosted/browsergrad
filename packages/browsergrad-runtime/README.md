# @unlocalhosted/browsergrad-runtime

[![npm](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-runtime.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-runtime)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pyodide-in-Worker runtime for browser-based Python execution. Same-origin assets, AbortSignal cancellation, cooperative cancel via `SharedArrayBuffer`, structured assertion + artifact protocols. Platform-agnostic primitives — bring your own UI.

> **Status: v0.1.1.** Stable API in [`src/types.ts`](./src/types.ts) — adding optional fields is non-breaking; everything else triggers a major bump. Adds the lab manifest validator + semver gate (see below).

## Install

```sh
npm install @unlocalhosted/browsergrad-runtime pyodide
```

Pyodide is a `peerDependency` — you install it directly so you control the version and asset-sync story.

## Pyodide assets (same-origin)

This library **does not load Pyodide from a CDN.** You point it at a same-origin URL that serves the runtime assets. The simplest pattern is a sync script in your build:

```js
// scripts/sync-pyodide.mjs
import { copyFile, mkdir, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = join(__dirname, "..", "node_modules", "pyodide");
const pkg = JSON.parse(await readFile(join(src, "package.json"), "utf8"));
const dest = join(__dirname, "..", "public", "pyodide", `v${pkg.version}`);

await mkdir(dest, { recursive: true });
for (const f of ["pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm",
                 "python_stdlib.zip", "pyodide-lock.json"]) {
  await copyFile(join(src, f), join(dest, f));
}
```

Run at install time:

```json
{ "scripts": { "postinstall": "node scripts/sync-pyodide.mjs" } }
```

## Usage

```ts
import { createSession } from "@unlocalhosted/browsergrad-runtime";

const session = await createSession({
  pyodideIndexURL: "/pyodide/v0.26.4/",
  packages: ["numpy"],
  onPackageProgress: (e) => console.log(e.package, e.status),
});

const result = await session.exec({
  code: `
    import numpy as np
    import browsergrad as bg

    x = np.arange(10)
    actual = int(x.sum())
    expected = 45

    if actual == expected:
        bg.assert_pass("sum_of_0_to_9")
    else:
        bg.assert_fail("sum_of_0_to_9", "wrong sum",
                       expected=expected, actual=actual)

    bg.emit_json("series", {"x": x.tolist()})
    print("done")
  `,
  timeoutMs: 5000,
  signal: someAbortController.signal,
  onStdout: (chunk) => console.log("[py]", chunk),
  onAssertion: (a) => console.log("[assert]", a),
  onArtifact: (a) => console.log("[artifact]", a),
});

console.log(result.ok, result.durationMs);
console.log(result.assertions);          // structured Assertion[] in arrival order
console.log(result.artifacts);           // structured Artifact[] in arrival order

await session.dispose();
```

A full runnable example is in [`examples/vanilla/`](./examples/vanilla/README.md).

## Cancellation

Three ways to cancel an `exec`, from most cooperative to most blunt:

1. **`session.interrupt()`** — explicit call. Writes SIGINT to the SharedArrayBuffer-backed interrupt buffer. Python raises `KeyboardInterrupt` on the next bytecode boundary. Requires `crossOriginIsolated` page.
2. **`ExecOptions.signal: AbortSignal`** — standard abort. First tries the interrupt buffer, falls back to `worker.terminate()` after a 500 ms grace period.
3. **`ExecOptions.timeoutMs: number`** — same flow as `signal`, just wall-clock-driven.

`session.canInterrupt` exposes whether cooperative cancel is available in the current document. Browsers without cross-origin isolation (no `COOP: same-origin` + `COEP: require-corp`) get terminate-only mode automatically — no error thrown.

## The `browsergrad` Python module

Registered into the Pyodide runtime at session boot, so user code can `import browsergrad as bg` with zero setup. The helpers emit structured events that surface in `ExecResult.assertions` / `ExecResult.artifacts` and via the `onAssertion` / `onArtifact` callbacks.

```python
import browsergrad as bg

bg.assert_pass(name, duration_ms=None)
bg.assert_fail(name, message, expected=None, actual=None, duration_ms=None)
bg.assert_error(name, message, exc=None, duration_ms=None)

bg.log(name, data, level="info")             # → ArtifactLog
bg.emit_json(name, data)                      # → ArtifactJson (data must be JSON-able)
bg.emit_image(name, mime, data_base64)        # → ArtifactImage

oracle = bg.oracle("_bg_tokenizers")          # profile-registered JS oracle module
model = oracle.train_byte_bpe(corpus, 300)

ctx = bg.assignment_context()                 # id/root/fixtures/tests/gates
for gate in ctx["behavioral_gates"]:
    ...
```

The library does not interpret assertion/artifact events — it just relays them.
`bg.oracle(name)` imports a JS module registered through
`createSession({ jsModules })` and raises a clear error if the profile/runtime
forgot to register it. `bg.assignment_context()` parses the launcher-provided
assignment environment into a plain dict. Build your test framework /
visualizer / grader on top.

## Lab manifest

Optional helper for platforms that ship versioned "labs" (or any other unit of executable Python content) and want a contract surface for pinning runtime versions. Hand-written validator + semver gate — no `ajv` dependency.

```ts
import {
  parseManifest,
  assertCompatibleRuntime,
  LabRuntimeMismatch,
} from "@unlocalhosted/browsergrad-runtime";

const manifestJson = JSON.parse(await fetch("/labs/single-neuron/manifest.json").then((r) => r.text()));

const result = parseManifest(manifestJson);
if (!result.ok) {
  throw new Error(`Invalid manifest: ${result.errors.join("; ")}`);
}

try {
  assertCompatibleRuntime(result.manifest, "0.8.0");
} catch (e) {
  if (e instanceof LabRuntimeMismatch) {
    // Show "lab requires <pin>, runtime is <version>" with no fallback
  }
}
```

The schema is intentionally small (8 fields):

| Field | Purpose |
|---|---|
| `id`, `version` | Lab identity (kebab-case + semver) |
| `requires_browsergrad` | Semver range — `^0.8.0`, `~0.8.1`, or exact |
| `required_ops` | UOps the lab uses (≤ 64; informational for the dispatcher) |
| `rubric_path`, `starter_path`, `reference_path` | Where the lab's Python files live |
| `datasets` | Optional safetensors URLs (≤ 32) |

Hard-fail on mismatch is the v0 contract — no legacy CDN, no GH-Action mirror. When 5+ labs exist and an actual coexistence problem appears, we'll revisit.

## Assignment profile preflight

Platforms can validate assignment profiles and decide whether a lab is runnable
before launching a Worker:

```ts
import {
  assignmentRubricKind,
  assignmentRunReadiness,
  assignmentRunnerRoute,
  createAssignmentCapabilityEnvironment,
  createAssignmentBenchmarkPreflightMatrix,
  createVerifiedAssignmentBenchmarkPreflightMatrix,
  createAssignmentDatasetCachePlan,
  createAssignmentExternalRunnerRequest,
  createAssignmentMountPlan,
  createAssignmentPreflightReport,
  createAssignmentRunPlan,
  evaluateAssignmentCapabilities,
  evaluateAssignmentMountContents,
  parseAssignmentProfile,
  requiredAssignmentCapabilities,
  runAssignmentJavascriptProfile,
  runAssignmentJavascriptRubric,
  runAssignmentRubric,
} from "@unlocalhosted/browsergrad-runtime";

const parsed = parseAssignmentProfile(profileJson);
if (!parsed.ok) throw new Error(parsed.errors.join("; "));

const environment = createAssignmentCapabilityEnvironment({
  browserCapabilities: ["pyodide", "torch-compat", "webgpu", "wgsl-kernel"],
  simulatedCapabilities: ["worker-mesh", "distributed-simulator"],
  externalCapabilities: ["native-cuda-external"],
});
const required = requiredAssignmentCapabilities(parsed.profile);
const preflight = evaluateAssignmentCapabilities(parsed.profile, environment);
const plan = createAssignmentRunPlan(parsed.profile, environment);
const report = createAssignmentPreflightReport(parsed.profile, environment);
const matrix = createAssignmentBenchmarkPreflightMatrix(
  [parsed.profile],
  environment,
  { files: { [report.plan.files.rubricPath]: rubricSource } },
);
const verifiedMatrix = await createVerifiedAssignmentBenchmarkPreflightMatrix(
  [parsed.profile],
  environment,
  {
    files: { [report.plan.files.rubricPath]: rubricSource },
    datasets: { tiny: tinyFixtureText },
  },
);
const route = assignmentRunnerRoute(plan);
if (!plan.ok) {
  throw new Error(`Missing capabilities: ${plan.capabilityEvaluation.missingCapabilities.join(", ")}`);
}
if (route.target === "external") {
  const externalRequest = createAssignmentExternalRunnerRequest(plan);
  // Send externalRequest to your native/hosted runner queue.
}

const mounts = createAssignmentMountPlan(plan);
const datasetCache = createAssignmentDatasetCachePlan(mounts);
const rubricKind = assignmentRubricKind(plan);
const readiness = assignmentRunReadiness(plan);

console.log(required, preflight.ok, preflight.missingCapabilities);
console.log(plan.session.packages, plan.files.rubricPath, plan.execution.allowedTests);
console.log(readiness.status, rubricKind, mounts.files, mounts.datasets);
console.log(route.target, report.runnerRoute.target, report.mountPlan.files, datasetCache.datasets);
console.log(matrix.rows[0]?.readinessStatus, matrix.rows[0]?.contentOk);
console.log(matrix.rows[0]?.gates.map((gate) => [gate.name, gate.status, gate.selectedCapabilities]));
console.log(verifiedMatrix.rows[0]?.hashOk, verifiedMatrix.rows[0]?.hashChecks);
const fileContents: Record<string, string | Uint8Array> = {
  [plan.files.rubricPath]: rubricSource,
};
if (plan.files.starterPath) fileContents[plan.files.starterPath] = starterSource;
const contentReadiness = evaluateAssignmentMountContents(mounts, {
  files: fileContents,
  datasets: { tiny: tinyFixtureText },
});
if (!contentReadiness.ok) {
  throw new Error(`Missing mount contents: ${[
    ...contentReadiness.missingRequiredFiles,
    ...contentReadiness.missingDatasets,
  ].join(", ")}`);
}

const run = await runAssignmentRubric(session, plan, {
  files: fileContents,
  datasets: { tiny: tinyFixtureText },
});
console.log(run.mount.writtenPaths, run.exec.assertions);

const jsRun = await runAssignmentJavascriptProfile(
  parsed.profile,
  environment,
  { files: { [report.plan.files.rubricPath]: rubricSource } },
  importedRubric,
  { oracles: { _bg_cpu_parallelism: cs149Oracle } },
);
console.log(jsRun.report.runnerRoute.target, jsRun.run.assertions);
```

Capability gates support `requires` for all-of requirements and `any_of` for
alternative groups. Non-capability gates such as streaming and timeout remain
rubric/watchdog checks.

Use `createAssignmentCapabilityEnvironment()` to build the platform environment
from browser, simulated, and external capability groups. It de-duplicates and
sorts capabilities, then attaches `capabilityModes`; if a capability appears in
multiple groups, direct browser support wins over simulated and external modes.
`assignmentRunReadiness(plan)` turns the selected capabilities into a platform
status: `runnable`, `simulated`, `external-only`, or `blocked`. When multiple
`any_of` alternatives are available, BrowserGrad selects the strongest path by
mode: `browser`, then `simulated`, then `external`.

`assignmentRunnerRoute(plan)` maps preflight output to a runner target:
`pyodide`, `javascript`, `external`, `unsupported`, or `blocked`. Use this for
platform branching before calling `runAssignmentRubric()` or
`runAssignmentJavascriptRubric()`.
Use `runAssignmentJavascriptProfile()` when the platform has a full JS-routed
profile and wants BrowserGrad to own preflight, route validation, mount
collection, declared-oracle preflight, oracle/substrate wiring, and rubric
execution in one call. The runner rejects missing profile-declared JS oracles
before invoking the rubric, so broken platform wiring cannot silently pass.
For `external` routes, `createAssignmentExternalRunnerRequest(plan)` returns the
native/hosted runner handoff: selected external capabilities, resolved files,
timeouts, behavioral gates, a `BROWSERGRAD_*` environment map, mount plan, and
dataset cache plan. Preflight reports include this as `externalRunnerRequest`
when applicable. BrowserGrad does not execute native code; the platform owns
that runner.

`createAssignmentPreflightReport(profile, environment)` bundles the common
platform preflight calls into `{ plan, rubricKind, readiness, runnerRoute,
requiredCapabilities, mountPlan, datasetCachePlan }`. Use it when the UI needs
a single readonly object before fetching fixtures, mounting files, or launching
code.
`createAssignmentBenchmarkPreflightMatrix(profiles, environment, contents?)`
batch-flattens those same preflight decisions into platform-ready rows:
readiness status, runner target, rubric kind, required/selected/missing
capabilities, mount-content gaps, dataset cache strategies, and whether an
external runner handoff is required. Each row also contains `gates`, a
capability-gate table with `status`, `selectedAnyOf`, `selectedCapabilities`,
`missingRequired`, `missingAnyOf`, and optional author `message`. Use it for
benchmark dashboards, PRD handoffs, and platform smoke tests that cover many
course profiles at once without reimplementing BrowserGrad's route selection.
Use `createVerifiedAssignmentBenchmarkPreflightMatrix(profiles, environment,
contents)` after file/dataset contents are fetched to run the same matrix plus
dataset SHA checks. Its rows add `hashOk` and `hashChecks`, and row `ok` stays
false until capability, content, and declared dataset hashes all pass.

`createAssignmentRunPlan()` does not execute student code. It produces the
platform handoff object: package preload list, JS oracle modules, resolved
profile file paths, allowed test ids, timeout hints, dataset declarations,
capability preflight, and behavioral gates.

`createAssignmentMountPlan()` turns a run plan into deterministic file and
dataset mount declarations. It does not fetch or write content; the platform
uses it to decide what to place into `Session.fs` before rubric execution.
`createAssignmentDatasetCachePlan(mountPlan)` turns dataset mount declarations
into deterministic cache metadata for platform fetch/OPFS layers. Valid
`sha256:<64 hex>` hashes become content-addressed cache entries; missing hashes
use source-addressed URL keys; malformed/unsupported hashes are marked
`invalid-hash` or `unsupported-hash` so preflight can block before trust.
`evaluateAssignmentMountContents(mountPlan, contents)` dry-runs that mount plan
against platform-provided file/dataset contents and returns missing required
files, missing datasets, optional skips, and writable paths without touching
`Session.fs`.
`verifyAssignmentMountContentHashes(mountPlan, contents)` hashes dataset contents
that declare `sha256:<64 hex>` and returns per-dataset `match`, `mismatch`,
`missing`, `invalid`, or `unsupported` checks before write.
`createAssignmentMountPreflightReport(mountPlan, contents)` combines content
readiness and hash checks into one `{ ok, content, hashes }` object for platform
preflight panels.
Mount contents may be UTF-8 strings or `Uint8Array` bytes. Use bytes for
snapshots and small upstream fixtures such as `.pt`, `.npz`, or
`.safetensors`. Hosts can verify mounted or cached bytes with
`Session.fs.readBytes(path)` without decoding through UTF-8.

`materializeAssignmentMountPlan()` writes provided string contents into
`Session.fs` in mount-plan order, preserving byte contents for binary fixtures.
Required rubric content fails loudly when missing; optional starter/reference
files are skipped when absent. Dataset contents are keyed by dataset name.

`createAssignmentRubricExecRequest()` turns that plan into the minimal
`Session.exec` request for rubric execution. It sets assignment metadata in
environment variables, puts the assignment root on `sys.path`, runs the resolved
rubric path, and uses the profile's effective execution timeout. It refuses
plans whose capability preflight failed so unavailable labs do not accidentally
launch.
Profile parsing validates option shapes for `streaming`, `forbidden-read`, and
`timeout` behavioral gates before they reach rubrics.
When both `test_ms` and `worker_ms` are declared, the shorter value becomes the
exec watchdog; when only `worker_ms` is declared, it is used directly.
Rubrics can read:

- `BROWSERGRAD_ASSIGNMENT_ID`
- `BROWSERGRAD_ASSIGNMENT_ROOT`
- `BROWSERGRAD_FIXTURES_PATH` when the profile declares one
- `BROWSERGRAD_ALLOWED_TESTS_JSON`
- `BROWSERGRAD_BEHAVIORAL_GATES_JSON`

Prefer `browsergrad.assignment_context()` inside Python rubrics unless you need
the raw environment values.
For browser-safe streaming checks, Python rubrics can wrap chunk iterables with
`browsergrad.streaming_gate(name, iterable)`. The helper reads
`max_chunks_before_first_yield` from the active behavioral gate when omitted and
raises `StreamingGateViolation` if student code consumes too much input before
the first wrapped output yield:

```py
gate = bg.streaming_gate("encode_iterable_streaming", chunks)
output = gate.wrap_output(student.encode_iterable(gate.input))
first_token = next(iter(output))
```
For browser-safe file behavior, Python rubrics can wrap text with
`browsergrad.forbidden_read_gate(name, text)`. The helper reads the forbidden
`methods` list from the matching `forbidden-read` gate when omitted, permits
incremental iteration/`readline()`, and raises `ForbiddenReadViolation` for
eager `read()` or `readlines()` calls:

```py
file_obj = bg.forbidden_read_gate("no_eager_file_read", fixture_text)
student.process(file_obj)
```

`runAssignmentRubric()` is the one-call Pyodide path for platforms that do not
need to stage each step manually. It derives the mount plan, materializes files
and datasets, builds the rubric exec request, calls `session.exec()`, and
returns `{ mount, exec }`. Use the lower-level helpers when the UI needs a
preflight preview before writing files or launching code.

`assignmentRubricKind(plan)` returns `python`, `javascript`, or `unknown` from
the resolved rubric path. Use it to route JS/WebGPU/native-style labs away from
the Pyodide rubric runner.

`runAssignmentJavascriptRubric()` is the browser-native JS path. It validates
the active run plan, prepares an in-memory read-only mount view, exposes
assignment context, declared oracles, and assertion/artifact helpers to the
rubric function, then returns `{ mount, assertions, artifacts }`. This is the
starting point for GPU Puzzles and CS149-style JS/WebGPU rubrics. Pass browser
resources such as WebGPU adapters/devices through `substrates`, then read them
inside the rubric with `ctx.substrate("webgpu")`. JS rubrics can read mounted
text with `ctx.readText(path)` or mounted bytes with `ctx.readBytes(path)`.
Use `runAssignmentJavascriptProfile()` for full assignment profiles when the
profile's declared oracle modules should be checked before rubric execution.
For JS/TS streaming rubrics, pair this with `createStreamingGate()` from
`@unlocalhosted/browsergrad-tokenizers` and wrap student input/output iterables.

## What this is, and is not

**This is:** a small, well-typed primitive for running Python in a browser worker. ~1,000 LOC. Boots Pyodide, mounts files, exec, stream stdout/stderr, structured assert/artifact protocols, cooperative + hard cancel, persistent namespace.

**This is not:**
- A notebook UI
- A test framework (you compose one from `onAssertion`)
- A grading harness (those live in your platform, not in the library)
- A PyTorch / WebGPU library — those ship as separate packages: [`browsergrad-grad`](../browsergrad-grad/) (eager autograd, stable), [`browsergrad-jit`](../browsergrad-jit/) (lazy IR + fusion + WebGPU seam), [`browsergrad-kernels`](../browsergrad-kernels/) (WGSL primitives + the production realizer bridge)

## API reference

See [`src/types.ts`](./src/types.ts) for the full annotated type surface. Stability contract:

- Adding optional fields → minor version bump
- Removing fields or making them required → major version bump
- Anything not exported from `src/index.ts` is private

## Why not just use Pyodide directly?

You can. Pyodide is excellent. This library packages the boilerplate that every consumer rewrites:

- Worker host + RPC envelope
- Lazy boot with progress callbacks
- AbortSignal-based cancellation (Pyodide alone has no clean cancel path)
- Cooperative cancel via `setInterruptBuffer` with feature detection
- Persistent namespace across `exec` calls
- Typed errors for syntax / runtime / timeout / abort / interrupt
- Structured assertion + artifact protocols, so visible-test UIs don't have to parse `print()` output

If you're booting Pyodide three times across two apps, copy the patterns. If you're booting it many times, install this.

## License

MIT
