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
```

The library does not interpret these — it just relays them. Build your test framework / visualizer / grader on top.

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
  createAssignmentMountPlan,
  createAssignmentRunPlan,
  createAssignmentRubricExecRequest,
  evaluateAssignmentCapabilities,
  parseAssignmentProfile,
  requiredAssignmentCapabilities,
} from "@unlocalhosted/browsergrad-runtime";

const parsed = parseAssignmentProfile(profileJson);
if (!parsed.ok) throw new Error(parsed.errors.join("; "));

const required = requiredAssignmentCapabilities(parsed.profile);
const preflight = evaluateAssignmentCapabilities(parsed.profile, {
  capabilities: ["pyodide", "torch-compat", "webgpu", "wgsl-kernel"],
});
const plan = createAssignmentRunPlan(parsed.profile, {
  capabilities: ["pyodide", "torch-compat", "webgpu", "wgsl-kernel"],
});
const mounts = createAssignmentMountPlan(plan);
const execRequest = createAssignmentRubricExecRequest(plan);

console.log(required, preflight.ok, preflight.missingCapabilities);
console.log(plan.session.packages, plan.files.rubricPath, plan.execution.allowedTests);
console.log(mounts.files, mounts.datasets);
await session.exec(execRequest);
```

Capability gates support `requires` for all-of requirements and `any_of` for
alternative groups. Non-capability gates such as streaming and timeout remain
rubric/watchdog checks.

`createAssignmentRunPlan()` does not execute student code. It produces the
platform handoff object: package preload list, JS oracle modules, resolved
profile file paths, allowed test ids, timeout hints, dataset declarations,
capability preflight, and behavioral gates.

`createAssignmentMountPlan()` turns a run plan into deterministic file and
dataset mount declarations. It does not fetch or write content; the platform
uses it to decide what to place into `Session.fs` before rubric execution.

`createAssignmentRubricExecRequest()` turns that plan into the minimal
`Session.exec` request for rubric execution. It sets assignment metadata in
environment variables, puts the assignment root on `sys.path`, runs the resolved
rubric path, and uses the profile's test timeout.

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
