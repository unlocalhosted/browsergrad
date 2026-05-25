# Vanilla example

Smallest-possible consumer of `@unlocalhosted/browsergrad-runtime` — no framework, no bundler config beyond what Vite needs for module workers.

This example demonstrates:
- Booting a session with a same-origin Pyodide URL
- Running Python that uses the `browsergrad` module for structured assertions + artifacts
- Cooperative cancel via `AbortSignal`
- Streaming stdout

## Running it

You need a host page served with `COOP: same-origin` + `COEP: require-corp` for the `SharedArrayBuffer`-based cooperative cancel to work. Without those headers everything still functions — just falls back to `worker.terminate()` for cancellation.

Pyodide assets must be served same-origin. The `sync-pyodide.mjs` script in the package README is the easiest way; this example assumes assets are at `/pyodide/v0.26.4/`.

```html
<!-- index.html -->
<!doctype html>
<script type="module" src="./main.js"></script>
<pre id="out"></pre>
```

```js
// main.js
import { createSession } from "@unlocalhosted/browsergrad-runtime";

const out = document.getElementById("out");
const log = (s) => (out.textContent += s + "\n");

const session = await createSession({
  pyodideIndexURL: "/pyodide/v0.26.4/",
  packages: ["numpy"],
  onPackageProgress: (e) => log(`[pkg] ${e.package}: ${e.status}`),
});

log(`session ready (canInterrupt: ${session.canInterrupt})`);

const ctrl = new AbortController();
const result = await session.exec({
  code: `
import numpy as np
import browsergrad as bg
import time

bg.log("start", "running smoke test")

x = np.arange(10, dtype=np.float32)
y = x * 2.0
expected = float(np.sum(np.arange(10) * 2))
actual = float(np.sum(y))

if abs(expected - actual) < 1e-6:
    bg.assert_pass("sum_matches", duration_ms=1.0)
else:
    bg.assert_fail("sum_matches", "sum mismatch", expected=expected, actual=actual)

bg.emit_json("series", {"x": x.tolist(), "y": y.tolist()})

print("done")
`,
  signal: ctrl.signal,
  timeoutMs: 5000,
  onStdout: (chunk) => log("[stdout] " + chunk.trim()),
  onAssertion: (a) => log("[assert] " + JSON.stringify(a)),
  onArtifact: (a) => log("[artifact] " + a.kind + ":" + a.name),
});

log(`exec ok=${result.ok} (${result.durationMs.toFixed(0)} ms)`);
log(`assertions: ${result.assertions.length}, artifacts: ${result.artifacts.length}`);

await session.dispose();
```

## What to expect

```
[pkg] numpy: loading
[pkg] numpy: loaded
session ready (canInterrupt: true)
[artifact] log:start
[assert] {"kind":"pass","name":"sum_matches","durationMs":1}
[artifact] json:series
[stdout] done
exec ok=true (12 ms)
assertions: 1, artifacts: 2
```
