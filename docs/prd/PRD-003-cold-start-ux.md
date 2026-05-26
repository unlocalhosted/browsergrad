# PRD-003: Cold-Start UX — Reduce Time-to-First-Train

**Feature ID**: PRD-003
**Status**: Draft
**Authors**: browsergrad maintainers
**Date**: May 2026
**Roadmap position**: P0.3 — Months 1–3
**Companion PRDs**: PRD-008 (OPFS pipeline cache + safetensors streaming)

---

## 1. TL;DR

Today a first visit to any browsergrad-powered lesson pays a ~12-second cold-start penalty before the Python REPL becomes interactive: Pyodide's WASM binary downloads (~10 MB compressed), gets compiled by the browser's WASM JIT, CPython boots inside the worker, and `browsergrad_grad`'s 11 Python source files are written to the virtual filesystem and imported as a smoke check. Nothing is cached between visits. This PRD closes the two biggest wins available in P0: (1) a service worker that caches Pyodide assets and the browsergrad Python bundle after the first download, cutting second-visit boot to under 8 seconds by eliminating all network round-trips; and (2) a structured four-phase boot progress protocol so the user sees forward motion immediately rather than a blank screen for 12 seconds. Compiled WGSL pipeline caching (PRD-008) and `.safetensors` streaming are explicitly out of scope.

---

## 2. Background

### 2.1 The current cold-start cost, broken down by phase

The current boot sequence runs serially inside a dedicated `Worker`. Measured on an M1 MacBook Air (2020, 8 GB RAM) over a 100 Mbps connection with an empty browser cache:

| Phase | What happens | Observed cost |
|---|---|---|
| **P1 — Pyodide asset download** | `pyodide.mjs` (~1 MB), `pyodide.asm.wasm` (~10 MB gz, ~35 MB uncompressed), and `python_stdlib.zip` (~4 MB compressed) fetched from the host origin. Since Pyodide 0.24.0 they download in parallel. | ~4–6 s on 100 Mbps |
| **P2 — WASM compile** | Browser WASM JIT compiles `pyodide.asm.wasm` (~35 MB uncompressed). Chrome uses streaming compilation so this overlaps with download. On older x86 the compile is the dominant CPU cost. | ~2–3 s on M1; up to 5 s on older x86 |
| **P3 — Pyodide init** | `loadPyodide()` completes: CPython boots, `python_stdlib.zip` mounts, `_bg_native` registered, Python preamble runs. | ~1–2 s |
| **P4 — Python source install + smoke import** | `installGrad()` calls `session.fs.write()` for each of 11 `.py` files (serial round-trips), adds mount root to `sys.path`, runs `import browsergrad_grad`. | ~1–2 s |
| **Total first visit** | | **~8–12 s** |
| **Total second visit (no cache)** | Identical — nothing cached today | **~8–12 s** |

Community corroboration: the Pyodide community reports 3–5 s just for `loadPyodide()` ([pyodide/pyodide#1406](https://github.com/pyodide/pyodide/discussions/1406), [#3940](https://github.com/pyodide/pyodide/issues/3940)), plus download cost on first visit.

The education research dossier cited in PRD.md §7 P0.3 marks **<15 seconds** as the minimum-viable cold-start budget for an educational code editor. Community data from PyScript and Pyodide educational deployments confirms that 12–15 second blank screens cause students to close the tab.

### 2.2 Why the Pyodide interpreter cannot be snapshot-cached

An obvious question: why not snapshot the Pyodide interpreter after `browsergrad_grad` imports and restore on second visit? CPython's heap is a graph of C structs with pointer cycles, file descriptors, and engine-internal state. There is no supported serialization format for a live CPython interpreter. Pyodide has no `freeze`/`thaw` API ([pyodide/pyodide#806](https://github.com/pyodide/pyodide/issues/806), open since 2021). Cloudflare achieves fast cold starts for Python Workers by snapshotting at deploy time — a server-side build step — which is not available to a browser runtime.

The achievable alternative: cache the bytes (Pyodide WASM + Python source bundle), skip the download, and let the install run fresh every time. On a cache hit the install is fast because it skips network entirely.

---

## 3. User Stories

**Student, first visit.** A student clicks a craftingattention lesson link for the first time on a school laptop. The code editor appears within 1 second. A progress indicator reads "Downloading Python environment... 38%," then "Initializing Python...," then "Installing browsergrad...," then "Ready." All four phases complete within 12 seconds on a 4-year-old laptop.

**Student, second visit.** Same student opens the lesson the next morning. Browser serves all Pyodide assets from the service worker cache — no network round-trips. Boot completes within 8 seconds. The progress indicator still fires all four phase messages but skips the download phase instantly.

**Host page author (craftingattention).** A developer registers the service worker once in `_app.tsx`, passes `serviceWorkerUrl` to `createSession`, subscribes to `onPackageProgress`, and renders a progress bar in React.

**Developer on localhost.** Service worker registration proceeds normally (localhost is a secure context). On `file://`, registration is skipped gracefully with a warning log, no thrown error.

**Third-party embedder.** Developer at a different platform omits `serviceWorkerUrl`. `createSession` works exactly as today — opt-in means no implicit behavior change.

---

## 4. Goals and Non-Goals

### Goals

1. **Second-visit boot time**: `createSession()` + `installGrad()` resolves in 8 seconds or less on an M1 MacBook Air (2020, 8 GB RAM), Chrome stable, 100 Mbps, cache warm.
2. **First-visit boot time**: same machine, empty cache — 12 seconds or less (no regression).
3. **Four-phase progress events**: `onPackageProgress` subscriber receives all four `BootPhase` values in order on every boot.
4. **Service worker is opt-in**: no SW registration ever happens without the caller passing `serviceWorkerUrl`.
5. **Graceful fallback**: `file://`, plain `http://` non-localhost, and omitted `serviceWorkerUrl` all produce a working uncached boot, zero errors.
6. **SW caches**: Pyodide WASM and JS assets (cache-first); the browsergrad Python source bundle (versioned, cache-first).

### Non-Goals

- **WGSL pipeline caching in OPFS** — scoped to PRD-008.
- **Pyodide interpreter snapshot** — not feasible (see §2.2).
- **Lazy module loading** — Python source bundle is ~90 KB; lazy install adds complexity for negligible gain.
- **`micropip` package caching** — uses Pyodide's own HTTP caching; not touched here.
- **Full PWA / offline-first** — this is a performance cache, not a PWA.
- **Sub-3s second-visit** — PRD-008's target with pipeline caching.

---

## 5. Architecture

### 5.1 Boot sequence today

```
Main thread                         Worker thread
─────────────────────────────────   ──────────────────────────────────────────────
createSession() called
new Worker(worker/index.js)
postMessage: init{pyodideIndexURL}
                                    fetch pyodide.mjs           ← NETWORK ~1 MB
                                    fetch pyodide.asm.wasm      ← NETWORK ~10 MB gz
                                    fetch python_stdlib.zip     ← NETWORK ~4 MB gz
                                    (parallel since Pyodide 0.24)
                                    Browser JIT compiles wasm   (~2–3 s CPU)
                                    loadPyodide() resolves
                                    py.registerJsModule("_bg_native")
                                    runPythonAsync(PY_PREAMBLE)
                                    postMessage: init:done
createSession() resolves, no cache
installGrad() begins:
  fs.write("tensor.py")             FS.writeFile (11× serial RPC round-trips)
  ... ×11 files ...
  exec(sys.path.insert)             runPythonAsync(sys.path.insert)
  exec(import browsergrad_grad)     runPythonAsync(import browsergrad_grad)
installGrad() resolves
```

### 5.2 Boot sequence after this PRD

```
Host page               Main thread                         Worker thread
────────────────────    ─────────────────────────────────   ──────────────────────────────────
App startup:
navigator.serviceWorker
  .register("/browsergrad-sw.js")

createSession({
  serviceWorkerUrl,
  onPackageProgress,
})
                        verifyServiceWorkerReady()
                        emit phase: "downloading-pyodide"
                        new Worker(worker/index.js)
                        postMessage: init{pyodideIndexURL}

                                                            fetch pyodide.mjs
                                                            ↳ SW: CACHE HIT (2nd visit)
                                                               or NETWORK+cache-write (1st)
                                                            fetch pyodide.asm.wasm  (same)
                                                            fetch python_stdlib.zip (same)

                        ← init:progress{phase:"downloading-pyodide", bytesLoaded, bytesTotal}

                                                            WASM compile (~2–3 s)
                        ← init:progress{phase:"initializing-python"}

                                                            loadPyodide() resolves
                                                            registerJsModule, PY_PREAMBLE
                        ← init:progress{phase:"installing-browsergrad"}

                        init:done received
                        installGrad() runs (11 FS writes + import check)

                        ← phase: "ready" emitted
                        createSession() resolves
```

On cache hit (second visit), download is near-instant: SW serves from `caches.match()` without network. WASM compile + Python init account for the remaining ~3–5 s.

### 5.3 Service worker design

**File**: `packages/browsergrad-runtime/src/service-worker.ts`

Compiled separately to `dist/browsergrad-sw.js`, published to npm so host pages copy it to their public root.

**Registration** is the host page's responsibility:

```ts
// Host page (e.g. craftingattention _app.tsx) — runs once at app startup.
// Delayed to window "load" to avoid competing with critical-path resources.
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/browsergrad-sw.js", { scope: "/" })
      .catch((err) => console.warn("[browsergrad] SW registration failed:", err));
  });
}
```

**Cache name**: `browsergrad-v${VERSION}`. On `activate`, the SW deletes all `browsergrad-v*` caches that do not match the current version.

**Fetch interception rules**:

1. Pyodide assets — any URL matching `*.asm.wasm`, `*python_stdlib.zip`, `*pyodide.mjs`, `*pyodide-lock.json`: **cache-first**. These are version-pinned by the `pyodideIndexURL` path.
2. The browsergrad Python bundle — URL pattern `/__browsergrad_bundle__/v${VERSION}`: **network-first on SW install, cache-first thereafter**. The SW constructs this response internally from the Python source bundled at build time. No actual network request.
3. All other requests: pass through unchanged.

**`self.skipWaiting()` + `clients.claim()`**: called during SW install to minimize the window where the SW is in `waiting` state.

### 5.4 COOP/COEP — what this PRD does NOT affect

Service worker caching does NOT require COOP/COEP headers. Those are required only for `SharedArrayBuffer` (cooperative cancellation). This PRD adds no new cross-origin isolation requirements.

| Headers set | Cache warm | Cooperative cancel |
|---|---|---|
| Neither | Yes (this PRD) | No |
| COOP + COEP | Yes | Yes |

### 5.5 Service worker availability and fallbacks

| Condition | Behavior |
|---|---|
| `serviceWorkerUrl` omitted | No SW registration. Boot proceeds as today. |
| `location.protocol === "file:"` | `sw-registration.ts` skips. Warning logged. Boot uncached. |
| `http://` non-localhost | `"serviceWorker" in navigator` is `false` on non-secure HTTP. Short-circuits. |
| `http://localhost` or HTTPS | SW registers normally. |
| SW stuck in `waiting` (old version active) | `verifyServiceWorkerReady()` times out at 5 s, proceeds uncached. |
| `serviceWorkerUrl` cross-origin | Logs a warning and skips — cross-origin SWs not supported. |

### 5.6 Boot progress protocol extension

**Current** (`types.ts`, `protocol.ts`): `PackageProgressEvent` carries `{ package: string, status: "loading" | "loaded" | "failed", message?: string }`.

**After this PRD** — backward-compatible extension:

```ts
export type BootPhase =
  | "downloading-pyodide"
  | "initializing-python"
  | "installing-browsergrad"
  | "ready";

export interface PackageProgressEvent {
  // Existing fields (unchanged)
  readonly package: string;
  readonly status: "loading" | "loaded" | "failed";
  readonly message?: string;

  // New optional fields (absent on per-package events)
  readonly phase?: BootPhase;
  readonly bytesLoaded?: number;  // absent if Pyodide lacks a progress API
  readonly bytesTotal?: number;
}
```

Phase events are emitted with `package: ""`.

**Where each phase is emitted**:

- `"downloading-pyodide"`: worker emits at `bootPyodide()` entry.
- `"initializing-python"`: worker emits when `loadPyodide()` resolves.
- `"installing-browsergrad"`: main-thread emits immediately before `installGrad()`.
- `"ready"`: main-thread emits when `installGrad()` resolves.

The `"installing-browsergrad"` and `"ready"` events originate on the main thread (not the worker), so they post directly through the `onInitProgress` callback rather than through the worker message channel.

---

## 6. API Surface

### 6.1 Updated `SessionOptions`

```ts
// packages/browsergrad-runtime/src/types.ts

export interface SessionOptions {
  pyodideIndexURL: string;
  packages?: readonly string[];
  disableInterruptBuffer?: boolean;
  worker?: Worker;
  onPackageProgress?: (event: PackageProgressEvent) => void;

  /**
   * URL of the browsergrad service worker, relative to the host page's origin.
   * When present, createSession() calls verifyServiceWorkerReady(serviceWorkerUrl)
   * before spawning the worker. The host page is responsible for calling
   * navigator.serviceWorker.register(serviceWorkerUrl) at app startup.
   *
   * Omitting this field disables all service-worker behavior. No implicit
   * registration is ever attempted.
   *
   * Example: "/browsergrad-sw.js"
   */
  serviceWorkerUrl?: string;
}
```

### 6.2 Host-page integration pattern

```ts
import { createSession } from "@unlocalhosted/browsergrad-runtime";
import { installGrad } from "@unlocalhosted/browsergrad-grad";

// 1. Register the SW once at app startup — delayed to window load.
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/browsergrad-sw.js", { scope: "/" })
      .catch((e) => console.warn("[browsergrad] SW registration failed:", e));
  });
}

// 2. Create the session with progress tracking.
const session = await createSession({
  pyodideIndexURL: "/pyodide/v0.26.4/",
  serviceWorkerUrl: "/browsergrad-sw.js",
  onPackageProgress(event) {
    if (event.phase) {
      switch (event.phase) {
        case "downloading-pyodide":
          if (event.bytesTotal) {
            const pct = Math.round((event.bytesLoaded! / event.bytesTotal) * 100);
            ui.setStatus(`Downloading Python environment… ${pct}%`);
          } else {
            ui.setStatus("Downloading Python environment…");
          }
          break;
        case "initializing-python":
          ui.setStatus("Initializing Python…");
          break;
        case "installing-browsergrad":
          ui.setStatus("Installing browsergrad…");
          break;
        case "ready":
          ui.setStatus("Ready", { complete: true });
          break;
      }
    } else if (event.package) {
      ui.setStatus(`Loading ${event.package}…`);
    }
  },
});

// 3. Install the Python grad package.
await installGrad(session);
```

### 6.3 `installGrad` signature extension

```ts
// packages/browsergrad-grad/src/install.ts

export interface InstallOptions {
  mountRoot?: string;
  skipImportCheck?: boolean;
  /** If provided, called with phase events during the install. */
  onProgress?: (event: PackageProgressEvent) => void;
}
```

### 6.4 New and modified files summary

| File | Action | What changes |
|---|---|---|
| `packages/browsergrad-runtime/src/types.ts` | Modify | Add `BootPhase`; extend `PackageProgressEvent`; add `serviceWorkerUrl?` to `SessionOptions` |
| `packages/browsergrad-runtime/src/protocol.ts` | None | No wire format changes |
| `packages/browsergrad-runtime/src/worker/index.ts` | Modify | `bootPyodide()` emits `"downloading-pyodide"` and `"initializing-python"` phase events |
| `packages/browsergrad-runtime/src/client.ts` | Modify | Calls `verifyServiceWorkerReady()` when `serviceWorkerUrl` present; emits `"installing-browsergrad"` and `"ready"` |
| `packages/browsergrad-runtime/src/service-worker.ts` | Create | SW source: fetch interception, cache-first for Pyodide assets, versioned Python bundle |
| `packages/browsergrad-runtime/src/sw-registration.ts` | Create | `verifyServiceWorkerReady(url)` — protocol check, 5 s timeout |
| `packages/browsergrad-runtime/package.json` | Modify | Add SW build entry: `esbuild src/service-worker.ts --bundle --platform=browser --format=iife --outfile=dist/browsergrad-sw.js` |
| `packages/browsergrad-grad/src/install.ts` | Modify | Add optional `onProgress` to `InstallOptions`; emit phase events |

---

## 7. Implementation Plan

### Week 1 — Boot progress protocol (no caching yet)

- [ ] Add `BootPhase` type and extend `PackageProgressEvent` in `types.ts`. All new fields optional — zero breaking change.
- [ ] Add `serviceWorkerUrl?: string` to `SessionOptions` (wired up in Week 2).
- [ ] In `worker/index.ts` `bootPyodide()`: emit phase events at entry and at `loadPyodide()` resolution.
- [ ] In `client.ts` `init()`: emit `"installing-browsergrad"` directly via `onInitProgress`; emit `"ready"` after `installGrad()` resolves.
- [ ] Extend `InstallOptions` in `browsergrad-grad/src/types.ts`.
- [ ] Unit tests: mock `onPackageProgress`, boot a test session, assert all four `phase` strings arrive in order.
- [ ] Update README with the four-phase progress-bar snippet.

Deliverable: every boot emits all four phase events. No SW yet.

### Week 2 — Service worker + cache integration

- [ ] Write `service-worker.ts`:
  - `install`: open `browsergrad-v${VERSION}` cache; pre-cache synthetic Python bundle for `/__browsergrad_bundle__/v${VERSION}`.
  - `activate`: delete `browsergrad-v*` caches not matching current version. Call `self.skipWaiting()` + `clients.claim()`.
  - `fetch`: intercept Pyodide asset URLs with cache-first; pass others through.
- [ ] Add SW build step. Verify output is <50 KB gzip.
- [ ] Write `sw-registration.ts`: `verifyServiceWorkerReady(swUrl): Promise<void>`. Checks protocol, waits up to 5 s for `navigator.serviceWorker.ready`.
- [ ] Wire `verifyServiceWorkerReady()` into `createSession()` in `client.ts`.
- [ ] Add Playwright timing tests (see §9): first-visit ≤12 s, second-visit ≤8 s, four phases in order.
- [ ] Manual validation on M1 MacBook Air; log numbers in `PROGRESS.md`.
- [ ] Add SW versioning test: activate two SW versions in sequence, assert only current-version cache survives.

---

## 8. Acceptance Criteria

1. **First-visit cold start**: ≤12 s on M1 MacBook Air, Chrome stable, 100 Mbps, empty cache. Verified by Playwright.
2. **Second-visit warm cache**: ≤8 s with SW active and assets cached.
3. **Four phase events in order**: `onPackageProgress` receives all four `BootPhase` values in order on every boot.
4. **Service worker opt-in enforced**: omitting `serviceWorkerUrl` produces zero `navigator.serviceWorker.register` calls.
5. **Graceful fallback**: `createSession()` completes without error on `file://`, `http://` non-localhost, omitted SW URL.
6. **No regression**: all 234 existing integration tests pass.
7. **Cache versioning**: after library version bump, old SW cache entry evicted on SW activation.

---

## 9. Test Strategy

### Playwright timing tests — `packages/browsergrad-runtime/tests/e2e/cold-start.spec.ts`

```ts
test("first visit: createSession + installGrad ≤ 12s", async ({ browser }) => {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const t0 = Date.now();
  await page.goto("/test-harness.html");
  await page.waitForFunction(() => (window as any).__bgReady === true, { timeout: 14_000 });
  expect(Date.now() - t0).toBeLessThanOrEqual(12_000);
});

test("second visit (cache warm): ≤ 8s", async ({ browser }) => {
  const ctx = await browser.newContext();
  const firstPage = await ctx.newPage();
  await firstPage.goto("/test-harness.html");
  await firstPage.waitForFunction(() => (window as any).__bgReady === true, { timeout: 14_000 });

  const page = await ctx.newPage();
  const t0 = Date.now();
  await page.goto("/test-harness.html");
  await page.waitForFunction(() => (window as any).__bgReady === true, { timeout: 9_000 });
  expect(Date.now() - t0).toBeLessThanOrEqual(8_000);
});

test("four boot phases received in correct order", async ({ browser }) => {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const phases: string[] = [];
  await page.exposeFunction("recordPhase", (p: string) => phases.push(p));
  await page.goto("/test-harness-phases.html");
  await page.waitForFunction(() => (window as any).__bgReady === true, { timeout: 14_000 });
  expect(phases).toEqual([
    "downloading-pyodide",
    "initializing-python",
    "installing-browsergrad",
    "ready",
  ]);
});
```

In CI, Pyodide WASM assets are served from a local fixture directory via Playwright's route interception to avoid real network calls.

### Manual hardware lab

Before shipping, a maintainer runs the timing test on:
- **Primary reference**: M1 MacBook Air (2020, 8 GB RAM) — acceptance criterion baseline
- **Network throttle**: Chrome DevTools "Fast 3G" (~1.6 Mbps) to validate first-visit under constrained bandwidth

Numbers logged in `PROGRESS.md`.

---

## 10. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | SW stuck in `waiting` state on library version update | Medium | Medium | `self.skipWaiting()` + `clients.claim()` on install. `verifyServiceWorkerReady()` 5s timeout. |
| R2 | `file://` origin (local dev without server) | High | Low | Detect `location.protocol === "file:"` and skip registration. No error. |
| R3 | COOP/COEP headers break embedded cross-origin iframes | High | Medium | This PRD adds no new COOP/COEP requirements. SW caching works on any HTTPS page. |
| R4 | Pyodide asset URL structure changes between releases | Low | High | Cache key includes version path. A Pyodide version bump produces cache misses by design. |
| R5 | SW bundle size unexpectedly large | Low | Low | 11 Python files ~90 KB uncompressed, ~30 KB gzip. Total SW <50 KB gzip. Verified at build. |
| R6 | 8-second second-visit target not met on older x86 | Medium | Low | Annotate hardware-specific results in `PROGRESS.md`. M1 is the reference baseline. |
| R7 | Firefox quirks with SW scope or Cache API | Low | Low | Firefox supports SW + Cache API since v44. Standard `caches.match()` only — no Workbox. |
| R8 | Byte-level download progress unavailable in Pyodide 0.26.x | High | Low | `"downloading-pyodide"` event fires without `bytesLoaded`/`bytesTotal` in this version. UX fallback is a spinner. Tracked in [pyodide#2927](https://github.com/pyodide/pyodide/discussions/2927). |

---

## 11. Open Questions

1. **Ship the compiled SW as `dist/browsergrad-sw.js` in the npm package?** Makes integration trivial — host pages copy from `node_modules/`. Recommendation: publish it.

2. **Opt-in vs. opt-out for service worker registration?** Current design requires `serviceWorkerUrl` explicitly. Opt-in is safer for the first SW release.

3. **Should `verifyServiceWorkerReady()` be exported as public API?** Currently internal. May be needed if craftingattention wants to display "service worker updating" UI states.

4. **Test harness serving strategy in CI?** (a) check minimal Pyodide build into `fixtures/`, (b) download in CI setup, (c) mock fetches. Resolve before Week 2 starts.

---

## 12. References

- [Pyodide Discussion #1406 — Performance guidance](https://github.com/pyodide/pyodide/discussions/1406)
- [Pyodide Issue #3940 — Why is initialization so slow?](https://github.com/pyodide/pyodide/issues/3940)
- [Pyodide Issue #806 — Persistent instance cache](https://github.com/pyodide/pyodide/issues/806)
- [Pyodide Discussion #2927 — loadPyodide loading progress](https://github.com/pyodide/pyodide/discussions/2927)
- [Pyodide 0.24.0 release](https://blog.pyodide.org/posts/0.24-release/)
- [web.dev — Service worker registration](https://web.dev/articles/service-workers-registration)
- [Chrome Developers — Workbox caching strategies](https://developer.chrome.com/docs/workbox/caching-strategies-overview)
- [MDN — Using Service Workers](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API/Using_Service_Workers)
- [web.dev — Why COOP/COEP](https://web.dev/articles/why-coop-coep)
- [HoloViz Discourse — Caching Pyodide for PWA](https://discourse.holoviz.org/t/caching-pyiodide-and-python-env-for-pwa/4358)
- `packages/browsergrad-runtime/src/client.ts` — `SessionImpl`, `createSession`
- `packages/browsergrad-runtime/src/protocol.ts` — `InitProgressEvent`, `PackageProgressEvent`
- `packages/browsergrad-runtime/src/types.ts` — `SessionOptions`, `PackageProgressEvent`
- `packages/browsergrad-runtime/src/worker/index.ts` — `bootPyodide()`, `handleInit()`
- `packages/browsergrad-grad/src/install.ts` — `installGrad()`, 11-file FS write
- `packages/browsergrad-grad/src/python/index.ts` — `SOURCE_FILES`, `MOUNT_ROOT`
