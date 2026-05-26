# PRD-013 — craftingattention Lab Platform Alignment

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-013 |
| **Phase** | P2 (Months 11–14 of the 14-month roadmap in PRD.md §6) |
| **Package** | `@unlocalhosted/browsergrad-runtime` (new submodule `lab/`), `@unlocalhosted/browsergrad-jit` (Python `lab/` package), `@unlocalhosted/bg-lab-cli` (new npm-publishable validator) |
| **Depends on** | PRD-004 (assertion + artifact relay), PRD-005 (IR — trace identity for the harness), PRD-008 (OPFS + safetensors streamer — asset pipeline), PRD-009 (gradient checkpointing — required by fast.ai Ch 18 + CS336 A1 transformer labs) |
| **Enables** | craftingattention's first 5 public lessons; fast.ai Part 2 Ch 9–25 as in-browser labs; Stanford CS336 A1–A5 as labs; deep-ml problem catalog as drop-in exercises |
| **Companion docs** | [PRD.md](../../PRD.md) §3.7, §7 P2.3 · [VISION.md](../../VISION.md) §3 · [PRD-004](PRD-004-lab-runtime-api.md) (the substrate) |

---

## TL;DR

PRD-004 shipped Python helpers (`bg.assert_close`, `bg.plot_loss`, `bg.checkpoint`) and the artifact/assertion wire protocol that lets one lab run end-to-end in browsergrad. PRD-004 did not specify the **contract** between browsergrad and the platform hosting hundreds of labs across multiple curricula. PRD-013 is that bridge. It defines (1) a JSON-Schema'd **lab manifest** declaring required ops, datasets, memory budget, wall-clock target, and rubric reference; (2) a **curriculum coverage matrix** mapping every fast.ai Part 2 chapter, every Stanford CS336 sub-problem, and every deep-ml problem to the browsergrad PRDs each requires; (3) a **server-less in-browser grading harness** with numerical-match, gradient-norm-sanity, convergence, wall-clock, no-nan-inf, and shape-match primitives; (4) a **sandbox / security model** — Pyodide worker boundary, `bg.fetch` allowlist, `bg.allow_dynamic_exec` opt-in; (5) an **asset pipeline** bundling MNIST, CIFAR-10, tiny-shakespeare and friends as safetensors blobs streamed via PRD-008's OPFS reader; (6) **privacy-first opt-in telemetry** emitting only lab id + event type + assertion name + anonymized session token; (7) a thin **authoring SDK** (TypeScript types only, consumed by craftingattention's private React UI); and (8) a **semver-range versioning contract** with a legacy-runtime registry so an exercise written in May 2026 still grades correctly in 2028. Real implementation only — no stubs. Manifest format is machine-readable (JSON Schema draft 2020-12) so CI tools refuse a malformed exercise before publish.

---

## Background

### Why PRD-004 isn't enough

PRD-004 was scoped at "one lab works end-to-end" — deep-ml problem #25 single-neuron backprop running in Pyodide-in-node. That was correct for P0. Scaling to a curriculum needs five things PRD-004 did not provide:

- **Hundreds of exercises** instead of one hand-coded `lab-e2e-*.test.ts`.
- **Authors who aren't browsergrad maintainers** writing rubrics without reading `protocol.ts`.
- **Grading that survives version drift** — an exercise authored against `browsergrad-jit@0.4.0` must still grade against `1.3.0`, or refuse to run with a clear message.
- **Datasets larger than a Python string literal**. MNIST is 60 MB; CIFAR-10 is 175 MB; tiny-shakespeare is 1 MB; sentencepiece models are 5–20 MB.
- **Rubrics richer than `(actual - expected).max() < atol`** — gradient-norm sanity (nan/inf, exploding norms), convergence bands, optimizer-state-after-step, wall-clock.

PRD-013 supplies the *shape* of those things. Exercise content lives in craftingattention's private repo; the *contract* lives here.

### Why JSON manifest and not a Python decorator

A draft considered `@bg.lab(...)` decorators. Rejected because (1) CI validation must parse manifests without executing Python; (2) version pinning must be readable outside Pyodide so the host can refuse incompatible runtimes; (3) JSON Schema gives VS Code completions and GitHub Action linters for free; (4) a future server-side grader benefits from a language-neutral manifest. The shape is closest to [Quarto's `_quarto.yml`](https://quarto.org/docs/projects/quarto-projects.html), plus an inline rubric reference (see also [IMS Caliper Analytics](https://www.imsglobal.org/spec/caliper/v1p2) and [Jupyter Book `_config.yml`](https://jupyterbook.org/en/stable/customize/config.html)).

### The three target curricula

Per PRD.md §3.7:

- **Fast.ai Part 2 ([course.fast.ai/Lessons/part2.html](https://course.fast.ai/Lessons/part2.html))** — 17 chapters spanning Ch 9 (Stable Diffusion deep dive), Ch 11 (matmul from scratch), Ch 13 (backprop from scratch), Ch 17 (init + norm), Ch 18 (accelerated SGD), Ch 19 (DDPM from scratch), Ch 20 (mixed precision), Ch 24 (attention from scratch), Ch 25 (latent diffusion).
- **Stanford CS336 ([stanford-cs336.github.io/spring2024/](https://stanford-cs336.github.io/spring2024/))** — 5 assignments: A1 Basics (tokenizer + transformer + AdamW), A2 Systems (flash-attention from scratch), A3 Scaling, A4 Data, A5 Alignment (SFT/DPO/RLHF). Each is ~10 graded units.
- **deep-ml.com ([Open-Deep-ML/DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem))** — ~250 problems across linear algebra, classical ML, deep learning (CNN/RNN/LSTM/transformer), optimization (SGD/Adam/Adagrad), normalization (BN/LN/GN), and LLM (KV cache, top-k, RoPE, GQA).

Together: ~400 distinct exercises. The integration contract must be uniform across all three.

---

## User Stories

**U1 — Author writes a fast.ai Ch 11 lab.** Maya picks the `fastai-part2-ch11` template, fills the manifest (dataset = `mnist-1k-shard`, target 3 s wall-clock, required ops = `MATMUL, ADD, RELU`), writes starter + reference solution side-by-side, authors a rubric. She clicks Publish; CI runs `bg-lab validate`, the reference solution grades to "pass," and the lab ships.

**U2 — Student runs a CS336 transformer-from-scratch exercise.** Karan loads a CS336 A1 sub-problem. The host reads `manifest.json`, sees `requires_browsergrad: ">=0.7.0 <2.0.0"`, validates the runtime is `1.2.0` (in range), and pre-streams `tiny-shakespeare.safetensors` via PRD-008's OPFS reader. Karan implements `attention(Q,K,V,mask)`; presses Run; the grader emits one `AssertionPass` per rubric assertion and an `ArtifactJson` heatmap of the attention pattern.

**U3 — Student stuck on backward-pass exercise.** Three rubric assertions fail. Opt-in telemetry records `(lab_id, attempt=4, failed_assertions=["weight_grad","bias_grad"], stuck_duration_s=412)`. After 5 minutes inactivity the pedagogy layer offers a hint. **No source code, no tensor values, no PII leaves the browser.**

**U4 — Curriculum drift over two years.** A fast.ai Ch 18 lab authored in 2026 pins `>=0.7.0 <1.0.0`. In 2028 browsergrad is at `2.3.0`. A new student loads the lab; the runtime sees the pin doesn't satisfy `2.3.0`, refuses default, and transparently boots `0.9.5` from the legacy-runtime CDN. The student notices only a small "running on legacy runtime" indicator.

**U5 — CI gate on an exercise PR.** A craftingattention PR adds a new lab. GitHub Actions runs `npx @unlocalhosted/bg-lab validate ./labs/**/manifest.json`. One manifest has a typo in `required_ops`; the action fails with a JSON Pointer path and an error line. The contributor fixes and re-pushes.

---

## Goals and Non-Goals

### Goals

1. Ship `lab-manifest.schema.json` (JSON Schema draft 2020-12) describing exercise metadata: id, version, browsergrad version range, datasets, required ops, memory budget, expected wall-clock, rubric reference, policy.
2. Publish the curriculum coverage matrix (JSON source + Markdown render) for fast.ai Part 2 Ch 9–25, CS336 A1–A5, and the deep-ml catalog.
3. Ship `browsergrad_jit.lab.harness` with `assert_pytorch_match`, `assert_gradient_norm_sane`, `assert_convergence`, `assert_wall_clock`, `assert_no_nan_inf`, `assert_shape_match`, and `grade_run`.
4. Document and ship the sandbox model: Pyodide worker boundary, `bg.fetch` allowlist, `bg.allow_dynamic_exec` opt-in, OPFS mediation, WGSL isolation.
5. Ship the dataset pipeline (`bg.lab.dataset(name)` resolving safetensors via PRD-008) with initial registry: MNIST (1k / 10k / 60k shards), CIFAR-10 (full + 5k subset), tiny-shakespeare, TinyStories-1M, 64-px face thumbnails.
6. Ship opt-in telemetry that emits only `(lab_id, event_type, anonymized_session_token, timestamp, optional failed_assertion_name)`. No code, no tensor values, no PII.
7. Publish authoring SDK types (`@unlocalhosted/browsergrad-runtime/lab/authoring`) consumed by craftingattention's React UI.
8. Ship the versioning contract: a `requires_browsergrad` semver range + legacy-runtime CDN bundling every released version forever.
9. Publish `npx @unlocalhosted/bg-lab validate` CLI for author CI.

### Non-Goals

1. The craftingattention lesson UI itself (private repo).
2. Server-side grading. Everything runs in the browser.
3. Marketplace / monetization.
4. Identity / login / progress tracking.
5. Real-time collaborative editing.
6. i18n of rubric error messages.

---

## Architecture

### Module layout

```
packages/browsergrad-runtime/
  src/lab/
    manifest.ts              # parser + ajv JSON Schema validator
    schema/
      lab-manifest.schema.json
      curriculum-coverage.schema.json
    harness.ts               # TS-side grader event router (wraps PRD-004 assertion stream)
    policy.ts                # network + dynamic-exec policy enforcement
    dataset-resolver.ts      # name -> URL -> OPFS path (calls PRD-008)
    telemetry.ts             # opt-in emitter; respects DNT + explicit consent
    versioning.ts            # semver range + legacy runtime loader
    authoring/               # SDK types craftingattention imports
      index.ts
      types.ts

packages/browsergrad-jit/
  src/python/lab/
    harness.py               # rubric assertion primitives
    dataset.py               # bg.lab.dataset(name) + safetensors mmap into TensorProxy
    policy.py                # bg.lab.policy() + bg.fetch wrapper

packages/bg-lab-cli/          # new npm package
  src/{validate,fixtures-load,coverage-matrix}.ts
```

### Manifest schema (excerpt)

Full schema at `packages/browsergrad-runtime/src/lab/schema/lab-manifest.schema.json`. The load-bearing fields:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://unlocalhosted.dev/schemas/bg-lab-manifest-1.json",
  "type": "object",
  "required": ["id", "version", "title", "requires_browsergrad", "rubric_path", "starter_path"],
  "additionalProperties": false,
  "properties": {
    "id":                   { "type": "string", "pattern": "^[a-z0-9-]+(/[a-z0-9-]+)*$" },
    "version":              { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "title":                { "type": "string", "minLength": 4, "maxLength": 120 },
    "curriculum":           { "enum": ["fastai-part2", "cs336", "deep-ml", "craftingattention-original"] },
    "curriculum_chapter":   { "type": "string" },
    "learning_goals":       { "type": "array", "items": { "type": "string" }, "minItems": 1, "maxItems": 8 },
    "requires_browsergrad": { "type": "string", "description": "semver range, e.g. >=0.7.0 <2.0.0" },
    "required_ops":         { "type": "array", "items": { "type": "string" } },
    "required_prds":        { "type": "array", "items": { "pattern": "^PRD-\\d{3}$" } },
    "datasets":             { "type": "array", "items": { "$ref": "#/$defs/dataset_ref" } },
    "memory_budget_mb":     { "type": "integer", "minimum": 16, "maximum": 1500 },
    "expected_wall_clock_ms": { "type": "integer", "minimum": 50, "maximum": 60000 },
    "starter_path":         { "type": "string" },
    "reference_path":       { "type": "string" },
    "rubric_path":          { "type": "string" },
    "policy":               { "$ref": "#/$defs/lab_policy" },
    "telemetry":            { "type": "object", "properties": { "opt_in": { "type": "boolean", "default": false } } }
  },
  "$defs": {
    "dataset_ref": { "type": "object", "required": ["name", "sha256", "max_bytes"],
      "properties": { "name": {"type":"string"}, "sha256": {"type":"string","pattern":"^[0-9a-f]{64}$"}, "max_bytes": {"type":"integer"} } },
    "lab_policy": { "type": "object",
      "properties": { "fetch_allowlist": {"type":"array","items":{"type":"string","format":"uri"}},
                      "allow_dynamic_exec": {"type":"boolean","default":false},
                      "max_workers": {"type":"integer","minimum":1,"maximum":4,"default":1} } }
  }
}
```

`additionalProperties: false` is deliberate — unknown fields fail validation so the schema evolves additively under a `schema_version` field (see Risks R1).

### Curriculum coverage matrix (excerpt)

Full table at `packages/bg-lab-cli/src/data/curriculum-coverage.json`; CI publishes Markdown to `docs/curriculum-coverage.md`. Representative rows:

| Curriculum | Chapter/Problem | Title | Required PRDs | Notes |
|---|---|---|---|---|
| fast.ai-pt2 | ch11 | Matmul from scratch | PRD-001, 005 | Triple-loop → broadcast → einsum |
| fast.ai-pt2 | ch13 | Backprop from scratch | PRD-005, 007 | VJP rules by hand |
| fast.ai-pt2 | ch17 | Init + normalization | PRD-001, 005 | |
| fast.ai-pt2 | ch18 | Accelerated SGD | PRD-005, 007, 009 | **PRD-009 required** for headroom |
| fast.ai-pt2 | ch19 | DDPM from scratch | PRD-005, 006, 008, 009 | Streams 50 MB unet |
| fast.ai-pt2 | ch20 | Mixed precision | PRD-010 | Authentic fp16 path |
| fast.ai-pt2 | ch24 | Attention from scratch | PRD-005, 006, 012 | Megakernel optional |
| fast.ai-pt2 | ch25 | Latent diffusion | PRD-006, 008, 009, 010 | Streams ~250 MB SD-tiny |
| cs336 | a1.2 | BPE tokenizer | PRD-001 | CPU-only |
| cs336 | a1.5 | Scaled dot-product attention | PRD-005, 006 | |
| cs336 | a1.7 | AdamW + LR schedule | PRD-001, 005 | Optimizer-state assertion |
| cs336 | a2.3 | Flash-attention from scratch | PRD-006, 012, 015 | PRD-015 (custom WGSL) |
| cs336 | a5.2 | DPO loss + step | PRD-005, 007, 014 | PRD-014 (vmap) |
| deep-ml | 25 | Single neuron backprop | PRD-001, 004 | Already in PRD-004 |
| deep-ml | 107 | Masked self-attention | PRD-001, 005 | |
| deep-ml | 130 | Simple CNN training | PRD-001, 002, 005 | |
| deep-ml | 188 | Gradient checkpointing | PRD-009 | Pedagogical case for PRD-009 |
| deep-ml | LLM category | All ~18 problems | PRD-001, 005, 006, 008 | Top-k, RoPE, KV cache, GQA |

The full matrix has ~400 rows.

### Grading harness — Python side

`packages/browsergrad-jit/src/python/lab/harness.py`:

```python
"""Imported as `from browsergrad_jit.lab import harness as bg_grade`."""
from __future__ import annotations
import time
import numpy as np
import browsergrad as bg
import browsergrad_jit as torch

def assert_pytorch_match(name, actual, expected_numpy, *, atol=1e-4, rtol=0.0):
    a = actual.numpy() if hasattr(actual, "numpy") else np.asarray(actual)
    e = np.asarray(expected_numpy)
    if a.shape != e.shape:
        bg.emit_assertion(name, "error", f"shape {a.shape} != expected {e.shape}"); return
    diff = float(np.max(np.abs(a - e)))
    tol  = atol + rtol * float(np.max(np.abs(e)))
    bg.emit_assertion(name, "pass" if diff <= tol else "fail",
                      f"max_diff={diff:.2e} {'<=' if diff <= tol else '>'} tol={tol:.2e}")

def assert_gradient_norm_sane(name, params, *, max_norm=1e3):
    bad = []
    for p in params:
        if p.grad is None: bad.append("no .grad"); continue
        g = p.grad.numpy()
        if not np.all(np.isfinite(g)): bad.append(f"non-finite: {int(np.sum(~np.isfinite(g)))}")
        n = float(np.linalg.norm(g))
        if n > max_norm: bad.append(f"L2 {n:.2e} > {max_norm:.2e}")
    bg.emit_assertion(name, "fail" if bad else "pass", "; ".join(bad) or f"{len(list(params))} OK")

def assert_convergence(name, losses, *, max_final, max_increase_steps=0):
    final = float(losses[-1])
    inc = sum(1 for i in range(1, len(losses)) if losses[i] > losses[i-1])
    ok = final <= max_final and inc <= max_increase_steps
    bg.emit_assertion(name, "pass" if ok else "fail",
                      f"final={final:.4f}, increases={inc}")

def assert_wall_clock(name, fn, *, max_ms, warmup=1, iters=3):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    elapsed = (time.perf_counter() - t0) * 1000 / iters
    bg.emit_assertion(name, "pass" if elapsed <= max_ms else "fail",
                      f"{elapsed:.1f}ms vs {max_ms}ms")

def assert_no_nan_inf(name, tensor):
    a = tensor.numpy()
    if np.all(np.isfinite(a)):
        bg.emit_assertion(name, "pass", "all finite")
    else:
        bg.emit_assertion(name, "fail",
                          f"{int(np.sum(np.isnan(a)))} nan, {int(np.sum(np.isinf(a)))} inf")

def assert_shape_match(name, tensor, expected_shape):
    ok = tuple(tensor.shape) == tuple(expected_shape)
    bg.emit_assertion(name, "pass" if ok else "fail",
                      f"shape={tensor.shape}{'' if ok else f' != {expected_shape}'}")
```

A rubric per lab (in craftingattention's repo) looks like:

```python
# rubric.py for fastai-part2/ch11/matmul-from-scratch
from browsergrad_jit.lab import harness as bg_grade
import numpy as np

def grade(student_module):
    A = np.random.RandomState(42).randn(64, 128).astype(np.float32)
    B = np.random.RandomState(43).randn(128, 32).astype(np.float32)
    expected = A @ B
    actual = student_module.matmul(A, B)
    bg_grade.assert_shape_match("matmul_shape", actual, (64, 32))
    bg_grade.assert_pytorch_match("matmul_value", actual, expected, atol=1e-4)
    bg_grade.assert_wall_clock("matmul_speed", lambda: student_module.matmul(A, B), max_ms=50)
```

### Sandbox / security model

Inherits PRD.md §3.8's "security by default" and instantiates concretely:

1. **Pyodide worker boundary.** Student + rubric code execute inside the runtime's existing Web Worker. The host never holds a `pyodide` reference; interaction is via PRD-004's structured messages. Host can unconditionally call `session.dispose()`.
2. **`bg.fetch` allowlist.** New `bg.fetch(url, *, max_bytes=10_000_000, timeout_ms=15000)` posts a `FetchRequest`; the host consults `manifest.policy.fetch_allowlist`. Out-of-allowlist requests raise `bg.SecurityError`. `urllib`, `requests`, and `pyodide.http.pyfetch` are monkey-patched at preamble time to route through `bg.fetch`; the patches are sealed against `del`.
3. **`bg.allow_dynamic_exec` opt-in.** When `false` (default), `sys.setprofile` records direct `eval`/`exec`/`compile` calls; the UI flags them as unusual. When `true`, allowed and only debug-logged.
4. **OPFS mediation.** No direct `FileSystemSyncAccessHandle` in Python. The only writes are `bg.checkpoint` (PRD-004) and `bg.lab.dataset(name)` (this PRD via PRD-008).
5. **WGSL pipelines** compile only in the runtime worker; user code never holds a `GPUDevice`. Custom WGSL (PRD-015) inherits the same isolation.
6. **Cross-origin isolation.** Host is expected to ship COOP+COEP. When absent, runtime degrades to single-threaded silently. Manifest may declare `requires_coop_coep: true`; host shows a clear message if headers are missing.
7. **Telemetry never sees** Python code, tensor values, dataset bytes, or assertion *messages* (only assertion *names*).

### Asset pipeline

`packages/browsergrad-jit/src/python/lab/dataset.py`:

```python
"""bg.lab.dataset(name) -> DatasetDict streamed from OPFS via PRD-008."""
import browsergrad as bg

_REGISTRY = {
    "mnist-1k-shard":      {"url": "https://assets.unlocalhosted.dev/datasets/v1/mnist-1k-shard.safetensors",
                            "sha256": "9c8e...", "bytes": 1_310_720,
                            "tensors": {"x": ("uint8", (1000, 28, 28)), "y": ("uint8", (1000,))}},
    "mnist-10k-shard":     {"url": ".../mnist-10k-shard.safetensors", "sha256": "...", "bytes": 13_000_000, "tensors": {...}},
    "mnist-60k-full":      {"url": ".../mnist-60k.safetensors",       "sha256": "...", "bytes": 78_000_000, "tensors": {...}},
    "cifar-10-5k-subset":  {"url": ".../cifar-10-5k.safetensors",     "sha256": "...", "bytes": 15_000_000, "tensors": {...}},
    "cifar-10-full":       {"url": ".../cifar-10-full.safetensors",   "sha256": "...", "bytes": 175_000_000, "tensors": {...}},
    "tiny-shakespeare":    {"url": ".../tiny-shakespeare.safetensors","sha256": "...", "bytes": 1_115_394,  "tensors": {...}},
    "tinystories-1m":      {"url": ".../tinystories-1m.safetensors",  "sha256": "...", "bytes": 11_000_000, "tensors": {...}},
    "faces-64-thumbnails": {"url": ".../faces-64.safetensors",        "sha256": "...", "bytes": 48_000_000, "tensors": {...}},
}

class DatasetDict:
    def __init__(self, name): self._name, self._cache = name, {}
    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = bg.load_safetensors(_REGISTRY[self._name]["url"])[key]
        return self._cache[key]

def dataset(name):
    if name not in _REGISTRY:
        raise ValueError(f"unknown dataset '{name}'; available: {list(_REGISTRY)}")
    return DatasetDict(name)
```

The first call triggers PRD-008's streamer to fetch the URL, verify sha256 against the manifest, persist to OPFS, and return a `TensorProxy` backed by a `BUFFER` UOp. Subsequent calls hit OPFS directly. Blobs are served from Cloudflare R2 primary + jsdelivr fallback, `Cache-Control: public, max-age=31536000, immutable`. A new dtype convention ships under `v2/`; `v1/` lives forever.

### Telemetry

`packages/browsergrad-runtime/src/lab/telemetry.ts`:

```typescript
export interface LabEvent {
  readonly labId: string;            // "fastai-part2/ch11/matmul-from-scratch"
  readonly eventType: "boot" | "run-start" | "run-complete"
                    | "assertion-pass" | "assertion-fail"
                    | "stuck-detected" | "abandon";
  readonly assertionName?: string;   // rubric-author string only
  readonly sessionToken: string;     // 128-bit random, per tab, never persisted
  readonly timestamp: number;
  readonly browsergradVersion: string;
}

// Hard rules:
//   1. emit() is a no-op until manifest.telemetry.opt_in === true AND setTelemetryConsent(true) called
//      AND navigator.doNotTrack !== "1".
//   2. POST to a single endpoint configured per-deployment (default /api/lab-events, same-origin).
//   3. No field carries code, tensor data, exception messages, stack traces. CI random-fuzzes
//      10 000 payloads against forbidden patterns (R3 in Risks).
//   4. session tokens are forgotten on tab close.
```

### Authoring SDK (consumed by craftingattention)

`packages/browsergrad-runtime/src/lab/authoring/types.ts`:

```typescript
export interface AuthoringSession {
  loadManifest(path: string): Promise<LabManifest>;
  validateManifest(m: LabManifest): ManifestValidationReport;
  runReferenceAgainstRubric(m: LabManifest): Promise<RubricRunReport>;
  renderCoverageMatrixRow(m: LabManifest): CurriculumRow;
}
export interface ManifestValidationReport {
  ok: boolean;
  errors: ReadonlyArray<{ path: string; message: string }>;  // JSON Pointer paths
}
export interface RubricRunReport {
  ok: boolean;
  assertions: ReadonlyArray<{ name: string; verdict: "pass" | "fail" | "error"; detail: string }>;
  wallClockMs: number;
}
```

craftingattention's React UI imports these types and builds Monaco/Codemirror panes on top. The SDK exposes only types + pure-TS validation.

### Versioning contract

Host boot sequence:

1. Fetch + validate manifest.
2. Parse `requires_browsergrad` as semver range.
3. If range satisfied by current `LATEST_DEFAULT` → boot default runtime.
4. Else → resolve the highest satisfying version from `https://cdn.unlocalhosted.dev/legacy-runtimes/<version>/runtime.bundle.js`, boot it, run the lab against it. UI shows a "running on legacy browsergrad X.Y.Z" indicator.
5. Every `browsergrad-jit` release is mirrored to the legacy CDN within 24 h via GH Action; bundles are immutable.

Maintainer contract: a minor version bump must not break any manifest pinned within the same major; a major bump may break but legacy bundles serve forever. A lab pinned `>=0.7.0 <2.0.0` works against `1.99.x` even after `2.0.0` ships.

---

## API Surface

### TypeScript (host page)

```typescript
import { loadLab, runLab } from "@unlocalhosted/browsergrad-runtime/lab";

const lab = await loadLab({
  manifestUrl: "/labs/fastai-part2/ch11/matmul-from-scratch/manifest.json",
  onTelemetryConsentRequired: async () => askUser(),
});

const result = await runLab(lab, {
  studentCode: editor.getValue(),
  onAssertion: (e) => testPanel.update(e),
  onArtifact:  (e) => previewPane.render(e),
  onProgress:  (e) => lossChart.append(e),
});
// result: { ok: boolean; assertions: AssertionEvent[]; wallClockMs: number }
```

### Python (inside lab code)

```python
import browsergrad_jit as torch
from browsergrad_jit.lab import harness as bg_grade, dataset as bg_dataset

data = bg_dataset.dataset("mnist-1k-shard")
x, y = data["x"], data["y"]

model = torch.nn.Linear(784, 10)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
logits = model(x.reshape(-1, 784).float() / 255.0)
loss = torch.nn.functional.cross_entropy(logits, y.long())
loss.backward(); opt.step()

bg_grade.assert_shape_match("logits_shape", logits, (1000, 10))
bg_grade.assert_no_nan_inf("loss_finite", loss)
bg_grade.assert_gradient_norm_sane("grads_sane", model.parameters())
```

### CLI

```
npx @unlocalhosted/bg-lab validate ./labs/**/manifest.json
npx @unlocalhosted/bg-lab grade    ./labs/foo/manifest.json --student ./submission.py
npx @unlocalhosted/bg-lab coverage-matrix --out docs/curriculum-coverage.md
```

---

## Implementation Plan

### Week 1 — JSON Schema + manifest parser

- [ ] Author `lab-manifest.schema.json` (draft 2020-12).
- [ ] Implement `manifest.ts` with `ajv`; emit structured errors with JSON Pointer paths.
- [ ] Unit tests: 20 valid + 30 invalid manifest fixtures.

### Week 2 — Grading harness (Python)

- [ ] Write `lab/harness.py` with all 6 primitives.
- [ ] Integration tests in `tests-integration/lab-harness.test.ts`: drive each via Pyodide-in-node; assert emitted assertion events.
- [ ] Re-run deep-ml #25 rubric through the new harness; assert identical event stream to PRD-004.

### Week 3 — Dataset asset pipeline

- [ ] Build initial registry; produce `.safetensors` blobs for MNIST-1k, tiny-shakespeare, CIFAR-5k.
- [ ] Wire `bg.lab.dataset(name)` through PRD-008's streamer.
- [ ] Tests: cold download + cache, warm hit, sha256-mismatch raise.
- [ ] Stand up R2 + jsdelivr; publish `v1/`.

### Week 4 — Sandbox + policy

- [ ] Implement `bg.fetch` + monkey-patches for `urllib`/`requests`/`pyodide.http`.
- [ ] `bg.allow_dynamic_exec` profile-guard.
- [ ] Host-side `policy.ts` (defense-in-depth: refuses unallowed fetches at the worker boundary too).
- [ ] Security regression suite: 12 escape attempts (fetch to non-allowlisted, `exec` of arbitrary string, `del bg.fetch`, OPFS path traversal, etc.); each must block.

### Week 5 — Telemetry + consent

- [ ] Implement `telemetry.ts`; same-origin POST emitter.
- [ ] Random-fuzz: 10 000 payloads vs forbidden-pattern regex.
- [ ] `setTelemetryConsent` API; refuse to fire until called.

### Week 6 — Versioning + legacy CDN

- [ ] `versioning.ts` with semver `satisfies()`.
- [ ] GH Action: mirror every `browsergrad-jit` release to `cdn.unlocalhosted.dev/legacy-runtimes/<v>/`.
- [ ] Test: host on latest, manifest pinned old, legacy bundle resolves and grades identically.

### Week 7 — Authoring SDK + first 3 labs

- [ ] Publish `@unlocalhosted/browsergrad-runtime/lab/authoring`.
- [ ] Three labs in craftingattention through public APIs only: fast.ai Ch 11, deep-ml #25, CS336 A1.5.

### Week 8 — Coverage matrix + CI gate

- [ ] Author `curriculum-coverage.json` (~400 rows).
- [ ] Implement `bg-lab coverage-matrix --out`.
- [ ] Wire `bg-lab validate **/manifest.json` into craftingattention CI; failure blocks merge.
- [ ] Publish `@unlocalhosted/bg-lab@1.0.0`.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | JSON Schema validates 100% of authored manifests, rejects 100% of crafted-invalid manifests | `manifest.test.ts` |
| AC2 | All 6 grading primitives emit `AssertionPass/Fail/Error` matching PRD-004 wire format | `lab-harness.test.ts` |
| AC3 | `bg.lab.dataset("mnist-1k-shard")` cold path ≤2s, warm path ≤200ms | `dataset-streaming.test.ts` |
| AC4 | `bg.fetch` to non-allowlisted URL raises `bg.SecurityError`; monkey-patched `urllib.request.urlopen` raises same | `sandbox-policy.test.ts` |
| AC5 | Telemetry is no-op until both `manifest.telemetry.opt_in === true` and `setTelemetryConsent(true)` | `telemetry-consent.test.ts` |
| AC6 | A manifest pinned `>=0.7.0 <1.0.0` boots the legacy runtime when host is `1.5.0` | `versioning-legacy.test.ts` |
| AC7 | `bg-lab validate` rejects a manifest with `required_ops:["MATMUL"]` whose reference uses `CONV2D` | `cli-validate.test.ts` |
| AC8 | Coverage matrix has ≥1 row per public fast.ai chapter / CS336 sub-problem / deep-ml problem | `coverage-matrix.test.ts` |
| AC9 | Three labs authored via the SDK ship publicly in craftingattention and run end-to-end | manual sign-off |
| AC10 | 10 000-payload telemetry fuzz finds zero forbidden field shapes | `telemetry-fuzz.test.ts` |
| AC11 | Security review pass: zero PII fields, zero student-derived data in telemetry | manual review + automated grep |
| AC12 | `npx bg-lab validate` runs in ≤5s over 50 manifests on CI | CI timing |

---

## Test Strategy

### Unit tests (`packages/browsergrad-runtime/tests/`)

- `manifest.test.ts` — schema validation + JSON Pointer error paths.
- `versioning.test.ts` — semver range satisfaction + legacy URL resolution.
- `policy.test.ts` — fetch allowlist parsing.
- `coverage-matrix.test.ts` — JSON consistency (no dangling PRD references, no duplicate ids).

### Integration tests (`packages/browsergrad-runtime/tests-integration/`)

- `lab-harness.test.ts` — each grading primitive against real Pyodide-in-node.
- `dataset-streaming.test.ts` — cold + warm + corrupted-sha256 paths.
- `sandbox-policy.test.ts` — the 12 escape attempts.
- `telemetry-consent.test.ts` — emitter dormant until consent.
- `telemetry-fuzz.test.ts` — 10 000-payload fuzz.
- `versioning-legacy.test.ts` — modern host + pinned-old manifest = legacy bundle boots.
- `cli-validate.test.ts` — CLI exit codes.

### End-to-end fixtures (`packages/bg-lab-cli/fixtures/`)

Three authored labs (fast.ai Ch 11 matmul, deep-ml #25, CS336 A1.5 attention) with full manifest + starter + reference + rubric. Every CI run boots the runtime, loads each, runs the reference, asserts the rubric grades to "all pass."

### Cross-PRD regression

- Re-run PRD-004's `lab-e2e-single-neuron.test.ts` through the new harness; identical assertion stream required.
- Re-run PRD-008's safetensors-streaming tests with `bg.lab.dataset()` as the consumer.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | JSON Schema ossifies before real authors stress-test it | High | Medium | `additionalProperties:false` rejects unknown fields loudly; ship a `schema_version` field so v2 is additive |
| R2 | Legacy-runtime CDN cost grows unbounded | Medium | Low | ~3 MB/bundle × 50 versions = 150 MB; trivial at R2 prices; documented in maintainer ADR |
| R3 | Author-controlled `failed_assertion_name` smuggles PII into telemetry | Low | High | Linter rejects names containing `$`, `{`, `<`, or matching email/phone/UUID regex; hard CI gate |
| R4 | Pyodide `js` module bypass of `bg.fetch` allowlist | Medium | High | Host-side message router refuses any proxied `js.fetch`; `js` module wrapped at preamble; external pentest before v1.0.0 |
| R5 | Asset CDN outage during a live class | Low | High | Dual CDN (R2 + jsdelivr); OPFS cache means returning students unaffected |
| R6 | Rubric exceeds Pyodide's 60s gas limit | Medium | Medium | Schema clamps `expected_wall_clock_ms` ≤30 s; CI runs reference within budget |
| R7 | Semver bump silently changes rubric behavior | High | High | Nightly CI runs each legacy bundle against its own fixtures; drift fails the release; rubrics carry `last_validated_at` |
| R8 | `required_prds` matrix drifts from reality | Medium | Medium | `bg-lab coverage-matrix --verify` cross-checks every entry against the PRD index in CI |
| R9 | Telemetry consent UI dark-patterned downstream | Low | High | Spec requires explicit `setTelemetryConsent(true)` JS call; consent UI ships open-source in craftingattention and is reviewable; runtime refuses telemetry without the call regardless of manifest |
| R10 | Fast.ai Ch 25 latent diffusion needs an op no PRD covers | Medium | Medium | Matrix has a `gap:true` flag on incomplete rows; gaps inform P3 PRD list |

---

## Open Questions

1. **Server-side grading escape hatch.** Some CS336 A3 scaling exercises ask for training-loss-curve predictions on models too large for any browser. Current proposal: ship a precomputed loss-curve `.safetensors` and assert match. Alternative: a thin server grader for these specific problems. Resolution deferred to CS336 authoring.

2. **Authoring UI open-source or private.** This PRD assumes craftingattention's authoring UI is private. A future open release would let other platforms onboard authors. Resolution: keep private through P2; revisit at 50+ labs.

3. **Curriculum-specific rubric idioms.** Fast.ai may want `assert_within_factor(0.9, 1.1)` while CS336 prefers strict `assert_pytorch_match`. The harness ships the minimum; curriculum-specific helpers can layer as Python packages without runtime changes.

4. **i18n of error messages.** Assertion strings are English. Hooks (`bg.set_error_locale("es")`) deferred to PRD-018+; gap documented.

5. **Telemetry analytics consumer.** PRD specifies the emitter, not the consumer. craftingattention will build a Postgres + dashboard. Whether browsergrad ships any default consumer is undecided.

6. **Legacy-runtime sunset policy.** Bundles are cheap to store but the catalogue grows indefinitely. Sunset after 5 years? Resolution: defer until catalogue > 1000 labs.

---

## References

1. **fast.ai Part 2 (Practical Deep Learning Part 2)** — [course.fast.ai/Lessons/part2.html](https://course.fast.ai/Lessons/part2.html). 17 chapters spanning matmul-from-scratch (Ch 11) through latent diffusion (Ch 25). Authoritative public syllabus this PRD covers.

2. **Stanford CS336: Language Modeling from Scratch** — [stanford-cs336.github.io/spring2024/](https://stanford-cs336.github.io/spring2024/). Five assignments (A1 Basics, A2 Systems, A3 Scaling, A4 Data, A5 Alignment) with public sub-problems.

3. **Open-Deep-ML problem catalog** — [github.com/Open-Deep-ML/DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem). ~250 problems across linear algebra, classical ML, deep learning (CNN/RNN/LSTM/transformer), optimization, normalization, LLM.

4. **PRD-004 — Lab Runtime API** — `docs/prd/PRD-004-lab-runtime-api.md`. Assertion + artifact wire protocol PRD-013 extends.

5. **PRD-005 — JIT Foundation** — `docs/prd/PRD-005-jit-foundation.md`. IR + tracer the harness depends on for trace identity.

6. **PRD-008 — OPFS Pipeline Cache + Safetensors Streaming** — `docs/prd/PRD-008-persistent-caching.md`. Underlying transport for the dataset pipeline.

7. **PRD-009 — Gradient Checkpointing** — `docs/prd/PRD-009-gradient-checkpointing.md`. Load-bearing for fast.ai Ch 18 + transformer labs past `(B=8, seq=512)`.

8. **JSON Schema Draft 2020-12** — [json-schema.org/draft/2020-12](https://json-schema.org/draft/2020-12). Wire format for the manifest schema.

9. **Quarto `_quarto.yml`** — [quarto.org/docs/projects/quarto-projects.html](https://quarto.org/docs/projects/quarto-projects.html). Reference for machine-readable curriculum metadata.

10. **safetensors format** — [github.com/huggingface/safetensors](https://github.com/huggingface/safetensors). Asset blob format; same as PRD-008.

11. **W3C COOP / COEP** — [developer.mozilla.org/en-US/docs/Web/HTTP/Cross-Origin_Opener_Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/Cross-Origin_Opener_Policy). Headers the host must serve for SAB-dependent features.

12. **Pyodide security model** — [pyodide.org/en/stable/usage/api/js-api.html](https://pyodide.org/en/stable/usage/api/js-api.html). Trust boundary `bg.fetch` enforces.

13. **Cloudflare R2 pricing** — [developers.cloudflare.com/r2/pricing](https://developers.cloudflare.com/r2/pricing). Cost basis for the legacy-runtime CDN.

14. **GDPR Article 7 (consent)** — [gdpr-info.eu/art-7-gdpr](https://gdpr-info.eu/art-7-gdpr/). Legal substrate the telemetry consent flow respects.

15. **PyTorch `torch.testing.assert_close`** — [pytorch.org/docs/stable/testing.html](https://pytorch.org/docs/stable/testing.html). Tolerance defaults `assert_pytorch_match` mirrors.

16. **Karpathy nanoGPT** — [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). Reference model used in several CS336 sub-problems.
