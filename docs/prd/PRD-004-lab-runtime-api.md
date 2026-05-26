# PRD-004 — Lab Runtime API + End-to-End Validation

**Status**: Draft v1, May 2026
**Owner**: browsergrad maintainers
**Parent PRD**: [PRD.md](../../PRD.md) §7 P0.4 + P0.5
**Target milestone**: P0 (months 1–3)
**Estimated effort**: 3 weeks total

---

## TL;DR

The runtime and Python library already support structured assertion and artifact emission, but no ergonomic helpers exist for the patterns a lab author actually writes — showing a tensor as an image, plotting a loss curve, saving and resuming a checkpoint, or asserting numerical proximity. This PRD specifies a new `browsergrad_grad.lab` Python module containing six helpers that wrap the existing artifact-emit protocol, adds two minor runtime extensions (OPFS-backed checkpoint read/write and an `onProgress` streaming hint in `ExecOptions`), then validates the whole stack by running deep-ml problem #25 (single-neuron backpropagation) end-to-end as a regression fixture. Completion converts browsergrad from "a library that works" to "a library that is ready to power a lab platform."

---

## Background: minimum-viable lab UX

Education research and competitive analysis identify six features that separate a usable in-browser coding lab from a bare Python REPL:

1. **Code editor + run.** Non-negotiable. Every competitor has it.
2. **stdout/stderr streaming.** Students need to see print statements as they arrive, not after the whole exec completes.
3. **Structured assertion runner.** The host page needs pass/fail/error verdicts — not parsed from stdout — to show a test-results panel that the grader can read programmatically.
4. **At least one visualization.** "See the training curve" is the canonical feedback loop. deep-ml, fast.ai, Colab, and Observable all include it.
5. **Persistent state across runs.** Students want to save a checkpoint and resume; they also want the namespace to persist across cells so they can iterate without re-running everything.
6. **Numerical proximity assertions.** `assert_close(a, b, atol=1e-4)` is the bread-and-butter of ML verification; a raw `assert (a - b).abs().max() < tol` is acceptable but verbose.

Evidence: deep-ml ships all six. Colab ships 1–5. Observable ships 1–4. fast.ai's notebooks ship all six. The only platform without persistent state is an isolated-cell runner — deep-ml problem mode fits that pattern. Lab mode (craftingattention) needs all six.

Browsergrad's runtime already covers items 1, 2, and 3 at the infrastructure level: `ExecOptions.onStdout`, `ExecOptions.onStderr`, `ExecOptions.onAssertion`, and `ExecOptions.onArtifact` are all live in `packages/browsergrad-runtime/src/types.ts:133–168`. Items 4, 5, and 6 require Python-side helpers and, for item 5, an OPFS adapter.

---

## User stories

**Story A — Lab author emitting a loss curve.** Maya is writing a craftingattention lesson on gradient descent. She wants students to watch the loss decrease over 50 epochs. She writes:

```python
import browsergrad as bg
losses = []
for epoch in range(50):
    loss = train_step(model, optimizer, X, y)
    losses.append(float(loss))
bg.plot_loss("training_loss", losses)
```

The host page receives an `ArtifactJson` event with `name="training_loss"` and `data={"steps": [0..49], "loss": [...]}`, renders it with its chart library of choice, and the student sees the curve.

**Story B — Student saving and resuming a checkpoint.** A student runs 30 epochs of a small CNN, wants to close the tab and resume tomorrow. They add:

```python
bg.checkpoint("my_cnn_v1", model, optimizer)
```

The runtime serializes the `state_dict` plus optimizer state to OPFS under the key `browsergrad/checkpoints/{origin_hash}/my_cnn_v1`. Next session:

```python
bg.load_checkpoint("my_cnn_v1", model, optimizer)
```

restores both. The student continues from epoch 31.

**Story C — Automated grader verifying a lab answer.** A craftingattention lab asks students to implement a single-neuron forward/backward pass. The lab's test block runs:

```python
import browsergrad as bg
import numpy as np

actual_w = np.array(w_grad)
expected_w = np.array([-0.2, 0.4])
bg.assert_close("weight_grad", actual_w, expected_w, atol=1e-4)
bg.assert_close("bias_grad", float(b_grad), -0.1, atol=1e-4)
```

The host page receives `AssertionPass` or `AssertionFail` events and renders a test-results panel with checkmarks or diffs.

---

## Goals + non-goals

**Goals**

- Ship six Python helpers (`show_image`, `plot_loss`, `checkpoint`, `load_checkpoint`, `assert_close`, and the `onProgress` streaming pattern) that a lab author can use without touching the protocol directly.
- Validate the entire stack end-to-end by running deep-ml problem #25 (single-neuron with backpropagation) through the runtime.
- Add a regression test fixture that locks in the end-to-end result so future changes can't silently break the lab path.
- Document the lab API in the README with one runnable example per function.
- Confirm that `clearNamespace()` fully resets lab state with no variable leakage.

**Non-goals**

- Plotting library — the host page chooses its own renderer. We emit the data; we don't render.
- The OPFS adapter for the P0.3 cold-start cache — that's a separate feature. The checkpoint API shares the OPFS write path but doesn't implement the service-worker caching.
- Server-side checkpoint sync — OPFS is local-only. Cross-device is a later problem.
- Notebook-level abstractions (`bg.lab.notebook()`, cell history, output diffing) — deferred to the open question section.
- WebGPU acceleration for the helpers themselves — they emit bookkeeping data; performance is not the concern.

---

## Architecture

### Two-layer design

The feature lives at two layers that are independently testable:

**Layer 1 — Python helpers** (`browsergrad_grad.lab` module, new file)

```
packages/browsergrad-grad/src/python/lab.py
```

A pure-Python module that:
- Imports `browsergrad` (the preamble-injected module already present in the runtime worker's namespace) for its `emit_image`, `emit_json`, and `assert_*` primitives.
- Provides the six user-facing functions documented in the API surface section below.
- Has no dependencies beyond `numpy` (already loaded in every runtime session) and `base64` / `io` (stdlib).
- Is registered on `sys.modules["browsergrad_grad.lab"]` and re-exported from `browsergrad_grad.__init__` as `bg_lab` and also via `import browsergrad as bg; bg.show_image(...)` by extending the preamble.

The six helpers are thin wrappers: they do data preparation (encode tensor to PNG, build the loss JSON, serialize state_dict to bytes) then call the underlying `emit_*` or `assert_*` primitives. They own none of the transport logic.

**Layer 2 — Runtime extensions** (`browsergrad-runtime`)

Two minor changes:

1. **`ExecOptions.onProgress` callback** (optional convenience alias): a new optional field in `ExecOptions` that is called by the host page whenever an `ArtifactJson` with name matching `"*_loss"` or `"*_progress"` arrives. This is purely a host-page convenience — it does not change the wire protocol. The Python side already calls `emit_json`; `onProgress` is just a named filter that the host page can bind instead of parsing inside `onArtifact`. Concretely: `onProgress?: (artifact: ArtifactJson) => void` is added to `ExecOptions` in `types.ts`, and `client.ts` routes `ArtifactJson` artifacts through it when provided.

2. **Checkpoint wire protocol**: `bg.checkpoint` and `bg.load_checkpoint` cannot write to OPFS from inside the Pyodide worker directly — OPFS `FileSystemSyncAccessHandle` is available from Worker threads, but the Pyodide worker needs explicit protocol messages to expose it. Two new protocol messages are added: `CheckpointSaveRequest` / `CheckpointSaveResponse` and `CheckpointLoadRequest` / `CheckpointLoadResponse`. Python triggers these via two new `_bg_native` functions: `saveCheckpoint(name, dataBase64)` and `loadCheckpoint(name) -> dataBase64`. The runtime worker calls through to an OPFS-backed handler on the main thread (or, in node tests, a MemFS mock).

### Data flow: `bg.show_image(name, tensor)`

```
Python: bg.show_image("filter_0", weight_tensor)
  |
  v
lab.py: _tensor_to_png_base64(weight_tensor)
  - numpy: normalize to [0, 255] uint8
  - try: from PIL import Image (Pyodide ships Pillow via loadPackage)
  -   if PIL: Image.fromarray(arr).save(buf, format="PNG") -> buf.getvalue()
  -   fallback: encode as raw single-channel PPM in base64
  - base64.b64encode(png_bytes).decode("ascii")
  |
  v
bg.emit_image("filter_0", "image/png", b64_string)
  |
  v
_bg_native.postArtifact(json_string)
  |
  v
worker/index.ts: reply({ kind: "exec:artifact", artifact: ArtifactImage })
  |
  v
client.ts: slot.artifacts.push(artifact); slot.onArtifact?.(artifact)
  |
  v
Host page: onArtifact callback -> render <img src="data:image/png;base64,..."> 
```

### Data flow: `bg.plot_loss(name, values)`

```
Python: bg.plot_loss("loss_curve", [2.1, 1.8, 1.4, ...])
  |
  v
lab.py: build dict {"steps": list(range(len(values))), "loss": [float(v) for v in values]}
  |
  v
bg.emit_json("loss_curve", dict)
  |
  v
ArtifactJson arrives at host page via onArtifact / onProgress
Host page chooses: Chart.js, Plotly, Recharts, D3 — all can consume {steps, loss} arrays
```

### Data flow: `bg.checkpoint(name, model, optimizer=None)`

```
Python: bg.checkpoint("epoch_30", model, optimizer)
  |
  v
lab.py:
  - model.state_dict() -> dict[str, Tensor]
  - Convert each Tensor: t.data.astype(np.float32).tobytes()
  - Build payload JSON: {"version":1, "model": {k: {"shape":..., "data_b64":...}}, "optimizer": ...}
  - base64-encode the whole JSON string
  |
  v
_bg_native.saveCheckpoint(name, payload_b64)
  |
  v
worker/index.ts: handleCheckpointSave() posts CheckpointSaveRequest
  |
  v
client.ts: receives checkpoint:save, writes to OPFS:
  path = `browsergrad/checkpoints/{sha256(origin)[0:16]}/{name}.json`
  navigator.storage.getDirectory() -> getDirectoryHandle -> getFileHandle -> createWritable
  |
  v
worker receives checkpoint:save:done, Python awaits the sync response
```

### Checkpoint storage layout in OPFS

```
browsergrad/
  checkpoints/
    {origin_hash_prefix}/     # first 16 hex chars of sha256(window.location.origin)
      {name}.json             # checkpoint payload (base64-encoded JSON)
```

The `origin_hash_prefix` prevents a lesson on `lesson-01.craftingattention.com` from overwriting a checkpoint from `lesson-02.craftingattention.com` — both share the same origin but use different subdirectory prefixes. In node tests, the OPFS adapter is replaced with an in-memory `Map<string, string>` keyed by `{name}`.

Payload format (version 1):

```json
{
  "version": 1,
  "browsergrad_version": "0.5.0",
  "created_at_ms": 1748304000000,
  "model": {
    "fc1.weight": { "shape": [128, 64], "dtype": "float32", "data_b64": "..." },
    "fc1.bias":   { "shape": [128],     "dtype": "float32", "data_b64": "..." }
  },
  "optimizer": {
    "kind": "Adam",
    "state": { ... }
  }
}
```

`browsergrad_version` allows `load_checkpoint` to warn (not error) on version mismatch. Tensors are the only binary blobs, base64-encoded inline. JSON-readable for easy debugging.

### Wire protocol extensions for checkpoints

Two new message pairs added to `protocol.ts`:

```typescript
// Client → Worker
export interface CheckpointSaveRequest {
  readonly id: number;
  readonly kind: "checkpoint.save";
  readonly name: string;
  readonly payloadBase64: string;
}

export interface CheckpointLoadRequest {
  readonly id: number;
  readonly kind: "checkpoint.load";
  readonly name: string;
}

// Worker → Client
export interface CheckpointSaveResponse {
  readonly id: number;
  readonly kind: "checkpoint.save:done";
}

export interface CheckpointLoadResponse {
  readonly id: number;
  readonly kind: "checkpoint.load:done";
  readonly payloadBase64: string | null;  // null if not found
}
```

Same request/response pattern as `fs.write` / `fs.read` already in `protocol.ts:42-64`.

### End-to-end validation harness

The target lab is **deep-ml problem #25 (single-neuron-with-backpropagation)**. Why this problem:

- Exercises sigmoid activation, a single `nn.Linear(2, 1)` equivalent, cross-entropy loss, and one backward pass — the same ops covered by existing PyTorch-conformance fixtures.
- Produces deterministic expected outputs for given seed inputs.
- Small enough to run in under 1 second in Pyodide-in-node.
- Uses all four lab helpers: `assert_close` for gradient verification, `plot_loss` for the loss curve, optional `show_image` for weight viz.

Problem statement: given inputs `X = [[1, 2], [2, 3], [3, 4], [4, 5]]`, labels `y = [0, 0, 1, 1]`, initial weights `w = [0.1, -0.2]`, `b = 0.0`, run one forward pass (sigmoid output), compute BCE loss, run one backward pass, return the updated weights and bias after one SGD step with `lr=0.1`.

The regression fixture lives at:

```
packages/browsergrad-runtime/tests-integration/lab-e2e-single-neuron.test.ts
```

It boots Pyodide-in-node, installs the preamble, installs `browsergrad_grad`, runs the single-neuron lab code, and asserts:
- `ExecResult.ok === true`
- One `AssertionPass` per `assert_close` call
- One `ArtifactJson` with `name === "training_loss"` and non-empty `.data.loss`
- `clearNamespace()` followed by re-run produces identical results (no state pollution)

---

## API surface

All functions live in `browsergrad_grad/lab.py`. They are also registered on the `browsergrad` module injected by the preamble, so both `import browsergrad as bg; bg.show_image(...)` and `from browsergrad_grad import lab; lab.show_image(...)` work.

### `bg.show_image(name, tensor, *, normalize=True)`

Encode a 2-D or 3-D tensor (H×W or C×H×W) as a PNG and emit it as an `ArtifactImage`.

```python
def show_image(
    name: str,
    tensor,                 # Tensor | np.ndarray, shape (H, W) or (C, H, W) or (H, W, C)
    *,
    normalize: bool = True, # rescale to [0, 255]; set False if already uint8
) -> None:
```

Example:

```python
import browsergrad as bg
conv_weights = model.conv1.weight  # shape (8, 1, 3, 3)
for i in range(8):
    bg.show_image(f"conv1_filter_{i}", conv_weights[i, 0])
```

### `bg.plot_loss(name, values, *, x_label="step", y_label="loss")`

Emit a JSON artifact containing a step/loss series.

```python
def plot_loss(
    name: str,
    values: list | np.ndarray,
    *,
    x_label: str = "step",
    y_label: str = "loss",
) -> None:
```

Emits `ArtifactJson` with `data = {"steps": [0, 1, ..., N-1], "loss": [...], "x_label": ..., "y_label": ...}`.

### `bg.checkpoint(name, model, optimizer=None)`

Serialize model (and optionally optimizer) state to OPFS.

```python
def checkpoint(
    name: str,
    model,
    optimizer=None,
) -> None:
```

Raises `RuntimeError` if OPFS is unavailable (node test environment with no mock registered).

### `bg.load_checkpoint(name, model, optimizer=None)`

Restore model (and optionally optimizer) state from OPFS.

```python
def load_checkpoint(
    name: str,
    model,
    optimizer=None,
) -> bool:              # True if found, False if not found
```

Returns `False` rather than raising when the checkpoint does not exist:

```python
if not bg.load_checkpoint("my_run", model, optimizer):
    print("Starting fresh — no checkpoint found")
```

### `bg.assert_close(name, actual, expected, *, atol=1e-4, rtol=0.0)`

Emit a pass/fail assertion verifying numerical closeness.

```python
def assert_close(
    name: str,
    actual,
    expected,
    *,
    atol: float = 1e-4,
    rtol: float = 0.0,
) -> None:
```

Passes if `max(|actual - expected|) <= atol + rtol * max(|expected|)`. On failure, emits `AssertionFail` with `expectedRepr` showing the max diff and `atol` threshold. Exceptions during comparison emit `AssertionError`.

### `session.exec({ onProgress })` — TypeScript side

`onProgress` is a new optional field on `ExecOptions`. It is a convenience filter over `onArtifact` that fires only for `ArtifactJson` artifacts:

```typescript
const result = await session.exec({
  code: trainingCode,
  onProgress: (artifact) => {
    // artifact.kind === "json" guaranteed
    if (artifact.name === "train_loss") renderLossCurve(artifact.data);
  },
});
```

Purely additive change to `ExecOptions` in `types.ts`. No protocol change.

---

## Implementation plan

### Week 1: Python helpers + preamble extension

**Days 1–2: `browsergrad_grad/lab.py` scaffold**

- Create `/packages/browsergrad-grad/src/python/lab.py`
- Implement `plot_loss` and `assert_close` (no dependencies beyond `numpy` and the existing `browsergrad` preamble module)
- Write unit tests in `packages/browsergrad-runtime/tests-integration/python-bridge.test.ts`: add four test cases asserting the artifact shapes emitted by each new helper
- Confirm `clearNamespace()` isolation: run helper, clear, run again, assert no state cross-contamination

**Days 3–4: `show_image` with PIL / fallback path**

- Implement `_tensor_to_png_base64` with PIL fast path (Pyodide ships `Pillow`) and a pure-Python raw PPM fallback
- Test verifies `ArtifactImage` payload: `kind === "image"`, `mime === "image/png"`, `dataBase64` decodes to bytes starting with PNG magic (`\x89PNG`)

**Day 5: Preamble extension**

- Extend `python-preamble.ts` to attach `show_image`, `plot_loss`, `assert_close` to the `browsergrad` module
- Test that `import browsergrad as bg; bg.show_image(...)` works alongside `from browsergrad_grad import lab; lab.show_image(...)`

### Week 2: Checkpoint API + TypeScript runtime extension

**Days 1–2: Protocol + client extension**

- Add `CheckpointSaveRequest`, `CheckpointLoadRequest`, `CheckpointSaveResponse`, `CheckpointLoadResponse` to `protocol.ts`
- Extend `ClientToWorker` and `WorkerToClient` union types
- Add `onProgress?: (artifact: ArtifactJson) => void` to `ExecOptions` in `types.ts:133`
- Route `ArtifactJson` through `onProgress` in `client.ts` routing switch
- Add `PendingRequest.onProgress` field alongside the existing `onArtifact`

**Days 3–4: OPFS adapter + worker-side handler**

- Add `handleCheckpointSave` and `handleCheckpointLoad` to `worker/index.ts`
- The main-thread side (a new `src/opfs.ts` module) implements OPFS read/write using `navigator.storage.getDirectory()` with the namespaced path scheme
- Provide a node-test mock: `createOpfsMock()` returns a `Map<string, string>`-backed adapter

**Day 5: `bg.checkpoint` + `bg.load_checkpoint` Python side**

- Implement `checkpoint()` and `load_checkpoint()` in `lab.py` using the new `_bg_native.saveCheckpoint` / `_bg_native.loadCheckpoint` hooks
- Add tests in `client-routing.test.ts` extending the `FakeWorker` pattern to verify round-trip identity of the serialized payload

### Week 3: End-to-end validation + regression fixture

**Days 1–2: Single-neuron lab fixture**

- Create `packages/browsergrad-runtime/tests-integration/lab-e2e-single-neuron.test.ts`
- Write the single-neuron lab Python code (sigmoid forward, BCE loss, backward, SGD step) using `bg.assert_close` for all expected values
- Run against Pyodide-in-node; fix every failure
- Assert artifact shapes: at least one `AssertionPass` per `assert_close` call, one `ArtifactJson` from `plot_loss`

**Days 3–4: Issue fixing + clearNamespace validation**

- Run `clearNamespace()` then re-execute the lab; assert identical results
- Run the lab twice in sequence without clearing; assert second run does not inherit first run's variables
- If any op in the lab fails, fix in `browsergrad_grad` and add a regression test

**Day 5: Documentation + README update**

- Update `packages/browsergrad-grad/README.md` with a "Lab API" section showing all six helpers with one runnable example each
- Update `packages/browsergrad-runtime/README.md` with `onProgress` documentation
- Add a `CHANGELOG` entry for PRD-004

---

## Acceptance criteria

**AC-1: Python helpers emit correct artifact shapes.**
For each of `show_image`, `plot_loss`, `checkpoint`, `load_checkpoint`, `assert_close`: at least one test in `tests-integration/` runs the helper in real Pyodide-in-node and asserts the emitted artifact's `kind`, `name`, and `data` shape match the specification.

**AC-2: Single-neuron lab runs end-to-end.**
`lab-e2e-single-neuron.test.ts` passes with `ExecResult.ok === true`. Every `bg.assert_close` call emits `AssertionPass`. The test runs in under 60 seconds.

**AC-3: Re-run after `clearNamespace()` is identical.**
The single-neuron lab is run twice in the same test, separated by `clearNamespace()`. Both runs emit identical assertions and identical artifact `data` values.

**AC-4: `onProgress` routes JSON artifacts.**
A unit test in `client-routing.test.ts` using `FakeWorker` verifies that providing `onProgress` receives `ArtifactJson` artifacts, and `ArtifactLog` / `ArtifactImage` do not route to `onProgress`.

**AC-5: Checkpoint round-trip.**
A test writes a checkpoint via `bg.checkpoint("test", model)`, calls `clearNamespace()`, re-instantiates `model`, calls `bg.load_checkpoint("test", model)`, and verifies that `model.state_dict()` values match the pre-checkpoint values within `1e-6`.

**AC-6: `show_image` emits valid PNG.**
The base64 payload in the `ArtifactImage` artifact, when decoded, starts with the PNG magic bytes `\x89PNG`.

---

## Test strategy

**Existing infrastructure reuse:** all new tests follow the same pattern as `python-bridge.test.ts` (real Pyodide-in-node) and `client-routing.test.ts` (FakeWorker for pure TypeScript routing tests). No new test harness infrastructure needed.

**New test files:**

- `packages/browsergrad-runtime/tests-integration/lab-e2e-single-neuron.test.ts` — the full lab end-to-end fixture (highest-value test).
- Python helper shapes added as new `describe` blocks in `python-bridge.test.ts`.
- Checkpoint protocol messages added as new `describe` blocks in `client-routing.test.ts`.

**Test coverage checklist:**
- [ ] `show_image` emits `ArtifactImage` with correct mime and valid base64
- [ ] `plot_loss` emits `ArtifactJson` with `steps` and `loss` arrays of matching length
- [ ] `assert_close` emits `AssertionPass` when values match within `atol`
- [ ] `assert_close` emits `AssertionFail` with diff info when values exceed `atol`
- [ ] `assert_close` emits `AssertionError` on shape mismatch
- [ ] `checkpoint` round-trip preserves all parameter values
- [ ] `load_checkpoint` returns `False` when named checkpoint does not exist
- [ ] `onProgress` routes only `ArtifactJson`; does not fire for `ArtifactLog` or `ArtifactImage`
- [ ] Single-neuron lab passes end-to-end with all assertions passing
- [ ] `clearNamespace()` isolation: second lab run identical to first

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Image encoding in pure Python is slow for large tensors (>1 MB) | Medium | Low for lab use cases | Use PIL when Pyodide's `Pillow` package is loaded; fall back to downsampled PPM. For tensors > 512×512, auto-downsample before encoding and emit a warning via `bg.log`. |
| R2 | Checkpoint API causes students to accidentally overwrite good training progress | Medium | High emotionally (losing hours of work) | `bg.checkpoint` never silently overwrites by default — raises `RuntimeError("checkpoint '{name}' already exists; pass overwrite=True to replace")`. |
| R3 | OPFS `FileSystemSyncAccessHandle` unavailable in some cross-origin-isolated environments | Low | Medium | Python functions check `_bg_native.opfsAvailable()` at call time and emit a `bg.log(..., level="warn")` explaining rather than raising silently. |
| R4 | The single-neuron lab exposes a missing op or gradient bug | Medium | Low (it's a P0 validation — finding bugs is the goal) | Fix the bug, add a regression test, document in CHANGELOG. |
| R5 | `assert_close` tolerance defaults disagree with PyTorch's `torch.testing.assert_close` | Low | Low | Use the same defaults as `torch.testing.assert_close`: `atol=1e-4`, `rtol=0`. |
| R6 | PIL not available in Pyodide without explicit `loadPackage(["Pillow"])` | High | Low | Docstring and README note: "For PNG encoding, include `'Pillow'` in `packages` when creating the session. Falls back to PPM if Pillow is unavailable." |

---

## Open questions

**OQ-1: Should `bg.lab.notebook()` exist as a higher-level wrapper?**

A `notebook()` context manager that captures a whole training loop and automatically emits intermediate `plot_loss` and `show_image` artifacts on a configurable frequency would be ergonomic for simple labs:

```python
with bg.lab.notebook(emit_every=10) as nb:
    for epoch in range(100):
        loss = train_step(model, opt, X, y)
        nb.record_loss(loss)
        if epoch % 20 == 0:
            nb.show_weights(model.conv1.weight[0, 0], name=f"filter_epoch_{epoch}")
```

More opinionated; risks fighting against the lab author's preferred structure. Deferred until craftingattention builds its first three labs and we see what patterns emerge.

**OQ-2: Per-step streaming vs final-values one-shot.**

`plot_loss(name, values)` is one-shot. The alternative is per-step streaming: call `bg.emit_progress(name, step, value)` inside the loop and have the host page update a live chart.

Per-step is more interactive (matches Colab/Jupyter cell-output streaming) and updates even on interrupt. Downside is API complexity: host page needs to accumulate partial series.

Decision: ship one-shot `plot_loss` first. Add `bg.emit_step(name, step, value)` as a subsequent additive API once the host page renderer can handle incremental updates.

---

## References

1. **Artifact protocol types** — `packages/browsergrad-runtime/src/types.ts:264–287`
2. **Python preamble** — `packages/browsergrad-runtime/src/worker/python-preamble.ts`
3. **Worker protocol** — `packages/browsergrad-runtime/src/worker/index.ts:89–108`
4. **Client routing** — `packages/browsergrad-runtime/src/client.ts:295–318`
5. **Existing bridge tests** — `packages/browsergrad-runtime/tests-integration/python-bridge.test.ts`
6. **deep-ml problem #25** — [Open-Deep-ML/DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem): "Single Neuron with Backpropagation"
7. **OPFS spec** — [MDN: Origin Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
8. **Pyodide Pillow availability** — [Pyodide package list](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)
9. **ExecOptions interface** — `packages/browsergrad-runtime/src/types.ts:133–168`
