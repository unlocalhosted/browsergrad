/**
 * WebGPU realizer seam tests (PRD-011.5 spike).
 *
 * The real bridge dispatches WGSL kernels via Pyodide → JS → WebGPU.
 * That path needs a real GPUDevice and lives in browser CI. These
 * tests instead instantiate a Python-side NumPy-backed mock bridge
 * that satisfies the same surface (see `_bridge.py` Protocol) and
 * verify:
 *
 *   - The seam dispatches correctly (every supported UOp routes
 *     to the bridge method and the returned handle threads through).
 *   - Residency contract: chained matmuls trigger N uploads + 1
 *     materialise, with intermediate handles released the moment
 *     their last consumer finishes.
 *   - End-to-end numerical match between bg.realize_webgpu and
 *     bg.tensor.numpy() (which uses the NumPy realizer).
 *   - Refusal modes: unsupported opcode raises with a pointer to
 *     bg.realize() as the fallback.
 *   - Flash Attention opt-in CUSTOM op routes through bridge.flash_attention.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

const MOCK_BRIDGE_PY = `
# A NumPy-backed mock bridge that satisfies the WebGpuBridge protocol.
# Used for testing the seam in pyodide-in-node where no GPUDevice exists.
#
# Records every call so tests can assert on the residency contract:
#   - upload_count / materialize_count / release_count
#   - alive set: handles minted minus handles released
import numpy as np

class MockBridge:
    def __init__(self):
        self._handles = {}        # handle_id -> ndarray
        self._next = 0
        self.upload_count = 0
        self.materialize_count = 0
        self.release_count = 0
        self.matmul_count = 0
        self.fused_count = 0
        self.flash_count = 0
        self.cast_count = 0
        self.calls = []           # ordered list of (op, handle_id_in_or_out)

    def _mint(self, arr):
        hid = self._next
        self._next += 1
        self._handles[hid] = np.array(arr, copy=True)
        return hid

    def alive(self):
        return set(self._handles.keys())

    # ---- protocol ----
    def upload(self, data, shape, dtype):
        self.upload_count += 1
        arr = np.frombuffer(data, dtype=np.dtype(dtype))
        if shape and shape != (1,):
            arr = arr.reshape(shape)
        h = self._mint(arr)
        self.calls.append(("upload", h))
        return h

    def materialize(self, handle, shape, dtype):
        self.materialize_count += 1
        arr = self._handles[handle]
        self.calls.append(("materialize", handle))
        return arr.astype(np.dtype(dtype), copy=False).tobytes()

    def release(self, handle):
        self.release_count += 1
        self.calls.append(("release", handle))
        self._handles.pop(handle, None)

    def matmul(self, a, b, m, k, n, dtype):
        self.matmul_count += 1
        out = self._handles[a] @ self._handles[b]
        h = self._mint(out.astype(np.dtype(dtype), copy=False))
        self.calls.append(("matmul", h))
        return h

    def fused_elementwise(self, inputs, ops, shape, dtype):
        self.fused_count += 1
        # ops is a list of (opcode, lhs_ref, rhs_ref) — same shape as the
        # NumPy realizer's _h_fused_elementwise.
        externals = [self._handles[h] for h in inputs]
        steps = []
        def resolve(ref):
            return externals[-ref - 1] if ref < 0 else steps[ref]
        for opcode, lhs_ref, rhs_ref in ops:
            a = resolve(lhs_ref)
            if opcode == "ADD":
                steps.append(a + resolve(rhs_ref))
            elif opcode == "MUL":
                steps.append(a * resolve(rhs_ref))
            elif opcode == "DIV":
                steps.append(a / resolve(rhs_ref))
            elif opcode == "NEG":
                steps.append(-a)
            elif opcode == "EXP":
                steps.append(np.exp(a))
            elif opcode == "LOG":
                steps.append(np.log(a))
            else:
                raise ValueError(f"mock fused: unknown op {opcode}")
        h = self._mint(steps[-1].astype(np.dtype(dtype), copy=False))
        self.calls.append(("fused_elementwise", h))
        return h

    def cast(self, handle, src_dtype, dst_dtype, shape):
        self.cast_count += 1
        arr = self._handles[handle].astype(np.dtype(dst_dtype), copy=False)
        h = self._mint(arr)
        self.calls.append(("cast", h))
        return h

    def flash_attention(self, q, k, v, mask, b, h_, sq, sk, d, scale, dtype):
        self.flash_count += 1
        Q = self._handles[q]
        K = self._handles[k]
        V = self._handles[v]
        # Composed reference: scores = Q @ K^T * scale; (+ mask); softmax; @ V.
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale
        if mask is not None:
            scores = scores + self._handles[mask]
        # Stable softmax along the last axis.
        m_ = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m_)
        p = e / e.sum(axis=-1, keepdims=True)
        out = np.matmul(p, V).astype(np.dtype(dtype), copy=False)
        hh = self._mint(out)
        self.calls.append(("flash_attention", hh))
        return hh
`;

describe("PRD-011.5 WebGPU realizer seam", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    // Re-register a fresh mock bridge per test so counters start at zero.
    await target.run(`
import browsergrad_jit as bg
${MOCK_BRIDGE_PY}
_mock = MockBridge()
bg.register_webgpu_bridge(_mock)
`);
  });

  it("realizes a single matmul through the bridge and matches NumPy", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      upload: number;
      matmul: number;
      materialize: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
rng = np.random.RandomState(0)
A = rng.uniform(-1, 1, size=(3, 4)).astype(np.float32)
B = rng.uniform(-1, 1, size=(4, 5)).astype(np.float32)
a = bg.from_numpy(A.copy())
b = bg.from_numpy(B.copy())
y_ref = (a @ b).numpy()        # NumPy realizer
y_gpu = bg.realize_webgpu(a @ b)
{
    "max_diff": float(np.max(np.abs(y_ref - y_gpu))),
    "upload": _mock.upload_count,
    "matmul": _mock.matmul_count,
    "materialize": _mock.materialize_count,
}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
    expect(result.upload).toBe(2); // A, B
    expect(result.matmul).toBe(1);
    expect(result.materialize).toBe(1);
  });

  it("residency contract: chained matmuls upload 3 inputs and materialise 1 output", async () => {
    // (X @ W1) @ W2 — the (X @ W1) intermediate must STAY on the GPU and
    // get released immediately after the second matmul consumes it. The
    // mock bridge's alive-set + release counter prove this.
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      upload: number;
      matmul: number;
      materialize: number;
      release: number;
      alive_after: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
rng = np.random.RandomState(1)
X = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
W1 = rng.uniform(-1, 1, size=(3, 4)).astype(np.float32)
W2 = rng.uniform(-1, 1, size=(4, 5)).astype(np.float32)

x = bg.from_numpy(X.copy())
w1 = bg.from_numpy(W1.copy())
w2 = bg.from_numpy(W2.copy())

# Reference via NumPy realizer.
y_ref = ((x @ w1) @ w2).numpy()

# GPU realizer with mock bridge.
y_gpu = bg.realize_webgpu((x @ w1) @ w2)

# Three seed buffers uploaded once each.
# Two matmul calls.
# One materialise.
# Intermediate matmul output should be released — total releases:
#   1 (intermediate) + 1 (root materialise post-call) = 2 in v0.
gbt = bg._realize_webgpu.get_registered_gpu_buffer_table()
alive_after = gbt.stats()["handles_alive"]

{
    "max_diff": float(np.max(np.abs(y_ref - y_gpu))),
    "upload": _mock.upload_count,
    "matmul": _mock.matmul_count,
    "materialize": _mock.materialize_count,
    "release": _mock.release_count,
    "alive_after": alive_after,
}
`);
    expect(result.max_diff).toBeLessThan(1e-5);
    expect(result.upload).toBe(3);
    expect(result.matmul).toBe(2);
    expect(result.materialize).toBe(1);
    // Two intermediates: the (X @ W1) handle plus the root after materialise.
    expect(result.release).toBe(2);
    // After the realize call, the only handles still alive are the three
    // seed BUFFERs (X, W1, W2) which persist for cross-call caching.
    expect(result.alive_after).toBe(3);
  });

  it("reuses uploaded seed buffers on a second realize call", async () => {
    // Calling realize_webgpu twice on graphs sharing inputs should not
    // re-upload them — that's the entire point of the GpuBufferTable
    // persisting across calls.
    const target = await getJitTarget();
    const result = await target.run<{
      uploads_first: number;
      uploads_second: number;
      matmul_total: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
A = np.eye(4, dtype=np.float32)
B = np.eye(4, dtype=np.float32) * 2.0
a = bg.from_numpy(A)
b = bg.from_numpy(B)

bg.realize_webgpu(a @ b)
u1 = _mock.upload_count
bg.realize_webgpu(a @ b)
u2 = _mock.upload_count

{"uploads_first": u1, "uploads_second": u2, "matmul_total": _mock.matmul_count}
`);
    expect(result.uploads_first).toBe(2);
    expect(result.uploads_second).toBe(2); // no new uploads on the second call
    expect(result.matmul_total).toBe(2);
  });

  it("realize_webgpu raises if no bridge is registered", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import numpy as np
bg.unregister_webgpu_bridge()
a = bg.from_numpy(np.eye(2, dtype=np.float32))
try:
    bg.realize_webgpu(a @ a)
    result = "no_error"
except bg.JitNotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/No WebGPU bridge registered/);
    expect(err).toMatch(/register_webgpu_bridge/);
  });

  it("unsupported opcode raises with a pointer to bg.realize()", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import numpy as np
# REDUCE is not in the v0 WebGPU realizer whitelist.
a = bg.from_numpy(np.ones((3, 3), dtype=np.float32))
try:
    bg.realize_webgpu(a.sum())
    result = "no_error"
except bg.JitNotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/REDUCE/);
    expect(err).toMatch(/bg\.realize\(\)/);
  });

  it("flash_attention CUSTOM op routes through bridge.flash_attention", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      flash_count: number;
      shape: number[];
      max_diff: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

B, H, Sq, Sk, D = 2, 4, 8, 8, 16
scale = 1.0 / np.sqrt(D)
rng = np.random.RandomState(2)
Q_np = rng.standard_normal((B, H, Sq, D)).astype(np.float32)
K_np = rng.standard_normal((B, H, Sk, D)).astype(np.float32)
V_np = rng.standard_normal((B, H, Sk, D)).astype(np.float32)

Q = bg.from_numpy(Q_np.copy())
K = bg.from_numpy(K_np.copy())
V = bg.from_numpy(V_np.copy())

out = bg.kernels.flash_attention(Q, K, V)
arr_gpu = bg.realize_webgpu(out)

# NumPy reference for parity.
scores = np.matmul(Q_np, np.swapaxes(K_np, -1, -2)) * scale
m_ = scores.max(axis=-1, keepdims=True)
e = np.exp(scores - m_)
p = e / e.sum(axis=-1, keepdims=True)
ref = np.matmul(p, V_np)

{
    "flash_count": _mock.flash_count,
    "shape": list(arr_gpu.shape),
    "max_diff": float(np.max(np.abs(arr_gpu - ref))),
}
`);
    expect(result.flash_count).toBe(1);
    expect(result.shape).toEqual([2, 4, 8, 16]);
    expect(result.max_diff).toBeLessThan(1e-4);
  });

  it("flash_attention accepts an additive mask", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      max_diff: number;
      flash_count: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

B, H, Sq, Sk, D = 1, 2, 4, 4, 8
scale = 1.0 / np.sqrt(D)
rng = np.random.RandomState(3)
Q_np = rng.standard_normal((B, H, Sq, D)).astype(np.float32)
K_np = rng.standard_normal((B, H, Sk, D)).astype(np.float32)
V_np = rng.standard_normal((B, H, Sk, D)).astype(np.float32)
# Causal mask: lower-triangular zeros, upper-triangular -inf.
mask_np = np.where(
    np.tri(Sq, Sk, dtype=bool),
    0.0,
    -1e9,
).astype(np.float32).reshape(1, 1, Sq, Sk)

Q = bg.from_numpy(Q_np.copy())
K = bg.from_numpy(K_np.copy())
V = bg.from_numpy(V_np.copy())
mask = bg.from_numpy(mask_np.copy())

out = bg.kernels.flash_attention(Q, K, V, mask=mask)
arr_gpu = bg.realize_webgpu(out)

scores = np.matmul(Q_np, np.swapaxes(K_np, -1, -2)) * scale + mask_np
m_ = scores.max(axis=-1, keepdims=True)
e = np.exp(scores - m_)
p = e / e.sum(axis=-1, keepdims=True)
ref = np.matmul(p, V_np)
{
    "max_diff": float(np.max(np.abs(arr_gpu - ref))),
    "flash_count": _mock.flash_count,
}
`);
    expect(result.max_diff).toBeLessThan(1e-4);
    expect(result.flash_count).toBe(1);
  });

  it("supported_opcodes returns the v0 whitelist", async () => {
    const target = await getJitTarget();
    const ops = await target.run<string[]>(`
import browsergrad_jit as bg
sorted(bg.webgpu_supported_opcodes())
`);
    expect(ops).toEqual([
      "BUFFER",
      "CAST",
      "CONST",
      "CUSTOM",
      "FUSED_ELEMENTWISE",
      "LOAD",
      "MATMUL",
    ]);
  });

  it("is_available is false until a bridge is registered", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ before: boolean; during: boolean; after: boolean }>(`
import browsergrad_jit as bg
${MOCK_BRIDGE_PY}
bg.unregister_webgpu_bridge()
before = bg.webgpu_is_available()
bg.register_webgpu_bridge(MockBridge())
during = bg.webgpu_is_available()
bg.unregister_webgpu_bridge()
after = bg.webgpu_is_available()
{"before": before, "during": during, "after": after}
`);
    expect(result.before).toBe(false);
    expect(result.during).toBe(true);
    expect(result.after).toBe(false);
  });
});
