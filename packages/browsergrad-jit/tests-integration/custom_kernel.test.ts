/**
 * Custom WGSL kernel tests (PRD-015).
 *
 * Verifies that user-supplied WGSL routes through OP_CUSTOM → the
 * WebGPU realizer's `_h_custom` "user" branch → bridge.run_user_kernel.
 * The Python-side mock bridge satisfies the protocol with a NumPy
 * fallback so pyodide-in-node can exercise the entire flow without
 * a real GPUDevice.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

const MOCK_BRIDGE_WITH_USER = `
import numpy as np

class MockBridge:
    def __init__(self):
        self._handles = {}
        self._next = 0
        self.upload_count = 0
        self.materialize_count = 0
        self.release_count = 0
        self.user_kernel_count = 0
        self._user_kernel_simulator = None

    def set_simulator(self, fn):
        """Tests register a NumPy lambda that mimics the WGSL kernel."""
        self._user_kernel_simulator = fn

    def _mint(self, arr):
        hid = self._next
        self._next += 1
        self._handles[hid] = np.array(arr, copy=True)
        return hid

    def upload(self, data, shape, dtype):
        self.upload_count += 1
        arr = np.frombuffer(data, dtype=np.dtype(dtype))
        if shape and shape != (1,):
            arr = arr.reshape(shape)
        return self._mint(arr)

    def materialize(self, handle, shape, dtype):
        self.materialize_count += 1
        return self._handles[handle].astype(np.dtype(dtype), copy=False).tobytes()

    def release(self, handle):
        self.release_count += 1
        self._handles.pop(handle, None)

    def matmul(self, a, b, m, k, n, dtype):
        out = self._handles[a] @ self._handles[b]
        return self._mint(out.astype(np.dtype(dtype), copy=False))

    def fused_elementwise(self, inputs, ops, shape, dtype):
        return self._mint(np.zeros(shape, dtype=np.dtype(dtype)))

    def cast(self, handle, src_dtype, dst_dtype, shape):
        return self._mint(self._handles[handle].astype(np.dtype(dst_dtype)))

    def flash_attention(self, *args, **kwargs):
        raise NotImplementedError

    def run_user_kernel(self, inputs, wgsl, name, hash, workgroup_size,
                        dispatch_shape, output_length, output_shape, dtype):
        self.user_kernel_count += 1
        if self._user_kernel_simulator is None:
            # Default sim: copy first input, multiply by 2.
            out = self._handles[inputs[0]] * 2.0
        else:
            arrs = [self._handles[h] for h in inputs]
            out = self._user_kernel_simulator(name, *arrs)
        return self._mint(out.astype(np.dtype(dtype), copy=False).reshape(output_shape))
`;

describe("PRD-015 custom WGSL kernels", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    await target.run(`
import browsergrad_jit as bg
${MOCK_BRIDGE_WITH_USER}
_mock = MockBridge()
bg.register_webgpu_bridge(_mock)
`);
  });

  it("decorator builds an OP_CUSTOM UOp tagged 'user'", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      kernel_name: string;
      output_shape: number[];
      requires_grad: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np

double = bg.custom_kernel(
    wgsl="@compute @workgroup_size(1) fn main() {}",
    name="double_each",
    workgroup_size=(1, 1, 1),
    output_shape_fn=lambda s0: s0,
    dispatch_shape_fn=lambda s0: (max(int(np.prod(s0)), 1), 1, 1),
    num_inputs=1,
)
x = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
y = double(x)
{
    "op": y._uop.op,
    "kernel_name": y._uop.arg["kernel_name"],
    "output_shape": list(y._uop.shape),
    "requires_grad": y.requires_grad,
}
`);
    expect(result.op).toBe("CUSTOM");
    expect(result.kernel_name).toBe("double_each");
    expect(result.output_shape).toEqual([3]);
    expect(result.requires_grad).toBe(false); // forward-only in v0
  });

  it("realize_webgpu dispatches user kernel via bridge.run_user_kernel", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      values: number[];
      calls: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

double = bg.custom_kernel(
    wgsl="@compute @workgroup_size(1) fn main() {}",
    name="double_each",
    workgroup_size=(1, 1, 1),
    output_shape_fn=lambda s0: s0,
    dispatch_shape_fn=lambda s0: (max(int(np.prod(s0)), 1), 1, 1),
    num_inputs=1,
)
_mock.set_simulator(lambda name, *arrs: arrs[0] * 2.0)
x = bg.from_numpy(np.array([1.5, 2.5, 3.5], dtype=np.float32))
y = bg.realize_webgpu(double(x))
{"values": y.tolist(), "calls": _mock.user_kernel_count}
`);
    expect(result.values).toEqual([3.0, 5.0, 7.0]);
    expect(result.calls).toBe(1);
  });

  it("same WGSL → same hash → idempotent registration", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      hash_a: string;
      hash_b: string;
      registry_size_before: number;
      registry_size_after: number;
    }>(`
import browsergrad_jit as bg
from browsergrad_jit._custom_kernel import get_registry

src = "@compute @workgroup_size(1) fn main() {}"
get_registry().clear()  # reset for this test
before = len(get_registry())

k1 = bg.custom_kernel(
    wgsl=src, name="k", workgroup_size=(1, 1, 1),
    output_shape_fn=lambda s: s, dispatch_shape_fn=lambda s: (1, 1, 1),
    num_inputs=1,
)
k2 = bg.custom_kernel(
    wgsl=src, name="k2", workgroup_size=(1, 1, 1),
    output_shape_fn=lambda s: s, dispatch_shape_fn=lambda s: (1, 1, 1),
    num_inputs=1,
)
after = len(get_registry())
{
    "hash_a": k1.hash,
    "hash_b": k2.hash,
    "registry_size_before": before,
    "registry_size_after": after,
}
`);
    expect(result.hash_a).toBe(result.hash_b);
    expect(result.registry_size_after).toBe(result.registry_size_before + 1);
  });

  it("conflicting workgroup_size for same WGSL raises", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
from browsergrad_jit._custom_kernel import get_registry
get_registry().clear()
src = "@compute @workgroup_size(8) fn main() {}"
bg.custom_kernel(
    wgsl=src, name="a", workgroup_size=(8, 1, 1),
    output_shape_fn=lambda s: s, dispatch_shape_fn=lambda s: (1, 1, 1),
    num_inputs=1,
)
try:
    bg.custom_kernel(
        wgsl=src, name="b", workgroup_size=(16, 1, 1),
        output_shape_fn=lambda s: s, dispatch_shape_fn=lambda s: (1, 1, 1),
        num_inputs=1,
    )
    result = "no_error"
except ValueError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/workgroup_size/);
  });

  it("calling a user kernel through bg.realize() (NumPy) refuses with a pointer", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import numpy as np

k = bg.custom_kernel(
    wgsl="@compute @workgroup_size(1) fn main() {}",
    name="noop", workgroup_size=(1, 1, 1),
    output_shape_fn=lambda s: s, dispatch_shape_fn=lambda s: (1, 1, 1),
    num_inputs=1,
)
x = bg.from_numpy(np.zeros(2, dtype=np.float32))
y = k(x)
try:
    # .numpy() routes through the NumPy realizer, which has no CUSTOM/user branch.
    y.numpy()
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    // CUSTOM op routes to NumPy's _h_custom which requires arg["fn"].
    // For user kernels arg has no "fn"; the realizer surfaces a clear error.
    expect(err).toMatch(/CUSTOM/);
  });

  it("invalid num_inputs refuses at decoration time", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
try:
    bg.custom_kernel(
        wgsl="@compute @workgroup_size(1) fn main() {}",
        name="bad", workgroup_size=(1, 1, 1),
        output_shape_fn=lambda *s: s[0],
        dispatch_shape_fn=lambda *s: (1, 1, 1),
        num_inputs=9,
    )
    result = "no_error"
except ValueError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/num_inputs=9/);
  });
});
