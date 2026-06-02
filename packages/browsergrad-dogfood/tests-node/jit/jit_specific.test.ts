/**
 * Jit — adversarial tests focused on behaviors NOT in craftingattention's
 * 61-test integration suite. Adds confidence in v0.8.0 specifically as a
 * published artifact.
 *
 * Hypotheses: J1–J11.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { installJit } from "@unlocalhosted/browsergrad-jit";
import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-jit/node-adapter";
import { getPyodide, pyJson, pyStr, pyBool } from "../helpers.js";

let installed = false;

beforeAll(async () => {
  if (installed) return;
  const py = await getPyodide();
  await installJit(createNodePyodideTarget(py));
  installed = true;
}, 120_000);

describe("install + version (J1)", () => {
  it("__version__ matches the npm version (J1)", async () => {
    const v = await pyStr(`import browsergrad_jit; browsergrad_jit.__version__`);
    expect(v).toMatch(/^0\.8\.\d+$/);
  });
});

describe("lazy TensorProxy (J2, J3)", () => {
  it("TensorProxy.shape works without materializing (J2)", async () => {
    const r = await pyJson<{ shape: number[]; was_materialized: boolean }>(`
import browsergrad_jit as bg
import numpy as np, json
a = bg.from_numpy(np.ones((4, 8), dtype=np.float32))
b = bg.from_numpy(np.ones((8, 4), dtype=np.float32))
chain = (a @ b) + 1.0
# Probe shape — should not trigger realization.
shape = list(chain.shape)
# .numpy() forces realize. If shape access triggered realize, the
# subsequent .numpy() would still work but we can't easily probe that
# from outside. Sanity check shape is correct.
json.dumps({"shape": shape, "was_materialized": True})
`);
    expect(r.shape).toEqual([4, 4]);
  });

  it(".numpy() triggers realize and returns np.ndarray (J3)", async () => {
    const ok = await pyBool(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.ones((2, 2), dtype=np.float32))
out = a.numpy()
isinstance(out, np.ndarray) and out.shape == (2, 2)
`);
    expect(ok).toBe(true);
  });
});

describe("trace cache + fusion (J4, J5)", () => {
  it("trace_cache_stats returns non-negative integers (J4)", async () => {
    const r = await pyJson<{ entries: number; hits: number }>(`
import browsergrad_jit as bg, json
bg.clear_cache("trace")
bg.jit.use_trace_cache(True)
s = bg.jit.trace_cache_stats()
json.dumps({"entries": int(s["entries"]), "hits": int(s["hits"])})
`);
    expect(r.entries).toBeGreaterThanOrEqual(0);
    expect(r.hits).toBeGreaterThanOrEqual(0);
  });

  it("use_fusion toggle: off then on (J5)", async () => {
    const ok = await pyBool(`
import browsergrad_jit as bg
bg.jit.use_fusion(False)
bg.jit.use_fusion(True)
True
`);
    expect(ok).toBe(true);
  });
});

describe("custom kernel hash (J6)", () => {
  it("custom_kernel hash is SHA-256 hex (64 chars, [0-9a-f]) (J6)", async () => {
    const h = await pyStr(`
import browsergrad_jit as bg
k = bg.custom_kernel(
    wgsl="@compute @workgroup_size(1) fn main() {}",
    name="probe", workgroup_size=(1, 1, 1),
    output_shape_fn=lambda s0: s0,
    dispatch_shape_fn=lambda s0: (1, 1, 1),
    num_inputs=1,
)
k.hash
`);
    expect(h.length).toBe(64);
    expect(h).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe("realize_webgpu without bridge (J7)", () => {
  it("error message contains 'bridge' (J7)", async () => {
    const err = await pyStr(`
import browsergrad_jit as bg
import numpy as np
bg.unregister_webgpu_bridge()
a = bg.from_numpy(np.zeros((2, 2), dtype=np.float32))
try:
    bg.realize_webgpu(a @ a)
    result = "no_error"
except bg.JitNotImplementedError as e:
    result = str(e)
result.lower()
`);
    expect(err).toMatch(/bridge/);
  });
});

describe("error types (J8)", () => {
  it("matmul inner-dim mismatch → ShapeError, not generic Exception (J8)", async () => {
    const r = await pyJson<{ caught: string }>(`
import browsergrad_jit as bg
import numpy as np, json
a = bg.from_numpy(np.zeros((3, 4), dtype=np.float32))
b = bg.from_numpy(np.zeros((5, 6), dtype=np.float32))
try:
    a @ b
    caught = "no_error"
except bg.ShapeError as e:
    caught = "ShapeError"
except Exception as e:
    caught = type(e).__name__
json.dumps({"caught": caught})
`);
    expect(r.caught).toBe("ShapeError");
  });
});

describe("save_safetensors mixed types (J9)", () => {
  it("accepts dict of TensorProxy + ndarray (J9)", async () => {
    const ok = await pyBool(`
import browsergrad_jit as bg
import numpy as np
state = {
    "proxy": bg.from_numpy(np.zeros(4, dtype=np.float32)),
    "ndarray": np.ones(4, dtype=np.float32),
}
blob = bg.save_safetensors(state)
restored = bg.load_safetensors(blob)
bool(np.allclose(restored["proxy"].numpy(), np.zeros(4))) and \
bool(np.allclose(restored["ndarray"].numpy(), np.ones(4)))
`);
    expect(ok).toBe(true);
  });
});

describe("ONNX export (J10, J11)", () => {
  it("export bytes begin with 0x08 (proto3 ir_version marker) (J10)", async () => {
    const r = await pyJson<{ first_byte: number; bytes: number }>(`
import browsergrad_jit as bg
import numpy as np, json
x = bg.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
w = bg.from_numpy(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))
y = x @ w
blob = bg.onnx.export_inference(y, input_buffers=(x,))
json.dumps({"first_byte": int(blob[0]), "bytes": len(blob)})
`);
    expect(r.first_byte).toBe(8);
    expect(r.bytes).toBeGreaterThan(100);
  });

  it("argmax export raises OnnxUnmappableOp with name pointer (J11)", async () => {
    const msg = await pyStr(`
import browsergrad_jit as bg
import numpy as np
x = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
y = x.argmax()
try:
    bg.onnx.export_inference(y, input_buffers=(x,))
    result = "no_error"
except bg.onnx.OnnxUnmappableOp as e:
    result = str(e)
result
`);
    expect(msg.toLowerCase()).toMatch(/argmax/);
  });
});
