/**
 * Safetensors integration tests (PRD-008 v0).
 *
 * Verifies the round-trip: build a dict of TensorProxies, write to a
 * temp file via `save_safetensors`, load back via `load_safetensors`,
 * compare values + dtypes + shapes. Exercises all supported dtype
 * variants and confirms the BF16 surface raises a clean error.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("PRD-008 safetensors round-trip", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("round-trips a 3-tensor dict through bytes", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      names: string[];
      x_match: boolean;
      y_match: boolean;
      z_match: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
import tempfile, os, json

# Three tensors with distinct dtypes + shapes.
src = {
    "x": bg.tensor(np.arange(12, dtype=np.float32).reshape(3, 4)),
    "y": bg.tensor(np.arange(6, dtype=np.float32).reshape(2, 3) * 10),
    "z": bg.tensor(np.array([1, 2, 3, 4], dtype=np.int64)),
}

with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    path = f.name
try:
    bg.save_safetensors(src, path)
    loaded = bg.load_safetensors(path)
    result = {
        "names": sorted(list(loaded.keys())),
        "x_match": bool(np.allclose(loaded["x"].numpy(), src["x"].numpy())),
        "y_match": bool(np.allclose(loaded["y"].numpy(), src["y"].numpy())),
        "z_match": bool(np.array_equal(loaded["z"].numpy(), src["z"].numpy())),
    }
finally:
    os.unlink(path)
result
`);
    expect(result.names).toEqual(["x", "y", "z"]);
    expect(result.x_match).toBe(true);
    expect(result.y_match).toBe(true);
    expect(result.z_match).toBe(true);
  });

  it("preserves dtypes across save+load", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np
import tempfile, os

# Coverage for every dtype safetensors supports (except BF16 which is
# a separate test). Bool gets its own row because NumPy's bool dtype
# is sometimes called bool_; the safetensors writer should normalize.
src = {
    "f32": bg.tensor(np.arange(4, dtype=np.float32)),
    "f16": bg.tensor(np.array([1, 2, 3], dtype=np.float16)),
    "i64": bg.tensor(np.array([1, 2], dtype=np.int64)),
    "i32": bg.tensor(np.array([1, 2], dtype=np.int32)),
    "i8":  bg.tensor(np.array([1, 2], dtype=np.int8)),
    "u8":  bg.tensor(np.array([1, 2], dtype=np.uint8)),
    "bool": bg.tensor(np.array([True, False, True], dtype=bool)),
}
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    path = f.name
try:
    bg.save_safetensors(src, path)
    loaded = bg.load_safetensors(path)
    out = {name: t.dtype for name, t in loaded.items()}
finally:
    os.unlink(path)
out
`);
    expect(result.f32).toBe("float32");
    expect(result.f16).toBe("float16");
    expect(result.i64).toBe("int64");
    expect(result.i32).toBe("int32");
    expect(result.i8).toBe("int8");
    expect(result.u8).toBe("uint8");
    expect(result.bool).toBe("bool");
  });

  it("supports the `dtype=` cast argument", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ before: string; after: string }>(`
import browsergrad_jit as bg
import numpy as np
import tempfile, os

src = {"w": bg.tensor(np.arange(4, dtype=np.float16))}
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    path = f.name
try:
    bg.save_safetensors(src, path)
    cast_loaded = bg.load_safetensors(path, dtype="float32")
    plain_loaded = bg.load_safetensors(path)
    result = {
        "before": plain_loaded["w"].dtype,
        "after": cast_loaded["w"].dtype,
    }
finally:
    os.unlink(path)
result
`);
    expect(result.before).toBe("float16");
    expect(result.after).toBe("float32");
  });

  it("raises a clean error on BF16", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
import struct, json, tempfile, os

# Hand-build a tiny BF16 safetensors file.
header = {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
body = b"\\x00" * 4

with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    path = f.name
try:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(body)
    try:
        bg.load_safetensors(path)
        result = "no_error"
    except NotImplementedError as e:
        result = str(e)
finally:
    os.unlink(path)
result
`);
    expect(err).toMatch(/BF16/);
    expect(err).toMatch(/PRD-010/);
  });

  it("refuses HTTP URLs with a clear error (PRD-008.2 reference)", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit as bg
try:
    bg.load_safetensors("https://example.com/model.safetensors")
    result = "no_error"
except NotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/HTTP/);
    expect(err).toMatch(/PRD-008\.2/);
  });

  it("integrates with nn.Module.load_state_dict", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ match: boolean }>(`
import browsergrad_jit as bg
import numpy as np
import tempfile, os

bg.manual_seed(0)
model_a = bg.nn.Linear(4, 3)
state = {n: p for n, p in model_a.named_parameters()}

with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    path = f.name
try:
    bg.save_safetensors(state, path)
    loaded = bg.load_safetensors(path)
    model_b = bg.nn.Linear(4, 3)
    # state_dict format on jit's Linear is {"weight": ndarray, "bias": ndarray}.
    model_b.load_state_dict({k: v.numpy() for k, v in loaded.items()})
    # Now models a and b should have identical parameters.
    same = True
    for (n_a, p_a), (n_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if not np.allclose(p_a.numpy(), p_b.numpy()):
            same = False
            break
    result = {"match": same}
finally:
    os.unlink(path)
result
`);
    expect(result.match).toBe(true);
  });
});
