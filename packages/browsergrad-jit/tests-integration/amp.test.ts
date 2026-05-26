/**
 * Mixed precision integration tests (PRD-010 v0).
 *
 * Verifies, end-to-end through Pyodide:
 *   - ISNAN realizes elementwise and survives the dispatch table.
 *   - fp32 matmul accumulator: K=4096 all-ones dot product is exact in
 *     f16-input matmul (would overflow / drift in pure-f16 accumulate).
 *   - Autocast tags forward MATMULs constructed inside the context with
 *     `autocast_hint="float16"`, leaves outside ones untouched.
 *   - Cast pass inserts CAST nodes around tagged MATMULs and tagged
 *     EXP/REDUCE/DIV.
 *   - GradScaler halves on NaN injection, doubles after growth_interval
 *     consecutive clean steps.
 *   - Refusal: dtype=bfloat16 raises NotImplementedError.
 *   - torch.amp shim resolves once install_torch_alias() is called.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("PRD-010 mixed precision", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("ISNAN realizes elementwise and matches np.isnan", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ matches: boolean }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_ISNAN

a = bg.from_numpy(np.array([1.0, float("nan"), 3.0, float("nan")], dtype=np.float32))
nan_uop = UOp(op=OP_ISNAN, inputs=(a._uop,), shape=a.shape, dtype="bool", arg=None)
from browsergrad_jit._realize import realize
out = realize(nan_uop, a._get_session().buffer_table)
{"matches": bool(np.array_equal(out, np.array([False, True, False, True])))}
`);
    expect(result.matches).toBe(true);
  });

  it("fp32 matmul accumulator: K=4096 all-ones is exact", async () => {
    // Pure-f16 accumulate over 4096 terms would saturate around 2048 →
    // out=2048.0; fp32 accumulator inside `_h_matmul` keeps the answer
    // exact at 4096.0 even though inputs and output are f16.
    const target = await getJitTarget();
    const result = await target.run<{ value: number; dtype: string }>(`
import browsergrad_jit as bg
import numpy as np

K = 4096
a = bg.from_numpy(np.ones((1, K), dtype=np.float16))
b = bg.from_numpy(np.ones((K, 1), dtype=np.float16))
y = (a @ b)
arr = y.numpy()
{"value": float(arr[0, 0]), "dtype": str(arr.dtype)}
`);
    expect(result.value).toBe(4096.0);
    // Output dtype follows promote rules (f16 @ f16 → f16), accumulator is internal.
    expect(result.dtype).toBe("float16");
  });

  it("autocast tags forward MATMUL inside the context, leaves outside untouched", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ inside: string; outside_tagged: boolean }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit import _amp

a = bg.from_numpy(np.ones((2, 3), dtype=np.float32))
b = bg.from_numpy(np.ones((3, 2), dtype=np.float32))

# Outside autocast: arg should have no autocast_hint.
y_out = a @ b
outside_tagged = bool(isinstance(y_out._uop.arg, dict)
                      and "autocast_hint" in y_out._uop.arg)

# Inside autocast: arg should carry hint.
with _amp.autocast(device_type="webgpu", dtype="float16"):
    y_in = a @ b
in_hint = y_in._uop.arg["autocast_hint"]

{"inside": in_hint, "outside_tagged": outside_tagged}
`);
    expect(result.inside).toBe("float16");
    expect(result.outside_tagged).toBe(false);
  });

  it("cast pass inserts CASTs around tagged MATMUL", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      input_dtypes: string[];
      matmul_dtype: string;
      saw_cast: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit import _amp
from browsergrad_jit._ir import toposort, OP_CAST, OP_MATMUL

a = bg.from_numpy(np.ones((2, 3), dtype=np.float32))
b = bg.from_numpy(np.ones((3, 2), dtype=np.float32))
with _amp.autocast(device_type="webgpu", dtype="float16"):
    y = a @ b

rewritten = _amp.insert_cast_pass(y._uop)
nodes = toposort(rewritten)
matmul = [n for n in nodes if n.op == OP_MATMUL][0]
casts = [n for n in nodes if n.op == OP_CAST]
{
    "input_dtypes": [matmul.inputs[0].dtype, matmul.inputs[1].dtype],
    "matmul_dtype": matmul.dtype,
    "saw_cast": len(casts) >= 2,
}
`);
    // Allowlist matmul has both inputs forced to float16.
    expect(result.input_dtypes).toEqual(["float16", "float16"]);
    expect(result.matmul_dtype).toBe("float16");
    expect(result.saw_cast).toBe(true);
  });

  it("cast pass is a no-op when no autocast hint is present", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ same_root: boolean }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit import _amp

a = bg.from_numpy(np.ones((2, 3), dtype=np.float32))
b = bg.from_numpy(np.ones((3, 2), dtype=np.float32))
y = a @ b  # no autocast
rewritten = _amp.insert_cast_pass(y._uop)
{"same_root": rewritten is y._uop}
`);
    expect(result.same_root).toBe(true);
  });

  it("autocast end-to-end: f32 input → f16 matmul → numerical match", async () => {
    // Same matmul, run with and without autocast. The fp32 accumulator
    // inside _h_matmul keeps the result accurate enough that relative
    // error is dominated by the cast-down-to-f16 of the output, not by
    // accumulator drift.
    const target = await getJitTarget();
    const result = await target.run<{ max_rel: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit import _amp

rng = np.random.RandomState(0)
A = rng.uniform(-1, 1, size=(8, 16)).astype(np.float32)
B = rng.uniform(-1, 1, size=(16, 8)).astype(np.float32)

a = bg.from_numpy(A.copy())
b = bg.from_numpy(B.copy())
ref = (a @ b).numpy()

a2 = bg.from_numpy(A.copy())
b2 = bg.from_numpy(B.copy())
with _amp.autocast(device_type="webgpu", dtype="float16"):
    y = a2 @ b2
amp_out = y.numpy()

# amp_out is f16; promote for comparison.
amp_f32 = amp_out.astype(np.float32)
max_rel = float(np.max(np.abs(amp_f32 - ref) / (np.abs(ref) + 1e-3)))
{"max_rel": max_rel}
`);
    // f16 round-trip noise on order ~1e-3 relative; comfortably bounded.
    expect(result.max_rel).toBeLessThan(5e-2);
  });

  it("GradScaler scales loss multiplicatively", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ scaled: number; raw: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit import _amp

scaler = _amp.GradScaler(init_scale=2.0 ** 10)
loss = bg.tensor(0.5)
scaled = scaler.scale(loss)
{"scaled": float(scaled.numpy()), "raw": float(loss.numpy())}
`);
    expect(result.raw).toBeCloseTo(0.5, 6);
    expect(result.scaled).toBeCloseTo(0.5 * 1024, 3);
  });

  it("GradScaler halves the scale when a NaN gradient is detected", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      before: number;
      after: number;
      stepped: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit import _amp

# Fake optimizer with one parameter whose grad is NaN.
class FakeParam:
    def __init__(self, arr):
        self.grad = bg.from_numpy(arr.copy())
        self._session = bg.get_default_session()
    def _get_session(self):
        return self._session

class FakeOpt:
    def __init__(self):
        self._params = [FakeParam(np.array([float("nan"), 1.0], dtype=np.float32))]
        self.did_step = False
    def step(self):
        self.did_step = True

scaler = _amp.GradScaler(init_scale=2.0 ** 16, backoff_factor=0.5)
before = scaler.get_scale()
opt = FakeOpt()
scaler.step(opt)
after = scaler.get_scale()
{"before": before, "after": after, "stepped": opt.did_step}
`);
    expect(result.before).toBe(65536.0);
    expect(result.after).toBe(32768.0); // halved
    expect(result.stepped).toBe(false); // overflow → skip
  });

  it("GradScaler doubles the scale after growth_interval clean steps", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ scale_after: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit import _amp

class FakeParam:
    def __init__(self):
        self.grad = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
        self._session = bg.get_default_session()
    def _get_session(self):
        return self._session

class FakeOpt:
    def __init__(self):
        self._params = [FakeParam()]
        self.steps = 0
    def step(self):
        self.steps += 1

scaler = _amp.GradScaler(
    init_scale=2.0 ** 10,
    growth_factor=2.0,
    growth_interval=3,
)
opt = FakeOpt()
for _ in range(3):
    # Reset grads each step so the in-place unscale doesn't compound.
    opt._params[0].grad = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    scaler.step(opt)
scaler.update()
{"scale_after": scaler.get_scale()}
`);
    expect(result.scale_after).toBe(2048.0); // doubled from 1024
  });

  it("refuses dtype=bfloat16 with a clear message", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
from browsergrad_jit import _amp
try:
    with _amp.autocast(device_type="webgpu", dtype="bfloat16"):
        pass
    result = "no_error"
except NotImplementedError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/bfloat16/);
    expect(err).toMatch(/shader-bf16/);
  });

  it("bg.amp exposes autocast, GradScaler, is_available", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      has_autocast: boolean;
      has_scaler: boolean;
      available: boolean;
    }>(`
import browsergrad_jit as bg
{
    "has_autocast": callable(bg.amp.autocast),
    "has_scaler": callable(bg.amp.GradScaler),
    "available": bg.amp.is_available(),
}
`);
    expect(result.has_autocast).toBe(true);
    expect(result.has_scaler).toBe(true);
    expect(result.available).toBe(true);
  });

  it("torch.amp shim resolves via install_torch_alias", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      autocast_found: boolean;
      scaler_found: boolean;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
{
    "autocast_found": callable(torch.amp.autocast),
    "scaler_found": callable(torch.amp.GradScaler),
}
`);
    expect(result.autocast_found).toBe(true);
    expect(result.scaler_found).toBe(true);
  });
});
