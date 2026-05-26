/**
 * Pile B + Pile C — limited / impossible feature surface.
 *
 * Pile B (possible but limited): autocast is a no-op, image transforms are
 * NumPy-only, linalg wraps numpy.linalg, multi-GPU device hooks no-op.
 *
 * Pile C (impossible in browser): compile / fx / jit / cuda.* / distributed /
 * onnx / quantization stubs raise NotImplementedError with an architectural
 * reason — never silent success (that was greed's mistake).
 */

import { beforeAll, describe, expect, it } from "vitest";
import { clearNamespace, getGradTarget } from "./pyodide-host";

let target: Awaited<ReturnType<typeof getGradTarget>>;

beforeAll(async () => {
  target = await getGradTarget();
}, 120_000);

async function reset(): Promise<void> {
  await clearNamespace(target);
}

const PRELUDE = `
import browsergrad_grad as grad
grad.install_torch_alias()
import torch
`;

describe("Pile B — possible but limited", () => {
  beforeAll(reset);

  it("torch.amp.autocast is a no-op context manager (no real fp16 in-browser)", async () => {
    const result = await target.run<{ stayed_f32: boolean }>(`
${PRELUDE}
import numpy as np
import browsergrad_grad as grad2
with torch.amp.autocast(device_type='cpu'):
    x = grad2.Tensor(np.ones((2, 3), dtype=np.float32))
    y = x + x
{"stayed_f32": str(y.dtype) == "float32"}
`);
    expect(result.stayed_f32).toBe(true);
  });

  it("torch.linalg.norm wraps numpy.linalg.norm", async () => {
    const result = await target.run<{ ours: number; oracle: number }>(`
${PRELUDE}
import numpy as np
import browsergrad_grad as grad2
x_np = np.array([3.0, 4.0], dtype=np.float32)
ours = float(torch.linalg.norm(grad2.Tensor(x_np)).data)
oracle = float(np.linalg.norm(x_np))
{"ours": ours, "oracle": oracle}
`);
    expect(result.ours).toBeCloseTo(result.oracle, 5);
  });

  it("torch.linalg.inv wraps numpy.linalg.inv", async () => {
    const result = await target.run<{ identity_max_dev: number }>(`
${PRELUDE}
import numpy as np
import browsergrad_grad as grad2
A_np = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
A = grad2.Tensor(A_np)
Ainv = torch.linalg.inv(A)
prod = A_np @ np.asarray(Ainv.data)
identity_max_dev = float(np.max(np.abs(prod - np.eye(2, dtype=np.float32))))
{"identity_max_dev": identity_max_dev}
`);
    expect(result.identity_max_dev).toBeLessThan(1e-4);
  });

  it("model.to(device) accepts a device but doesn't move (no-op shim)", async () => {
    const result = await target.run<{ same: boolean }>(`
${PRELUDE}
import browsergrad_grad.nn as nn
m = nn.Linear(3, 2)
m2 = m.to('cuda:0')  # accepted, but ignored
{"same": m2 is m}
`);
    expect(result.same).toBe(true);
  });
});

describe("Pile C — physically impossible in browser", () => {
  beforeAll(reset);

  it("torch.compile raises NotImplementedError mentioning the reason", async () => {
    const errored = await target.run<boolean>(`
${PRELUDE}
import browsergrad_grad.nn as nn
m = nn.Linear(3, 2)
_ok = False
try:
    torch.compile(m)
except NotImplementedError as e:
    _ok = "compiler" in str(e) or "browser" in str(e)
_ok
`);
    expect(errored).toBe(true);
  });

  it("torch.cuda.is_available() returns False (legitimate answer)", async () => {
    const result = await target.run<boolean>(`
${PRELUDE}
bool(torch.cuda.is_available())
`);
    expect(result).toBe(false);
  });

  it("torch.cuda.device_count() returns 0", async () => {
    const result = await target.run<number>(`
${PRELUDE}
int(torch.cuda.device_count())
`);
    expect(result).toBe(0);
  });

  it("torch.distributed.init_process_group raises with browser-reason message", async () => {
    const errored = await target.run<boolean>(`
${PRELUDE}
_ok = False
try:
    torch.distributed.init_process_group("nccl")
except NotImplementedError as e:
    _ok = "browser" in str(e) or "multi-machine" in str(e)
_ok
`);
    expect(errored).toBe(true);
  });

  it("torch.onnx.export raises NotImplementedError", async () => {
    const errored = await target.run<boolean>(`
${PRELUDE}
_ok = False
try:
    torch.onnx.export(None, None, "/tmp/x.onnx")
except NotImplementedError:
    _ok = True
_ok
`);
    expect(errored).toBe(true);
  });

  it("torch.jit.script raises NotImplementedError", async () => {
    const errored = await target.run<boolean>(`
${PRELUDE}
_ok = False
try:
    torch.jit.script(lambda x: x)
except NotImplementedError:
    _ok = True
_ok
`);
    expect(errored).toBe(true);
  });

  it("torch.quantization.quantize raises NotImplementedError", async () => {
    const errored = await target.run<boolean>(`
${PRELUDE}
_ok = False
try:
    torch.quantization.quantize(None, None)
except NotImplementedError:
    _ok = True
_ok
`);
    expect(errored).toBe(true);
  });
});
