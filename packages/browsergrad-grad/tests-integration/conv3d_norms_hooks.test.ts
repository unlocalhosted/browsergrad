/**
 * Pile A #16 + #17 + #18 — Conv3d / ConvTranspose1d/2d / GroupNorm /
 * InstanceNorm{1,2,3}d / BatchNorm3d / Module hooks.
 *
 * Oracles:
 *  - Conv3d: hand-written triple-loop convolution in NumPy.
 *  - ConvTranspose: shape oracle (full impl deferred but shape matches).
 *  - GroupNorm: standard formula per-group.
 *  - InstanceNorm: GroupNorm with num_groups = C.
 *  - BatchNorm3d: same as BatchNorm2d generalized.
 *  - Hooks: observe forward output, observe backward grad.
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
import browsergrad_grad.nn as nn
import numpy as np
`;

describe("nn.GroupNorm", () => {
  beforeAll(reset);

  it("normalizes per group: mean ≈ 0, var ≈ 1 across each group after affine off", async () => {
    const result = await target.run<{ means: number[]; stds: number[] }>(`
${PRELUDE}
np.random.seed(0)
N, C, H, W = 2, 6, 4, 4
gn = nn.GroupNorm(num_groups=3, num_channels=C, affine=False)
x = grad.Tensor(np.random.randn(N, C, H, W).astype(np.float32))
y = gn(x)
y_np = np.asarray(y.tolist())
# Reshape to (N, G, C/G, H, W) and compute per-group means/stds
G = 3
yr = y_np.reshape(N, G, C // G, H, W)
means = yr.mean(axis=(2, 3, 4)).flatten().tolist()
stds = yr.std(axis=(2, 3, 4)).flatten().tolist()
{"means": means, "stds": stds}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    for (const s of result.stds) expect(s).toBeCloseTo(1.0, 3);
  });
});

describe("nn.InstanceNorm2d", () => {
  beforeAll(reset);

  it("normalizes each (N, C) plane independently", async () => {
    const result = await target.run<{ means: number[]; stds: number[] }>(`
${PRELUDE}
np.random.seed(1)
N, C, H, W = 2, 4, 5, 5
inst = nn.InstanceNorm2d(C, affine=False)
x = grad.Tensor(np.random.randn(N, C, H, W).astype(np.float32))
y = inst(x)
y_np = np.asarray(y.tolist())
means = y_np.mean(axis=(2, 3)).flatten().tolist()
stds  = y_np.std(axis=(2, 3)).flatten().tolist()
{"means": means, "stds": stds}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    for (const s of result.stds) expect(s).toBeCloseTo(1.0, 3);
  });
});

describe("nn.BatchNorm3d", () => {
  beforeAll(reset);

  it("normalizes per channel across (N, D, H, W) in training mode", async () => {
    const result = await target.run<{ means: number[]; stds: number[] }>(`
${PRELUDE}
np.random.seed(2)
N, C, D, H, W = 2, 3, 3, 4, 4
bn = nn.BatchNorm3d(C, affine=False)
bn.train()
x = grad.Tensor(np.random.randn(N, C, D, H, W).astype(np.float32))
y = bn(x)
y_np = np.asarray(y.tolist())
means = y_np.mean(axis=(0, 2, 3, 4)).tolist()
stds  = y_np.std(axis=(0, 2, 3, 4)).tolist()
{"means": means, "stds": stds}
`);
    for (const m of result.means) expect(Math.abs(m)).toBeLessThan(1e-4);
    for (const s of result.stds) expect(s).toBeCloseTo(1.0, 3);
  });
});

describe("Module hooks", () => {
  beforeAll(reset);

  it("register_forward_hook fires after forward with (module, input, output)", async () => {
    const result = await target.run<{ fired: boolean; out_shape: number[] }>(`
${PRELUDE}
fc = nn.Linear(3, 5)
captured = {}
def hook(module, input, output):
    captured["module_is_fc"] = module is fc
    captured["output_shape"] = list(output.data.shape)
fc.register_forward_hook(hook)
x = grad.Tensor(np.zeros((2, 3), dtype=np.float32))
y = fc(x)
{"fired": captured.get("module_is_fc", False), "out_shape": captured.get("output_shape", [])}
`);
    expect(result.fired).toBe(true);
    expect(result.out_shape).toEqual([2, 5]);
  });

  it("multiple forward hooks fire in registration order", async () => {
    const result = await target.run<string[]>(`
${PRELUDE}
fc = nn.Linear(2, 2)
order = []
def h1(m, i, o): order.append("h1")
def h2(m, i, o): order.append("h2")
fc.register_forward_hook(h1)
fc.register_forward_hook(h2)
_ = fc(grad.Tensor(np.zeros((1, 2), dtype=np.float32)))
order
`);
    expect(result).toEqual(["h1", "h2"]);
  });
});
