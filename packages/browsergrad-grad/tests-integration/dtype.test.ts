/**
 * Pile A #3 — multi-dtype Tensor.
 *
 * Tensor stores .dtype; cast methods .long() / .float() / .bool() / .int()
 * produce new Tensors with the requested dtype. The default (constructor with
 * no dtype) remains float32 so all existing code paths behave the same.
 *
 * Oracle: NumPy dtype strings directly.
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
import numpy as np
`;

describe("Tensor.dtype + cast methods", () => {
  beforeAll(reset);

  it("default constructor dtype is float32", async () => {
    const result = await target.run<string>(`
${PRELUDE}
str(grad.Tensor([1.0, 2.0, 3.0]).dtype)
`);
    expect(result).toBe("float32");
  });

  it("constructor accepts dtype='int64' and stores as int64", async () => {
    const result = await target.run<{ dtype: string; vals: number[] }>(`
${PRELUDE}
t = grad.Tensor([1, 2, 3], dtype='int64')
{"dtype": str(t.dtype), "vals": t.tolist()}
`);
    expect(result.dtype).toBe("int64");
    expect(result.vals).toEqual([1, 2, 3]);
  });

  it("constructor accepts dtype='bool' and stores as bool", async () => {
    const result = await target.run<{ dtype: string; vals: boolean[] }>(`
${PRELUDE}
t = grad.Tensor([True, False, True, True], dtype='bool')
{"dtype": str(t.dtype), "vals": t.tolist()}
`);
    expect(result.dtype).toBe("bool");
    expect(result.vals).toEqual([true, false, true, true]);
  });

  it(".long() casts to int64", async () => {
    const result = await target.run<{ dtype: string; vals: number[] }>(`
${PRELUDE}
t = grad.Tensor([1.5, 2.9, -3.1]).long()
{"dtype": str(t.dtype), "vals": t.tolist()}
`);
    expect(result.dtype).toBe("int64");
    expect(result.vals).toEqual([1, 2, -3]);  // truncation toward zero per numpy int cast
  });

  it(".bool() casts to bool (nonzero true)", async () => {
    const result = await target.run<{ dtype: string; vals: boolean[] }>(`
${PRELUDE}
t = grad.Tensor([0.0, 1.0, -2.5, 0.0]).bool()
{"dtype": str(t.dtype), "vals": t.tolist()}
`);
    expect(result.dtype).toBe("bool");
    expect(result.vals).toEqual([false, true, true, false]);
  });

  it(".float() casts back to float32", async () => {
    const result = await target.run<{ dtype: string }>(`
${PRELUDE}
t = grad.Tensor([1, 2, 3], dtype='int64').float()
{"dtype": str(t.dtype)}
`);
    expect(result.dtype).toBe("float32");
  });

  it(".int() casts to int32", async () => {
    const result = await target.run<{ dtype: string }>(`
${PRELUDE}
t = grad.Tensor([1.0, 2.0, 3.0]).int()
{"dtype": str(t.dtype)}
`);
    expect(result.dtype).toBe("int32");
  });

  it("comparison ops produce bool tensors", async () => {
    const result = await target.run<{ dtype: string }>(`
${PRELUDE}
a = grad.Tensor([1.0, 2.0, 3.0])
b = grad.Tensor([1.0, 5.0, 3.0])
out = (a == b)
{"dtype": str(out.dtype)}
`);
    expect(result.dtype).toBe("bool");
  });

  it("cross_entropy_loss still accepts a long tensor as targets", async () => {
    const result = await target.run<{ loss_is_scalar: boolean }>(`
${PRELUDE}
import browsergrad_grad.functional as F
logits = grad.Tensor(np.random.randn(4, 3).astype(np.float32))
targets = grad.Tensor([0, 2, 1, 0], dtype='int64')
loss = F.cross_entropy_loss(logits, targets)
{"loss_is_scalar": loss.data.size == 1}
`);
    expect(result.loss_is_scalar).toBe(true);
  });
});
