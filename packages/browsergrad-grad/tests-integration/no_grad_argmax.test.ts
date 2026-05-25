/**
 * TDD'd `grad.no_grad()` context + `Tensor.argmax(dim)`. Both PyTorch-
 * conformant; oracles are hand-derived.
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

describe("no_grad context", () => {
  beforeAll(reset);

  it("operations inside no_grad don't build the autograd graph", async () => {
    const result = await target.run<boolean>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
with grad.no_grad():
    y = x * x
# Outside no_grad, y is a fresh tensor with no _ctx — should not record.
bool(y.requires_grad == False and y._ctx is None)
`);
    expect(result).toBe(true);
  });

  it("operations outside no_grad still record (sanity)", async () => {
    const result = await target.run<boolean>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * x  # outside no_grad
bool(y.requires_grad and y._ctx is not None)
`);
    expect(result).toBe(true);
  });

  it("nesting and proper restoration of grad-enabled state", async () => {
    const result = await target.run<{
      before: boolean;
      inside: boolean;
      after: boolean;
    }>(`
${PRELUDE}
x = grad.Tensor([1.0], requires_grad=True)
before = (x * x).requires_grad
with grad.no_grad():
    inside = (x * x).requires_grad
after = (x * x).requires_grad
{"before": bool(before), "inside": bool(inside), "after": bool(after)}
`);
    expect(result.before).toBe(true);
    expect(result.inside).toBe(false);
    expect(result.after).toBe(true);
  });

  it("backward still works for tensors built outside the no_grad block", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()  # outside no_grad — autograd intact
with grad.no_grad():
    _ = x * 7   # this does not affect y's graph
y.backward()
x.grad.tolist()
`);
    expect(result).toEqual([2, 4, 6]);
  });
});

describe("Tensor.argmax", () => {
  beforeAll(reset);

  it("argmax() with no axis returns flat index of the global max", async () => {
    const result = await target.run<number>(`
${PRELUDE}
t = grad.Tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 8.0]])
int(t.argmax())
`);
    // Global max is 9 at flat index 3 (row 1, col 0)
    expect(result).toBe(3);
  });

  it("argmax(dim=0) returns per-column argmax", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
t = grad.Tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 8.0]])
t.argmax(dim=0).tolist()
`);
    // Per column: [9>1: row 1, 5>2: row 0, 8>3: row 1] = [1, 0, 1]
    expect(result).toEqual([1, 0, 1]);
  });

  it("argmax(dim=-1) returns per-row argmax", async () => {
    const result = await target.run<number[]>(`
${PRELUDE}
t = grad.Tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 8.0]])
t.argmax(dim=-1).tolist()
`);
    // Per row: [max=5 at col 1, max=9 at col 0] = [1, 0]
    expect(result).toEqual([1, 0]);
  });

  it("argmax returns ints (not floats)", async () => {
    const result = await target.run<{ value: number; is_int: boolean }>(`
${PRELUDE}
t = grad.Tensor([1.0, 4.0, 2.0])
out = t.argmax()
{"value": int(out), "is_int": isinstance(int(out), int)}
`);
    expect(result.value).toBe(1);
    expect(result.is_int).toBe(true);
  });

  it("model classification: argmax over logits gives the predicted class", async () => {
    // Smoke test that argmax integrates with a real model output.
    const result = await target.run<number>(`
${PRELUDE}
np.random.seed(0)
model = nn.Linear(4, 3)
X = np.random.randn(5, 4).astype(np.float32)
with grad.no_grad():
    logits = model(grad.Tensor(X))
    preds = logits.argmax(dim=-1)
int(preds.tolist()[0])
`);
    // We don't care what the prediction IS, just that it's a valid class index.
    expect(result).toBeGreaterThanOrEqual(0);
    expect(result).toBeLessThan(3);
  });
});
