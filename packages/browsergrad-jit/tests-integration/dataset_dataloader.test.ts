/**
 * utils.data — Dataset + DataLoader (single-process).
 *
 * Mirrors browsergrad-grad's browser-safe subset, with JIT-specific coverage
 * that TensorProxy samples collate back into TensorProxy batches.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

let target: Awaited<ReturnType<typeof getJitTarget>>;

beforeAll(async () => {
  target = await getJitTarget();
}, 120_000);

beforeEach(async () => {
  await clearNamespace(target);
});

const PRELUDE = `
import browsergrad_jit as bg
from browsergrad_jit.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
`;

describe("browsergrad_jit.utils.data", () => {
  it("user can subclass Dataset and __len__ / __getitem__ work", async () => {
    const result = await target.run<{ len: number; first: number[]; last: number[] }>(`
${PRELUDE}
class XY(Dataset):
    def __init__(self):
        self.x = np.arange(20).reshape(10, 2).astype(np.float32)
        self.y = np.arange(10).astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.x[i], int(self.y[i])
ds = XY()
{"len": len(ds), "first": ds[0][0].tolist(), "last": ds[9][0].tolist()}
`);
    expect(result.len).toBe(10);
    expect(result.first).toEqual([0, 1]);
    expect(result.last).toEqual([18, 19]);
  });

  it("produces ordered batches with drop_last=False", async () => {
    const result = await target.run<{ count: number; batches: number[][] }>(`
${PRELUDE}
class Idx(Dataset):
    def __len__(self): return 7
    def __getitem__(self, i): return i
loader = DataLoader(Idx(), batch_size=3, shuffle=False)
batches = [np.asarray(b).tolist() for b in loader]
{"count": len(batches), "batches": batches}
`);
    expect(result.count).toBe(3);
    expect(result.batches).toEqual([[0, 1, 2], [3, 4, 5], [6]]);
  });

  it("honors drop_last=True", async () => {
    const result = await target.run<{ count: number; batches: number[][] }>(`
${PRELUDE}
class Idx(Dataset):
    def __len__(self): return 7
    def __getitem__(self, i): return i
loader = DataLoader(Idx(), batch_size=3, shuffle=False, drop_last=True)
batches = [np.asarray(b).tolist() for b in loader]
{"count": len(batches), "batches": batches}
`);
    expect(result.count).toBe(2);
    expect(result.batches).toEqual([[0, 1, 2], [3, 4, 5]]);
  });

  it("shuffle=True yields a permutation across an epoch", async () => {
    const result = await target.run<{ unique_sorted: number[]; total: number }>(`
${PRELUDE}
class Idx(Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return i
np.random.seed(42)
loader = DataLoader(Idx(), batch_size=3, shuffle=True)
seen = []
for b in loader:
    seen.extend(list(np.asarray(b).flatten()))
{"unique_sorted": sorted(set(int(s) for s in seen)), "total": len(seen)}
`);
    expect(result.unique_sorted).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(result.total).toBe(8);
  });

  it("tuple samples collate column-wise", async () => {
    const result = await target.run<{ x_shape: number[]; y_shape: number[]; x0: number[] }>(`
${PRELUDE}
class XY(Dataset):
    def __init__(self):
        self.x = np.arange(20).reshape(10, 2).astype(np.float32)
        self.y = np.arange(10).astype(np.int64)
    def __len__(self): return 10
    def __getitem__(self, i):
        return self.x[i], int(self.y[i])
loader = DataLoader(XY(), batch_size=4, shuffle=False)
xb, yb = next(iter(loader))
xb_arr = np.asarray(xb)
yb_arr = np.asarray(yb)
{"x_shape": list(xb_arr.shape), "y_shape": list(yb_arr.shape), "x0": xb_arr[0].tolist()}
`);
    expect(result.x_shape).toEqual([4, 2]);
    expect(result.y_shape).toEqual([4]);
    expect(result.x0).toEqual([0, 1]);
  });

  it("TensorDataset wraps TensorProxy inputs and yields TensorProxy batches", async () => {
    const result = await target.run<{
      x_is_proxy: boolean;
      y_is_proxy: boolean;
      x_shape: number[];
      y_shape: number[];
      x_values: number[][];
      y_values: number[];
    }>(`
${PRELUDE}
x = bg.from_numpy(np.arange(12, dtype=np.float32).reshape(6, 2))
y = bg.from_numpy(np.arange(6, dtype=np.int64))
loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
xb, yb = next(iter(loader))
{
    "x_is_proxy": isinstance(xb, bg.TensorProxy),
    "y_is_proxy": isinstance(yb, bg.TensorProxy),
    "x_shape": list(xb.shape),
    "y_shape": list(yb.shape),
    "x_values": xb.numpy().tolist(),
    "y_values": yb.numpy().tolist(),
}
`);
    expect(result.x_is_proxy).toBe(true);
    expect(result.y_is_proxy).toBe(true);
    expect(result.x_shape).toEqual([4, 2]);
    expect(result.y_shape).toEqual([4]);
    expect(result.x_values).toEqual([
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7],
    ]);
    expect(result.y_values).toEqual([0, 1, 2, 3]);
  });

  it("num_workers > 0 raises with a browser/Pyodide-specific message", async () => {
    const message = await target.run<string>(`
${PRELUDE}
class Idx(Dataset):
    def __len__(self): return 4
    def __getitem__(self, i): return i
try:
    DataLoader(Idx(), batch_size=2, num_workers=2)
    result = "no_error"
except NotImplementedError as e:
    result = str(e)
result
`);
    expect(message).toMatch(/num_workers > 0/);
    expect(message.toLowerCase()).toMatch(/pyodide|browser/);
  });
});

describe("torch.utils.data alias", () => {
  it("resolves Dataset, DataLoader, and TensorDataset after install_torch_alias", async () => {
    const result = await target.run<{
      module_registered: boolean;
      utils_has_data: boolean;
      batch_shape: number[];
      batch_values: number[][];
    }>(`
import browsergrad_jit as bg
import numpy as np
import sys
bg.install_torch_alias()
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
x = bg.from_numpy(np.arange(10, dtype=np.float32).reshape(5, 2))
ds = TensorDataset(x)
batch, = next(iter(DataLoader(ds, batch_size=2)))
{
    "module_registered": "torch.utils.data" in sys.modules,
    "utils_has_data": hasattr(torch.utils, "data"),
    "batch_shape": list(batch.shape),
    "batch_values": batch.numpy().tolist(),
}
`);
    expect(result.module_registered).toBe(true);
    expect(result.utils_has_data).toBe(true);
    expect(result.batch_shape).toEqual([2, 2]);
    expect(result.batch_values).toEqual([
      [0, 1],
      [2, 3],
    ]);
  });
});
