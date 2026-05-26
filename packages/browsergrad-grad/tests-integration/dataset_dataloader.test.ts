/**
 * Pile A #4 — Dataset + DataLoader (single-process).
 *
 * Independent oracle: hand-derived batch indices, computed in pure Python
 * (no DataLoader involved). Confirms behavior, not implementation.
 *
 * In-browser we are necessarily single-process — DataLoader(num_workers=0)
 * is the only supported mode. num_workers > 0 must raise (architecturally
 * impossible in a browser without a worker bridge).
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
from browsergrad_grad.utils.data import Dataset, DataLoader
import numpy as np
`;

describe("Dataset (TensorDataset-style behavior)", () => {
  beforeAll(reset);

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
});

describe("DataLoader basics (shuffle=False)", () => {
  beforeAll(reset);

  it("produces ceil(N/batch_size) batches with drop_last=False (default)", async () => {
    const result = await target.run<{ count: number; sizes: number[] }>(`
${PRELUDE}
class Idx(Dataset):
    def __len__(self): return 7
    def __getitem__(self, i): return float(i)
loader = DataLoader(Idx(), batch_size=3, shuffle=False)
batches = list(loader)
{"count": len(batches), "sizes": [int(np.asarray(b).size) for b in batches]}
`);
    expect(result.count).toBe(3);
    expect(result.sizes).toEqual([3, 3, 1]);
  });

  it("produces N // batch_size batches with drop_last=True", async () => {
    const result = await target.run<{ count: number; sizes: number[] }>(`
${PRELUDE}
class Idx(Dataset):
    def __len__(self): return 7
    def __getitem__(self, i): return float(i)
loader = DataLoader(Idx(), batch_size=3, shuffle=False, drop_last=True)
batches = list(loader)
{"count": len(batches), "sizes": [int(np.asarray(b).size) for b in batches]}
`);
    expect(result.count).toBe(2);
    expect(result.sizes).toEqual([3, 3]);
  });

  it("with shuffle=False, batches are in dataset order", async () => {
    const result = await target.run<number[][]>(`
${PRELUDE}
class Idx(Dataset):
    def __len__(self): return 6
    def __getitem__(self, i): return i
loader = DataLoader(Idx(), batch_size=2, shuffle=False)
[np.asarray(b).tolist() for b in loader]
`);
    expect(result).toEqual([[0, 1], [2, 3], [4, 5]]);
  });
});

describe("DataLoader: shuffle and tuple-batching", () => {
  beforeAll(reset);

  it("with shuffle=True, batch indices are a permutation of 0..N-1 (oracle: set-equality)", async () => {
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

  it("when __getitem__ returns a tuple, the loader yields a tuple of stacked batches", async () => {
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
});

describe("DataLoader limits in browser context", () => {
  beforeAll(reset);

  it("num_workers > 0 raises (no worker pool inside Pyodide)", async () => {
    const errored = await target.run<boolean>(`
${PRELUDE}
class Idx(Dataset):
    def __len__(self): return 4
    def __getitem__(self, i): return i
_ok = False
try:
    DataLoader(Idx(), batch_size=2, num_workers=2)
except (NotImplementedError, ValueError):
    _ok = True
_ok
`);
    expect(errored).toBe(true);
  });
});
