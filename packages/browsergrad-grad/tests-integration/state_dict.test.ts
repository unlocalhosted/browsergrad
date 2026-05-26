/**
 * Pile A #5 — state_dict / load_state_dict / torch.save / torch.load.
 *
 * Independent oracle: round-trip identity — random init → snapshot → re-init
 * → load → outputs match exactly. End-to-end behavior, no internal coupling.
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

describe("state_dict / load_state_dict", () => {
  beforeAll(reset);

  it("state_dict returns a dict keyed by qualified parameter name", async () => {
    const keys = await target.run<string[]>(`
${PRELUDE}
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        return self.fc2(self.fc1(x))
net = Net()
sorted(list(net.state_dict().keys()))
`);
    expect(keys).toEqual(["fc1.bias", "fc1.weight", "fc2.bias", "fc2.weight"]);
  });

  it("state_dict values match the actual parameter data", async () => {
    const result = await target.run<{ w_match: boolean; b_match: boolean }>(`
${PRELUDE}
fc = nn.Linear(3, 5)
sd = fc.state_dict()
{
  "w_match": bool(np.array_equal(np.asarray(sd["weight"]), fc.weight.data)),
  "b_match": bool(np.array_equal(np.asarray(sd["bias"]), fc.bias.data)),
}
`);
    expect(result.w_match).toBe(true);
    expect(result.b_match).toBe(true);
  });

  it("round-trip: load_state_dict restores parameters and forward outputs match", async () => {
    const result = await target.run<{ same: boolean }>(`
${PRELUDE}
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        return self.fc2(self.fc1(x))

src = Net()
x = grad.Tensor(np.random.randn(3, 4).astype(np.float32))
y_before = src(x).data.copy()

# snapshot, then fully re-randomize destination
sd = src.state_dict()
dst = Net()
y_dst_initial = dst(x).data
assert not np.allclose(y_before, y_dst_initial), "test setup: dst must differ before load"

dst.load_state_dict(sd)
y_after = dst(x).data
{"same": bool(np.allclose(y_before, y_after, atol=1e-6))}
`);
    expect(result.same).toBe(true);
  });

  it("load_state_dict rejects mismatched keys", async () => {
    const errored = await target.run<boolean>(`
${PRELUDE}
fc = nn.Linear(2, 3)
_ok = False
try:
    fc.load_state_dict({"nonsense.weight": np.zeros((3, 2))})
except (KeyError, RuntimeError, ValueError):
    _ok = True
_ok
`);
    expect(errored).toBe(true);
  });

  it("load_state_dict rejects mismatched shapes", async () => {
    const errored = await target.run<boolean>(`
${PRELUDE}
fc = nn.Linear(2, 3)
_ok = False
try:
    fc.load_state_dict({"weight": np.zeros((4, 4)), "bias": np.zeros(3)})
except (RuntimeError, ValueError):
    _ok = True
_ok
`);
    expect(errored).toBe(true);
  });
});

describe("torch.save / torch.load", () => {
  beforeAll(reset);

  it("torch.save then torch.load round-trips a state_dict", async () => {
    const result = await target.run<{ same: boolean }>(`
${PRELUDE}
grad.install_torch_alias()
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

src = Net()
x = grad.Tensor(np.random.randn(3, 4).astype(np.float32))
y_before = src(x).data.copy()

torch.save(src.state_dict(), "/tmp/bg_test.pt")

dst = Net()
dst.load_state_dict(torch.load("/tmp/bg_test.pt"))
y_after = dst(x).data
{"same": bool(np.allclose(y_before, y_after, atol=1e-6))}
`);
    expect(result.same).toBe(true);
  });
});
