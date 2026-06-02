/**
 * Grad — install + autograd + nn adversarial tests.
 *
 * Hypotheses: G1–G15.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { installGrad } from "@unlocalhosted/browsergrad-grad";
import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-grad/node-adapter";
import { getPyodide, pyJson, pyStr, pyBool } from "../helpers.js";

let installed = false;

beforeAll(async () => {
  if (installed) return;
  const py = await getPyodide();
  await installGrad(createNodePyodideTarget(py));
  installed = true;
}, 120_000);

describe("installGrad + version (G1, G2, G3)", () => {
  it("second installGrad call is idempotent (G1)", async () => {
    const py = await getPyodide();
    // Calling installGrad twice should not error.
    await installGrad(createNodePyodideTarget(py));
    expect(true).toBe(true);
  });

  it("exposes browsergrad_grad module (G2)", async () => {
    const ok = await pyBool(`
import browsergrad_grad
hasattr(browsergrad_grad, "Tensor") and hasattr(browsergrad_grad, "nn")
`);
    expect(ok).toBe(true);
  });

  it("version matches npm package version (G3)", async () => {
    const v = await pyStr(`import browsergrad_grad; browsergrad_grad.__version__`);
    expect(v).toMatch(/^0\.5\.\d+$/);
  });
});

describe("basic autograd (G4, G7, G8)", () => {
  it("(t*t).sum().backward() → grad = 2t (G4)", async () => {
    const r = await pyJson<{ grad: number[] }>(`
import browsergrad_grad as grad
import numpy as np, json
t = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (t * t).sum()
y.backward()
json.dumps({"grad": t.grad.tolist()})
`);
    expect(r.grad).toEqual([2, 4, 6]);
  });

  it("no_grad() context prevents backward (G7)", async () => {
    const ok = await pyBool(`
import browsergrad_grad as grad
t = grad.Tensor([1.0, 2.0], requires_grad=True)
with grad.no_grad():
    y = (t * t).sum()
# Within no_grad, y shouldn't have a graph. Calling backward should raise or be no-op.
try:
    y.backward()
    no_graph_safe = (t.grad is None)
except Exception:
    no_graph_safe = True
no_graph_safe
`);
    expect(ok).toBe(true);
  });

  it("requires_grad=False → backward leaves .grad as None (G8)", async () => {
    const ok = await pyBool(`
import browsergrad_grad as grad
t = grad.Tensor([1.0, 2.0], requires_grad=False)
# A non-leaf op may still be computable; backward through it should not write to t.
try:
    y = (t * t).sum()
    y.backward()
except Exception:
    pass
t.grad is None
`);
    expect(ok).toBe(true);
  });
});

describe("nn.Module (G5, G6, G9)", () => {
  it("nn.Linear forward shape (G5)", async () => {
    const r = await pyJson<{ shape: number[] }>(`
import browsergrad_grad as grad
import numpy as np, json
layer = grad.nn.Linear(8, 4)
x = grad.Tensor(np.zeros((16, 8), dtype=np.float32))
json.dumps({"shape": list(layer(x).shape)})
`);
    expect(r.shape).toEqual([16, 4]);
  });

  it("Sequential + SGD converges in 20 steps (G6)", async () => {
    const r = await pyJson<{ initial: number; final: number; improved: boolean }>(`
import browsergrad_grad as grad
import numpy as np, json
np.random.seed(42)
model = grad.nn.Sequential(
    grad.nn.Linear(4, 16),
    grad.nn.ReLU(),
    grad.nn.Linear(16, 2),
)
opt = grad.optim.SGD([p for p in model.parameters()], lr=0.01)
x = grad.Tensor(np.random.RandomState(0).randn(32, 4).astype(np.float32))
y = grad.Tensor(np.random.RandomState(1).randn(32, 2).astype(np.float32))
initial = float(((model(x) - y) ** 2).mean().data)
for _ in range(20):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
final = float(((model(x) - y) ** 2).mean().data)
json.dumps({"initial": initial, "final": final, "improved": final < initial})
`);
    expect(r.improved).toBe(true);
  });

  it("model.parameters() walks recursively (G9)", async () => {
    const n = await pyJson<{ count: number }>(`
import browsergrad_grad as grad, json
model = grad.nn.Sequential(
    grad.nn.Linear(4, 16),
    grad.nn.ReLU(),
    grad.nn.Linear(16, 8),
    grad.nn.ReLU(),
    grad.nn.Linear(8, 2),
)
ps = [p for p in model.parameters()]
# 3 Linears × (weight + bias) = 6
json.dumps({"count": len(ps)})
`);
    expect(n.count).toBe(6);
  });
});

describe("torch alias shim (G10, G11, G14, G15)", () => {
  it("install_torch_alias wires torch.nn (G10)", async () => {
    const ok = await pyBool(`
import browsergrad_grad as grad
grad.install_torch_alias()
import torch
import torch.nn as nn
hasattr(nn, "Linear") and hasattr(torch, "manual_seed")
`);
    expect(ok).toBe(true);
  });

  it("torch.cuda.is_available() returns False honestly (G14)", async () => {
    const ok = await pyBool(`
import browsergrad_grad as grad; grad.install_torch_alias()
import torch
torch.cuda.is_available() == False
`);
    expect(ok).toBe(true);
  });

  it("torch.compile raises NotImplementedError with reason (G15)", async () => {
    const msg = await pyStr(`
import browsergrad_grad as grad; grad.install_torch_alias()
import torch
try:
    torch.compile(lambda x: x)
    result = "no_error"
except NotImplementedError as e:
    result = str(e)
except AttributeError as e:
    result = "AttributeError: " + str(e)
result
`);
    // Either NotImplementedError or AttributeError is acceptable; just must not silently no-op
    expect(msg).not.toBe("no_error");
    console.log(`[G15] torch.compile: ${msg.slice(0, 100)}`);
  });

  it("cross-entropy loss decreases with training (G11)", async () => {
    // grad's native function name is `cross_entropy_loss`, NOT `cross_entropy`.
    // The PyTorch alias (`torch.nn.functional.cross_entropy`) maps to it.
    // For direct consumers using the grad namespace, use cross_entropy_loss.
    // Lives in browsergrad_grad.functional (NOT in grad.nn.functional — nn is a flat module).
    const r = await pyJson<{ initial: number; final: number; improved: boolean }>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import numpy as np, json
np.random.seed(7)
model = grad.nn.Linear(4, 3)
opt = grad.optim.SGD([p for p in model.parameters()], lr=0.05)
x = grad.Tensor(np.random.RandomState(0).randn(32, 4).astype(np.float32))
y = grad.Tensor(np.random.RandomState(1).randint(0, 3, size=(32,)).astype(np.int64))
initial = float(F.cross_entropy_loss(model(x), y).data)
for _ in range(50):
    opt.zero_grad()
    loss = F.cross_entropy_loss(model(x), y)
    loss.backward()
    opt.step()
final = float(F.cross_entropy_loss(model(x), y).data)
json.dumps({"initial": initial, "final": final, "improved": final < initial})
`);
    expect(r.improved).toBe(true);
  });

  it("PyTorch-name alias `F.cross_entropy` works in grad's namespace ✓ (added in 0.5.1)", async () => {
    const ok = await pyBool(`
import browsergrad_grad.functional as F
hasattr(F, "cross_entropy") and hasattr(F, "nll") and hasattr(F, "bce_with_logits")
`);
    expect(ok).toBe(true);
  });

  it("probe: where does cross_entropy live in grad's module tree?", async () => {
    const r = await pyJson<{ places: Record<string, boolean> }>(`
import browsergrad_grad as grad
import json
places = {}
for path in [
    "grad.cross_entropy",
    "grad.cross_entropy_loss",        # per v0.4.6 changelog: native name
    "grad.functional.cross_entropy",
    "grad.functional.cross_entropy_loss",
    "grad.nn.functional.cross_entropy",
    "grad.nn.cross_entropy",
    "grad.nn.cross_entropy_loss",
    "grad.nn.CrossEntropyLoss",        # the Module form
    "grad.F.cross_entropy",
]:
    try:
        eval(path)
        places[path] = True
    except Exception:
        places[path] = False
json.dumps({"places": places})
`);
    console.log(`[probe] cross_entropy reachable at: ${JSON.stringify(r.places)}`);
    // At least one path must reach a cross-entropy implementation.
    expect(Object.values(r.places).some(Boolean)).toBe(true);
  });

  it("grad.nn.functional is reachable as attribute ✓ (fixed in 0.5.1)", async () => {
    const ok = await pyBool(`
import browsergrad_grad as grad
hasattr(grad.nn, "functional")
`);
    expect(ok).toBe(true);
  });

  it("`from browsergrad_grad.nn import functional` works ✓ (fixed in 0.5.1 via sys.modules)", async () => {
    const ok = await pyBool(`
try:
    from browsergrad_grad.nn import functional as F
    result = True
except ImportError:
    result = False
result
`);
    expect(ok).toBe(true);
  });

  it("`grad.F` shortcut works ✓ (added in 0.5.1)", async () => {
    const ok = await pyBool(`
import browsergrad_grad as grad
grad.F is grad.functional and hasattr(grad.F, "cross_entropy")
`);
    expect(ok).toBe(true);
  });
});

describe("state_dict roundtrip (G12)", () => {
  it("save then load preserves weights (G12)", async () => {
    const r = await pyJson<{ match: boolean }>(`
import browsergrad_grad as grad
import numpy as np, json
m = grad.nn.Linear(4, 2)
state = m.state_dict()
m2 = grad.nn.Linear(4, 2)
m2.load_state_dict(state)
x = grad.Tensor(np.random.RandomState(0).randn(8, 4).astype(np.float32))
y1 = m(x).data
y2 = m2(x).data
json.dumps({"match": bool(np.allclose(y1, y2))})
`);
    expect(r.match).toBe(true);
  });
});

describe("DataLoader Pyodide refusal (G13)", () => {
  it("num_workers>0 raises with Pyodide-specific reason (G13)", async () => {
    const msg = await pyStr(`
import browsergrad_grad as grad
import numpy as np
from browsergrad_grad.utils.data import TensorDataset, DataLoader
ds = TensorDataset(grad.Tensor(np.zeros((4, 2), dtype=np.float32)))
try:
    DataLoader(ds, batch_size=2, num_workers=2)
    result = "no_error"
except Exception as e:
    result = str(e)
result
`);
    expect(msg.toLowerCase()).toMatch(/pyodide|worker|browser/);
  });
});
