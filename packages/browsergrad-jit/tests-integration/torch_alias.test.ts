/**
 * install_torch_alias / uninstall_torch_alias integration tests.
 *
 * Verifies the owner-token protocol per the PRD-005 critique:
 *  - install() is idempotent (re-installing by the same owner is a no-op)
 *  - the torch namespace gets browsergrad_jit's full public surface
 *  - conflict detection refuses to overwrite another owner without force=True
 *  - uninstall() releases cleanly and is safe to call when no alias is set
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("install_torch_alias", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    // Always start each test from a clean slate.
    await target.run(`
import browsergrad_jit
browsergrad_jit.uninstall_torch_alias()
`);
  });

  it("registers torch / torch.nn / torch.nn.functional / torch.optim / torch.utils", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      hasTorch: boolean;
      hasNn: boolean;
      hasFunc: boolean;
      hasOptim: boolean;
      hasUtils: boolean;
      hasCheckpoint: boolean;
      hasData: boolean;
      hasAmp: boolean;
      hasFuncTransforms: boolean;
      owner: string;
    }>(`
import browsergrad_jit
browsergrad_jit.install_torch_alias()
import sys
result = {
    "hasTorch": "torch" in sys.modules,
    "hasNn": "torch.nn" in sys.modules,
    "hasFunc": "torch.nn.functional" in sys.modules,
    "hasOptim": "torch.optim" in sys.modules,
    "hasUtils": "torch.utils" in sys.modules,
    "hasCheckpoint": "torch.utils.checkpoint" in sys.modules,
    "hasData": "torch.utils.data" in sys.modules,
    "hasAmp": "torch.amp" in sys.modules,
    "hasFuncTransforms": "torch.func" in sys.modules,
    "owner": getattr(sys.modules["torch"], "__bg_owner__", "<none>"),
}
result
`);
    expect(result.hasTorch).toBe(true);
    expect(result.hasNn).toBe(true);
    expect(result.hasFunc).toBe(true);
    expect(result.hasOptim).toBe(true);
    expect(result.hasUtils).toBe(true);
    expect(result.hasCheckpoint).toBe(true);
    expect(result.hasData).toBe(true);
    expect(result.hasAmp).toBe(true);
    expect(result.hasFuncTransforms).toBe(true);
    expect(result.owner).toBe("browsergrad_jit");
  });

  it("import torch resolves to browsergrad_jit", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ ver: string; sumValue: number }>(`
import browsergrad_jit
browsergrad_jit.install_torch_alias()
import torch
import torch.nn as nn
import torch.nn.functional as F
t = torch.tensor([1.0, 2.0, 3.0, 4.0])
{"ver": torch.__version__, "sumValue": float(t.sum().item())}
`);
    expect(result.ver).toMatch(/^0\.8\.\d+$/);
    expect(result.sumValue).toBe(10.0);
  });

  it("covers curriculum compatibility APIs from issue #5", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      caps: Record<string, boolean>;
      sigmoid: number[];
      detachedRequiresGrad: boolean;
      detachBlocksGrad: boolean;
      cloneGrad: number[];
      cloneDtype: string;
      noGradRequiresGrad: boolean;
      noGradGradFn: string | null;
      nestedNoGradRequiresGrad: boolean;
      afterNoGradRequiresGrad: boolean;
      bnTrainMean: number;
      bnEvalMean: number;
      bnGradShapes: number[][];
      bn3dShape: number[];
      bnStateKeys: string[];
      loadedKeys: string[];
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch, torch.nn as nn
import numpy as np

caps = {
  "nn.Dropout": hasattr(nn, "Dropout"),
  "nn.BatchNorm1d": hasattr(nn, "BatchNorm1d"),
  "torch.no_grad": hasattr(torch, "no_grad"),
  "torch.inference_mode": hasattr(torch, "inference_mode"),
  "torch.save": hasattr(torch, "save"),
  "torch.load": hasattr(torch, "load"),
  "torch.sigmoid": hasattr(torch, "sigmoid"),
}

t = torch.tensor([1.0, -1.0], requires_grad=True)
caps["Tensor.clone"] = hasattr(t, "clone")
caps["Tensor.detach"] = hasattr(t, "detach")
caps["Tensor.is_leaf"] = hasattr(t, "is_leaf")
caps["Tensor.grad_fn"] = hasattr(t, "grad_fn")

s = torch.sigmoid(torch.tensor([0.0, 2.0])).numpy().round(6).tolist()
detached = t.detach()
clone_loss = t.clone().sum()
clone_loss.backward()
cloneDtype = torch.tensor([1, 2], dtype=torch.int64).clone().dtype
detachBlocksGrad = not detached.requires_grad and detached.grad_fn is None and detached.is_leaf

with torch.no_grad():
    with torch.inference_mode():
        no_grad_y = t * 3.0
nestedNoGradRequiresGrad = no_grad_y.requires_grad
after_no_grad_y = t * 4.0
afterNoGradRequiresGrad = after_no_grad_y.requires_grad

bn = nn.BatchNorm1d(2, affine=True, momentum=1.0)
x = torch.tensor([[1.0, 2.0], [3.0, 6.0]], requires_grad=True)
y_train = bn(x)
train_mean = float(y_train.numpy().mean())
(y_train * y_train).mean().backward()
bn_grad_shapes = [list(x.grad.shape), list(bn.weight.grad.shape), list(bn.bias.grad.shape)]
bn.eval()
y_eval = bn(x)
eval_mean = float(y_eval.numpy().mean())
bn3 = nn.BatchNorm1d(2, affine=False)
y3 = bn3(torch.tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3)))

bn_state = bn.state_dict()

torch.save({"weight": np.asarray([1.0, 2.0], dtype=np.float32)}, "/tmp/bg_jit_state.pt")
loaded = torch.load("/tmp/bg_jit_state.pt")

{
  "caps": caps,
  "sigmoid": s,
  "detachedRequiresGrad": detached.requires_grad,
  "detachBlocksGrad": detachBlocksGrad,
  "cloneGrad": t.grad.numpy().tolist(),
  "cloneDtype": cloneDtype,
  "noGradRequiresGrad": no_grad_y.requires_grad,
  "noGradGradFn": no_grad_y.grad_fn,
  "nestedNoGradRequiresGrad": nestedNoGradRequiresGrad,
  "afterNoGradRequiresGrad": afterNoGradRequiresGrad,
  "bnTrainMean": train_mean,
  "bnEvalMean": eval_mean,
  "bnGradShapes": bn_grad_shapes,
  "bn3dShape": list(y3.shape),
  "bnStateKeys": sorted(bn_state.keys()),
  "loadedKeys": sorted(loaded.keys()),
}
`);
    expect(result.caps).toEqual({
      "nn.Dropout": true,
      "nn.BatchNorm1d": true,
      "torch.no_grad": true,
      "torch.inference_mode": true,
      "torch.save": true,
      "torch.load": true,
      "torch.sigmoid": true,
      "Tensor.clone": true,
      "Tensor.detach": true,
      "Tensor.is_leaf": true,
      "Tensor.grad_fn": true,
    });
    expect(result.sigmoid[0]).toBe(0.5);
    expect(result.sigmoid[1]).toBeCloseTo(0.880797, 6);
    expect(result.detachedRequiresGrad).toBe(false);
    expect(result.detachBlocksGrad).toBe(true);
    expect(result.cloneGrad).toEqual([1, 1]);
    expect(result.cloneDtype).toBe("int64");
    expect(result.noGradRequiresGrad).toBe(false);
    expect(result.noGradGradFn).toBeUndefined();
    expect(result.nestedNoGradRequiresGrad).toBe(false);
    expect(result.afterNoGradRequiresGrad).toBe(true);
    expect(Math.abs(result.bnTrainMean)).toBeLessThan(1e-6);
    expect(result.bnEvalMean).toBeCloseTo(0, 5);
    expect(result.bnGradShapes).toEqual([[2, 2], [2], [2]]);
    expect(result.bn3dShape).toEqual([2, 2, 3]);
    expect(result.bnStateKeys).toEqual(["bias", "running_mean", "running_var", "weight"]);
    expect(result.loadedKeys).toEqual(["weight"]);
  });

  it("is idempotent — re-installing returns cleanly", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit
browsergrad_jit.install_torch_alias()
browsergrad_jit.install_torch_alias()
browsergrad_jit.install_torch_alias()
import sys
"torch" in sys.modules
`);
    expect(ok).toBe(true);
  });

  it("uninstall removes the alias and registered submodules", async () => {
    const target = await getJitTarget();
    const states = await target.run<{
      after_install: boolean;
      after_uninstall: boolean;
      stale_submodules: string[];
    }>(`
import browsergrad_jit
import sys
browsergrad_jit.install_torch_alias()
after_install = "torch" in sys.modules
browsergrad_jit.uninstall_torch_alias()
browsergrad_jit.uninstall_torch_alias()  # double-uninstall safe
after_uninstall = "torch" in sys.modules
stale_submodules = sorted(k for k in sys.modules if k == "torch" or k.startswith("torch."))
{
    "after_install": after_install,
    "after_uninstall": after_uninstall,
    "stale_submodules": stale_submodules,
}
`);
    expect(states.after_install).toBe(true);
    expect(states.after_uninstall).toBe(false);
    expect(states.stale_submodules).toEqual([]);
  });

  it("refuses to shadow a foreign owner without force=True", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit
import sys
import types
# Plant a fake foreign torch ownership marker.
fake = types.ModuleType("torch")
fake.__bg_owner__ = "some_other_package"
sys.modules["torch"] = fake
try:
    browsergrad_jit.install_torch_alias()
    result = "no_error"
except browsergrad_jit.TorchAliasConflict as e:
    result = str(e)
finally:
    sys.modules.pop("torch", None)
result
`);
    expect(err).toMatch(/some_other_package/);
    expect(err).toMatch(/force=True/);
  });

  it("force=True overrides the conflict for tests", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit
import sys
import types
fake = types.ModuleType("torch")
fake.__bg_owner__ = "some_other_package"
sys.modules["torch"] = fake
browsergrad_jit.install_torch_alias(force=True)
ok = getattr(sys.modules["torch"], "__bg_owner__", None) == "browsergrad_jit"
sys.modules.pop("torch", None)
ok
`);
    expect(ok).toBe(true);
  });
});
