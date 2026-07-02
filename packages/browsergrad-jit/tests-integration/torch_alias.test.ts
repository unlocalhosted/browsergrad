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

  it("supports PyTorch reduction aliases on Tensor and top-level torch", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      methodSum: number[][];
      methodMean: number[][];
      methodArgmax: number[][];
      torchSum: number[][];
      torchMean: number[][];
      torchArgmax: number[][];
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch

x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], requires_grad=True)
{
  "methodSum": x.sum(dim=1, keepdim=True).numpy().tolist(),
  "methodMean": x.mean(dim=0, keepdim=True).numpy().tolist(),
  "methodArgmax": x.argmax(dim=1, keepdim=True).numpy().tolist(),
  "torchSum": torch.sum(x, dim=1, keepdim=True).numpy().tolist(),
  "torchMean": torch.mean(x, dim=0, keepdim=True).numpy().tolist(),
  "torchArgmax": torch.argmax(x, dim=1, keepdim=True).numpy().tolist(),
}
`);
    expect(result.methodSum).toEqual([[6], [9]]);
    expect(result.methodMean).toEqual([[2.5, 1.5, 3.5]]);
    expect(result.methodArgmax).toEqual([[1], [2]]);
    expect(result.torchSum).toEqual(result.methodSum);
    expect(result.torchMean).toEqual(result.methodMean);
    expect(result.torchArgmax).toEqual(result.methodArgmax);
  });

  it("supports top-level PyTorch math and softmax aliases", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      matmul: number[][];
      mm: number[][];
      bmmShape: number[];
      expFirst: number;
      logFirst: number;
      tanhZero: number;
      softmaxRows: number[];
      logSoftmaxRows: number[];
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
import numpy as np

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
batched_a = torch.tensor(np.ones((2, 2, 3), dtype=np.float32))
batched_b = torch.tensor(np.ones((2, 3, 4), dtype=np.float32))
x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]])
sm = torch.softmax(x, dim=-1).numpy()
lsm = torch.log_softmax(x, dim=-1).numpy()
{
  "matmul": torch.matmul(a, b).numpy().tolist(),
  "mm": torch.mm(a, b).numpy().tolist(),
  "bmmShape": list(torch.bmm(batched_a, batched_b).shape),
  "expFirst": float(torch.exp(torch.tensor([1.0])).item()),
  "logFirst": float(torch.log(torch.tensor([1.0])).item()),
  "tanhZero": float(torch.tanh(torch.tensor([0.0])).item()),
  "softmaxRows": sm.sum(axis=-1).round(6).tolist(),
  "logSoftmaxRows": np.exp(lsm).sum(axis=-1).round(6).tolist(),
}
`);
    expect(result.matmul).toEqual([[19, 22], [43, 50]]);
    expect(result.mm).toEqual(result.matmul);
    expect(result.bmmShape).toEqual([2, 2, 4]);
    expect(result.expFirst).toBeCloseTo(Math.E, 5);
    expect(result.logFirst).toBe(0);
    expect(result.tanhZero).toBe(0);
    expect(result.softmaxRows).toEqual([1, 1]);
    expect(result.logSoftmaxRows).toEqual([1, 1]);
  });

  it("supports PyTorch shape and multi-tensor composition aliases", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      flattened: number[][];
      topFlattenShape: number[];
      unsqueezedShape: number[];
      squeezedShape: number[];
      transposed: number[][];
      permutedShape: number[];
      catRows: number[][];
      stackShape: number[];
      catGradA: number[][];
      catGradB: number[][];
      stackGradA: number[][];
      stackGradB: number[][];
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch

x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
flat_method = x.flatten(start_dim=1)
flat_top = torch.flatten(x, start_dim=0, end_dim=1)
unsqueezed = torch.unsqueeze(flat_method, dim=-1)
squeezed = unsqueezed.squeeze(dim=-1)
transposed = torch.transpose(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 1)
permuted = torch.permute(x, 2, 0, 1)

a = torch.tensor([[1.0, 2.0]], requires_grad=True)
b = torch.tensor([[3.0, 4.0]], requires_grad=True)
cat_out = torch.cat([a, b], dim=0)
stack_out = torch.stack([a, b], dim=1)
cat_loss = cat_out.sum()
cat_loss.backward()
cat_grad_a = a.grad.numpy().copy().tolist()
cat_grad_b = b.grad.numpy().copy().tolist()
stack_loss = stack_out.mean()
stack_loss.backward()

{
  "flattened": flat_method.numpy().tolist(),
  "topFlattenShape": list(flat_top.shape),
  "unsqueezedShape": list(unsqueezed.shape),
  "squeezedShape": list(squeezed.shape),
  "transposed": transposed.numpy().tolist(),
  "permutedShape": list(permuted.shape),
  "catRows": cat_out.numpy().tolist(),
  "stackShape": list(stack_out.shape),
  "catGradA": cat_grad_a,
  "catGradB": cat_grad_b,
  "stackGradA": a.grad.numpy().tolist(),
  "stackGradB": b.grad.numpy().tolist(),
}
`);
    expect(result.flattened).toEqual([[1, 2, 3, 4]]);
    expect(result.topFlattenShape).toEqual([2, 2]);
    expect(result.unsqueezedShape).toEqual([1, 4, 1]);
    expect(result.squeezedShape).toEqual([1, 4]);
    expect(result.transposed).toEqual([
      [1, 3],
      [2, 4],
    ]);
    expect(result.permutedShape).toEqual([2, 1, 2]);
    expect(result.catRows).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(result.stackShape).toEqual([1, 2, 2]);
    expect(result.catGradA).toEqual([[1, 1]]);
    expect(result.catGradB).toEqual([[1, 1]]);
    expect(result.stackGradA).toEqual([[1.25, 1.25]]);
    expect(result.stackGradB).toEqual([[1.25, 1.25]]);
  });

  it("supports functional one_hot and stable BCE-with-logits loss", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      oneHot: number[][];
      tensorOneHot: number[][];
      loss: number;
      sumLoss: number;
      noneLoss: number[];
      grad: number[];
      moduleLoss: number;
      moduleGradFinite: boolean;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

one_hot = F.one_hot(np.array([0, 2, 1], dtype=np.int64), num_classes=3)
tensor_one_hot = F.one_hot(torch.tensor([2, 0], dtype=torch.int64), num_classes=3)

logits = torch.tensor([0.0, 100.0, -100.0], requires_grad=True)
targets = torch.tensor([0.0, 1.0, 0.0])
loss = F.binary_cross_entropy_with_logits(logits, targets)
loss.backward()

sum_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
none_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

module_logits = torch.tensor([1000.0, -1000.0], requires_grad=True)
module_targets = torch.tensor([1.0, 0.0])
module_loss = nn.BCEWithLogitsLoss()(module_logits, module_targets)
module_loss.backward()

{
  "oneHot": one_hot.numpy().tolist(),
  "tensorOneHot": tensor_one_hot.numpy().tolist(),
  "loss": float(loss.item()),
  "sumLoss": float(sum_loss.item()),
  "noneLoss": none_loss.numpy().round(6).tolist(),
  "grad": logits.grad.numpy().round(6).tolist(),
  "moduleLoss": float(module_loss.item()),
  "moduleGradFinite": bool(np.isfinite(module_logits.grad.numpy()).all()),
}
`);
    expect(result.oneHot).toEqual([
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ]);
    expect(result.tensorOneHot).toEqual([
      [0, 0, 1],
      [1, 0, 0],
    ]);
    expect(result.loss).toBeCloseTo(Math.log(2) / 3, 6);
    expect(result.sumLoss).toBeCloseTo(Math.log(2), 6);
    expect(result.noneLoss[0]).toBeCloseTo(Math.log(2), 6);
    expect(result.noneLoss.slice(1)).toEqual([0, 0]);
    expect(result.grad[0]).toBeCloseTo(1 / 6, 6);
    expect(result.grad.slice(1)).toEqual([0, 0]);
    expect(result.moduleLoss).toBe(0);
    expect(result.moduleGradFinite).toBe(true);
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
