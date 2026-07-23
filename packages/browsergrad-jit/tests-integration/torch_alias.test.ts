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
import pkg from "../package.json";
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
    expect(result.ver).toBe(pkg.version);
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
  "nn.LayerNorm": hasattr(nn, "LayerNorm"),
  "nn.Conv1d": hasattr(nn, "Conv1d"),
  "nn.Conv2d": hasattr(nn, "Conv2d"),
  "nn.ConvTranspose2d": hasattr(nn, "ConvTranspose2d"),
  "nn.Conv3d": hasattr(nn, "Conv3d"),
  "F.conv2d": hasattr(__import__("torch.nn.functional", fromlist=["conv2d"]), "conv2d"),
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
      "nn.LayerNorm": true,
      "nn.Conv1d": true,
      "nn.Conv2d": true,
      "nn.ConvTranspose2d": true,
      "nn.Conv3d": true,
      "F.conv2d": true,
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
    expect(result.bnStateKeys).toEqual([
      "bias",
      "num_batches_tracked",
      "running_mean",
      "running_var",
      "weight",
    ]);
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

  it("supports PyTorch unary math aliases and gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      methodAbs: number[];
      torchAbs: number[];
      methodSqrt: number[];
      torchSqrt: number[];
      torchPow: number[];
      grad: number[];
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch

x = torch.tensor([-4.0, 9.0], requires_grad=True)
positive = torch.tensor([4.0, 9.0], requires_grad=True)
y = x.abs().sum() + torch.sqrt(positive).sum()
y.backward()

{
  "methodAbs": x.abs().numpy().tolist(),
  "torchAbs": torch.abs(x).numpy().tolist(),
  "methodSqrt": positive.sqrt().numpy().tolist(),
  "torchSqrt": torch.sqrt(positive).numpy().tolist(),
  "torchPow": torch.pow(torch.tensor([2.0, 3.0]), 3).numpy().tolist(),
  "grad": positive.grad.numpy().round(6).tolist(),
}
`);
    expect(result.methodAbs).toEqual([4, 9]);
    expect(result.torchAbs).toEqual([4, 9]);
    expect(result.methodSqrt).toEqual([2, 3]);
    expect(result.torchSqrt).toEqual([2, 3]);
    expect(result.torchPow).toEqual([8, 27]);
    expect(result.grad[0]).toBeCloseTo(0.25, 6);
    expect(result.grad[1]).toBeCloseTo(1 / 6, 6);
  });

  it("supports eager-compatible math, clamp, and activation aliases", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      sin: number[];
      cos: number[];
      rsqrt: number[];
      clamp: number[];
      clip: number[];
      clampMin: number[];
      clampGrad: number[];
      sign: number[];
      minimum: number[];
      minGradA: number[];
      minGradB: number[];
      silu: number[];
      leaky: number[];
      moduleLeaky: number[];
      moduleFlatShape: number[];
      activationGrad: number[];
      caps: Record<string, boolean>;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

trig = torch.tensor([0.0, np.pi / 2], requires_grad=True)
c = torch.tensor([-2.0, -0.5, 0.0, 2.0], requires_grad=True)
(c.clamp(min=-1.0, max=1.0).sum() + c.sign().sum()).backward()

a = torch.tensor([1.0, 4.0], requires_grad=True)
b = torch.tensor([2.0, 3.0], requires_grad=True)
mn = torch.minimum(a, b)
mn.sum().backward()

act = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
functional_leaky = F.leaky_relu(act, negative_slope=0.2)
module_leaky = nn.LeakyReLU(negative_slope=0.3)(act)
(functional_leaky.sum() + module_leaky.sum()).backward()

flat = nn.Flatten(start_dim=1)(torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4)))

{
  "sin": torch.sin(trig).numpy().round(6).tolist(),
  "cos": torch.cos(trig).numpy().round(6).tolist(),
  "rsqrt": torch.rsqrt(torch.tensor([4.0, 9.0])).numpy().round(6).tolist(),
  "clamp": c.clamp(min=-1.0, max=1.0).numpy().tolist(),
  "clip": c.clip(min=-0.5, max=0.5).numpy().tolist(),
  "clampMin": c.clamp_min(0.0).numpy().tolist(),
  "clampGrad": c.grad.numpy().tolist(),
  "sign": c.sign().numpy().tolist(),
  "minimum": mn.numpy().tolist(),
  "minGradA": a.grad.numpy().tolist(),
  "minGradB": b.grad.numpy().tolist(),
  "silu": F.silu(torch.tensor([-1.0, 0.0, 1.0])).numpy().round(6).tolist(),
  "leaky": functional_leaky.numpy().tolist(),
  "moduleLeaky": module_leaky.numpy().tolist(),
  "moduleFlatShape": list(flat.shape),
  "activationGrad": act.grad.numpy().round(6).tolist(),
  "caps": {
    "torch.sin": hasattr(torch, "sin"),
    "torch.cos": hasattr(torch, "cos"),
    "torch.rsqrt": hasattr(torch, "rsqrt"),
    "torch.minimum": hasattr(torch, "minimum"),
    "Tensor.clamp": hasattr(c, "clamp"),
    "Tensor.clip": hasattr(c, "clip"),
    "Tensor.clamp_min": hasattr(c, "clamp_min"),
    "Tensor.sign": hasattr(c, "sign"),
    "F.silu": hasattr(F, "silu"),
    "F.leaky_relu": hasattr(F, "leaky_relu"),
    "nn.LeakyReLU": hasattr(nn, "LeakyReLU"),
    "nn.Flatten": hasattr(nn, "Flatten"),
  },
}
`);
    expect(result.sin).toEqual([0, 1]);
    expect(result.cos[0]).toBe(1);
    expect(Math.abs(result.cos[1]!)).toBeLessThan(1e-5);
    expect(result.rsqrt[0]).toBe(0.5);
    expect(result.rsqrt[1]).toBeCloseTo(1 / 3, 6);
    expect(result.clamp).toEqual([-1, -0.5, 0, 1]);
    expect(result.clip).toEqual([-0.5, -0.5, 0, 0.5]);
    expect(result.clampMin).toEqual([0, 0, 0, 2]);
    expect(result.clampGrad).toEqual([0, 1, 1, 0]);
    expect(result.sign).toEqual([-1, -1, 0, 1]);
    expect(result.minimum).toEqual([1, 3]);
    expect(result.minGradA).toEqual([1, 0]);
    expect(result.minGradB).toEqual([0, 1]);
    expect(result.silu[0]).toBeCloseTo(-0.268941, 6);
    expect(result.silu[1]).toBe(0);
    expect(result.silu[2]).toBeCloseTo(0.731059, 6);
    expect(result.leaky[0]).toBeCloseTo(-0.2, 6);
    expect(result.leaky.slice(1)).toEqual([0, 2]);
    expect(result.moduleLeaky[0]).toBeCloseTo(-0.3, 6);
    expect(result.moduleLeaky.slice(1)).toEqual([0, 2]);
    expect(result.moduleFlatShape).toEqual([2, 12]);
    expect(result.activationGrad).toEqual([0.5, 0.5, 2]);
    expect(result.caps).toEqual({
      "torch.sin": true,
      "torch.cos": true,
      "torch.rsqrt": true,
      "torch.minimum": true,
      "Tensor.clamp": true,
      "Tensor.clip": true,
      "Tensor.clamp_min": true,
      "Tensor.sign": true,
      "F.silu": true,
      "F.leaky_relu": true,
      "nn.LeakyReLU": true,
      "nn.Flatten": true,
    });
  });

  it("supports PyTorch where and like-factory aliases", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      selected: number[];
      grad: number[];
      zeros: number[][];
      ones: number[][];
      zerosDtype: string;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch

x = torch.tensor([-2.0, 3.0], requires_grad=True)
selected = torch.where(x > 0, x, -x)
selected.sum().backward()
base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
zeros = torch.zeros_like(base)
ones = torch.ones_like(base, dtype=torch.int64)

{
  "selected": selected.numpy().tolist(),
  "grad": x.grad.numpy().tolist(),
  "zeros": zeros.numpy().tolist(),
  "ones": ones.numpy().tolist(),
  "zerosDtype": zeros.dtype,
}
`);
    expect(result.selected).toEqual([2, 3]);
    expect(result.grad).toEqual([-1, 1]);
    expect(result.zeros).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(result.ones).toEqual([
      [1, 1],
      [1, 1],
    ]);
    expect(result.zerosDtype).toBe("float32");
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

  it("supports eager-compatible indexing, repeat, and reduction aliases", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      prod: number[];
      prodGrad: number[][];
      cumsum: number[];
      cumsumGrad: number[];
      gathered: number[][];
      gatherGrad: number[][];
      repeatedShape: number[];
      repeatInterleave: number[][];
      repeatGrad: number[][];
      expandedShape: number[];
      flip: number[][];
      expandFlipGrad: number[][];
      sortedValues: number[][];
      sortedIndices: number[][];
      topValues: number[][];
      topIndices: number[][];
      caps: Record<string, boolean>;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
import numpy as np

p = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
prod = torch.prod(p, dim=1)
prod.sum().backward()

c = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
cs = torch.cumsum(c, dim=0)
cs.sum().backward()

g = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], requires_grad=True)
idx = torch.tensor([[2, 0], [1, 1]], dtype=torch.int64)
gathered = torch.gather(g, 1, idx)
gathered.sum().backward()

r = torch.tensor([[1.0, 2.0]], requires_grad=True)
repeated = r.repeat(2, 1)
interleaved = torch.repeat_interleave(r, repeats=3, dim=1)
(repeated.sum() + interleaved.sum()).backward()

e = torch.tensor([[1.0], [2.0]], requires_grad=True)
expanded = e.expand(2, 3)
flipped = e.flip(0)
(expanded.sum() + flipped.sum()).backward()

s = torch.tensor([[3.0, 1.0, 2.0], [4.0, 6.0, 5.0]])
sort_values, sort_indices = torch.sort(s, dim=1, descending=True)
top_values, top_indices = torch.topk(s, 2, dim=1, largest=False, sorted=True)

{
  "prod": prod.numpy().tolist(),
  "prodGrad": p.grad.numpy().tolist(),
  "cumsum": cs.numpy().tolist(),
  "cumsumGrad": c.grad.numpy().tolist(),
  "gathered": gathered.numpy().tolist(),
  "gatherGrad": g.grad.numpy().tolist(),
  "repeatedShape": list(repeated.shape),
  "repeatInterleave": interleaved.numpy().tolist(),
  "repeatGrad": r.grad.numpy().tolist(),
  "expandedShape": list(expanded.shape),
  "flip": flipped.numpy().tolist(),
  "expandFlipGrad": e.grad.numpy().tolist(),
  "sortedValues": sort_values.numpy().tolist(),
  "sortedIndices": sort_indices.numpy().tolist(),
  "topValues": top_values.numpy().tolist(),
  "topIndices": top_indices.numpy().tolist(),
  "caps": {
    "torch.prod": hasattr(torch, "prod"),
    "torch.gather": hasattr(torch, "gather"),
    "torch.repeat_interleave": hasattr(torch, "repeat_interleave"),
    "torch.cumsum": hasattr(torch, "cumsum"),
    "torch.sort": hasattr(torch, "sort"),
    "torch.topk": hasattr(torch, "topk"),
    "Tensor.repeat": hasattr(r, "repeat"),
    "Tensor.expand": hasattr(e, "expand"),
    "Tensor.flip": hasattr(e, "flip"),
    "Tensor.topk": hasattr(s, "topk"),
  },
}
`);
    expect(result.prod).toEqual([2, 12]);
    expect(result.prodGrad).toEqual([
      [2, 1],
      [4, 3],
    ]);
    expect(result.cumsum).toEqual([1, 3, 6]);
    expect(result.cumsumGrad).toEqual([3, 2, 1]);
    expect(result.gathered).toEqual([
      [12, 10],
      [21, 21],
    ]);
    expect(result.gatherGrad).toEqual([
      [1, 0, 1],
      [0, 2, 0],
    ]);
    expect(result.repeatedShape).toEqual([2, 2]);
    expect(result.repeatInterleave).toEqual([[1, 1, 1, 2, 2, 2]]);
    expect(result.repeatGrad).toEqual([[5, 5]]);
    expect(result.expandedShape).toEqual([2, 3]);
    expect(result.flip).toEqual([[2], [1]]);
    expect(result.expandFlipGrad).toEqual([[4], [4]]);
    expect(result.sortedValues).toEqual([
      [3, 2, 1],
      [6, 5, 4],
    ]);
    expect(result.sortedIndices).toEqual([
      [0, 2, 1],
      [1, 2, 0],
    ]);
    expect(result.topValues).toEqual([
      [1, 2],
      [4, 5],
    ]);
    expect(result.topIndices).toEqual([
      [1, 2],
      [0, 2],
    ]);
    expect(result.caps).toEqual({
      "torch.prod": true,
      "torch.gather": true,
      "torch.repeat_interleave": true,
      "torch.cumsum": true,
      "torch.sort": true,
      "torch.topk": true,
      "Tensor.repeat": true,
      "Tensor.expand": true,
      "Tensor.flip": true,
      "Tensor.topk": true,
    });
  });

  it("supports eager-compatible convenience, masking, and namespace aliases", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      device: string;
      sameCpu: boolean;
      sameCuda: boolean;
      contiguousSame: boolean;
      doubleDtype: string;
      intDtype: string;
      toDtype: string;
      varValues: number[];
      stdValues: number[];
      varGrad: number[][];
      masked: number[];
      maskedGrad: number[];
      maskedInplace: number[];
      scatter: number[][];
      triu: number[][];
      tril: number[][];
      triGrad: number[][];
      einsum: number[][];
      multinomialShape: number[];
      multinomialDtype: string;
      zeroGradIsNone: boolean;
      ceLossAlias: number;
      caps: Record<string, boolean>;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
import torch.nn.functional as F
import numpy as np

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
var_values = x.var(dim=1, unbiased=False)
std_values = x.std(dim=1, unbiased=False)
var_values.sum().backward()

m = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
mask = torch.tensor([False, True, False], dtype="bool")
masked = m.masked_fill(mask, -9.0)
masked.sum().backward()
masked_inplace = m.masked_fill_(mask, 0.0)

base = torch.zeros(1, 3)
idx = torch.tensor([[0, 2]], dtype=torch.int64)
scatter = base.scatter(1, idx, torch.tensor([[5.0, 6.0]]))

tri = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
upper = torch.triu(tri)
lower = torch.tril(tri, diagonal=-1)
(upper.sum() + lower.sum()).backward()

a = torch.tensor([[1.0, 2.0]])
b = torch.tensor([[3.0], [4.0]])
sampled = torch.multinomial(torch.tensor([0.1, 0.9]), num_samples=1)

zg = torch.tensor([1.0], requires_grad=True)
(zg * 2.0).sum().backward()
zg.zero_grad()

ce_loss = F.cross_entropy_loss(
    torch.tensor([[1.0, 3.0], [4.0, 2.0]]),
    torch.tensor([1, 0], dtype=torch.int64),
)

{
  "device": x.device,
  "sameCpu": x.cpu() is x,
  "sameCuda": x.cuda() is x,
  "contiguousSame": x.contiguous() is x,
  "doubleDtype": x.double().dtype,
  "intDtype": torch.tensor([1.2]).int().dtype,
  "toDtype": x.to(dtype="float64").dtype,
  "varValues": var_values.numpy().tolist(),
  "stdValues": std_values.numpy().round(6).tolist(),
  "varGrad": x.grad.numpy().tolist(),
  "masked": masked.numpy().tolist(),
  "maskedGrad": m.grad.numpy().tolist(),
  "maskedInplace": masked_inplace.numpy().tolist(),
  "scatter": scatter.numpy().tolist(),
  "triu": upper.numpy().tolist(),
  "tril": lower.numpy().tolist(),
  "triGrad": tri.grad.numpy().tolist(),
  "einsum": torch.einsum("ij,jk", a, b).numpy().tolist(),
  "multinomialShape": list(sampled.shape),
  "multinomialDtype": sampled.dtype,
  "zeroGradIsNone": zg.grad is None,
  "ceLossAlias": float(ce_loss.item()),
  "caps": {
    "torch.F": hasattr(torch, "F"),
    "torch.functional": hasattr(torch, "functional"),
    "torch.std": hasattr(torch, "std"),
    "torch.var": hasattr(torch, "var"),
    "torch.triu": hasattr(torch, "triu"),
    "torch.tril": hasattr(torch, "tril"),
    "torch.einsum": hasattr(torch, "einsum"),
    "torch.multinomial": hasattr(torch, "multinomial"),
    "F.cross_entropy_loss": hasattr(F, "cross_entropy_loss"),
    "F.bce_with_logits_loss": hasattr(F, "bce_with_logits_loss"),
  },
}
`);
    expect(result.device).toBe("cpu");
    expect(result.sameCpu).toBe(true);
    expect(result.sameCuda).toBe(true);
    expect(result.contiguousSame).toBe(true);
    expect(result.doubleDtype).toBe("float64");
    expect(result.intDtype).toBe("int32");
    expect(result.toDtype).toBe("float64");
    expect(result.varValues).toEqual([0.25, 0.25]);
    expect(result.stdValues).toEqual([0.5, 0.5]);
    expect(result.varGrad).toEqual([
      [-0.5, 0.5],
      [-0.5, 0.5],
    ]);
    expect(result.masked).toEqual([1, -9, 3]);
    expect(result.maskedGrad).toEqual([1, 0, 1]);
    expect(result.maskedInplace).toEqual([1, 0, 3]);
    expect(result.scatter).toEqual([[5, 0, 6]]);
    expect(result.triu).toEqual([
      [1, 2],
      [0, 4],
    ]);
    expect(result.tril).toEqual([
      [0, 0],
      [3, 0],
    ]);
    expect(result.triGrad).toEqual([
      [1, 1],
      [1, 1],
    ]);
    expect(result.einsum).toEqual([[11]]);
    expect(result.multinomialShape).toEqual([1]);
    expect(result.multinomialDtype).toBe("int64");
    expect(result.zeroGradIsNone).toBe(true);
    expect(result.ceLossAlias).toBeLessThan(0.2);
    expect(result.caps).toEqual({
      "torch.F": true,
      "torch.functional": true,
      "torch.std": true,
      "torch.var": true,
      "torch.triu": true,
      "torch.tril": true,
      "torch.einsum": true,
      "torch.multinomial": true,
      "F.cross_entropy_loss": true,
      "F.bce_with_logits_loss": true,
    });
  });

  it("supports eager-compatible functional spatial and vector utility aliases", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      padded: number[][];
      padGrad: number[][];
      normalizedRows: number[];
      cosine: number[];
      interpolatedShape: number[];
      interpolated: number[][][][];
      attentionShape: number[];
      attentionRows: number[];
      caps: Record<string, boolean>;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
import torch.nn.functional as F
import numpy as np

p = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
padded = F.pad(p, (1, 0, 0, 1), value=-1.0)
padded.sum().backward()

n = F.normalize(torch.tensor([[3.0, 4.0], [0.0, 5.0]]), dim=1)
cos = F.cosine_similarity(
    torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
    torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    dim=1,
)

img = torch.tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
interp = F.interpolate(img, scale_factor=2, mode="nearest")

q = torch.tensor(np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32))
k = torch.tensor(np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32))
v = torch.tensor(np.array([[[[10.0, 0.0], [0.0, 20.0]]]], dtype=np.float32))
attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

{
  "padded": padded.numpy().tolist(),
  "padGrad": p.grad.numpy().tolist(),
  "normalizedRows": (n * n).sum(dim=1).numpy().round(6).tolist(),
  "cosine": cos.numpy().round(6).tolist(),
  "interpolatedShape": list(interp.shape),
  "interpolated": interp.numpy().tolist(),
  "attentionShape": list(attn.shape),
  "attentionRows": attn.numpy().sum(axis=-1).round(6).tolist()[0][0],
  "caps": {
    "F.pad": hasattr(F, "pad"),
    "F.interpolate": hasattr(F, "interpolate"),
    "F.normalize": hasattr(F, "normalize"),
    "F.cosine_similarity": hasattr(F, "cosine_similarity"),
    "F.scaled_dot_product_attention": hasattr(F, "scaled_dot_product_attention"),
  },
}
`);
    expect(result.padded).toEqual([
      [-1, 1, 2],
      [-1, 3, 4],
      [-1, -1, -1],
    ]);
    expect(result.padGrad).toEqual([
      [1, 1],
      [1, 1],
    ]);
    expect(result.normalizedRows).toEqual([1, 1]);
    expect(result.cosine[0]).toBe(0);
    expect(result.cosine[1]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(result.interpolatedShape).toEqual([1, 1, 4, 4]);
    expect(result.interpolated[0]?.[0]).toEqual([
      [1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4],
    ]);
    expect(result.attentionShape).toEqual([1, 1, 2, 2]);
    expect(result.attentionRows[0]).toBe(10);
    expect(result.attentionRows[1]).toBeCloseTo(16.697617, 6);
    expect(result.caps).toEqual({
      "F.pad": true,
      "F.interpolate": true,
      "F.normalize": true,
      "F.cosine_similarity": true,
      "F.scaled_dot_product_attention": true,
    });
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

  it("supports eager-compatible functional and module loss aliases", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      l1Loss: number;
      l1Grad: number[];
      bceLoss: number;
      bceGrad: number[];
      smoothLoss: number;
      smoothGrad: number[];
      klLoss: number;
      klGrad: number[][];
      caps: Record<string, boolean>;
    }>(`
import browsergrad_jit as bg
bg.install_torch_alias()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

l1_x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
l1_t = torch.tensor([0.0, -1.0, 5.0])
l1_loss = nn.L1Loss()(l1_x, l1_t)
l1_loss.backward()

bce_x = torch.tensor([0.25, 0.75], requires_grad=True)
bce_t = torch.tensor([0.0, 1.0])
bce_loss = F.binary_cross_entropy(bce_x, bce_t)
bce_loss.backward()

smooth_x = torch.tensor([0.0, 2.0, -2.0], requires_grad=True)
smooth_t = torch.tensor([0.0, 0.0, 0.0])
smooth_loss = nn.SmoothL1Loss(reduction="sum")(smooth_x, smooth_t)
smooth_loss.backward()

kl_x = torch.tensor(np.log([[0.5, 0.5], [0.25, 0.75]]), requires_grad=True)
kl_t = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
kl_loss = nn.KLDivLoss(reduction="batchmean")(kl_x, kl_t)
kl_loss.backward()

{
  "l1Loss": float(l1_loss.item()),
  "l1Grad": l1_x.grad.numpy().round(6).tolist(),
  "bceLoss": float(bce_loss.item()),
  "bceGrad": bce_x.grad.numpy().round(6).tolist(),
  "smoothLoss": float(smooth_loss.item()),
  "smoothGrad": smooth_x.grad.numpy().round(6).tolist(),
  "klLoss": float(kl_loss.item()),
  "klGrad": kl_x.grad.numpy().round(6).tolist(),
  "caps": {
    "nn.BCELoss": hasattr(nn, "BCELoss"),
    "nn.L1Loss": hasattr(nn, "L1Loss"),
    "nn.SmoothL1Loss": hasattr(nn, "SmoothL1Loss"),
    "nn.KLDivLoss": hasattr(nn, "KLDivLoss"),
    "F.bce_loss": hasattr(F, "bce_loss"),
    "F.kl_div": hasattr(F, "kl_div"),
  },
}
`);
    expect(result.l1Loss).toBeCloseTo(4 / 3, 6);
    expect(result.l1Grad[0]).toBeCloseTo(1 / 3, 6);
    expect(result.l1Grad[1]).toBeCloseTo(-1 / 3, 6);
    expect(result.l1Grad[2]).toBeCloseTo(-1 / 3, 6);
    expect(result.bceLoss).toBeCloseTo(-Math.log(0.75), 6);
    expect(result.bceGrad[0]).toBeCloseTo(2 / 3, 6);
    expect(result.bceGrad[1]).toBeCloseTo(-2 / 3, 6);
    expect(result.smoothLoss).toBe(3);
    expect(result.smoothGrad).toEqual([0, 1, -1]);
    expect(result.klLoss).toBeCloseTo(-Math.log(0.75) / 2, 6);
    expect(result.klGrad[0]).toEqual([-0.25, -0.25]);
    const klSecondRow = result.klGrad[1];
    expect(klSecondRow).toBeDefined();
    expect(klSecondRow![0]).toBeCloseTo(0, 6);
    expect(klSecondRow![1]).toBe(-0.5);
    expect(result.caps).toEqual({
      "nn.BCELoss": true,
      "nn.L1Loss": true,
      "nn.SmoothL1Loss": true,
      "nn.KLDivLoss": true,
      "F.bce_loss": true,
      "F.kl_div": true,
    });
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
