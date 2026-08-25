import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("training compatibility regressions", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("keeps scalar leaf gradients as rank-zero ndarrays", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      first: number;
      accumulated: number;
      shape: number[];
      parameter: number;
      parameterDtype: string;
      rejected: string;
    }>(`
import browsergrad_jit as bg
import numpy as np

p = bg.tensor(np.asarray(3.0, dtype=np.float32), requires_grad=True)
(p * p).sum().backward()
first = p.grad.item()
(p * p).sum().backward()

parameter = bg.nn.Parameter(bg.from_numpy(np.asarray(2.0, dtype=np.float64)))
(parameter * parameter).sum().backward()
try:
    bg.from_numpy(3.0)
    rejected = "none"
except TypeError as error:
    rejected = str(error)

{
    "first": first,
    "accumulated": p.grad.item(),
    "shape": list(p.grad.shape),
    "parameter": parameter.grad.item(),
    "parameterDtype": parameter.grad.dtype,
    "rejected": rejected,
}
`);
    expect(result.first).toBe(6);
    expect(result.accumulated).toBe(12);
    expect(result.shape).toEqual([]);
    expect(result.parameter).toBe(4);
    expect(result.parameterDtype).toBe("float64");
    expect(result.rejected).toContain("expected np.ndarray or NumPy scalar");
  });

  it("keeps sigmoid forward and backward finite at saturation", async () => {
    const target = await getJitTarget();
    const result = await target.run<Array<{
      dtype: string;
      values: number[];
      gradients: number[];
      warnings: string[];
    }>>(`
import browsergrad_jit as bg
import numpy as np
import warnings

results = []
for dtype in (np.float32, np.float64):
    x = bg.tensor(np.asarray([-1000.0, -1.0, 0.0, 1.0, 1000.0], dtype=dtype), requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = bg.sigmoid(x)
        y.sum().backward()
        results.append({
            "dtype": x.dtype,
            "values": y.numpy().tolist(),
            "gradients": x.grad.numpy().tolist(),
            "warnings": [str(item.message) for item in caught],
        })
results
`);
    for (const item of result) {
      expect(item.warnings).toEqual([]);
      expect(item.values[0]).toBe(0);
      expect(item.values[4]).toBe(1);
      expect(item.gradients.every(Number.isFinite)).toBe(true);
      expect(item.gradients[0]).toBe(0);
      expect(item.gradients[2]).toBeCloseTo(0.25, 6);
      expect(item.gradients[4]).toBe(0);
      expect(item.values[1]).toBeCloseTo(0.268941421, 6);
      expect(item.values[3]).toBeCloseTo(0.731058579, 6);
    }
  });

  it("computes rank-one matmul and Linear operand gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, number[]>>(`
import browsergrad_jit as bg
import browsergrad_jit._functional as F

a = bg.tensor([1.0, 2.0], requires_grad=True)
b = bg.tensor([3.0, 4.0], requires_grad=True)
(a @ b).backward()

vm_a = bg.tensor([1.0, 2.0], requires_grad=True)
vm_b = bg.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
(vm_a @ vm_b).sum().backward()

mv_a = bg.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
mv_b = bg.tensor([5.0, 6.0], requires_grad=True)
(mv_a @ mv_b).sum().backward()

batched_vector = bg.tensor([1.0, 2.0], requires_grad=True)
batched_matrix = bg.tensor(
    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
    requires_grad=True,
)
(batched_vector @ batched_matrix).sum().backward()

x = bg.tensor([1.0, 2.0], requires_grad=True)
weight = bg.tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)
bias = bg.tensor([0.5, -0.5], requires_grad=True)
F.linear(x, weight, bias).sum().backward()

{
    "dotA": a.grad.numpy().tolist(),
    "dotB": b.grad.numpy().tolist(),
    "vmA": vm_a.grad.numpy().tolist(),
    "vmB": vm_b.grad.numpy().tolist(),
    "mvA": mv_a.grad.numpy().tolist(),
    "mvB": mv_b.grad.numpy().tolist(),
    "batchedVector": batched_vector.grad.numpy().tolist(),
    "batchedMatrix": batched_matrix.grad.numpy().tolist(),
    "linearX": x.grad.numpy().tolist(),
    "linearWeight": weight.grad.numpy().tolist(),
    "linearBias": bias.grad.numpy().tolist(),
}
`);
    expect(result.dotA).toEqual([3, 4]);
    expect(result.dotB).toEqual([1, 2]);
    expect(result.vmA).toEqual([6, 15]);
    expect(result.vmB).toEqual([[1, 1, 1], [2, 2, 2]]);
    expect(result.mvA).toEqual([[5, 6], [5, 6]]);
    expect(result.mvB).toEqual([4, 6]);
    expect(result.batchedVector).toEqual([14, 22]);
    expect(result.batchedMatrix).toEqual([
      [[1, 1], [2, 2]],
      [[1, 1], [2, 2]],
    ]);
    expect(result.linearX).toEqual([8, 10]);
    expect(result.linearWeight).toEqual([[1, 2], [1, 2]]);
    expect(result.linearBias).toEqual([1, 1]);
  });

  it("loads partial module state non-strictly and reports incompatibilities", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      missing: string[];
      unexpected: string[];
      loaded: boolean;
      strictError: string;
      shapeError: string;
      atomic: boolean;
      complete: string[][];
    }>(`
import browsergrad_jit as bg
import numpy as np

src = bg.nn.Sequential(bg.nn.Linear(2, 2), bg.nn.BatchNorm1d(2))
dst = bg.nn.Sequential(bg.nn.Linear(2, 2), bg.nn.BatchNorm1d(2))
full = src.state_dict()
partial = {"0.weight": full["0.weight"], "extra": np.zeros(1, dtype=np.float32)}
incompatible = dst.load_state_dict(partial, strict=False)
loaded = bool(np.array_equal(dst.state_dict()["0.weight"], full["0.weight"]))

try:
    dst.load_state_dict(partial, strict=True)
    strict_error = "none"
except KeyError as error:
    strict_error = str(error)

before = dst.state_dict()["0.weight"].copy()
bad = dict(full)
bad["0.weight"] = np.full_like(full["0.weight"], 99)
bad["1.running_var"] = np.zeros(3, dtype=np.float32)
try:
    dst.load_state_dict(bad)
    shape_error = "none"
except bg.ShapeError as error:
    shape_error = str(error)
atomic = bool(np.array_equal(dst.state_dict()["0.weight"], before))
complete = dst.load_state_dict(full)

{
    "missing": incompatible.missing_keys,
    "unexpected": incompatible.unexpected_keys,
    "loaded": loaded,
    "strictError": strict_error,
    "shapeError": shape_error,
    "atomic": atomic,
    "complete": [complete.missing_keys, complete.unexpected_keys],
}
`);
    expect(result.missing).toEqual([
      "0.bias",
      "1.weight",
      "1.bias",
      "1.running_mean",
      "1.running_var",
      "1.num_batches_tracked",
    ]);
    expect(result.unexpected).toEqual(["extra"]);
    expect(result.loaded).toBe(true);
    expect(result.strictError).toContain("missing keys");
    expect(result.strictError).toContain("unexpected keys");
    expect(result.shapeError).toContain("running_var");
    expect(result.atomic).toBe(true);
    expect(result.complete).toEqual([[], []]);
  });

  it("rejects weights_only and unknown load options before deserialization", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      weightsError: string;
      unknownError: string;
      sentinel: boolean;
      trusted: number[];
    }>(`
import browsergrad_jit as bg
import os
import pickle

payload = "/tmp/browsergrad-malicious.pkl"
sentinel = "/tmp/browsergrad-malicious-ran"
if os.path.exists(sentinel):
    os.unlink(sentinel)

class Malicious:
    def __reduce__(self):
        return (exec, ("open('/tmp/browsergrad-malicious-ran', 'w').write('ran')",))

with open(payload, "wb") as handle:
    pickle.dump(Malicious(), handle)
try:
    bg.load(payload, weights_only=True)
    weights_error = "none"
except bg.JitNotImplementedError as error:
    weights_error = str(error)
try:
    bg.load("/tmp/does-not-exist", map_location="cpu")
    unknown_error = "none"
except TypeError as error:
    unknown_error = str(error)

trusted_path = "/tmp/browsergrad-trusted.pkl"
bg.save({"values": [1, 2, 3]}, trusted_path)
trusted = bg.load(trusted_path, weights_only=False)["values"]
{
    "weightsError": weights_error,
    "unknownError": unknown_error,
    "sentinel": os.path.exists(sentinel),
    "trusted": trusted,
}
`);
    expect(result.weightsError).toContain("unrestricted pickle");
    expect(result.unknownError).toContain("map_location");
    expect(result.sentinel).toBe(false);
    expect(result.trusted).toEqual([1, 2, 3]);
  });

  it("supports contiguous queries and differentiable paired indexing", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      contiguous: boolean[];
      selected: number[];
      row: number[];
      gradient: number[][];
      boundsError: string;
      dtypeError: string;
    }>(`
import browsergrad_jit as bg

x = bg.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], requires_grad=True)
rows = bg.tensor([0, 0, 1])
columns = bg.tensor([2, 2, 1])
selected = x[rows, columns]
selected.sum().backward()
try:
    x[bg.tensor([0]), bg.tensor([3])].numpy()
    bounds_error = "none"
except bg.RealizationError as error:
    bounds_error = str(error)
try:
    x[bg.tensor([0.0]), bg.tensor([1])]
    dtype_error = "none"
except bg.ShapeError as error:
    dtype_error = str(error)

{
    "contiguous": [x.is_contiguous(), x.permute(1, 0).is_contiguous()],
    "selected": selected.numpy().tolist(),
    "row": x[-1].numpy().tolist(),
    "gradient": x.grad.numpy().tolist(),
    "boundsError": bounds_error,
    "dtypeError": dtype_error,
}
`);
    expect(result.contiguous).toEqual([true, true]);
    expect(result.selected).toEqual([12, 12, 21]);
    expect(result.row).toEqual([20, 21, 22]);
    expect(result.gradient).toEqual([[0, 0, 2], [0, 1, 0]]);
    expect(result.boundsError).toContain("index values must be in");
    expect(result.dtypeError).toContain("dtype int64");
  });

  it("evicts trace entries with module lifetime and never reuses ownership", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      before: { enabled: number; entries: number; hits: number; misses: number };
      afterCollection: { enabled: number; entries: number; hits: number; misses: number };
      afterReplacement: { enabled: number; entries: number; hits: number; misses: number };
      unique: boolean;
      value: number[];
    }>(`
import browsergrad_jit as bg
import gc

bg.jit.clear_trace_cache()
class Shift(bg.nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount
    def forward(self, x):
        return x + self.amount

x = bg.tensor([1.0])
module = Shift(1.0)
first_token = module._trace_cache_token
module(x)
module(x)
before = bg.jit.trace_cache_stats()
del module
gc.collect()
after_collection = bg.jit.trace_cache_stats()
replacement = Shift(7.0)
second_token = replacement._trace_cache_token
value = replacement(x).numpy().tolist()
after_replacement = bg.jit.trace_cache_stats()
{
    "before": before,
    "afterCollection": after_collection,
    "afterReplacement": after_replacement,
    "unique": first_token != second_token,
    "value": value,
}
`);
    expect(result.before.entries).toBe(1);
    expect(result.before.hits).toBe(1);
    expect(result.afterCollection.entries).toBe(0);
    expect(result.unique).toBe(true);
    expect(result.value).toEqual([8]);
    expect(result.afterReplacement.misses).toBe(result.before.misses + 1);
  });

  it("clips aggregate gradient norms through torch.nn.utils", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      importable: boolean;
      norm: number;
      gradient: number[];
      parameter: number[];
      empty: number;
      infNorm: number;
      infGradient: number[];
      finiteError: string;
      foreachError: string;
    }>(`
import browsergrad_jit as bg
import numpy as np
bg.install_torch_alias(force=True)
import torch
import torch.nn.utils as nn_utils

parameter = bg.nn.Parameter(bg.tensor([10.0, 20.0]))
parameter.grad = bg.tensor([3.0, 4.0])
before = parameter.numpy().tolist()
norm = nn_utils.clip_grad_norm_([parameter], 1.0)
empty = torch.nn.utils.clip_grad_norm_([], 1.0)

inf_parameter = bg.nn.Parameter(bg.tensor([1.0, 2.0]))
inf_parameter.grad = bg.tensor([3.0, -4.0])
inf_norm = nn_utils.clip_grad_norm_([inf_parameter], 2.0, norm_type=float("inf"))

bad = bg.nn.Parameter(bg.tensor([1.0]))
bad.grad = bg.tensor([np.nan])
try:
    nn_utils.clip_grad_norm_([bad], 1.0, error_if_nonfinite=True)
    finite_error = "none"
except RuntimeError as error:
    finite_error = str(error)
try:
    nn_utils.clip_grad_norm_([parameter], 1.0, foreach=True)
    foreach_error = "none"
except bg.JitNotImplementedError as error:
    foreach_error = str(error)

{
    "importable": torch.nn.utils is nn_utils,
    "norm": norm.item(),
    "gradient": parameter.grad.numpy().tolist(),
    "parameter": parameter.numpy().tolist(),
    "empty": empty.item(),
    "infNorm": inf_norm.item(),
    "infGradient": inf_parameter.grad.numpy().tolist(),
    "finiteError": finite_error,
    "foreachError": foreach_error,
}
`);
    expect(result.importable).toBe(true);
    expect(result.norm).toBeCloseTo(5, 6);
    expect(result.gradient[0]).toBeCloseTo(0.6, 5);
    expect(result.gradient[1]).toBeCloseTo(0.8, 5);
    expect(result.parameter).toEqual([10, 20]);
    expect(result.empty).toBe(0);
    expect(result.infNorm).toBe(4);
    expect(result.infGradient[0]).toBeCloseTo(1.5, 5);
    expect(result.infGradient[1]).toBeCloseTo(-2, 5);
    expect(result.finiteError).toContain("non-finite");
    expect(result.foreachError).toContain("foreach=True");
  });

  it("round-trips SGD, Adam, and AdamW optimizer state", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      sgd: boolean;
      adam: boolean;
      adamw: boolean;
      portableIds: boolean;
      empty: boolean;
      mismatchError: string;
      atomic: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np

def round_trip(optimizer_type, options):
    reference_parameter = bg.tensor([1.0, -2.0], requires_grad=True)
    source_parameter = bg.tensor([1.0, -2.0], requires_grad=True)
    reference = optimizer_type([reference_parameter], **options)
    source = optimizer_type([source_parameter], **options)
    for parameter, optimizer in ((reference_parameter, reference), (source_parameter, source)):
        parameter.grad = bg.tensor([0.25, -0.5])
        optimizer.step()
    checkpoint = source.state_dict()
    resumed_parameter = bg.tensor(source_parameter.numpy(), requires_grad=True)
    resumed = optimizer_type([resumed_parameter], lr=9.0)
    resumed.load_state_dict(checkpoint)
    reference_parameter.grad = bg.tensor([-0.1, 0.2])
    resumed_parameter.grad = bg.tensor([-0.1, 0.2])
    reference.step()
    resumed.step()
    return bool(np.allclose(reference_parameter.numpy(), resumed_parameter.numpy())), checkpoint

sgd, sgd_state = round_trip(bg.optim.SGD, {"lr": 0.1, "momentum": 0.9})
adam, adam_state = round_trip(bg.optim.Adam, {"lr": 0.01, "betas": (0.8, 0.95)})
adamw, adamw_state = round_trip(bg.optim.AdamW, {"lr": 0.01, "betas": (0.8, 0.95), "weight_decay": 0.1})

empty_source = bg.optim.Adam([bg.tensor([1.0], requires_grad=True)], lr=0.01)
empty_state = empty_source.state_dict()
empty_target = bg.optim.Adam([bg.tensor([1.0], requires_grad=True)], lr=0.5)
empty_target.load_state_dict(empty_state)
empty = empty_target.state_dict() == empty_state

target_parameter = bg.tensor([1.0, 2.0], requires_grad=True)
target = bg.optim.Adam([target_parameter], lr=0.5)
bad = adam_state.copy()
bad["state"] = {key: dict(value) for key, value in adam_state["state"].items()}
bad["state"][0]["exp_avg"] = np.zeros(3, dtype=np.float64)
before_lr = target.lr
try:
    target.load_state_dict(bad)
    mismatch_error = "none"
except bg.ShapeError as error:
    mismatch_error = str(error)

{
    "sgd": sgd,
    "adam": adam,
    "adamw": adamw,
    "portableIds": sgd_state["param_groups"][0]["params"] == [0] and list(sgd_state["state"]) == [0],
    "empty": empty,
    "mismatchError": mismatch_error,
    "atomic": target.lr == before_lr and target.state_dict()["state"] == {},
}
`);
    expect(result.sgd).toBe(true);
    expect(result.adam).toBe(true);
    expect(result.adamw).toBe(true);
    expect(result.portableIds).toBe(true);
    expect(result.empty).toBe(true);
    expect(result.mismatchError).toContain("metadata does not match");
    expect(result.atomic).toBe(true);
  });
});
