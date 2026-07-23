import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_NLL_LOSS_CONFORMANCE } from "../../../test-support/framework-nll-loss-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 eager torch.nn.functional.nll_loss conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches weighted class-index reductions, gradients, snapshots, empties, dtypes, and aliases", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_NLL_LOSS_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
base = fixture["base"]

def tensor(shape, values, dtype="float32", requires_grad=False):
    return grad.Tensor(
        np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape)),
        dtype=dtype,
        requires_grad=requires_grad,
    )

source = tensor(base["inputShape"], base["input"], requires_grad=True)
target = tensor(base["targetShape"], base["target"], "int64")
weight = tensor([3], base["weight"])
none = F.nll_loss(
    source,
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduction="none",
)
none_array = none.numpy()
none.backward(tensor(base["targetShape"], base["upstream"]))

summed = F.nll_loss(
    tensor(base["inputShape"], base["input"]),
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduction="sum",
)
mean_source = tensor(base["inputShape"], base["input"], requires_grad=True)
mean = F.nll_loss(
    mean_source,
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
)
mean.backward()
unweighted = F.nll_loss(
    tensor(base["inputShape"], base["input"]),
    target,
    ignore_index=base["ignoreIndex"],
)

snapshot_source = tensor(base["inputShape"], base["input"], requires_grad=True)
snapshot_target = tensor(base["targetShape"], base["target"], "int64")
snapshot_weight = tensor([3], base["weight"])
snapshot = F.nll_loss(
    snapshot_source,
    snapshot_target,
    weight=snapshot_weight,
    ignore_index=base["ignoreIndex"],
    reduction="none",
)
snapshot_source.data[:] = 100
snapshot_target.data[:] = 0
snapshot_weight.data[:] = 100
snapshot.backward(tensor(base["targetShape"], base["upstream"]))

unbatched_fixture = fixture["unbatched"]
unbatched_source = tensor([3], unbatched_fixture["input"], requires_grad=True)
unbatched_target = tensor([], [unbatched_fixture["target"]], "int64")
unbatched = F.nll_loss(unbatched_source, unbatched_target)
unbatched.backward()

empty = fixture["empty"]
zero_batch_input = tensor(empty["zeroBatchInputShape"], [])
zero_batch_target = tensor(empty["zeroBatchTargetShape"], [], "int64")
zero_support_input = tensor(empty["zeroSupportInputShape"], [])
zero_support_target = tensor(empty["zeroSupportTargetShape"], [], "int64")
empty_results = {
    "zeroBatchNoneShape": list(F.nll_loss(
        zero_batch_input, zero_batch_target, reduction="none"
    ).shape),
    "zeroBatchSum": float(F.nll_loss(
        zero_batch_input, zero_batch_target, reduction="sum"
    ).item()),
    "zeroBatchMeanNan": bool(np.isnan(F.nll_loss(
        zero_batch_input, zero_batch_target, reduction="mean"
    ).item())),
    "zeroSupportMeanNan": bool(np.isnan(F.nll_loss(
        zero_support_input, zero_support_target, reduction="mean"
    ).item())),
}

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype = case["dtype"]
    dtype_source = tensor([2, 3], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], dtype, True)
    dtype_target = tensor([2], [0, 2], "int64")
    dtype_output = F.nll_loss(dtype_source, dtype_target, reduction="sum")
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "gradient": dtype_source.grad.dtype,
    })

legacy_sum = F.nll_loss(
    tensor(base["inputShape"], base["input"]),
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    size_average=False,
)
legacy_none = F.nll_loss(
    tensor(base["inputShape"], base["input"]),
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduce=False,
)
with grad.no_grad():
    detached = F.nll_loss(
        tensor([3], unbatched_fixture["input"], requires_grad=True),
        unbatched_target,
    )

grad.install_torch_alias()
import torch
module = torch.nn.NLLLoss(
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduction="sum",
)
module_value = module(
    tensor(base["inputShape"], base["input"]),
    target,
)

{
    "none": none_array.reshape(-1).tolist(),
    "noneGradient": source.grad.numpy().reshape(-1).tolist(),
    "sum": float(summed.item()),
    "mean": float(mean.item()),
    "meanGradient": mean_source.grad.numpy().reshape(-1).tolist(),
    "unweightedMean": float(unweighted.item()),
    "snapshotGradient": snapshot_source.grad.numpy().reshape(-1).tolist(),
    "unbatched": float(unbatched.item()),
    "unbatchedGradient": unbatched_source.grad.numpy().tolist(),
    "empty": empty_results,
    "dtypes": dtype_results,
    "legacySum": float(legacy_sum.item()),
    "legacyNoneShape": list(legacy_none.shape),
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "module": float(module_value.item()),
}
`);

    const fixture = FRAMEWORK_NLL_LOSS_CONFORMANCE;
    expectCloseArray(result.none, fixture.base.weightedNone);
    expectCloseArray(result.noneGradient, fixture.base.weightedNoneGradient);
    expect(result.sum).toBeCloseTo(fixture.base.weightedSum, 5);
    expect(result.mean).toBeCloseTo(fixture.base.weightedMean, 5);
    expectCloseArray(result.meanGradient, fixture.base.weightedMeanGradient);
    expect(result.unweightedMean).toBeCloseTo(fixture.base.unweightedMean, 5);
    expectCloseArray(result.snapshotGradient, fixture.base.weightedNoneGradient);
    expect(result.unbatched).toBeCloseTo(fixture.unbatched.value, 5);
    expectCloseArray(result.unbatchedGradient, fixture.unbatched.gradient);
    expect(result.empty).toEqual({
      zeroBatchNoneShape: fixture.empty.zeroBatchTargetShape,
      zeroBatchSum: 0,
      zeroBatchMeanNan: true,
      zeroSupportMeanNan: true,
    });
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map(({ dtype }) => ({
      output: dtype,
      gradient: dtype,
    })));
    expect(result.legacySum).toBeCloseTo(fixture.base.weightedSum, 5);
    expect(result.legacyNoneShape).toEqual(fixture.base.targetShape);
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.module).toBeCloseTo(fixture.base.weightedSum, 5);
  });

  it("rejects malformed, hostile, out-of-range, and over-budget requests", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_NLL_LOSS_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

def tensor(values, dtype="float32", requires_grad=False):
    return grad.Tensor(
        np.asarray(values, dtype=np.dtype(dtype)),
        dtype=dtype,
        requires_grad=requires_grad,
    )

source = tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]])
target = tensor([0, 2], "int64")
errors = {
    "input-type": error(lambda: F.nll_loss([[-0.1]], target)),
    "target-type": error(lambda: F.nll_loss(source, [0, 2])),
    "target-shape": error(lambda: F.nll_loss(source, tensor([[0], [2]], "int64"))),
    "input-rank": error(lambda: F.nll_loss(tensor(1.0), tensor(0, "int64"))),
    "reduction": error(lambda: F.nll_loss(source, target, reduction="median")),
    "reduction-type": error(lambda: F.nll_loss(source, target, reduction=object())),
    "input-dtype": error(lambda: F.nll_loss(tensor([[1, 2], [3, 4]], "int32"), target)),
    "target-dtype": error(lambda: F.nll_loss(source, tensor([0, 1], "int32"))),
    "weight-type": error(lambda: F.nll_loss(source, target, weight=[1, 2, 3])),
    "weight-shape": error(lambda: F.nll_loss(source, target, weight=tensor([1, 2]))),
    "weight-dtype": error(lambda: F.nll_loss(source, target, weight=tensor([1, 2, 3], "float64"))),
    "weight-grad": error(lambda: F.nll_loss(
        source, target, weight=tensor([1, 2, 3], requires_grad=True)
    )),
    "ignore-type": error(lambda: F.nll_loss(source, target, ignore_index=True)),
    "ignore-range": error(lambda: F.nll_loss(source, target, ignore_index=1 << 70)),
    "size-average-type": error(lambda: F.nll_loss(source, target, size_average=1)),
    "reduce-type": error(lambda: F.nll_loss(source, target, reduce=1)),
    "target-range": error(lambda: F.nll_loss(source, tensor([0, 3], "int64"))),
}

class HostileArray(np.ndarray):
    pass
errors["array-subclass"] = error(lambda: F.nll_loss(
    grad.Tensor(
        np.asarray([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]], dtype=np.float32).view(HostileArray),
        dtype="float32",
    ),
    target,
))

def strided(shape, dtype):
    base = np.empty((1,), dtype=np.dtype(dtype))
    return np.lib.stride_tricks.as_strided(
        base,
        shape=shape,
        strides=(0,) * len(shape),
    )

work = fixture["limits"]["workExtent"]
workspace = fixture["limits"]["workspaceExtent"]
errors["work"] = error(lambda: F.nll_loss(
    grad.Tensor(strided((1, work), "float32"), dtype="float32"),
    tensor([0], "int64"),
    reduction="sum",
))
errors["workspace"] = error(lambda: F.nll_loss(
    grad.Tensor(strided((workspace, 1), "float64"), dtype="float64"),
    grad.Tensor(strided((workspace,), "int64"), dtype="int64"),
    reduction="sum",
))
errors["zero-hidden-work"] = error(lambda: F.nll_loss(
    grad.Tensor(strided((0, 1, work), "float32"), dtype="float32"),
    grad.Tensor(strided((0, work), "int64"), dtype="int64"),
    reduction="sum",
))
errors
`);

    for (const invalid of FRAMEWORK_NLL_LOSS_CONFORMANCE.invalid) {
      if (invalid.id === "session" || invalid.id === "weight-session") continue;
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["array-subclass"]).toMatch(/^TypeError: .*exact ndarray/u);
    expect(errors.work).toMatch(/^ValueError: .*work/u);
    expect(errors.workspace).toMatch(/^ValueError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ValueError: .*work/u);
  });
});
