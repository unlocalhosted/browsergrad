import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE } from "../../../test-support/framework-binary-cross-entropy-with-logits-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 eager BCE-with-logits conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches stable reductions, both gradients, extremes, promotion, snapshots, empties, and aliases", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE);
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

logits = tensor(base["shape"], base["logits"], requires_grad=True)
targets = tensor(base["shape"], base["targets"], requires_grad=True)
none = F.bce_with_logits_loss(logits, targets, reduction="none")
none_array = none.numpy()
none.backward(tensor(base["shape"], base["upstream"]))

summed = F.bce_with_logits_loss(
    tensor(base["shape"], base["logits"]),
    tensor(base["shape"], base["targets"]),
    reduction="sum",
)
mean_logits = tensor(base["shape"], base["logits"], requires_grad=True)
mean_targets = tensor(base["shape"], base["targets"], requires_grad=True)
mean = F.bce_with_logits_loss(mean_logits, mean_targets)
mean.backward()

extreme = fixture["extremes"]
extreme_logits = tensor([4], extreme["logits"], requires_grad=True)
extreme_targets = tensor([4], extreme["targets"], requires_grad=True)
extreme_loss = F.bce_with_logits_loss(
    extreme_logits, extreme_targets, reduction="none"
)
extreme_values = extreme_loss.numpy()
extreme_loss.backward(tensor([4], [1, 1, 1, 1]))

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype_logits = tensor([2], [-2, 2], case["logitsDtype"], True)
    dtype_targets = tensor([2], [0, 1], case["targetDtype"], True)
    dtype_output = F.bce_with_logits_loss(dtype_logits, dtype_targets, reduction="sum")
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "logitsGradient": dtype_logits.grad.dtype,
        "targetGradient": dtype_targets.grad.dtype,
    })

scalar_fixture = fixture["scalar"]
scalar_logits = tensor([], [scalar_fixture["logits"]], requires_grad=True)
scalar_target = tensor([], [scalar_fixture["target"]], requires_grad=True)
scalar = F.bce_with_logits_loss(scalar_logits, scalar_target)
scalar.backward()

empty = fixture["empty"]
empty_logits = tensor(empty["shape"], [])
empty_target = tensor(empty["shape"], [])
empty_none = F.bce_with_logits_loss(empty_logits, empty_target, reduction="none")
empty_sum = F.bce_with_logits_loss(empty_logits, empty_target, reduction="sum")
empty_mean = F.bce_with_logits_loss(empty_logits, empty_target, reduction="mean")
empty_grad_logits = tensor(empty["shape"], [], requires_grad=True)
empty_grad_target = tensor(empty["shape"], [], requires_grad=True)
F.bce_with_logits_loss(empty_grad_logits, empty_grad_target).backward()

snapshot_logits = tensor(base["shape"], base["logits"], requires_grad=True)
snapshot_target = tensor(base["shape"], base["targets"], requires_grad=True)
snapshot = F.bce_with_logits_loss(snapshot_logits, snapshot_target, reduction="none")
snapshot_logits.data[:] = 9
snapshot_target.data[:] = 0.1
snapshot.backward(tensor(base["shape"], base["upstream"]))

with grad.no_grad():
    detached = F.bce_with_logits_loss(
        tensor([1], [0], requires_grad=True),
        tensor([1], [1], requires_grad=True),
    )

grad.install_torch_alias()
import torch
alias = torch.nn.BCEWithLogitsLoss(reduction="sum")(
    torch.tensor([-2.0, 2.0]), torch.tensor([0.0, 1.0])
)
functional_alias = torch.nn.functional.binary_cross_entropy_with_logits(
    torch.tensor([-2.0, 2.0]),
    torch.tensor([0.0, 1.0]),
    reduction="sum",
)

{
    "none": none_array.reshape(-1).tolist(),
    "noneLogitsGradient": logits.grad.numpy().reshape(-1).tolist(),
    "noneTargetGradient": targets.grad.numpy().reshape(-1).tolist(),
    "sum": float(summed.item()),
    "mean": float(mean.item()),
    "meanLogitsGradient": mean_logits.grad.numpy().reshape(-1).tolist(),
    "meanTargetGradient": mean_targets.grad.numpy().reshape(-1).tolist(),
    "extremeValues": extreme_values.tolist(),
    "extremeLogitsGradients": extreme_logits.grad.numpy().tolist(),
    "extremeTargetGradients": extreme_targets.grad.numpy().tolist(),
    "dtypes": dtype_results,
    "scalar": float(scalar.item()),
    "scalarGradients": [float(scalar_logits.grad.item()), float(scalar_target.grad.item())],
    "emptyNoneShape": list(empty_none.shape),
    "emptyNoneSize": int(empty_none.data.size),
    "emptySum": float(empty_sum.item()),
    "emptyMeanIsNan": bool(np.isnan(empty_mean.item())),
    "emptyGradientShapes": [list(empty_grad_logits.grad.shape), list(empty_grad_target.grad.shape)],
    "snapshotGradients": [
        snapshot_logits.grad.numpy().reshape(-1).tolist(),
        snapshot_target.grad.numpy().reshape(-1).tolist(),
    ],
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "alias": float(alias.item()),
    "functionalAlias": float(functional_alias.item()),
}
`);

    const fixture = FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE;
    expectCloseArray(result.none, fixture.base.noneValues, 4);
    expectCloseArray(result.noneLogitsGradient, fixture.base.noneLogitsGradient, 4);
    expectCloseArray(result.noneTargetGradient, fixture.base.noneTargetGradient, 4);
    expect(result.sum).toBeCloseTo(fixture.base.sumValue, 4);
    expect(result.mean).toBeCloseTo(fixture.base.meanValue, 4);
    expectCloseArray(result.meanLogitsGradient, fixture.base.meanLogitsGradient, 5);
    expectCloseArray(result.meanTargetGradient, fixture.base.meanTargetGradient, 5);
    expectCloseArray(result.extremeValues, fixture.extremes.noneValues, 4);
    expectCloseArray(result.extremeLogitsGradients, fixture.extremes.logitsGradients, 4);
    expectCloseArray(result.extremeTargetGradients, fixture.extremes.targetGradients, 4);
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map((entry) => ({
      output: entry.outputDtype,
      logitsGradient: entry.logitsDtype,
      targetGradient: entry.targetDtype,
    })));
    expect(result.scalar).toBeCloseTo(fixture.scalar.output, 5);
    expectCloseArray(result.scalarGradients, [
      fixture.scalar.logitsGradient,
      fixture.scalar.targetGradient,
    ]);
    expect(result.emptyNoneShape).toEqual(fixture.empty.noneShape);
    expect(result.emptyNoneSize).toBe(0);
    expect(result.emptySum).toBe(fixture.empty.sumValue);
    expect(result.emptyMeanIsNan).toBe(true);
    expect(result.emptyGradientShapes).toEqual([fixture.empty.shape, fixture.empty.shape]);
    const snapshots = result.snapshotGradients as number[][];
    expectCloseArray(snapshots[0], fixture.base.noneLogitsGradient, 4);
    expectCloseArray(snapshots[1], fixture.base.noneTargetGradient, 4);
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    const expectedAlias = 2 * fixture.base.noneValues[0];
    expect(result.alias).toBeCloseTo(expectedAlias, 5);
    expect(result.functionalAlias).toBeCloseTo(expectedAlias, 5);
  });

  it("rejects malformed, array-subclass, unsupported weighting, and over-budget requests", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE);
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

def tensor(values, dtype="float32"):
    return grad.Tensor(np.asarray(values, dtype=np.dtype(dtype)), dtype=dtype)

logits = tensor([-2.0, 2.0])
targets = tensor([0.0, 1.0])
errors = {
    "logits-type": error(lambda: F.bce_with_logits_loss([-2.0, 2.0], targets)),
    "target-type": error(lambda: F.bce_with_logits_loss(logits, [0.0, 1.0])),
    "shape": error(lambda: F.bce_with_logits_loss(logits, tensor([1.0]))),
    "reduction": error(lambda: F.bce_with_logits_loss(logits, targets, reduction="median")),
    "reduction-type": error(lambda: F.bce_with_logits_loss(logits, targets, reduction=object())),
    "logits-dtype": error(lambda: F.bce_with_logits_loss(tensor([-2, 2], "int32"), targets)),
    "target-dtype": error(lambda: F.bce_with_logits_loss(logits, tensor([0, 1], "int32"))),
    "weight": error(lambda: F.bce_with_logits_loss(logits, targets, weight=targets)),
    "pos-weight": error(lambda: F.bce_with_logits_loss(logits, targets, pos_weight=targets)),
}

class HostileArray(np.ndarray):
    pass

errors["array-subclass"] = error(lambda: F.bce_with_logits_loss(
    grad.Tensor(np.array([-2.0, 2.0], dtype=np.float32).view(HostileArray), dtype="float32"),
    targets,
))

def strided(shape, dtype):
    base = np.empty((1,), dtype=np.dtype(dtype))
    return np.lib.stride_tricks.as_strided(base, shape=shape, strides=(0,) * len(shape))

work_shape = (fixture["limits"]["workExtent"],)
errors["work"] = error(lambda: F.bce_with_logits_loss(
    grad.Tensor(strided(work_shape, "float32"), dtype="float32"),
    grad.Tensor(strided(work_shape, "float32"), dtype="float32"),
    reduction="sum",
))
workspace_shape = (fixture["limits"]["workspaceExtent"],)
errors["workspace"] = error(lambda: F.bce_with_logits_loss(
    grad.Tensor(strided(workspace_shape, "float64"), dtype="float64"),
    grad.Tensor(strided(workspace_shape, "float64"), dtype="float64"),
    reduction="sum",
))
zero_hidden_shape = (0, fixture["limits"]["workExtent"])
errors["zero-hidden-work"] = error(lambda: F.bce_with_logits_loss(
    grad.Tensor(strided(zero_hidden_shape, "float32"), dtype="float32"),
    grad.Tensor(strided(zero_hidden_shape, "float32"), dtype="float32"),
    reduction="sum",
))
errors
`);

    for (const invalid of FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE.invalid) {
      if (invalid.id === "session") continue;
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["array-subclass"]).toMatch(/^TypeError: .*exact ndarray/u);
    expect(errors.work).toMatch(/^ValueError: .*work/u);
    expect(errors.workspace).toMatch(/^ValueError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ValueError: .*work/u);
  });
});
