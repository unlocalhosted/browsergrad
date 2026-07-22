import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE } from "../../../test-support/framework-binary-cross-entropy-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 eager binary cross entropy conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches reductions, promotion, both gradients, endpoints, snapshots, empties, and aliases", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE);
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

probabilities = tensor(base["shape"], base["inputValues"], requires_grad=True)
targets = tensor(base["shape"], base["targetValues"], requires_grad=True)
none = F.bce_loss(probabilities, targets, reduction="none")
none_array = none.numpy()
none.backward(tensor(base["shape"], base["upstreamValues"]))

summed = F.bce_loss(
    tensor(base["shape"], base["inputValues"]),
    tensor(base["shape"], base["targetValues"]),
    reduction="sum",
)
mean_probabilities = tensor(base["shape"], base["inputValues"], requires_grad=True)
mean_targets = tensor(base["shape"], base["targetValues"], requires_grad=True)
mean = F.bce_loss(mean_probabilities, mean_targets)
mean.backward()

boundary = fixture["boundaries"]
boundary_probabilities = tensor([2], boundary["inputValues"], requires_grad=True)
boundary_targets = tensor([2], boundary["targetValues"], requires_grad=True)
boundary_loss = F.bce_loss(boundary_probabilities, boundary_targets, reduction="none")
boundary_values = boundary_loss.numpy()
boundary_loss.backward(tensor([2], [1.0, 1.0]))

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype_probabilities = tensor([2], [0.25, 0.75], case["inputDtype"], True)
    dtype_targets = tensor([2], [0.0, 1.0], case["targetDtype"], True)
    dtype_output = F.bce_loss(dtype_probabilities, dtype_targets, reduction="sum")
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "inputGradient": dtype_probabilities.grad.dtype,
        "targetGradient": dtype_targets.grad.dtype,
    })

scalar_fixture = fixture["scalar"]
scalar_probability = tensor([], [scalar_fixture["inputValue"]], requires_grad=True)
scalar_target = tensor([], [scalar_fixture["targetValue"]], requires_grad=True)
scalar = F.bce_loss(scalar_probability, scalar_target)
scalar.backward()

empty = fixture["empty"]
empty_probability = tensor(empty["shape"], [])
empty_target = tensor(empty["shape"], [])
empty_none = F.bce_loss(empty_probability, empty_target, reduction="none")
empty_sum = F.bce_loss(empty_probability, empty_target, reduction="sum")
empty_mean = F.bce_loss(empty_probability, empty_target, reduction="mean")
empty_grad_probability = tensor(empty["shape"], [], requires_grad=True)
empty_grad_target = tensor(empty["shape"], [], requires_grad=True)
F.bce_loss(empty_grad_probability, empty_grad_target).backward()

snapshot_probability = tensor(base["shape"], base["inputValues"], requires_grad=True)
snapshot_target = tensor(base["shape"], base["targetValues"], requires_grad=True)
snapshot = F.bce_loss(snapshot_probability, snapshot_target, reduction="none")
snapshot_probability.data[:] = 0.9
snapshot_target.data[:] = 0.1
snapshot.backward(tensor(base["shape"], base["upstreamValues"]))

with grad.no_grad():
    detached = F.bce_loss(
        tensor([1], [0.25], requires_grad=True),
        tensor([1], [1.0], requires_grad=True),
    )

grad.install_torch_alias()
import torch
alias = torch.nn.BCELoss(reduction="sum")(
    torch.tensor([0.25, 0.75]),
    torch.tensor([0.0, 1.0]),
)

{
    "none": none_array.reshape(-1).tolist(),
    "noneInputGradient": probabilities.grad.numpy().reshape(-1).tolist(),
    "noneTargetGradient": targets.grad.numpy().reshape(-1).tolist(),
    "sum": float(summed.item()),
    "mean": float(mean.item()),
    "meanInputGradient": mean_probabilities.grad.numpy().reshape(-1).tolist(),
    "meanTargetGradient": mean_targets.grad.numpy().reshape(-1).tolist(),
    "boundaryValues": boundary_values.tolist(),
    "boundaryInputGradients": boundary_probabilities.grad.numpy().tolist(),
    "boundaryTargetPositiveInfinity": bool(np.isposinf(boundary_targets.grad.numpy()[0])),
    "boundaryTargetNegativeInfinity": bool(np.isneginf(boundary_targets.grad.numpy()[1])),
    "dtypes": dtype_results,
    "scalar": float(scalar.item()),
    "scalarGradients": [float(scalar_probability.grad.item()), float(scalar_target.grad.item())],
    "emptyNoneShape": list(empty_none.shape),
    "emptyNoneSize": int(empty_none.data.size),
    "emptySum": float(empty_sum.item()),
    "emptyMeanIsNan": bool(np.isnan(empty_mean.item())),
    "emptyGradientShapes": [list(empty_grad_probability.grad.shape), list(empty_grad_target.grad.shape)],
    "snapshotGradients": [
        snapshot_probability.grad.numpy().reshape(-1).tolist(),
        snapshot_target.grad.numpy().reshape(-1).tolist(),
    ],
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "alias": float(alias.item()),
}
`);

    const fixture = FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE;
    expectCloseArray(result.none, fixture.base.noneValues);
    expectCloseArray(result.noneInputGradient, fixture.base.noneInputGradient, 4);
    expectCloseArray(result.noneTargetGradient, fixture.base.noneTargetGradient, 4);
    expect(result.sum).toBeCloseTo(fixture.base.sumValue, 5);
    expect(result.mean).toBeCloseTo(fixture.base.meanValue, 5);
    expectCloseArray(result.meanInputGradient, fixture.base.meanInputGradient, 5);
    expectCloseArray(result.meanTargetGradient, fixture.base.meanTargetGradient, 5);
    expect(result.boundaryValues).toEqual(fixture.boundaries.noneValues);
    expectCloseArray(result.boundaryInputGradients, fixture.boundaries.inputGradients, -5);
    expect(result.boundaryTargetPositiveInfinity).toBe(true);
    expect(result.boundaryTargetNegativeInfinity).toBe(true);
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map((entry) => ({
      output: entry.outputDtype,
      inputGradient: entry.inputDtype,
      targetGradient: entry.targetDtype,
    })));
    expect(result.scalar).toBeCloseTo(fixture.scalar.outputValue, 5);
    expectCloseArray(result.scalarGradients, [
      fixture.scalar.inputGradient,
      fixture.scalar.targetGradient,
    ]);
    expect(result.emptyNoneShape).toEqual(fixture.empty.noneShape);
    expect(result.emptyNoneSize).toBe(0);
    expect(result.emptySum).toBe(fixture.empty.sumValue);
    expect(result.emptyMeanIsNan).toBe(true);
    expect(result.emptyGradientShapes).toEqual([fixture.empty.shape, fixture.empty.shape]);
    const snapshots = result.snapshotGradients as number[][];
    expectCloseArray(snapshots[0], fixture.base.noneInputGradient, 4);
    expectCloseArray(snapshots[1], fixture.base.noneTargetGradient, 4);
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.alias).toBeCloseTo(2 * Math.log(4 / 3), 5);
  });

  it("rejects malformed, hostile-domain, array-subclass, and over-budget inputs", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE);
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

probabilities = tensor([0.25, 0.75])
targets = tensor([0.0, 1.0])
errors = {
    "input-type": error(lambda: F.bce_loss([0.25, 0.75], targets)),
    "target-type": error(lambda: F.bce_loss(probabilities, [0.0, 1.0])),
    "shape": error(lambda: F.bce_loss(probabilities, tensor([1.0]))),
    "reduction": error(lambda: F.bce_loss(probabilities, targets, reduction="median")),
    "reduction-type": error(lambda: F.bce_loss(probabilities, targets, reduction=object())),
    "input-dtype": error(lambda: F.bce_loss(tensor([0, 1], "int32"), targets)),
    "target-dtype": error(lambda: F.bce_loss(probabilities, tensor([0, 1], "int32"))),
    "input-low": error(lambda: F.bce_loss(tensor([-0.1, 0.5]), targets)),
    "input-high": error(lambda: F.bce_loss(tensor([0.5, 1.1]), targets)),
    "input-nan": error(lambda: F.bce_loss(tensor([float("nan"), 0.5]), targets)),
    "input-infinite": error(lambda: F.bce_loss(tensor([float("inf"), 0.5]), targets)),
    "target-low": error(lambda: F.bce_loss(probabilities, tensor([-0.1, 0.5]))),
    "target-high": error(lambda: F.bce_loss(probabilities, tensor([0.5, 1.1]))),
    "target-nan": error(lambda: F.bce_loss(probabilities, tensor([float("nan"), 0.5]))),
    "target-infinite": error(lambda: F.bce_loss(probabilities, tensor([float("inf"), 0.5]))),
}

class HostileArray(np.ndarray):
    pass

errors["array-subclass"] = error(lambda: F.bce_loss(
    grad.Tensor(np.array([0.25, 0.75], dtype=np.float32).view(HostileArray), dtype="float32"),
    targets,
))

def strided(shape, dtype):
    base = np.empty((1,), dtype=np.dtype(dtype))
    return np.lib.stride_tricks.as_strided(base, shape=shape, strides=(0,) * len(shape))

work_shape = (fixture["limits"]["workExtent"],)
errors["work"] = error(lambda: F.bce_loss(
    grad.Tensor(strided(work_shape, "float32"), dtype="float32"),
    grad.Tensor(strided(work_shape, "float32"), dtype="float32"),
    reduction="sum",
))
workspace_shape = (fixture["limits"]["workspaceExtent"],)
errors["workspace"] = error(lambda: F.bce_loss(
    grad.Tensor(strided(workspace_shape, "float64"), dtype="float64"),
    grad.Tensor(strided(workspace_shape, "float64"), dtype="float64"),
    reduction="sum",
))
zero_hidden_shape = (0, fixture["limits"]["workExtent"])
errors["zero-hidden-work"] = error(lambda: F.bce_loss(
    grad.Tensor(strided(zero_hidden_shape, "float32"), dtype="float32"),
    grad.Tensor(strided(zero_hidden_shape, "float32"), dtype="float32"),
    reduction="sum",
))
errors
`);

    for (const invalid of FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE.invalid) {
      if (invalid.id === "session") continue;
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["array-subclass"]).toMatch(/^TypeError: .*exact ndarray/u);
    expect(errors.work).toMatch(/^ValueError: .*work/u);
    expect(errors.workspace).toMatch(/^ValueError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ValueError: .*work/u);
  });
});
