import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE } from "../../../test-support/framework-smooth-l1-loss-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.nn.functional.smooth_l1_loss conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches shared reductions, promotion, both gradients, snapshots, empties, and aliases", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE);
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

left = tensor(base["shape"], base["inputValues"], requires_grad=True)
right = tensor(base["shape"], base["targetValues"], requires_grad=True)
none = F.smooth_l1_loss(left, right, beta=base["beta"], reduction="none")
none_array = none.numpy()
none.backward(tensor(base["shape"], base["upstreamValues"]))

summed = F.smooth_l1_loss(
    tensor(base["shape"], base["inputValues"]),
    tensor(base["shape"], base["targetValues"]),
    beta=base["beta"],
    reduction="sum",
)
mean_left = tensor(base["shape"], base["inputValues"], requires_grad=True)
mean_right = tensor(base["shape"], base["targetValues"], requires_grad=True)
mean = F.smooth_l1_loss(mean_left, mean_right, beta=base["beta"])
mean.backward()

beta_results = [
    float(F.smooth_l1_loss(
        tensor(base["shape"], base["inputValues"]),
        tensor(base["shape"], base["targetValues"]),
        beta=profile["beta"],
        reduction="sum",
    ).item())
    for profile in fixture["betaProfiles"]
]
zero_beta_left = tensor(base["shape"], base["inputValues"], requires_grad=True)
zero_beta_right = tensor(base["shape"], base["targetValues"], requires_grad=True)
F.smooth_l1_loss(
    zero_beta_left, zero_beta_right, beta=0.0, reduction="sum"
).backward()

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype_left = tensor([2], [0.25, -2.0], case["inputDtype"], True)
    dtype_right = tensor([2], [0.0, -4.0], case["targetDtype"], True)
    dtype_output = F.smooth_l1_loss(dtype_left, dtype_right, reduction="sum")
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "inputGradient": dtype_left.grad.dtype,
        "targetGradient": dtype_right.grad.dtype,
    })

scalar_fixture = fixture["scalar"]
scalar_left = tensor([], [scalar_fixture["inputValue"]], requires_grad=True)
scalar_right = tensor([], [scalar_fixture["targetValue"]], requires_grad=True)
scalar = F.smooth_l1_loss(
    scalar_left, scalar_right, beta=scalar_fixture["beta"]
)
scalar.backward()

empty = fixture["empty"]
empty_left = tensor(empty["shape"], [])
empty_right = tensor(empty["shape"], [])
empty_none = F.smooth_l1_loss(empty_left, empty_right, reduction="none")
empty_sum = F.smooth_l1_loss(empty_left, empty_right, reduction="sum")
empty_mean = F.smooth_l1_loss(empty_left, empty_right, reduction="mean")
empty_grad_left = tensor(empty["shape"], [], requires_grad=True)
empty_grad_right = tensor(empty["shape"], [], requires_grad=True)
F.smooth_l1_loss(empty_grad_left, empty_grad_right, reduction="mean").backward()

snapshot_left = tensor(base["shape"], base["inputValues"], requires_grad=True)
snapshot_right = tensor(base["shape"], base["targetValues"], requires_grad=True)
snapshot = F.smooth_l1_loss(
    snapshot_left, snapshot_right, beta=base["beta"], reduction="none"
)
snapshot_left.data[:] = -1000.0
snapshot_right.data[:] = 1000.0
snapshot.backward(tensor(base["shape"], base["upstreamValues"]))

with grad.no_grad():
    detached = F.smooth_l1_loss(
        tensor([1], [1.0], requires_grad=True),
        tensor([1], [0.0], requires_grad=True),
    )

grad.install_torch_alias()
import torch
alias = torch.nn.SmoothL1Loss(beta=0.5, reduction="sum")(
    torch.tensor([0.0, 1.0]),
    torch.tensor([0.0, 0.0]),
)

{
    "none": none_array.reshape(-1).tolist(),
    "noneShape": list(none.shape),
    "noneInputGradient": left.grad.numpy().reshape(-1).tolist(),
    "noneTargetGradient": right.grad.numpy().reshape(-1).tolist(),
    "sum": float(summed.item()),
    "mean": float(mean.item()),
    "meanInputGradient": mean_left.grad.numpy().reshape(-1).tolist(),
    "meanTargetGradient": mean_right.grad.numpy().reshape(-1).tolist(),
    "betaResults": beta_results,
    "zeroBetaGradients": [
        zero_beta_left.grad.numpy().reshape(-1).tolist(),
        zero_beta_right.grad.numpy().reshape(-1).tolist(),
    ],
    "dtypes": dtype_results,
    "scalar": float(scalar.item()),
    "scalarGradients": [float(scalar_left.grad.item()), float(scalar_right.grad.item())],
    "emptyNoneShape": list(empty_none.shape),
    "emptyNoneSize": int(empty_none.data.size),
    "emptySum": float(empty_sum.item()),
    "emptyMeanIsNan": bool(np.isnan(empty_mean.item())),
    "emptyGradientShapes": [list(empty_grad_left.grad.shape), list(empty_grad_right.grad.shape)],
    "snapshotGradients": [
        snapshot_left.grad.numpy().reshape(-1).tolist(),
        snapshot_right.grad.numpy().reshape(-1).tolist(),
    ],
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "alias": float(alias.item()),
}
`);

    const fixture = FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE;
    expect(result.none).toEqual(fixture.base.noneValues);
    expect(result.noneShape).toEqual(fixture.base.shape);
    expect(result.noneInputGradient).toEqual(fixture.base.noneInputGradient);
    expect(result.noneTargetGradient).toEqual(fixture.base.noneTargetGradient);
    expect(result.sum).toBe(fixture.base.sumValue);
    expect(result.mean).toBe(fixture.base.meanValue);
    expect(result.meanInputGradient).toEqual(fixture.base.meanInputGradient);
    expect(result.meanTargetGradient).toEqual(fixture.base.meanTargetGradient);
    expect(result.betaResults).toEqual(fixture.betaProfiles.map((entry) => entry.sumValue));
    expect(result.zeroBetaGradients).toEqual([
      fixture.zeroBetaGradients.input,
      fixture.zeroBetaGradients.target,
    ]);
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map((entry) => ({
      output: entry.outputDtype,
      inputGradient: entry.inputDtype,
      targetGradient: entry.targetDtype,
    })));
    expect(result.scalar).toBe(fixture.scalar.outputValue);
    expect(result.scalarGradients).toEqual([
      fixture.scalar.inputGradient,
      fixture.scalar.targetGradient,
    ]);
    expect(result.emptyNoneShape).toEqual(fixture.empty.noneShape);
    expect(result.emptyNoneSize).toBe(0);
    expect(result.emptySum).toBe(fixture.empty.sumValue);
    expect(result.emptyMeanIsNan).toBe(true);
    expect(result.emptyGradientShapes).toEqual([fixture.empty.shape, fixture.empty.shape]);
    expect(result.snapshotGradients).toEqual([
      fixture.base.noneInputGradient,
      fixture.base.noneTargetGradient,
    ]);
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.alias).toBe(0.75);
  });

  it("rejects malformed, hostile, and over-budget requests before numerical work", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE);
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

left = grad.Tensor(np.ones((2,), dtype=np.float32), dtype="float32")
right = grad.Tensor(np.zeros((2,), dtype=np.float32), dtype="float32")

def strided(shape, dtype):
    base = np.empty((1,), dtype=np.dtype(dtype))
    return np.lib.stride_tricks.as_strided(base, shape=shape, strides=(0,) * len(shape))

class HostileBeta:
    def __float__(self):
        raise RuntimeError("must not execute")

errors = {
    "input-type": error(lambda: F.smooth_l1_loss([1.0, 2.0], right)),
    "shape": error(lambda: F.smooth_l1_loss(
        left, grad.Tensor(np.zeros((1,), dtype=np.float32), dtype="float32")
    )),
    "reduction": error(lambda: F.smooth_l1_loss(left, right, reduction="median")),
    "reduction-type": error(lambda: F.smooth_l1_loss(left, right, reduction=object())),
    "input-dtype": error(lambda: F.smooth_l1_loss(
        grad.Tensor(np.ones((2,), dtype=np.int32), dtype="int32"), right
    )),
    "target-dtype": error(lambda: F.smooth_l1_loss(
        left, grad.Tensor(np.ones((2,), dtype=np.int32), dtype="int32")
    )),
    "target-type": error(lambda: F.smooth_l1_loss(left, [0.0, 0.0])),
    "beta-negative": error(lambda: F.smooth_l1_loss(left, right, beta=-0.5)),
    "beta-nan": error(lambda: F.smooth_l1_loss(left, right, beta=float("nan"))),
    "beta-infinite": error(lambda: F.smooth_l1_loss(left, right, beta=float("inf"))),
    "beta-type": error(lambda: F.smooth_l1_loss(left, right, beta=object())),
    "beta-bool": error(lambda: F.smooth_l1_loss(left, right, beta=True)),
    "beta-underflow": error(lambda: F.smooth_l1_loss(left, right, beta=1e-100)),
    "beta-overflow": error(lambda: F.smooth_l1_loss(left, right, beta=1e100)),
    "beta-hostile": error(lambda: F.smooth_l1_loss(left, right, beta=HostileBeta())),
}

class HostileArray(np.ndarray):
    pass

errors["array-subclass"] = error(lambda: F.smooth_l1_loss(
    grad.Tensor(np.ones((2,), dtype=np.float32).view(HostileArray), dtype="float32"),
    right,
))
work_shape = (fixture["limits"]["workExtent"],)
errors["work"] = error(lambda: F.smooth_l1_loss(
    grad.Tensor(strided(work_shape, "float32"), dtype="float32"),
    grad.Tensor(strided(work_shape, "float32"), dtype="float32"),
    reduction="sum",
))
workspace_shape = (fixture["limits"]["workspaceExtent"],)
errors["workspace"] = error(lambda: F.smooth_l1_loss(
    grad.Tensor(strided(workspace_shape, "float64"), dtype="float64"),
    grad.Tensor(strided(workspace_shape, "float64"), dtype="float64"),
    reduction="sum",
))
zero_hidden_shape = (0, fixture["limits"]["workExtent"])
errors["zero-hidden-work"] = error(lambda: F.smooth_l1_loss(
    grad.Tensor(strided(zero_hidden_shape, "float32"), dtype="float32"),
    grad.Tensor(strided(zero_hidden_shape, "float32"), dtype="float32"),
    reduction="sum",
))
errors
`);

    expect(errors.shape).toContain("must equal target shape");
    expect(errors.reduction).toContain("must be 'none', 'sum', or 'mean'");
    expect(errors["reduction-type"]).toContain("reduction must be a string");
    expect(errors["input-dtype"]).toContain("input dtype");
    expect(errors["target-dtype"]).toContain("target dtype");
    expect(errors["target-type"]).toContain("target must be a Tensor");
    expect(errors["beta-negative"]).toContain("beta must be non-negative");
    expect(errors["beta-nan"]).toContain("beta must be finite");
    expect(errors["beta-infinite"]).toContain("beta must be finite");
    expect(errors["beta-type"]).toContain("beta must be an exact real scalar");
    expect(errors["beta-bool"]).toContain("beta must be an exact real scalar");
    expect(errors["beta-underflow"]).toContain("not representable");
    expect(errors["beta-overflow"]).toContain("not representable");
    expect(errors["beta-hostile"]).toContain("beta must be an exact real scalar");
    expect(errors["array-subclass"]).toMatch(/^TypeError: .*exact ndarray/u);
    expect(errors.work).toMatch(/^ValueError: .*work/u);
    expect(errors.workspace).toMatch(/^ValueError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ValueError: .*work/u);
  });
});
