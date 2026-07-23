import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_KL_DIV_CONFORMANCE } from "../../../test-support/framework-kl-div-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 eager KL-divergence conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches reductions, both target modes, both gradients, snapshots, zeros, empties, and aliases", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_KL_DIV_CONFORMANCE);
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

input_value = tensor(base["shape"], base["input"], requires_grad=True)
target_value = tensor(base["shape"], base["target"], requires_grad=True)
none = F.kl_div_loss(input_value, target_value, reduction="none")
none_array = none.numpy()
none.backward(tensor(base["shape"], base["upstream"]))

reductions = {}
for reduction in ("sum", "mean", "batchmean"):
    reductions[reduction] = float(F.kl_div_loss(
        tensor(base["shape"], base["input"]),
        tensor(base["shape"], base["target"]),
        reduction=reduction,
    ).item())

log_input = tensor(base["shape"], base["input"], requires_grad=True)
log_target = tensor(
    base["shape"], np.log(np.asarray(base["target"])), requires_grad=True
)
log_loss = F.kl_div_loss(log_input, log_target, reduction="none", log_target=True)
log_values = log_loss.numpy()
log_loss.backward(tensor(base["shape"], base["upstream"]))

zero = fixture["zeroTarget"]
zero_input = tensor([2], zero["input"], requires_grad=True)
zero_target = tensor([2], zero["target"], requires_grad=True)
zero_loss = F.kl_div_loss(zero_input, zero_target, reduction="none")
zero_values = zero_loss.numpy()
zero_loss.backward(tensor([2], [1, 1]))

snapshot_input = tensor(base["shape"], base["input"], requires_grad=True)
snapshot_target = tensor(base["shape"], base["target"], requires_grad=True)
snapshot = F.kl_div_loss(snapshot_input, snapshot_target, reduction="none")
snapshot_input.data[:] = 7
snapshot_target.data[:] = 0.5
snapshot.backward(tensor(base["shape"], base["upstream"]))

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype_input = tensor([2], [-0.3, -1.2], case["inputDtype"], True)
    dtype_target = tensor([2], [0.7, 0.3], case["targetDtype"], True)
    dtype_output = F.kl_div_loss(dtype_input, dtype_target, reduction="sum")
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "inputGradient": dtype_input.grad.dtype,
        "targetGradient": dtype_target.grad.dtype,
    })

scalar_fixture = fixture["scalar"]
scalar_input = tensor([], [scalar_fixture["input"]], requires_grad=True)
scalar_target = tensor([], [scalar_fixture["target"]], requires_grad=True)
scalar = F.kl_div_loss(scalar_input, scalar_target, reduction="batchmean")
scalar.backward()

empty = fixture["empty"]
zero_batch_input = tensor(empty["zeroBatchShape"], [])
zero_batch_target = tensor(empty["zeroBatchShape"], [])
zero_support_input = tensor(empty["zeroSupportShape"], [])
zero_support_target = tensor(empty["zeroSupportShape"], [])
empty_results = {
    "zeroBatchSum": float(F.kl_div_loss(zero_batch_input, zero_batch_target, reduction="sum").item()),
    "zeroBatchMeanNan": bool(np.isnan(F.kl_div_loss(zero_batch_input, zero_batch_target, reduction="mean").item())),
    "zeroBatchBatchmeanNan": bool(np.isnan(F.kl_div_loss(zero_batch_input, zero_batch_target, reduction="batchmean").item())),
    "zeroSupportBatchmean": float(F.kl_div_loss(zero_support_input, zero_support_target, reduction="batchmean").item()),
}

with grad.no_grad():
    detached = F.kl_div_loss(
        tensor([1], [-0.3], requires_grad=True),
        tensor([1], [0.7], requires_grad=True),
    )

grad.install_torch_alias()
import torch
module_alias = torch.nn.KLDivLoss(reduction="batchmean")(
    torch.tensor(np.asarray(base["input"]).reshape(base["shape"])),
    torch.tensor(np.asarray(base["target"]).reshape(base["shape"])),
)
functional_alias = torch.nn.functional.kl_div(
    torch.tensor(np.asarray(base["input"]).reshape(base["shape"])),
    torch.tensor(np.asarray(base["target"]).reshape(base["shape"])),
    reduction="batchmean",
)

{
    "none": none_array.reshape(-1).tolist(),
    "noneInputGradient": input_value.grad.numpy().reshape(-1).tolist(),
    "noneTargetGradient": target_value.grad.numpy().reshape(-1).tolist(),
    "reductions": reductions,
    "logValues": log_values.reshape(-1).tolist(),
    "logInputGradient": log_input.grad.numpy().reshape(-1).tolist(),
    "logTargetGradient": log_target.grad.numpy().reshape(-1).tolist(),
    "zeroValues": zero_values.tolist(),
    "zeroInputGradient": zero_input.grad.numpy().tolist(),
    "zeroTargetGradientNan": bool(np.isnan(zero_target.grad.numpy()[0])),
    "snapshotGradients": [
        snapshot_input.grad.numpy().reshape(-1).tolist(),
        snapshot_target.grad.numpy().reshape(-1).tolist(),
    ],
    "dtypes": dtype_results,
    "scalar": float(scalar.item()),
    "scalarGradients": [float(scalar_input.grad.item()), float(scalar_target.grad.item())],
    "empty": empty_results,
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "moduleAlias": float(module_alias.item()),
    "functionalAlias": float(functional_alias.item()),
}
`);

    const fixture = FRAMEWORK_KL_DIV_CONFORMANCE;
    expectCloseArray(result.none, fixture.base.noneValues);
    expectCloseArray(result.noneInputGradient, fixture.base.noneInputGradient);
    expectCloseArray(result.noneTargetGradient, fixture.base.noneTargetGradient);
    expect(result.reductions).toEqual({
      sum: expect.closeTo(fixture.base.sumValue, 5),
      mean: expect.closeTo(fixture.base.meanValue, 5),
      batchmean: expect.closeTo(fixture.base.batchmeanValue, 5),
    });
    expectCloseArray(result.logValues, fixture.base.noneValues);
    expectCloseArray(result.logInputGradient, fixture.base.noneInputGradient);
    expectCloseArray(result.logTargetGradient, fixture.base.noneLogTargetGradient);
    expectCloseArray(result.zeroValues, fixture.zeroTarget.noneValues);
    expectCloseArray(result.zeroInputGradient, fixture.zeroTarget.inputGradients);
    expect(result.zeroTargetGradientNan).toBe(true);
    const snapshots = result.snapshotGradients as number[][];
    expectCloseArray(snapshots[0], fixture.base.noneInputGradient);
    expectCloseArray(snapshots[1], fixture.base.noneTargetGradient);
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map((entry) => ({
      output: entry.outputDtype,
      inputGradient: entry.inputDtype,
      targetGradient: entry.targetDtype,
    })));
    expect(result.scalar).toBeCloseTo(fixture.scalar.output, 5);
    expectCloseArray(result.scalarGradients, [fixture.scalar.inputGradient, fixture.scalar.targetGradient]);
    expect(result.empty).toEqual({
      zeroBatchSum: fixture.empty.sumValue,
      zeroBatchMeanNan: true,
      zeroBatchBatchmeanNan: true,
      zeroSupportBatchmean: fixture.empty.zeroSupportBatchmeanValue,
    });
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.moduleAlias).toBeCloseTo(fixture.base.batchmeanValue, 5);
    expect(result.functionalAlias).toBeCloseTo(fixture.base.batchmeanValue, 5);
  });

  it("rejects malformed, ndarray-subclass, deprecated keywords, and hostile resources", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_KL_DIV_CONFORMANCE);
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

input_value = tensor([-0.3, -1.2])
target_value = tensor([0.7, 0.3])
errors = {
    "input-type": error(lambda: F.kl_div_loss([-0.3, -1.2], target_value)),
    "target-type": error(lambda: F.kl_div_loss(input_value, [0.7, 0.3])),
    "shape": error(lambda: F.kl_div_loss(input_value, tensor([0.7]))),
    "reduction": error(lambda: F.kl_div_loss(input_value, target_value, reduction="median")),
    "reduction-type": error(lambda: F.kl_div_loss(input_value, target_value, reduction=object())),
    "log-target-type": error(lambda: F.kl_div_loss(input_value, target_value, log_target=1)),
    "input-dtype": error(lambda: F.kl_div_loss(tensor([-1, -2], "int32"), target_value)),
    "target-dtype": error(lambda: F.kl_div_loss(input_value, tensor([1, 0], "int32"))),
    "size-average": error(lambda: F.kl_div_loss(input_value, target_value, size_average=True)),
    "reduce": error(lambda: F.kl_div_loss(input_value, target_value, reduce=True)),
}

class HostileArray(np.ndarray):
    pass
errors["array-subclass"] = error(lambda: F.kl_div_loss(
    grad.Tensor(np.array([-0.3, -1.2], dtype=np.float32).view(HostileArray), dtype="float32"),
    target_value,
))

def strided(shape, dtype):
    base = np.empty((1,), dtype=np.dtype(dtype))
    return np.lib.stride_tricks.as_strided(base, shape=shape, strides=(0,) * len(shape))

work_shape = (fixture["limits"]["workExtent"],)
errors["work"] = error(lambda: F.kl_div_loss(
    grad.Tensor(strided(work_shape, "float32"), dtype="float32"),
    grad.Tensor(strided(work_shape, "float32"), dtype="float32"),
    reduction="sum",
))
workspace_shape = (fixture["limits"]["workspaceExtent"],)
errors["workspace"] = error(lambda: F.kl_div_loss(
    grad.Tensor(strided(workspace_shape, "float64"), dtype="float64"),
    grad.Tensor(strided(workspace_shape, "float64"), dtype="float64"),
    reduction="sum",
))
zero_hidden_shape = (0, fixture["limits"]["workExtent"])
errors["zero-hidden-work"] = error(lambda: F.kl_div_loss(
    grad.Tensor(strided(zero_hidden_shape, "float32"), dtype="float32"),
    grad.Tensor(strided(zero_hidden_shape, "float32"), dtype="float32"),
    reduction="sum",
))
errors
`);

    for (const invalid of FRAMEWORK_KL_DIV_CONFORMANCE.invalid) {
      if (invalid.id === "session") continue;
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["array-subclass"]).toMatch(/^TypeError: .*exact ndarray/u);
    expect(errors.work).toMatch(/^ValueError: .*work/u);
    expect(errors.workspace).toMatch(/^ValueError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ValueError: .*work/u);
  });
});
