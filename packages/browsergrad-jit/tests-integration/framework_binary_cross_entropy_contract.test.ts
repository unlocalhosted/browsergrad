import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE } from "../../../test-support/framework-binary-cross-entropy-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 typed torch.nn.functional.binary_cross_entropy contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches clamped probability reductions, both gradients, endpoints, promotion, and aliases", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
base = fixture["base"]

def tensor(shape, values, dtype="float32", requires_grad=False):
    return bg.from_numpy(
        np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape)),
        requires_grad=requires_grad,
    )

probabilities = tensor(base["shape"], base["inputValues"], requires_grad=True)
targets = tensor(base["shape"], base["targetValues"], requires_grad=True)
none = F.binary_cross_entropy(probabilities, targets, reduction="none")
none_array = none.numpy()
upstream = tensor(base["shape"], base["upstreamValues"])
closure_gradients = none._ctx.fn(
    upstream.numpy(), (probabilities.numpy(), targets.numpy())
)
none.backward(upstream)

summed = F.binary_cross_entropy(
    tensor(base["shape"], base["inputValues"]),
    tensor(base["shape"], base["targetValues"]),
    reduction="sum",
)
mean_probabilities = tensor(base["shape"], base["inputValues"], requires_grad=True)
mean_targets = tensor(base["shape"], base["targetValues"], requires_grad=True)
mean = F.binary_cross_entropy(mean_probabilities, mean_targets)
mean.backward()

boundary = fixture["boundaries"]
boundary_probabilities = tensor([2], boundary["inputValues"], requires_grad=True)
boundary_targets = tensor([2], boundary["targetValues"], requires_grad=True)
boundary_loss = F.binary_cross_entropy(
    boundary_probabilities, boundary_targets, reduction="none"
)
boundary_values = boundary_loss.numpy()
boundary_loss.backward(tensor([2], [1.0, 1.0]))

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype_probabilities = tensor([2], [0.25, 0.75], case["inputDtype"], True)
    dtype_targets = tensor([2], [0.0, 1.0], case["targetDtype"], True)
    dtype_output = F.binary_cross_entropy(
        dtype_probabilities, dtype_targets, reduction="sum"
    )
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "inputGradient": dtype_probabilities.grad.dtype,
        "targetGradient": dtype_targets.grad.dtype,
    })

scalar_fixture = fixture["scalar"]
scalar_probability = tensor([], [scalar_fixture["inputValue"]], requires_grad=True)
scalar_target = tensor([], [scalar_fixture["targetValue"]], requires_grad=True)
scalar = F.binary_cross_entropy(scalar_probability, scalar_target)
scalar.backward()

empty = fixture["empty"]
empty_probability = tensor(empty["shape"], [])
empty_target = tensor(empty["shape"], [])
empty_none = F.binary_cross_entropy(empty_probability, empty_target, reduction="none")
empty_sum = F.binary_cross_entropy(empty_probability, empty_target, reduction="sum")
empty_mean = F.binary_cross_entropy(empty_probability, empty_target, reduction="mean")
empty_grad_probability = tensor(empty["shape"], [], requires_grad=True)
empty_grad_target = tensor(empty["shape"], [], requires_grad=True)
F.binary_cross_entropy(empty_grad_probability, empty_grad_target).backward()

with bg.no_grad():
    detached = F.binary_cross_entropy(
        tensor([1], [0.25], requires_grad=True),
        tensor([1], [1.0], requires_grad=True),
    )

bg.install_torch_alias()
import torch
alias = torch.nn.BCELoss(reduction="sum")(
    torch.tensor([0.25, 0.75]),
    torch.tensor([0.0, 1.0]),
)
functional_alias = F.bce_loss(
    tensor([2], [0.25, 0.75]),
    tensor([2], [0.0, 1.0]),
    reduction="sum",
)
parameter = bg.nn.Parameter(tensor([2], [0.25, 0.75]))
parameter_loss = F.binary_cross_entropy(
    parameter, tensor([2], [0.0, 1.0]), reduction="sum"
)

{
    "op": none._uop.op,
    "arg": none._uop.arg,
    "none": none_array.reshape(-1).tolist(),
    "noneInputGradient": probabilities.grad.numpy().reshape(-1).tolist(),
    "noneTargetGradient": targets.grad.numpy().reshape(-1).tolist(),
    "closureGradients": [value.reshape(-1).tolist() for value in closure_gradients],
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
    "emptyNoneSize": int(empty_none.numpy().size),
    "emptySum": float(empty_sum.item()),
    "emptyMeanIsNan": bool(np.isnan(empty_mean.item())),
    "emptyGradientShapes": [list(empty_grad_probability.grad.shape), list(empty_grad_target.grad.shape)],
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "alias": float(alias.item()),
    "functionalAlias": float(functional_alias.item()),
    "parameter": float(parameter_loss.item()),
}
`);

    const fixture = FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE;
    expect(result.op).toBe("BINARY_CROSS_ENTROPY");
    expect(result.arg).toEqual({ reduction: "none", batch_rank: 0 });
    expectCloseArray(result.none, fixture.base.noneValues);
    expectCloseArray(result.noneInputGradient, fixture.base.noneInputGradient, 4);
    expectCloseArray(result.noneTargetGradient, fixture.base.noneTargetGradient, 4);
    const closure = result.closureGradients as number[][];
    expectCloseArray(closure[0], fixture.base.noneInputGradient, 4);
    expectCloseArray(closure[1], fixture.base.noneTargetGradient, 4);
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
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.alias).toBeCloseTo(2 * Math.log(4 / 3), 5);
    expect(result.functionalAlias).toBeCloseTo(2 * Math.log(4 / 3), 5);
    expect(result.parameter).toBeCloseTo(2 * Math.log(4 / 3), 5);
  });

  it("supports symbolic VJP and nested vmap while refusing unrepresentable export and device paths", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._ir import OP_BINARY_CROSS_ENTROPY, OP_BINARY_CROSS_ENTROPY_VJP, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

probabilities = bg.from_numpy(np.array([0.25, 0.75, 0.5, 0.2], dtype=np.float32), requires_grad=True)
targets = bg.from_numpy(np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float32), requires_grad=True)
loss = F.binary_cross_entropy(probabilities, targets, reduction="mean")
input_gradient = bg.func.grad(
    lambda value: F.binary_cross_entropy(value, targets, reduction="sum")
)(probabilities)
target_gradient = bg.func.grad(
    lambda value: F.binary_cross_entropy(probabilities, value, reduction="sum")
)(targets)

batched_probabilities = bg.from_numpy(
    np.array([[0.25, 0.75], [0.5, 0.2]], dtype=np.float32)
)
batched_targets = bg.from_numpy(
    np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
)
mapped_mean = bg.func.vmap(
    lambda a, b: F.binary_cross_entropy(a, b, reduction="mean")
)(batched_probabilities, batched_targets)
captured = bg.from_numpy(np.array([0.0, 1.0], dtype=np.float32))
mapped_sum = bg.func.vmap(
    lambda value: F.binary_cross_entropy(value, captured, reduction="sum")
)(batched_probabilities)
mapped_none = bg.func.vmap(
    lambda a, b: F.binary_cross_entropy(a, b, reduction="none")
)(batched_probabilities, batched_targets)
nested_probabilities = bg.from_numpy(
    np.tile(batched_probabilities.numpy(), (2, 1, 1))
)
nested_targets = bg.from_numpy(np.tile(batched_targets.numpy(), (2, 1, 1)))
nested_mean = bg.func.vmap(bg.func.vmap(
    lambda a, b: F.binary_cross_entropy(a, b, reduction="mean")
))(nested_probabilities, nested_targets)

support = next(
    item for item in bg.framework_operation_support()["operations"]
    if item["opcode"] == OP_BINARY_CROSS_ENTROPY
)
symbolic = get_rule(OP_BINARY_CROSS_ENTROPY)(
    loss._uop, loss._uop.inputs, bg.ones()._uop
)

{
    "registered": get_rule(OP_BINARY_CROSS_ENTROPY) is not None,
    "inputGradient": input_gradient.numpy().tolist(),
    "targetGradient": target_gradient.numpy().tolist(),
    "symbolicOps": [[node.op for node in toposort(value)] for value in symbolic],
    "mappedMean": mapped_mean.numpy().tolist(),
    "mappedMeanBatchRank": mapped_mean._uop.arg["batch_rank"],
    "mappedSum": mapped_sum.numpy().tolist(),
    "mappedNone": mapped_none.numpy().tolist(),
    "nestedMean": nested_mean.numpy().tolist(),
    "nestedMeanBatchRank": nested_mean._uop.arg["batch_rank"],
    "onnxError": error(lambda: bg.onnx.export_inference(
        loss, input_buffers=(probabilities, targets)
    )),
    "planError": error(lambda: bg.gpu_plan_summary(loss)),
    "webgpuSupported": (
        OP_BINARY_CROSS_ENTROPY in supported_opcodes()
        or OP_BINARY_CROSS_ENTROPY_VJP in supported_opcodes()
    ),
    "support": support,
}
`);

    expect(result.registered).toBe(true);
    expectCloseArray(result.inputGradient, [4 / 3, -4 / 3, 1, -3.4375], 4);
    expectCloseArray(result.targetGradient, [Math.log(3), -Math.log(3), 0, Math.log(4)], 4);
    for (const ops of result.symbolicOps as string[][]) {
      expect(ops).toContain("BINARY_CROSS_ENTROPY_VJP");
      expect(ops).not.toContain("CUSTOM");
    }
    expectCloseArray(result.mappedMean, [Math.log(4 / 3),
      (Math.log(2) + 1.2628643221541278) / 2]);
    expect(result.mappedMeanBatchRank).toBe(1);
    expectCloseArray(result.mappedSum, [2 * Math.log(4 / 3),
      -Math.log(0.5) - Math.log(0.2)]);
    expect((result.mappedNone as number[][])).toHaveLength(2);
    expectCloseArray((result.mappedNone as number[][])[0], [Math.log(4 / 3), Math.log(4 / 3)]);
    expectCloseArray((result.mappedNone as number[][])[1], [Math.log(2), 1.2628643221541278]);
    expect(result.nestedMeanBatchRank).toBe(2);
    for (const row of result.nestedMean as number[][]) {
      expectCloseArray(row, [Math.log(4 / 3), (Math.log(2) + 1.2628643221541278) / 2]);
    }
    expect(result.onnxError).toMatch(/^OnnxUnmappableOp: .*fail-closed runtime validation/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*BINARY_CROSS_ENTROPY/u);
    expect(result.webgpuSupported).toBe(false);
    expect(result.support).toMatchObject({
      contractId: "browsergrad.jit.framework.functional.binary-cross-entropy.v1",
      publicSurface: "torch.nn.functional.binary_cross_entropy",
      opcode: "BINARY_CROSS_ENTROPY",
      semanticState: "typed",
      retiredOpaqueOperationId: "jit.custom.binary-cross-entropy.v0",
      decisions: {
        onnxExport: "refused-runtime-probability-domain-cannot-fail-closed",
      },
    });
  });

  it("rejects malformed, hostile-domain, mutated, and over-budget requests", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_BINARY_CROSS_ENTROPY
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

def tensor(values, dtype="float32", session=None):
    return bg.from_numpy(np.asarray(values, dtype=np.dtype(dtype)), session=session)

probabilities = tensor([0.25, 0.75])
targets = tensor([0.0, 1.0])
other_targets = tensor([0.0, 1.0], session=bg.new_session())

errors = {
    "input-type": error(lambda: F.binary_cross_entropy([0.25, 0.75], targets)),
    "target-type": error(lambda: F.binary_cross_entropy(probabilities, [0.0, 1.0])),
    "shape": error(lambda: F.binary_cross_entropy(probabilities, tensor([1.0]))),
    "session": error(lambda: F.binary_cross_entropy(probabilities, other_targets)),
    "reduction": error(lambda: F.binary_cross_entropy(probabilities, targets, reduction="median")),
    "reduction-type": error(lambda: F.binary_cross_entropy(probabilities, targets, reduction=object())),
    "input-dtype": error(lambda: F.binary_cross_entropy(tensor([0, 1], "int32"), targets)),
    "target-dtype": error(lambda: F.binary_cross_entropy(probabilities, tensor([0, 1], "int32"))),
    "input-low": error(lambda: F.binary_cross_entropy(tensor([-0.1, 0.5]), targets).numpy()),
    "input-high": error(lambda: F.binary_cross_entropy(tensor([0.5, 1.1]), targets).numpy()),
    "input-nan": error(lambda: F.binary_cross_entropy(tensor([float("nan"), 0.5]), targets).numpy()),
    "input-infinite": error(lambda: F.binary_cross_entropy(tensor([float("inf"), 0.5]), targets).numpy()),
    "target-low": error(lambda: F.binary_cross_entropy(probabilities, tensor([-0.1, 0.5])).numpy()),
    "target-high": error(lambda: F.binary_cross_entropy(probabilities, tensor([0.5, 1.1])).numpy()),
    "target-nan": error(lambda: F.binary_cross_entropy(probabilities, tensor([float("nan"), 0.5])).numpy()),
    "target-infinite": error(lambda: F.binary_cross_entropy(probabilities, tensor([float("inf"), 0.5])).numpy()),
}

class HostileArray(np.ndarray):
    pass

hostile_source = bg.from_numpy(np.array([0.25, 0.75], dtype=np.float32).view(HostileArray))
errors["array-subclass"] = error(lambda: F.binary_cross_entropy(
    hostile_source, targets, reduction="sum"
).numpy())

valid = F.binary_cross_entropy(probabilities, targets, reduction="sum")
valid._uop.arg["reduction"] = "median"
dy = UOp(OP_BUFFER, (), (), "float32", arg="dy")
errors["mutated-cpu"] = error(valid.numpy)
errors["mutated-vjp"] = error(lambda: get_rule(OP_BINARY_CROSS_ENTROPY)(
    valid._uop, valid._uop.inputs, dy
))
errors["mutated-vmap"] = error(lambda: get_vmap_rule(OP_BINARY_CROSS_ENTROPY)(
    valid._uop, {id(node): node for node in valid._uop.inputs}, 2
))
errors["mutated-onnx"] = error(lambda: bg.onnx.export_inference(
    valid, input_buffers=(probabilities, targets)
))
errors["mutated-plan"] = error(lambda: bg.gpu_plan_summary(valid))

def forged(shape, dtype="float32"):
    lhs = UOp(OP_BUFFER, (), shape, dtype, arg="lhs-" + str(shape))
    rhs = UOp(OP_BUFFER, (), shape, dtype, arg="rhs-" + str(shape))
    return UOp(
        OP_BINARY_CROSS_ENTROPY,
        (lhs, rhs),
        (),
        dtype,
        arg={"reduction": "sum", "batch_rank": 0},
    )

errors["rank"] = error(lambda: forged((1,) * 33))
errors["work"] = error(lambda: forged((fixture["limits"]["workExtent"],)))
errors["workspace"] = error(lambda: forged(
    (fixture["limits"]["workspaceExtent"],), "float64"
))
errors["zero-hidden-work"] = error(lambda: forged(
    (0, fixture["limits"]["workExtent"])
))

wrong_left = UOp(OP_BUFFER, (), (2,), "float32", arg="wrong-left")
wrong_right = UOp(OP_BUFFER, (), (2,), "float32", arg="wrong-right")
errors["declared-shape"] = error(lambda: UOp(
    OP_BINARY_CROSS_ENTROPY,
    (wrong_left, wrong_right),
    (1,),
    "float32",
    arg={"reduction": "sum", "batch_rank": 0},
))
errors["declared-dtype"] = error(lambda: UOp(
    OP_BINARY_CROSS_ENTROPY,
    (wrong_left, wrong_right),
    (),
    "float64",
    arg={"reduction": "sum", "batch_rank": 0},
))
errors
`);

    for (const invalid of FRAMEWORK_BINARY_CROSS_ENTROPY_CONFORMANCE.invalid) {
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["mutated-cpu"]).toMatch(/^RealizationError: .*reduction/u);
    expect(errors["mutated-vjp"]).toMatch(/^ShapeError: .*reduction/u);
    expect(errors["mutated-vmap"]).toMatch(/^ShapeError: .*reduction/u);
    expect(errors["mutated-onnx"]).toMatch(/^ShapeError: .*reduction/u);
    expect(errors["mutated-plan"]).toMatch(/^ShapeError: .*reduction/u);
    expect(errors.rank).toMatch(/^ShapeError: .*rank/u);
    expect(errors.work).toMatch(/^ShapeError: .*work/u);
    expect(errors.workspace).toMatch(/^ShapeError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ShapeError: .*work/u);
    expect(errors["declared-shape"]).toMatch(/^ShapeError: .*declared shape/u);
    expect(errors["declared-dtype"]).toMatch(/^ShapeError: .*declared dtype/u);
    expect(errors["array-subclass"]).toMatch(/^RealizationError: .*exact ndarray/u);
  });
});
