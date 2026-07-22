import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_L1_LOSS_CONFORMANCE } from "../../../test-support/framework-l1-loss-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.nn.functional.l1_loss contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches shared reductions, both-input gradients, promotion, empties, and aliases", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_L1_LOSS_CONFORMANCE);
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

left = tensor(base["shape"], base["inputValues"], requires_grad=True)
right = tensor(base["shape"], base["targetValues"], requires_grad=True)
none = F.l1_loss(left, right, reduction="none")
none_array = none.numpy()
upstream = tensor(base["shape"], base["upstreamValues"])
closure_gradients = none._ctx.fn(
    upstream.numpy(),
    (left.numpy(), right.numpy()),
)
none.backward(upstream)

summed = F.l1_loss(
    tensor(base["shape"], base["inputValues"]),
    tensor(base["shape"], base["targetValues"]),
    reduction="sum",
)

mean_left = tensor(base["shape"], base["inputValues"], requires_grad=True)
mean_right = tensor(base["shape"], base["targetValues"], requires_grad=True)
mean = F.l1_loss(mean_left, mean_right)
mean.backward()

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype_left = tensor([2], [1.0, -2.0], case["inputDtype"], True)
    dtype_right = tensor([2], [0.0, -4.0], case["targetDtype"], True)
    dtype_output = F.l1_loss(dtype_left, dtype_right, reduction="sum")
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "inputGradient": dtype_left.grad.dtype,
        "targetGradient": dtype_right.grad.dtype,
    })

scalar_fixture = fixture["scalar"]
scalar_left = tensor([], [scalar_fixture["inputValue"]], requires_grad=True)
scalar_right = tensor([], [scalar_fixture["targetValue"]], requires_grad=True)
scalar = F.l1_loss(scalar_left, scalar_right)
scalar.backward()

empty = fixture["empty"]
empty_left = tensor(empty["shape"], [])
empty_right = tensor(empty["shape"], [])
empty_none = F.l1_loss(empty_left, empty_right, reduction="none")
empty_sum = F.l1_loss(empty_left, empty_right, reduction="sum")
empty_mean = F.l1_loss(empty_left, empty_right, reduction="mean")

with bg.no_grad():
    detached = F.l1_loss(
        tensor([1], [1.0], requires_grad=True),
        tensor([1], [0.0], requires_grad=True),
    )

bg.install_torch_alias()
import torch
alias = torch.nn.L1Loss(reduction="sum")(
    torch.tensor([1.0, -2.0]),
    torch.tensor([0.0, -1.0]),
)
parameter = bg.nn.Parameter(tensor([2], [1.0, -2.0]))
parameter_loss = F.l1_loss(parameter, tensor([2], [0.0, -1.0]), reduction="sum")

{
    "op": none._uop.op,
    "arg": none._uop.arg,
    "none": none_array.reshape(-1).tolist(),
    "noneShape": list(none.shape),
    "noneInputGradient": left.grad.numpy().reshape(-1).tolist(),
    "noneTargetGradient": right.grad.numpy().reshape(-1).tolist(),
    "closureGradients": [value.reshape(-1).tolist() for value in closure_gradients],
    "sum": float(summed.item()),
    "mean": float(mean.item()),
    "meanInputGradient": mean_left.grad.numpy().reshape(-1).tolist(),
    "meanTargetGradient": mean_right.grad.numpy().reshape(-1).tolist(),
    "dtypes": dtype_results,
    "scalar": float(scalar.item()),
    "scalarGradients": [float(scalar_left.grad.item()), float(scalar_right.grad.item())],
    "emptyNoneShape": list(empty_none.shape),
    "emptyNoneSize": int(empty_none.numpy().size),
    "emptySum": float(empty_sum.item()),
    "emptyMeanIsNan": bool(np.isnan(empty_mean.item())),
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "alias": float(alias.item()),
    "parameter": float(parameter_loss.item()),
}
`);

    const fixture = FRAMEWORK_L1_LOSS_CONFORMANCE;
    expect(result.op).toBe("L1_LOSS");
    expect(result.arg).toEqual({ reduction: "none", batch_rank: 0 });
    expect(result.none).toEqual(fixture.base.noneValues);
    expect(result.noneShape).toEqual(fixture.base.shape);
    expect(result.noneInputGradient).toEqual(fixture.base.noneInputGradient);
    expect(result.noneTargetGradient).toEqual(fixture.base.noneTargetGradient);
    expect(result.closureGradients).toEqual([
      fixture.base.noneInputGradient,
      fixture.base.noneTargetGradient,
    ]);
    expect(result.sum).toBe(fixture.base.sumValue);
    expect(result.mean).toBe(fixture.base.meanValue);
    expect(result.meanInputGradient).toEqual(fixture.base.meanInputGradient);
    expect(result.meanTargetGradient).toEqual(fixture.base.meanTargetGradient);
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
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.alias).toBe(2);
    expect(result.parameter).toBe(2);
  });

  it("supports symbolic VJP, per-example vmap, ONNX decomposition, and explicit backend refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._ir import OP_L1_LOSS, OP_L1_LOSS_VJP, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

left = bg.from_numpy(np.array([1.0, -2.0, 3.0], dtype=np.float32), requires_grad=True)
right = bg.from_numpy(np.array([0.0, -1.0, 5.0], dtype=np.float32), requires_grad=True)
loss = F.l1_loss(left, right, reduction="mean")
input_gradient = bg.func.grad(
    lambda value: F.l1_loss(value, right, reduction="sum")
)(left)
target_gradient = bg.func.grad(
    lambda value: F.l1_loss(left, value, reduction="sum")
)(right)

batched_left = bg.from_numpy(np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float32))
batched_right = bg.from_numpy(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
mapped_mean = bg.func.vmap(lambda a, b: F.l1_loss(a, b, reduction="mean"))(
    batched_left, batched_right
)
captured = bg.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
mapped_sum = bg.func.vmap(lambda value: F.l1_loss(value, captured, reduction="sum"))(
    batched_left
)
mapped_none = bg.func.vmap(lambda a, b: F.l1_loss(a, b, reduction="none"))(
    batched_left, batched_right
)
nested_left = bg.from_numpy(np.arange(1.0, 9.0, dtype=np.float32).reshape(2, 2, 2))
nested_right = bg.from_numpy(np.zeros((2, 2, 2), dtype=np.float32))
nested_mean = bg.func.vmap(bg.func.vmap(
    lambda a, b: F.l1_loss(a, b, reduction="mean")
))(nested_left, nested_right)

model = bg.onnx.export_inference(loss, input_buffers=(left, right))
half_left = bg.from_numpy(np.array([1.0, -2.0], dtype=np.float16))
half_right = bg.from_numpy(np.array([0.0, -1.0], dtype=np.float16))
half_model = bg.onnx.export_inference(
    F.l1_loss(half_left, half_right, reduction="mean"),
    input_buffers=(half_left, half_right),
)
support = next(
    item for item in bg.framework_operation_support()["operations"]
    if item["opcode"] == OP_L1_LOSS
)
symbolic = get_rule(OP_L1_LOSS)(loss._uop, loss._uop.inputs, bg.ones()._uop)

{
    "registered": get_rule(OP_L1_LOSS) is not None,
    "inputGradient": input_gradient.numpy().tolist(),
    "targetGradient": target_gradient.numpy().tolist(),
    "symbolicOps": [[node.op for node in toposort(value)] for value in symbolic],
    "mappedMean": mapped_mean.numpy().tolist(),
    "mappedMeanShape": list(mapped_mean.shape),
    "mappedMeanBatchRank": mapped_mean._uop.arg["batch_rank"],
    "mappedSum": mapped_sum.numpy().tolist(),
    "mappedNone": mapped_none.numpy().tolist(),
    "mappedNoneShape": list(mapped_none.shape),
    "nestedMean": nested_mean.numpy().tolist(),
    "nestedMeanBatchRank": nested_mean._uop.arg["batch_rank"],
    "onnxHasSub": b"Sub" in model,
    "onnxHasAbs": b"Abs" in model,
    "onnxHasReduceMean": b"ReduceMean" in model,
    "halfOnnxHasCast": b"Cast" in half_model,
    "onnxHasCustom": b"CUSTOM" in model,
    "planError": error(lambda: bg.gpu_plan_summary(loss)),
    "webgpuSupported": OP_L1_LOSS in supported_opcodes() or OP_L1_LOSS_VJP in supported_opcodes(),
    "support": support,
}
`);

    expect(result.registered).toBe(true);
    expect(result.inputGradient).toEqual([1, -1, -1]);
    expect(result.targetGradient).toEqual([-1, 1, 1]);
    for (const ops of result.symbolicOps as string[][]) {
      expect(ops).toContain("L1_LOSS_VJP");
      expect(ops).not.toContain("CUSTOM");
    }
    expect(result.mappedMean).toEqual([1.5, 2]);
    expect(result.mappedMeanShape).toEqual([2]);
    expect(result.mappedMeanBatchRank).toBe(1);
    expect(result.mappedSum).toEqual([1, 4]);
    expect(result.mappedNone).toEqual([[1, 2], [2, 2]]);
    expect(result.mappedNoneShape).toEqual([2, 2]);
    expect(result.nestedMean).toEqual([[1.5, 3.5], [5.5, 7.5]]);
    expect(result.nestedMeanBatchRank).toBe(2);
    expect(result.onnxHasSub).toBe(true);
    expect(result.onnxHasAbs).toBe(true);
    expect(result.onnxHasReduceMean).toBe(true);
    expect(result.halfOnnxHasCast).toBe(true);
    expect(result.onnxHasCustom).toBe(false);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*L1_LOSS/u);
    expect(result.webgpuSupported).toBe(false);
    expect(result.support).toMatchObject({
      contractId: "browsergrad.jit.framework.functional.l1-loss.v1",
      publicSurface: "torch.nn.functional.l1_loss",
      opcode: "L1_LOSS",
      semanticState: "typed",
      retiredOpaqueOperationId: "jit.custom.l1-loss.v0",
    });
  });

  it("rejects malformed, mutated, hostile, and over-budget requests before execution", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_L1_LOSS_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_L1_LOSS
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

left = bg.from_numpy(np.ones((2,), dtype=np.float32))
right = bg.from_numpy(np.zeros((2,), dtype=np.float32))
other_session = bg.new_session()
other_right = bg.from_numpy(
    np.zeros((2,), dtype=np.float32),
    session=other_session,
)

errors = {
    "input-type": error(lambda: F.l1_loss([1.0, 2.0], right)),
    "shape": error(lambda: F.l1_loss(left, bg.from_numpy(np.zeros((1,), dtype=np.float32)))),
    "reduction": error(lambda: F.l1_loss(left, right, reduction="median")),
    "reduction-type": error(lambda: F.l1_loss(left, right, reduction=object())),
    "input-dtype": error(lambda: F.l1_loss(
        bg.from_numpy(np.ones((2,), dtype=np.int32)), right
    )),
    "target-dtype": error(lambda: F.l1_loss(
        left, bg.from_numpy(np.ones((2,), dtype=np.int32))
    )),
    "target-type": error(lambda: F.l1_loss(left, [0.0, 0.0])),
    "session": error(lambda: F.l1_loss(left, other_right)),
}

class HostileArray(np.ndarray):
    pass

hostile_source = bg.from_numpy(np.ones((2,), dtype=np.float32).view(HostileArray))
errors["array-subclass"] = error(lambda: F.l1_loss(
    hostile_source, right, reduction="sum"
).numpy())

valid = F.l1_loss(left, right, reduction="sum")
valid._uop.arg["reduction"] = "forged"
dy = UOp(OP_BUFFER, (), (), "float32", arg="dy")
errors["mutated-cpu"] = error(valid.numpy)
errors["mutated-vjp"] = error(lambda: get_rule(OP_L1_LOSS)(valid._uop, valid._uop.inputs, dy))
errors["mutated-vmap"] = error(lambda: get_vmap_rule(OP_L1_LOSS)(
    valid._uop,
    {id(node): node for node in valid._uop.inputs},
    2,
))
errors["mutated-onnx"] = error(lambda: bg.onnx.export_inference(valid, input_buffers=(left, right)))
errors["mutated-plan"] = error(lambda: bg.gpu_plan_summary(valid))

def forged(shape, dtype="float32"):
    lhs = UOp(OP_BUFFER, (), shape, dtype, arg="lhs-" + str(shape))
    rhs = UOp(OP_BUFFER, (), shape, dtype, arg="rhs-" + str(shape))
    return UOp(
        OP_L1_LOSS,
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

wrong_shape_left = UOp(OP_BUFFER, (), (2,), "float32", arg="wrong-shape-left")
wrong_shape_right = UOp(OP_BUFFER, (), (2,), "float32", arg="wrong-shape-right")
errors["declared-shape"] = error(lambda: UOp(
    OP_L1_LOSS,
    (wrong_shape_left, wrong_shape_right),
    (1,),
    "float32",
    arg={"reduction": "sum", "batch_rank": 0},
))
errors["declared-dtype"] = error(lambda: UOp(
    OP_L1_LOSS,
    (wrong_shape_left, wrong_shape_right),
    (),
    "float64",
    arg={"reduction": "sum", "batch_rank": 0},
))
errors
`);

    for (const invalid of FRAMEWORK_L1_LOSS_CONFORMANCE.invalid) {
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
    expect(errors.session).toMatch(/^ShapeError: .*same session/u);
  });
});
