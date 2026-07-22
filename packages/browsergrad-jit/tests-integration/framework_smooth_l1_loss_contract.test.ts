import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE } from "../../../test-support/framework-smooth-l1-loss-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.nn.functional.smooth_l1_loss contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches shared piecewise reductions, both gradients, promotion, empties, and aliases", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE);
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
none = F.smooth_l1_loss(left, right, beta=base["beta"], reduction="none")
none_array = none.numpy()
upstream = tensor(base["shape"], base["upstreamValues"])
closure_gradients = none._ctx.fn(upstream.numpy(), (left.numpy(), right.numpy()))
none.backward(upstream)

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

negative_zero = F.smooth_l1_loss(
    tensor([1], [1.0]), tensor([1], [0.0]), beta=-0.0
)

with bg.no_grad():
    detached = F.smooth_l1_loss(
        tensor([1], [1.0], requires_grad=True),
        tensor([1], [0.0], requires_grad=True),
    )

bg.install_torch_alias()
import torch
alias = torch.nn.SmoothL1Loss(beta=0.5, reduction="sum")(
    torch.tensor([0.0, 1.0]),
    torch.tensor([0.0, 0.0]),
)
parameter = bg.nn.Parameter(tensor([2], [0.5, -2.0]))
parameter_loss = F.smooth_l1_loss(
    parameter, tensor([2], [0.0, -1.0]), reduction="sum"
)

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
    "betaResults": beta_results,
    "zeroBetaGradients": [
        zero_beta_left.grad.numpy().reshape(-1).tolist(),
        zero_beta_right.grad.numpy().reshape(-1).tolist(),
    ],
    "dtypes": dtype_results,
    "scalar": float(scalar.item()),
    "scalarGradients": [float(scalar_left.grad.item()), float(scalar_right.grad.item())],
    "emptyNoneShape": list(empty_none.shape),
    "emptyNoneSize": int(empty_none.numpy().size),
    "emptySum": float(empty_sum.item()),
    "emptyMeanIsNan": bool(np.isnan(empty_mean.item())),
    "emptyGradientShapes": [list(empty_grad_left.grad.shape), list(empty_grad_right.grad.shape)],
    "negativeZeroCanonical": bool(np.signbit(negative_zero._uop.arg["beta"]) == False),
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "alias": float(alias.item()),
    "parameter": float(parameter_loss.item()),
}
`);

    const fixture = FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE;
    expect(result.op).toBe("SMOOTH_L1_LOSS");
    expect(result.arg).toEqual({ reduction: "none", batch_rank: 0, beta: 1 });
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
    expect(result.negativeZeroCanonical).toBe(true);
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.alias).toBe(0.75);
    expect(result.parameter).toBe(0.625);
  });

  it("supports symbolic VJP, nested vmap, ONNX decomposition, and explicit backend refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._ir import OP_SMOOTH_L1_LOSS, OP_SMOOTH_L1_LOSS_VJP, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

left = bg.from_numpy(np.array([0.0, -0.5, 2.0, -1.0], dtype=np.float32), requires_grad=True)
right = bg.from_numpy(np.zeros((4,), dtype=np.float32), requires_grad=True)
loss = F.smooth_l1_loss(left, right, beta=1.0, reduction="mean")
input_gradient = bg.func.grad(
    lambda value: F.smooth_l1_loss(value, right, beta=1.0, reduction="sum")
)(left)
target_gradient = bg.func.grad(
    lambda value: F.smooth_l1_loss(left, value, beta=1.0, reduction="sum")
)(right)

batched_left = bg.from_numpy(np.array([[0.0, 0.5], [2.0, -1.0]], dtype=np.float32))
batched_right = bg.from_numpy(np.zeros((2, 2), dtype=np.float32))
mapped_mean = bg.func.vmap(
    lambda a, b: F.smooth_l1_loss(a, b, beta=1.0, reduction="mean")
)(batched_left, batched_right)
captured = bg.from_numpy(np.zeros((2,), dtype=np.float32))
mapped_sum = bg.func.vmap(
    lambda value: F.smooth_l1_loss(value, captured, beta=1.0, reduction="sum")
)(batched_left)
mapped_none = bg.func.vmap(
    lambda a, b: F.smooth_l1_loss(a, b, beta=1.0, reduction="none")
)(batched_left, batched_right)
nested_left = bg.from_numpy(np.arange(8.0, dtype=np.float32).reshape(2, 2, 2))
nested_right = bg.from_numpy(np.zeros((2, 2, 2), dtype=np.float32))
nested_mean = bg.func.vmap(bg.func.vmap(
    lambda a, b: F.smooth_l1_loss(a, b, beta=1.0, reduction="mean")
))(nested_left, nested_right)

model = bg.onnx.export_inference(loss, input_buffers=(left, right))
zero_beta_model = bg.onnx.export_inference(
    F.smooth_l1_loss(left, right, beta=0.0, reduction="sum"),
    input_buffers=(left, right),
)
half_left = bg.from_numpy(np.array([0.5, -2.0], dtype=np.float16))
half_right = bg.from_numpy(np.zeros((2,), dtype=np.float16))
half_model = bg.onnx.export_inference(
    F.smooth_l1_loss(half_left, half_right, reduction="mean"),
    input_buffers=(half_left, half_right),
)
support = next(
    item for item in bg.framework_operation_support()["operations"]
    if item["opcode"] == OP_SMOOTH_L1_LOSS
)
symbolic = get_rule(OP_SMOOTH_L1_LOSS)(
    loss._uop, loss._uop.inputs, bg.ones()._uop
)

{
    "registered": get_rule(OP_SMOOTH_L1_LOSS) is not None,
    "inputGradient": input_gradient.numpy().tolist(),
    "targetGradient": target_gradient.numpy().tolist(),
    "symbolicOps": [[node.op for node in toposort(value)] for value in symbolic],
    "mappedMean": mapped_mean.numpy().tolist(),
    "mappedMeanBatchRank": mapped_mean._uop.arg["batch_rank"],
    "mappedSum": mapped_sum.numpy().tolist(),
    "mappedNone": mapped_none.numpy().tolist(),
    "nestedMean": nested_mean.numpy().tolist(),
    "nestedMeanBatchRank": nested_mean._uop.arg["batch_rank"],
    "onnxHasSub": b"Sub" in model,
    "onnxHasAbs": b"Abs" in model,
    "onnxHasLess": b"Less" in model,
    "onnxHasMul": b"Mul" in model,
    "onnxHasWhere": b"Where" in model,
    "onnxHasReduceMean": b"ReduceMean" in model,
    "zeroBetaHasWhere": b"Where" in zero_beta_model,
    "halfOnnxHasCast": b"Cast" in half_model,
    "onnxHasCustom": b"CUSTOM" in model,
    "planError": error(lambda: bg.gpu_plan_summary(loss)),
    "webgpuSupported": (
        OP_SMOOTH_L1_LOSS in supported_opcodes()
        or OP_SMOOTH_L1_LOSS_VJP in supported_opcodes()
    ),
    "support": support,
}
`);

    expect(result.registered).toBe(true);
    expect(result.inputGradient).toEqual([0, -0.5, 1, -1]);
    expect(result.targetGradient).toEqual([0, 0.5, -1, 1]);
    for (const ops of result.symbolicOps as string[][]) {
      expect(ops).toContain("SMOOTH_L1_LOSS_VJP");
      expect(ops).not.toContain("CUSTOM");
    }
    expect(result.mappedMean).toEqual([0.0625, 1]);
    expect(result.mappedMeanBatchRank).toBe(1);
    expect(result.mappedSum).toEqual([0.125, 2]);
    expect(result.mappedNone).toEqual([[0, 0.125], [1.5, 0.5]]);
    expect(result.nestedMean).toEqual([[0.25, 2], [4, 6]]);
    expect(result.nestedMeanBatchRank).toBe(2);
    expect(result.onnxHasSub).toBe(true);
    expect(result.onnxHasAbs).toBe(true);
    expect(result.onnxHasLess).toBe(true);
    expect(result.onnxHasMul).toBe(true);
    expect(result.onnxHasWhere).toBe(true);
    expect(result.onnxHasReduceMean).toBe(true);
    expect(result.zeroBetaHasWhere).toBe(false);
    expect(result.halfOnnxHasCast).toBe(true);
    expect(result.onnxHasCustom).toBe(false);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*SMOOTH_L1_LOSS/u);
    expect(result.webgpuSupported).toBe(false);
    expect(result.support).toMatchObject({
      contractId: "browsergrad.jit.framework.functional.smooth-l1-loss.v1",
      publicSurface: "torch.nn.functional.smooth_l1_loss",
      opcode: "SMOOTH_L1_LOSS",
      semanticState: "typed",
      retiredOpaqueOperationId: "jit.custom.smooth-l1-loss.v0",
    });
  });

  it("rejects malformed, mutated, hostile, and over-budget requests before execution", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_SMOOTH_L1_LOSS
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
other_right = bg.from_numpy(
    np.zeros((2,), dtype=np.float32), session=bg.new_session()
)

class HostileBeta:
    def __float__(self):
        raise RuntimeError("must not execute")

errors = {
    "input-type": error(lambda: F.smooth_l1_loss([1.0, 2.0], right)),
    "shape": error(lambda: F.smooth_l1_loss(
        left, bg.from_numpy(np.zeros((1,), dtype=np.float32))
    )),
    "reduction": error(lambda: F.smooth_l1_loss(left, right, reduction="median")),
    "reduction-type": error(lambda: F.smooth_l1_loss(left, right, reduction=object())),
    "input-dtype": error(lambda: F.smooth_l1_loss(
        bg.from_numpy(np.ones((2,), dtype=np.int32)), right
    )),
    "target-dtype": error(lambda: F.smooth_l1_loss(
        left, bg.from_numpy(np.ones((2,), dtype=np.int32))
    )),
    "target-type": error(lambda: F.smooth_l1_loss(left, [0.0, 0.0])),
    "session": error(lambda: F.smooth_l1_loss(left, other_right)),
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

hostile_source = bg.from_numpy(np.ones((2,), dtype=np.float32).view(HostileArray))
errors["array-subclass"] = error(lambda: F.smooth_l1_loss(
    hostile_source, right, reduction="sum"
).numpy())

valid = F.smooth_l1_loss(left, right, beta=1.0, reduction="sum")
valid._uop.arg["beta"] = -1.0
dy = UOp(OP_BUFFER, (), (), "float32", arg="dy")
errors["mutated-cpu"] = error(valid.numpy)
errors["mutated-vjp"] = error(lambda: get_rule(OP_SMOOTH_L1_LOSS)(
    valid._uop, valid._uop.inputs, dy
))
errors["mutated-vmap"] = error(lambda: get_vmap_rule(OP_SMOOTH_L1_LOSS)(
    valid._uop, {id(node): node for node in valid._uop.inputs}, 2
))
errors["mutated-onnx"] = error(lambda: bg.onnx.export_inference(
    valid, input_buffers=(left, right)
))
errors["mutated-plan"] = error(lambda: bg.gpu_plan_summary(valid))

def forged(shape, dtype="float32"):
    lhs = UOp(OP_BUFFER, (), shape, dtype, arg="lhs-" + str(shape))
    rhs = UOp(OP_BUFFER, (), shape, dtype, arg="rhs-" + str(shape))
    return UOp(
        OP_SMOOTH_L1_LOSS,
        (lhs, rhs),
        (),
        dtype,
        arg={"reduction": "sum", "batch_rank": 0, "beta": 1.0},
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
    OP_SMOOTH_L1_LOSS,
    (wrong_left, wrong_right),
    (1,),
    "float32",
    arg={"reduction": "sum", "batch_rank": 0, "beta": 1.0},
))
errors["declared-dtype"] = error(lambda: UOp(
    OP_SMOOTH_L1_LOSS,
    (wrong_left, wrong_right),
    (),
    "float64",
    arg={"reduction": "sum", "batch_rank": 0, "beta": 1.0},
))
errors
`);

    for (const invalid of FRAMEWORK_SMOOTH_L1_LOSS_CONFORMANCE.invalid) {
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["beta-hostile"]).toContain("beta must be an exact real scalar");
    expect(errors["mutated-cpu"]).toMatch(/^RealizationError: .*beta/u);
    expect(errors["mutated-vjp"]).toMatch(/^ShapeError: .*beta/u);
    expect(errors["mutated-vmap"]).toMatch(/^ShapeError: .*beta/u);
    expect(errors["mutated-onnx"]).toMatch(/^ShapeError: .*beta/u);
    expect(errors["mutated-plan"]).toMatch(/^ShapeError: .*beta/u);
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
