import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE } from "../../../test-support/framework-binary-cross-entropy-with-logits-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 typed torch.nn.functional.binary_cross_entropy_with_logits contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches stable reductions, both gradients, extremes, promotion, empties, and aliases", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE);
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

logits = tensor(base["shape"], base["logits"], requires_grad=True)
targets = tensor(base["shape"], base["targets"], requires_grad=True)
none = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
none_array = none.numpy()
upstream = tensor(base["shape"], base["upstream"])
closure_gradients = none._ctx.fn(upstream.numpy(), (logits.numpy(), targets.numpy()))
none.backward(upstream)

summed = F.binary_cross_entropy_with_logits(
    tensor(base["shape"], base["logits"]),
    tensor(base["shape"], base["targets"]),
    reduction="sum",
)
mean_logits = tensor(base["shape"], base["logits"], requires_grad=True)
mean_targets = tensor(base["shape"], base["targets"], requires_grad=True)
mean = F.binary_cross_entropy_with_logits(mean_logits, mean_targets)
mean.backward()

extreme = fixture["extremes"]
extreme_logits = tensor([4], extreme["logits"], requires_grad=True)
extreme_targets = tensor([4], extreme["targets"], requires_grad=True)
extreme_loss = F.binary_cross_entropy_with_logits(
    extreme_logits, extreme_targets, reduction="none"
)
extreme_values = extreme_loss.numpy()
extreme_loss.backward(tensor([4], [1, 1, 1, 1]))

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype_logits = tensor([2], [-2, 2], case["logitsDtype"], True)
    dtype_targets = tensor([2], [0, 1], case["targetDtype"], True)
    dtype_output = F.binary_cross_entropy_with_logits(
        dtype_logits, dtype_targets, reduction="sum"
    )
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "logitsGradient": dtype_logits.grad.dtype,
        "targetGradient": dtype_targets.grad.dtype,
    })

scalar_fixture = fixture["scalar"]
scalar_logits = tensor([], [scalar_fixture["logits"]], requires_grad=True)
scalar_target = tensor([], [scalar_fixture["target"]], requires_grad=True)
scalar = F.binary_cross_entropy_with_logits(scalar_logits, scalar_target)
scalar.backward()

empty = fixture["empty"]
empty_logits = tensor(empty["shape"], [])
empty_target = tensor(empty["shape"], [])
empty_none = F.binary_cross_entropy_with_logits(empty_logits, empty_target, reduction="none")
empty_sum = F.binary_cross_entropy_with_logits(empty_logits, empty_target, reduction="sum")
empty_mean = F.binary_cross_entropy_with_logits(empty_logits, empty_target, reduction="mean")
empty_grad_logits = tensor(empty["shape"], [], requires_grad=True)
empty_grad_target = tensor(empty["shape"], [], requires_grad=True)
F.binary_cross_entropy_with_logits(empty_grad_logits, empty_grad_target).backward()

with bg.no_grad():
    detached = F.binary_cross_entropy_with_logits(
        tensor([1], [0], requires_grad=True),
        tensor([1], [1], requires_grad=True),
    )

bg.install_torch_alias()
import torch
alias = torch.nn.BCEWithLogitsLoss(reduction="sum")(
    torch.tensor([-2.0, 2.0]), torch.tensor([0.0, 1.0])
)
functional_alias = F.bce_with_logits_loss(
    tensor([2], [-2, 2]), tensor([2], [0, 1]), reduction="sum"
)
parameter = bg.nn.Parameter(tensor([2], [-2, 2]))
parameter_loss = F.binary_cross_entropy_with_logits(
    parameter, tensor([2], [0, 1]), reduction="sum"
)

{
    "op": none._uop.op,
    "arg": none._uop.arg,
    "none": none_array.reshape(-1).tolist(),
    "noneLogitsGradient": logits.grad.numpy().reshape(-1).tolist(),
    "noneTargetGradient": targets.grad.numpy().reshape(-1).tolist(),
    "closureGradients": [value.reshape(-1).tolist() for value in closure_gradients],
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
    "emptyNoneSize": int(empty_none.numpy().size),
    "emptySum": float(empty_sum.item()),
    "emptyMeanIsNan": bool(np.isnan(empty_mean.item())),
    "emptyGradientShapes": [list(empty_grad_logits.grad.shape), list(empty_grad_target.grad.shape)],
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "alias": float(alias.item()),
    "functionalAlias": float(functional_alias.item()),
    "parameter": float(parameter_loss.item()),
}
`);

    const fixture = FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE;
    expect(result.op).toBe("BINARY_CROSS_ENTROPY_WITH_LOGITS");
    expect(result.arg).toEqual({ reduction: "none", batch_rank: 0 });
    expectCloseArray(result.none, fixture.base.noneValues, 4);
    expectCloseArray(result.noneLogitsGradient, fixture.base.noneLogitsGradient, 4);
    expectCloseArray(result.noneTargetGradient, fixture.base.noneTargetGradient, 4);
    const closure = result.closureGradients as number[][];
    expectCloseArray(closure[0], fixture.base.noneLogitsGradient, 4);
    expectCloseArray(closure[1], fixture.base.noneTargetGradient, 4);
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
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    const twoExtremeCorrect = 2 * fixture.base.noneValues[0];
    expect(result.alias).toBeCloseTo(twoExtremeCorrect, 5);
    expect(result.functionalAlias).toBeCloseTo(twoExtremeCorrect, 5);
    expect(result.parameter).toBeCloseTo(twoExtremeCorrect, 5);
  });

  it("supports symbolic VJP, nested vmap, stable ONNX export, and explicit device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit._ir import OP_BINARY_CROSS_ENTROPY_WITH_LOGITS, OP_BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

logits = bg.from_numpy(np.array([-2.0, 0.0, 2.0, 80.0], dtype=np.float32), requires_grad=True)
targets = bg.from_numpy(np.array([0.0, 0.25, 1.0, 0.75], dtype=np.float32), requires_grad=True)
loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
logits_gradient = bg.func.grad(
    lambda value: F.binary_cross_entropy_with_logits(value, targets, reduction="sum")
)(logits)
target_gradient = bg.func.grad(
    lambda value: F.binary_cross_entropy_with_logits(logits, value, reduction="sum")
)(targets)

batched_logits = bg.from_numpy(np.array([[-2.0, 0.0], [2.0, 80.0]], dtype=np.float32))
batched_targets = bg.from_numpy(np.array([[0.0, 0.25], [1.0, 0.75]], dtype=np.float32))
mapped_mean = bg.func.vmap(
    lambda a, b: F.binary_cross_entropy_with_logits(a, b, reduction="mean")
)(batched_logits, batched_targets)
captured = bg.from_numpy(np.array([0.0, 1.0], dtype=np.float32))
mapped_sum = bg.func.vmap(
    lambda value: F.binary_cross_entropy_with_logits(value, captured, reduction="sum")
)(batched_logits)
mapped_none = bg.func.vmap(
    lambda a, b: F.binary_cross_entropy_with_logits(a, b, reduction="none")
)(batched_logits, batched_targets)
nested_logits = bg.from_numpy(np.tile(batched_logits.numpy(), (2, 1, 1)))
nested_targets = bg.from_numpy(np.tile(batched_targets.numpy(), (2, 1, 1)))
nested_mean = bg.func.vmap(bg.func.vmap(
    lambda a, b: F.binary_cross_entropy_with_logits(a, b, reduction="mean")
))(nested_logits, nested_targets)

model = bg.onnx.export_inference(loss, input_buffers=(logits, targets))
half_logits = bg.from_numpy(np.array([-2.0, 2.0], dtype=np.float16))
half_targets = bg.from_numpy(np.array([0.0, 1.0], dtype=np.float16))
half_model = bg.onnx.export_inference(
    F.binary_cross_entropy_with_logits(half_logits, half_targets),
    input_buffers=(half_logits, half_targets),
)
scalar_model = bg.onnx.export_inference(
    F.binary_cross_entropy_with_logits(
        bg.from_numpy(np.asarray(0.0, dtype=np.float32)),
        bg.from_numpy(np.asarray(1.0, dtype=np.float32)),
    ),
)

def fields(data):
    index = 0
    while index < len(data):
        tag = 0
        shift = 0
        while True:
            byte = data[index]
            index += 1
            tag |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        number = tag >> 3
        wire = tag & 0x7
        if wire == 0:
            value = 0
            shift = 0
            while True:
                byte = data[index]
                index += 1
                value |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    break
                shift += 7
            yield number, wire, value
        elif wire == 2:
            length = 0
            shift = 0
            while True:
                byte = data[index]
                index += 1
                length |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    break
                shift += 7
            payload = data[index:index + length]
            index += length
            yield number, wire, payload
        else:
            raise RuntimeError("unexpected protobuf wire type " + str(wire))

def scalar_one_rank(model):
    graph = next(
        payload for number, wire, payload in fields(model)
        if number == 7 and wire == 2
    )
    for number, wire, payload in fields(graph):
        if number != 5 or wire != 2:
            continue
        tensor_fields = list(fields(payload))
        name = next(
            value.decode("utf-8") for field, kind, value in tensor_fields
            if field == 8 and kind == 2
        )
        if name.startswith("bce_with_logits_one_"):
            return sum(
                1 for field, kind, _ in tensor_fields
                if field == 1 and kind == 0
            )
    raise RuntimeError("missing BCE-with-logits one initializer")
support = next(
    item for item in bg.framework_operation_support()["operations"]
    if item["opcode"] == OP_BINARY_CROSS_ENTROPY_WITH_LOGITS
)
symbolic = get_rule(OP_BINARY_CROSS_ENTROPY_WITH_LOGITS)(
    loss._uop, loss._uop.inputs, bg.ones()._uop
)

{
    "logitsGradient": logits_gradient.numpy().tolist(),
    "targetGradient": target_gradient.numpy().tolist(),
    "symbolicOps": [[node.op for node in toposort(value)] for value in symbolic],
    "mappedMean": mapped_mean.numpy().tolist(),
    "mappedMeanBatchRank": mapped_mean._uop.arg["batch_rank"],
    "mappedSum": mapped_sum.numpy().tolist(),
    "mappedNone": mapped_none.numpy().tolist(),
    "nestedMean": nested_mean.numpy().tolist(),
    "nestedMeanBatchRank": nested_mean._uop.arg["batch_rank"],
    "onnxOps": {name: (name.encode() in model) for name in ["Neg", "Softplus", "Sub", "Mul", "Add", "ReduceMean"]},
    "halfOnnxHasCast": b"Cast" in half_model,
    "scalarOnnxOneRank": scalar_one_rank(scalar_model),
    "planError": error(lambda: bg.gpu_plan_summary(loss)),
    "webgpuSupported": (
        OP_BINARY_CROSS_ENTROPY_WITH_LOGITS in supported_opcodes()
        or OP_BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP in supported_opcodes()
    ),
    "support": support,
}
`);

    const fixture = FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE;
    expectCloseArray(result.logitsGradient, [
      fixture.base.noneLogitsGradient[0],
      fixture.base.noneLogitsGradient[1] / 2,
      fixture.base.noneLogitsGradient[2] / 3,
      fixture.base.noneLogitsGradient[3] / 4,
    ], 4);
    expectCloseArray(result.targetGradient, [2, 0, -2, -80], 4);
    for (const ops of result.symbolicOps as string[][]) {
      expect(ops).toContain("BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP");
      expect(ops).not.toContain("CUSTOM");
    }
    expectCloseArray(result.mappedMean, [
      (fixture.base.noneValues[0] + fixture.base.noneValues[1]) / 2,
      (fixture.base.noneValues[2] + fixture.base.noneValues[3]) / 2,
    ], 4);
    expect(result.mappedMeanBatchRank).toBe(1);
    expectCloseArray(result.mappedSum, [
      fixture.base.noneValues[0] + fixture.base.noneValues[1],
      2 + fixture.base.noneValues[0],
    ], 4);
    expect(result.mappedNone).toEqual(expect.arrayContaining([expect.any(Array)]));
    expect(result.nestedMeanBatchRank).toBe(2);
    expect((result.nestedMean as number[][])).toHaveLength(2);
    expect(result.onnxOps).toEqual({
      Neg: true,
      Softplus: true,
      Sub: true,
      Mul: true,
      Add: true,
      ReduceMean: true,
    });
    expect(result.halfOnnxHasCast).toBe(true);
    expect(result.scalarOnnxOneRank).toBe(0);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*BINARY_CROSS_ENTROPY_WITH_LOGITS/u);
    expect(result.webgpuSupported).toBe(false);
    expect(result.support).toMatchObject({
      contractId: "browsergrad.jit.framework.functional.binary-cross-entropy-with-logits.v1",
      opcode: "BINARY_CROSS_ENTROPY_WITH_LOGITS",
      semanticState: "typed",
      retiredOpaqueOperationId: "jit.custom.binary-cross-entropy-with-logits.v0",
      decisions: {
        onnxExport: "supported-opset17-stable-bce-with-logits-float16-float32-float64",
      },
    });
  });

  it("rejects malformed, mutated, and over-budget requests before numerical work", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_BINARY_CROSS_ENTROPY_WITH_LOGITS
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

logits = tensor([-2.0, 2.0])
targets = tensor([0.0, 1.0])
other_targets = tensor([0.0, 1.0], session=bg.new_session())
errors = {
    "logits-type": error(lambda: F.binary_cross_entropy_with_logits([-2.0, 2.0], targets)),
    "target-type": error(lambda: F.binary_cross_entropy_with_logits(logits, [0.0, 1.0])),
    "shape": error(lambda: F.binary_cross_entropy_with_logits(logits, tensor([1.0]))),
    "session": error(lambda: F.binary_cross_entropy_with_logits(logits, other_targets)),
    "reduction": error(lambda: F.binary_cross_entropy_with_logits(logits, targets, reduction="median")),
    "reduction-type": error(lambda: F.binary_cross_entropy_with_logits(logits, targets, reduction=object())),
    "logits-dtype": error(lambda: F.binary_cross_entropy_with_logits(tensor([-2, 2], "int32"), targets)),
    "target-dtype": error(lambda: F.binary_cross_entropy_with_logits(logits, tensor([0, 1], "int32"))),
    "weight": error(lambda: F.binary_cross_entropy_with_logits(logits, targets, weight=targets)),
    "pos-weight": error(lambda: F.binary_cross_entropy_with_logits(logits, targets, pos_weight=targets)),
}

class HostileArray(np.ndarray):
    pass

hostile = bg.from_numpy(np.array([-2.0, 2.0], dtype=np.float32).view(HostileArray))
errors["array-subclass"] = error(lambda: F.binary_cross_entropy_with_logits(
    hostile, targets, reduction="sum"
).numpy())

valid = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
valid._uop.arg["reduction"] = "median"
dy = UOp(OP_BUFFER, (), (), "float32", arg="dy")
errors["mutated-cpu"] = error(valid.numpy)
errors["mutated-vjp"] = error(lambda: get_rule(OP_BINARY_CROSS_ENTROPY_WITH_LOGITS)(
    valid._uop, valid._uop.inputs, dy
))
errors["mutated-vmap"] = error(lambda: get_vmap_rule(OP_BINARY_CROSS_ENTROPY_WITH_LOGITS)(
    valid._uop, {id(node): node for node in valid._uop.inputs}, 2
))
errors["mutated-onnx"] = error(lambda: bg.onnx.export_inference(
    valid, input_buffers=(logits, targets)
))
errors["mutated-plan"] = error(lambda: bg.gpu_plan_summary(valid))

def forged(shape, dtype="float32"):
    left = UOp(OP_BUFFER, (), shape, dtype, arg="left-" + str(shape))
    right = UOp(OP_BUFFER, (), shape, dtype, arg="right-" + str(shape))
    return UOp(
        OP_BINARY_CROSS_ENTROPY_WITH_LOGITS,
        (left, right),
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

left = UOp(OP_BUFFER, (), (2,), "float32", arg="wrong-left")
right = UOp(OP_BUFFER, (), (2,), "float32", arg="wrong-right")
errors["declared-shape"] = error(lambda: UOp(
    OP_BINARY_CROSS_ENTROPY_WITH_LOGITS,
    (left, right),
    (1,),
    "float32",
    arg={"reduction": "sum", "batch_rank": 0},
))
errors["declared-dtype"] = error(lambda: UOp(
    OP_BINARY_CROSS_ENTROPY_WITH_LOGITS,
    (left, right),
    (),
    "float64",
    arg={"reduction": "sum", "batch_rank": 0},
))
errors
`);

    for (const invalid of FRAMEWORK_BINARY_CROSS_ENTROPY_WITH_LOGITS_CONFORMANCE.invalid) {
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["array-subclass"]).toMatch(/^RealizationError: .*exact ndarray/u);
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
  });
});
