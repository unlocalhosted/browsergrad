import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_KL_DIV_CONFORMANCE } from "../../../test-support/framework-kl-div-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 typed torch.nn.functional.kl_div contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches native reductions, both target modes, both gradients, zeros, empties, and aliases", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_KL_DIV_CONFORMANCE);
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

input_value = tensor(base["shape"], base["input"], requires_grad=True)
target_value = tensor(base["shape"], base["target"], requires_grad=True)
none = F.kl_div(input_value, target_value, reduction="none")
none_array = none.numpy()
upstream = tensor(base["shape"], base["upstream"])
closure = none._ctx.fn(upstream.numpy(), (input_value.numpy(), target_value.numpy()))
none.backward(upstream)

reductions = {}
for reduction in ("sum", "mean", "batchmean"):
    reductions[reduction] = float(F.kl_div(
        tensor(base["shape"], base["input"]),
        tensor(base["shape"], base["target"]),
        reduction=reduction,
    ).item())

log_input = tensor(base["shape"], base["input"], requires_grad=True)
log_target = tensor(
    base["shape"], np.log(np.asarray(base["target"])), requires_grad=True
)
log_loss = F.kl_div(log_input, log_target, reduction="none", log_target=True)
log_values = log_loss.numpy()
log_loss.backward(upstream)

zero = fixture["zeroTarget"]
zero_input = tensor([2], zero["input"], requires_grad=True)
zero_target = tensor([2], zero["target"], requires_grad=True)
zero_loss = F.kl_div(zero_input, zero_target, reduction="none")
zero_values = zero_loss.numpy()
zero_loss.backward(tensor([2], [1, 1]))

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype_input = tensor([2], [-0.3, -1.2], case["inputDtype"], True)
    dtype_target = tensor([2], [0.7, 0.3], case["targetDtype"], True)
    dtype_output = F.kl_div(dtype_input, dtype_target, reduction="sum")
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "inputGradient": dtype_input.grad.dtype,
        "targetGradient": dtype_target.grad.dtype,
    })

scalar_fixture = fixture["scalar"]
scalar_input = tensor([], [scalar_fixture["input"]], requires_grad=True)
scalar_target = tensor([], [scalar_fixture["target"]], requires_grad=True)
scalar = F.kl_div(scalar_input, scalar_target, reduction="batchmean")
scalar.backward()

empty = fixture["empty"]
zero_batch_input = tensor(empty["zeroBatchShape"], [])
zero_batch_target = tensor(empty["zeroBatchShape"], [])
zero_support_input = tensor(empty["zeroSupportShape"], [])
zero_support_target = tensor(empty["zeroSupportShape"], [])
empty_results = {
    "zeroBatchSum": float(F.kl_div(zero_batch_input, zero_batch_target, reduction="sum").item()),
    "zeroBatchMeanNan": bool(np.isnan(F.kl_div(zero_batch_input, zero_batch_target, reduction="mean").item())),
    "zeroBatchBatchmeanNan": bool(np.isnan(F.kl_div(zero_batch_input, zero_batch_target, reduction="batchmean").item())),
    "zeroSupportBatchmean": float(F.kl_div(zero_support_input, zero_support_target, reduction="batchmean").item()),
}

with bg.no_grad():
    detached = F.kl_div(
        tensor([1], [-0.3], requires_grad=True),
        tensor([1], [0.7], requires_grad=True),
    )

bg.install_torch_alias()
import torch
module_alias = torch.nn.KLDivLoss(reduction="batchmean")(
    torch.tensor(np.asarray(base["input"]).reshape(base["shape"])),
    torch.tensor(np.asarray(base["target"]).reshape(base["shape"])),
)
functional_alias = F.kl_div_loss(
    tensor(base["shape"], base["input"]),
    tensor(base["shape"], base["target"]),
    reduction="batchmean",
)

{
    "op": none._uop.op,
    "arg": none._uop.arg,
    "none": none_array.reshape(-1).tolist(),
    "noneInputGradient": input_value.grad.numpy().reshape(-1).tolist(),
    "noneTargetGradient": target_value.grad.numpy().reshape(-1).tolist(),
    "closure": [value.reshape(-1).tolist() for value in closure],
    "reductions": reductions,
    "logValues": log_values.reshape(-1).tolist(),
    "logInputGradient": log_input.grad.numpy().reshape(-1).tolist(),
    "logTargetGradient": log_target.grad.numpy().reshape(-1).tolist(),
    "zeroValues": zero_values.tolist(),
    "zeroInputGradient": zero_input.grad.numpy().tolist(),
    "zeroTargetGradientNan": bool(np.isnan(zero_target.grad.numpy()[0])),
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
    expect(result.op).toBe("KL_DIV");
    expect(result.arg).toEqual({ reduction: "none", batch_rank: 0, log_target: false });
    expectCloseArray(result.none, fixture.base.noneValues);
    expectCloseArray(result.noneInputGradient, fixture.base.noneInputGradient);
    expectCloseArray(result.noneTargetGradient, fixture.base.noneTargetGradient);
    const closure = result.closure as number[][];
    expectCloseArray(closure[0], fixture.base.noneInputGradient);
    expectCloseArray(closure[1], fixture.base.noneTargetGradient);
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
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map((entry) => ({
      output: entry.outputDtype,
      inputGradient: entry.inputDtype,
      targetGradient: entry.targetDtype,
    })));
    expect(result.scalar).toBeCloseTo(fixture.scalar.output, 5);
    expectCloseArray(result.scalarGradients, [
      fixture.scalar.inputGradient,
      fixture.scalar.targetGradient,
    ]);
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

  it("supports symbolic VJP, nested vmap, both ONNX target modes, and explicit device refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_KL_DIV_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
import struct
from browsergrad_jit._ir import OP_KL_DIV, OP_KL_DIV_VJP, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})
base = fixture["base"]
shape = tuple(base["shape"])
inputs = np.asarray(base["input"], dtype=np.float32).reshape(shape)
targets = np.asarray(base["target"], dtype=np.float32).reshape(shape)
input_value = bg.from_numpy(inputs, requires_grad=True)
target_value = bg.from_numpy(targets, requires_grad=True)
loss = F.kl_div(input_value, target_value, reduction="batchmean")

input_gradient = bg.func.grad(
    lambda value: F.kl_div(value, target_value, reduction="sum")
)(input_value)
target_gradient = bg.func.grad(
    lambda value: F.kl_div(input_value, value, reduction="sum")
)(target_value)
symbolic = get_rule(OP_KL_DIV)(loss._uop, loss._uop.inputs, bg.ones()._uop)

batched_inputs = bg.from_numpy(np.stack([inputs, inputs + 0.1]))
batched_targets = bg.from_numpy(np.stack([targets, targets]))
mapped_batchmean = bg.func.vmap(
    lambda a, b: F.kl_div(a, b, reduction="batchmean")
)(batched_inputs, batched_targets)
captured = bg.from_numpy(targets)
mapped_sum = bg.func.vmap(
    lambda value: F.kl_div(value, captured, reduction="sum")
)(batched_inputs)
nested_inputs = bg.from_numpy(np.stack([batched_inputs.numpy(), batched_inputs.numpy()]))
nested_targets = bg.from_numpy(np.stack([batched_targets.numpy(), batched_targets.numpy()]))
nested = bg.func.vmap(bg.func.vmap(
    lambda a, b: F.kl_div(a, b, reduction="mean")
))(nested_inputs, nested_targets)

probability_model = bg.onnx.export_inference(
    F.kl_div(input_value, target_value, reduction="batchmean"),
    input_buffers=(input_value, target_value),
)
log_target_value = bg.from_numpy(np.log(targets))
log_model = bg.onnx.export_inference(
    F.kl_div(input_value, log_target_value, reduction="sum", log_target=True),
    input_buffers=(input_value, log_target_value),
)
half_input = bg.from_numpy(inputs.astype(np.float16))
half_target = bg.from_numpy(targets.astype(np.float16))
half_model = bg.onnx.export_inference(
    F.kl_div(half_input, half_target), input_buffers=(half_input, half_target)
)
scalar_model = bg.onnx.export_inference(F.kl_div(
    bg.from_numpy(np.asarray(-0.3, dtype=np.float32)),
    bg.from_numpy(np.asarray(0.7, dtype=np.float32)),
    reduction="batchmean",
))

${ONNX_PROTOBUF_TEST_HELPERS}

def parse_kl_model(model):
    graph = next(
        payload for number, wire, payload in fields(model)
        if number == 7 and wire == 2
    )
    op_types = []
    initializers = {}
    output_dtype = None
    for number, wire, payload in fields(graph):
        if number == 1 and wire == 2:
            node_fields = list(fields(payload))
            op_types.append(next(
                value.decode("utf-8") for field, kind, value in node_fields
                if field == 4 and kind == 2
            ))
        elif number == 5 and wire == 2:
            tensor_fields = list(fields(payload))
            name = next(
                value.decode("utf-8") for field, kind, value in tensor_fields
                if field == 8 and kind == 2
            )
            initializers[name] = {
                "rank": sum(
                    1 for field, kind, _ in tensor_fields
                    if field == 1 and kind == 0
                ),
                "dtype": next(
                    value for field, kind, value in tensor_fields
                    if field == 2 and kind == 0
                ),
                "raw": next(
                    value for field, kind, value in tensor_fields
                    if field == 9 and kind == 2
                ),
            }
        elif number == 12 and wire == 2:
            output_dtype = value_info_dtype(payload)
    denominator = next(
        (value for name, value in initializers.items()
         if name.startswith("kl_div_batch_denominator_")),
        None,
    )
    zero = next(
        (value for name, value in initializers.items()
         if name.startswith("kl_div_zero_")),
        None,
    )
    denominator_result = {"present": False}
    if denominator is not None:
        denominator_result = {
            "present": True,
            "rank": denominator["rank"],
            "dtype": denominator["dtype"],
            "value": struct.unpack("<f", denominator["raw"])[0],
        }
    return {
        "opTypes": op_types,
        "outputDtype": output_dtype,
        "denominator": denominator_result,
        "zeroRank": -1 if zero is None else zero["rank"],
    }
support = next(
    item for item in bg.framework_operation_support()["operations"]
    if item["opcode"] == OP_KL_DIV
)

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "inputGradient": input_gradient.numpy().reshape(-1).tolist(),
    "targetGradient": target_gradient.numpy().reshape(-1).tolist(),
    "symbolicOps": [[node.op for node in toposort(value)] for value in symbolic],
    "mappedBatchmean": mapped_batchmean.numpy().tolist(),
    "mappedBatchRank": mapped_batchmean._uop.arg["batch_rank"],
    "mappedSum": mapped_sum.numpy().tolist(),
    "nestedShape": list(nested.shape),
    "nestedBatchRank": nested._uop.arg["batch_rank"],
    "probabilityOnnx": parse_kl_model(probability_model),
    "logOnnx": parse_kl_model(log_model),
    "halfOnnx": parse_kl_model(half_model),
    "scalarOnnx": parse_kl_model(scalar_model),
    "planError": error(lambda: bg.gpu_plan_summary(loss)),
    "webgpuSupported": OP_KL_DIV in supported_opcodes() or OP_KL_DIV_VJP in supported_opcodes(),
    "support": support,
}
`);

    const fixture = FRAMEWORK_KL_DIV_CONFORMANCE;
    expectCloseArray(result.inputGradient, fixture.base.target.map((value) => -value));
    expectCloseArray(
      result.targetGradient,
      fixture.base.noneTargetGradient.map((value, index) => value / fixture.base.upstream[index]!),
    );
    for (const ops of result.symbolicOps as string[][]) {
      expect(ops).toContain("KL_DIV_VJP");
      expect(ops).not.toContain("CUSTOM");
    }
    const secondBatchmean = fixture.base.batchmeanValue - 0.1;
    expectCloseArray(result.mappedBatchmean, [fixture.base.batchmeanValue, secondBatchmean]);
    expect(result.mappedBatchRank).toBe(1);
    expectCloseArray(result.mappedSum, [fixture.base.sumValue, fixture.base.sumValue - 0.2]);
    expect(result.nestedShape).toEqual([2, 2]);
    expect(result.nestedBatchRank).toBe(2);
    expect(result.probabilityOnnx).toEqual({
      opTypes: ["Log", "Mul", "Equal", "Where", "Mul", "Sub", "ReduceSum", "Div", "Identity"],
      outputDtype: 1,
      denominator: { present: true, rank: 0, dtype: 1, value: 2 },
      zeroRank: 0,
    });
    expect(result.logOnnx).toEqual({
      opTypes: ["Exp", "Sub", "Mul", "ReduceSum", "Identity"],
      outputDtype: 1,
      denominator: { present: false },
      zeroRank: -1,
    });
    expect(result.halfOnnx).toEqual({
      opTypes: ["Cast", "Cast", "Log", "Mul", "Equal", "Where", "Mul", "Sub", "ReduceMean", "Cast", "Identity"],
      outputDtype: 10,
      denominator: { present: false },
      zeroRank: 0,
    });
    expect(result.scalarOnnx).toEqual({
      opTypes: ["Log", "Mul", "Equal", "Where", "Mul", "Sub", "Identity", "Identity"],
      outputDtype: 1,
      denominator: { present: false },
      zeroRank: 0,
    });
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*KL_DIV/u);
    expect(result.webgpuSupported).toBe(false);
    expect(result.support).toMatchObject({
      contractId: "browsergrad.jit.framework.functional.kl-div.v1",
      opcode: "KL_DIV",
      semanticState: "typed",
      retiredOpaqueOperationId: "jit.custom.kl-div.v0",
      decisions: {
        onnxExport: "supported-opset17-kl-div-float16-float32-float64",
      },
    });
  });

  it("rejects malformed, mutated, and over-budget requests before numerical work", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_KL_DIV_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_KL_DIV
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

input_value = tensor([-0.3, -1.2])
target_value = tensor([0.7, 0.3])
other_target = tensor([0.7, 0.3], session=bg.new_session())
errors = {
    "input-type": error(lambda: F.kl_div([-0.3, -1.2], target_value)),
    "target-type": error(lambda: F.kl_div(input_value, [0.7, 0.3])),
    "shape": error(lambda: F.kl_div(input_value, tensor([0.7]))),
    "session": error(lambda: F.kl_div(input_value, other_target)),
    "reduction": error(lambda: F.kl_div(input_value, target_value, reduction="median")),
    "reduction-type": error(lambda: F.kl_div(input_value, target_value, reduction=object())),
    "log-target-type": error(lambda: F.kl_div(input_value, target_value, log_target=1)),
    "input-dtype": error(lambda: F.kl_div(tensor([-1, -2], "int32"), target_value)),
    "target-dtype": error(lambda: F.kl_div(input_value, tensor([1, 0], "int32"))),
    "size-average": error(lambda: F.kl_div(input_value, target_value, size_average=True)),
    "reduce": error(lambda: F.kl_div(input_value, target_value, reduce=True)),
}

class HostileArray(np.ndarray):
    pass
hostile = bg.from_numpy(np.array([-0.3, -1.2], dtype=np.float32).view(HostileArray))
errors["array-subclass"] = error(lambda: F.kl_div(hostile, target_value).numpy())

valid = F.kl_div(input_value, target_value, reduction="sum")
valid._uop.arg["log_target"] = 1
dy = UOp(OP_BUFFER, (), (), "float32", arg="dy")
errors["mutated-cpu"] = error(valid.numpy)
errors["mutated-vjp"] = error(lambda: get_rule(OP_KL_DIV)(
    valid._uop, valid._uop.inputs, dy
))
errors["mutated-vmap"] = error(lambda: get_vmap_rule(OP_KL_DIV)(
    valid._uop, {id(node): node for node in valid._uop.inputs}, 2
))
errors["mutated-onnx"] = error(lambda: bg.onnx.export_inference(
    valid, input_buffers=(input_value, target_value)
))
errors["mutated-plan"] = error(lambda: bg.gpu_plan_summary(valid))

def forged(shape, dtype="float32"):
    left = UOp(OP_BUFFER, (), shape, dtype, arg="left-" + str(shape))
    right = UOp(OP_BUFFER, (), shape, dtype, arg="right-" + str(shape))
    return UOp(
        OP_KL_DIV,
        (left, right),
        (),
        dtype,
        arg={"reduction": "sum", "batch_rank": 0, "log_target": False},
    )

errors["rank"] = error(lambda: forged((1,) * 33))
errors["work"] = error(lambda: forged((fixture["limits"]["workExtent"],)))
errors["workspace"] = error(lambda: forged(
    (fixture["limits"]["workspaceExtent"],), "float64"
))
errors["zero-hidden-work"] = error(lambda: forged(
    (0, fixture["limits"]["workExtent"])
))
errors
`);

    for (const invalid of FRAMEWORK_KL_DIV_CONFORMANCE.invalid) {
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["array-subclass"]).toMatch(/^RealizationError: .*exact ndarray/u);
    for (const key of ["mutated-cpu", "mutated-vjp", "mutated-vmap", "mutated-onnx", "mutated-plan"] as const) {
      expect(errors[key]).toMatch(/^(RealizationError|ShapeError): .*log_target/u);
    }
    expect(errors.rank).toMatch(/^ShapeError: .*rank/u);
    expect(errors.work).toMatch(/^ShapeError: .*work/u);
    expect(errors.workspace).toMatch(/^ShapeError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ShapeError: .*work/u);
  });
});
