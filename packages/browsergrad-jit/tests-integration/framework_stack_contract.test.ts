import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_STACK_CONFORMANCE } from "../../../test-support/framework-stack-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.stack framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("emits typed STACK with exact promotion, scalar/empty support, gradients, and ownership", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_STACK_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      op: string;
      arg: { axis: number };
      shape: number[];
      values: number[][];
      dtype: string;
      gradients: number[][];
      dtypeCases: Array<{ dtype: string; values: unknown[] }>;
      scalar: { shape: number[]; values: number[] };
      empty: { shape: number[]; values: unknown[] };
      mixedGradient: { floating: number[]; integralPresent: boolean };
      ownsData: boolean;
      rerunValues: number[][];
      errors: Record<string, string>;
      hostileCalls: { axis: number; sequence: number };
    }>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
inputs = tuple(
    bg.from_numpy(np.asarray(values, dtype=np.float32), requires_grad=True)
    for values in valid["inputValues"]
)
output = bg.stack(list(inputs), dim=np.int32(valid["axis"]))
values = output.numpy().tolist()
cotangent = bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32))
(output * cotangent).sum().backward()

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed_inputs = [
        bg.from_numpy(np.asarray(values, dtype=np.dtype(dtype)))
        for values, dtype in zip(case["values"], case["dtypes"])
    ]
    observed = bg.stack(typed_inputs, dim=0)
    dtype_cases.append({"dtype": observed.dtype, "values": observed.numpy().tolist()})

scalar_spec = fixture["scalar"]
scalar = bg.stack(tuple(bg.tensor(value) for value in scalar_spec["values"]), dim=scalar_spec["axis"])

empty_spec = fixture["empty"]
empty = bg.stack(tuple(
    bg.from_numpy(np.empty(tuple(empty_spec["inputShape"]), dtype=np.float32))
    for _ in range(2)
), dim=empty_spec["axis"])

mixed_spec = fixture["mixedGradient"]
mixed_float = bg.from_numpy(
    np.asarray(mixed_spec["floatingValues"], dtype=np.float32),
    requires_grad=True,
)
mixed_int = bg.from_numpy(
    np.asarray(mixed_spec["integralValues"], dtype=np.int64),
    requires_grad=True,
)
bg.stack((mixed_float, mixed_int), dim=0).sum().backward()

class HostileAxis:
    calls = 0
    def __index__(self):
        HostileAxis.calls += 1
        return 0
    def __int__(self):
        HostileAxis.calls += 1
        return 0

class HostileSequence:
    calls = 0
    def __iter__(self):
        HostileSequence.calls += 1
        return iter(inputs)

errors = {}
attempts = {
    "non-sequence": lambda: bg.stack((tensor for tensor in inputs), 1),
    "empty-sequence": lambda: bg.stack([], 0),
    "non-tensor": lambda: bg.stack([inputs[0], [1.0, 2.0]], 0),
    "bool-axis": lambda: bg.stack(inputs, True),
    "float-axis": lambda: bg.stack(inputs, 1.0),
    "low-axis": lambda: bg.stack(inputs, -3),
    "high-axis": lambda: bg.stack(inputs, 2),
    "hostile-axis": lambda: bg.stack(inputs, HostileAxis()),
    "hostile-sequence": lambda: bg.stack(HostileSequence(), 1),
    "shape-mismatch": lambda: bg.stack((inputs[0], bg.ones(3)), 0),
    "unsupported-dtype": lambda: bg.stack((
        bg.from_numpy(np.ones((1,), dtype=np.uint16)),
        bg.from_numpy(np.ones((1,), dtype=np.uint16)),
    ), 0),
    "out-mutation": lambda: bg.stack(inputs, 1, out=inputs[0]),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

array = output.numpy()
owns_data = bool(array.flags["OWNDATA"])
array[0, 0] = 999
rerun = bg.stack(inputs, dim=-1)

{
    "schema": fixture["schema"],
    "op": output._uop.op,
    "arg": dict(output._uop.arg),
    "shape": list(output.shape),
    "values": values,
    "dtype": output.dtype,
    "gradients": [tensor.grad.numpy().tolist() for tensor in inputs],
    "dtypeCases": dtype_cases,
    "scalar": {"shape": list(scalar.shape), "values": scalar.numpy().tolist()},
    "empty": {"shape": list(empty.shape), "values": empty.numpy().tolist()},
    "mixedGradient": {
        "floating": mixed_float.grad.numpy().tolist(),
        "integralPresent": mixed_int.grad is not None,
    },
    "ownsData": owns_data,
    "rerunValues": rerun.numpy().tolist(),
    "errors": errors,
    "hostileCalls": {"axis": HostileAxis.calls, "sequence": HostileSequence.calls},
}
`);

    expect(result.schema).toBe(FRAMEWORK_STACK_CONFORMANCE.schema);
    expect(result.op).toBe("STACK");
    expect(result.arg).toEqual({ axis: FRAMEWORK_STACK_CONFORMANCE.valid.axis });
    expect(result.shape).toEqual(FRAMEWORK_STACK_CONFORMANCE.valid.expectedShape);
    expect(result.values).toEqual(FRAMEWORK_STACK_CONFORMANCE.valid.expectedValues);
    expect(result.dtype).toBe("float32");
    expect(result.gradients).toEqual(FRAMEWORK_STACK_CONFORMANCE.valid.expectedGradients);
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_STACK_CONFORMANCE.dtypeCases.map((testCase) => ({
        dtype: testCase.expectedDtype,
        values: testCase.expectedStackValues,
      })),
    );
    expect(result.scalar).toEqual({
      shape: [...FRAMEWORK_STACK_CONFORMANCE.scalar.expectedShape],
      values: FRAMEWORK_STACK_CONFORMANCE.scalar.expectedValues,
    });
    expect(result.empty).toEqual({
      shape: [...FRAMEWORK_STACK_CONFORMANCE.empty.expectedShape],
      values: FRAMEWORK_STACK_CONFORMANCE.empty.expectedValues,
    });
    expect(result.mixedGradient).toEqual({
      floating: FRAMEWORK_STACK_CONFORMANCE.mixedGradient.expectedFloatingGradient,
      integralPresent: FRAMEWORK_STACK_CONFORMANCE.mixedGradient.expectedIntegralGradientPresent,
    });
    expect(result.ownsData).toBe(true);
    expect(result.rerunValues).toEqual(FRAMEWORK_STACK_CONFORMANCE.valid.expectedValues);
    expect(result.hostileCalls).toEqual({ axis: 0, sequence: 0 });
    for (const invalid of FRAMEWORK_STACK_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).not.toBe("no_error");
    }
  });

  it("provides typed indexed VJP, captured-input vmap, exact ONNX decomposition, and device refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_STACK_CONFORMANCE);
    const result = await target.run<{
      registered: boolean;
      gradient: number[];
      gradientDtype: string;
      gradientOps: string[];
      mapped: number[][][];
      mappedArg: { axis: number };
      mappedOps: string[];
      onnx: {
        opTypes: string[];
        concatInputs: string[];
        axis: number;
        outputDtype: number;
        castTo: number;
        axesInputCount: number;
        axesInitializerRank: number;
        axesInitializerValue: number;
      };
      halfOnnxError: string;
      planError: string;
      webgpuSupported: boolean[];
    }>(`
import browsergrad_jit as bg
import json
import numpy as np
from browsergrad_jit._ir import OP_NARROW, OP_STACK, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

${ONNX_PROTOBUF_TEST_HELPERS}

def parse_stack(model):
    graph = next(payload for number, wire, payload in fields(model) if number == 7 and wire == 2)
    op_types = []
    concat_inputs = None
    axis = None
    cast_to = None
    output_dtype = None
    axes_name = None
    axes_rank = None
    axes_value = None
    axes_input_count = 0
    for number, wire, payload in fields(graph):
        if number == 1 and wire == 2:
            node_fields = list(fields(payload))
            op_type = next(
                value.decode("utf-8") for field, kind, value in node_fields
                if field == 4 and kind == 2
            )
            op_types.append(op_type)
            attrs = attribute_map(node_fields)
            inputs = [
                value.decode("utf-8") for field, kind, value in node_fields
                if field == 1 and kind == 2
            ]
            if op_type == "Cast":
                cast_to = attrs["to"]
            elif op_type == "Unsqueeze":
                if len(inputs) != 2:
                    raise RuntimeError("Unsqueeze must have exact data and axes inputs")
                if axes_name is None:
                    axes_name = inputs[1]
                if inputs[1] == axes_name:
                    axes_input_count += 1
            elif op_type == "Concat":
                concat_inputs = inputs
                axis = attrs["axis"]
        elif number == 5 and wire == 2:
            tensor_fields = list(fields(payload))
            name = next(
                value.decode("utf-8") for field, kind, value in tensor_fields
                if field == 8 and kind == 2
            )
            if name.startswith("const_stack_axes_"):
                axes_name = name
                axes_rank = sum(1 for field, kind, _ in tensor_fields if field == 1 and kind == 0)
                raw = next(
                    value for field, kind, value in tensor_fields
                    if field == 9 and kind == 2
                )
                axes_value = int.from_bytes(raw[:8], "little", signed=True)
        elif number == 12 and wire == 2:
            output_dtype = value_info_dtype(payload)
    if concat_inputs is None or axis is None or axes_rank is None or axes_value is None:
        raise RuntimeError("missing exact Stack ONNX decomposition")
    return {
        "opTypes": op_types,
        "concatInputs": concat_inputs,
        "axis": axis,
        "outputDtype": output_dtype,
        "castTo": cast_to,
        "axesInputCount": axes_input_count,
        "axesInitializerRank": axes_rank,
        "axesInitializerValue": axes_value,
    }

source = bg.from_numpy(np.asarray([1, 2], dtype=np.float16))
other = bg.from_numpy(np.asarray([3, 4], dtype=np.float64))
cotangent = bg.from_numpy(np.asarray([[1, 2], [8, 16]], dtype=np.float64))
gradient = bg.func.grad(
    lambda value: (bg.stack((value, other), dim=1) * cotangent).sum()
)(source)

vmap_spec = fixture["vmap"]
mapped_source = bg.from_numpy(np.asarray(vmap_spec["mappedValues"], dtype=np.float32))
captured = bg.from_numpy(np.asarray(vmap_spec["capturedValues"], dtype=np.float32))
mapped = bg.func.vmap(lambda row: bg.stack((row, captured), dim=0))(mapped_source)

float_source = bg.from_numpy(np.asarray([0.5, 1.5], dtype=np.float32))
int_source = bg.from_numpy(np.asarray([2, 3], dtype=np.int64))
mixed = bg.stack((float_source, int_source), dim=fixture["onnx"]["axis"])
onnx = parse_stack(bg.onnx.export_inference(mixed, input_buffers=(float_source, int_source)))
half_a = bg.from_numpy(np.asarray([1], dtype=np.float16))
half_b = bg.from_numpy(np.asarray([2], dtype=np.float16))

{
    "registered": get_rule(OP_STACK) is not None,
    "gradient": gradient.numpy().tolist(),
    "gradientDtype": gradient.dtype,
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mapped": mapped.numpy().tolist(),
    "mappedArg": dict(mapped._uop.arg),
    "mappedOps": sorted({node.op for node in toposort(mapped._uop)}),
    "onnx": onnx,
    "halfOnnxError": error(lambda: bg.onnx.export_inference(
        bg.stack((half_a, half_b), 0), input_buffers=(half_a, half_b)
    )),
    "planError": error(lambda: bg.gpu_plan_summary(mixed)),
    "webgpuSupported": [OP_STACK in supported_opcodes(), OP_NARROW in supported_opcodes()],
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual([1, 8]);
    expect(result.gradientDtype).toBe("float16");
    expect(result.gradientOps).toContain("NARROW");
    expect(result.gradientOps).toContain("RESHAPE");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual(FRAMEWORK_STACK_CONFORMANCE.vmap.expectedValues);
    expect(result.mappedArg).toEqual(FRAMEWORK_STACK_CONFORMANCE.vmap.expectedArg);
    expect(result.mappedOps).toContain("BROADCAST_TO");
    expect(result.mappedOps).not.toContain("CUSTOM");
    expect(result.onnx.opTypes).toEqual(FRAMEWORK_STACK_CONFORMANCE.onnx.opTypes);
    expect(result.onnx.concatInputs).toHaveLength(FRAMEWORK_STACK_CONFORMANCE.onnx.concatInputCount);
    expect(result.onnx.axis).toBe(FRAMEWORK_STACK_CONFORMANCE.onnx.axis);
    expect(result.onnx.outputDtype).toBe(FRAMEWORK_STACK_CONFORMANCE.onnx.outputDtype);
    expect(result.onnx.castTo).toBe(FRAMEWORK_STACK_CONFORMANCE.onnx.castTo);
    expect(result.onnx.axesInputCount).toBe(2);
    expect(result.onnx.axesInitializerRank).toBe(FRAMEWORK_STACK_CONFORMANCE.onnx.axesInitializerRank);
    expect(result.onnx.axesInitializerValue).toBe(FRAMEWORK_STACK_CONFORMANCE.onnx.axesInitializerValue);
    expect(result.halfOnnxError).toMatch(/^OnnxUnmappableOp: .*float16/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*STACK/u);
    expect(result.webgpuSupported).toEqual([false, false]);
  });

  it("rejects malformed stack nodes at construction and every admitted boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_STACK
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

left = bg.from_numpy(np.ones((2,), dtype=np.float32))
right = bg.from_numpy(np.ones((2,), dtype=np.float32))
uint = bg.from_numpy(np.ones((2,), dtype=np.uint16))
valid = bg.stack((left, right), dim=0)
mutated = bg.stack((left, right), dim=0)
mutated._uop.arg["axis"] = True
dy = bg.from_numpy(np.ones(valid.shape, dtype=np.float32))._uop

large = UOp(OP_BUFFER, (), (1 << 27,), "float32", arg="stack:large")
small = UOp(OP_BUFFER, (), (1,), "float32", arg="stack:small")

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "missingInputs": error(lambda: UOp(OP_STACK, (), (0,), "float32", arg={"axis": 0})),
    "wrongFields": error(lambda: UOp(OP_STACK, (left._uop, right._uop), (2, 2), "float32", arg={})),
    "wrongShape": error(lambda: UOp(OP_STACK, (left._uop, right._uop), (4,), "float32", arg={"axis": 0})),
    "wrongDtype": error(lambda: UOp(OP_STACK, (left._uop, right._uop), (2, 2), "float64", arg={"axis": 0})),
    "badAxis": error(lambda: UOp(OP_STACK, (left._uop, right._uop), (2, 2), "float32", arg={"axis": True})),
    "badSourceShape": error(lambda: UOp(OP_STACK, (left._uop, bg.ones(3)._uop), (2, 2), "float32", arg={"axis": 0})),
    "badSourceDtype": error(lambda: UOp(OP_STACK, (left._uop, uint._uop), (2, 2), "float32", arg={"axis": 0})),
    "tooManyInputs": error(lambda: UOp(OP_STACK, (small,) * 1025, (1025, 1), "float32", arg={"axis": 0})),
    "resourceBound": error(lambda: UOp(OP_STACK, (large, large), (2, 1 << 27), "float32", arg={"axis": 0})),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_STACK)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_STACK)(
        mutated._uop,
        {id(source): source for source in mutated._uop.inputs},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(left, right))),
    "openArgPlan": error(lambda: bg.gpu_plan_summary(mutated)),
}
`);

    expect(result.missingInputs).toMatch(/^ShapeError: .*at least one input/u);
    expect(result.wrongFields).toMatch(/^ShapeError: .*arg fields/u);
    expect(result.wrongShape).toMatch(/^ShapeError: .*output shape/u);
    expect(result.wrongDtype).toMatch(/^ShapeError: .*output dtype/u);
    expect(result.badAxis).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.badSourceShape).toMatch(/^ShapeError: .*does not match/u);
    expect(result.badSourceDtype).toMatch(/^ShapeError: .*unsupported dtype/u);
    expect(result.tooManyInputs).toMatch(/^ShapeError: .*input count/u);
    expect(result.resourceBound).toMatch(/^ShapeError: .*byte ceiling/u);
    expect(result.openArgCpu).toMatch(/^RealizationError: .*normalized integer/u);
    expect(result.openArgVjp).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.openArgVmap).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.openArgOnnx).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.openArgPlan).toMatch(/^ShapeError: .*normalized integer/u);
  });
});
