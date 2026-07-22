import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_CUMSUM_CONFORMANCE } from "../../../test-support/framework-cumsum-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.cumsum framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("emits typed CUMSUM with inclusive values, exact promotion, closure gradients, and owning CPU results", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CUMSUM_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      op: string;
      arg: { axis: number; reverse: boolean };
      values: number[][];
      alternateAxisValues: number[][];
      gradient: number[][];
      dtypeCases: Array<{ dtype: string; values: unknown[] }>;
      emptyShape: number[];
      emptyValues: unknown[];
      ownsData: boolean;
      rerunValues: number[][];
      errors: Record<string, string>;
      hostileCalls: { axis: number; dtype: number };
    }>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
source = bg.from_numpy(base, requires_grad=True)
output = source.cumsum(np.int32(valid["axis"]))
output_values = output.numpy().tolist()
output.sum().backward()
alternate = bg.cumsum(source, dim=-2)

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed = bg.from_numpy(np.asarray(case["values"], dtype=np.dtype(case["sourceDtype"])))
    kwargs = {} if case["dtype"] is None else {"dtype": case["dtype"]}
    observed = typed.cumsum(0, **kwargs)
    dtype_cases.append({"dtype": observed.dtype, "values": observed.numpy().tolist()})

empty_spec = fixture["empty"]
empty = bg.cumsum(
    bg.from_numpy(np.empty(empty_spec["shape"], dtype=np.float32)),
    dim=empty_spec["axis"],
)

class HostileAxis:
    calls = 0
    def __int__(self):
        HostileAxis.calls += 1
        return 0

class HostileDtype:
    calls = 0
    def __str__(self):
        HostileDtype.calls += 1
        return "float32"

errors = {}
attempts = {
    "scalar-input": lambda: bg.cumsum(bg.tensor(1.0), 0),
    "bool-axis": lambda: source.cumsum(True),
    "float-axis": lambda: source.cumsum(1.0),
    "low-axis": lambda: source.cumsum(-3),
    "high-axis": lambda: source.cumsum(2),
    "hostile-axis": lambda: source.cumsum(HostileAxis()),
    "list-dtype": lambda: source.cumsum(1, dtype=["float32"]),
    "hostile-dtype": lambda: source.cumsum(1, dtype=HostileDtype()),
    "unknown-dtype": lambda: source.cumsum(1, dtype="complex64"),
    "unsupported-source-dtype": lambda: bg.from_numpy(np.ones((2, 3), dtype=np.uint16)).cumsum(1),
    "out-mutation": lambda: bg.cumsum(source, 1, out=source),
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
rerun = output.numpy()

{
    "schema": fixture["schema"],
    "op": output._uop.op,
    "arg": dict(output._uop.arg),
    "values": output_values,
    "alternateAxisValues": alternate.numpy().tolist(),
    "gradient": source.grad.numpy().tolist(),
    "dtypeCases": dtype_cases,
    "emptyShape": list(empty.shape),
    "emptyValues": empty.numpy().tolist(),
    "ownsData": owns_data,
    "rerunValues": rerun.tolist(),
    "errors": errors,
    "hostileCalls": {"axis": HostileAxis.calls, "dtype": HostileDtype.calls},
}
`);

    expect(result.schema).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.schema);
    expect(result.op).toBe("CUMSUM");
    expect(result.arg).toEqual({ axis: FRAMEWORK_CUMSUM_CONFORMANCE.valid.axis, reverse: false });
    expect(result.values).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.values);
    expect(result.alternateAxisValues).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.alternateAxisValues);
    expect(result.gradient).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.gradient);
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_CUMSUM_CONFORMANCE.dtypeCases.map((testCase) => ({
        dtype: testCase.expectedDtype,
        values: testCase.expectedValues,
      })),
    );
    expect(result.emptyShape).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.empty.shape);
    expect(result.emptyValues).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.empty.values);
    expect(result.ownsData).toBe(true);
    expect(result.rerunValues).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.values);
    expect(result.hostileCalls).toEqual({ axis: 0, dtype: 0 });
    for (const invalid of FRAMEWORK_CUMSUM_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).not.toBe("no_error");
    }
  });

  it("provides scan VJP, shifted vmap, exact ONNX Cast/CumSum, and explicit device refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CUMSUM_CONFORMANCE);
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      gradientOps: string[];
      mapped: number[][];
      mappedArg: { axis: number; reverse: boolean };
      typedGradient: number[];
      typedGradientDtype: string;
      autogradRequiresGrad: boolean[];
      onnx: {
        opTypes: string[];
        cumsumInputs: string[];
        axis: number;
        axisRank: number;
        axisDtype: number;
        outputDtype: number;
        castTo: number;
        exclusive: number;
        reverse: number;
      };
      floatOpTypes: string[];
      halfOnnxError: string;
      planError: string;
      webgpuSupported: boolean;
    }>(`
import browsergrad_jit as bg
import json
import numpy as np
import struct
from browsergrad_jit._ir import OP_CUMSUM, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

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

def attribute_map(node_fields):
    result = {}
    for field, kind, payload in node_fields:
        if field != 5 or kind != 2:
            continue
        attribute_fields = list(fields(payload))
        name = next(
            value.decode("utf-8") for child, child_kind, value in attribute_fields
            if child == 1 and child_kind == 2
        )
        result[name] = next(
            value for child, child_kind, value in attribute_fields
            if child == 3 and child_kind == 0
        )
    return result

def value_info_dtype(value_info):
    type_proto = next(
        payload for number, wire, payload in fields(value_info)
        if number == 2 and wire == 2
    )
    tensor_type = next(
        payload for number, wire, payload in fields(type_proto)
        if number == 1 and wire == 2
    )
    return next(
        value for number, wire, value in fields(tensor_type)
        if number == 1 and wire == 0
    )

def parse_scan(model):
    graph = next(payload for number, wire, payload in fields(model) if number == 7 and wire == 2)
    op_types = []
    cumsum_inputs = None
    cumsum_attrs = None
    cast_to = None
    initializers = {}
    output_dtype = None
    for number, wire, payload in fields(graph):
        if number == 1 and wire == 2:
            node_fields = list(fields(payload))
            op_type = next(
                value.decode("utf-8") for field, kind, value in node_fields
                if field == 4 and kind == 2
            )
            op_types.append(op_type)
            attrs = attribute_map(node_fields)
            if op_type == "Cast":
                cast_to = attrs["to"]
            elif op_type == "CumSum":
                cumsum_inputs = [
                    value.decode("utf-8") for field, kind, value in node_fields
                    if field == 1 and kind == 2
                ]
                cumsum_attrs = attrs
        elif number == 5 and wire == 2:
            tensor_fields = list(fields(payload))
            name = next(
                value.decode("utf-8") for field, kind, value in tensor_fields
                if field == 8 and kind == 2
            )
            initializers[name] = {
                "rank": sum(1 for field, kind, _ in tensor_fields if field == 1 and kind == 0),
                "dtype": next(value for field, kind, value in tensor_fields if field == 2 and kind == 0),
                "raw": next(value for field, kind, value in tensor_fields if field == 9 and kind == 2),
            }
        elif number == 12 and wire == 2:
            output_dtype = value_info_dtype(payload)
    if cumsum_inputs is None or len(cumsum_inputs) != 2 or cumsum_attrs is None:
        raise RuntimeError("missing exact CumSum wiring")
    axis_initializer = initializers[cumsum_inputs[1]]
    return {
        "opTypes": op_types,
        "cumsumInputs": cumsum_inputs,
        "axis": struct.unpack("<q", axis_initializer["raw"])[0],
        "axisRank": axis_initializer["rank"],
        "axisDtype": axis_initializer["dtype"],
        "outputDtype": output_dtype,
        "castTo": cast_to,
        "exclusive": cumsum_attrs["exclusive"],
        "reverse": cumsum_attrs["reverse"],
    }

source = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32))
cotangent = bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32))
gradient = bg.func.grad(lambda value: (value.cumsum(valid["axis"]) * cotangent).sum())(source)
mapped = bg.func.vmap(lambda row: row.cumsum(0))(source)

typed_spec = fixture["typedGradient"]
typed_source = bg.from_numpy(np.asarray(typed_spec["values"], dtype=np.dtype(typed_spec["sourceDtype"])))
typed_cotangent = bg.from_numpy(np.asarray(typed_spec["cotangent"], dtype=np.dtype(typed_spec["outputDtype"])))
typed_gradient = bg.func.grad(
    lambda value: (value.cumsum(0, dtype=typed_spec["outputDtype"]) * typed_cotangent).sum()
)(typed_source)
float_to_integral = source.cumsum(1, dtype="int64")
integral_to_float = bg.from_numpy(
    np.asarray([1, 2], dtype=np.int32), requires_grad=True
).cumsum(0, dtype="float32")

int_source = bg.from_numpy(np.asarray([[2147483647, 1]], dtype=np.int32))
int_output = int_source.cumsum(1)
onnx = parse_scan(bg.onnx.export_inference(int_output, input_buffers=(int_source,)))
float_model = parse_scan(bg.onnx.export_inference(source.cumsum(1), input_buffers=(source,)))
half = bg.from_numpy(np.asarray([[1, 2]], dtype=np.float16))

{
    "registered": get_rule(OP_CUMSUM) is not None,
    "gradient": gradient.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mapped": mapped.numpy().tolist(),
    "mappedArg": dict(mapped._uop.arg),
    "typedGradient": typed_gradient.numpy().tolist(),
    "typedGradientDtype": typed_gradient.dtype,
    "autogradRequiresGrad": [float_to_integral.requires_grad, integral_to_float.requires_grad],
    "onnx": onnx,
    "floatOpTypes": float_model["opTypes"],
    "halfOnnxError": error(lambda: bg.onnx.export_inference(half.cumsum(1), input_buffers=(half,))),
    "planError": error(lambda: bg.gpu_plan_summary(source.cumsum(1))),
    "webgpuSupported": OP_CUMSUM in supported_opcodes(),
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.cotangentGradient);
    expect(result.gradientOps).toContain("CUMSUM");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.valid.values);
    expect(result.mappedArg).toEqual({ axis: 1, reverse: false });
    expect(result.typedGradient).toEqual(FRAMEWORK_CUMSUM_CONFORMANCE.typedGradient.expected);
    expect(result.typedGradientDtype).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.typedGradient.expectedDtype);
    expect(result.autogradRequiresGrad).toEqual([
      FRAMEWORK_CUMSUM_CONFORMANCE.autograd.floatToIntegralRequiresGrad,
      FRAMEWORK_CUMSUM_CONFORMANCE.autograd.integralToFloatRequiresGrad,
    ]);
    expect(result.onnx.opTypes).toEqual(["Cast", "CumSum", "Identity"]);
    expect(result.onnx.cumsumInputs).toHaveLength(2);
    expect(result.onnx.axis).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.onnx.axis);
    expect(result.onnx.axisRank).toBe(0);
    expect(result.onnx.axisDtype).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.onnx.axisDtype);
    expect(result.onnx.outputDtype).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.onnx.outputDtype);
    expect(result.onnx.castTo).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.onnx.castTo);
    expect(result.onnx.exclusive).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.onnx.exclusive);
    expect(result.onnx.reverse).toBe(FRAMEWORK_CUMSUM_CONFORMANCE.onnx.reverse);
    expect(result.floatOpTypes).toEqual(["CumSum", "Identity"]);
    expect(result.halfOnnxError).toMatch(/^OnnxUnmappableOp: .*float16/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*CUMSUM/u);
    expect(result.webgpuSupported).toBe(false);
  });

  it("rejects malformed scan nodes at construction and every admitted boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, string> & { capturedVmap: string }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_CUMSUM
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
scalar = bg.tensor(1.0)
uint_source = bg.from_numpy(np.arange(6, dtype=np.uint16).reshape(2, 3))
valid = source.cumsum(1)
mutated = source.cumsum(1)
mutated._uop.arg["reverse"] = 1
dy = bg.from_numpy(np.ones(source.shape, dtype=np.float32))._uop

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "scalarNode": error(lambda: UOp(OP_CUMSUM, (scalar._uop,), (), "float32", arg={"axis": 0, "reverse": False})),
    "wrongFields": error(lambda: UOp(OP_CUMSUM, (source._uop,), source.shape, "float32", arg={"axis": 1})),
    "wrongShape": error(lambda: UOp(OP_CUMSUM, (source._uop,), (3, 2), "float32", arg={"axis": 1, "reverse": False})),
    "badAxis": error(lambda: UOp(OP_CUMSUM, (source._uop,), source.shape, "float32", arg={"axis": True, "reverse": False})),
    "badReverse": error(lambda: UOp(OP_CUMSUM, (source._uop,), source.shape, "float32", arg={"axis": 1, "reverse": 0})),
    "badSourceDtype": error(lambda: UOp(OP_CUMSUM, (uint_source._uop,), uint_source.shape, "int64", arg={"axis": 1, "reverse": False})),
    "badOutputDtype": error(lambda: UOp(OP_CUMSUM, (source._uop,), source.shape, "uint16", arg={"axis": 1, "reverse": False})),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_CUMSUM)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_CUMSUM)(
        mutated._uop,
        {id(mutated._uop.inputs[0]): mutated._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(source,))),
    "openArgPlan": error(lambda: bg.gpu_plan_summary(mutated)),
    "capturedVmap": error(lambda: get_vmap_rule(OP_CUMSUM)(
        valid._uop,
        {id(valid._uop.inputs[0]): valid._uop.inputs[0]},
        1,
    )),
}
`);

    expect(result.scalarNode).toMatch(/^ShapeError: .*rank at least one/u);
    expect(result.wrongFields).toMatch(/^ShapeError: .*arg fields/u);
    expect(result.wrongShape).toMatch(/^ShapeError: .*preserve its input shape/u);
    expect(result.badAxis).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.badReverse).toMatch(/^ShapeError: .*must be a boolean/u);
    expect(result.badSourceDtype).toMatch(/^ShapeError: .*source dtype/u);
    expect(result.badOutputDtype).toMatch(/^ShapeError: .*output dtype/u);
    expect(result.openArgCpu).toMatch(/^RealizationError: .*reverse must be a boolean/u);
    expect(result.openArgVjp).toMatch(/^ShapeError: .*reverse must be a boolean/u);
    expect(result.openArgVmap).toMatch(/^ShapeError: .*reverse must be a boolean/u);
    expect(result.openArgOnnx).toMatch(/^ShapeError: .*reverse must be a boolean/u);
    expect(result.openArgPlan).toMatch(/^ShapeError: .*reverse must be a boolean/u);
    expect(result.capturedVmap).toMatch(/^JitNotImplementedError: .*leading mapped axis/u);
  });
});
