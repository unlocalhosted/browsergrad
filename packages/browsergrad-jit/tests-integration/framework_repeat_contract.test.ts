import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_REPEAT_CONFORMANCE } from "../../../test-support/framework-repeat-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.repeat framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("tiles values with owning dtype-preserving CPU results and block-sum closure gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      repeats: number[];
      shape: number[];
      values: number[][];
      ownsData: boolean;
      gradient: number[][];
      halfGradient: number[];
      halfGradientDtype: string;
      leadingValues: number[][];
      scalarValues: number[];
      unsignedRepeats: number[];
      intDtype: string;
      boolDtype: string;
      emptyShape: number[];
      emptySize: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

source = bg.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
weights = bg.from_numpy(np.arange(1, 25, dtype=np.float32).reshape(4, 6))
repeated = source.repeat(2, 3)
array = repeated.numpy()
(repeated * weights).sum().backward()
half_source = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float16), requires_grad=True)
half_source.repeat(2).sum().backward()

leading = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).repeat(2, 3).numpy()
scalar = bg.tensor(5.0).repeat(3).numpy()
unsigned = source.repeat(np.uint64(2), np.uint16(3))
integer = bg.from_numpy(np.array([1, 2], dtype=np.int32)).repeat(2).numpy()
boolean = bg.from_numpy(np.array([True, False], dtype=np.bool_)).repeat(2).numpy()
empty = source.repeat(2, 0).numpy()

{
    "op": repeated._uop.op,
    "repeats": list(repeated._uop.arg["repeats"]),
    "shape": list(array.shape),
    "values": array.tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "gradient": source.grad.numpy().tolist(),
    "halfGradient": half_source.grad.numpy().tolist(),
    "halfGradientDtype": half_source.grad.dtype,
    "leadingValues": leading.tolist(),
    "scalarValues": scalar.tolist(),
    "unsignedRepeats": list(unsigned._uop.arg["repeats"]),
    "intDtype": str(integer.dtype),
    "boolDtype": str(boolean.dtype),
    "emptyShape": list(empty.shape),
    "emptySize": int(empty.size),
}
`);

    expect(result).toEqual({
      op: "REPEAT",
      repeats: [2, 3],
      shape: [4, 6],
      values: [
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
      ],
      ownsData: true,
      gradient: [[54, 60], [90, 96]],
      halfGradient: [2, 2],
      halfGradientDtype: "float16",
      leadingValues: [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]],
      scalarValues: [5, 5, 5],
      unsignedRepeats: [2, 3],
      intDtype: "int32",
      boolDtype: "bool",
      emptyShape: [4, 0],
      emptySize: 0,
    });
  });

  it("provides typed block-sum VJP, batch-safe vmap, exact ONNX Tile, and device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      leadingGradient: number[];
      scalarGradient: number;
      zeroGradient: number[][];
      gradientOps: string[];
      mapped: number[][];
      mappedRepeats: number[];
      mappedLeading: number[][][];
      mappedLeadingInputOp: string;
      mappedLeadingRepeats: number[];
      tileInputs: string[];
      tileRepeats: string[];
      onnxSupportedDtypes: Record<string, boolean>;
      onnxHasCustom: boolean;
      planError: string;
    }>(`
import browsergrad_jit as bg
import numpy as np
import struct
from browsergrad_jit._ir import OP_REPEAT, toposort
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
weights = bg.from_numpy(np.arange(1, 25, dtype=np.float32).reshape(4, 6))
gradient_tensor = bg.func.grad(lambda value: (value.repeat(2, 3) * weights).sum())(source)
leading_source = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
leading_gradient = bg.func.grad(lambda value: value.repeat(2, 3).sum())(leading_source)
scalar_source = bg.tensor(2.0)
scalar_gradient = bg.func.grad(lambda value: value.repeat(3).sum())(scalar_source)
zero_gradient = bg.func.grad(lambda value: value.repeat(2, 0).sum())(source)
mapped_tensor = bg.func.vmap(lambda row: row.repeat(2))(source)
mapped = mapped_tensor.numpy()
mapped_leading_tensor = bg.func.vmap(lambda row: row.repeat(2, 3))(source)
mapped_leading = mapped_leading_tensor.numpy()
onnx = bg.onnx.export_inference(source.repeat(2, 3), input_buffers=(source,))

onnx_supported_dtypes = {}
for dtype in (np.float32, np.int32, np.int64, np.bool_):
    export_source = bg.from_numpy(np.array([1, 0], dtype=dtype))
    payload = bg.onnx.export_inference(
        export_source.repeat(2),
        input_buffers=(export_source,),
    )
    onnx_supported_dtypes[str(np.dtype(dtype))] = len(payload) > 0

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
            raise RuntimeError(f"unexpected protobuf wire type {wire}")

graph = next(payload for number, wire, payload in fields(onnx) if number == 7 and wire == 2)
tile_inputs = None
initializer_values = {}
for number, wire, payload in fields(graph):
    if number == 1 and wire == 2:
        node_fields = list(fields(payload))
        op_type = next(
            value.decode("utf-8") for field, kind, value in node_fields
            if field == 4 and kind == 2
        )
        if op_type == "Tile":
            tile_inputs = [
                value.decode("utf-8") for field, kind, value in node_fields
                if field == 1 and kind == 2
            ]
    elif number == 5 and wire == 2:
        tensor_fields = list(fields(payload))
        name = next(
            value.decode("utf-8") for field, kind, value in tensor_fields
            if field == 8 and kind == 2
        )
        if name.startswith("const_repeat_"):
            dtype = next(value for field, kind, value in tensor_fields if field == 2 and kind == 0)
            raw = next(value for field, kind, value in tensor_fields if field == 9 and kind == 2)
            if dtype != 7 or len(raw) % 8 != 0:
                raise RuntimeError("REPEAT Tile initializer is not an int64 vector")
            initializer_values[name] = struct.unpack("<" + "q" * (len(raw) // 8), raw)
if tile_inputs is None:
    raise RuntimeError("missing Tile node")

try:
    bg.gpu_plan_summary(source.repeat(2, 3))
    plan_error = "no_error"
except Exception as exc:
    plan_error = type(exc).__name__ + ": " + str(exc)

{
    "registered": get_rule(OP_REPEAT) is not None,
    "gradient": gradient_tensor.numpy().tolist(),
    "leadingGradient": leading_gradient.numpy().tolist(),
    "scalarGradient": float(scalar_gradient.item()),
    "zeroGradient": zero_gradient.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient_tensor._uop)}),
    "mapped": mapped.tolist(),
    "mappedRepeats": list(mapped_tensor._uop.arg["repeats"]),
    "mappedLeading": mapped_leading.tolist(),
    "mappedLeadingInputOp": mapped_leading_tensor._uop.inputs[0].op,
    "mappedLeadingRepeats": list(mapped_leading_tensor._uop.arg["repeats"]),
    "tileInputs": tile_inputs[1:],
    "tileRepeats": [str(value) for value in initializer_values[tile_inputs[1]]],
    "onnxSupportedDtypes": onnx_supported_dtypes,
    "onnxHasCustom": b"CUSTOM" in onnx,
    "planError": plan_error,
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual([[54, 60], [90, 96]]);
    expect(result.leadingGradient).toEqual([6, 6]);
    expect(result.scalarGradient).toBe(3);
    expect(result.zeroGradient).toEqual([[0, 0], [0, 0]]);
    expect(result.gradientOps).toContain("REDUCE");
    expect(result.gradientOps).toContain("RESHAPE");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual([[1, 2, 1, 2], [3, 4, 3, 4]]);
    expect(result.mappedRepeats).toEqual([1, 2]);
    expect(result.mappedLeading).toEqual([
      [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]],
      [[3, 4, 3, 4, 3, 4], [3, 4, 3, 4, 3, 4]],
    ]);
    expect(result.mappedLeadingInputOp).toBe("RESHAPE");
    expect(result.mappedLeadingRepeats).toEqual([1, 2, 3]);
    expect(result.tileInputs).toEqual(["const_repeat_0"]);
    expect(result.tileRepeats).toEqual(["2", "3"]);
    expect(result.onnxSupportedDtypes).toEqual({
      float32: true,
      int32: true,
      int64: true,
      bool: true,
    });
    expect(result.onnxHasCustom).toBe(false);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*REPEAT/u);
  });

  it("rejects coercion, resource abuse, malformed contracts, and consumer-boundary mutation", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_REPEAT
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(4, dtype=np.float32).reshape(2, 2))
empty = bg.from_numpy(np.empty((0,), dtype=np.float32))
float64_source = bg.from_numpy(np.arange(4, dtype=np.float64).reshape(2, 2))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

class HostileScalar:
    calls = 0
    def __index__(self):
        HostileScalar.calls += 1
        return 2
    def __int__(self):
        HostileScalar.calls += 1
        return 2

class HostileTuple(tuple):
    def __iter__(self):
        HostileScalar.calls += 1
        return super().__iter__()

mutated = source.repeat(2, 3)
mutated._uop.arg["repeats"] = (-1, 3)
dy = bg.from_numpy(np.ones((4, 6), dtype=np.float32))._uop

errors = {
    "empty": error(lambda: source.repeat()),
    "rankShort": error(lambda: source.repeat(2)),
    "bool": error(lambda: source.repeat(2, True)),
    "float": error(lambda: source.repeat(2, 3.0)),
    "string": error(lambda: source.repeat(2, "3")),
    "hostile": error(lambda: source.repeat(2, HostileScalar())),
    "hostileTuple": error(lambda: bg.tensor(1.0).repeat(HostileTuple((2,)))),
    "negative": error(lambda: source.repeat(2, -1)),
    "factorCeiling": error(lambda: empty.repeat(1073741825)),
    "rankCeiling": error(lambda: bg.tensor(1.0).repeat(*([1] * 33))),
    "listArg": error(lambda: UOp(OP_REPEAT, (source._uop,), (4, 6), "float32", arg={"repeats": [2, 3]})),
    "wrongShape": error(lambda: UOp(OP_REPEAT, (source._uop,), (2, 6), "float32", arg={"repeats": (2, 3)})),
    "wrongDtype": error(lambda: UOp(OP_REPEAT, (source._uop,), (4, 6), "float64", arg={"repeats": (2, 3)})),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_REPEAT)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_REPEAT)(
        mutated._uop,
        {id(mutated._uop.inputs[0]): mutated._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(source,))),
    "onnxFloat64": error(lambda: bg.onnx.export_inference(
        float64_source.repeat(2, 3),
        input_buffers=(float64_source,),
    )),
}
{"errors": errors, "hostileCalls": HostileScalar.calls}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.empty).toMatch(/^ShapeError: .*at least one/u);
    expect(result.errors.rankShort).toMatch(/^ShapeError: .*shorter than input rank/u);
    expect(result.errors.bool).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.float).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.string).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.hostile).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.hostileTuple).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.negative).toMatch(/^ShapeError: .*must be in/u);
    expect(result.errors.factorCeiling).toMatch(/^ShapeError: .*must be in/u);
    expect(result.errors.rankCeiling).toMatch(/^ShapeError: .*axis ceiling/u);
    expect(result.errors.listArg).toMatch(/^ShapeError: .*canonical tuple/u);
    expect(result.errors.wrongShape).toMatch(/^ShapeError: .*derived shape/u);
    expect(result.errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(result.errors.openArgCpu).toMatch(/^RealizationError: .*must be in/u);
    expect(result.errors.openArgVjp).toMatch(/^ShapeError: .*must be in/u);
    expect(result.errors.openArgVmap).toMatch(/^ShapeError: .*must be in/u);
    expect(result.errors.openArgOnnx).toMatch(/^ShapeError: .*must be in/u);
    expect(result.errors.onnxFloat64).toMatch(/^OnnxUnmappableOp: .*dtype/u);
  });

  it("matches the shared eager/lazy repeat conformance fixture", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_REPEAT_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[][];
      dtypes: string[];
      errors: Record<string, string>;
    }>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
output = bg.from_numpy(base).repeat(*valid["requestedRepeats"]).numpy()

dtypes = []
for case in fixture["dtypeCases"]:
    source = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(bg.from_numpy(source).repeat(*valid["requestedRepeats"]).dtype)

errors = {}
for case in fixture["invalid"]:
    source = np.arange(np.prod(case["inputShape"]), dtype=np.float32).reshape(case["inputShape"])
    try:
        bg.from_numpy(source).repeat(*case["requestedRepeats"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.tolist(),
    "dtypes": dtypes,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_REPEAT_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_REPEAT_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_REPEAT_CONFORMANCE.valid.outputValues);
    expect(result.dtypes).toEqual(
      FRAMEWORK_REPEAT_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const fixture of FRAMEWORK_REPEAT_CONFORMANCE.invalid) {
      expect(result.errors[fixture.id]).toContain(fixture.message);
    }
  });
});
