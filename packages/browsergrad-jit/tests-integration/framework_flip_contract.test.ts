import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed flip framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("normalizes one axis and preserves values, dtypes, owning results, and closure gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      axis: number;
      unsignedAxis: number;
      values: number[][];
      ownsData: boolean;
      gradient: number[][];
      intValues: number[][];
      intDtype: string;
      boolValues: boolean[][];
      boolDtype: string;
    }>(`
import browsergrad_jit as bg
import numpy as np

source = bg.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), requires_grad=True)
weights = bg.from_numpy(np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], dtype=np.float32))
flipped = source.flip(np.int32(-1))
unsigned_axis = source.flip(np.uint64(1))
array = flipped.numpy()
(flipped * weights).sum().backward()

integer = bg.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.int32)).flip(0).numpy()
boolean = bg.from_numpy(np.array([[True, False], [False, True]], dtype=np.bool_)).flip(1).numpy()

{
    "op": flipped._uop.op,
    "axis": flipped._uop.arg["axis"],
    "unsignedAxis": unsigned_axis._uop.arg["axis"],
    "values": array.tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "gradient": source.grad.numpy().tolist(),
    "intValues": integer.tolist(),
    "intDtype": str(integer.dtype),
    "boolValues": boolean.tolist(),
    "boolDtype": str(boolean.dtype),
}
`);

    expect(result).toEqual({
      op: "FLIP",
      axis: 1,
      unsignedAxis: 1,
      values: [[3, 2, 1], [6, 5, 4]],
      ownsData: true,
      gradient: [[4, 2, 1], [32, 16, 8]],
      intValues: [[3, 4], [1, 2]],
      intDtype: "int32",
      boolValues: [[false, true], [true, false]],
      boolDtype: "bool",
    });
  });

  it("provides involutive symbolic VJP, axis-shifting vmap, ONNX Slice, and explicit device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      gradientOps: string[];
      mapped: number[][];
      sliceInputs: string[];
      sliceValues: string[];
      onnxHasCustom: boolean;
      onnxSupportedDtypes: Record<string, boolean>;
      planError: string;
    }>(`
import browsergrad_jit as bg
import numpy as np
import struct
from browsergrad_jit._ir import OP_FLIP, toposort
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
weights = bg.from_numpy(np.array([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], dtype=np.float32))
gradient_tensor = bg.func.grad(lambda value: (value.flip(1) * weights).sum())(source)
mapped = bg.func.vmap(lambda row: row.flip(0))(source).numpy()
onnx = bg.onnx.export_inference(source.flip(1), input_buffers=(source,))
onnx_supported_dtypes = {}
for dtype in (np.float32, np.int32, np.int64, np.bool_):
    export_source = bg.from_numpy(np.array([1, 0], dtype=dtype))
    payload = bg.onnx.export_inference(
        export_source.flip(0),
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
slice_inputs = None
initializer_values = {}
for number, wire, payload in fields(graph):
    if number == 1 and wire == 2:
        node_fields = list(fields(payload))
        op_type = next(
            value.decode("utf-8") for field, kind, value in node_fields
            if field == 4 and kind == 2
        )
        if op_type == "Slice":
            slice_inputs = [
                value.decode("utf-8") for field, kind, value in node_fields
                if field == 1 and kind == 2
            ]
    elif number == 5 and wire == 2:
        tensor_fields = list(fields(payload))
        name = next(
            value.decode("utf-8") for field, kind, value in tensor_fields
            if field == 8 and kind == 2
        )
        if name.startswith("const_flip_"):
            dtype = next(value for field, kind, value in tensor_fields if field == 2 and kind == 0)
            raw = next(value for field, kind, value in tensor_fields if field == 9 and kind == 2)
            if dtype != 7 or len(raw) != 8:
                raise RuntimeError("FLIP Slice initializer is not one int64 scalar")
            initializer_values[name] = struct.unpack("<q", raw)[0]
if slice_inputs is None:
    raise RuntimeError("missing Slice node")

try:
    bg.gpu_plan_summary(source.flip(1))
    plan_error = "no_error"
except Exception as exc:
    plan_error = type(exc).__name__ + ": " + str(exc)

{
    "registered": get_rule(OP_FLIP) is not None,
    "gradient": gradient_tensor.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient_tensor._uop)}),
    "mapped": mapped.tolist(),
    "sliceInputs": slice_inputs[1:],
    "sliceValues": [str(initializer_values[name]) for name in slice_inputs[1:]],
    "onnxHasCustom": b"CUSTOM" in onnx,
    "onnxSupportedDtypes": onnx_supported_dtypes,
    "planError": plan_error,
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual([[4, 2, 1], [32, 16, 8]]);
    expect(result.gradientOps).toContain("FLIP");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual([[3, 2, 1], [6, 5, 4]]);
    expect(result.sliceInputs).toEqual([
      "const_flip_starts_0",
      "const_flip_ends_0",
      "const_flip_axes_0",
      "const_flip_steps_0",
    ]);
    expect(result.sliceValues).toEqual(["-1", "-9223372036854775808", "1", "-1"]);
    expect(result.onnxHasCustom).toBe(false);
    expect(result.onnxSupportedDtypes).toEqual({
      float32: true,
      int32: true,
      int64: true,
      bool: true,
    });
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*FLIP/u);
  });

  it("rejects coercion, wrapped axes, scalar rank, invalid contracts, and consumer-boundary mutation", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_FLIP
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
float64_source = bg.from_numpy(np.arange(6, dtype=np.float64).reshape(2, 3))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

class HostileIndex:
    calls = 0
    def __index__(self):
        HostileIndex.calls += 1
        return 0

mutated = source.flip(1)
mutated._uop.arg["axis"] = -1
dy = bg.from_numpy(np.ones(source.shape, dtype=np.float32))._uop

errors = {
    "bool": error(lambda: source.flip(True)),
    "float": error(lambda: source.flip(1.0)),
    "string": error(lambda: source.flip("1")),
    "hostile": error(lambda: source.flip(HostileIndex())),
    "positiveRange": error(lambda: source.flip(2)),
    "negativeRange": error(lambda: source.flip(-3)),
    "scalar": error(lambda: bg.tensor(1.0).flip(0)),
    "wrongFields": error(lambda: UOp(OP_FLIP, (source._uop,), source.shape, "float32", arg={})),
    "wrongShape": error(lambda: UOp(OP_FLIP, (source._uop,), (1, 2, 3), "float32", arg={"axis": 1})),
    "wrongDtype": error(lambda: UOp(OP_FLIP, (source._uop,), source.shape, "float64", arg={"axis": 1})),
    "boolArg": error(lambda: UOp(OP_FLIP, (source._uop,), source.shape, "float32", arg={"axis": True})),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_FLIP)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_FLIP)(
        mutated._uop,
        {id(mutated._uop.inputs[0]): mutated._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(source,))),
    "onnxFloat64": error(lambda: bg.onnx.export_inference(
        float64_source.flip(1),
        input_buffers=(float64_source,),
    )),
}
{"errors": errors, "hostileCalls": HostileIndex.calls}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.bool).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.float).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.string).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.hostile).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.positiveRange).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.negativeRange).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.scalar).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.wrongFields).toMatch(/^ShapeError: .*arg fields/u);
    expect(result.errors.wrongShape).toMatch(/^ShapeError: .*preserve its input shape/u);
    expect(result.errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(result.errors.boolArg).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.errors.openArgCpu).toMatch(/^RealizationError: .*out of range/u);
    expect(result.errors.openArgVjp).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.openArgVmap).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.openArgOnnx).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.onnxFloat64).toMatch(/^OnnxUnmappableOp: .*dtype/u);
  });
});
