import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE } from "../../../test-support/framework-repeat-interleave-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.repeat_interleave framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("preserves values, dtype, ownership, normalization, and closure gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      arg: { repeats: number; axis: number };
      values: number[][];
      ownsData: boolean;
      gradient: number[][];
      halfGradient: number[];
      halfGradientDtype: string;
      negativeAxis: number[][];
      unsignedArg: { repeats: number; axis: number };
      intDtype: string;
      boolDtype: string;
      emptyShape: number[];
      emptyGradient: number[][];
    }>(`
import browsergrad_jit as bg
import numpy as np

source = bg.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
weights = bg.from_numpy(np.arange(1, 9, dtype=np.float32).reshape(2, 4))
repeated = source.repeat_interleave(2, dim=1)
array = repeated.numpy()
(repeated * weights).sum().backward()

half_source = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float16), requires_grad=True)
half_source.repeat_interleave(2, dim=0).sum().backward()
negative_axis = bg.from_numpy(np.array([[1, 2]], dtype=np.int32)).repeat_interleave(2, dim=-1)
unsigned = source.repeat_interleave(np.uint64(2), dim=np.int16(0))
boolean = bg.from_numpy(np.array([True, False], dtype=np.bool_)).repeat_interleave(2, dim=0)
empty_source = bg.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
empty = empty_source.repeat_interleave(0, dim=1)
empty.sum().backward()

{
    "op": repeated._uop.op,
    "arg": repeated._uop.arg,
    "values": array.tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "gradient": source.grad.numpy().tolist(),
    "halfGradient": half_source.grad.numpy().tolist(),
    "halfGradientDtype": half_source.grad.dtype,
    "negativeAxis": negative_axis.numpy().tolist(),
    "unsignedArg": unsigned._uop.arg,
    "intDtype": negative_axis.dtype,
    "boolDtype": boolean.dtype,
    "emptyShape": list(empty.shape),
    "emptyGradient": empty_source.grad.numpy().tolist(),
}
`);

    expect(result).toEqual({
      op: "REPEAT_INTERLEAVE",
      arg: { repeats: 2, axis: 1 },
      values: [[1, 1, 2, 2], [3, 3, 4, 4]],
      ownsData: true,
      gradient: [[3, 7], [11, 15]],
      halfGradient: [2, 2],
      halfGradientDtype: "float16",
      negativeAxis: [[1, 1, 2, 2]],
      unsignedArg: { repeats: 2, axis: 0 },
      intDtype: "int32",
      boolDtype: "bool",
      emptyShape: [2, 0],
      emptyGradient: [[0, 0], [0, 0]],
    });
  });

  it("provides typed VJP, batch-safe vmap, exact ONNX decomposition, and device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      gradientOps: string[];
      mapped: number[][];
      mappedArg: { repeats: number; axis: number };
      onnxOps: string[];
      onnxInitializers: Record<string, string[]>;
      onnxSupportedDtypes: Record<string, boolean>;
      onnxHasCustom: boolean;
      planError: string;
      webgpuSupported: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
import struct
from browsergrad_jit._ir import OP_REPEAT_INTERLEAVE, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
weights = bg.from_numpy(np.arange(1, 9, dtype=np.float32).reshape(2, 4))
gradient_tensor = bg.func.grad(
    lambda value: (value.repeat_interleave(2, dim=1) * weights).sum()
)(source)
mapped_tensor = bg.func.vmap(lambda row: row.repeat_interleave(2, dim=0))(source)
onnx = bg.onnx.export_inference(source.repeat_interleave(2, dim=1), input_buffers=(source,))

onnx_supported_dtypes = {}
for dtype in (np.float32, np.int32, np.int64, np.bool_):
    export_source = bg.from_numpy(np.array([[1, 0]], dtype=dtype))
    payload = bg.onnx.export_inference(
        export_source.repeat_interleave(2, dim=1),
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
onnx_ops = []
initializers = {}
for number, wire, payload in fields(graph):
    if number == 1 and wire == 2:
        node_fields = list(fields(payload))
        onnx_ops.append(next(
            value.decode("utf-8") for field, kind, value in node_fields
            if field == 4 and kind == 2
        ))
    elif number == 5 and wire == 2:
        tensor_fields = list(fields(payload))
        name = next(
            value.decode("utf-8") for field, kind, value in tensor_fields
            if field == 8 and kind == 2
        )
        if name.startswith("const_repeat_interleave_"):
            dtype = next(value for field, kind, value in tensor_fields if field == 2 and kind == 0)
            raw = next(value for field, kind, value in tensor_fields if field == 9 and kind == 2)
            if dtype != 7 or len(raw) % 8 != 0:
                raise RuntimeError("repeat_interleave initializer is not an int64 vector")
            initializers[name] = [
                str(value) for value in struct.unpack("<" + "q" * (len(raw) // 8), raw)
            ]

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "registered": get_rule(OP_REPEAT_INTERLEAVE) is not None,
    "gradient": gradient_tensor.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient_tensor._uop)}),
    "mapped": mapped_tensor.numpy().tolist(),
    "mappedArg": mapped_tensor._uop.arg,
    "onnxOps": onnx_ops,
    "onnxInitializers": initializers,
    "onnxSupportedDtypes": onnx_supported_dtypes,
    "onnxHasCustom": b"CUSTOM" in onnx,
    "planError": error(lambda: bg.gpu_plan_summary(source.repeat_interleave(2, dim=1))),
    "webgpuSupported": OP_REPEAT_INTERLEAVE in supported_opcodes(),
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual([[3, 7], [11, 15]]);
    expect(result.gradientOps).toContain("REDUCE");
    expect(result.gradientOps).toContain("RESHAPE");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual([[1, 1, 2, 2], [3, 3, 4, 4]]);
    expect(result.mappedArg).toEqual({ repeats: 2, axis: 1 });
    expect(result.onnxOps).toEqual(["Unsqueeze", "Tile", "Reshape", "Identity"]);
    expect(result.onnxInitializers).toEqual({
      const_repeat_interleave_axes_0: ["2"],
      const_repeat_interleave_repeats_0: ["1", "1", "2"],
      const_repeat_interleave_shape_0: ["2", "4"],
    });
    expect(result.onnxSupportedDtypes).toEqual({
      float32: true,
      int32: true,
      int64: true,
      bool: true,
    });
    expect(result.onnxHasCustom).toBe(false);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*REPEAT_INTERLEAVE/u);
    expect(result.webgpuSupported).toBe(false);
  });

  it("rejects coercion, resource abuse, malformed contracts, and boundary mutation", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_REPEAT_INTERLEAVE
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(4, dtype=np.float32).reshape(2, 2))
empty = bg.from_numpy(np.empty((0, 1), dtype=np.float32))
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
        return 1
    def __int__(self):
        HostileScalar.calls += 1
        return 1

mutated_cpu = source.repeat_interleave(2, dim=1)
mutated_cpu._uop.arg["repeats"] = -1
mutated_vjp = source.repeat_interleave(2, dim=1)
mutated_vjp._uop.arg["axis"] = 2
mutated_vmap = source.repeat_interleave(2, dim=1)
mutated_vmap._uop.arg["repeats"] = -1
mutated_onnx = source.repeat_interleave(2, dim=1)
mutated_onnx._uop.arg["extra"] = 1
dy = bg.from_numpy(np.ones((2, 4), dtype=np.float32))._uop

errors = {
    "boolRepeat": error(lambda: source.repeat_interleave(True, dim=1)),
    "floatRepeat": error(lambda: source.repeat_interleave(2.0, dim=1)),
    "hostileRepeat": error(lambda: source.repeat_interleave(HostileScalar(), dim=1)),
    "negativeRepeat": error(lambda: source.repeat_interleave(-1, dim=1)),
    "repeatCeiling": error(lambda: empty.repeat_interleave(1073741825, dim=0)),
    "boolAxis": error(lambda: source.repeat_interleave(2, dim=True)),
    "floatAxis": error(lambda: source.repeat_interleave(2, dim=1.0)),
    "hostileAxis": error(lambda: source.repeat_interleave(2, dim=HostileScalar())),
    "axisRange": error(lambda: source.repeat_interleave(2, dim=2)),
    "scalarAxis": error(lambda: bg.tensor(1.0).repeat_interleave(2, dim=0)),
    "wrongShape": error(lambda: UOp(
        OP_REPEAT_INTERLEAVE,
        (source._uop,),
        (2, 3),
        "float32",
        arg={"repeats": 2, "axis": 1},
    )),
    "wrongDtype": error(lambda: UOp(
        OP_REPEAT_INTERLEAVE,
        (source._uop,),
        (2, 4),
        "float64",
        arg={"repeats": 2, "axis": 1},
    )),
    "openArgCpu": error(lambda: mutated_cpu.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_REPEAT_INTERLEAVE)(
        mutated_vjp._uop,
        mutated_vjp._uop.inputs,
        dy,
    )),
    "openArgVmap": error(lambda: get_vmap_rule(OP_REPEAT_INTERLEAVE)(
        mutated_vmap._uop,
        {id(mutated_vmap._uop.inputs[0]): mutated_vmap._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated_onnx, input_buffers=(source,))),
    "onnxFloat64": error(lambda: bg.onnx.export_inference(
        float64_source.repeat_interleave(2, dim=1),
        input_buffers=(float64_source,),
    )),
}
{"errors": errors, "hostileCalls": HostileScalar.calls}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.boolRepeat).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.floatRepeat).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.hostileRepeat).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.negativeRepeat).toMatch(/^ShapeError: .*must be in/u);
    expect(result.errors.repeatCeiling).toMatch(/^ShapeError: .*must be in/u);
    expect(result.errors.boolAxis).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.floatAxis).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.hostileAxis).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.axisRange).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.scalarAxis).toMatch(/^ShapeError: .*rank 0/u);
    expect(result.errors.wrongShape).toMatch(/^ShapeError: .*derived shape/u);
    expect(result.errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(result.errors.openArgCpu).toMatch(/^RealizationError: .*must be in/u);
    expect(result.errors.openArgVjp).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.openArgVmap).toMatch(/^ShapeError: .*must be in/u);
    expect(result.errors.openArgOnnx).toMatch(/^ShapeError: .*fields must be exactly/u);
    expect(result.errors.onnxFloat64).toMatch(/^OnnxUnmappableOp: .*dtype/u);
  });

  it("matches the shared eager/lazy repeat_interleave fixture", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE);
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
output = bg.from_numpy(base).repeat_interleave(valid["repeats"], dim=valid["dim"]).numpy()

dtypes = []
for case in fixture["dtypeCases"]:
    source = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(
        bg.from_numpy(source)
        .repeat_interleave(valid["repeats"], dim=valid["dim"])
        .dtype
    )

errors = {}
for case in fixture["invalid"]:
    source = np.arange(np.prod(case["inputShape"]), dtype=np.float32).reshape(case["inputShape"])
    try:
        bg.from_numpy(source).repeat_interleave(case["repeats"], dim=case["dim"])
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

    expect(result.schema).toBe(FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.valid.outputValues);
    expect(result.dtypes).toEqual(
      FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const fixture of FRAMEWORK_REPEAT_INTERLEAVE_CONFORMANCE.invalid) {
      expect(result.errors[fixture.id]).toContain(fixture.message);
    }
  });
});
