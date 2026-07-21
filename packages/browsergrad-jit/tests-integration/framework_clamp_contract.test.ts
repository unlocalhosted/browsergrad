import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed clamp framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("preserves floating dtypes, normalized bounds, owning results, aliases, and closure gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      arg: { min: number; max: number };
      values: number[];
      ownsData: boolean;
      gradient: number[];
      minOnlyOp: string;
      minOnlyValues: number[];
      maxOnlyOp: string;
      maxOnlyValues: number[];
      dtypePairs: string[][];
    }>(`
import browsergrad_jit as bg
import numpy as np

source = bg.from_numpy(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32), requires_grad=True)
clamped = source.clamp(np.int32(-1), np.float64(1))
array = clamped.numpy()
clamped.sum().backward()

min_only = source.clamp_min(0)
max_only = source.clip(max=0)

dtype_pairs = []
for dtype in (np.float16, np.float32, np.float64):
    typed = bg.from_numpy(np.array([-2.0, 0.0, 2.0], dtype=dtype)).clamp(-1, 1)
    dtype_pairs.append([typed.dtype, str(typed.numpy().dtype)])

{
    "op": clamped._uop.op,
    "arg": clamped._uop.arg,
    "values": array.tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "gradient": source.grad.numpy().tolist(),
    "minOnlyOp": min_only._uop.op,
    "minOnlyValues": min_only.numpy().tolist(),
    "maxOnlyOp": max_only._uop.op,
    "maxOnlyValues": max_only.numpy().tolist(),
    "dtypePairs": dtype_pairs,
}
`);

    expect(result).toEqual({
      op: "CLAMP",
      arg: { min: -1, max: 1 },
      values: [-1, -1, 0, 1, 1],
      ownsData: true,
      gradient: [0, 1, 1, 1, 0],
      minOnlyOp: "CLAMP",
      minOnlyValues: [0, 0, 0, 1, 2],
      maxOnlyOp: "CLAMP",
      maxOnlyValues: [-2, -1, 0, 0, 0],
      dtypePairs: [
        ["float16", "float16"],
        ["float32", "float32"],
        ["float64", "float64"],
      ],
    });
  });

  it("provides typed symbolic VJP, functional-grad, vmap, ONNX Clip, and explicit plan refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[];
      gradientOps: string[];
      mapped: number[][];
      onnxHasClip: boolean;
      onnxHasMin: boolean;
      onnxHasMax: boolean;
      onnxHasCustom: boolean;
      clipInputs: string[][];
      initializerValues: number[][];
      planError: string;
    }>(`
import browsergrad_jit as bg
import numpy as np
import struct
from browsergrad_jit._ir import OP_CLAMP, toposort
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
gradient_tensor = bg.func.grad(lambda value: value.clamp(-1, 1).sum())(source)

batched = bg.from_numpy(np.array([[-2.0, 0.0], [0.5, 2.0]], dtype=np.float32))
mapped = bg.func.vmap(lambda row: row.clamp(-0.5, 0.75))(batched).numpy()

both_onnx = bg.onnx.export_inference(source.clamp(-1, 1), input_buffers=(source,))
min_onnx = bg.onnx.export_inference(source.clamp(min=-1), input_buffers=(source,))
max_onnx = bg.onnx.export_inference(source.clamp(max=1), input_buffers=(source,))

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

def clip_signature(model):
    graph = next(payload for number, wire, payload in fields(model) if number == 7 and wire == 2)
    clip_inputs = None
    initializer_values = []
    for number, wire, payload in fields(graph):
        if number == 1 and wire == 2:
            node_fields = list(fields(payload))
            op_type = next(
                value.decode("utf-8") for field, kind, value in node_fields
                if field == 4 and kind == 2
            )
            if op_type == "Clip":
                clip_inputs = [
                    value.decode("utf-8") for field, kind, value in node_fields
                    if field == 1 and kind == 2
                ]
        elif number == 5 and wire == 2:
            tensor_fields = list(fields(payload))
            name = next(
                value.decode("utf-8") for field, kind, value in tensor_fields
                if field == 8 and kind == 2
            )
            if name.startswith("const_clamp_"):
                dtype = next(value for field, kind, value in tensor_fields if field == 2 and kind == 0)
                raw = next(value for field, kind, value in tensor_fields if field == 9 and kind == 2)
                if dtype != 1 or len(raw) != 4:
                    raise RuntimeError("CLAMP initializer is not one float32 scalar")
                initializer_values.append(struct.unpack("<f", raw)[0])
    if clip_inputs is None:
        raise RuntimeError("missing Clip node")
    return clip_inputs, initializer_values

signatures = [clip_signature(model) for model in (both_onnx, min_onnx, max_onnx)]

try:
    bg.gpu_plan_summary(source.clamp(-1, 1))
    plan_error = "no_error"
except Exception as exc:
    plan_error = type(exc).__name__ + ": " + str(exc)

{
    "registered": get_rule(OP_CLAMP) is not None,
    "gradient": gradient_tensor.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient_tensor._uop)}),
    "mapped": mapped.tolist(),
    "onnxHasClip": all(b"Clip" in model for model in (both_onnx, min_onnx, max_onnx)),
    "onnxHasMin": b"const_clamp_min" in both_onnx and b"const_clamp_min" in min_onnx,
    "onnxHasMax": b"const_clamp_max" in both_onnx and b"const_clamp_max" in max_onnx,
    "onnxHasCustom": any(b"CUSTOM" in model for model in (both_onnx, min_onnx, max_onnx)),
    "clipInputs": [signature[0] for signature in signatures],
    "initializerValues": [signature[1] for signature in signatures],
    "planError": plan_error,
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual([0, 1, 1, 1, 0]);
    expect(result.gradientOps).toEqual(expect.arrayContaining(["CAST", "CMP", "MUL"]));
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual([[-0.5, 0], [0.5, 0.75]]);
    expect(result.onnxHasClip).toBe(true);
    expect(result.onnxHasMin).toBe(true);
    expect(result.onnxHasMax).toBe(true);
    expect(result.onnxHasCustom).toBe(false);
    expect(result.clipInputs[0]?.slice(1)).toEqual(["const_clamp_min_0", "const_clamp_max_0"]);
    expect(result.clipInputs[1]?.slice(1)).toEqual(["const_clamp_min_0"]);
    expect(result.clipInputs[2]?.slice(1)).toEqual(["", "const_clamp_max_0"]);
    expect(result.initializerValues).toEqual([[-1, 1], [-1], [1]]);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*CLAMP/u);
  });

  it("rejects hostile bounds, dtype drift, invalid contracts, and mutation at consuming boundaries", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_CLAMP
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.array([-2.0, 0.0, 2.0], dtype=np.float32))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

class HostileFloat:
    calls = 0
    def __float__(self):
        HostileFloat.calls += 1
        return 0.0

mutated = source.clamp(-1, 1)
mutated._uop.arg["min"] = 0
dy = bg.from_numpy(np.ones(source.shape, dtype=np.float32))._uop

errors = {
    "missing": error(lambda: source.clamp()),
    "bool": error(lambda: source.clamp(True, 1)),
    "string": error(lambda: source.clamp("-1", 1)),
    "complex": error(lambda: source.clamp(complex(-1, 0), 1)),
    "nan": error(lambda: source.clamp(float("nan"), 1)),
    "infinity": error(lambda: source.clamp(-1, float("inf"))),
    "hostile": error(lambda: source.clamp(HostileFloat(), 1)),
    "reversed": error(lambda: source.clamp(2, 1)),
    "integerInput": error(lambda: bg.from_numpy(np.array([1], dtype=np.int32)).clamp(0, 1)),
    "wrongFields": error(lambda: UOp(OP_CLAMP, (source._uop,), source.shape, "float32", arg={"min": -1.0})),
    "wrongShape": error(lambda: UOp(OP_CLAMP, (source._uop,), (1, 3), "float32", arg={"min": -1.0, "max": 1.0})),
    "wrongDtype": error(lambda: UOp(OP_CLAMP, (source._uop,), source.shape, "float64", arg={"min": -1.0, "max": 1.0})),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_CLAMP)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_CLAMP)(
        mutated._uop,
        {id(mutated._uop.inputs[0]): mutated._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(source,))),
}
{"errors": errors, "hostileCalls": HostileFloat.calls}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.missing).toMatch(/^ValueError: .*at least one/u);
    expect(result.errors.bool).toMatch(/^TypeError: .*real scalar/u);
    expect(result.errors.string).toMatch(/^TypeError: .*real scalar/u);
    expect(result.errors.complex).toMatch(/^TypeError: .*real scalar/u);
    expect(result.errors.nan).toMatch(/^ValueError: .*finite/u);
    expect(result.errors.infinity).toMatch(/^ValueError: .*finite/u);
    expect(result.errors.hostile).toMatch(/^TypeError: .*real scalar/u);
    expect(result.errors.reversed).toMatch(/^ValueError: .*must be <=/u);
    expect(result.errors.integerInput).toMatch(/^ShapeError: .*floating dtypes only/u);
    expect(result.errors.wrongFields).toMatch(/^ShapeError: .*arg fields/u);
    expect(result.errors.wrongShape).toMatch(/^ShapeError: .*preserve its input shape/u);
    expect(result.errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(result.errors.openArgCpu).toMatch(/^RealizationError: .*finite float/u);
    expect(result.errors.openArgVjp).toMatch(/^ShapeError: .*finite float/u);
    expect(result.errors.openArgVmap).toMatch(/^ShapeError: .*finite float/u);
    expect(result.errors.openArgOnnx).toMatch(/^ShapeError: .*finite float/u);
  });
});
