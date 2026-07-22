import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_TRIL_CONFORMANCE } from "../../../test-support/framework-tril-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.tril framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("emits typed TRIL with canonical diagonals and owning dtype-preserving CPU semantics", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      diagonal: number;
      upperDiagonal: number[];
      positiveSaturation: number;
      negativeSaturation: number;
      positiveValues: number[];
      negativeValues: number[];
      values: number[];
      gradient: number[];
      ownsData: boolean;
      dtypes: string[];
      emptyShapes: number[][];
    }>(`
import browsergrad_jit as bg
import numpy as np

shape = (2, 2, 3)
values = np.arange(1, 13, dtype=np.float32).reshape(shape)
source = bg.from_numpy(values, requires_grad=True)
weights = bg.from_numpy(np.arange(1, 13, dtype=np.float32).reshape(shape))
output = source.tril(np.int32(-1))
array = output.numpy()
(output * weights).sum().backward()
upper = bg.tril(source, np.uint64(1))
positive = source.tril(10 ** 200)
negative = source.tril(-(10 ** 200))

dtypes = []
for dtype in (np.float16, np.float32, np.float64, np.int32, np.int64, np.bool_):
    typed_values = np.arange(1, 13).reshape(shape).astype(dtype)
    dtypes.append(bg.from_numpy(typed_values).tril(-1).dtype)
empty_shapes = [
    list(bg.from_numpy(np.empty((0, 3), dtype=np.float32)).tril(-1).shape),
    list(bg.from_numpy(np.empty((2, 0), dtype=np.float32)).tril(1).shape),
]

{
    "op": output._uop.op,
    "diagonal": output._uop.arg["diagonal"],
    "upperDiagonal": upper.numpy().reshape(-1).tolist(),
    "positiveSaturation": positive._uop.arg["diagonal"],
    "negativeSaturation": negative._uop.arg["diagonal"],
    "positiveValues": positive.numpy().reshape(-1).tolist(),
    "negativeValues": negative.numpy().reshape(-1).tolist(),
    "values": array.reshape(-1).tolist(),
    "gradient": source.grad.numpy().reshape(-1).tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "dtypes": dtypes,
    "emptyShapes": empty_shapes,
}
`);

    expect(result.op).toBe("TRIL");
    expect(result.diagonal).toBe(-1);
    expect(result.values).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.outputValues);
    expect(result.gradient).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.sourceGradient);
    expect(result.upperDiagonal).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.upperDiagonalValues);
    expect(result.positiveSaturation).toBe(2);
    expect(result.negativeSaturation).toBe(-2);
    expect(result.positiveValues).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.inputValues);
    expect(result.negativeValues).toEqual(new Array(12).fill(0));
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_TRIL_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    expect(result.emptyShapes).toEqual([[0, 3], [2, 0]]);
  });

  it("provides idempotent symbolic VJP, batch-safe vmap, ONNX Trilu, and explicit device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[];
      gradientOps: string[];
      mapped: number[];
      triluInputs: string[];
      diagonalValue: number;
      diagonalRank: number;
      diagonalDtype: number;
      upperAttribute: number;
      halfOnnxError: string;
      planError: string;
      webgpuSupported: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
import struct
from browsergrad_jit._ir import OP_TRIL, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

shape = (2, 2, 3)
source = bg.from_numpy(np.arange(1, 13, dtype=np.float32).reshape(shape))
weights = bg.from_numpy(np.arange(1, 13, dtype=np.float32).reshape(shape))
gradient = bg.func.grad(lambda value: (value.tril(-1) * weights).sum())(source)
mapped = bg.func.vmap(lambda matrix: matrix.tril(-1))(source)
onnx = bg.onnx.export_inference(source.tril(-1), input_buffers=(source,))
half = bg.from_numpy(np.arange(1, 13, dtype=np.float16).reshape(shape))

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

graph = next(payload for number, wire, payload in fields(onnx) if number == 7 and wire == 2)
trilu_inputs = None
upper_attribute = None
initializers = {}
for number, wire, payload in fields(graph):
    if number == 1 and wire == 2:
        node_fields = list(fields(payload))
        op_type = next(
            value.decode("utf-8") for field, kind, value in node_fields
            if field == 4 and kind == 2
        )
        if op_type == "Trilu":
            trilu_inputs = [
                value.decode("utf-8") for field, kind, value in node_fields
                if field == 1 and kind == 2
            ]
            for field, kind, value in node_fields:
                if field != 5 or kind != 2:
                    continue
                attribute_fields = list(fields(value))
                name = next(
                    item.decode("utf-8") for child, child_kind, item in attribute_fields
                    if child == 1 and child_kind == 2
                )
                if name == "upper":
                    upper_attribute = next(
                        item for child, child_kind, item in attribute_fields
                        if child == 3 and child_kind == 0
                    )
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
if trilu_inputs is None or len(trilu_inputs) != 2:
    raise RuntimeError("missing exact Trilu inputs")
diagonal_initializer = initializers[trilu_inputs[1]]

{
    "registered": get_rule(OP_TRIL) is not None,
    "gradient": gradient.numpy().reshape(-1).tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mapped": mapped.numpy().reshape(-1).tolist(),
    "triluInputs": trilu_inputs,
    "diagonalValue": struct.unpack("<q", diagonal_initializer["raw"])[0],
    "diagonalRank": diagonal_initializer["rank"],
    "diagonalDtype": diagonal_initializer["dtype"],
    "upperAttribute": upper_attribute,
    "halfOnnxError": error(lambda: bg.onnx.export_inference(
        half.tril(-1),
        input_buffers=(half,),
    )),
    "planError": error(lambda: bg.gpu_plan_summary(source.tril(-1))),
    "webgpuSupported": OP_TRIL in supported_opcodes(),
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.sourceGradient);
    expect(result.gradientOps).toContain("TRIL");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.outputValues);
    expect(result.triluInputs).toHaveLength(2);
    expect(result.triluInputs[1]).toBe("const_tril_diagonal_0");
    expect(result.diagonalValue).toBe(-1);
    expect(result.diagonalRank).toBe(0);
    expect(result.diagonalDtype).toBe(7);
    expect(result.upperAttribute).toBe(0);
    expect(result.halfOnnxError).toMatch(/^OnnxUnmappableOp: .*float16/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*TRIL/u);
    expect(result.webgpuSupported).toBe(false);
  });

  it("rejects hostile inputs and malformed TRIL at every admitted boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_TRIL
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
float64_source = bg.from_numpy(np.arange(1, 7, dtype=np.float64).reshape(2, 3))
rank_one = bg.from_numpy(np.arange(3, dtype=np.float32))
uint_source = bg.from_numpy(np.arange(6, dtype=np.uint16).reshape(2, 3))

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

valid = source.tril(0)
mutated = source.tril(0)
mutated._uop.arg["diagonal"] = 4
dy = bg.from_numpy(np.ones(source.shape, dtype=np.float32))._uop

errors = {
    "rank": error(lambda: rank_one.tril()),
    "dtype": error(lambda: uint_source.tril()),
    "bool": error(lambda: source.tril(True)),
    "float": error(lambda: source.tril(1.0)),
    "string": error(lambda: source.tril("1")),
    "hostile": error(lambda: source.tril(HostileIndex())),
    "wrongFields": error(lambda: UOp(OP_TRIL, (source._uop,), source.shape, "float32", arg={})),
    "wrongShape": error(lambda: UOp(OP_TRIL, (source._uop,), (3, 2), "float32", arg={"diagonal": 0})),
    "wrongDtype": error(lambda: UOp(OP_TRIL, (source._uop,), source.shape, "float64", arg={"diagonal": 0})),
    "boolArg": error(lambda: UOp(OP_TRIL, (source._uop,), source.shape, "float32", arg={"diagonal": True})),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_TRIL)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_TRIL)(
        mutated._uop,
        {id(mutated._uop.inputs[0]): mutated._uop.inputs[0]},
        1,
    )),
    "capturedVmap": error(lambda: get_vmap_rule(OP_TRIL)(
        valid._uop,
        {id(valid._uop.inputs[0]): valid._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(source,))),
    "openArgPlan": error(lambda: bg.gpu_plan_summary(mutated)),
    "onnxFloat64": error(lambda: bg.onnx.export_inference(
        float64_source.tril(),
        input_buffers=(float64_source,),
    )),
}
{"errors": errors, "hostileCalls": HostileIndex.calls, "validDiagonal": valid._uop.arg["diagonal"]}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.rank).toMatch(/^ShapeError: .*rank at least two/u);
    expect(result.errors.dtype).toMatch(/^ShapeError: .*not supported/u);
    expect(result.errors.bool).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.float).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.string).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.hostile).toMatch(/^ShapeError: .*integer scalar/u);
    expect(result.errors.wrongFields).toMatch(/^ShapeError: .*arg fields/u);
    expect(result.errors.wrongShape).toMatch(/^ShapeError: .*preserve its input shape/u);
    expect(result.errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(result.errors.boolArg).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.errors.openArgCpu).toMatch(/^RealizationError: .*canonical range/u);
    expect(result.errors.openArgVjp).toMatch(/^ShapeError: .*canonical range/u);
    expect(result.errors.openArgVmap).toMatch(/^ShapeError: .*canonical range/u);
    expect(result.errors.capturedVmap).toMatch(/^JitNotImplementedError: .*leading mapped axis/u);
    expect(result.errors.openArgOnnx).toMatch(/^ShapeError: .*canonical range/u);
    expect(result.errors.openArgPlan).toMatch(/^ShapeError: .*canonical range/u);
    expect(result.errors.onnxFloat64).toMatch(/^OnnxUnmappableOp: .*float64/u);
  });

  it("matches the shared eager/lazy triangular-selection fixture", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_TRIL_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      values: number[];
      dtypes: string[];
      errors: Record<string, string>;
    }>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
values = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
output = bg.from_numpy(values).tril(valid["diagonal"])

dtypes = []
for case in fixture["dtypeCases"]:
    dtype = case["dtype"]
    typed = np.asarray(valid["inputValues"], dtype=np.dtype(dtype)).reshape(valid["inputShape"])
    dtypes.append(bg.from_numpy(typed).tril(valid["diagonal"]).dtype)

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "rank":
            bg.from_numpy(np.ones(case["value"], dtype=np.float32)).tril()
        elif case["kind"] == "diagonal":
            bg.from_numpy(values).tril(case["value"])
        elif case["kind"] == "dtype":
            bad = np.asarray(valid["inputValues"], dtype=np.dtype(case["value"])).reshape(valid["inputShape"])
            bg.from_numpy(bad).tril()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "values": output.numpy().reshape(-1).tolist(),
    "dtypes": dtypes,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_TRIL_CONFORMANCE.schema);
    expect(result.values).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.outputValues);
    expect(result.dtypes).toEqual(
      FRAMEWORK_TRIL_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const invalid of FRAMEWORK_TRIL_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).toContain(invalid.message);
    }
  });
});
