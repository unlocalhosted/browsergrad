import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import {
  FRAMEWORK_TRIL_CONFORMANCE,
  FRAMEWORK_TRIU_CONFORMANCE,
} from "../../../test-support/framework-triangular-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

const TRIANGULAR_FIXTURES = Object.freeze({
  tril: FRAMEWORK_TRIL_CONFORMANCE,
  triu: FRAMEWORK_TRIU_CONFORMANCE,
});

describe("Gate 6 typed triangular-selection framework contracts", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("emits typed TRIL/TRIU with canonical diagonals and owning dtype-preserving CPU semantics", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(TRIANGULAR_FIXTURES);
    const result = await target.run<Record<string, {
      op: string;
      diagonal: number;
      alternateValues: number[];
      positiveSaturation: number;
      negativeSaturation: number;
      positiveValues: number[];
      negativeValues: number[];
      values: number[];
      gradient: number[];
      ownsData: boolean;
      dtypes: string[];
      emptyShapes: number[][];
    }>>(`
import browsergrad_jit as bg
import json
import numpy as np

fixtures = json.loads(${JSON.stringify(fixtureJson)})
results = {}
for operation, fixture in fixtures.items():
    valid = fixture["valid"]
    shape = tuple(valid["inputShape"])
    values = np.asarray(valid["inputValues"], dtype=np.float32).reshape(shape)
    source = bg.from_numpy(values, requires_grad=True)
    weights = bg.from_numpy(np.arange(1, 13, dtype=np.float32).reshape(shape))
    output = getattr(source, operation)(np.int32(valid["diagonal"]))
    array = output.numpy()
    (output * weights).sum().backward()
    alternate_diagonal = 1 if operation == "tril" else 0
    alternate = getattr(bg, operation)(source, np.uint64(alternate_diagonal))
    positive = getattr(source, operation)(10 ** 200)
    negative = getattr(source, operation)(-(10 ** 200))

    dtypes = []
    for case in fixture["dtypeCases"]:
        typed_values = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(shape)
        dtypes.append(getattr(bg.from_numpy(typed_values), operation)(valid["diagonal"]).dtype)
    empty_shapes = [
        list(getattr(bg.from_numpy(np.empty((0, 3), dtype=np.float32)), operation)(-1).shape),
        list(getattr(bg.from_numpy(np.empty((2, 0), dtype=np.float32)), operation)(1).shape),
    ]

    results[operation] = {
        "op": output._uop.op,
        "diagonal": output._uop.arg["diagonal"],
        "alternateValues": alternate.numpy().reshape(-1).tolist(),
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
results
`);

    for (const [operation, fixture] of Object.entries(TRIANGULAR_FIXTURES)) {
      const observed = result[operation];
      expect(observed).toBeDefined();
      expect(observed?.op).toBe(operation.toUpperCase());
      expect(observed?.diagonal).toBe(fixture.valid.diagonal);
      expect(observed?.values).toEqual(fixture.valid.outputValues);
      expect(observed?.gradient).toEqual(fixture.valid.sourceGradient);
      expect(observed?.ownsData).toBe(true);
      expect(observed?.dtypes).toEqual(
        fixture.dtypeCases.map(({ expectedDtype }) => expectedDtype),
      );
      expect(observed?.emptyShapes).toEqual([[0, 3], [2, 0]]);
    }
    expect(result.tril?.alternateValues).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.upperDiagonalValues);
    expect(result.tril?.positiveSaturation).toBe(2);
    expect(result.tril?.negativeSaturation).toBe(-2);
    expect(result.tril?.positiveValues).toEqual(FRAMEWORK_TRIL_CONFORMANCE.valid.inputValues);
    expect(result.tril?.negativeValues).toEqual(new Array(12).fill(0));
    expect(result.triu?.alternateValues).toEqual(FRAMEWORK_TRIU_CONFORMANCE.valid.mainDiagonalValues);
    expect(result.triu?.positiveSaturation).toBe(3);
    expect(result.triu?.negativeSaturation).toBe(-1);
    expect(result.triu?.positiveValues).toEqual(new Array(12).fill(0));
    expect(result.triu?.negativeValues).toEqual(FRAMEWORK_TRIU_CONFORMANCE.valid.inputValues);
  });

  it("provides idempotent symbolic VJPs, batch-safe vmap, exact ONNX Trilu, and explicit device refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(TRIANGULAR_FIXTURES);
    const result = await target.run<Record<string, {
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
    }>>(`
import browsergrad_jit as bg
import json
import numpy as np
import struct
from browsergrad_jit._ir import OP_TRIL, OP_TRIU, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixtures = json.loads(${JSON.stringify(fixtureJson)})

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

def parse_trilu(model):
    graph = next(payload for number, wire, payload in fields(model) if number == 7 and wire == 2)
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
    diagonal = initializers[trilu_inputs[1]]
    return trilu_inputs, upper_attribute, diagonal

results = {}
opcodes = {"tril": OP_TRIL, "triu": OP_TRIU}
for operation, fixture in fixtures.items():
    valid = fixture["valid"]
    shape = tuple(valid["inputShape"])
    source = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32).reshape(shape))
    weights = bg.from_numpy(np.arange(1, 13, dtype=np.float32).reshape(shape))
    diagonal = valid["diagonal"]
    gradient = bg.func.grad(
        lambda value, op=operation, k=diagonal: (getattr(value, op)(k) * weights).sum()
    )(source)
    mapped = bg.func.vmap(
        lambda matrix, op=operation, k=diagonal: getattr(matrix, op)(k)
    )(source)
    output = getattr(source, operation)(diagonal)
    model = bg.onnx.export_inference(output, input_buffers=(source,))
    trilu_inputs, upper_attribute, diagonal_initializer = parse_trilu(model)
    half = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float16).reshape(shape))
    opcode = opcodes[operation]
    results[operation] = {
        "registered": get_rule(opcode) is not None,
        "gradient": gradient.numpy().reshape(-1).tolist(),
        "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
        "mapped": mapped.numpy().reshape(-1).tolist(),
        "triluInputs": trilu_inputs,
        "diagonalValue": struct.unpack("<q", diagonal_initializer["raw"])[0],
        "diagonalRank": diagonal_initializer["rank"],
        "diagonalDtype": diagonal_initializer["dtype"],
        "upperAttribute": upper_attribute,
        "halfOnnxError": error(lambda op=operation, k=diagonal: bg.onnx.export_inference(
            getattr(half, op)(k),
            input_buffers=(half,),
        )),
        "planError": error(lambda value=output: bg.gpu_plan_summary(value)),
        "webgpuSupported": opcode in supported_opcodes(),
    }
results
`);

    for (const [operation, fixture] of Object.entries(TRIANGULAR_FIXTURES)) {
      const observed = result[operation];
      const opcode = operation.toUpperCase();
      expect(observed?.registered).toBe(true);
      expect(observed?.gradient).toEqual(fixture.valid.sourceGradient);
      expect(observed?.gradientOps).toContain(opcode);
      expect(observed?.gradientOps).not.toContain("CUSTOM");
      expect(observed?.mapped).toEqual(fixture.valid.outputValues);
      expect(observed?.triluInputs).toHaveLength(2);
      expect(observed?.triluInputs[1]).toBe(`const_${operation}_diagonal_0`);
      expect(observed?.diagonalValue).toBe(fixture.valid.diagonal);
      expect(observed?.diagonalRank).toBe(0);
      expect(observed?.diagonalDtype).toBe(7);
      expect(observed?.upperAttribute).toBe(operation === "triu" ? 1 : 0);
      expect(observed?.halfOnnxError).toMatch(/^OnnxUnmappableOp: .*float16/u);
      expect(observed?.planError).toMatch(new RegExp(`^GpuPlanUnsupported: .*${opcode}`, "u"));
      expect(observed?.webgpuSupported).toBe(false);
    }
  });

  it("rejects hostile inputs and malformed triangular nodes at every admitted boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      errors: Record<string, Record<string, string>>;
      hostileCalls: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_TRIL, OP_TRIU
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

errors = {}
for operation, opcode in (("tril", OP_TRIL), ("triu", OP_TRIU)):
    method = getattr(source, operation)
    valid = method(0)
    mutated = method(0)
    mutated._uop.arg["diagonal"] = 10
    dy = bg.from_numpy(np.ones(source.shape, dtype=np.float32))._uop
    errors[operation] = {
        "rank": error(lambda op=operation: getattr(rank_one, op)()),
        "dtype": error(lambda op=operation: getattr(uint_source, op)()),
        "bool": error(lambda op=operation: getattr(source, op)(True)),
        "float": error(lambda op=operation: getattr(source, op)(1.0)),
        "string": error(lambda op=operation: getattr(source, op)("1")),
        "hostile": error(lambda op=operation: getattr(source, op)(HostileIndex())),
        "wrongFields": error(lambda code=opcode: UOp(code, (source._uop,), source.shape, "float32", arg={})),
        "wrongShape": error(lambda code=opcode: UOp(code, (source._uop,), (3, 2), "float32", arg={"diagonal": 0})),
        "wrongDtype": error(lambda code=opcode: UOp(code, (source._uop,), source.shape, "float64", arg={"diagonal": 0})),
        "boolArg": error(lambda code=opcode: UOp(code, (source._uop,), source.shape, "float32", arg={"diagonal": True})),
        "openArgCpu": error(lambda value=mutated: value.numpy()),
        "openArgVjp": error(lambda value=mutated, code=opcode: get_rule(code)(value._uop, value._uop.inputs, dy)),
        "openArgVmap": error(lambda value=mutated, code=opcode: get_vmap_rule(code)(
            value._uop,
            {id(value._uop.inputs[0]): value._uop.inputs[0]},
            1,
        )),
        "capturedVmap": error(lambda value=valid, code=opcode: get_vmap_rule(code)(
            value._uop,
            {id(value._uop.inputs[0]): value._uop.inputs[0]},
            1,
        )),
        "openArgOnnx": error(lambda value=mutated: bg.onnx.export_inference(value, input_buffers=(source,))),
        "openArgPlan": error(lambda value=mutated: bg.gpu_plan_summary(value)),
        "onnxFloat64": error(lambda op=operation: bg.onnx.export_inference(
            getattr(float64_source, op)(),
            input_buffers=(float64_source,),
        )),
    }
{"errors": errors, "hostileCalls": HostileIndex.calls}
`);

    expect(result.hostileCalls).toBe(0);
    for (const errors of Object.values(result.errors)) {
      expect(errors.rank).toMatch(/^ShapeError: .*rank at least two/u);
      expect(errors.dtype).toMatch(/^ShapeError: .*not supported/u);
      expect(errors.bool).toMatch(/^ShapeError: .*integer scalar/u);
      expect(errors.float).toMatch(/^ShapeError: .*integer scalar/u);
      expect(errors.string).toMatch(/^ShapeError: .*integer scalar/u);
      expect(errors.hostile).toMatch(/^ShapeError: .*integer scalar/u);
      expect(errors.wrongFields).toMatch(/^ShapeError: .*arg fields/u);
      expect(errors.wrongShape).toMatch(/^ShapeError: .*preserve its input shape/u);
      expect(errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
      expect(errors.boolArg).toMatch(/^ShapeError: .*normalized integer/u);
      expect(errors.openArgCpu).toMatch(/^RealizationError: .*canonical range/u);
      expect(errors.openArgVjp).toMatch(/^ShapeError: .*canonical range/u);
      expect(errors.openArgVmap).toMatch(/^ShapeError: .*canonical range/u);
      expect(errors.capturedVmap).toMatch(/^JitNotImplementedError: .*leading mapped axis/u);
      expect(errors.openArgOnnx).toMatch(/^ShapeError: .*canonical range/u);
      expect(errors.openArgPlan).toMatch(/^ShapeError: .*canonical range/u);
      expect(errors.onnxFloat64).toMatch(/^OnnxUnmappableOp: .*float64/u);
    }
  });

  it("matches the shared eager/lazy triangular-selection fixtures", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(TRIANGULAR_FIXTURES);
    const result = await target.run<Record<string, {
      schema: string;
      values: number[];
      dtypes: string[];
      errors: Record<string, string>;
    }>>(`
import browsergrad_jit as bg
import json
import numpy as np

fixtures = json.loads(${JSON.stringify(fixtureJson)})
results = {}
for operation, fixture in fixtures.items():
    valid = fixture["valid"]
    shape = tuple(valid["inputShape"])
    values = np.asarray(valid["inputValues"], dtype=np.float32).reshape(shape)
    output = getattr(bg.from_numpy(values), operation)(valid["diagonal"])

    dtypes = []
    for case in fixture["dtypeCases"]:
        typed = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(shape)
        dtypes.append(getattr(bg.from_numpy(typed), operation)(valid["diagonal"]).dtype)

    errors = {}
    for case in fixture["invalid"]:
        try:
            if case["kind"] == "rank":
                getattr(bg.from_numpy(np.ones(case["value"], dtype=np.float32)), operation)()
            elif case["kind"] == "diagonal":
                getattr(bg.from_numpy(values), operation)(case["value"])
            elif case["kind"] == "dtype":
                bad = np.asarray(valid["inputValues"], dtype=np.dtype(case["value"])).reshape(shape)
                getattr(bg.from_numpy(bad), operation)()
            errors[case["id"]] = "no_error"
        except Exception as exc:
            errors[case["id"]] = str(exc)

    results[operation] = {
        "schema": fixture["schema"],
        "values": output.numpy().reshape(-1).tolist(),
        "dtypes": dtypes,
        "errors": errors,
    }
results
`);

    for (const [operation, fixture] of Object.entries(TRIANGULAR_FIXTURES)) {
      const observed = result[operation];
      expect(observed?.schema).toBe(fixture.schema);
      expect(observed?.values).toEqual(fixture.valid.outputValues);
      expect(observed?.dtypes).toEqual(
        fixture.dtypeCases.map(({ expectedDtype }) => expectedDtype),
      );
      for (const invalid of fixture.invalid) {
        expect(observed?.errors[invalid.id]).toContain(invalid.message);
      }
    }
  });
});
