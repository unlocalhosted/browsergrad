import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_PROD_CONFORMANCE } from "../../../test-support/framework-prod-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.prod framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("returns owning dtype-preserving products and zero-aware closure gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      axes: number[];
      keepdims: boolean;
      values: number[];
      ownsData: boolean;
      fullValue: number;
      scalarValue: number;
      multiAxisValue: number;
      intDtype: string;
      boolDtype: string;
      noZeroGradient: number[][];
      halfGradient: number[];
      halfGradientDtype: string;
      oneZeroGradient: number[][];
      twoZeroGradient: number[][];
      fullZeroGradient: number[];
      emptyValues: number[];
      emptyGradientShape: number[];
    }>(`
import browsergrad_jit as bg
import numpy as np

source = bg.from_numpy(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32), requires_grad=True)
weights = bg.from_numpy(np.array([2.0, 3.0], dtype=np.float32))
product = source.prod(dim=1)
array = product.numpy()
(product * weights).sum().backward()
half_source = bg.from_numpy(np.array([2.0, 3.0], dtype=np.float16), requires_grad=True)
half_source.prod().backward()

one_zero_source = bg.from_numpy(np.array([[0.0, 3.0], [4.0, 5.0]], dtype=np.float32), requires_grad=True)
one_zero_source.prod(dim=1).sum().backward()
two_zero_source = bg.from_numpy(np.array([[0.0, 0.0], [4.0, 5.0]], dtype=np.float32), requires_grad=True)
two_zero_source.prod(dim=1).sum().backward()
full_zero_source = bg.from_numpy(np.array([2.0, 0.0, 4.0], dtype=np.float32), requires_grad=True)
full_zero_source.prod().backward()

empty_source = bg.from_numpy(np.empty((2, 0), dtype=np.float32), requires_grad=True)
empty_product = empty_source.prod(dim=1)
empty_product.sum().backward()

integer = bg.from_numpy(np.array([[2, 3], [4, 5]], dtype=np.int32)).prod(dim=1)
boolean = bg.from_numpy(np.array([[True, False], [True, True]], dtype=np.bool_)).prod(dim=1)

{
    "op": product._uop.op,
    "axes": list(product._uop.arg["axes"]),
    "keepdims": product._uop.arg["keepdims"],
    "values": array.tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "fullValue": float(source.prod().item()),
    "scalarValue": float(bg.tensor(7.0).prod().item()),
    "multiAxisValue": float(source.prod(dim=(0, 1)).item()),
    "intDtype": integer.dtype,
    "boolDtype": boolean.dtype,
    "noZeroGradient": source.grad.numpy().tolist(),
    "halfGradient": half_source.grad.numpy().tolist(),
    "halfGradientDtype": half_source.grad.dtype,
    "oneZeroGradient": one_zero_source.grad.numpy().tolist(),
    "twoZeroGradient": two_zero_source.grad.numpy().tolist(),
    "fullZeroGradient": full_zero_source.grad.numpy().tolist(),
    "emptyValues": empty_product.numpy().tolist(),
    "emptyGradientShape": list(empty_source.grad.shape),
}
`);

    expect(result).toEqual({
      op: "PROD",
      axes: [1],
      keepdims: false,
      values: [6, 20],
      ownsData: true,
      fullValue: 120,
      scalarValue: 7,
      multiAxisValue: 120,
      intDtype: "int32",
      boolDtype: "bool",
      noZeroGradient: [[6, 4], [15, 12]],
      halfGradient: [3, 2],
      halfGradientDtype: "float16",
      oneZeroGradient: [[3, 0], [5, 4]],
      twoZeroGradient: [[0, 0], [5, 4]],
      fullZeroGradient: [0, 8, 0],
      emptyValues: [1, 1],
      emptyGradientShape: [2, 0],
    });
  });

  it("provides zero-aware typed VJP, batch-safe vmap, exact ONNX, and device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      gradientOps: string[];
      halfGradient: number[];
      halfGradientDtype: string;
      mapped: number[];
      mappedKeepdims: number[][];
      mappedAxes: number[];
      onnxOps: string[];
      reduceAxes: string[];
      reduceKeepdims: number;
      onnxSupportedDtypes: Record<string, boolean>;
      onnxBoolError: string;
      onnxFloat16Error: string;
      planError: string;
      webgpuSupported: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_PROD, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([[0.0, 3.0], [4.0, 5.0]], dtype=np.float32))
gradient = bg.func.grad(lambda value: value.prod(dim=1).sum())(source)
half_source = bg.from_numpy(np.array([2.0, 3.0], dtype=np.float16))
half_gradient = bg.func.grad(lambda value: value.prod())(half_source)
mapped_tensor = bg.func.vmap(lambda row: row.prod(dim=0))(source)
mapped_keepdims = bg.func.vmap(lambda row: row.prod(dim=0, keepdim=True))(source)
onnx = bg.onnx.export_inference(source.prod(dim=1), input_buffers=(source,))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

onnx_supported_dtypes = {}
for dtype in (np.float32, np.int32, np.int64):
    export_source = bg.from_numpy(np.array([[2, 3]], dtype=dtype))
    payload = bg.onnx.export_inference(export_source.prod(dim=1), input_buffers=(export_source,))
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
        wire = tag & 7
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

def packed_varints(data):
    values = []
    index = 0
    while index < len(data):
        value = 0
        shift = 0
        while True:
            byte = data[index]
            index += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        values.append(value)
    return values

graph = next(payload for number, wire, payload in fields(onnx) if number == 7 and wire == 2)
onnx_ops = []
reduce_axes = None
reduce_keepdims = None
for number, wire, payload in fields(graph):
    if number != 1 or wire != 2:
        continue
    node_fields = list(fields(payload))
    op_type = next(
        value.decode("utf-8") for field, kind, value in node_fields
        if field == 4 and kind == 2
    )
    onnx_ops.append(op_type)
    if op_type == "ReduceProd":
        for field, kind, attr_payload in node_fields:
            if field != 5 or kind != 2:
                continue
            attr_fields = list(fields(attr_payload))
            name = next(
                value.decode("utf-8") for attr_field, attr_kind, value in attr_fields
                if attr_field == 1 and attr_kind == 2
            )
            if name == "axes":
                raw = next(
                    value for attr_field, attr_kind, value in attr_fields
                    if attr_field == 8 and attr_kind == 2
                )
                reduce_axes = [str(value) for value in packed_varints(raw)]
            elif name == "keepdims":
                reduce_keepdims = next(
                    value for attr_field, attr_kind, value in attr_fields
                    if attr_field == 3 and attr_kind == 0
                )

bool_source = bg.from_numpy(np.array([[True, False]], dtype=np.bool_))
half_source = bg.from_numpy(np.array([[2.0, 3.0]], dtype=np.float16))

{
    "registered": get_rule(OP_PROD) is not None,
    "gradient": gradient.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "halfGradient": half_gradient.numpy().tolist(),
    "halfGradientDtype": half_gradient.dtype,
    "mapped": mapped_tensor.numpy().tolist(),
    "mappedKeepdims": mapped_keepdims.numpy().tolist(),
    "mappedAxes": list(mapped_tensor._uop.arg["axes"]),
    "onnxOps": onnx_ops,
    "reduceAxes": reduce_axes,
    "reduceKeepdims": reduce_keepdims,
    "onnxSupportedDtypes": onnx_supported_dtypes,
    "onnxBoolError": error(lambda: bg.onnx.export_inference(bool_source.prod(dim=1), input_buffers=(bool_source,))),
    "onnxFloat16Error": error(lambda: bg.onnx.export_inference(half_source.prod(dim=1), input_buffers=(half_source,))),
    "planError": error(lambda: bg.gpu_plan_summary(source.prod(dim=1))),
    "webgpuSupported": OP_PROD in supported_opcodes(),
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual([[3, 0], [5, 4]]);
    expect(result.gradientOps).toContain("PROD");
    expect(result.gradientOps).toContain("WHERE");
    expect(result.gradientOps).toContain("REDUCE");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.halfGradient).toEqual([3, 2]);
    expect(result.halfGradientDtype).toBe("float16");
    expect(result.mapped).toEqual([0, 20]);
    expect(result.mappedKeepdims).toEqual([[0], [20]]);
    expect(result.mappedAxes).toEqual([1]);
    expect(result.onnxOps).toEqual(["ReduceProd", "Identity"]);
    expect(result.reduceAxes).toEqual(["1"]);
    expect(result.reduceKeepdims).toBe(0);
    expect(result.onnxSupportedDtypes).toEqual({ float32: true, int32: true, int64: true });
    expect(result.onnxBoolError).toMatch(/^OnnxUnmappableOp: .*dtype/u);
    expect(result.onnxFloat16Error).toMatch(/^OnnxUnmappableOp: .*dtype/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*PROD/u);
    expect(result.webgpuSupported).toBe(false);
  });

  it("rejects coercion, ambiguous axes, malformed contracts, and boundary mutation", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_PROD
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(1, 5, dtype=np.float32).reshape(2, 2))
dy = bg.from_numpy(np.ones((2,), dtype=np.float32))._uop

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

class HostileAxis:
    calls = 0
    def __index__(self):
        HostileAxis.calls += 1
        return 1
    def __int__(self):
        HostileAxis.calls += 1
        return 1
    def __iter__(self):
        HostileAxis.calls += 1
        return iter((1,))

class HostileList(list):
    def __iter__(self):
        HostileAxis.calls += 1
        return super().__iter__()

mutated_cpu = source.prod(dim=1)
mutated_cpu._uop.arg["axes"] = (1, 1)
mutated_vjp = source.prod(dim=1)
mutated_vjp._uop.arg["keepdims"] = 1
mutated_vmap = source.prod(dim=1)
mutated_vmap._uop.arg["axes"] = (2,)
mutated_onnx = source.prod(dim=1)
mutated_onnx._uop.arg["extra"] = True

errors = {
    "boolAxis": error(lambda: source.prod(dim=True)),
    "floatAxis": error(lambda: source.prod(dim=1.0)),
    "hostileAxis": error(lambda: source.prod(dim=HostileAxis())),
    "hostileList": error(lambda: source.prod(dim=HostileList([1]))),
    "emptyAxes": error(lambda: source.prod(dim=())),
    "duplicateAxes": error(lambda: source.prod(dim=(0, -2))),
    "axisRange": error(lambda: source.prod(dim=2)),
    "bothAliases": error(lambda: source.prod(dim=0, axis=1)),
    "keepdims": error(lambda: source.prod(dim=0, keepdim=1)),
    "scalarAxis": error(lambda: bg.tensor(2.0).prod(dim=0)),
    "listArg": error(lambda: UOp(OP_PROD, (source._uop,), (2,), "float32", arg={"axes": [1], "keepdims": False})),
    "wrongShape": error(lambda: UOp(OP_PROD, (source._uop,), (2, 1), "float32", arg={"axes": (1,), "keepdims": False})),
    "wrongDtype": error(lambda: UOp(OP_PROD, (source._uop,), (2,), "float64", arg={"axes": (1,), "keepdims": False})),
    "openArgCpu": error(lambda: mutated_cpu.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_PROD)(mutated_vjp._uop, mutated_vjp._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_PROD)(
        mutated_vmap._uop,
        {id(mutated_vmap._uop.inputs[0]): mutated_vmap._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated_onnx, input_buffers=(source,))),
}
{"errors": errors, "hostileCalls": HostileAxis.calls}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.boolAxis).toMatch(/^ShapeError: .*axis must be/u);
    expect(result.errors.floatAxis).toMatch(/^ShapeError: .*axis must be/u);
    expect(result.errors.hostileAxis).toMatch(/^ShapeError: .*axis must be/u);
    expect(result.errors.hostileList).toMatch(/^ShapeError: .*axis must be/u);
    expect(result.errors.emptyAxes).toMatch(/^ShapeError: .*non-empty/u);
    expect(result.errors.duplicateAxes).toMatch(/^ShapeError: .*unique/u);
    expect(result.errors.axisRange).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.bothAliases).toMatch(/^ShapeError: .*only one/u);
    expect(result.errors.keepdims).toMatch(/^ShapeError: .*booleans/u);
    expect(result.errors.scalarAxis).toMatch(/^ShapeError: .*rank 0/u);
    expect(result.errors.listArg).toMatch(/^ShapeError: .*canonical tuple/u);
    expect(result.errors.wrongShape).toMatch(/^ShapeError: .*derived shape/u);
    expect(result.errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(result.errors.openArgCpu).toMatch(/^RealizationError: .*unique/u);
    expect(result.errors.openArgVjp).toMatch(/^ShapeError: .*boolean/u);
    expect(result.errors.openArgVmap).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.openArgOnnx).toMatch(/^ShapeError: .*fields must be exactly/u);
  });

  it("matches the shared eager/lazy product fixture", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_PROD_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[];
      keepdimsShape: number[];
      keepdimsValues: number[][];
      fullValue: number;
      dtypes: string[];
      errors: Record<string, string>;
    }>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
base = np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"])
source = bg.from_numpy(base)
output = source.prod(dim=valid["axis"])
keepdims = source.prod(dim=valid["axis"], keepdim=True)

dtypes = []
for case in fixture["dtypeCases"]:
    typed = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(bg.from_numpy(typed).prod(dim=valid["axis"]).dtype)

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "both":
            source.prod(dim=0, axis=1)
        elif case["kind"] == "keepdims":
            source.prod(dim=0, keepdim=case["value"])
        elif case["kind"] == "scalar":
            bg.tensor(2.0).prod(dim=case["value"])
        else:
            source.prod(dim=case["value"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.numpy().tolist(),
    "keepdimsShape": list(keepdims.shape),
    "keepdimsValues": keepdims.numpy().tolist(),
    "fullValue": float(source.prod().item()),
    "dtypes": dtypes,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_PROD_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_PROD_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_PROD_CONFORMANCE.valid.outputValues);
    expect(result.keepdimsShape).toEqual(FRAMEWORK_PROD_CONFORMANCE.valid.keepdimsShape);
    expect(result.keepdimsValues).toEqual(FRAMEWORK_PROD_CONFORMANCE.valid.keepdimsValues);
    expect(result.fullValue).toBe(FRAMEWORK_PROD_CONFORMANCE.valid.fullValue);
    expect(result.dtypes).toEqual(
      FRAMEWORK_PROD_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const fixture of FRAMEWORK_PROD_CONFORMANCE.invalid) {
      expect(result.errors[fixture.id]).toContain(fixture.message);
    }
  });
});
