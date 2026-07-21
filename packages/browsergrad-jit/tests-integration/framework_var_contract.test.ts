import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_VAR_CONFORMANCE } from "../../../test-support/framework-var-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.var framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("returns owning dtype-preserving variance with canonical correction gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      axes: number[];
      correction: number;
      keepdims: boolean;
      values: number[];
      ownsData: boolean;
      fullValue: number;
      populationValues: number[];
      legacyPopulationValues: number[];
      negativeCorrectionValues: number[];
      multiAxisPopulation: number;
      gradient: number[][];
      halfDtype: string;
      halfGradientDtype: string;
      doubleDtype: string;
      zeroDenominatorInfinite: boolean;
      scalarDefaultNan: boolean;
      scalarPopulation: number;
    }>(`
import browsergrad_jit as bg
import numpy as np

source = bg.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), requires_grad=True)
variance = source.var(dim=1)
array = variance.numpy()
variance.sum().backward()
population = source.var(dim=1, correction=0)
legacy_population = source.var(dim=1, unbiased=False)
negative_correction = source.var(dim=1, correction=np.int64(-1))
multi_axis_population = source.var(dim=(0, 1), correction=0)
half = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float16), requires_grad=True)
half_variance = half.var(correction=0)
half_variance.backward()
double_variance = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64)).var()
zero_denominator = source.var(dim=1, correction=3)
scalar = bg.tensor(2.0)

{
    "op": variance._uop.op,
    "axes": list(variance._uop.arg["axes"]),
    "correction": variance._uop.arg["correction"],
    "keepdims": variance._uop.arg["keepdims"],
    "values": array.tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "fullValue": float(source.var().item()),
    "populationValues": population.numpy().tolist(),
    "legacyPopulationValues": legacy_population.numpy().tolist(),
    "negativeCorrectionValues": negative_correction.numpy().tolist(),
    "multiAxisPopulation": float(multi_axis_population.item()),
    "gradient": source.grad.numpy().tolist(),
    "halfDtype": half_variance.dtype,
    "halfGradientDtype": half.grad.dtype,
    "doubleDtype": double_variance.dtype,
    "zeroDenominatorInfinite": bool(np.isinf(zero_denominator.numpy()).all()),
    "scalarDefaultNan": bool(np.isnan(scalar.var().item())),
    "scalarPopulation": float(scalar.var(correction=0).item()),
}
`);

    expect(result.op).toBe("VAR");
    expect(result.axes).toEqual([1]);
    expect(result.correction).toBe(1);
    expect(result.keepdims).toBe(false);
    expect(result.values).toEqual([1, 1]);
    expect(result.ownsData).toBe(true);
    expect(result.fullValue).toBe(3.5);
    for (const value of result.populationValues) expect(value).toBeCloseTo(2 / 3, 6);
    for (const value of result.legacyPopulationValues) expect(value).toBeCloseTo(2 / 3, 6);
    expect(result.negativeCorrectionValues).toEqual([0.5, 0.5]);
    expect(result.multiAxisPopulation).toBeCloseTo(35 / 12, 6);
    expect(result.gradient).toEqual([[-1, 0, 1], [-1, 0, 1]]);
    expect(result.halfDtype).toBe("float16");
    expect(result.halfGradientDtype).toBe("float16");
    expect(result.doubleDtype).toBe("float64");
    expect(result.zeroDenominatorInfinite).toBe(true);
    expect(result.scalarDefaultNan).toBe(true);
    expect(result.scalarPopulation).toBe(0);
  });

  it("provides typed VJP, batch-safe vmap, exact ONNX decomposition, and device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      gradientOps: string[];
      mapped: number[];
      mappedKeepdims: number[][];
      mappedAxes: number[];
      mappedCorrection: number;
      onnxOps: string[];
      halfOnnxError: string;
      doubleOnnxError: string;
      planError: string;
      webgpuSupported: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_VAR, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
gradient = bg.func.grad(lambda value: value.var(dim=1).sum())(source)
mapped = bg.func.vmap(lambda row: row.var(dim=0))(source)
mapped_keepdims = bg.func.vmap(lambda row: row.var(dim=0, keepdim=True))(source)
onnx = bg.onnx.export_inference(source.var(dim=1), input_buffers=(source,))

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

graph = next(payload for number, wire, payload in fields(onnx) if number == 7 and wire == 2)
onnx_ops = []
for number, wire, payload in fields(graph):
    if number != 1 or wire != 2:
        continue
    node_fields = list(fields(payload))
    onnx_ops.append(next(
        value.decode("utf-8") for field, kind, value in node_fields
        if field == 4 and kind == 2
    ))

half = bg.from_numpy(np.array([[1.0, 2.0]], dtype=np.float16))
double = bg.from_numpy(np.array([[1.0, 2.0]], dtype=np.float64))

{
    "registered": get_rule(OP_VAR) is not None,
    "gradient": gradient.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mapped": mapped.numpy().tolist(),
    "mappedKeepdims": mapped_keepdims.numpy().tolist(),
    "mappedAxes": list(mapped._uop.arg["axes"]),
    "mappedCorrection": mapped._uop.arg["correction"],
    "onnxOps": onnx_ops,
    "halfOnnxError": error(lambda: bg.onnx.export_inference(half.var(dim=1), input_buffers=(half,))),
    "doubleOnnxError": error(lambda: bg.onnx.export_inference(double.var(dim=1), input_buffers=(double,))),
    "planError": error(lambda: bg.gpu_plan_summary(source.var(dim=1))),
    "webgpuSupported": OP_VAR in supported_opcodes(),
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual([[-1, 0, 1], [-1, 0, 1]]);
    expect(result.gradientOps).toContain("REDUCE");
    expect(result.gradientOps).toContain("DIV");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual([1, 1]);
    expect(result.mappedKeepdims).toEqual([[1], [1]]);
    expect(result.mappedAxes).toEqual([1]);
    expect(result.mappedCorrection).toBe(1);
    expect(result.onnxOps).toEqual(["ReduceMean", "Sub", "Mul", "ReduceSum", "Div", "Identity"]);
    expect(result.halfOnnxError).toMatch(/^OnnxUnmappableOp: .*dtype/u);
    expect(result.doubleOnnxError).toMatch(/^OnnxUnmappableOp: .*dtype/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*VAR/u);
    expect(result.webgpuSupported).toBe(false);
  });

  it("rejects coercion, malformed contracts, and mutation at every admitted boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_VAR
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
dy = bg.from_numpy(np.ones((2,), dtype=np.float32))._uop

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
    def __iter__(self):
        HostileScalar.calls += 1
        return iter((1,))

class HostileList(list):
    def __iter__(self):
        HostileScalar.calls += 1
        return super().__iter__()

mutated_cpu = source.var(dim=1)
mutated_cpu._uop.arg["correction"] = True
mutated_vjp = source.var(dim=1)
mutated_vjp._uop.arg["axes"] = (1, 1)
mutated_vmap = source.var(dim=1)
mutated_vmap._uop.arg["keepdims"] = 1
mutated_onnx = source.var(dim=1)
mutated_onnx._uop.arg["correction"] = 2147483648
mutated_plan = source.var(dim=1)
mutated_plan._uop.arg["extra"] = True

errors = {
    "boolAxis": error(lambda: source.var(dim=True)),
    "floatAxis": error(lambda: source.var(dim=1.0)),
    "hostileAxis": error(lambda: source.var(dim=HostileScalar())),
    "hostileList": error(lambda: source.var(dim=HostileList([1]))),
    "emptyAxes": error(lambda: source.var(dim=())),
    "duplicateAxes": error(lambda: source.var(dim=(0, -2))),
    "axisRange": error(lambda: source.var(dim=2)),
    "bothAxes": error(lambda: source.var(dim=0, axis=1)),
    "keepdims": error(lambda: source.var(dim=0, keepdim=1)),
    "boolCorrection": error(lambda: source.var(correction=True)),
    "floatCorrection": error(lambda: source.var(correction=0.5)),
    "hostileCorrection": error(lambda: source.var(correction=HostileScalar())),
    "correctionRange": error(lambda: source.var(correction=2147483648)),
    "unbiased": error(lambda: source.var(unbiased=1)),
    "bothCorrections": error(lambda: source.var(correction=0, unbiased=False)),
    "integerInput": error(lambda: bg.from_numpy(np.array([1, 2], dtype=np.int32)).var()),
    "booleanInput": error(lambda: bg.from_numpy(np.array([True, False], dtype=np.bool_)).var()),
    "scalarAxis": error(lambda: bg.tensor(2.0).var(dim=0)),
    "listArg": error(lambda: UOp(OP_VAR, (source._uop,), (2,), "float32", arg={"axes": [1], "correction": 1, "keepdims": False})),
    "wrongShape": error(lambda: UOp(OP_VAR, (source._uop,), (2, 1), "float32", arg={"axes": (1,), "correction": 1, "keepdims": False})),
    "wrongDtype": error(lambda: UOp(OP_VAR, (source._uop,), (2,), "float64", arg={"axes": (1,), "correction": 1, "keepdims": False})),
    "openArgCpu": error(lambda: mutated_cpu.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_VAR)(mutated_vjp._uop, mutated_vjp._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_VAR)(mutated_vmap._uop, {id(mutated_vmap._uop.inputs[0]): mutated_vmap._uop.inputs[0]}, 1)),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated_onnx, input_buffers=(source,))),
    "openArgPlan": error(lambda: bg.gpu_plan_summary(mutated_plan)),
}
{"errors": errors, "hostileCalls": HostileScalar.calls}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.boolAxis).toMatch(/^ShapeError: .*axis must be/u);
    expect(result.errors.floatAxis).toMatch(/^ShapeError: .*axis must be/u);
    expect(result.errors.hostileAxis).toMatch(/^ShapeError: .*axis must be/u);
    expect(result.errors.hostileList).toMatch(/^ShapeError: .*axis must be/u);
    expect(result.errors.emptyAxes).toMatch(/^ShapeError: .*non-empty/u);
    expect(result.errors.duplicateAxes).toMatch(/^ShapeError: .*unique/u);
    expect(result.errors.axisRange).toMatch(/^ShapeError: .*out of range/u);
    expect(result.errors.bothAxes).toMatch(/^ShapeError: .*only one/u);
    expect(result.errors.keepdims).toMatch(/^ShapeError: .*booleans/u);
    expect(result.errors.boolCorrection).toMatch(/^ShapeError: .*correction must be/u);
    expect(result.errors.floatCorrection).toMatch(/^ShapeError: .*correction must be/u);
    expect(result.errors.hostileCorrection).toMatch(/^ShapeError: .*correction must be/u);
    expect(result.errors.correctionRange).toMatch(/^ShapeError: .*correction must be in/u);
    expect(result.errors.unbiased).toMatch(/^ShapeError: .*unbiased must be/u);
    expect(result.errors.bothCorrections).toMatch(/^ShapeError: .*only one/u);
    expect(result.errors.integerInput).toMatch(/^ShapeError: .*floating dtypes/u);
    expect(result.errors.booleanInput).toMatch(/^ShapeError: .*floating dtypes/u);
    expect(result.errors.scalarAxis).toMatch(/^ShapeError: .*rank 0/u);
    expect(result.errors.listArg).toMatch(/^ShapeError: .*canonical tuple/u);
    expect(result.errors.wrongShape).toMatch(/^ShapeError: .*derived shape/u);
    expect(result.errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(result.errors.openArgCpu).toMatch(/^RealizationError: .*normalized integer/u);
    expect(result.errors.openArgVjp).toMatch(/^ShapeError: .*unique/u);
    expect(result.errors.openArgVmap).toMatch(/^ShapeError: .*boolean/u);
    expect(result.errors.openArgOnnx).toMatch(/^ShapeError: .*correction must be in/u);
    expect(result.errors.openArgPlan).toMatch(/^ShapeError: .*fields must be exactly/u);
  });

  it("matches the shared eager/lazy variance fixture", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_VAR_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[];
      negativeAxisValues: number[];
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
output = source.var(dim=valid["axis"], correction=valid["correction"])
negative_axis = source.var(dim=valid["negativeAxis"])
keepdims = source.var(dim=valid["axis"], keepdim=True)

dtypes = []
for case in fixture["dtypeCases"]:
    typed = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(bg.from_numpy(typed).var(dim=valid["axis"]).dtype)

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "axis":
            source.var(dim=case["value"])
        elif case["kind"] == "both-axis-aliases":
            source.var(dim=0, axis=1)
        elif case["kind"] == "keepdims":
            source.var(dim=0, keepdim=case["value"])
        elif case["kind"] == "correction":
            source.var(dim=0, correction=case["value"])
        elif case["kind"] == "unbiased":
            source.var(dim=0, unbiased=case["value"])
        elif case["kind"] == "both-correction-aliases":
            source.var(dim=0, correction=0, unbiased=False)
        elif case["kind"] == "dtype":
            typed = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.dtype(case["value"])).reshape(valid["inputShape"]))
            typed.var(dim=1)
        elif case["kind"] == "scalar":
            bg.tensor(2.0).var(dim=case["value"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.numpy().tolist(),
    "negativeAxisValues": negative_axis.numpy().tolist(),
    "keepdimsShape": list(keepdims.shape),
    "keepdimsValues": keepdims.numpy().tolist(),
    "fullValue": float(source.var().item()),
    "dtypes": dtypes,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_VAR_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.outputValues);
    expect(result.negativeAxisValues).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.outputValues);
    expect(result.keepdimsShape).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.keepdimsShape);
    expect(result.keepdimsValues).toEqual(FRAMEWORK_VAR_CONFORMANCE.valid.keepdimsValues);
    expect(result.fullValue).toBe(FRAMEWORK_VAR_CONFORMANCE.valid.fullValue);
    expect(result.dtypes).toEqual(
      FRAMEWORK_VAR_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const invalid of FRAMEWORK_VAR_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).toContain(invalid.message);
    }
  });
});
