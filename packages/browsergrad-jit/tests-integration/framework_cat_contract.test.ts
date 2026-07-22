import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_CAT_CONFORMANCE } from "../../../test-support/framework-cat-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.cat framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("emits typed CONCAT with exact promotion, empty compatibility, closure gradients, and owning CPU results", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CAT_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      op: string;
      arg: { axis: number };
      shape: number[];
      values: number[][];
      dtype: string;
      gradients: unknown[];
      dtypeCases: Array<{ dtype: string; values: unknown[] }>;
      legacy: {
        shape: number[];
        dtype: string;
        values: number[][];
        emptyGradient: unknown[];
        emptyGradientDtype: string;
        matrixGradient: number[][];
        matrixGradientDtype: string;
      };
      allEmpty: { shape: number[]; dtype: string; values: unknown[] };
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
    bg.from_numpy(np.asarray(values, dtype=np.float32).reshape(shape), requires_grad=True)
    for values, shape in zip(valid["inputValues"], valid["inputShapes"])
)
output = bg.cat(list(inputs), dim=np.int32(valid["axis"]))
values = output.numpy().tolist()
cotangent = bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32))
(output * cotangent).sum().backward()

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed_inputs = [
        bg.from_numpy(np.asarray(values, dtype=np.dtype(dtype)))
        for values, dtype in zip(case["values"], case["dtypes"])
    ]
    observed = bg.cat(typed_inputs, dim=0)
    dtype_cases.append({"dtype": observed.dtype, "values": observed.numpy().tolist()})

legacy_spec = fixture["legacyEmpty"]
legacy_empty = bg.from_numpy(
    np.empty((0,), dtype=np.dtype(legacy_spec["emptyDtype"])),
    requires_grad=True,
)
legacy_matrix = bg.from_numpy(
    np.asarray(legacy_spec["matrixValues"], dtype=np.dtype(legacy_spec["matrixDtype"])),
    requires_grad=True,
)
legacy_output = bg.cat((legacy_empty, legacy_matrix), dim=legacy_spec["axis"])
legacy_output.sum().backward()

all_empty_spec = fixture["allEmpty"]
all_empty = bg.cat(tuple(
    bg.from_numpy(np.empty((0,), dtype=np.dtype(dtype)))
    for dtype in all_empty_spec["dtypes"]
), dim=all_empty_spec["axis"])

mixed_gradient_spec = fixture["mixedGradient"]
mixed_float = bg.from_numpy(
    np.asarray(mixed_gradient_spec["floatingValues"], dtype=np.float32),
    requires_grad=True,
)
mixed_int = bg.from_numpy(
    np.asarray(mixed_gradient_spec["integralValues"], dtype=np.int64),
    requires_grad=True,
)
bg.cat((mixed_float, mixed_int), dim=0).sum().backward()

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
    "non-sequence": lambda: bg.cat((tensor for tensor in inputs), 1),
    "empty-sequence": lambda: bg.cat([], 0),
    "non-tensor": lambda: bg.cat([inputs[0], [[1.0, 2.0]]], 0),
    "scalar-input": lambda: bg.cat([bg.tensor(1.0), bg.tensor(2.0)], 0),
    "bool-axis": lambda: bg.cat(inputs, True),
    "float-axis": lambda: bg.cat(inputs, 1.0),
    "low-axis": lambda: bg.cat(inputs, -3),
    "high-axis": lambda: bg.cat(inputs, 2),
    "hostile-axis": lambda: bg.cat(inputs, HostileAxis()),
    "hostile-sequence": lambda: bg.cat(HostileSequence(), 1),
    "shape-mismatch": lambda: bg.cat((
        bg.from_numpy(np.ones((2, 2), dtype=np.float32)),
        bg.from_numpy(np.ones((1, 1), dtype=np.float32)),
    ), 0),
    "rank-mismatch": lambda: bg.cat((
        bg.from_numpy(np.ones((2, 2), dtype=np.float32)),
        bg.from_numpy(np.ones((2,), dtype=np.float32)),
    ), 0),
    "unsupported-dtype": lambda: bg.cat((
        bg.from_numpy(np.ones((1,), dtype=np.uint16)),
        bg.from_numpy(np.ones((1,), dtype=np.uint16)),
    ), 0),
    "out-mutation": lambda: bg.cat(inputs, 1, out=inputs[0]),
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
rerun = bg.cat(inputs, dim=-1)

{
    "schema": fixture["schema"],
    "op": output._uop.op,
    "arg": dict(output._uop.arg),
    "shape": list(output.shape),
    "values": values,
    "dtype": output.dtype,
    "gradients": [tensor.grad.numpy().tolist() for tensor in inputs],
    "dtypeCases": dtype_cases,
    "legacy": {
        "shape": list(legacy_output.shape),
        "dtype": legacy_output.dtype,
        "values": legacy_output.numpy().tolist(),
        "emptyGradient": legacy_empty.grad.numpy().tolist(),
        "emptyGradientDtype": legacy_empty.grad.dtype,
        "matrixGradient": legacy_matrix.grad.numpy().tolist(),
        "matrixGradientDtype": legacy_matrix.grad.dtype,
    },
    "allEmpty": {
        "shape": list(all_empty.shape),
        "dtype": all_empty.dtype,
        "values": all_empty.numpy().tolist(),
    },
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

    expect(result.schema).toBe(FRAMEWORK_CAT_CONFORMANCE.schema);
    expect(result.op).toBe("CONCAT");
    expect(result.arg).toEqual({ axis: FRAMEWORK_CAT_CONFORMANCE.valid.axis });
    expect(result.shape).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedShape);
    expect(result.values).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedValues);
    expect(result.dtype).toBe("float32");
    expect(result.gradients).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedGradients);
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_CAT_CONFORMANCE.dtypeCases.map((testCase) => ({
        dtype: testCase.expectedDtype,
        values: testCase.expectedValues,
      })),
    );
    expect(result.legacy).toEqual({
      shape: [...FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedShape],
      dtype: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedDtype,
      values: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedValues,
      emptyGradient: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedEmptyGradient,
      emptyGradientDtype: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.emptyDtype,
      matrixGradient: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.expectedMatrixGradient,
      matrixGradientDtype: FRAMEWORK_CAT_CONFORMANCE.legacyEmpty.matrixDtype,
    });
    expect(result.allEmpty).toEqual({
      shape: [...FRAMEWORK_CAT_CONFORMANCE.allEmpty.expectedShape],
      dtype: FRAMEWORK_CAT_CONFORMANCE.allEmpty.expectedDtype,
      values: FRAMEWORK_CAT_CONFORMANCE.allEmpty.expectedValues,
    });
    expect(result.mixedGradient).toEqual({
      floating: FRAMEWORK_CAT_CONFORMANCE.mixedGradient.expectedFloatingGradient,
      integralPresent: FRAMEWORK_CAT_CONFORMANCE.mixedGradient.expectedIntegralGradientPresent,
    });
    expect(result.ownsData).toBe(true);
    expect(result.rerunValues).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedValues);
    expect(result.hostileCalls).toEqual({ axis: 0, sequence: 0 });
    for (const invalid of FRAMEWORK_CAT_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).not.toBe("no_error");
    }
  });

  it("provides split VJP, captured-input vmap, exact ONNX Cast/Concat, and explicit device refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CAT_CONFORMANCE);
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      gradientDtype: string;
      gradientOps: string[];
      mapped: number[][];
      mappedArg: { axis: number };
      mappedOps: string[];
      onnx: {
        opTypes: string[];
        concatInputs: string[];
        axis: number;
        outputDtype: number;
        castTo: number;
      };
      legacyOnnxInputCount: number;
      halfOnnxError: string;
      planError: string;
      webgpuSupported: boolean[];
    }>(`
import browsergrad_jit as bg
import json
import numpy as np
from browsergrad_jit._ir import OP_CONCAT, OP_NARROW, toposort
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

def parse_concat(model):
    graph = next(payload for number, wire, payload in fields(model) if number == 7 and wire == 2)
    op_types = []
    concat_inputs = None
    axis = None
    cast_to = None
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
            elif op_type == "Concat":
                concat_inputs = [
                    value.decode("utf-8") for field, kind, value in node_fields
                    if field == 1 and kind == 2
                ]
                axis = attrs["axis"]
        elif number == 12 and wire == 2:
            output_dtype = value_info_dtype(payload)
    if concat_inputs is None or axis is None:
        raise RuntimeError("missing exact Concat wiring")
    return {
        "opTypes": op_types,
        "concatInputs": concat_inputs,
        "axis": axis,
        "outputDtype": output_dtype,
        "castTo": cast_to,
    }

valid = fixture["valid"]
source = bg.from_numpy(np.asarray(valid["inputValues"][0], dtype=np.float16))
other = bg.from_numpy(np.asarray(valid["inputValues"][1], dtype=np.float64))
cotangent = bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float64))
gradient = bg.func.grad(
    lambda value: (bg.cat((value, other), dim=valid["axis"]) * cotangent).sum()
)(source)

vmap_spec = fixture["vmap"]
mapped_source = bg.from_numpy(np.asarray(vmap_spec["mappedValues"], dtype=np.float32))
captured = bg.from_numpy(np.asarray(vmap_spec["capturedValues"], dtype=np.float32))
mapped = bg.func.vmap(lambda row: bg.cat((row, captured), dim=0))(mapped_source)

float_source = bg.from_numpy(np.asarray([0.5, 1.5], dtype=np.float32))
int_source = bg.from_numpy(np.asarray([2, 3], dtype=np.int64))
mixed = bg.cat((float_source, int_source), dim=0)
onnx = parse_concat(bg.onnx.export_inference(mixed, input_buffers=(float_source, int_source)))

legacy_empty = bg.from_numpy(np.empty((0,), dtype=np.int64))
matrix = bg.from_numpy(np.asarray([[1, 2], [3, 4]], dtype=np.float32))
legacy = bg.cat((legacy_empty, matrix), dim=1)
legacy_onnx = parse_concat(bg.onnx.export_inference(legacy, input_buffers=(legacy_empty, matrix)))
half_a = bg.from_numpy(np.asarray([1], dtype=np.float16))
half_b = bg.from_numpy(np.asarray([2], dtype=np.float16))

{
    "registered": get_rule(OP_CONCAT) is not None,
    "gradient": gradient.numpy().tolist(),
    "gradientDtype": gradient.dtype,
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mapped": mapped.numpy().tolist(),
    "mappedArg": dict(mapped._uop.arg),
    "mappedOps": sorted({node.op for node in toposort(mapped._uop)}),
    "onnx": onnx,
    "legacyOnnxInputCount": len(legacy_onnx["concatInputs"]),
    "halfOnnxError": error(lambda: bg.onnx.export_inference(
        bg.cat((half_a, half_b), 0), input_buffers=(half_a, half_b)
    )),
    "planError": error(lambda: bg.gpu_plan_summary(mixed)),
    "webgpuSupported": [OP_CONCAT in supported_opcodes(), OP_NARROW in supported_opcodes()],
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual(FRAMEWORK_CAT_CONFORMANCE.valid.expectedGradients[0]);
    expect(result.gradientDtype).toBe("float16");
    expect(result.gradientOps).toContain("NARROW");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual(FRAMEWORK_CAT_CONFORMANCE.vmap.expectedValues);
    expect(result.mappedArg).toEqual(FRAMEWORK_CAT_CONFORMANCE.vmap.expectedArg);
    expect(result.mappedOps).toContain("BROADCAST_TO");
    expect(result.mappedOps).not.toContain("CUSTOM");
    expect(result.onnx.opTypes).toEqual(FRAMEWORK_CAT_CONFORMANCE.onnx.opTypes);
    expect(result.onnx.concatInputs).toHaveLength(FRAMEWORK_CAT_CONFORMANCE.onnx.concatInputCount);
    expect(result.onnx.axis).toBe(FRAMEWORK_CAT_CONFORMANCE.onnx.axis);
    expect(result.onnx.outputDtype).toBe(FRAMEWORK_CAT_CONFORMANCE.onnx.outputDtype);
    expect(result.onnx.castTo).toBe(FRAMEWORK_CAT_CONFORMANCE.onnx.castTo);
    expect(result.legacyOnnxInputCount).toBe(1);
    expect(result.halfOnnxError).toMatch(/^OnnxUnmappableOp: .*float16/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*CONCAT/u);
    expect(result.webgpuSupported).toEqual([false, false]);
  });

  it("rejects malformed concat and narrow nodes at construction and admitted boundaries", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._framework_contracts import validate_narrow_contract
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_CONCAT, OP_NARROW
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

left = bg.from_numpy(np.ones((2, 2), dtype=np.float32))
right = bg.from_numpy(np.ones((2, 1), dtype=np.float32))
uint = bg.from_numpy(np.ones((2, 1), dtype=np.uint16))
valid = bg.cat((left, right), dim=1)
mutated = bg.cat((left, right), dim=1)
mutated._uop.arg["axis"] = True
dy = bg.from_numpy(np.ones(valid.shape, dtype=np.float32))._uop
gradient = get_rule(OP_CONCAT)(valid._uop, valid._uop.inputs, dy)[0]
gradient.arg["length"] = -1

large = UOp(OP_BUFFER, (), (1 << 27,), "float32", arg="cat:large")
small = UOp(OP_BUFFER, (), (1,), "float32", arg="cat:small")

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "missingInputs": error(lambda: UOp(OP_CONCAT, (), (0,), "float32", arg={"axis": 0})),
    "wrongFields": error(lambda: UOp(OP_CONCAT, (left._uop, right._uop), (2, 3), "float32", arg={})),
    "wrongShape": error(lambda: UOp(OP_CONCAT, (left._uop, right._uop), (3, 2), "float32", arg={"axis": 1})),
    "wrongDtype": error(lambda: UOp(OP_CONCAT, (left._uop, right._uop), (2, 3), "float64", arg={"axis": 1})),
    "badAxis": error(lambda: UOp(OP_CONCAT, (left._uop, right._uop), (2, 3), "float32", arg={"axis": True})),
    "badSourceDtype": error(lambda: UOp(OP_CONCAT, (left._uop, uint._uop), (2, 3), "float32", arg={"axis": 1})),
    "tooManyInputs": error(lambda: UOp(OP_CONCAT, (small,) * 1025, (1025,), "float32", arg={"axis": 0})),
    "resourceBound": error(lambda: UOp(OP_CONCAT, (large, large), (1 << 28,), "float32", arg={"axis": 0})),
    "badNarrow": error(lambda: UOp(OP_NARROW, (dy,), (2, 1), "float32", arg={"axis": 1, "start": 0, "length": -1})),
    "mutatedNarrow": error(lambda: validate_narrow_contract(gradient)),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_CONCAT)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_CONCAT)(
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
    expect(result.badSourceDtype).toMatch(/^ShapeError: .*unsupported dtype/u);
    expect(result.tooManyInputs).toMatch(/^ShapeError: .*input count/u);
    expect(result.resourceBound).toMatch(/^ShapeError: .*byte ceiling/u);
    expect(result.badNarrow).toMatch(/^ShapeError: .*length/u);
    expect(result.mutatedNarrow).toMatch(/^ShapeError: .*length/u);
    expect(result.openArgCpu).toMatch(/^RealizationError: .*normalized integer/u);
    expect(result.openArgVjp).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.openArgVmap).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.openArgOnnx).toMatch(/^ShapeError: .*normalized integer/u);
    expect(result.openArgPlan).toMatch(/^ShapeError: .*normalized integer/u);
  });
});
