import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_GATHER_CONFORMANCE } from "../../../test-support/framework-gather-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.gather framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("returns owning dtype-preserving gathers and deterministic duplicate-index gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      dim: number;
      values: number[][];
      ownsData: boolean;
      gradient: number[][];
      halfDtype: string;
      halfGradientDtype: string;
      intDtype: string;
      boolDtype: string;
      partialValues: number[][];
      emptyShape: number[];
      emptyGradient: number[][];
    }>(`
import browsergrad_jit as bg
import numpy as np

source = bg.from_numpy(
    np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32),
    requires_grad=True,
)
index = bg.from_numpy(np.array([[2, 0], [1, 1]], dtype=np.int64))
gathered = source.gather(1, index)
array = gathered.numpy()
gathered.sum().backward()

half_source = bg.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float16), requires_grad=True)
half_index = bg.from_numpy(np.array([[2, 0]], dtype=np.int64))
half = half_source.gather(-1, half_index)
half.sum().backward()
integer = bg.from_numpy(np.array([[1, 2, 3]], dtype=np.int32)).gather(-1, half_index)
boolean = bg.from_numpy(np.array([[True, False, True]], dtype=np.bool_)).gather(-1, half_index)

partial_source = bg.from_numpy(np.arange(9, dtype=np.float32).reshape(3, 3))
partial_index = bg.from_numpy(np.array([[2, 0], [1, 1]], dtype=np.int64))
partial = partial_source.gather(1, partial_index)

empty_source = bg.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
empty_index = bg.from_numpy(np.empty((2, 0), dtype=np.int64))
empty = empty_source.gather(1, empty_index)
empty.sum().backward()

{
    "op": gathered._uop.op,
    "dim": gathered._uop.arg["dim"],
    "values": array.tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "gradient": source.grad.numpy().tolist(),
    "halfDtype": half.dtype,
    "halfGradientDtype": half_source.grad.dtype,
    "intDtype": integer.dtype,
    "boolDtype": boolean.dtype,
    "partialValues": partial.numpy().tolist(),
    "emptyShape": list(empty.shape),
    "emptyGradient": empty_source.grad.numpy().tolist(),
}
`);

    expect(result).toEqual({
      op: "INDEX",
      dim: 1,
      values: [[12, 10], [21, 21]],
      ownsData: true,
      gradient: [[1, 0, 1], [0, 2, 0]],
      halfDtype: "float16",
      halfGradientDtype: "float16",
      intDtype: "int32",
      boolDtype: "bool",
      partialValues: [[2, 0], [4, 4]],
      emptyShape: [2, 0],
      emptyGradient: [[0, 0, 0], [0, 0, 0]],
    });
  });

  it("provides typed scatter-add VJP, paired vmap, GatherElements export, and device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[][];
      gradientOps: string[];
      mapped: number[][];
      mappedDim: number;
      onnxOps: string[];
      gatherAxis: number;
      planError: string;
      webgpuSupported: boolean;
      rangeErrors: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_INDEX, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32))
index = bg.from_numpy(np.array([[2, 0], [1, 1]], dtype=np.int64))
gradient = bg.func.grad(lambda value: value.gather(1, index).sum())(source)
mapped = bg.func.vmap(lambda value, ids: value.gather(0, ids))(source, index)
onnx = bg.onnx.export_inference(source.gather(1, index), input_buffers=(source, index))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

negative = bg.from_numpy(np.array([[-1, 0], [0, 1]], dtype=np.int64))
too_large = bg.from_numpy(np.array([[3, 0], [0, 1]], dtype=np.int64))
range_errors = [
    error(lambda: source.gather(1, negative).numpy()),
    error(lambda: bg.func.grad(lambda value: value.gather(1, too_large).sum())(source).numpy()),
]

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
gather_axis = None
for number, wire, payload in fields(graph):
    if number != 1 or wire != 2:
        continue
    node_fields = list(fields(payload))
    op_type = next(
        value.decode("utf-8") for field, kind, value in node_fields
        if field == 4 and kind == 2
    )
    onnx_ops.append(op_type)
    if op_type == "GatherElements":
        for field, kind, attr_payload in node_fields:
            if field != 5 or kind != 2:
                continue
            attr_fields = list(fields(attr_payload))
            name = next(
                value.decode("utf-8") for attr_field, attr_kind, value in attr_fields
                if attr_field == 1 and attr_kind == 2
            )
            if name == "axis":
                gather_axis = next(
                    value for attr_field, attr_kind, value in attr_fields
                    if attr_field == 3 and attr_kind == 0
                )

{
    "registered": get_rule(OP_INDEX) is not None,
    "gradient": gradient.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mapped": mapped.numpy().tolist(),
    "mappedDim": mapped._uop.arg["dim"],
    "onnxOps": onnx_ops,
    "gatherAxis": gather_axis,
    "planError": error(lambda: bg.gpu_plan_summary(source.gather(1, index))),
    "webgpuSupported": OP_INDEX in supported_opcodes(),
    "rangeErrors": range_errors,
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual([[1, 0, 1], [0, 2, 0]]);
    expect(result.gradientOps).toContain("SCATTER_ADD");
    expect(result.gradientOps).toContain("BROADCAST_TO");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual([[12, 10], [21, 21]]);
    expect(result.mappedDim).toBe(1);
    expect(result.onnxOps).toEqual(["GatherElements", "Identity"]);
    expect(result.gatherAxis).toBe(1);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*INDEX/u);
    expect(result.webgpuSupported).toBe(false);
    expect(result.rangeErrors[0]).toMatch(/^RealizationError: .*\[0, 3\)/u);
    expect(result.rangeErrors[1]).toMatch(/^RealizationError: .*\[0, 3\)/u);
  });

  it("rejects coercion, malformed metadata, hostile axes, and boundary mutation", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._framework_contracts import validate_gather_contract
from browsergrad_jit._ir import UOp, OP_INDEX
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
index = bg.from_numpy(np.array([[2, 0], [1, 1]], dtype=np.int64))
dy = bg.from_numpy(np.ones((2, 2), dtype=np.float32))._uop

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

class HostileAxis:
    calls = 0
    def __int__(self):
        HostileAxis.calls += 1
        return 1

valid = source.gather(1, index)
bad_arg = source.gather(1, index)
bad_arg._uop.arg["dim"] = True
open_arg = source.gather(1, index)
open_arg._uop.arg["extra"] = 1
def wrong_shape():
    return UOp(
        op=OP_INDEX,
        inputs=(source._uop, index._uop),
        shape=(2, 1),
        dtype=source.dtype,
        arg={"dim": 1},
    )

def wrong_dtype():
    return UOp(
        op=OP_INDEX,
        inputs=(source._uop, index._uop),
        shape=index.shape,
        dtype="float16",
        arg={"dim": 1},
    )

errors = {
    "hostile-axis": error(lambda: source.gather(HostileAxis(), index)),
    "non-tensor-index": error(lambda: source.gather(1, [[0, 1], [1, 0]])),
    "int32-index": error(lambda: source.gather(1, bg.from_numpy(np.zeros((2, 2), dtype=np.int32)))),
    "rank-mismatch": error(lambda: source.gather(1, bg.from_numpy(np.zeros((2,), dtype=np.int64)))),
    "oversized-dimension": error(lambda: source.gather(1, bg.from_numpy(np.zeros((3, 1), dtype=np.int64)))),
    "bad-arg-cpu": error(lambda: bad_arg.numpy()),
    "bad-arg-vjp": error(lambda: get_rule(OP_INDEX)(bad_arg._uop, bad_arg._uop.inputs, dy)),
    "bad-arg-vmap": error(lambda: get_vmap_rule(OP_INDEX)(bad_arg._uop, {id(source._uop): source._uop, id(index._uop): index._uop}, 2)),
    "bad-arg-onnx": error(lambda: bg.onnx.export_inference(bad_arg, input_buffers=(source, index))),
    "bad-arg-plan": error(lambda: bg.gpu_plan_summary(bad_arg)),
    "open-arg": error(lambda: validate_gather_contract(open_arg._uop)),
    "wrong-shape": error(wrong_shape),
    "wrong-dtype": error(wrong_dtype),
    "valid": error(lambda: validate_gather_contract(valid._uop)),
}
{"errors": errors, "hostileCalls": HostileAxis.calls}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.valid).toBe("no_error");
    for (const [name, message] of Object.entries(result.errors)) {
      if (name === "valid") continue;
      expect(message, name).not.toBe("no_error");
    }
    expect(result.errors["non-tensor-index"]).toMatch(/^TypeError: .*TensorProxy/u);
    expect(result.errors["int32-index"]).toMatch(/^ShapeError: .*int64/u);
    expect(result.errors["bad-arg-cpu"]).toMatch(/^RealizationError: .*normalized/u);
    expect(result.errors["wrong-shape"]).toMatch(/^ShapeError: .*output shape/u);
    expect(result.errors["wrong-dtype"]).toMatch(/^ShapeError: .*source dtype/u);
  });

  it("matches the shared eager/lazy fixture", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_GATHER_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      outputShape: number[];
      outputValues: number[];
      sourceGradient: number[];
      dtypes: string[];
      errors: Record<string, string>;
    }>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source = bg.from_numpy(
    np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"]),
    requires_grad=True,
)
index = bg.from_numpy(np.asarray(valid["indexValues"], dtype=np.int64).reshape(valid["indexShape"]))
output = source.gather(valid["axis"], index)
output.sum().backward()

dtypes = []
for case in fixture["dtypeCases"]:
    typed = bg.from_numpy(
        np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    )
    dtypes.append(typed.gather(valid["axis"], index).dtype)

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "axis":
            source.gather(case["value"], index)
        elif case["kind"] == "index-dtype":
            bad = bg.from_numpy(
                np.asarray(valid["indexValues"], dtype=np.dtype(case["value"])).reshape(valid["indexShape"])
            )
            source.gather(valid["axis"], bad)
        elif case["kind"] == "index-rank":
            source.gather(valid["axis"], bg.from_numpy(np.asarray(case["value"], dtype=np.int64)))
        elif case["kind"] == "index-shape":
            source.gather(valid["axis"], bg.from_numpy(np.zeros(case["value"], dtype=np.int64)))
        elif case["kind"] == "non-tensor":
            source.gather(valid["axis"], case["value"])
        elif case["kind"] == "index-values":
            bad = bg.from_numpy(np.asarray(case["value"], dtype=np.int64).reshape(valid["indexShape"]))
            source.gather(valid["axis"], bad).numpy()
        elif case["kind"] == "scalar-source":
            bg.tensor(2.0).gather(case["value"], bg.from_numpy(np.asarray(0, dtype=np.int64)))
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "outputShape": list(output.shape),
    "outputValues": output.numpy().reshape(-1).tolist(),
    "sourceGradient": source.grad.numpy().reshape(-1).tolist(),
    "dtypes": dtypes,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_GATHER_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_GATHER_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_GATHER_CONFORMANCE.valid.outputValues);
    expect(result.sourceGradient).toEqual(FRAMEWORK_GATHER_CONFORMANCE.valid.sourceGradient);
    expect(result.dtypes).toEqual(
      FRAMEWORK_GATHER_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const invalid of FRAMEWORK_GATHER_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).toContain(invalid.message);
    }
  });
});
