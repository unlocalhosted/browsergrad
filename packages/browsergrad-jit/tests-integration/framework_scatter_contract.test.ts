import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_SCATTER_CONFORMANCE } from "../../../test-support/framework-scatter-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.scatter contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches shared overwrite values, dtypes, gradients, aliases, and refusals", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SCATTER_CONFORMANCE);
    const result = await target.run<{
      schema: string;
      op: string;
      dim: number;
      outputShape: number[];
      outputValues: number[];
      scalarOutputValues: number[];
      targetGradient: number[];
      sourceGradient: number[];
      ownsData: boolean;
      dtypes: string[];
      halfGradientDtypes: string[];
      emptyValues: number[];
      aliasValues: number[];
      errors: Record<string, string>;
    }>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
target_values = np.asarray(valid["targetValues"], dtype=np.float32).reshape(valid["targetShape"])
index_values = np.asarray(valid["indexValues"], dtype=np.int64).reshape(valid["indexShape"])
source_values = np.asarray(valid["sourceValues"], dtype=np.float32).reshape(valid["indexShape"])

target = bg.from_numpy(target_values, requires_grad=True)
index = bg.from_numpy(index_values)
source = bg.from_numpy(source_values, requires_grad=True)
output = target.scatter(valid["axis"], index, source)
output_array = output.numpy()
output.sum().backward()
scalar_output = bg.scatter(
    bg.from_numpy(target_values),
    valid["negativeAxis"],
    index,
    valid["scalarSource"],
)

dtypes = []
half_gradient_dtypes = []
for case in fixture["dtypeCases"]:
    dtype = case["dtype"]
    if dtype == "bool":
        typed_target_values = (target_values != 0)
        typed_source_values = np.asarray([[False, True], [True, False]], dtype=np.bool_)
    else:
        typed_target_values = target_values.astype(np.dtype(dtype))
        typed_source_values = source_values.astype(np.dtype(dtype))
    typed_target = bg.from_numpy(typed_target_values, requires_grad=dtype == "float16")
    typed_source = bg.from_numpy(typed_source_values, requires_grad=dtype == "float16")
    typed_output = typed_target.scatter(valid["axis"], index, typed_source)
    dtypes.append(typed_output.dtype)
    if dtype == "float16":
        typed_output.sum().backward()
        half_gradient_dtypes = [typed_target.grad.dtype, typed_source.grad.dtype]

empty_index = bg.from_numpy(np.empty((2, 0), dtype=np.int64))
empty_source = bg.from_numpy(np.empty((2, 0), dtype=np.float32))
empty_values = bg.from_numpy(target_values).scatter(1, empty_index, empty_source).numpy().reshape(-1).tolist()

bg.install_torch_alias()
import torch
alias_values = torch.scatter(
    torch.tensor(target_values),
    1,
    torch.tensor(index_values, dtype=torch.int64),
    torch.tensor(source_values),
).numpy().reshape(-1).tolist()

errors = {}
for case in fixture["invalid"]:
    try:
        kind = case["kind"]
        base = bg.from_numpy(target_values)
        if kind == "axis":
            bg.scatter(base, case["value"], index, bg.from_numpy(source_values))
        elif kind == "index-dtype":
            bad = bg.from_numpy(index_values.astype(np.dtype(case["value"])))
            bg.scatter(base, 1, bad, bg.from_numpy(source_values))
        elif kind == "index-shape":
            shape = tuple(case["value"])
            bad = bg.from_numpy(np.zeros(shape, dtype=np.int64))
            bad_source = bg.from_numpy(np.zeros(shape, dtype=np.float32))
            bg.scatter(base, 1, bad, bad_source)
        elif kind == "non-tensor-index":
            bg.scatter(base, 1, case["value"], bg.from_numpy(source_values))
        elif kind == "source-shape":
            bad_source = bg.from_numpy(np.zeros(tuple(case["value"]), dtype=np.float32))
            bg.scatter(base, 1, index, bad_source)
        elif kind == "source-dtype":
            bad_source = bg.from_numpy(source_values.astype(np.float64))
            bg.scatter(base, 1, index, bad_source)
        elif kind == "index-values":
            bad = bg.from_numpy(
                np.asarray(case["value"], dtype=np.int64).reshape(valid["indexShape"])
            )
            bg.scatter(base, 1, bad, bg.from_numpy(source_values)).numpy()
        elif kind == "reduce":
            bg.scatter(base, 1, index, bg.from_numpy(source_values), reduce=case["value"])
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = str(exc)

{
    "schema": fixture["schema"],
    "op": output._uop.op,
    "dim": output._uop.arg["dim"],
    "outputShape": list(output.shape),
    "outputValues": output_array.reshape(-1).tolist(),
    "scalarOutputValues": scalar_output.numpy().reshape(-1).tolist(),
    "targetGradient": target.grad.numpy().reshape(-1).tolist(),
    "sourceGradient": source.grad.numpy().reshape(-1).tolist(),
    "ownsData": bool(output_array.flags["OWNDATA"]),
    "dtypes": dtypes,
    "halfGradientDtypes": half_gradient_dtypes,
    "emptyValues": empty_values,
    "aliasValues": alias_values,
    "errors": errors,
}
`);

    expect(result.schema).toBe(FRAMEWORK_SCATTER_CONFORMANCE.schema);
    expect(result.op).toBe("SCATTER");
    expect(result.dim).toBe(1);
    expect(result.outputShape).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.outputValues);
    expect(result.scalarOutputValues).toEqual(
      FRAMEWORK_SCATTER_CONFORMANCE.valid.scalarOutputValues,
    );
    expect(result.targetGradient).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.targetGradient);
    expect(result.sourceGradient).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.sourceGradient);
    expect(result.ownsData).toBe(true);
    expect(result.dtypes).toEqual(
      FRAMEWORK_SCATTER_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    expect(result.halfGradientDtypes).toEqual(["float16", "float16"]);
    expect(result.emptyValues).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.targetValues);
    expect(result.aliasValues).toEqual(FRAMEWORK_SCATTER_CONFORMANCE.valid.outputValues);
    for (const invalid of FRAMEWORK_SCATTER_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id], invalid.id).toContain(invalid.message);
    }
  });

  it("provides typed VJP, captured-safe vmap, ScatterElements export, and device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      targetGradient: number[][];
      sourceGradient: number[][];
      targetGradientOps: string[];
      sourceGradientOps: string[];
      mapped: number[][];
      capturedMapped: number[][];
      mappedDim: number;
      onnxScatter: boolean;
      scalarOnnxExpand: boolean;
      planError: string;
      webgpuSupported: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_SCATTER, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

target = bg.from_numpy(np.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]], dtype=np.float32))
index = bg.from_numpy(np.array([[3, 0], [1, 2]], dtype=np.int64))
source = bg.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
target_gradient = bg.func.grad(lambda value: bg.scatter(value, 1, index, source).sum())(target)
source_gradient = bg.func.grad(lambda value: bg.scatter(target, 1, index, value).sum())(source)

mapped = bg.func.vmap(lambda row, ids, values: row.scatter(0, ids, values))(
    target,
    index,
    source,
)
captured_index = bg.from_numpy(np.array([3, 0], dtype=np.int64))
captured_source = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
captured_mapped = bg.func.vmap(
    lambda row: row.scatter(0, captured_index, captured_source)
)(target)

onnx = bg.onnx.export_inference(
    bg.scatter(target, 1, index, source),
    input_buffers=(target, index, source),
)
scalar_onnx = bg.onnx.export_inference(
    bg.scatter(target, 1, index, -5.0),
    input_buffers=(target, index),
)

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "registered": get_rule(OP_SCATTER) is not None,
    "targetGradient": target_gradient.numpy().tolist(),
    "sourceGradient": source_gradient.numpy().tolist(),
    "targetGradientOps": sorted({node.op for node in toposort(target_gradient._uop)}),
    "sourceGradientOps": sorted({node.op for node in toposort(source_gradient._uop)}),
    "mapped": mapped.numpy().tolist(),
    "capturedMapped": captured_mapped.numpy().tolist(),
    "mappedDim": mapped._uop.arg["dim"],
    "onnxScatter": b"ScatterElements" in onnx,
    "scalarOnnxExpand": b"ScatterElements" in scalar_onnx and b"Expand" in scalar_onnx,
    "planError": error(lambda: bg.gpu_plan_summary(bg.scatter(target, 1, index, source))),
    "webgpuSupported": OP_SCATTER in supported_opcodes(),
}
`);

    expect(result.registered).toBe(true);
    expect(result.targetGradient).toEqual([[0, 1, 1, 0], [1, 0, 0, 1]]);
    expect(result.sourceGradient).toEqual([[1, 1], [1, 1]]);
    expect(result.targetGradientOps).toContain("SCATTER");
    expect(result.sourceGradientOps).toContain("INDEX");
    expect(result.targetGradientOps).not.toContain("CUSTOM");
    expect(result.sourceGradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual([[2, 20, 30, 1], [50, 3, 4, 80]]);
    expect(result.capturedMapped).toEqual([[2, 20, 30, 1], [2, 60, 70, 1]]);
    expect(result.mappedDim).toBe(1);
    expect(result.onnxScatter).toBe(true);
    expect(result.scalarOnnxExpand).toBe(true);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*SCATTER/u);
    expect(result.webgpuSupported).toBe(false);
  });

  it("rejects hostile axes, mutation, malformed IR, duplicate writes, and oversized work", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      errors: Record<string, string>;
      hostileCalls: number;
      hostileSourceCalls: number;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._framework_contracts import validate_scatter_contract
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_CONST, OP_SCATTER
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

target = bg.from_numpy(np.zeros((2, 4), dtype=np.float32))
index = bg.from_numpy(np.array([[3, 0], [1, 2]], dtype=np.int64))
source = bg.from_numpy(np.ones((2, 2), dtype=np.float32))
dy = bg.from_numpy(np.ones((2, 4), dtype=np.float32))._uop

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

class HostileSource:
    calls = 0
    def __float__(self):
        HostileSource.calls += 1
        return 1.0
    def __int__(self):
        HostileSource.calls += 1
        return 1

valid = bg.scatter(target, 1, index, source)
bad_arg = bg.scatter(target, 1, index, source)
bad_arg._uop.arg["dim"] = True
open_arg = bg.scatter(target, 1, index, source)
open_arg._uop.arg["extra"] = 1
duplicate = bg.from_numpy(np.array([[1, 1], [0, 2]], dtype=np.int64))

def forged(shape, index_shape, dtype="float32"):
    target_uop = UOp(OP_BUFFER, (), shape, dtype, {"buffer_id": "target"})
    index_uop = UOp(OP_BUFFER, (), index_shape, "int64", {"buffer_id": "index"})
    source_uop = UOp(OP_CONST, (), (), dtype, {"value": 0})
    return UOp(OP_SCATTER, (target_uop, index_uop, source_uop), shape, dtype, {"dim": len(shape) - 1})

errors = {
    "hostile-axis": error(lambda: bg.scatter(target, HostileAxis(), index, source)),
    "hostile-source": error(lambda: bg.scatter(target, 1, index, HostileSource())),
    "negative-index": error(lambda: bg.scatter(target, 1, bg.from_numpy(np.array([[-1, 0], [1, 2]], dtype=np.int64)), source).numpy()),
    "duplicate-index": error(lambda: bg.scatter(target, 1, duplicate, source).numpy()),
    "bad-arg-cpu": error(lambda: bad_arg.numpy()),
    "bad-arg-vjp": error(lambda: get_rule(OP_SCATTER)(bad_arg._uop, bad_arg._uop.inputs, dy)),
    "bad-arg-vmap": error(lambda: get_vmap_rule(OP_SCATTER)(bad_arg._uop, {id(node): node for node in bad_arg._uop.inputs}, 2)),
    "bad-arg-onnx": error(lambda: bg.onnx.export_inference(bad_arg, input_buffers=(target, index, source))),
    "bad-arg-plan": error(lambda: bg.gpu_plan_summary(bad_arg)),
    "open-arg": error(lambda: validate_scatter_contract(open_arg._uop)),
    "wrong-output-shape": error(lambda: validate_scatter_contract(UOp(OP_SCATTER, valid._uop.inputs, (2, 3), "float32", {"dim": 1}))),
    "rank-ceiling": error(lambda: validate_scatter_contract(forged((1,) * 33, (0,) * 33))),
    "output-ceiling": error(lambda: validate_scatter_contract(forged((1, 1 << 28), (1, 0), "float64"))),
    "workspace-ceiling": error(lambda: validate_scatter_contract(forged((64, 1 << 19), (64, 1 << 19)))),
    "valid": error(lambda: validate_scatter_contract(valid._uop)),
}
{
    "errors": errors,
    "hostileCalls": HostileAxis.calls,
    "hostileSourceCalls": HostileSource.calls,
}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.hostileSourceCalls).toBe(0);
    expect(result.errors.valid).toBe("no_error");
    for (const [name, message] of Object.entries(result.errors)) {
      if (name === "valid") continue;
      expect(message, name).not.toBe("no_error");
    }
    expect(result.errors["negative-index"]).toMatch(/^RealizationError: .*\[0, 4\)/u);
    expect(result.errors["duplicate-index"]).toMatch(/^RealizationError: .*duplicate/u);
    expect(result.errors["output-ceiling"]).toContain("output requires");
    expect(result.errors["workspace-ceiling"]).toContain("workspace requires");
  });
});
