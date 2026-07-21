import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_EXPAND_CONFORMANCE } from "../../../test-support/framework-expand-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.expand contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("uses typed BROADCAST_TO for CPU realization, dtype preservation, closure autograd, and planning", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      rootOp: string;
      rootArgKeys: string[];
      shape: number[];
      values: number[][];
      ownsData: boolean;
      dtype: string;
      closureGradient: number[][];
      integerDtype: string;
      integerValues: number[][];
      emptyShape: number[];
      emptySize: number;
      planOps: string[];
      planHasCustom: boolean;
      planRootOp: string;
    }>(`
import browsergrad_jit as bg
import numpy as np

x = bg.from_numpy(np.array([[1.0], [2.0]], dtype=np.float32), requires_grad=True)
expanded = x.expand(-1, 3)
arr = expanded.numpy()
plan = bg.gpu_plan_summary(expanded)
expanded.sum().backward()

ints = bg.from_numpy(np.array([[4], [7]], dtype=np.int32)).expand(2, 2)
int_arr = ints.numpy()
empty = bg.from_numpy(np.array([[1.0], [2.0]], dtype=np.float32)).expand(2, 0).numpy()

{
    "rootOp": expanded._uop.op,
    "rootArgKeys": sorted(expanded._uop.arg.keys()),
    "shape": list(arr.shape),
    "values": arr.tolist(),
    "ownsData": bool(arr.flags["OWNDATA"]),
    "dtype": str(arr.dtype),
    "closureGradient": x.grad.numpy().tolist(),
    "integerDtype": str(int_arr.dtype),
    "integerValues": int_arr.tolist(),
    "emptyShape": list(empty.shape),
    "emptySize": int(empty.size),
    "planOps": plan["ops"],
    "planHasCustom": bool(plan["has_custom_ops"]),
    "planRootOp": plan["steps"][-1]["op"],
}
`);

    expect(result).toEqual({
      rootOp: "BROADCAST_TO",
      rootArgKeys: ["shape"],
      shape: [2, 3],
      values: [[1, 1, 1], [2, 2, 2]],
      ownsData: true,
      dtype: "float32",
      closureGradient: [[3], [3]],
      integerDtype: "int32",
      integerValues: [[4, 4], [7, 7]],
      emptyShape: [2, 0],
      emptySize: 0,
      planOps: ["BUFFER", "LOAD", "BROADCAST_TO"],
      planHasCustom: false,
      planRootOp: "BROADCAST_TO",
    });
  });

  it("has symbolic VJP, vmap, and ONNX Expand decisions", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      vjpRegistered: boolean;
      functionalGradient: number[][];
      vmapShape: number[];
      vmapValues: number[][];
      onnxHasExpand: boolean;
      onnxHasCustom: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_BROADCAST_TO
from browsergrad_jit._vjp import get_rule

x = bg.from_numpy(np.array([[1.0], [2.0]], dtype=np.float32))
functional_gradient = bg.func.grad(lambda value: value.expand(2, 3).sum())(x)

batched = bg.from_numpy(np.array([[1.0], [2.0]], dtype=np.float32))
mapped = bg.func.vmap(lambda row: row.expand(3))(batched)

export_input = bg.from_numpy(np.array([[5.0], [9.0]], dtype=np.float32))
onnx_bytes = bg.onnx.export_inference(
    export_input.expand(2, 3),
    input_buffers=(export_input,),
)

{
    "vjpRegistered": get_rule(OP_BROADCAST_TO) is not None,
    "functionalGradient": functional_gradient.numpy().tolist(),
    "vmapShape": list(mapped.shape),
    "vmapValues": mapped.numpy().tolist(),
    "onnxHasExpand": b"Expand" in onnx_bytes,
    "onnxHasCustom": b"CUSTOM" in onnx_bytes,
}
`);

    expect(result).toEqual({
      vjpRegistered: true,
      functionalGradient: [[3], [3]],
      vmapShape: [2, 3],
      vmapValues: [[1, 1, 1], [2, 2, 2]],
      onnxHasExpand: true,
      onnxHasCustom: false,
    });
  });

  it("rejects invalid shapes and post-construction contract mutation before execution", async () => {
    const target = await getJitTarget();
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_BROADCAST_TO

x = bg.from_numpy(np.array([[1.0], [2.0]], dtype=np.float32))
vector = bg.from_numpy(np.array([1.0, 2.0], dtype=np.float32))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

mutated_shape = x.expand(2, 3)
mutated_shape._uop.arg["shape"] = (2, 2)
mutated_fields = x.expand(2, 3)
mutated_fields._uop.arg["backend"] = "webgpu"

errors = {
    "float": error(lambda: x.expand(2, 3.0)),
    "bool": error(lambda: x.expand(2, True)),
    "leadingMinusOne": error(lambda: vector.expand(-1, 2)),
    "negative": error(lambda: x.expand(2, -2)),
    "incompatible": error(lambda: x.expand(3, 3)),
    "fewerDimensions": error(lambda: x.expand(2)),
    "mutatedShapeCpu": error(lambda: mutated_shape.numpy()),
    "mutatedFieldsPlan": error(lambda: bg.gpu_plan_summary(mutated_fields)),
    "listArg": error(lambda: UOp(
        OP_BROADCAST_TO,
        (x._uop,),
        (2, 3),
        "float32",
        arg={"shape": [2, 3]},
    )),
    "dtypeChange": error(lambda: UOp(
        OP_BROADCAST_TO,
        (x._uop,),
        (2, 3),
        "int32",
        arg={"shape": (2, 3)},
    )),
}
errors
`);

    expect(errors.float).toMatch(/^ShapeError: .*must be an integer/u);
    expect(errors.bool).toMatch(/^ShapeError: .*must be an integer/u);
    expect(errors.leadingMinusOne).toMatch(/^ShapeError: .*-1 is not allowed/u);
    expect(errors.negative).toMatch(/^ShapeError: .*non-negative or -1/u);
    expect(errors.incompatible).toMatch(/^ShapeError: .*cannot expand/u);
    expect(errors.fewerDimensions).toMatch(/^ShapeError: .*fewer dims/u);
    expect(errors.mutatedShapeCpu).toMatch(/^RealizationError: .*does not match node shape/u);
    expect(errors.mutatedFieldsPlan).toMatch(/^ShapeError: .*arg fields/u);
    expect(errors.listArg).toMatch(/^ShapeError: .*arg\.shape must be a tuple/u);
    expect(errors.dtypeChange).toMatch(/^ShapeError: .*must preserve dtype/u);
  });

  it("matches the shared eager/lazy expand conformance fixture", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_EXPAND_CONFORMANCE);
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
output = bg.from_numpy(base).expand(*valid["requestedShape"]).numpy()

dtypes = []
for case in fixture["dtypeCases"]:
    source = np.asarray(valid["inputValues"], dtype=np.dtype(case["dtype"])).reshape(valid["inputShape"])
    dtypes.append(bg.from_numpy(source).expand(*valid["requestedShape"]).dtype)

errors = {}
for case in fixture["invalid"]:
    source = np.arange(np.prod(case["inputShape"]), dtype=np.float32).reshape(case["inputShape"])
    try:
        bg.from_numpy(source).expand(*case["requestedShape"])
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

    expect(result.schema).toBe(FRAMEWORK_EXPAND_CONFORMANCE.schema);
    expect(result.outputShape).toEqual(FRAMEWORK_EXPAND_CONFORMANCE.valid.outputShape);
    expect(result.outputValues).toEqual(FRAMEWORK_EXPAND_CONFORMANCE.valid.outputValues);
    expect(result.dtypes).toEqual(
      FRAMEWORK_EXPAND_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const fixture of FRAMEWORK_EXPAND_CONFORMANCE.invalid) {
      expect(result.errors[fixture.id]).toContain(fixture.message);
    }
  });
});
