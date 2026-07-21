import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_MASKED_FILL_CONFORMANCE } from "../../../test-support/framework-masked-fill-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed Tensor.masked_fill framework contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("emits typed WHERE with owning dtype-preserving CPU and closure semantics", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      op: string;
      contractId: string;
      inputOps: string[];
      fillDtype: string;
      values: number[];
      gradient: number[];
      ownsData: boolean;
      infinityCount: number;
      dtypes: string[];
    }>(`
import browsergrad_jit as bg
import numpy as np

source = bg.from_numpy(np.arange(1, 7, dtype=np.float32).reshape(2, 3), requires_grad=True)
mask = bg.from_numpy(np.array([True, False, True], dtype=np.bool_))
output = source.masked_fill(mask, -1)
array = output.numpy()
output.sum().backward()
infinite = source.masked_fill(mask, float("-inf"))

dtypes = []
for dtype in ("float16", "float32", "float64", "int32", "int64"):
    values = np.arange(1, 7, dtype=np.dtype(dtype)).reshape(2, 3)
    dtypes.append(bg.from_numpy(values).masked_fill(mask, -1).dtype)
bool_values = bg.from_numpy(np.ones((2, 3), dtype=np.bool_))
dtypes.append(bool_values.masked_fill(mask, False).dtype)

{
    "op": output._uop.op,
    "contractId": output._uop.arg,
    "inputOps": [node.op for node in output._uop.inputs],
    "fillDtype": output._uop.inputs[1].dtype,
    "values": array.reshape(-1).tolist(),
    "gradient": source.grad.numpy().reshape(-1).tolist(),
    "ownsData": bool(array.flags["OWNDATA"]),
    "infinityCount": int(np.isneginf(infinite.numpy()).sum()),
    "dtypes": dtypes,
}
`);

    expect(result.op).toBe("WHERE");
    expect(result.contractId).toBe("browsergrad.jit.framework.tensor.masked-fill.v1");
    expect(result.inputOps).toEqual(["LOAD", "CONST", "LOAD"]);
    expect(result.fillDtype).toBe("float32");
    expect(result.values).toEqual(FRAMEWORK_MASKED_FILL_CONFORMANCE.valid.outputValues);
    expect(result.gradient).toEqual(FRAMEWORK_MASKED_FILL_CONFORMANCE.valid.sourceGradient);
    expect(result.ownsData).toBe(true);
    expect(result.infinityCount).toBe(4);
    expect(result.dtypes).toEqual(
      FRAMEWORK_MASKED_FILL_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
  });

  it("provides symbolic selection VJP, vmap, ONNX Where, and explicit device refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: boolean;
      gradient: number[];
      gradientOps: string[];
      mapped: number[];
      onnxWhereCount: number;
      halfOnnxError: string;
      planError: string;
      webgpuSupported: boolean;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_WHERE, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
mask = bg.from_numpy(np.array([True, False, True], dtype=np.bool_))
gradient = bg.func.grad(lambda value: value.masked_fill(mask, -1).sum())(source)
mapped = bg.func.vmap(lambda row: row.masked_fill(mask, -1))(source)
onnx = bg.onnx.export_inference(source.masked_fill(mask, -1), input_buffers=(source, mask))
half = bg.from_numpy(np.arange(1, 7, dtype=np.float16).reshape(2, 3))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "registered": get_rule(OP_WHERE) is not None,
    "gradient": gradient.numpy().reshape(-1).tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mapped": mapped.numpy().reshape(-1).tolist(),
    "onnxWhereCount": onnx.count(b"Where"),
    "halfOnnxError": error(lambda: bg.onnx.export_inference(
        half.masked_fill(mask, -1),
        input_buffers=(half, mask),
    )),
    "planError": error(lambda: bg.gpu_plan_summary(source.masked_fill(mask, -1))),
    "webgpuSupported": OP_WHERE in supported_opcodes(),
}
`);

    expect(result.registered).toBe(true);
    expect(result.gradient).toEqual(FRAMEWORK_MASKED_FILL_CONFORMANCE.valid.sourceGradient);
    expect(result.gradientOps).toContain("WHERE");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.mapped).toEqual(FRAMEWORK_MASKED_FILL_CONFORMANCE.valid.outputValues);
    expect(result.onnxWhereCount).toBe(1);
    expect(result.halfOnnxError).toMatch(/^OnnxUnmappableOp: .*float16/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*masked_fill/u);
    expect(result.webgpuSupported).toBe(false);
  });

  it("rejects hostile inputs and malformed WHERE at every admitted boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ errors: Record<string, string>; hostileCalls: number }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._framework_contracts import MASKED_FILL_CONTRACT_ID
from browsergrad_jit._ir import UOp, OP_CONST, OP_WHERE
from browsergrad_jit._realize import realize
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
mask = bg.from_numpy(np.array([True, False, True], dtype=np.bool_))
integer_mask = bg.from_numpy(np.array([1, 0, 1], dtype=np.int32))
fill = UOp(OP_CONST, (), (), "float32", arg={"value": -1.0})
wrong_fill = UOp(OP_CONST, (), (), "float64", arg={"value": -1.0})

def forged(condition=mask._uop, fill_node=fill, shape=(2, 3), arg=MASKED_FILL_CONTRACT_ID):
    return UOp(OP_WHERE, (condition, fill_node, source._uop), shape, "float32", arg=arg)

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

class HostileValue:
    calls = 0
    def __float__(self):
        HostileValue.calls += 1
        return -1.0
    def __int__(self):
        HostileValue.calls += 1
        return -1

errors = {
    "nonTensorMask": error(lambda: source.masked_fill([True, False, True], -1)),
    "maskDtype": error(lambda: source.masked_fill(integer_mask, -1)),
    "maskShape": error(lambda: source.masked_fill(bg.from_numpy(np.zeros((2, 2), dtype=np.bool_)), -1)),
    "expandingMask": error(lambda: source.masked_fill(bg.from_numpy(np.zeros((2, 2, 3), dtype=np.bool_)), -1)),
    "boolFill": error(lambda: source.masked_fill(mask, True)),
    "hostileFill": error(lambda: source.masked_fill(mask, HostileValue())),
    "integerFraction": error(lambda: bg.from_numpy(np.ones((2, 3), dtype=np.int32)).masked_fill(mask, 1.5)),
    "integerOverflow": error(lambda: bg.from_numpy(np.ones((2, 3), dtype=np.int32)).masked_fill(mask, 2147483648)),
    "cpuMaskDtype": error(lambda: realize(forged(condition=integer_mask._uop), source._get_session().buffer_table)),
    "cpuFillDtype": error(lambda: realize(forged(fill_node=wrong_fill), source._get_session().buffer_table)),
    "vjpShape": error(lambda: get_rule(OP_WHERE)(forged(shape=(3, 2)), forged().inputs, source._uop)),
    "vmapArg": error(lambda: get_vmap_rule(OP_WHERE)(forged(arg="forged"), {}, 2)),
    "onnxMaskDtype": error(lambda: bg.onnx.export_inference(
        bg.TensorProxy(forged(condition=integer_mask._uop), session=source._get_session()),
        input_buffers=(source, integer_mask),
    )),
    "planFillDtype": error(lambda: bg.gpu_plan_summary(
        bg.TensorProxy(forged(fill_node=wrong_fill), session=source._get_session())
    )),
}
{"errors": errors, "hostileCalls": HostileValue.calls}
`);

    expect(result.hostileCalls).toBe(0);
    expect(result.errors.nonTensorMask).toMatch(/^TypeError: .*mask must be/u);
    expect(result.errors.maskDtype).toMatch(/^ShapeError: .*mask dtype/u);
    expect(result.errors.maskShape).toMatch(/^ShapeError: .*cannot broadcast/u);
    expect(result.errors.expandingMask).toMatch(/^ShapeError: .*cannot broadcast/u);
    expect(result.errors.boolFill).toMatch(/^TypeError: .*real scalar/u);
    expect(result.errors.hostileFill).toMatch(/^TypeError: .*real scalar/u);
    expect(result.errors.integerFraction).toMatch(/^ValueError: .*exact finite integer/u);
    expect(result.errors.integerOverflow).toMatch(/^ValueError: .*out of range/u);
    expect(result.errors.cpuMaskDtype).toMatch(/^ShapeError: .*condition dtype/u);
    expect(result.errors.cpuFillDtype).toMatch(/^ShapeError: .*fill CONST dtype/u);
    expect(result.errors.vjpShape).toMatch(/^ShapeError: .*derived shape/u);
    expect(result.errors.vmapArg).toMatch(/^ShapeError: .*arg must be/u);
    expect(result.errors.onnxMaskDtype).toMatch(/^ShapeError: .*condition dtype/u);
    expect(result.errors.planFillDtype).toMatch(/^ShapeError: .*fill CONST dtype/u);
  });

  it("matches the shared eager/lazy masked-fill fixture", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_MASKED_FILL_CONFORMANCE);
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
source = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32).reshape(valid["inputShape"]))
mask = bg.from_numpy(np.asarray(valid["maskValues"], dtype=np.bool_))
output = source.masked_fill(mask, valid["fillValue"])

dtypes = []
for case in fixture["dtypeCases"]:
    dtype = case["dtype"]
    values = np.asarray(valid["inputValues"], dtype=np.dtype(dtype)).reshape(valid["inputShape"])
    fill = case.get("fillValue", valid["fillValue"])
    dtypes.append(bg.from_numpy(values).masked_fill(mask, fill).dtype)

errors = {}
for case in fixture["invalid"]:
    try:
        if case["kind"] == "mask-value":
            source.masked_fill(case["value"], -1)
        elif case["kind"] == "mask-dtype":
            source.masked_fill(bg.from_numpy(np.ones(valid["maskShape"], dtype=np.int32)), -1)
        elif case["kind"] == "mask-shape":
            source.masked_fill(bg.from_numpy(np.zeros(case["value"], dtype=np.bool_)), -1)
        elif case["kind"] == "fill":
            source.masked_fill(mask, case["value"])
        elif case["kind"] == "integer-fill":
            bg.from_numpy(np.ones(valid["inputShape"], dtype=np.int32)).masked_fill(mask, case["value"])
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

    expect(result.schema).toBe(FRAMEWORK_MASKED_FILL_CONFORMANCE.schema);
    expect(result.values).toEqual(FRAMEWORK_MASKED_FILL_CONFORMANCE.valid.outputValues);
    expect(result.dtypes).toEqual(
      FRAMEWORK_MASKED_FILL_CONFORMANCE.dtypeCases.map(({ expectedDtype }) => expectedDtype),
    );
    for (const invalid of FRAMEWORK_MASKED_FILL_CONFORMANCE.invalid) {
      expect(result.errors[invalid.id]).toContain(invalid.message);
    }
  });
});
