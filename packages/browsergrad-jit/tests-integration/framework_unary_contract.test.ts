import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed abs/sign framework contracts", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("preserves values, dtypes, owning CPU results, and closure gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      absOp: string;
      signOp: string;
      absValues: number[];
      signValues: number[];
      absOwnsData: boolean;
      signOwnsData: boolean;
      absGradient: number[];
      signGradient: number[];
      integerAbsDtype: string;
      integerAbsValues: number[];
      integerSignDtype: string;
      integerSignValues: number[];
    }>(`
import browsergrad_jit as bg
import numpy as np

abs_source = bg.from_numpy(np.array([-2.0, 0.0, 3.0], dtype=np.float32), requires_grad=True)
abs_result = abs_source.abs()
abs_array = abs_result.numpy()
abs_result.sum().backward()

sign_source = bg.from_numpy(np.array([-2.0, 0.0, 3.0], dtype=np.float32), requires_grad=True)
sign_result = sign_source.sign()
sign_array = sign_result.numpy()
sign_result.sum().backward()

integer_source = bg.from_numpy(np.array([-4, 0, 7], dtype=np.int32))
integer_abs = integer_source.abs().numpy()
integer_sign = integer_source.sign().numpy()

{
    "absOp": abs_result._uop.op,
    "signOp": sign_result._uop.op,
    "absValues": abs_array.tolist(),
    "signValues": sign_array.tolist(),
    "absOwnsData": bool(abs_array.flags["OWNDATA"]),
    "signOwnsData": bool(sign_array.flags["OWNDATA"]),
    "absGradient": abs_source.grad.numpy().tolist(),
    "signGradient": sign_source.grad.numpy().tolist(),
    "integerAbsDtype": str(integer_abs.dtype),
    "integerAbsValues": integer_abs.tolist(),
    "integerSignDtype": str(integer_sign.dtype),
    "integerSignValues": integer_sign.tolist(),
}
`);

    expect(result).toEqual({
      absOp: "ABS",
      signOp: "SIGN",
      absValues: [2, 0, 3],
      signValues: [-1, 0, 1],
      absOwnsData: true,
      signOwnsData: true,
      absGradient: [-1, 0, 1],
      signGradient: [0, 0, 0],
      integerAbsDtype: "int32",
      integerAbsValues: [4, 0, 7],
      integerSignDtype: "int32",
      integerSignValues: [-1, 0, 1],
    });
  });

  it("provides symbolic VJP, functional-grad, vmap, ONNX, and explicit plan decisions", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: string[];
      absGradient: number[];
      signGradient: number[];
      mappedAbs: number[][];
      mappedSign: number[][];
      onnxHasAbs: boolean;
      onnxHasSign: boolean;
      onnxHasCustom: boolean;
      planErrors: Record<string, string>;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_ABS, OP_SIGN
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([-2.0, 0.0, 3.0], dtype=np.float32))
abs_gradient = bg.func.grad(lambda value: value.abs().sum())(source).numpy()
sign_gradient = bg.func.grad(lambda value: value.sign().sum())(source).numpy()

batched = bg.from_numpy(np.array([[-2.0, 0.0], [3.0, -4.0]], dtype=np.float32))
mapped_abs = bg.func.vmap(lambda row: row.abs())(batched).numpy()
mapped_sign = bg.func.vmap(lambda row: row.sign())(batched).numpy()

abs_onnx = bg.onnx.export_inference(source.abs(), input_buffers=(source,))
sign_onnx = bg.onnx.export_inference(source.sign(), input_buffers=(source,))

def plan_error(tensor):
    try:
        bg.gpu_plan_summary(tensor)
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "registered": [
        name for name, op in (("ABS", OP_ABS), ("SIGN", OP_SIGN))
        if get_rule(op) is not None
    ],
    "absGradient": abs_gradient.tolist(),
    "signGradient": sign_gradient.tolist(),
    "mappedAbs": mapped_abs.tolist(),
    "mappedSign": mapped_sign.tolist(),
    "onnxHasAbs": b"Abs" in abs_onnx,
    "onnxHasSign": b"Sign" in sign_onnx,
    "onnxHasCustom": b"CUSTOM" in abs_onnx or b"CUSTOM" in sign_onnx,
    "planErrors": {
        "abs": plan_error(source.abs()),
        "sign": plan_error(source.sign()),
    },
}
`);

    expect(result.registered).toEqual(["ABS", "SIGN"]);
    expect(result.absGradient).toEqual([-1, 0, 1]);
    expect(result.signGradient).toEqual([0, 0, 0]);
    expect(result.mappedAbs).toEqual([[2, 0], [3, 4]]);
    expect(result.mappedSign).toEqual([[-1, 0], [1, -1]]);
    expect(result.onnxHasAbs).toBe(true);
    expect(result.onnxHasSign).toBe(true);
    expect(result.onnxHasCustom).toBe(false);
    expect(result.planErrors.abs).toMatch(/^GpuPlanUnsupported: .*ABS/u);
    expect(result.planErrors.sign).toMatch(/^GpuPlanUnsupported: .*SIGN/u);
  });

  it("rejects bool inputs and contract mutation at construction and every consuming boundary", async () => {
    const target = await getJitTarget();
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_ABS, OP_SIGN
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.array([-2.0, 0.0, 3.0], dtype=np.float32))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

mutated = source.abs()
mutated._uop.arg["backend"] = "webgpu"
dy = bg.from_numpy(np.ones(source.shape, dtype=np.float32))._uop

{
    "boolAbs": error(lambda: bg.from_numpy(np.array([True], dtype=np.bool_)).abs()),
    "boolSign": error(lambda: bg.from_numpy(np.array([True], dtype=np.bool_)).sign()),
    "wrongShape": error(lambda: UOp(OP_ABS, (source._uop,), (1, 3), "float32", arg={})),
    "wrongDtype": error(lambda: UOp(OP_SIGN, (source._uop,), source.shape, "int32", arg={})),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_ABS)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_ABS)(
        mutated._uop,
        {id(mutated._uop.inputs[0]): mutated._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(source,))),
}
`);

    expect(errors.boolAbs).toMatch(/^ShapeError: .*real numeric dtypes only/u);
    expect(errors.boolSign).toMatch(/^ShapeError: .*real numeric dtypes only/u);
    expect(errors.wrongShape).toMatch(/^ShapeError: .*preserve its input shape/u);
    expect(errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(errors.openArgCpu).toMatch(/^RealizationError: .*arg fields/u);
    expect(errors.openArgVjp).toMatch(/^ShapeError: .*arg fields/u);
    expect(errors.openArgVmap).toMatch(/^ShapeError: .*arg fields/u);
    expect(errors.openArgOnnx).toMatch(/^ShapeError: .*arg fields/u);
  });
});
