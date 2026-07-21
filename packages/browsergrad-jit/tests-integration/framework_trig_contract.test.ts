import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed sin/cos framework contracts", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("preserves floating dtypes, owning CPU results, values, and closure gradients", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      sinOp: string;
      cosOp: string;
      sinValues: number[];
      cosValues: number[];
      sinOwnsData: boolean;
      cosOwnsData: boolean;
      sinGradient: number[];
      cosGradient: number[];
      dtypePairs: string[][];
    }>(`
import browsergrad_jit as bg
import numpy as np

angles = np.array([-np.pi / 2, 0.0, np.pi / 2], dtype=np.float32)
sin_source = bg.from_numpy(angles, requires_grad=True)
sin_result = sin_source.sin()
sin_array = sin_result.numpy()
sin_result.sum().backward()

cos_source = bg.from_numpy(angles, requires_grad=True)
cos_result = cos_source.cos()
cos_array = cos_result.numpy()
cos_result.sum().backward()

dtype_pairs = []
for dtype in (np.float16, np.float32, np.float64):
    source = bg.from_numpy(np.array([0.25], dtype=dtype))
    dtype_pairs.append([str(source.sin().numpy().dtype), str(source.cos().numpy().dtype)])

{
    "sinOp": sin_result._uop.op,
    "cosOp": cos_result._uop.op,
    "sinValues": sin_array.tolist(),
    "cosValues": cos_array.tolist(),
    "sinOwnsData": bool(sin_array.flags["OWNDATA"]),
    "cosOwnsData": bool(cos_array.flags["OWNDATA"]),
    "sinGradient": sin_source.grad.numpy().tolist(),
    "cosGradient": cos_source.grad.numpy().tolist(),
    "dtypePairs": dtype_pairs,
}
`);

    expect(result.sinOp).toBe("SIN");
    expect(result.cosOp).toBe("COS");
    expect(result.sinValues[0]).toBeCloseTo(-1, 6);
    expect(result.sinValues[1]).toBeCloseTo(0, 6);
    expect(result.sinValues[2]).toBeCloseTo(1, 6);
    expect(result.cosValues[0]).toBeCloseTo(0, 6);
    expect(result.cosValues[1]).toBeCloseTo(1, 6);
    expect(result.cosValues[2]).toBeCloseTo(0, 6);
    expect(result.sinOwnsData).toBe(true);
    expect(result.cosOwnsData).toBe(true);
    expect(result.sinGradient[0]).toBeCloseTo(0, 6);
    expect(result.sinGradient[1]).toBeCloseTo(1, 6);
    expect(result.sinGradient[2]).toBeCloseTo(0, 6);
    expect(result.cosGradient[0]).toBeCloseTo(1, 6);
    expect(result.cosGradient[1]).toBeCloseTo(0, 6);
    expect(result.cosGradient[2]).toBeCloseTo(-1, 6);
    expect(result.dtypePairs).toEqual([
      ["float16", "float16"],
      ["float32", "float32"],
      ["float64", "float64"],
    ]);
  });

  it("provides typed symbolic VJP, functional-grad, vmap, ONNX, and explicit plan decisions", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      registered: string[];
      sinGradient: number[];
      cosGradient: number[];
      sinGradientOps: string[];
      cosGradientOps: string[];
      mappedSin: number[][];
      mappedCos: number[][];
      onnxHasSin: boolean;
      onnxHasCos: boolean;
      onnxHasCustom: boolean;
      planErrors: Record<string, string>;
    }>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import OP_COS, OP_SIN, toposort
from browsergrad_jit._vjp import get_rule

source = bg.from_numpy(np.array([-0.5, 0.0, 0.75], dtype=np.float32))
sin_gradient_tensor = bg.func.grad(lambda value: value.sin().sum())(source)
cos_gradient_tensor = bg.func.grad(lambda value: value.cos().sum())(source)

batched = bg.from_numpy(np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float32))
mapped_sin = bg.func.vmap(lambda row: row.sin())(batched).numpy()
mapped_cos = bg.func.vmap(lambda row: row.cos())(batched).numpy()

sin_onnx = bg.onnx.export_inference(source.sin(), input_buffers=(source,))
cos_onnx = bg.onnx.export_inference(source.cos(), input_buffers=(source,))

def plan_error(tensor):
    try:
        bg.gpu_plan_summary(tensor)
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "registered": [
        name for name, op in (("COS", OP_COS), ("SIN", OP_SIN))
        if get_rule(op) is not None
    ],
    "sinGradient": sin_gradient_tensor.numpy().tolist(),
    "cosGradient": cos_gradient_tensor.numpy().tolist(),
    "sinGradientOps": sorted({node.op for node in toposort(sin_gradient_tensor._uop)}),
    "cosGradientOps": sorted({node.op for node in toposort(cos_gradient_tensor._uop)}),
    "mappedSin": mapped_sin.tolist(),
    "mappedCos": mapped_cos.tolist(),
    "onnxHasSin": b"Sin" in sin_onnx,
    "onnxHasCos": b"Cos" in cos_onnx,
    "onnxHasCustom": b"CUSTOM" in sin_onnx or b"CUSTOM" in cos_onnx,
    "planErrors": {
        "sin": plan_error(source.sin()),
        "cos": plan_error(source.cos()),
    },
}
`);

    expect(result.registered).toEqual(["COS", "SIN"]);
    for (const [actual, expected] of result.sinGradient.map((value, index) =>
      [value, Math.cos([-0.5, 0, 0.75][index]!)] as const)) {
      expect(actual).toBeCloseTo(expected, 6);
    }
    for (const [actual, expected] of result.cosGradient.map((value, index) =>
      [value, -Math.sin([-0.5, 0, 0.75][index]!)] as const)) {
      expect(actual).toBeCloseTo(expected, 6);
    }
    expect(result.sinGradientOps).toContain("COS");
    expect(result.sinGradientOps).not.toContain("CUSTOM");
    expect(result.cosGradientOps).toContain("SIN");
    expect(result.cosGradientOps).not.toContain("CUSTOM");
    expect(result.mappedSin[0]?.[0]).toBeCloseTo(0, 6);
    expect(result.mappedSin[1]?.[0]).toBeCloseTo(Math.sin(-0.5), 6);
    expect(result.mappedCos[0]?.[1]).toBeCloseTo(Math.cos(0.5), 6);
    expect(result.mappedCos[1]?.[1]).toBeCloseTo(Math.cos(1), 6);
    expect(result.onnxHasSin).toBe(true);
    expect(result.onnxHasCos).toBe(true);
    expect(result.onnxHasCustom).toBe(false);
    expect(result.planErrors.sin).toMatch(/^GpuPlanUnsupported: .*SIN/u);
    expect(result.planErrors.cos).toMatch(/^GpuPlanUnsupported: .*COS/u);
  });

  it("rejects non-floating inputs and contract mutation at every consuming boundary", async () => {
    const target = await getJitTarget();
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_COS, OP_SIN
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.array([-0.5, 0.0, 0.75], dtype=np.float32))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

mutated = source.sin()
mutated._uop.arg["backend"] = "webgpu"
dy = bg.from_numpy(np.ones(source.shape, dtype=np.float32))._uop

{
    "boolSin": error(lambda: bg.from_numpy(np.array([True], dtype=np.bool_)).sin()),
    "intSin": error(lambda: bg.from_numpy(np.array([1], dtype=np.int32)).sin()),
    "intCos": error(lambda: bg.from_numpy(np.array([1], dtype=np.int64)).cos()),
    "wrongInputs": error(lambda: UOp(OP_SIN, [source._uop], source.shape, "float32", arg={})),
    "wrongShape": error(lambda: UOp(OP_SIN, (source._uop,), (1, 3), "float32", arg={})),
    "wrongDtype": error(lambda: UOp(OP_COS, (source._uop,), source.shape, "float64", arg={})),
    "openArgCpu": error(lambda: mutated.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_SIN)(mutated._uop, mutated._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_SIN)(
        mutated._uop,
        {id(mutated._uop.inputs[0]): mutated._uop.inputs[0]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated, input_buffers=(source,))),
}
`);

    expect(errors.boolSin).toMatch(/^ShapeError: .*floating dtypes only/u);
    expect(errors.intSin).toMatch(/^ShapeError: .*floating dtypes only/u);
    expect(errors.intCos).toMatch(/^ShapeError: .*floating dtypes only/u);
    expect(errors.wrongInputs).toMatch(/^ShapeError: .*inputs must be a plain tuple/u);
    expect(errors.wrongShape).toMatch(/^ShapeError: .*preserve its input shape/u);
    expect(errors.wrongDtype).toMatch(/^ShapeError: .*preserve its input dtype/u);
    expect(errors.openArgCpu).toMatch(/^RealizationError: .*arg fields/u);
    expect(errors.openArgVjp).toMatch(/^ShapeError: .*arg fields/u);
    expect(errors.openArgVmap).toMatch(/^ShapeError: .*arg fields/u);
    expect(errors.openArgOnnx).toMatch(/^ShapeError: .*arg fields/u);
  });
});
