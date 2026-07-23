import { readFileSync } from "node:fs";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

interface FixtureCase {
  readonly id: string;
  readonly expected: Record<string, unknown>;
}

const fixture = JSON.parse(readFileSync(new URL("./fixtures/jit-opaque-operation.v0.json", import.meta.url), "utf8")) as {
  readonly cases: readonly FixtureCase[];
};

function expected(id: string): Record<string, unknown> {
  const testCase = fixture.cases.find((candidate) => candidate.id === id);
  if (testCase === undefined) throw new Error(`Missing JIT opaque-operation fixture case ${id}`);
  return testCase.expected;
}

describe("Gate 0 JIT opaque-operation contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("pins callback realization, closure backward, and portable-path refusals", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
from browsergrad_jit import _vjp, _vmap
from browsergrad_jit._realize_webgpu import _h_custom, realize_tensor_plan_webgpu

def error_name(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__

x = bg.from_numpy(np.zeros((1, 1, 1, 2), dtype=np.float32), requires_grad=True)
y = F.interpolate(x, size=(1, 4), mode="nearest")
cpu_values = y.numpy().tolist()
y.sum().backward()
custom_plan = bg.gpu_plan_summary(y, allow_custom=True)

class PlanOnlyGpuBuffers:
    bridge = object()

{
    "label": y._uop.arg["name"],
    "labelField": "name" if "name" in y._uop.arg else "op",
    "cpuValues": cpu_values,
    "cpuGradient": x.grad.numpy().tolist(),
    "symbolicVjpRegistered": _vjp.get_rule("CUSTOM") is not None,
    "functionalGradError": error_name(lambda: bg.func.grad(
        lambda value: F.interpolate(value, size=(1, 4), mode="nearest").sum()
    )(x)),
    "vmapError": error_name(lambda: _vmap._VMAP_RULES["CUSTOM"](y._uop, {}, 2)),
    "onnxError": error_name(lambda: bg.onnx.export_inference(y, input_buffers=(x,))),
    "tensorPlanError": error_name(lambda: bg.gpu_plan_summary(y)),
    "tensorPlanAllowCustom": {
        "hasCustomOps": custom_plan["has_custom_ops"],
        "rootOp": custom_plan["ops"][-1],
    },
    "tensorPlanExecutionError": error_name(lambda: realize_tensor_plan_webgpu(
        y._uop,
        numpy_buffer_table=None,
        gpu_buffer_table=PlanOnlyGpuBuffers(),
    )),
    "legacyWebgpuError": error_name(lambda: _h_custom(
        y._uop,
        {id(node): f"input-{index}" for index, node in enumerate(y._uop.inputs)},
        None,
        object(),
        None,
    )),
}
`);
    expect(result).toEqual(expected("jit.custom.shared-refusals.v0"));
  });

  it("pins the remaining conditional backward refusal", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np

nearest_input = bg.from_numpy(np.array([[[[2.0]]]], dtype=np.float32), requires_grad=True)
F.interpolate(nearest_input, size=(2, 2), mode="nearest").sum().backward()

bilinear_input = bg.from_numpy(np.array([[[[2.0]]]], dtype=np.float32), requires_grad=True)
try:
    F.interpolate(bilinear_input, size=(2, 2), mode="bilinear").sum().backward()
    bilinear_error = "no_error"
except Exception as exc:
    bilinear_error = type(exc).__name__

{
    "nearestGradient": nearest_input.grad.numpy().reshape(-1).tolist(),
    "bilinearBackwardError": bilinear_error,
}
`);
    expect(result).toEqual(expected("jit.custom.conditional-backward-refusal.v0"));
  });

  it("separates accelerator-only routes from constructor-only labels", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._realize_webgpu import _h_custom

def error_name(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__

q = bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32), requires_grad=True)
k = bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32))
v = bg.from_numpy(np.ones((1, 1, 2, 4), dtype=np.float32))
flash = bg.kernels.flash_attention(q, k, v)

user_builder = bg.custom_kernel(
    wgsl="@compute @workgroup_size(1) fn main() {}",
    name="gate0_user",
    workgroup_size=(1, 1, 1),
    output_shape_fn=lambda shape: shape,
    dispatch_shape_fn=lambda shape: (1, 1, 1),
    num_inputs=1,
)
user = user_builder(q)

left = bg.from_numpy(np.ones((2, 3), dtype=np.float32))
right = bg.from_numpy(np.ones((3, 4), dtype=np.float32))
webnn = bg.experimental.webnn.matmul(left, right)

x = bg.from_numpy(np.ones((1, 2, 4), dtype=np.float32))
w_qkv = bg.from_numpy(np.ones((4, 12), dtype=np.float32))
w_o = bg.from_numpy(np.ones((4, 4), dtype=np.float32))
w_ff1 = bg.from_numpy(np.ones((4, 16), dtype=np.float32))
w_ff2 = bg.from_numpy(np.ones((16, 4), dtype=np.float32))
transformer = bg.kernels.transformer_block(x, w_qkv, w_o, w_ff1, w_ff2, num_heads=1)

class RouteBridge:
    def flash_attention(self, *args):
        return "flash-ok"
    def run_user_kernel(self, *args):
        return "user-ok"

bridge = RouteBridge()
flash_vt = {id(node): f"flash-{i}" for i, node in enumerate(flash._uop.inputs)}
user_vt = {id(node): f"user-{i}" for i, node in enumerate(user._uop.inputs)}
webnn_vt = {id(node): f"webnn-{i}" for i, node in enumerate(webnn._uop.inputs)}
transformer_vt = {id(node): f"transformer-{i}" for i, node in enumerate(transformer._uop.inputs)}

nodes = [flash, transformer, user, webnn]
{
    "labels": [node._uop.arg["op"] for node in nodes],
    "requiresGrad": [node.requires_grad for node in nodes],
    "cpuErrors": [error_name(node.numpy) for node in nodes],
    "flashLegacyRoute": _h_custom(flash._uop, flash_vt, None, bridge, None),
    "userLegacyRoute": _h_custom(user._uop, user_vt, None, bridge, None),
    "webnnLegacyError": error_name(lambda: _h_custom(webnn._uop, webnn_vt, None, bridge, None)),
    "transformerLegacyError": error_name(lambda: _h_custom(transformer._uop, transformer_vt, None, bridge, None)),
    "functionalDisconnectedGradient": bg.func.grad(lambda value: bg.kernels.flash_attention(value, k, v).sum())(q).numpy().reshape(-1).tolist(),
}
`);
    expect(result).toEqual(expected("jit.custom.accelerator-and-constructor-only.v0"));
  });

  it("executes every NumPy callback label and exposes realized dtype drift", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import numpy as np
import pyodide

observed = {}
gradient_present = {}

def leaf(values, dtype=np.float32):
    return bg.from_numpy(np.asarray(values, dtype=dtype), requires_grad=True)

def error_name(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__

def check(label, output, source):
    observed[label] = output._uop.arg.get("name")
    output.numpy()
    output.sum().backward()
    gradient_present[label] = source.grad is not None

a = leaf([[[[2.0]]]]); check("interpolate", F.interpolate(a, size=(2, 2), mode="nearest"), a)
{
    "environment": {"pyodide": pyodide.__version__, "numpy": np.__version__},
    "observedLabels": sorted(observed.keys()),
    "allLabelsMatch": all(observed[label] == label for label in observed),
    "closureGradientLabels": sorted(label for label, present in gradient_present.items() if present),
    "allClosureGradientsPresent": all(gradient_present.values()),
    "realizedDtypeDrift": {},
}
`);
    expect(result).toEqual(expected("jit.custom.all-callback-labels-and-dtypes.v0"));
  });
});
