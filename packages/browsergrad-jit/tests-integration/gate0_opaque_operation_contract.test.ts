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

  it("separates the accelerator route from the user extension", async () => {
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

class RouteBridge:
    def flash_attention(self, *args):
        return "flash-ok"
    def run_user_kernel(self, *args):
        return "user-ok"

bridge = RouteBridge()
flash_vt = {id(node): f"flash-{i}" for i, node in enumerate(flash._uop.inputs)}
user_vt = {id(node): f"user-{i}" for i, node in enumerate(user._uop.inputs)}
nodes = [flash, user]
{
    "labels": [node._uop.arg["op"] for node in nodes],
    "requiresGrad": [node.requires_grad for node in nodes],
    "cpuErrors": [error_name(node.numpy) for node in nodes],
    "flashLegacyRoute": _h_custom(flash._uop, flash_vt, None, bridge, None),
    "userLegacyRoute": _h_custom(user._uop, user_vt, None, bridge, None),
    "functionalDisconnectedGradient": bg.func.grad(lambda value: bg.kernels.flash_attention(value, k, v).sum())(q).numpy().reshape(-1).tolist(),
}
`);
    expect(result).toEqual(expected("jit.custom.accelerator-and-user-extension.v0"));
  });

});
