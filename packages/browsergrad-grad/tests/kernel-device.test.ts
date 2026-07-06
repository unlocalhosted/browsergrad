import { describe, expect, it } from "vitest";
import { createGradKernelDeviceBridge } from "../src/kernel-device";
import type { KernelDevice, Tensor as KernelTensor } from "@unlocalhosted/browsergrad-kernels";

describe("createGradKernelDeviceBridge", () => {
  it("adapts Python flat lists/shapes to kernels Tensor calls", async () => {
    const calls: Array<{ name: string; shapes: readonly (readonly number[])[] }> = [];
    const bridge = createGradKernelDeviceBridge({} as KernelDevice, {
      async matmul(_device, a, b) {
        calls.push({ name: "matmul", shapes: [a.shape, b.shape] });
        return tensor([2, 2], [19, 22, 43, 50]);
      },
      async softmax(_device, x) {
        calls.push({ name: "softmax", shapes: [x.shape] });
        return tensor(x.shape, [0.25, 0.75, 0.5, 0.5]);
      },
      async layernorm(_device, x, opts) {
        calls.push({
          name: "layernorm",
          shapes: [x.shape, opts?.gamma?.shape ?? [], opts?.beta?.shape ?? []],
        });
        expect(opts?.eps).toBe(0.001);
        return tensor(x.shape, [1, 2, 3, 4]);
      },
      async attention(_device, q, k, v) {
        calls.push({ name: "attention", shapes: [q.shape, k.shape, v.shape] });
        return tensor([2, 2], [1, 0, 0, 1]);
      },
    });

    await expect(bridge.matmul([1, 2, 3, 4], [2, 2], [5, 6, 7, 8], [2, 2]))
      .resolves.toEqual([19, 22, 43, 50]);
    await expect(bridge.softmax([1, 2, 3, 4], [2, 2]))
      .resolves.toEqual([0.25, 0.75, 0.5, 0.5]);
    await expect(bridge.layernorm([1, 2, 3, 4], [2, 2], [1, 1], [0, 0], 0.001))
      .resolves.toEqual([1, 2, 3, 4]);
    await expect(bridge.attention([1, 0, 0, 1], [2, 2], [1, 0, 0, 1], [2, 2], [1, 0, 0, 1], [2, 2]))
      .resolves.toEqual([1, 0, 0, 1]);

    expect(calls).toEqual([
      { name: "matmul", shapes: [[2, 2], [2, 2]] },
      { name: "softmax", shapes: [[2, 2]] },
      { name: "layernorm", shapes: [[2, 2], [2], [2]] },
      { name: "attention", shapes: [[2, 2], [2, 2], [2, 2]] },
    ]);
  });
});

function tensor(shape: readonly number[], values: readonly number[]): KernelTensor {
  return { shape, data: Float32Array.from(values) };
}
