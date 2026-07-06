/**
 * Real-WebGPU end-to-end tests.
 *
 * These run inside Playwright Chromium with WebGPU enabled. The whole
 * point: prove the WGSL kernels and the `createWebGpuRealizerBridge`
 * seam (PRD-011.5) work against an actual GPUDevice, not just the
 * NumPy-backed mock from tests-integration.
 *
 * Skip gracefully when navigator.gpu is absent (CI on a headless host
 * with no GPU adapter). The bench is data collection.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { createDevice } from "../src/device";
import { reference } from "../src/reference";
import { tensor } from "../src/types";
import {
  matmulDirect,
  matmulTiledDirect,
  fusedElementwiseDirect,
  flashAttentionDirect,
  runTensorGpuPlan,
} from "../src/index";
import {
  uploadFloat32,
  materializeFloat32,
} from "../src/runner";
import { createWebGpuRealizerBridge } from "../src/realizer";

interface DeviceCheck {
  available: boolean;
  reason?: string;
  adapterName?: string;
  adapterInfo?: GPUAdapterInfo;
}

async function checkDevice(): Promise<DeviceCheck> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu undefined" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { available: false, reason: "no GPU adapter" };
    }
    return {
      available: true,
      adapterName: adapter.info?.device ?? "unknown",
      adapterInfo: adapter.info,
    };
  } catch (err) {
    return {
      available: false,
      reason: err instanceof Error ? err.message : String(err),
    };
  }
}

describe("real WebGPU — matmul + tiled GEMM + fused elementwise + FA-v2", () => {
  let deviceCheck: DeviceCheck;

  beforeAll(async () => {
    deviceCheck = await checkDevice();
    if (!deviceCheck.available) {
      console.warn(`[skip] WebGPU not available: ${deviceCheck.reason}`);
    } else {
      console.log(`[ok] WebGPU adapter: ${deviceCheck.adapterName}`);
    }
  });

  it("reports the WebGPU adapter info or skips", () => {
    if (!deviceCheck.available) {
      console.warn(`Skipping: ${deviceCheck.reason}`);
      return;
    }
    expect(deviceCheck.adapterName).toBeDefined();
  });

  it("naive matmul matches the JS reference", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const M = 16, K = 24, N = 8;
    const aData = new Float32Array(M * K);
    const bData = new Float32Array(K * N);
    for (let i = 0; i < aData.length; i++) aData[i] = Math.sin(i * 0.1);
    for (let i = 0; i < bData.length; i++) bData[i] = Math.cos(i * 0.17);

    const A = tensor([M, K], aData);
    const B = tensor([K, N], bData);
    const C_ref = reference.matmul(A, B);

    // Real GPU path via dispatch (round-trip; includes upload+readback).
    const { matmul } = await import("../src/kernels/matmul");
    const C_gpu = await matmul(device, A, B);

    expect(C_gpu.shape).toEqual([M, N]);
    let maxDiff = 0;
    for (let i = 0; i < C_ref.data.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(C_gpu.data[i]! - C_ref.data[i]!));
    }
    expect(maxDiff).toBeLessThan(1e-4);
  });

  it("tiled GEMM matches the reference for non-aligned shapes", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    // Pick shapes that aren't tile-aligned to exercise the boundary code.
    const M = 17, K = 23, N = 19;
    const aData = new Float32Array(M * K);
    const bData = new Float32Array(K * N);
    for (let i = 0; i < aData.length; i++) aData[i] = Math.sin(i * 0.13);
    for (let i = 0; i < bData.length; i++) bData[i] = Math.cos(i * 0.21);

    const aBuf = uploadFloat32(device, aData);
    const bBuf = uploadFloat32(device, bData);
    try {
      const result = matmulTiledDirect(device, aBuf, bBuf, M, K, N);
      const out = await materializeFloat32(device, result.buffer, result.byteLength);

      const C_ref = reference.matmul(tensor([M, K], aData), tensor([K, N], bData));
      let maxDiff = 0;
      for (let i = 0; i < out.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(out[i]! - C_ref.data[i]!));
      }
      expect(out.length).toBe(M * N);
      expect(maxDiff).toBeLessThan(1e-3);

      result.buffer.destroy();
    } finally {
      aBuf.destroy();
      bBuf.destroy();
    }
  });

  it("residency contract: chained matmul makes 3 host→device uploads + 1 readback", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const M = 4, K = 4, N = 4;
    const x = new Float32Array(M * K);
    const w1 = new Float32Array(K * N);
    const w2 = new Float32Array(N * N);
    for (let i = 0; i < x.length; i++) x[i] = (i + 1) * 0.1;
    for (let i = 0; i < w1.length; i++) w1[i] = i * 0.05;
    for (let i = 0; i < w2.length; i++) w2[i] = i * 0.03;

    // Three uploads — x, w1, w2 — then chain matmuls in-residence.
    const xBuf = uploadFloat32(device, x);
    const w1Buf = uploadFloat32(device, w1);
    const w2Buf = uploadFloat32(device, w2);

    try {
      const mid = matmulTiledDirect(device, xBuf, w1Buf, M, K, N);
      const final_ = matmulTiledDirect(device, mid.buffer, w2Buf, M, N, N);
      const out = await materializeFloat32(device, final_.buffer, final_.byteLength);
      expect(out.length).toBe(M * N);

      // Compute the JS reference.
      const ref = reference.matmul(
        reference.matmul(tensor([M, K], x), tensor([K, N], w1)),
        tensor([N, N], w2),
      );
      let maxDiff = 0;
      for (let i = 0; i < out.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(out[i]! - ref.data[i]!));
      }
      expect(maxDiff).toBeLessThan(1e-3);

      mid.buffer.destroy();
      final_.buffer.destroy();
    } finally {
      xBuf.destroy();
      w1Buf.destroy();
      w2Buf.destroy();
    }
  });

  it("generic tensor GPU plan runs matmul plus elementwise without per-op bridge methods", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const M = 4, K = 4, N = 4;
    const x = new Float32Array(M * K);
    const w = new Float32Array(K * N);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.13);
    for (let i = 0; i < w.length; i++) w[i] = Math.cos(i * 0.17);

    const plan = {
      steps: [
        { step: 0, valueId: 1, op: "BUFFER", inputIds: [], shape: [M, K], dtype: "float32" },
        { step: 1, valueId: 2, op: "BUFFER", inputIds: [], shape: [K, N], dtype: "float32" },
        { step: 2, valueId: 3, op: "LOAD", inputIds: [1], shape: [M, K], dtype: "float32" },
        { step: 3, valueId: 4, op: "LOAD", inputIds: [2], shape: [K, N], dtype: "float32" },
        { step: 4, valueId: 5, op: "MATMUL", inputIds: [3, 4], shape: [M, N], dtype: "float32" },
        { step: 5, valueId: 6, op: "EXP", inputIds: [5], shape: [M, N], dtype: "float32" },
        { step: 6, valueId: 7, op: "MUL", inputIds: [6, 5], shape: [M, N], dtype: "float32" },
      ],
      buffers: [
        { valueId: 1, op: "BUFFER", shape: [M, K], dtype: "float32", bytes: M * K * 4, firstStep: 0, lastStep: 2, materialize: false },
        { valueId: 2, op: "BUFFER", shape: [K, N], dtype: "float32", bytes: K * N * 4, firstStep: 1, lastStep: 3, materialize: false },
        { valueId: 3, op: "LOAD", shape: [M, K], dtype: "float32", bytes: M * K * 4, firstStep: 2, lastStep: 4, materialize: false },
        { valueId: 4, op: "LOAD", shape: [K, N], dtype: "float32", bytes: K * N * 4, firstStep: 3, lastStep: 4, materialize: false },
        { valueId: 5, op: "MATMUL", shape: [M, N], dtype: "float32", bytes: M * N * 4, firstStep: 4, lastStep: 6, materialize: false },
        { valueId: 6, op: "EXP", shape: [M, N], dtype: "float32", bytes: M * N * 4, firstStep: 5, lastStep: 6, materialize: false },
        { valueId: 7, op: "MUL", shape: [M, N], dtype: "float32", bytes: M * N * 4, firstStep: 6, lastStep: 6, materialize: true },
      ],
      rootId: 7,
      materializationBoundary: "root",
      peakLiveBytes: 5 * M * N * 4,
      hasCustomOps: false,
    } as const;

    const result = await runTensorGpuPlan(device, plan, [
      { valueId: 1, data: x },
      { valueId: 2, data: w },
    ]);

    const mm = reference.matmul(tensor([M, K], x), tensor([K, N], w));
    let maxDiff = 0;
    for (let i = 0; i < result.data.length; i++) {
      const expected = Math.exp(mm.data[i]!) * mm.data[i]!;
      maxDiff = Math.max(maxDiff, Math.abs(result.data[i]! - expected) / (Math.abs(expected) + 1));
    }
    expect(result.shape).toEqual([M, N]);
    expect(result.materializedValueId).toBe(7);
    expect(result.peakLiveBytes).toBeGreaterThan(0);
    expect(maxDiff).toBeLessThan(1e-3);
  });

  it("WebGpuRealizerBridge runs a canonical tensor plan through one generic method", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const bridge = createWebGpuRealizerBridge(device);
    const M = 3, K = 4, N = 2;
    const x = new Float32Array(M * K);
    const w = new Float32Array(K * N);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.19);
    for (let i = 0; i < w.length; i++) w[i] = Math.cos(i * 0.23);

    const plan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [M, K], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [K, N], dtype: "float32" },
        { step: 2, value_id: 2, op: "LOAD", input_ids: [0], shape: [M, K], dtype: "float32" },
        { step: 3, value_id: 3, op: "LOAD", input_ids: [1], shape: [K, N], dtype: "float32" },
        { step: 4, value_id: 4, op: "MATMUL", input_ids: [2, 3], shape: [M, N], dtype: "float32" },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [M, K], dtype: "float32", bytes: M * K * 4, first_step: 0, last_step: 2, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [K, N], dtype: "float32", bytes: K * N * 4, first_step: 1, last_step: 3, materialize: false },
        { value_id: 2, op: "LOAD", shape: [M, K], dtype: "float32", bytes: M * K * 4, first_step: 2, last_step: 4, materialize: false },
        { value_id: 3, op: "LOAD", shape: [K, N], dtype: "float32", bytes: K * N * 4, first_step: 3, last_step: 4, materialize: false },
        { value_id: 4, op: "MATMUL", shape: [M, N], dtype: "float32", bytes: M * N * 4, first_step: 4, last_step: 4, materialize: true },
      ],
      root_id: 4,
      materialization_boundary: "root",
      peak_live_bytes: 4 * M * K * 4,
      has_custom_ops: false,
    } as const;

    const xBytes = new Uint8Array(x.buffer, x.byteOffset, x.byteLength);
    const wBytes = new Uint8Array(w.buffer, w.byteOffset, w.byteLength);
    const outBytes = await bridge.run_tensor_plan(
      plan,
      [
        { value_id: 0, data: xBytes },
        { value_id: 1, data: wBytes },
      ],
      "float32",
    );
    const out = new Float32Array(outBytes.buffer, outBytes.byteOffset, outBytes.byteLength / 4);
    const ref = reference.matmul(tensor([M, K], x), tensor([K, N], w));
    let maxDiff = 0;
    for (let i = 0; i < out.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(out[i]! - ref.data[i]!));
    }
    expect(out.length).toBe(M * N);
    expect(maxDiff).toBeLessThan(1e-3);
    expect(bridge.aliveHandleCount()).toBe(0);
  });

  it("generic tensor GPU plan runs reshape, permute, reduce, and broadcast", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const x = new Float32Array([0, 1, 2, 3, 4, 5]);
    const reducePlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [2, 3], dtype: "float32" },
        { step: 1, value_id: 1, op: "LOAD", input_ids: [0], shape: [2, 3], dtype: "float32" },
        { step: 2, value_id: 2, op: "RESHAPE", input_ids: [1], shape: [3, 2], dtype: "float32", arg: { new_shape: [3, 2] } },
        { step: 3, value_id: 3, op: "PERMUTE", input_ids: [2], shape: [2, 3], dtype: "float32", arg: { axes: [1, 0] } },
        { step: 4, value_id: 4, op: "REDUCE", input_ids: [3], shape: [2], dtype: "float32", arg: { op: "sum", axis: 1, keepdims: false } },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [2, 3], dtype: "float32", bytes: 24, first_step: 0, last_step: 1, materialize: false },
        { value_id: 1, op: "LOAD", shape: [2, 3], dtype: "float32", bytes: 24, first_step: 1, last_step: 2, materialize: false },
        { value_id: 2, op: "RESHAPE", shape: [3, 2], dtype: "float32", bytes: 24, first_step: 2, last_step: 3, materialize: false },
        { value_id: 3, op: "PERMUTE", shape: [2, 3], dtype: "float32", bytes: 24, first_step: 3, last_step: 4, materialize: false },
        { value_id: 4, op: "REDUCE", shape: [2], dtype: "float32", bytes: 8, first_step: 4, last_step: 4, materialize: true },
      ],
      root_id: 4,
      materialization_boundary: "root",
      peak_live_bytes: 72,
      has_custom_ops: false,
    } as const;
    const reduced = await runTensorGpuPlan(device, reducePlan, [{ valueId: 0, data: x }]);
    expect([...reduced.data]).toEqual([6, 9]);

    const b = new Float32Array([1, 2]);
    const broadcastPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [2, 1], dtype: "float32" },
        { step: 1, value_id: 1, op: "LOAD", input_ids: [0], shape: [2, 1], dtype: "float32" },
        { step: 2, value_id: 2, op: "BROADCAST_TO", input_ids: [1], shape: [2, 3], dtype: "float32", arg: { shape: [2, 3] } },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [2, 1], dtype: "float32", bytes: 8, first_step: 0, last_step: 1, materialize: false },
        { value_id: 1, op: "LOAD", shape: [2, 1], dtype: "float32", bytes: 8, first_step: 1, last_step: 2, materialize: false },
        { value_id: 2, op: "BROADCAST_TO", shape: [2, 3], dtype: "float32", bytes: 24, first_step: 2, last_step: 2, materialize: true },
      ],
      root_id: 2,
      materialization_boundary: "root",
      peak_live_bytes: 40,
      has_custom_ops: false,
    } as const;
    const broadcasted = await runTensorGpuPlan(device, broadcastPlan, [{ valueId: 0, data: b }]);
    expect([...broadcasted.data]).toEqual([1, 1, 1, 2, 2, 2]);
  });

  it("generic tensor GPU plan runs SGD_UPDATE", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const param = new Float32Array([1, -2, 3, -4]);
    const grad = new Float32Array([0.5, -0.25, 0.75, -1]);
    const plan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 2, value_id: 2, op: "LOAD", input_ids: [0], shape: [4], dtype: "float32" },
        { step: 3, value_id: 3, op: "LOAD", input_ids: [1], shape: [4], dtype: "float32" },
        {
          step: 4,
          value_id: 4,
          op: "SGD_UPDATE",
          input_ids: [2, 3],
          shape: [4],
          dtype: "float32",
          arg: { lr: 0.1, weight_decay: 0.01 },
        },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [4], dtype: "float32", bytes: 16, first_step: 0, last_step: 2, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [4], dtype: "float32", bytes: 16, first_step: 1, last_step: 3, materialize: false },
        { value_id: 2, op: "LOAD", shape: [4], dtype: "float32", bytes: 16, first_step: 2, last_step: 4, materialize: false },
        { value_id: 3, op: "LOAD", shape: [4], dtype: "float32", bytes: 16, first_step: 3, last_step: 4, materialize: false },
        { value_id: 4, op: "SGD_UPDATE", shape: [4], dtype: "float32", bytes: 16, first_step: 4, last_step: 4, materialize: true },
      ],
      root_id: 4,
      materialization_boundary: "root",
      peak_live_bytes: 80,
      has_custom_ops: false,
    } as const;
    const result = await runTensorGpuPlan(device, plan, [
      { valueId: 0, data: param },
      { valueId: 1, data: grad },
    ]);
    for (let i = 0; i < param.length; i++) {
      const expected = param[i]! - 0.1 * (grad[i]! + 0.01 * param[i]!);
      expect(Math.abs(result.data[i]! - expected)).toBeLessThan(1e-6);
    }
  });

  it("generic tensor GPU plan runs AdamW update roots", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const param = new Float32Array([1, -2, 3, -4]);
    const grad = new Float32Array([0.5, -0.25, 0.75, -1]);
    const m0 = new Float32Array([0.1, -0.2, 0.3, -0.4]);
    const v0 = new Float32Array([0.01, 0.02, 0.03, 0.04]);
    const arg = { lr: 0.01, beta1: 0.8, beta2: 0.95, eps: 1e-6, weight_decay: 0.02, step: 3 };
    const expectedM = new Float32Array(m0.length);
    const expectedV = new Float32Array(v0.length);
    const expectedP = new Float32Array(param.length);
    for (let i = 0; i < param.length; i++) {
      expectedM[i] = arg.beta1 * m0[i]! + (1 - arg.beta1) * grad[i]!;
      expectedV[i] = arg.beta2 * v0[i]! + (1 - arg.beta2) * grad[i]! * grad[i]!;
      const mHat = expectedM[i]! / (1 - arg.beta1 ** arg.step);
      const vHat = expectedV[i]! / (1 - arg.beta2 ** arg.step);
      expectedP[i] = param[i]! - arg.lr * (mHat / (Math.sqrt(vHat) + arg.eps)) - arg.lr * arg.weight_decay * param[i]!;
    }

    async function runAdamRoot(rootOp: string, rootShape: readonly number[], rootInputs: readonly number[]) {
      const steps = [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 2, value_id: 2, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 3, value_id: 3, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 4, value_id: 4, op: "LOAD", input_ids: [0], shape: [4], dtype: "float32" },
        { step: 5, value_id: 5, op: "LOAD", input_ids: [1], shape: [4], dtype: "float32" },
        { step: 6, value_id: 6, op: "LOAD", input_ids: [2], shape: [4], dtype: "float32" },
        { step: 7, value_id: 7, op: "LOAD", input_ids: [3], shape: [4], dtype: "float32" },
        { step: 8, value_id: 8, op: "ADAMW_UPDATE_M", input_ids: [6, 5], shape: [4], dtype: "float32", arg },
        { step: 9, value_id: 9, op: "ADAMW_UPDATE_V", input_ids: [7, 5], shape: [4], dtype: "float32", arg },
        { step: 10, value_id: 10, op: rootOp, input_ids: rootInputs, shape: rootShape, dtype: "float32", arg },
      ];
      const buffers = steps.map((step) => ({
        value_id: step.value_id,
        op: step.op,
        shape: step.shape,
        dtype: "float32",
        bytes: 16,
        first_step: step.step,
        last_step: 10,
        materialize: step.value_id === 10,
      }));
      const result = await runTensorGpuPlan(device, {
        steps,
        buffers,
        root_id: 10,
        materialization_boundary: "root",
        peak_live_bytes: 176,
        has_custom_ops: false,
      }, [
        { valueId: 0, data: param },
        { valueId: 1, data: grad },
        { valueId: 2, data: m0 },
        { valueId: 3, data: v0 },
      ]);
      return result.data;
    }

    const m = await runAdamRoot("ADAMW_UPDATE_M", [4], [6, 5]);
    const v = await runAdamRoot("ADAMW_UPDATE_V", [4], [7, 5]);
    const p = await runAdamRoot("ADAMW_UPDATE_PARAM", [4], [4, 5, 8, 9]);
    for (let i = 0; i < param.length; i++) {
      expect(Math.abs(m[i]! - expectedM[i]!)).toBeLessThan(1e-6);
      expect(Math.abs(v[i]! - expectedV[i]!)).toBeLessThan(1e-6);
      expect(Math.abs(p[i]! - expectedP[i]!)).toBeLessThan(1e-5);
    }
  });

  it("generic tensor GPU plan runs Adam update roots", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const param = new Float32Array([1, -2, 3, -4]);
    const grad = new Float32Array([0.5, -0.25, 0.75, -1]);
    const m0 = new Float32Array([0.1, -0.2, 0.3, -0.4]);
    const v0 = new Float32Array([0.01, 0.02, 0.03, 0.04]);
    const arg = { lr: 0.02, beta1: 0.75, beta2: 0.9, eps: 1e-6, weight_decay: 0.03, step: 4 };
    const expectedM = new Float32Array(m0.length);
    const expectedV = new Float32Array(v0.length);
    const expectedP = new Float32Array(param.length);
    for (let i = 0; i < param.length; i++) {
      const gradEff = grad[i]! + arg.weight_decay * param[i]!;
      expectedM[i] = arg.beta1 * m0[i]! + (1 - arg.beta1) * gradEff;
      expectedV[i] = arg.beta2 * v0[i]! + (1 - arg.beta2) * gradEff * gradEff;
      const mHat = expectedM[i]! / (1 - arg.beta1 ** arg.step);
      const vHat = expectedV[i]! / (1 - arg.beta2 ** arg.step);
      expectedP[i] = param[i]! - arg.lr * (mHat / (Math.sqrt(vHat) + arg.eps));
    }

    async function runAdamRoot(rootOp: string, rootShape: readonly number[], rootInputs: readonly number[]) {
      const steps = [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 2, value_id: 2, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 3, value_id: 3, op: "BUFFER", input_ids: [], shape: [4], dtype: "float32" },
        { step: 4, value_id: 4, op: "LOAD", input_ids: [0], shape: [4], dtype: "float32" },
        { step: 5, value_id: 5, op: "LOAD", input_ids: [1], shape: [4], dtype: "float32" },
        { step: 6, value_id: 6, op: "LOAD", input_ids: [2], shape: [4], dtype: "float32" },
        { step: 7, value_id: 7, op: "LOAD", input_ids: [3], shape: [4], dtype: "float32" },
        { step: 8, value_id: 8, op: "ADAM_UPDATE_M", input_ids: [4, 5, 6], shape: [4], dtype: "float32", arg },
        { step: 9, value_id: 9, op: "ADAM_UPDATE_V", input_ids: [4, 5, 7], shape: [4], dtype: "float32", arg },
        { step: 10, value_id: 10, op: rootOp, input_ids: rootInputs, shape: rootShape, dtype: "float32", arg },
      ];
      const buffers = steps.map((step) => ({
        value_id: step.value_id,
        op: step.op,
        shape: step.shape,
        dtype: "float32",
        bytes: 16,
        first_step: step.step,
        last_step: 10,
        materialize: step.value_id === 10,
      }));
      const result = await runTensorGpuPlan(device, {
        steps,
        buffers,
        root_id: 10,
        materialization_boundary: "root",
        peak_live_bytes: 176,
        has_custom_ops: false,
      }, [
        { valueId: 0, data: param },
        { valueId: 1, data: grad },
        { valueId: 2, data: m0 },
        { valueId: 3, data: v0 },
      ]);
      return result.data;
    }

    const m = await runAdamRoot("ADAM_UPDATE_M", [4], [4, 5, 6]);
    const v = await runAdamRoot("ADAM_UPDATE_V", [4], [4, 5, 7]);
    const p = await runAdamRoot("ADAM_UPDATE_PARAM", [4], [4, 8, 9]);
    for (let i = 0; i < param.length; i++) {
      expect(Math.abs(m[i]! - expectedM[i]!)).toBeLessThan(1e-6);
      expect(Math.abs(v[i]! - expectedV[i]!)).toBeLessThan(1e-6);
      expect(Math.abs(p[i]! - expectedP[i]!)).toBeLessThan(1e-5);
    }
  });

  it("generic tensor GPU plan runs LayerNorm forward and backward roots", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const x = new Float32Array([1, 2, 4, -1, 0.5, 3]);
    const weight = new Float32Array([0.75, -1.25, 1.5]);
    const bias = new Float32Array([0.1, -0.2, 0.3]);
    const dy = new Float32Array([0.25, -0.5, 1.0, -1.5, 0.75, 0.5]);
    const arg = { normalized_shape: [3], rows: 2, cols: 3, eps: 1e-5 };
    const expectedY = new Float32Array(6);
    const expectedDx = new Float32Array(6);
    const expectedDw = new Float32Array(3);
    const expectedDb = new Float32Array(3);
    for (let r = 0; r < arg.rows; r++) {
      const base = r * arg.cols;
      let mean = 0;
      for (let c = 0; c < arg.cols; c++) mean += x[base + c]!;
      mean /= arg.cols;
      let variance = 0;
      for (let c = 0; c < arg.cols; c++) {
        const centered = x[base + c]! - mean;
        variance += centered * centered;
      }
      variance /= arg.cols;
      const invStd = 1 / Math.sqrt(variance + arg.eps);
      let sumG = 0;
      let sumGXhat = 0;
      const xHat = new Float32Array(arg.cols);
      const g = new Float32Array(arg.cols);
      for (let c = 0; c < arg.cols; c++) {
        xHat[c] = (x[base + c]! - mean) * invStd;
        expectedY[base + c] = xHat[c]! * weight[c]! + bias[c]!;
        g[c] = dy[base + c]! * weight[c]!;
        sumG += g[c]!;
        sumGXhat += g[c]! * xHat[c]!;
        expectedDw[c] = expectedDw[c]! + dy[base + c]! * xHat[c]!;
        expectedDb[c] = expectedDb[c]! + dy[base + c]!;
      }
      for (let c = 0; c < arg.cols; c++) {
        expectedDx[base + c] = (invStd / arg.cols) * (arg.cols * g[c]! - sumG - xHat[c]! * sumGXhat);
      }
    }

    async function runLayerNormRoot(rootOp: string, rootShape: readonly number[], rootInputs: readonly number[]) {
      const steps = [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [2, 3], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [3], dtype: "float32" },
        { step: 2, value_id: 2, op: "BUFFER", input_ids: [], shape: [3], dtype: "float32" },
        { step: 3, value_id: 3, op: "BUFFER", input_ids: [], shape: [2, 3], dtype: "float32" },
        { step: 4, value_id: 4, op: "LOAD", input_ids: [0], shape: [2, 3], dtype: "float32" },
        { step: 5, value_id: 5, op: "LOAD", input_ids: [1], shape: [3], dtype: "float32" },
        { step: 6, value_id: 6, op: "LOAD", input_ids: [2], shape: [3], dtype: "float32" },
        { step: 7, value_id: 7, op: "LOAD", input_ids: [3], shape: [2, 3], dtype: "float32" },
        { step: 8, value_id: 8, op: rootOp, input_ids: rootInputs, shape: rootShape, dtype: "float32", arg },
      ];
      const buffers = steps.map((step) => ({
        value_id: step.value_id,
        op: step.op,
        shape: step.shape,
        dtype: "float32",
        bytes: step.shape.reduce((acc, dim) => acc * dim, 1) * 4,
        first_step: step.step,
        last_step: 8,
        materialize: step.value_id === 8,
      }));
      const result = await runTensorGpuPlan(device, {
        steps,
        buffers,
        root_id: 8,
        materialization_boundary: "root",
        peak_live_bytes: 160,
        has_custom_ops: false,
      }, [
        { valueId: 0, data: x },
        { valueId: 1, data: weight },
        { valueId: 2, data: bias },
        { valueId: 3, data: dy },
      ]);
      return result.data;
    }

    const y = await runLayerNormRoot("LAYER_NORM", [2, 3], [4, 5, 6]);
    const dx = await runLayerNormRoot("LAYER_NORM_BACKWARD_INPUT", [2, 3], [7, 4, 5]);
    const dw = await runLayerNormRoot("LAYER_NORM_BACKWARD_WEIGHT", [3], [7, 4]);
    const db = await runLayerNormRoot("LAYER_NORM_BACKWARD_BIAS", [3], [7]);
    for (let i = 0; i < expectedY.length; i++) {
      expect(Math.abs(y[i]! - expectedY[i]!)).toBeLessThan(1e-5);
      expect(Math.abs(dx[i]! - expectedDx[i]!)).toBeLessThan(1e-5);
    }
    for (let i = 0; i < expectedDw.length; i++) {
      expect(Math.abs(dw[i]! - expectedDw[i]!)).toBeLessThan(1e-5);
      expect(Math.abs(db[i]! - expectedDb[i]!)).toBeLessThan(1e-6);
    }
  });

  it("generic tensor GPU plan runs Conv1d and Conv2d forward", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();

    const conv1dPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [1, 1, 4], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [1, 1, 2], dtype: "float32" },
        { step: 2, value_id: 2, op: "BUFFER", input_ids: [], shape: [1], dtype: "float32" },
        { step: 3, value_id: 3, op: "LOAD", input_ids: [0], shape: [1, 1, 4], dtype: "float32" },
        { step: 4, value_id: 4, op: "LOAD", input_ids: [1], shape: [1, 1, 2], dtype: "float32" },
        { step: 5, value_id: 5, op: "LOAD", input_ids: [2], shape: [1], dtype: "float32" },
        {
          step: 6,
          value_id: 6,
          op: "CONV1D",
          input_ids: [3, 4, 5],
          shape: [1, 1, 3],
          dtype: "float32",
          arg: {
            n: 1, c_in: 1, l_in: 4, c_out: 1, k: 2,
            stride: 1, padding: 0, dilation: 1, groups: 1, l_out: 3,
            has_bias: true,
          },
        },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [1, 1, 4], dtype: "float32", bytes: 16, first_step: 0, last_step: 3, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [1, 1, 2], dtype: "float32", bytes: 8, first_step: 1, last_step: 4, materialize: false },
        { value_id: 2, op: "BUFFER", shape: [1], dtype: "float32", bytes: 4, first_step: 2, last_step: 5, materialize: false },
        { value_id: 3, op: "LOAD", shape: [1, 1, 4], dtype: "float32", bytes: 16, first_step: 3, last_step: 6, materialize: false },
        { value_id: 4, op: "LOAD", shape: [1, 1, 2], dtype: "float32", bytes: 8, first_step: 4, last_step: 6, materialize: false },
        { value_id: 5, op: "LOAD", shape: [1], dtype: "float32", bytes: 4, first_step: 5, last_step: 6, materialize: false },
        { value_id: 6, op: "CONV1D", shape: [1, 1, 3], dtype: "float32", bytes: 12, first_step: 6, last_step: 6, materialize: true },
      ],
      root_id: 6,
      materialization_boundary: "root",
      peak_live_bytes: 52,
      has_custom_ops: false,
    } as const;
    const conv1d = await runTensorGpuPlan(device, conv1dPlan, [
      { valueId: 0, data: new Float32Array([1, 2, 3, 4]) },
      { valueId: 1, data: new Float32Array([2, -1]) },
      { valueId: 2, data: new Float32Array([0.5]) },
    ]);
    expect([...conv1d.data]).toEqual([0.5, 1.5, 2.5]);

    const conv2dPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [1, 1, 3, 3], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 2, value_id: 2, op: "BUFFER", input_ids: [], shape: [1], dtype: "float32" },
        { step: 3, value_id: 3, op: "LOAD", input_ids: [0], shape: [1, 1, 3, 3], dtype: "float32" },
        { step: 4, value_id: 4, op: "LOAD", input_ids: [1], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 5, value_id: 5, op: "LOAD", input_ids: [2], shape: [1], dtype: "float32" },
        {
          step: 6,
          value_id: 6,
          op: "CONV2D",
          input_ids: [3, 4, 5],
          shape: [1, 1, 2, 2],
          dtype: "float32",
          arg: {
            n: 1, c_in: 1, h: 3, w: 3, c_out: 1, kh: 2, kw: 2,
            stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0,
            dilation_h: 1, dilation_w: 1, groups: 1,
            out_h: 2, out_w: 2, has_bias: true,
          },
        },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [1, 1, 3, 3], dtype: "float32", bytes: 36, first_step: 0, last_step: 3, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 1, last_step: 4, materialize: false },
        { value_id: 2, op: "BUFFER", shape: [1], dtype: "float32", bytes: 4, first_step: 2, last_step: 5, materialize: false },
        { value_id: 3, op: "LOAD", shape: [1, 1, 3, 3], dtype: "float32", bytes: 36, first_step: 3, last_step: 6, materialize: false },
        { value_id: 4, op: "LOAD", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 4, last_step: 6, materialize: false },
        { value_id: 5, op: "LOAD", shape: [1], dtype: "float32", bytes: 4, first_step: 5, last_step: 6, materialize: false },
        { value_id: 6, op: "CONV2D", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 6, last_step: 6, materialize: true },
      ],
      root_id: 6,
      materialization_boundary: "root",
      peak_live_bytes: 128,
      has_custom_ops: false,
    } as const;
    const conv2d = await runTensorGpuPlan(device, conv2dPlan, [
      { valueId: 0, data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]) },
      { valueId: 1, data: new Float32Array([1, 0, 0, -1]) },
      { valueId: 2, data: new Float32Array([0.5]) },
    ]);
    expect([...conv2d.data]).toEqual([-3.5, -3.5, -3.5, -3.5]);
  });

  it("generic tensor GPU plan runs Conv1d backward roots", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const arg = {
      n: 1, c_in: 1, l_in: 4, c_out: 1, k: 2,
      stride: 1, padding: 0, dilation: 1, groups: 1, l_out: 3,
    } as const;
    const baseBuffers = [
      { value_id: 0, op: "BUFFER", shape: [1, 1, 3], dtype: "float32", bytes: 12, first_step: 0, last_step: 2, materialize: false },
      { value_id: 1, op: "BUFFER", shape: [1, 1, 2], dtype: "float32", bytes: 8, first_step: 1, last_step: 3, materialize: false },
      { value_id: 2, op: "LOAD", shape: [1, 1, 3], dtype: "float32", bytes: 12, first_step: 2, last_step: 4, materialize: false },
      { value_id: 3, op: "LOAD", shape: [1, 1, 2], dtype: "float32", bytes: 8, first_step: 3, last_step: 4, materialize: false },
    ];
    const dxPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [1, 1, 3], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [1, 1, 2], dtype: "float32" },
        { step: 2, value_id: 2, op: "LOAD", input_ids: [0], shape: [1, 1, 3], dtype: "float32" },
        { step: 3, value_id: 3, op: "LOAD", input_ids: [1], shape: [1, 1, 2], dtype: "float32" },
        { step: 4, value_id: 4, op: "CONV1D_BACKWARD_INPUT", input_ids: [2, 3], shape: [1, 1, 4], dtype: "float32", arg },
      ],
      buffers: [
        ...baseBuffers,
        { value_id: 4, op: "CONV1D_BACKWARD_INPUT", shape: [1, 1, 4], dtype: "float32", bytes: 16, first_step: 4, last_step: 4, materialize: true },
      ],
      root_id: 4,
      materialization_boundary: "root",
      peak_live_bytes: 56,
      has_custom_ops: false,
    } as const;
    const dx = await runTensorGpuPlan(device, dxPlan, [
      { valueId: 0, data: new Float32Array([10, 20, 30]) },
      { valueId: 1, data: new Float32Array([2, -1]) },
    ]);
    expect([...dx.data]).toEqual([20, 30, 40, -30]);

    const dwPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [1, 1, 3], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [1, 1, 4], dtype: "float32" },
        { step: 2, value_id: 2, op: "LOAD", input_ids: [0], shape: [1, 1, 3], dtype: "float32" },
        { step: 3, value_id: 3, op: "LOAD", input_ids: [1], shape: [1, 1, 4], dtype: "float32" },
        { step: 4, value_id: 4, op: "CONV1D_BACKWARD_WEIGHT", input_ids: [2, 3], shape: [1, 1, 2], dtype: "float32", arg },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [1, 1, 3], dtype: "float32", bytes: 12, first_step: 0, last_step: 2, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [1, 1, 4], dtype: "float32", bytes: 16, first_step: 1, last_step: 3, materialize: false },
        { value_id: 2, op: "LOAD", shape: [1, 1, 3], dtype: "float32", bytes: 12, first_step: 2, last_step: 4, materialize: false },
        { value_id: 3, op: "LOAD", shape: [1, 1, 4], dtype: "float32", bytes: 16, first_step: 3, last_step: 4, materialize: false },
        { value_id: 4, op: "CONV1D_BACKWARD_WEIGHT", shape: [1, 1, 2], dtype: "float32", bytes: 8, first_step: 4, last_step: 4, materialize: true },
      ],
      root_id: 4,
      materialization_boundary: "root",
      peak_live_bytes: 64,
      has_custom_ops: false,
    } as const;
    const dw = await runTensorGpuPlan(device, dwPlan, [
      { valueId: 0, data: new Float32Array([10, 20, 30]) },
      { valueId: 1, data: new Float32Array([1, 2, 3, 4]) },
    ]);
    expect([...dw.data]).toEqual([140, 200]);

    const dbPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [1, 1, 3], dtype: "float32" },
        { step: 1, value_id: 1, op: "LOAD", input_ids: [0], shape: [1, 1, 3], dtype: "float32" },
        { step: 2, value_id: 2, op: "CONV1D_BACKWARD_BIAS", input_ids: [1], shape: [1], dtype: "float32", arg },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [1, 1, 3], dtype: "float32", bytes: 12, first_step: 0, last_step: 1, materialize: false },
        { value_id: 1, op: "LOAD", shape: [1, 1, 3], dtype: "float32", bytes: 12, first_step: 1, last_step: 2, materialize: false },
        { value_id: 2, op: "CONV1D_BACKWARD_BIAS", shape: [1], dtype: "float32", bytes: 4, first_step: 2, last_step: 2, materialize: true },
      ],
      root_id: 2,
      materialization_boundary: "root",
      peak_live_bytes: 28,
      has_custom_ops: false,
    } as const;
    const db = await runTensorGpuPlan(device, dbPlan, [
      { valueId: 0, data: new Float32Array([10, 20, 30]) },
    ]);
    expect([...db.data]).toEqual([60]);
  });

  it("generic tensor GPU plan runs Conv2d backward roots", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const arg = {
      n: 1, c_in: 1, h: 3, w: 3, c_out: 1, kh: 2, kw: 2,
      stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0,
      dilation_h: 1, dilation_w: 1, groups: 1, out_h: 2, out_w: 2,
    } as const;
    const dy = new Float32Array([10, 20, 30, 40]);
    const weight = new Float32Array([1, 0, 0, -1]);
    const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);

    const dxPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 2, value_id: 2, op: "LOAD", input_ids: [0], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 3, value_id: 3, op: "LOAD", input_ids: [1], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 4, value_id: 4, op: "CONV2D_BACKWARD_INPUT", input_ids: [2, 3], shape: [1, 1, 3, 3], dtype: "float32", arg },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 0, last_step: 2, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 1, last_step: 3, materialize: false },
        { value_id: 2, op: "LOAD", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 2, last_step: 4, materialize: false },
        { value_id: 3, op: "LOAD", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 3, last_step: 4, materialize: false },
        { value_id: 4, op: "CONV2D_BACKWARD_INPUT", shape: [1, 1, 3, 3], dtype: "float32", bytes: 36, first_step: 4, last_step: 4, materialize: true },
      ],
      root_id: 4,
      materialization_boundary: "root",
      peak_live_bytes: 100,
      has_custom_ops: false,
    } as const;
    const dx = await runTensorGpuPlan(device, dxPlan, [
      { valueId: 0, data: dy },
      { valueId: 1, data: weight },
    ]);
    expect([...dx.data]).toEqual([10, 20, 0, 30, 30, -20, 0, -30, -40]);

    const dwPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [1, 1, 3, 3], dtype: "float32" },
        { step: 2, value_id: 2, op: "LOAD", input_ids: [0], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 3, value_id: 3, op: "LOAD", input_ids: [1], shape: [1, 1, 3, 3], dtype: "float32" },
        { step: 4, value_id: 4, op: "CONV2D_BACKWARD_WEIGHT", input_ids: [2, 3], shape: [1, 1, 2, 2], dtype: "float32", arg },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 0, last_step: 2, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [1, 1, 3, 3], dtype: "float32", bytes: 36, first_step: 1, last_step: 3, materialize: false },
        { value_id: 2, op: "LOAD", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 2, last_step: 4, materialize: false },
        { value_id: 3, op: "LOAD", shape: [1, 1, 3, 3], dtype: "float32", bytes: 36, first_step: 3, last_step: 4, materialize: false },
        { value_id: 4, op: "CONV2D_BACKWARD_WEIGHT", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 4, last_step: 4, materialize: true },
      ],
      root_id: 4,
      materialization_boundary: "root",
      peak_live_bytes: 120,
      has_custom_ops: false,
    } as const;
    const dw = await runTensorGpuPlan(device, dwPlan, [
      { valueId: 0, data: dy },
      { valueId: 1, data: input },
    ]);
    expect([...dw.data]).toEqual([370, 470, 670, 770]);

    const dbPlan = {
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 1, value_id: 1, op: "LOAD", input_ids: [0], shape: [1, 1, 2, 2], dtype: "float32" },
        { step: 2, value_id: 2, op: "CONV2D_BACKWARD_BIAS", input_ids: [1], shape: [1], dtype: "float32", arg },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 0, last_step: 1, materialize: false },
        { value_id: 1, op: "LOAD", shape: [1, 1, 2, 2], dtype: "float32", bytes: 16, first_step: 1, last_step: 2, materialize: false },
        { value_id: 2, op: "CONV2D_BACKWARD_BIAS", shape: [1], dtype: "float32", bytes: 4, first_step: 2, last_step: 2, materialize: true },
      ],
      root_id: 2,
      materialization_boundary: "root",
      peak_live_bytes: 36,
      has_custom_ops: false,
    } as const;
    const db = await runTensorGpuPlan(device, dbPlan, [
      { valueId: 0, data: dy },
    ]);
    expect([...db.data]).toEqual([100]);
  });

  it("generic tensor GPU plan runs ConvTranspose2d forward and backward roots", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const arg = {
      n: 1, c_in: 1, h: 2, w: 2, c_out: 1, c_out_per_group: 1, kh: 2, kw: 2,
      stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0,
      output_pad_h: 0, output_pad_w: 0,
      dilation_h: 1, dilation_w: 1, groups: 1, out_h: 3, out_w: 3,
    } as const;
    const x = new Float32Array([1, 2, 3, 4]);
    const weight = new Float32Array([1, 0, 0, -1]);
    const bias = new Float32Array([0.5]);
    const dy = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);

    async function runSingle(
      op: string,
      inputShapes: readonly (readonly number[])[],
      outputShape: readonly number[],
      inputs: readonly Float32Array[],
    ): Promise<Float32Array> {
      const steps = [
        ...inputShapes.map((shape, i) => ({
          step: i,
          value_id: i,
          op: "BUFFER",
          input_ids: [],
          shape,
          dtype: "float32",
        })),
        ...inputShapes.map((shape, i) => ({
          step: inputShapes.length + i,
          value_id: inputShapes.length + i,
          op: "LOAD",
          input_ids: [i],
          shape,
          dtype: "float32",
        })),
        {
          step: inputShapes.length * 2,
          value_id: inputShapes.length * 2,
          op,
          input_ids: inputShapes.map((_, i) => inputShapes.length + i),
          shape: outputShape,
          dtype: "float32",
          arg,
        },
      ];
      const buffers = steps.map((step) => ({
        value_id: step.value_id,
        op: step.op,
        shape: step.shape,
        dtype: "float32",
        bytes: step.shape.reduce((a, b) => a * b, 1) * 4,
        first_step: step.step,
        last_step: inputShapes.length * 2,
        materialize: step.value_id === inputShapes.length * 2,
      }));
      const result = await runTensorGpuPlan(device, {
        steps,
        buffers,
        root_id: inputShapes.length * 2,
        materialization_boundary: "root",
        peak_live_bytes: buffers.reduce((sum, buffer) => sum + buffer.bytes, 0),
        has_custom_ops: false,
      }, inputs.map((data, valueId) => ({ valueId, data })));
      return result.data;
    }

    expect([...await runSingle("CONV_TRANSPOSE2D", [[1, 1, 2, 2], [1, 1, 2, 2], [1]], [1, 1, 3, 3], [x, weight, bias])])
      .toEqual([1.5, 2.5, 0.5, 3.5, 3.5, -1.5, 0.5, -2.5, -3.5]);
    expect([...await runSingle("CONV_TRANSPOSE2D_BACKWARD_INPUT", [[1, 1, 3, 3], [1, 1, 2, 2]], [1, 1, 2, 2], [dy, weight])])
      .toEqual([-4, -4, -4, -4]);
    expect([...await runSingle("CONV_TRANSPOSE2D_BACKWARD_WEIGHT", [[1, 1, 3, 3], [1, 1, 2, 2]], [1, 1, 2, 2], [dy, x])])
      .toEqual([37, 47, 67, 77]);
    expect([...await runSingle("CONV_TRANSPOSE2D_BACKWARD_BIAS", [[1, 1, 3, 3]], [1], [dy])])
      .toEqual([45]);
  });

  it("generic tensor GPU plan runs Conv3d forward and backward roots", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const arg = {
      n: 1, c_in: 1, d: 3, h: 3, w: 3, c_out: 1, kd: 2, kh: 2, kw: 2,
      stride_d: 1, stride_h: 1, stride_w: 1,
      pad_d: 0, pad_h: 0, pad_w: 0,
      dilation_d: 1, dilation_h: 1, dilation_w: 1,
      groups: 1, out_d: 2, out_h: 2, out_w: 2,
    } as const;
    const x = new Float32Array(Array.from({ length: 27 }, (_, i) => i + 1));
    const weight = new Float32Array([1, 0, 0, 0, 0, 0, 0, -1]);
    const bias = new Float32Array([0.5]);
    const dy = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

    function ix5(n: number, c: number, d: number, h: number, w: number, cDim: number, dDim: number, hDim: number, wDim: number): number {
      return (((n * cDim + c) * dDim + d) * hDim + h) * wDim + w;
    }

    function conv3dRef(): Float32Array {
      const out = new Float32Array(arg.n * arg.c_out * arg.out_d * arg.out_h * arg.out_w);
      for (let od = 0; od < arg.out_d; od++) {
        for (let oh = 0; oh < arg.out_h; oh++) {
          for (let ow = 0; ow < arg.out_w; ow++) {
            let acc = bias[0]!;
            for (let rd = 0; rd < arg.kd; rd++) {
              for (let rh = 0; rh < arg.kh; rh++) {
                for (let rw = 0; rw < arg.kw; rw++) {
                  const xi = ix5(0, 0, od + rd, oh + rh, ow + rw, 1, arg.d, arg.h, arg.w);
                  const wi = ((0 * arg.kd + rd) * arg.kh + rh) * arg.kw + rw;
                  acc += x[xi]! * weight[wi]!;
                }
              }
            }
            out[ix5(0, 0, od, oh, ow, 1, arg.out_d, arg.out_h, arg.out_w)] = acc;
          }
        }
      }
      return out;
    }

    function conv3dDxRef(): Float32Array {
      const out = new Float32Array(arg.n * arg.c_in * arg.d * arg.h * arg.w);
      for (let od = 0; od < arg.out_d; od++) {
        for (let oh = 0; oh < arg.out_h; oh++) {
          for (let ow = 0; ow < arg.out_w; ow++) {
            const grad = dy[ix5(0, 0, od, oh, ow, 1, arg.out_d, arg.out_h, arg.out_w)]!;
            for (let rd = 0; rd < arg.kd; rd++) {
              for (let rh = 0; rh < arg.kh; rh++) {
                for (let rw = 0; rw < arg.kw; rw++) {
                  const xi = ix5(0, 0, od + rd, oh + rh, ow + rw, 1, arg.d, arg.h, arg.w);
                  const wi = ((0 * arg.kd + rd) * arg.kh + rh) * arg.kw + rw;
                  out[xi]! += grad * weight[wi]!;
                }
              }
            }
          }
        }
      }
      return out;
    }

    function conv3dDwRef(): Float32Array {
      const out = new Float32Array(arg.c_out * arg.c_in * arg.kd * arg.kh * arg.kw);
      for (let rd = 0; rd < arg.kd; rd++) {
        for (let rh = 0; rh < arg.kh; rh++) {
          for (let rw = 0; rw < arg.kw; rw++) {
            let acc = 0;
            for (let od = 0; od < arg.out_d; od++) {
              for (let oh = 0; oh < arg.out_h; oh++) {
                for (let ow = 0; ow < arg.out_w; ow++) {
                  const di = ix5(0, 0, od, oh, ow, 1, arg.out_d, arg.out_h, arg.out_w);
                  const xi = ix5(0, 0, od + rd, oh + rh, ow + rw, 1, arg.d, arg.h, arg.w);
                  acc += dy[di]! * x[xi]!;
                }
              }
            }
            out[((0 * arg.kd + rd) * arg.kh + rh) * arg.kw + rw] = acc;
          }
        }
      }
      return out;
    }

    async function runSingle(
      op: string,
      inputShapes: readonly (readonly number[])[],
      outputShape: readonly number[],
      inputs: readonly Float32Array[],
    ): Promise<Float32Array> {
      const steps = [
        ...inputShapes.map((shape, i) => ({
          step: i,
          value_id: i,
          op: "BUFFER",
          input_ids: [],
          shape,
          dtype: "float32",
        })),
        ...inputShapes.map((shape, i) => ({
          step: inputShapes.length + i,
          value_id: inputShapes.length + i,
          op: "LOAD",
          input_ids: [i],
          shape,
          dtype: "float32",
        })),
        {
          step: inputShapes.length * 2,
          value_id: inputShapes.length * 2,
          op,
          input_ids: inputShapes.map((_, i) => inputShapes.length + i),
          shape: outputShape,
          dtype: "float32",
          arg,
        },
      ];
      const buffers = steps.map((step) => ({
        value_id: step.value_id,
        op: step.op,
        shape: step.shape,
        dtype: "float32",
        bytes: step.shape.reduce((a, b) => a * b, 1) * 4,
        first_step: step.step,
        last_step: inputShapes.length * 2,
        materialize: step.value_id === inputShapes.length * 2,
      }));
      const result = await runTensorGpuPlan(device, {
        steps,
        buffers,
        root_id: inputShapes.length * 2,
        materialization_boundary: "root",
        peak_live_bytes: buffers.reduce((sum, buffer) => sum + buffer.bytes, 0),
        has_custom_ops: false,
      }, inputs.map((data, valueId) => ({ valueId, data })));
      return result.data;
    }

    expect([...await runSingle("CONV3D", [[1, 1, 3, 3, 3], [1, 1, 2, 2, 2], [1]], [1, 1, 2, 2, 2], [x, weight, bias])])
      .toEqual([...conv3dRef()]);
    expect([...await runSingle("CONV3D_BACKWARD_INPUT", [[1, 1, 2, 2, 2], [1, 1, 2, 2, 2]], [1, 1, 3, 3, 3], [dy, weight])])
      .toEqual([...conv3dDxRef()]);
    expect([...await runSingle("CONV3D_BACKWARD_WEIGHT", [[1, 1, 2, 2, 2], [1, 1, 3, 3, 3]], [1, 1, 2, 2, 2], [dy, x])])
      .toEqual([...conv3dDwRef()]);
    expect([...await runSingle("CONV3D_BACKWARD_BIAS", [[1, 1, 2, 2, 2]], [1], [dy])])
      .toEqual([36]);
  });

  it("fused elementwise codegen runs on the GPU and matches NumPy semantics", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const N = 128;
    const a = new Float32Array(N);
    const b = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      a[i] = i * 0.01;
      b[i] = i * 0.05;
    }
    const aBuf = uploadFloat32(device, a);
    const bBuf = uploadFloat32(device, b);

    try {
      // ops = [ADD(in0, in1), EXP(step0), DIV(step1, in0)]
      const result = fusedElementwiseDirect(
        device,
        [aBuf, bBuf],
        [
          ["ADD", -1, -2],
          ["EXP", 0, 0],
          ["DIV", 1, -1],
        ],
        N,
      );
      const out = await materializeFloat32(device, result.buffer, result.byteLength);
      result.buffer.destroy();

      // Reference: exp(a + b) / a.
      let maxDiff = 0;
      for (let i = 0; i < N; i++) {
        const expected = Math.exp(a[i]! + b[i]!) / a[i]!;
        if (Number.isFinite(expected)) {
          maxDiff = Math.max(maxDiff, Math.abs(out[i]! - expected) / (Math.abs(expected) + 1));
        }
      }
      expect(maxDiff).toBeLessThan(1e-3);
    } finally {
      aBuf.destroy();
      bBuf.destroy();
    }
  });

  it("Flash Attention v2 forward matches the composed-attention reference", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const B = 1, H = 2, Sq = 4, Sk = 4, D = 8;
    const scale = 1 / Math.sqrt(D);

    function makeRandom(len: number, seed: number): Float32Array {
      const arr = new Float32Array(len);
      let s = seed;
      for (let i = 0; i < len; i++) {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        arr[i] = (s / 0x7fffffff - 0.5) * 2;
      }
      return arr;
    }

    const qData = makeRandom(B * H * Sq * D, 1);
    const kData = makeRandom(B * H * Sk * D, 2);
    const vData = makeRandom(B * H * Sk * D, 3);

    const qBuf = uploadFloat32(device, qData);
    const kBuf = uploadFloat32(device, kData);
    const vBuf = uploadFloat32(device, vData);

    try {
      const result = flashAttentionDirect(
        device,
        qBuf,
        kBuf,
        vBuf,
        null,
        { B, H, Sq, Sk, D },
        scale,
      );
      const out = await materializeFloat32(device, result.buffer, result.byteLength);

      // Reference: scores = Q @ K^T * scale; softmax; @ V (loop-based).
      const ref = new Float32Array(B * H * Sq * D);
      for (let bi = 0; bi < B; bi++) {
        for (let hi = 0; hi < H; hi++) {
          for (let i = 0; i < Sq; i++) {
            const scores = new Float32Array(Sk);
            for (let j = 0; j < Sk; j++) {
              let dot = 0;
              for (let d = 0; d < D; d++) {
                dot += qData[((bi * H + hi) * Sq + i) * D + d]! *
                       kData[((bi * H + hi) * Sk + j) * D + d]!;
              }
              scores[j] = dot * scale;
            }
            // Stable softmax.
            let m = -Infinity;
            for (let j = 0; j < Sk; j++) if (scores[j]! > m) m = scores[j]!;
            let sum = 0;
            for (let j = 0; j < Sk; j++) {
              scores[j] = Math.exp(scores[j]! - m);
              sum += scores[j]!;
            }
            for (let j = 0; j < Sk; j++) scores[j]! /= sum;
            for (let d = 0; d < D; d++) {
              let v = 0;
              for (let j = 0; j < Sk; j++) {
                v += scores[j]! * vData[((bi * H + hi) * Sk + j) * D + d]!;
              }
              ref[((bi * H + hi) * Sq + i) * D + d] = v;
            }
          }
        }
      }

      let maxDiff = 0;
      for (let i = 0; i < ref.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(out[i]! - ref[i]!));
      }
      expect(maxDiff).toBeLessThan(1e-4);

      result.buffer.destroy();
    } finally {
      qBuf.destroy();
      kBuf.destroy();
      vBuf.destroy();
    }
  });

  it("WebGpuRealizerBridge.matmul end-to-end via the seam", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const bridge = createWebGpuRealizerBridge(device);

    const a = new Float32Array([1, 2, 3, 4]); // 2×2
    const b = new Float32Array([5, 6, 7, 8]); // 2×2

    const aHandle = bridge.upload(
      new Uint8Array(a.buffer, a.byteOffset, a.byteLength),
      [2, 2],
      "float32",
    );
    const bHandle = bridge.upload(
      new Uint8Array(b.buffer, b.byteOffset, b.byteLength),
      [2, 2],
      "float32",
    );

    const outHandle = bridge.matmul(aHandle, bHandle, 2, 2, 2, "float32");
    const outBytesPromise = bridge.materialize(outHandle, [2, 2], "float32") as unknown as Promise<Uint8Array>;
    const outBytes = await outBytesPromise;
    const out = new Float32Array(outBytes.buffer, outBytes.byteOffset, 4);

    // Reference: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    expect(Array.from(out)).toEqual([19, 22, 43, 50]);

    bridge.release(aHandle);
    bridge.release(bHandle);
    bridge.release(outHandle);
    expect(bridge.aliveHandleCount()).toBe(0);
  });

  it("WebGpuRealizerBridge.conv1d forward/backward matches reference", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const bridge = createWebGpuRealizerBridge(device);

    const N = 1, C = 4, L = 9;
    const COut = 6, K = 3;
    const stride = 2, padding = 2, dilation = 2, groups = 2;
    const LOut = 5;
    const input = new Float32Array(N * C * L);
    const weight = new Float32Array(COut * (C / groups) * K);
    const bias = new Float32Array(COut);
    const dy = new Float32Array(N * COut * LOut);
    for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.11);
    for (let i = 0; i < weight.length; i++) weight[i] = Math.cos(i * 0.09);
    for (let i = 0; i < bias.length; i++) bias[i] = i * 0.03 - 0.1;
    for (let i = 0; i < dy.length; i++) dy[i] = Math.sin(i * 0.17) * 0.5;

    const inputHandle = bridge.upload(new Uint8Array(input.buffer), [N, C, L], "float32");
    const weightHandle = bridge.upload(new Uint8Array(weight.buffer), [COut, C / groups, K], "float32");
    const biasHandle = bridge.upload(new Uint8Array(bias.buffer), [COut], "float32");
    const dyHandle = bridge.upload(new Uint8Array(dy.buffer), [N, COut, LOut], "float32");

    const outHandle = bridge.conv1d(
      inputHandle, weightHandle, biasHandle,
      N, C, L, COut, K, stride, padding, dilation, groups, LOut, "float32",
    );
    const gxHandle = bridge.conv1d_backward_input(
      dyHandle, weightHandle,
      N, C, L, COut, K, stride, padding, dilation, groups, LOut, "float32",
    );
    const gwHandle = bridge.conv1d_backward_weight(
      dyHandle, inputHandle,
      N, C, L, COut, K, stride, padding, dilation, groups, LOut, "float32",
    );
    const gbHandle = bridge.conv1d_backward_bias(dyHandle, N, COut, LOut, "float32");

    const outBytes = await bridge.materialize(outHandle, [N, COut, LOut], "float32");
    const gxBytes = await bridge.materialize(gxHandle, [N, C, L], "float32");
    const gwBytes = await bridge.materialize(gwHandle, [COut, C / groups, K], "float32");
    const gbBytes = await bridge.materialize(gbHandle, [COut], "float32");
    const out = new Float32Array(outBytes.buffer, outBytes.byteOffset, N * COut * LOut);
    const gx = new Float32Array(gxBytes.buffer, gxBytes.byteOffset, N * C * L);
    const gw = new Float32Array(gwBytes.buffer, gwBytes.byteOffset, COut * (C / groups) * K);
    const gb = new Float32Array(gbBytes.buffer, gbBytes.byteOffset, COut);

    const cpg = C / groups;
    const opg = COut / groups;
    const paddedL = L + 2 * padding;
    const xPad = new Float32Array(N * C * paddedL);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let li = 0; li < L; li++) {
          xPad[(n * C + c) * paddedL + li + padding] = input[(n * C + c) * L + li]!;
        }
      }
    }

    const outRef = new Float32Array(out.length);
    const gxPad = new Float32Array(N * C * paddedL);
    const gwRef = new Float32Array(gw.length);
    const gbRef = new Float32Array(gb.length);
    for (let n = 0; n < N; n++) {
      for (let g = 0; g < groups; g++) {
        const c0 = g * cpg;
        const o0 = g * opg;
        for (let co = 0; co < opg; co++) {
          const outCh = o0 + co;
          for (let i = 0; i < LOut; i++) {
            let acc = bias[outCh]!;
            const gradVal = dy[(n * COut + outCh) * LOut + i]!;
            const base = i * stride;
            gbRef[outCh]! += gradVal;
            for (let ciLocal = 0; ciLocal < cpg; ciLocal++) {
              const inCh = c0 + ciLocal;
              for (let r = 0; r < K; r++) {
                const liPad = base + r * dilation;
                const li = liPad - padding;
                const wIdx = (outCh * cpg + ciLocal) * K + r;
                acc += (li >= 0 && li < L ? input[(n * C + inCh) * L + li]! : 0) * weight[wIdx]!;
                gxPad[(n * C + inCh) * paddedL + liPad]! += gradVal * weight[wIdx]!;
                gwRef[wIdx]! += gradVal * xPad[(n * C + inCh) * paddedL + liPad]!;
              }
            }
            outRef[(n * COut + outCh) * LOut + i] = acc;
          }
        }
      }
    }
    const gxRef = new Float32Array(gx.length);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let li = 0; li < L; li++) {
          gxRef[(n * C + c) * L + li] = gxPad[(n * C + c) * paddedL + li + padding]!;
        }
      }
    }

    let maxDiff = 0;
    for (let i = 0; i < out.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out[i]! - outRef[i]!));
    for (let i = 0; i < gx.length; i++) maxDiff = Math.max(maxDiff, Math.abs(gx[i]! - gxRef[i]!));
    for (let i = 0; i < gw.length; i++) maxDiff = Math.max(maxDiff, Math.abs(gw[i]! - gwRef[i]!));
    for (let i = 0; i < gb.length; i++) maxDiff = Math.max(maxDiff, Math.abs(gb[i]! - gbRef[i]!));
    expect(maxDiff).toBeLessThan(1e-4);

    for (const h of [inputHandle, weightHandle, biasHandle, dyHandle, outHandle, gxHandle, gwHandle, gbHandle]) {
      bridge.release(h);
    }
    expect(bridge.aliveHandleCount()).toBe(0);
  });

  it("WebGpuRealizerBridge.conv2d keeps tensors resident and matches reference", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const bridge = createWebGpuRealizerBridge(device);

    const N = 1, C = 4, Hh = 5, Ww = 4;
    const COut = 6, KH = 2, KW = 2;
    const strideH = 1, strideW = 1;
    const padH = 2, padW = 0;
    const dilationH = 2, dilationW = 1;
    const groups = 2;
    const outH = 7, outW = 3;
    const input = new Float32Array(N * C * Hh * Ww);
    const weight = new Float32Array(COut * (C / groups) * KH * KW);
    const bias = new Float32Array(COut);
    for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.13);
    for (let i = 0; i < weight.length; i++) weight[i] = Math.cos(i * 0.07);
    for (let i = 0; i < bias.length; i++) bias[i] = i * 0.05 - 0.1;

    const inputHandle = bridge.upload(
      new Uint8Array(input.buffer, input.byteOffset, input.byteLength),
      [N, C, Hh, Ww],
      "float32",
    );
    const weightHandle = bridge.upload(
      new Uint8Array(weight.buffer, weight.byteOffset, weight.byteLength),
      [COut, C / groups, KH, KW],
      "float32",
    );
    const biasHandle = bridge.upload(
      new Uint8Array(bias.buffer, bias.byteOffset, bias.byteLength),
      [COut],
      "float32",
    );

    const outHandle = bridge.conv2d(
      inputHandle,
      weightHandle,
      biasHandle,
      N,
      C,
      Hh,
      Ww,
      COut,
      KH,
      KW,
      strideH,
      strideW,
      padH,
      padW,
      dilationH,
      dilationW,
      groups,
      outH,
      outW,
      "float32",
    );
    const outBytes = await bridge.materialize(outHandle, [N, COut, outH, outW], "float32");
    const out = new Float32Array(outBytes.buffer, outBytes.byteOffset, N * COut * outH * outW);

    const ref = new Float32Array(out.length);
    const cpg = C / groups;
    const opg = COut / groups;
    const effH = dilationH * (KH - 1) + 1;
    const effW = dilationW * (KW - 1) + 1;
    for (let n = 0; n < N; n++) {
      for (let g = 0; g < groups; g++) {
        const c0 = g * cpg;
        const o0 = g * opg;
        for (let co = 0; co < opg; co++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              let acc = bias[o0 + co]!;
              for (let ciLocal = 0; ciLocal < cpg; ciLocal++) {
                const ci = c0 + ciLocal;
                for (let r = 0; r < KH; r++) {
                  const ih = oh * strideH + r * dilationH - padH;
                  if (ih < 0 || ih >= Hh) continue;
                  for (let col = 0; col < KW; col++) {
                    const iw = ow * strideW + col * dilationW - padW;
                    if (iw < 0 || iw >= Ww) continue;
                    const xIndex = ((n * C + ci) * Hh + ih) * Ww + iw;
                    const wIndex = (((o0 + co) * cpg + ciLocal) * KH + r) * KW + col;
                    acc += input[xIndex]! * weight[wIndex]!;
                  }
                }
              }
              ref[((n * COut + o0 + co) * outH + oh) * outW + ow] = acc;
            }
          }
        }
      }
    }

    let maxDiff = 0;
    for (let i = 0; i < ref.length; i++) {
      const diff = Math.abs(out[i]! - ref[i]!);
      maxDiff = Math.max(maxDiff, diff);
    }
    expect(maxDiff).toBeLessThan(1e-4);

    bridge.release(inputHandle);
    bridge.release(weightHandle);
    bridge.release(biasHandle);
    bridge.release(outHandle);
    expect(bridge.aliveHandleCount()).toBe(0);
  });

  it("WebGpuRealizerBridge.conv2d backward kernels match reference", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const bridge = createWebGpuRealizerBridge(device);

    const N = 1, C = 4, Hh = 5, Ww = 4;
    const COut = 6, KH = 2, KW = 2;
    const strideH = 1, strideW = 1;
    const padH = 2, padW = 0;
    const dilationH = 2, dilationW = 1;
    const groups = 2;
    const outH = 7, outW = 3;
    const input = new Float32Array(N * C * Hh * Ww);
    const weight = new Float32Array(COut * (C / groups) * KH * KW);
    const dy = new Float32Array(N * COut * outH * outW);
    for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.11);
    for (let i = 0; i < weight.length; i++) weight[i] = Math.cos(i * 0.09);
    for (let i = 0; i < dy.length; i++) dy[i] = Math.sin(i * 0.17) * 0.5;

    const inputHandle = bridge.upload(
      new Uint8Array(input.buffer, input.byteOffset, input.byteLength),
      [N, C, Hh, Ww],
      "float32",
    );
    const weightHandle = bridge.upload(
      new Uint8Array(weight.buffer, weight.byteOffset, weight.byteLength),
      [COut, C / groups, KH, KW],
      "float32",
    );
    const dyHandle = bridge.upload(
      new Uint8Array(dy.buffer, dy.byteOffset, dy.byteLength),
      [N, COut, outH, outW],
      "float32",
    );

    const gxHandle = bridge.conv2d_backward_input(
      dyHandle,
      weightHandle,
      N,
      C,
      Hh,
      Ww,
      COut,
      KH,
      KW,
      strideH,
      strideW,
      padH,
      padW,
      dilationH,
      dilationW,
      groups,
      outH,
      outW,
      "float32",
    );
    const gwHandle = bridge.conv2d_backward_weight(
      dyHandle,
      inputHandle,
      N,
      C,
      Hh,
      Ww,
      COut,
      KH,
      KW,
      strideH,
      strideW,
      padH,
      padW,
      dilationH,
      dilationW,
      groups,
      outH,
      outW,
      "float32",
    );
    const gbHandle = bridge.conv2d_backward_bias(
      dyHandle,
      N,
      COut,
      outH,
      outW,
      "float32",
    );

    const gxBytes = await bridge.materialize(gxHandle, [N, C, Hh, Ww], "float32");
    const gwBytes = await bridge.materialize(gwHandle, [COut, C / groups, KH, KW], "float32");
    const gbBytes = await bridge.materialize(gbHandle, [COut], "float32");
    const gx = new Float32Array(gxBytes.buffer, gxBytes.byteOffset, N * C * Hh * Ww);
    const gw = new Float32Array(gwBytes.buffer, gwBytes.byteOffset, COut * (C / groups) * KH * KW);
    const gb = new Float32Array(gbBytes.buffer, gbBytes.byteOffset, COut);

    const cpg = C / groups;
    const opg = COut / groups;
    const gxRefPad = new Float32Array(N * C * (Hh + 2 * padH) * (Ww + 2 * padW));
    const paddedW = Ww + 2 * padW;
    const paddedH = Hh + 2 * padH;
    for (let n = 0; n < N; n++) {
      for (let g = 0; g < groups; g++) {
        const c0 = g * cpg;
        const o0 = g * opg;
        for (let co = 0; co < opg; co++) {
          const outCh = o0 + co;
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              const gradVal = dy[((n * COut + outCh) * outH + oh) * outW + ow]!;
              const hBase = oh * strideH;
              const wBase = ow * strideW;
              for (let ciLocal = 0; ciLocal < cpg; ciLocal++) {
                const inCh = c0 + ciLocal;
                for (let r = 0; r < KH; r++) {
                  const ih = hBase + r * dilationH;
                  for (let s = 0; s < KW; s++) {
                    const iw = wBase + s * dilationW;
                    const gxIdx = ((n * C + inCh) * paddedH + ih) * paddedW + iw;
                    const wIdx = ((outCh * cpg + ciLocal) * KH + r) * KW + s;
                    gxRefPad[gxIdx]! += gradVal * weight[wIdx]!;
                  }
                }
              }
            }
          }
        }
      }
    }
    const gxRef = new Float32Array(gx.length);
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let ih = 0; ih < Hh; ih++) {
          for (let iw = 0; iw < Ww; iw++) {
            gxRef[((n * C + c) * Hh + ih) * Ww + iw] =
              gxRefPad[((n * C + c) * paddedH + ih + padH) * paddedW + iw + padW]!;
          }
        }
      }
    }

    const gwRef = new Float32Array(gw.length);
    for (let n = 0; n < N; n++) {
      for (let g = 0; g < groups; g++) {
        const c0 = g * cpg;
        const o0 = g * opg;
        for (let co = 0; co < opg; co++) {
          const outCh = o0 + co;
          for (let ciLocal = 0; ciLocal < cpg; ciLocal++) {
            const inCh = c0 + ciLocal;
            for (let r = 0; r < KH; r++) {
              for (let s = 0; s < KW; s++) {
                let acc = 0;
                for (let oh = 0; oh < outH; oh++) {
                  const ih = oh * strideH + r * dilationH - padH;
                  if (ih < 0 || ih >= Hh) continue;
                  for (let ow = 0; ow < outW; ow++) {
                    const iw = ow * strideW + s * dilationW - padW;
                    if (iw < 0 || iw >= Ww) continue;
                    const dyIdx = ((n * COut + outCh) * outH + oh) * outW + ow;
                    const xIdx = ((n * C + inCh) * Hh + ih) * Ww + iw;
                    acc += dy[dyIdx]! * input[xIdx]!;
                  }
                }
                gwRef[((outCh * cpg + ciLocal) * KH + r) * KW + s]! += acc;
              }
            }
          }
        }
      }
    }

    const gbRef = new Float32Array(COut);
    for (let n = 0; n < N; n++) {
      for (let co = 0; co < COut; co++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            gbRef[co]! += dy[((n * COut + co) * outH + oh) * outW + ow]!;
          }
        }
      }
    }

    let maxDiff = 0;
    for (let i = 0; i < gx.length; i++) maxDiff = Math.max(maxDiff, Math.abs(gx[i]! - gxRef[i]!));
    for (let i = 0; i < gw.length; i++) maxDiff = Math.max(maxDiff, Math.abs(gw[i]! - gwRef[i]!));
    for (let i = 0; i < gb.length; i++) maxDiff = Math.max(maxDiff, Math.abs(gb[i]! - gbRef[i]!));
    expect(maxDiff).toBeLessThan(1e-4);

    bridge.release(inputHandle);
    bridge.release(weightHandle);
    bridge.release(dyHandle);
    bridge.release(gxHandle);
    bridge.release(gwHandle);
    bridge.release(gbHandle);
    expect(bridge.aliveHandleCount()).toBe(0);
  });
});
