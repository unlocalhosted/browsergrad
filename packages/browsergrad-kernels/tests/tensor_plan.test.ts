import { describe, expect, it } from "vitest";
import { normalizeTensorGpuPlan } from "../src/index";

describe("tensor GPU plan normalization", () => {
  it("accepts browsergrad-jit snake_case plan payloads", () => {
    const plan = normalizeTensorGpuPlan({
      steps: [
        {
          step: 0,
          value_id: 101,
          op: "BUFFER",
          input_ids: [],
          shape: [2, 2],
          dtype: "float32",
          arg: "buf:x",
        },
        {
          step: 1,
          value_id: 102,
          op: "LOAD",
          input_ids: [101],
          shape: [2, 2],
          dtype: "float32",
          arg: null,
        },
      ],
      buffers: [
        {
          value_id: 101,
          op: "BUFFER",
          shape: [2, 2],
          dtype: "float32",
          bytes: 16,
          first_step: 0,
          last_step: 1,
          materialize: false,
        },
        {
          value_id: 102,
          op: "LOAD",
          shape: [2, 2],
          dtype: "float32",
          bytes: 16,
          first_step: 1,
          last_step: 1,
          materialize: true,
        },
      ],
      root_id: 102,
      materialization_boundary: "root",
      peak_live_bytes: 32,
      has_custom_ops: false,
    });

    expect(plan.rootId).toBe(102);
    expect(plan.materializationBoundary).toBe("root");
    expect(plan.peakLiveBytes).toBe(32);
    expect(plan.hasCustomOps).toBe(false);
    expect(plan.steps[0]).toMatchObject({
      valueId: 101,
      inputIds: [],
      op: "BUFFER",
    });
    expect(plan.buffers[1]).toMatchObject({
      valueId: 102,
      firstStep: 1,
      lastStep: 1,
      materialize: true,
    });
  });

  it("rejects CUSTOM-backed plans before runtime dispatch", () => {
    expect(() =>
      normalizeTensorGpuPlan({
        steps: [],
        buffers: [],
        root_id: 1,
        materialization_boundary: "root",
        peak_live_bytes: 0,
        has_custom_ops: true,
      }),
    ).toThrow(/hasCustomOps must be false/);
  });

  it("accepts SGD_UPDATE optimizer IR plan steps", () => {
    const plan = normalizeTensorGpuPlan({
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [2], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [2], dtype: "float32" },
        { step: 2, value_id: 2, op: "SGD_UPDATE", input_ids: [0, 1], shape: [2], dtype: "float32", arg: { lr: 0.1, weight_decay: 0 } },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [2], dtype: "float32", bytes: 8, first_step: 0, last_step: 2, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [2], dtype: "float32", bytes: 8, first_step: 1, last_step: 2, materialize: false },
        { value_id: 2, op: "SGD_UPDATE", shape: [2], dtype: "float32", bytes: 8, first_step: 2, last_step: 2, materialize: true },
      ],
      root_id: 2,
      materialization_boundary: "root",
      peak_live_bytes: 24,
      has_custom_ops: false,
    });
    expect(plan.steps[2]?.op).toBe("SGD_UPDATE");
  });

  it("accepts AdamW optimizer IR plan steps", () => {
    const plan = normalizeTensorGpuPlan({
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [2], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [2], dtype: "float32" },
        { step: 2, value_id: 2, op: "ADAMW_UPDATE_M", input_ids: [0, 1], shape: [2], dtype: "float32", arg: { beta1: 0.9, beta2: 0.999, step: 1 } },
        { step: 3, value_id: 3, op: "ADAMW_UPDATE_V", input_ids: [0, 1], shape: [2], dtype: "float32", arg: { beta1: 0.9, beta2: 0.999, step: 1 } },
        { step: 4, value_id: 4, op: "ADAMW_UPDATE_PARAM", input_ids: [0, 1, 2, 3], shape: [2], dtype: "float32", arg: { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01, step: 1 } },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [2], dtype: "float32", bytes: 8, first_step: 0, last_step: 4, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [2], dtype: "float32", bytes: 8, first_step: 1, last_step: 4, materialize: false },
        { value_id: 2, op: "ADAMW_UPDATE_M", shape: [2], dtype: "float32", bytes: 8, first_step: 2, last_step: 4, materialize: false },
        { value_id: 3, op: "ADAMW_UPDATE_V", shape: [2], dtype: "float32", bytes: 8, first_step: 3, last_step: 4, materialize: false },
        { value_id: 4, op: "ADAMW_UPDATE_PARAM", shape: [2], dtype: "float32", bytes: 8, first_step: 4, last_step: 4, materialize: true },
      ],
      root_id: 4,
      materialization_boundary: "root",
      peak_live_bytes: 40,
      has_custom_ops: false,
    });
    expect(plan.steps.map((step) => step.op)).toContain("ADAMW_UPDATE_PARAM");
  });

  it("accepts Adam optimizer IR plan steps", () => {
    const plan = normalizeTensorGpuPlan({
      steps: [
        { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [2], dtype: "float32" },
        { step: 1, value_id: 1, op: "BUFFER", input_ids: [], shape: [2], dtype: "float32" },
        { step: 2, value_id: 2, op: "BUFFER", input_ids: [], shape: [2], dtype: "float32" },
        { step: 3, value_id: 3, op: "BUFFER", input_ids: [], shape: [2], dtype: "float32" },
        { step: 4, value_id: 4, op: "ADAM_UPDATE_M", input_ids: [0, 1, 2], shape: [2], dtype: "float32", arg: { beta1: 0.9, weight_decay: 0.01 } },
        { step: 5, value_id: 5, op: "ADAM_UPDATE_V", input_ids: [0, 1, 3], shape: [2], dtype: "float32", arg: { beta2: 0.999, weight_decay: 0.01 } },
        { step: 6, value_id: 6, op: "ADAM_UPDATE_PARAM", input_ids: [0, 4, 5], shape: [2], dtype: "float32", arg: { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8, step: 1 } },
      ],
      buffers: [
        { value_id: 0, op: "BUFFER", shape: [2], dtype: "float32", bytes: 8, first_step: 0, last_step: 6, materialize: false },
        { value_id: 1, op: "BUFFER", shape: [2], dtype: "float32", bytes: 8, first_step: 1, last_step: 6, materialize: false },
        { value_id: 2, op: "BUFFER", shape: [2], dtype: "float32", bytes: 8, first_step: 2, last_step: 6, materialize: false },
        { value_id: 3, op: "BUFFER", shape: [2], dtype: "float32", bytes: 8, first_step: 3, last_step: 6, materialize: false },
        { value_id: 4, op: "ADAM_UPDATE_M", shape: [2], dtype: "float32", bytes: 8, first_step: 4, last_step: 6, materialize: false },
        { value_id: 5, op: "ADAM_UPDATE_V", shape: [2], dtype: "float32", bytes: 8, first_step: 5, last_step: 6, materialize: false },
        { value_id: 6, op: "ADAM_UPDATE_PARAM", shape: [2], dtype: "float32", bytes: 8, first_step: 6, last_step: 6, materialize: true },
      ],
      root_id: 6,
      materialization_boundary: "root",
      peak_live_bytes: 56,
      has_custom_ops: false,
    });
    expect(plan.steps.map((step) => step.op)).toContain("ADAM_UPDATE_PARAM");
  });
});
