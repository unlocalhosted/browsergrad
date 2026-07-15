import { describe, expect, it } from "vitest";

import {
  DENSE_PERMUTATION_VIEW_COPY_FIXTURES,
  fixtureExtentNumbers,
} from "../../../test-support/dense-permutation-view-copy-fixtures";

import {
  assertPreparedTensorPlanSemanticRequests,
  prepareTensorPlanSemanticRequests,
} from "../src/tensor_plan_semantics";

const LAYOUT_HASH = "f204e22acb50681d6a52703131d9c13d0b1424da476dfd004b5a5bc3db25c1a2";
const KERNEL_HASH = "d189f64cb8d148fe242978e0657f1ecd3747383908b3316ee6d8aa31a65d699a";

describe("tensor-plan semantic request preparation", () => {
  it.each(DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases)(
    "constructs the shared $id artifacts solely from the request",
    async (fixture) => {
      const prepared = await prepareTensorPlanSemanticRequests(
        fixturePermutationPlan(fixture.request.inputShape, fixture.outputShape, 20, 21),
        {
          schema: "browsergrad.jit.tensor-plan-semantic-requests",
          version: { major: 1, minor: 0 },
          requests: [{ ...fixture.request, valueId: 21 }],
        },
      );

      expect(prepared.requests).toHaveLength(1);
      expect(prepared.requests[0]).toMatchObject({
        ...fixture.request,
        valueId: 21,
        layoutSemanticHash: fixture.layoutSemanticHash,
        kernelSemanticHash: fixture.kernelSemanticHash,
      });
    },
  );

  it("strictly prepares the canonical rank-2 transpose beside the frozen plan", async () => {
    const prepared = await prepareTensorPlanSemanticRequests(
      permutationPlan(10, 11),
      JSON.stringify(requestEnvelope(11)),
    );

    expect(prepared).toMatchObject({
      schema: "browsergrad.jit.tensor-plan-semantic-requests",
      version: { major: 1, minor: 0 },
      requests: [{
        kind: "dense-permutation-view-copy",
        valueId: 11,
        inputShape: ["2", "3"],
        axes: [1, 0],
        dtype: "f32",
        layoutSemanticHash: LAYOUT_HASH,
        kernelSemanticHash: KERNEL_HASH,
      }],
    });
    expect(prepared.requests[0]?.semanticSpecializationHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(prepared.requests[0]?.wgslModuleHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(Object.isFrozen(prepared)).toBe(true);
    expect(Object.isFrozen(prepared.version)).toBe(true);
    expect(Object.isFrozen(prepared.requests)).toBe(true);
    expect(Object.isFrozen(prepared.requests[0])).toBe(true);
    expect(Object.isFrozen(prepared.requests[0]?.inputShape)).toBe(true);
    expect(Object.isFrozen(prepared.requests[0]?.axes)).toBe(true);
    expect(() => assertPreparedTensorPlanSemanticRequests(prepared)).not.toThrow();
  });

  it("keeps routing valueId outside semantic identity", async () => {
    const first = await prepareTensorPlanSemanticRequests(
      permutationPlan(0, 1),
      requestEnvelope(1),
    );
    const second = await prepareTensorPlanSemanticRequests(
      permutationPlan(100, 101),
      requestEnvelope(101),
    );

    expect(first.requests[0]?.valueId).toBe(1);
    expect(second.requests[0]?.valueId).toBe(101);
    expect(first.requests[0]?.layoutSemanticHash).toBe(second.requests[0]?.layoutSemanticHash);
    expect(first.requests[0]?.kernelSemanticHash).toBe(second.requests[0]?.kernelSemanticHash);
    expect(first.requests[0]?.semanticSpecializationHash)
      .toBe(second.requests[0]?.semanticSpecializationHash);
    expect(first.requests[0]?.wgslModuleHash).toBe(second.requests[0]?.wgslModuleHash);
  });

  it("requires exact request coverage and plan correlation", async () => {
    await expect(prepareTensorPlanSemanticRequests(
      permutationPlan(0, 1),
      { ...requestEnvelope(1), requests: [] },
    )).rejects.toThrow(/expected exactly 1 request/);

    await expect(prepareTensorPlanSemanticRequests(
      permutationPlan(0, 1),
      requestEnvelope(7),
    )).rejects.toThrow(/must match plan PERMUTE value 1/);

    await expect(prepareTensorPlanSemanticRequests(
      permutationPlan(0, 1),
      requestEnvelope(1, { inputShape: ["3", "2"] }),
    )).rejects.toThrow(/plan projection differ at axis 0/);

    const planWithLegacyMeaning = permutationPlan(0, 1);
    planWithLegacyMeaning.steps[1]!.arg = { axes: [1, 0] };
    await expect(prepareTensorPlanSemanticRequests(
      planWithLegacyMeaning,
      requestEnvelope(1),
    )).rejects.toThrow(/must erase legacy arg meaning/);
  });

  it.each([
    ["open envelope", { ...requestEnvelope(1), extra: true }, /expected closed fields requests, schema, version/],
    ["open version", { ...requestEnvelope(1), version: { major: 1, minor: 0, patch: 0 } }, /expected closed fields major, minor/],
    ["wrong version", { ...requestEnvelope(1), version: { major: 1, minor: 1 } }, /only semantic request version 1.0/],
    ["open request", requestEnvelope(1, { extra: true }), /expected closed fields axes, dtype, inputShape, kind, valueId/],
    ["wrong dtype", requestEnvelope(1, { dtype: "float32" }), /requires f32/],
    ["noncanonical extent", requestEnvelope(1, { inputShape: ["02", "3"] }), /WireI64 must be canonical/],
    ["zero extent", requestEnvelope(1, { inputShape: ["0", "3"] }), /extent must be positive/],
    ["duplicate axes", requestEnvelope(1, { axes: [0, 0] }), /exact non-negative permutation/],
  ] as const)("rejects %s", async (_name, envelope, expected) => {
    await expect(prepareTensorPlanSemanticRequests(
      permutationPlan(0, 1),
      envelope,
    )).rejects.toThrow(expected);
  });

  it("rejects forged prepared metadata", () => {
    expect(() => assertPreparedTensorPlanSemanticRequests({
      schema: "browsergrad.jit.tensor-plan-semantic-requests",
      version: { major: 1, minor: 0 },
      requests: [],
    })).toThrow(/not prepared by this module instance/);
  });
});

function requestEnvelope(
  valueId: number,
  override: Readonly<Record<string, unknown>> = {},
) {
  return {
    schema: "browsergrad.jit.tensor-plan-semantic-requests",
    version: { major: 1, minor: 0 },
    requests: [{
      kind: "dense-permutation-view-copy",
      valueId,
      inputShape: ["2", "3"],
      axes: [1, 0],
      dtype: "f32",
      ...override,
    }],
  };
}

function permutationPlan(
  inputValueId: number,
  outputValueId: number,
) {
  return {
    steps: [
      {
        step: 0,
        value_id: inputValueId,
        op: "BUFFER",
        input_ids: [],
        shape: [2, 3],
        dtype: "float32",
        arg: "buffer:x",
      },
      {
        step: 1,
        value_id: outputValueId,
        op: "PERMUTE",
        input_ids: [inputValueId],
        shape: [3, 2],
        dtype: "float32",
        arg: null as unknown,
      },
    ],
    buffers: [
      {
        value_id: inputValueId,
        op: "BUFFER",
        shape: [2, 3],
        dtype: "float32",
        bytes: 24,
        first_step: 0,
        last_step: 1,
        materialize: false,
      },
      {
        value_id: outputValueId,
        op: "PERMUTE",
        shape: [3, 2],
        dtype: "float32",
        bytes: 24,
        first_step: 1,
        last_step: 1,
        materialize: true,
      },
    ],
    root_id: outputValueId,
    materialization_boundary: "root",
    peak_live_bytes: 48,
    has_custom_ops: false,
  };
}

function fixturePermutationPlan(
  inputShape: readonly string[],
  outputShape: readonly string[],
  inputValueId: number,
  outputValueId: number,
) {
  const input = fixtureExtentNumbers(inputShape);
  const output = fixtureExtentNumbers(outputShape);
  const bytes = input.reduce((product, extent) => product * extent, 1) * 4;
  return {
    steps: [
      {
        step: 0,
        value_id: inputValueId,
        op: "BUFFER",
        input_ids: [],
        shape: input,
        dtype: "float32",
        arg: "buffer:x",
      },
      {
        step: 1,
        value_id: outputValueId,
        op: "PERMUTE",
        input_ids: [inputValueId],
        shape: output,
        dtype: "float32",
        arg: null,
      },
    ],
    buffers: [
      {
        value_id: inputValueId,
        op: "BUFFER",
        shape: input,
        dtype: "float32",
        bytes,
        first_step: 0,
        last_step: 1,
        materialize: false,
      },
      {
        value_id: outputValueId,
        op: "PERMUTE",
        shape: output,
        dtype: "float32",
        bytes,
        first_step: 1,
        last_step: 1,
        materialize: true,
      },
    ],
    root_id: outputValueId,
    materialization_boundary: "root",
    peak_live_bytes: bytes * 2,
    has_custom_ops: false,
  };
}
