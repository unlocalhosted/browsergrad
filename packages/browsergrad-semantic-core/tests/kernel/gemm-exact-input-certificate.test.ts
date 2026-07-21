import { describe, expect, it } from "vitest";

import {
  LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA,
  LOGICAL_GEMM_EXACT_F32_INPUT_PROFILE,
  LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT,
  copyCertifiedLogicalGemmExactF32Inputs,
  createVerifiedDenseLogicalGemmTileArtifacts,
  createVerifiedLogicalGemmExactF32InputCertificate,
  decodeLogicalGemmExactF32InputCertificate,
  logicalGemmExactF32InputCertificatePayload,
  verifyLogicalGemmExactF32InputCertificate,
  type CertifyLogicalGemmExactF32InputsRequest,
  type LogicalGemmExactF32InputCertificatePayloadV1,
  type VerifiedLogicalGemmExactF32InputCertificate,
} from "../../src/kernel";
import {
  KERNEL_DIAGNOSTIC_CODES,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  parseWireU64,
  sha256Hex,
  type JsonValue,
  type WireU64,
} from "../../src/schema";

const wire = (value: string): WireU64 => parseWireU64(value);

function f32Bytes(values: readonly number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return bytes;
}

async function diagnostic(run: () => Promise<unknown> | unknown): Promise<SemanticSchemaError> {
  try {
    await run();
    throw new Error("expected semantic failure");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    return error as SemanticSchemaError;
  }
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function envelope(payload: LogicalGemmExactF32InputCertificatePayloadV1): JsonValue {
  return {
    schema: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: { id: "exact-input-test", version: "1" },
    artifactId: "exact-input-certificate",
    requiredExtensions: [],
    payload,
  };
}

async function oneByOneRequest(lhs: number, rhs: number) {
  const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
    m: wire("1"), n: wire("1"), k: wire("1"),
    logicalTile: { m: wire("1"), n: wire("1"), k: wire("1") },
  });
  const request: CertifyLogicalGemmExactF32InputsRequest = {
    operationId: artifacts.operationId,
    inputs: { lhs: f32Bytes([lhs]), rhs: f32Bytes([rhs]) },
  };
  return { artifacts, request };
}

describe("logical GEMM exact f32 input certificate", () => {
  it("binds one strict logical specialization to concrete allocation bytes and an exact arithmetic proof", async () => {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("2"), n: wire("2"), k: wire("3"),
      logicalTile: { m: wire("8"), n: wire("8"), k: wire("4") },
    });
    const lhs = f32Bytes([1, 2, 3, 4, 5, 6]);
    const rhs = f32Bytes([7, 8, 9, 10, 11, 12]);
    const constructed = await createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      { operationId: artifacts.operationId, inputs: { lhs, rhs } },
    );
    const payload = logicalGemmExactF32InputCertificatePayload(constructed.certificate);

    expect(payload.logicalGemmSemanticHash).toBe(artifacts.kernelSemanticHash);
    expect(payload.specializationHash).toBe(constructed.specializationHash);
    expect(payload.inputs).toEqual({
      lhs: { algorithm: "sha256", allocationByteLength: "24", digest: await sha256Hex(lhs) },
      rhs: { algorithm: "sha256", allocationByteLength: "24", digest: await sha256Hex(rhs) },
    });
    expect(payload.proof).toEqual({
      profile: LOGICAL_GEMM_EXACT_F32_INPUT_PROFILE,
      byteOrder: "little-endian",
      exactIntegerLimit: "16777216",
      multiplyAdds: "12",
      maximumOutputSum: "154",
      guarantees: {
        inputs: "finite-nonnegative-integer-f32-with-positive-zero",
        products: "every-product-exact-f32",
        partialSums: "every-subset-sum-exact-f32",
        strictLogicalPolicy: "increasing-k-rne-separate-multiply-add-preserved",
        contraction: "value-preserving-on-certified-inputs",
        reassociation: "value-preserving-on-certified-inputs",
        f32Output: "bit-exact-on-certified-inputs",
      },
    });
    expect(LOGICAL_GEMM_EXACT_F32_INTEGER_LIMIT).toBe(1n << 24n);
  });

  it("retains private snapshots so later caller mutation cannot change the certified upload bytes", async () => {
    const { artifacts, request } = await oneByOneRequest(3, 5);
    const constructed = await createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      request,
    );
    request.inputs.lhs.fill(0xff);
    request.inputs.rhs.fill(0xff);

    const first = copyCertifiedLogicalGemmExactF32Inputs(constructed.certificate);
    expect([...first.lhs]).toEqual([...f32Bytes([3])]);
    expect([...first.rhs]).toEqual([...f32Bytes([5])]);
    first.lhs.fill(0);
    first.rhs.fill(0);
    const second = copyCertifiedLogicalGemmExactF32Inputs(constructed.certificate);
    expect([...second.lhs]).toEqual([...f32Bytes([3])]);
    expect([...second.rhs]).toEqual([...f32Bytes([5])]);
  });

  it("decodes only after recomputing the proof and byte commitments", async () => {
    const { artifacts, request } = await oneByOneRequest(3, 5);
    const constructed = await createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      request,
    );
    const payload = logicalGemmExactF32InputCertificatePayload(constructed.certificate);
    const decoded = await decodeLogicalGemmExactF32InputCertificate(
      canonicalJsonBytes(envelope(payload)),
      artifacts.layout,
      artifacts.kernel,
      request,
    );
    expect(logicalGemmExactF32InputCertificatePayload(decoded)).toEqual(payload);

    const changedBytes = { ...request, inputs: { lhs: f32Bytes([4]), rhs: request.inputs.rhs } };
    expect((await diagnostic(() => decodeLogicalGemmExactF32InputCertificate(
      canonicalJsonBytes(envelope(payload)),
      artifacts.layout,
      artifacts.kernel,
      changedBytes,
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidArtifact);
  });

  it("rejects closed-record mutations and authority forgeries", async () => {
    const { artifacts, request } = await oneByOneRequest(3, 5);
    const constructed = await createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      request,
    );
    const base = logicalGemmExactF32InputCertificatePayload(constructed.certificate);
    const mutations: Array<(payload: Record<string, unknown>) => void> = [
      (payload) => { payload.backend = "webgpu"; },
      (payload) => { payload.specializationHash = "0".repeat(64); },
      (payload) => {
        const proof = payload.proof as Record<string, unknown>;
        proof.maximumOutputSum = "14";
      },
      (payload) => {
        const inputs = payload.inputs as { lhs: Record<string, unknown> };
        inputs.lhs.digest = "0".repeat(64);
      },
      (payload) => {
        const proof = payload.proof as { guarantees: Record<string, unknown> };
        proof.guarantees.contraction = "allowed";
      },
    ];
    for (const mutate of mutations) {
      const payload = clone(base) as unknown as Record<string, unknown>;
      mutate(payload);
      expect((await diagnostic(() => verifyLogicalGemmExactF32InputCertificate(
        envelope(payload as unknown as LogicalGemmExactF32InputCertificatePayloadV1),
        artifacts.layout,
        artifacts.kernel,
        request,
      ))).diagnostic.code).toMatch(/^BG-KERNEL-/u);
    }
    expect((await diagnostic(() => logicalGemmExactF32InputCertificatePayload(
      base as unknown as VerifiedLogicalGemmExactF32InputCertificate,
    ))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.unverifiedArtifact);
    expect((await diagnostic(() => copyCertifiedLogicalGemmExactF32Inputs(
      Object.freeze({}) as VerifiedLogicalGemmExactF32InputCertificate,
    ))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.unverifiedArtifact);
  });

  it("accepts the inclusive 2^24 total and rejects an inexact product or larger partial sum", async () => {
    const boundary = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("1"), n: wire("1"), k: wire("2"),
      logicalTile: { m: wire("1"), n: wire("1"), k: wire("1") },
    });
    const accepted = await createVerifiedLogicalGemmExactF32InputCertificate(
      boundary.layout,
      boundary.kernel,
      {
        operationId: boundary.operationId,
        inputs: { lhs: f32Bytes([8_388_608, 8_388_608]), rhs: f32Bytes([1, 1]) },
      },
    );
    expect(logicalGemmExactF32InputCertificatePayload(accepted.certificate).proof.maximumOutputSum).toBe("16777216");

    const product = await oneByOneRequest(4_097, 4_097);
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      product.artifacts.layout,
      product.artifacts.kernel,
      product.request,
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile);

    const overflow = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: wire("1"), n: wire("1"), k: wire("3"),
      logicalTile: { m: wire("1"), n: wire("1"), k: wire("1") },
    });
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      overflow.layout,
      overflow.kernel,
      {
        operationId: overflow.operationId,
        inputs: { lhs: f32Bytes([8_388_608, 8_388_608, 1]), rhs: f32Bytes([1, 1, 1]) },
      },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile);
  });

  it("rejects values outside the exact nonnegative integer input domain", async () => {
    const negativeZero = new Uint8Array(4);
    new DataView(negativeZero.buffer).setUint32(0, 0x80000000, true);
    const cases: readonly Uint8Array[] = [
      negativeZero,
      f32Bytes([-1]),
      f32Bytes([0.5]),
      f32Bytes([Number.POSITIVE_INFINITY]),
      f32Bytes([Number.NaN]),
      f32Bytes([33_554_432]),
      Uint8Array.of(1, 0, 0, 0),
    ];
    for (const lhs of cases) {
      const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
        m: wire("1"), n: wire("1"), k: wire("1"),
        logicalTile: { m: wire("1"), n: wire("1"), k: wire("1") },
      });
      expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
        artifacts.layout,
        artifacts.kernel,
        { operationId: artifacts.operationId, inputs: { lhs, rhs: f32Bytes([1]) } },
      ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile);
    }
  });

  it("enforces byte, proof-work, time, cancellation, storage, and request-shape limits", async () => {
    const { artifacts, request } = await oneByOneRequest(3, 5);
    const cases: readonly [Partial<CertifyLogicalGemmExactF32InputsRequest>, string][] = [
      [{ maxInputBytes: 7 }, KERNEL_DIAGNOSTIC_CODES.resourceLimit],
      [{ maxProofSteps: 2 }, KERNEL_DIAGNOSTIC_CODES.resourceLimit],
      [{ maxCertificationMs: 0 }, KERNEL_DIAGNOSTIC_CODES.resourceLimit],
    ];
    for (const [extra, code] of cases) {
      expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
        artifacts.layout,
        artifacts.kernel,
        { ...request, ...extra },
      ))).diagnostic.code).toBe(code);
    }
    const controller = new AbortController();
    controller.abort();
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      { ...request, signal: controller.signal },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.resourceLimit);

    class DerivedBytes extends Uint8Array {}
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      { ...request, inputs: { lhs: new DerivedBytes(4), rhs: request.inputs.rhs } },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const shared = typeof SharedArrayBuffer === "undefined" ? undefined : new Uint8Array(new SharedArrayBuffer(4));
    if (shared !== undefined) {
      expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
        artifacts.layout,
        artifacts.kernel,
        { ...request, inputs: { lhs: shared, rhs: request.inputs.rhs } },
      ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
    }

    const resizableBuffer = Reflect.construct(ArrayBuffer, [4, { maxByteLength: 8 }]) as ArrayBuffer;
    const resizable = new Uint8Array(resizableBuffer);
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      { ...request, inputs: { lhs: resizable, rhs: request.inputs.rhs } },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const detached = f32Bytes([3]);
    structuredClone(detached.buffer, { transfer: [detached.buffer] });
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      { ...request, inputs: { lhs: detached, rhs: request.inputs.rhs } },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const misaligned = new Uint8Array(new ArrayBuffer(5), 1, 4);
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      { ...request, inputs: { lhs: misaligned, rhs: request.inputs.rhs } },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);

    const overlappingStorage = f32Bytes([3]);
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      { ...request, inputs: { lhs: overlappingStorage, rhs: overlappingStorage } },
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.aliasConflict);

    const getterRequest = Object.defineProperty({}, "operationId", {
      enumerable: true,
      get: () => artifacts.operationId,
    });
    Object.defineProperty(getterRequest, "inputs", { enumerable: true, value: request.inputs });
    expect((await diagnostic(() => createVerifiedLogicalGemmExactF32InputCertificate(
      artifacts.layout,
      artifacts.kernel,
      getterRequest as CertifyLogicalGemmExactF32InputsRequest,
    ))).diagnostic.code).toBe(KERNEL_DIAGNOSTIC_CODES.invalidBinding);
  });
});
