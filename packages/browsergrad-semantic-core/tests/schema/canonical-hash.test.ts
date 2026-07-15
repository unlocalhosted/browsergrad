import { describe, expect, it } from "vitest";

import {
  canonicalizeJson,
  derivePureValueId,
  deriveScopedEntityId,
  hashNamedComponents,
  hashSemanticArtifact,
  parseWireJson,
  sha256Hex,
  validateWireEnvelope,
  type JsonValue,
  type VerifiedArtifact,
} from "../../src/schema";
import { verifyWireArtifact } from "../../src/schema/envelope";

function envelope(source: string): VerifiedArtifact<JsonValue> {
  return verifyWireArtifact(parseWireJson(source), {
    schema: "browsergrad.layout",
    supportedMajor: 1,
    supportedMinor: 0,
    validatePayload: (value) => value,
  });
}

describe("canonical JSON and hashing", () => {
  it("sorts object keys by UTF-16 code units and preserves array order", () => {
    const first = { "\ue000": 1, "😀": 2, a: [2, 1] };
    const second = { a: [2, 1], "😀": 2, "\ue000": 1 };
    expect(canonicalizeJson(first)).toBe('{"a":[2,1],"😀":2,"":1}');
    expect(canonicalizeJson(second)).toBe(canonicalizeJson(first));
  });

  it("matches the SHA-256 standard vector", async () => {
    expect(await sha256Hex(new TextEncoder().encode("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });

  it("hashes semantic content but excludes transport and provenance fields", async () => {
    const first = envelope('{"schema":"browsergrad.layout","version":{"major":1,"minor":0},"producer":{"id":"a","version":"1"},"artifactId":"transport-a","payload":{"shape":[2,3]},"requiredExtensions":[],"optionalMetadata":{"note":"a"}}');
    const second = envelope('{"schema":"browsergrad.layout","version":{"major":1,"minor":0},"producer":{"id":"b","version":"9"},"artifactId":"transport-b","payload":{"shape":[2,3]},"requiredExtensions":[],"optionalMetadata":{"note":"b"}}');
    const changed = envelope('{"schema":"browsergrad.layout","version":{"major":1,"minor":0},"producer":{"id":"a","version":"1"},"artifactId":"transport-a","payload":{"shape":[2,4]},"requiredExtensions":[]}');

    expect(await hashSemanticArtifact(first)).toBe(await hashSemanticArtifact(second));
    expect(await hashSemanticArtifact(first)).not.toBe(await hashSemanticArtifact(changed));
  });

  it("keeps identical entity content distinct by scoped canonical position", async () => {
    const valueA = await derivePureValueId("dim", { kind: "const", value: "4" });
    const valueB = await derivePureValueId("dim", { value: "4", kind: "const" });
    expect(valueA).toBe(valueB);

    const allocationA = await deriveScopedEntityId("artifact-1", "allocation", "allocations[0]");
    const allocationB = await deriveScopedEntityId("artifact-1", "allocation", "allocations[1]");
    expect(allocationA).not.toBe(allocationB);
    expect(allocationA).toMatch(/^bg\.entity\.allocation\.sha256\.[0-9a-f]{64}$/u);
  });

  it("requires cache-key components to be explicitly named", async () => {
    await expect(hashNamedComponents({})).rejects.toThrow(/BG-SCHEMA-NONCANONICAL-VALUE/u);
    await expect(hashNamedComponents({ "": 1 })).rejects.toThrow(/BG-SCHEMA-NONCANONICAL-VALUE/u);
  });

  it("rejects structurally forged envelopes at the semantic hash boundary", async () => {
    const forged = validateWireEnvelope(parseWireJson('{"schema":"browsergrad.layout","version":{"major":1,"minor":0},"producer":{"id":"tests","version":"1"},"artifactId":"a","payload":{},"requiredExtensions":[]}'), {
      schema: "browsergrad.layout",
      supportedMajor: 1,
      supportedMinor: 0,
    }) as unknown as VerifiedArtifact<JsonValue>;
    await expect(hashSemanticArtifact(forged)).rejects.toThrow(/BG-SCHEMA-UNVERIFIED-ARTIFACT/u);
  });
});
