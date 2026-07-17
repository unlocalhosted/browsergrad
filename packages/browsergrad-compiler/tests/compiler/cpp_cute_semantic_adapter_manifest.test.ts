import {
  canonicalJsonBytes,
  sha256Hex,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_BYTE_LIMIT,
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_DECODE_LIMITS,
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_MANIFEST_ID,
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
  canonicalCppCuteSemanticAdapterManifestBytes,
  cppCuteSemanticAdapterManifestResourceBytes,
  cppCuteSemanticAdapterWarningArgv,
  decodeCppCuteSemanticAdapterManifest,
  deriveCppCuteSemanticAdapterManifestId,
  unwrapPreparedCppCuteSemanticAdapterManifest,
  type PreparedCppCuteSemanticAdapterManifest,
} from "../../src/cpp_cute_semantic_adapter_manifest.js";

type MutableJson = Record<string, unknown>;

function mutableResource(): MutableJson {
  return JSON.parse(new TextDecoder().decode(
    cppCuteSemanticAdapterManifestResourceBytes(),
  )) as MutableJson;
}

function body(value: MutableJson): MutableJson {
  return value.body as MutableJson;
}

function objectField(value: MutableJson, name: string): MutableJson {
  return value[name] as MutableJson;
}

function arrayField(value: MutableJson, name: string): MutableJson[] {
  return value[name] as MutableJson[];
}

function canonicalBytes(value: MutableJson): Uint8Array {
  return canonicalJsonBytes(value as JsonValue, {
    limits: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_DECODE_LIMITS,
  });
}

async function expectDecodeError(
  value: Uint8Array,
  code: string,
  path?: string,
): Promise<void> {
  await expect(decodeCppCuteSemanticAdapterManifest(value)).rejects.toMatchObject({
    code,
    ...(path === undefined ? {} : { path }),
  });
}

describe("C++/CuTe semantic-adapter manifest", () => {
  it("strict-decodes one pinned design authority without claiming a Clang invocation", async () => {
    const resource = cppCuteSemanticAdapterManifestResourceBytes();
    const prepared = await decodeCppCuteSemanticAdapterManifest(resource);

    expect(prepared).toEqual({
      manifestId: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_MANIFEST_ID,
      semanticAdapterId: "browsergrad.compiler.cpp-cute.clang-semantic-adapter@1",
      clangVersion: "22.1.8",
      resourceSha256: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
      resourceByteLength: 2_575,
      designAuthority: true,
      clangInvocationAuthorized: false,
    });
    expect(await sha256Hex(resource)).toBe(CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256);
    expect(canonicalCppCuteSemanticAdapterManifestBytes(prepared)).toEqual(resource);

    const manifest = unwrapPreparedCppCuteSemanticAdapterManifest(prepared).manifest;
    expect(await deriveCppCuteSemanticAdapterManifestId(manifest.body)).toBe(
      CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_MANIFEST_ID,
    );
    expect(manifest.body.clang).toEqual({ compilerId: "clang", version: "22.1.8" });
  });

  it("closes temporal rejection and keeps its diagnostic groups reserved", async () => {
    const prepared = await decodeCppCuteSemanticAdapterManifest(
      cppCuteSemanticAdapterManifestResourceBytes(),
    );
    const bodyValue = unwrapPreparedCppCuteSemanticAdapterManifest(prepared).manifest.body;

    expect(bodyValue.temporalMacros).toEqual({
      policyId: "browsergrad.compiler.cpp-cute.temporal-macros.reject@1",
      mode: "reject",
      macroNames: ["__DATE__", "__TIMESTAMP__", "__TIME__"],
      consultation: "forbidden",
      mutation: "forbidden",
      enforcement: "preprocessor-callback-before-expansion",
      diagnosticCodes: {
        consultation: "browsergrad.cpp-cute:temporal-macro-forbidden",
        mutation: "browsergrad.cpp-cute:temporal-macro-mutation-forbidden",
      },
      defenseInDepthArgv: [
        "-Werror=builtin-macro-redefined",
        "-Werror=date-time",
        "-Werror=macro-redefined",
      ],
    });
    expect(bodyValue.warningPolicyRegistry.reservedClangDiagnosticGroups).toEqual([
      "builtin-macro-redefined",
      "date-time",
      "macro-redefined",
    ]);
    expect(bodyValue.warningPolicyRegistry.entries.map((entry) => entry.clangDiagnosticGroup))
      .not.toEqual(expect.arrayContaining(["date-time", "builtin-macro-redefined", "macro-redefined"]));
  });

  it("pins six namespaced warning IDs to exact disposition argv arrays", async () => {
    const prepared = await decodeCppCuteSemanticAdapterManifest(
      cppCuteSemanticAdapterManifestResourceBytes(),
    );
    const entries = unwrapPreparedCppCuteSemanticAdapterManifest(prepared)
      .manifest.body.warningPolicyRegistry.entries;

    expect(entries.map((entry) => [entry.policyId, entry.clangDiagnosticGroup])).toEqual([
      ["clang.deprecated-declarations", "deprecated-declarations"],
      ["clang.sign-compare", "sign-compare"],
      ["clang.unknown-pragmas", "unknown-pragmas"],
      ["clang.unused-function", "unused-function"],
      ["clang.unused-parameter", "unused-parameter"],
      ["clang.unused-variable", "unused-variable"],
    ]);
    for (const entry of entries) {
      const group = entry.clangDiagnosticGroup;
      expect(entry.argv).toEqual({
        ignore: [`-Wno-${group}`],
        warn: [`-W${group}`, `-Wno-error=${group}`],
        error: [`-W${group}`, `-Werror=${group}`],
      });
    }
    expect(cppCuteSemanticAdapterWarningArgv(
      prepared,
      "clang.unused-parameter",
      "warn",
    )).toEqual(["-Wunused-parameter", "-Wno-error=unused-parameter"]);
  });

  it("rejects noncanonical, duplicate-key, trailing, and oversized bytes", async () => {
    const resource = cppCuteSemanticAdapterManifestResourceBytes();
    const parsed = mutableResource();
    await expectDecodeError(
      new TextEncoder().encode(JSON.stringify(parsed, null, 2)),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-NONCANONICAL-BYTES",
      "$bytes",
    );

    const text = new TextDecoder().decode(resource);
    await expectDecodeError(
      new TextEncoder().encode(text.replace(
        "{\"body\":",
        "{\"body\":null,\"body\":",
      )),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-INVALID",
      "$bytes",
    );

    const trailing = new Uint8Array(resource.byteLength + 1);
    trailing.set(resource);
    trailing[resource.byteLength] = 0x20;
    await expectDecodeError(
      trailing,
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-NONCANONICAL-BYTES",
      "$bytes",
    );
    await expectDecodeError(
      new Uint8Array(CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_BYTE_LIMIT + 1),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-RESOURCE-LIMIT",
      "$bytes",
    );
  });

  it("rejects unknown fields, version drift, Clang drift, and reordered closed inventories", async () => {
    const unknown = mutableResource();
    body(unknown).unknown = true;
    await expectDecodeError(
      canonicalBytes(unknown),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-INVALID",
      "$.body",
    );

    const version = mutableResource();
    objectField(version, "version").minor = 1;
    await expectDecodeError(
      canonicalBytes(version),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-UNSUPPORTED-VERSION",
      "$.version.minor",
    );

    const clang = mutableResource();
    objectField(body(clang), "clang").version = "20.1.9";
    await expectDecodeError(
      canonicalBytes(clang),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-INVALID",
      "$.body.clang.version",
    );

    const reordered = mutableResource();
    const registry = objectField(body(reordered), "warningPolicyRegistry");
    arrayField(registry, "entries").reverse();
    await expectDecodeError(
      canonicalBytes(reordered),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-INVALID",
      "$.body.warningPolicyRegistry.entries[0].policyId",
    );
  });

  it("rejects manifest identity drift before it can mint authority", async () => {
    const value = mutableResource();
    value.manifestId = `bg.cpp.semantic-adapter.sha256.${"0".repeat(64)}`;
    await expectDecodeError(
      canonicalBytes(value),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-HASH-MISMATCH",
      "$.manifestId",
    );
  });

  it("snapshots input bytes and admits only plain unshared Uint8Array instances", async () => {
    const bytes = cppCuteSemanticAdapterManifestResourceBytes();
    const pending = decodeCppCuteSemanticAdapterManifest(bytes);
    bytes.fill(0);
    await expect(pending).resolves.toMatchObject({ designAuthority: true });

    class ByteSubclass extends Uint8Array {}
    await expectDecodeError(
      new ByteSubclass(cppCuteSemanticAdapterManifestResourceBytes()),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-INVALID",
      "$bytes",
    );
    await expectDecodeError(
      new Proxy(cppCuteSemanticAdapterManifestResourceBytes(), {}),
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-INVALID",
      "$bytes",
    );
  });

  it("honors cancellation without minting partial authority", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(decodeCppCuteSemanticAdapterManifest(
      cppCuteSemanticAdapterManifestResourceBytes(),
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-CANCELLED",
      path: "$options.signal",
    });
  });

  it("keeps prepared authority opaque, immutable, and realm-local", async () => {
    const prepared = await decodeCppCuteSemanticAdapterManifest(
      cppCuteSemanticAdapterManifestResourceBytes(),
    );
    const record = unwrapPreparedCppCuteSemanticAdapterManifest(prepared);
    expect(() => {
      (record.manifest.body.clang as { version: string }).version = "forged";
    }).toThrow(TypeError);

    const copy = canonicalCppCuteSemanticAdapterManifestBytes(prepared);
    copy.fill(0);
    expect(canonicalCppCuteSemanticAdapterManifestBytes(prepared)[0]).not.toBe(0);

    await expect(Promise.resolve().then(() =>
      unwrapPreparedCppCuteSemanticAdapterManifest({
        ...prepared,
      } as PreparedCppCuteSemanticAdapterManifest),
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-UNVERIFIED",
      path: "$prepared",
    });
  });
});
