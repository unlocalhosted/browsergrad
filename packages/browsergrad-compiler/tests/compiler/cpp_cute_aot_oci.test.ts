import { describe, expect, it } from "vitest";
import {
  authorizeCppCuteAotOciMetadata,
  CppCuteAotOciError,
  inspectAuthorizedCppCuteAotOciMetadata,
  unwrapAuthorizedCppCuteAotOciMetadata,
  verifyCppCuteAotOciMetadata,
} from "../../src/cpp_cute_aot_oci.js";
import { CPP_CUTE_AOT_OCI_RESOURCE_LIMITS } from "../../src/cpp_cute_aot_policy.js";
import {
  createCppCuteAotOciFixture,
  defaultLayer,
  DEFAULT_DIFF_ID,
} from "./support/cpp_cute_aot_oci_fixtures.js";
import { createCppCuteAotRunnerFixture } from "./support/cpp_cute_aot_runner_fixtures.js";

const textEncoder = new TextEncoder();

describe("C++/CuTe AOT OCI manifest/config metadata", () => {
  it("separates cacheable metadata from plan authorization and local Docker presence", async () => {
    const fixture = await createCppCuteAotOciFixture();
    const verified = await verifyCppCuteAotOciMetadata(fixture.evidence);

    expect(verified).toEqual({
      manifest: {
        mediaType: "application/vnd.oci.image.manifest.v1+json",
        digest: fixture.manifestDigest,
        size: fixture.evidence.manifestBytes.byteLength,
      },
      config: {
        mediaType: "application/vnd.oci.image.config.v1+json",
        digest: fixture.configDigest,
        size: fixture.evidence.configBytes.byteLength,
      },
      layerCount: 1,
      totalLayerBytes: 123,
    });
    expect(Object.isFrozen(verified)).toBe(true);
    expect(Object.isFrozen(verified.manifest)).toBe(true);
    const authorized = authorizeCppCuteAotOciMetadata(fixture.plan, verified);
    expect(authorized).toEqual({
      jobId: fixture.plan.jobId,
      profileHash: fixture.plan.profileHash,
      executionPlanSha256: fixture.plan.executionPlanSha256,
      imageReference: fixture.plan.imageReference,
      manifestDigest: fixture.manifestDigest,
      configDigest: fixture.configDigest,
      layerCount: 1,
      totalLayerBytes: 123,
    });
    expect(unwrapAuthorizedCppCuteAotOciMetadata(authorized)).toEqual({
      plan: fixture.plan,
      metadata: verified,
    });
    expect(inspectAuthorizedCppCuteAotOciMetadata(authorized)).toMatchObject({
      plan: fixture.plan,
      metadata: verified,
      diffIds: [DEFAULT_DIFF_ID],
    });
    expect(() => authorizeCppCuteAotOciMetadata(fixture.plan, { ...verified } as never)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-AOT-OCI-UNVERIFIED" }),
    );
    expect(() => unwrapAuthorizedCppCuteAotOciMetadata({ ...authorized } as never)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-AOT-OCI-UNVERIFIED" }),
    );
    expect(() => inspectAuthorizedCppCuteAotOciMetadata({ ...authorized } as never)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-AOT-OCI-UNVERIFIED" }),
    );
    const wrongConfigPlan = await createCppCuteAotRunnerFixture({
      containerManifestDigest: fixture.manifestDigest,
      containerConfigDigest: `sha256:${"0".repeat(64)}`,
    });
    expect(() => authorizeCppCuteAotOciMetadata(wrongConfigPlan.plan, verified)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-DIGEST-MISMATCH",
        path: "$.plan.imageConfigDigest",
      }),
    );
  });

  it("retains exact ordered layer closure behind deeply frozen plan-neutral authority", async () => {
    const secondLayerDigest = `sha256:${"c".repeat(64)}`;
    const secondDiffId = `sha256:${"d".repeat(64)}`;
    const fixture = await createCppCuteAotOciFixture({
      layers: [defaultLayer(), defaultLayer({ digest: secondLayerDigest, size: 456 })],
      diffIds: [DEFAULT_DIFF_ID, secondDiffId],
    });
    const raw = await verifyCppCuteAotOciMetadata(fixture.evidence);
    const first = authorizeCppCuteAotOciMetadata(fixture.plan, raw);
    const secondRunner = await createCppCuteAotRunnerFixture({
      trustStoreSha256: "e".repeat(64),
      containerManifestDigest: fixture.manifestDigest,
      containerConfigDigest: fixture.configDigest,
    });
    const second = authorizeCppCuteAotOciMetadata(secondRunner.plan, raw);
    const projection = inspectAuthorizedCppCuteAotOciMetadata(first);

    expect(first.profileHash).not.toBe(second.profileHash);
    expect(unwrapAuthorizedCppCuteAotOciMetadata(first).metadata).toBe(raw);
    expect(unwrapAuthorizedCppCuteAotOciMetadata(second).metadata).toBe(raw);
    expect(projection.layers).toEqual([
      defaultLayer(),
      defaultLayer({ digest: secondLayerDigest, size: 456 }),
    ]);
    expect(projection.diffIds).toEqual([DEFAULT_DIFF_ID, secondDiffId]);
    expect(first.totalLayerBytes).toBe(579);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(projection)).toBe(true);
    expect(Object.isFrozen(projection.layers)).toBe(true);
    expect(Object.isFrozen(projection.layers[0])).toBe(true);
    expect(Object.isFrozen(projection.diffIds)).toBe(true);
    expect(() => { (projection.diffIds as string[])[0] = secondDiffId; }).toThrow(TypeError);
    expect(inspectAuthorizedCppCuteAotOciMetadata(first).diffIds).toEqual([DEFAULT_DIFF_ID, secondDiffId]);
  });

  it("rejects manifest, config, embedded-config digest, and exact-size drift", async () => {
    const fixture = await createCppCuteAotOciFixture();
    for (const key of ["manifestBytes", "configBytes"] as const) {
      const bytes = new Uint8Array(fixture.evidence[key].byteLength + 1);
      bytes[0] = 0x20;
      bytes.set(fixture.evidence[key], 1);
      const evidence = {
        ...fixture.evidence,
        [key]: bytes,
      };
      if (key === "configBytes") {
        await expect(verifyCppCuteAotOciMetadata(evidence)).rejects.toMatchObject({
          code: "BG-COMPILER-CPP-CUTE-AOT-OCI-DIGEST-MISMATCH",
          path: "$.manifest.config.digest",
        });
        continue;
      }
      const raw = await verifyCppCuteAotOciMetadata(evidence);
      expect(() => authorizeCppCuteAotOciMetadata(fixture.plan, raw)).toThrowError(
        expect.objectContaining({
          code: "BG-COMPILER-CPP-CUTE-AOT-OCI-DIGEST-MISMATCH",
          path: "$.plan.imageReference",
        }),
      );
    }

    const wrongConfig = await createCppCuteAotOciFixture({
      manifestConfigDigest: `sha256:${"f".repeat(64)}`,
    });
    await expect(verifyCppCuteAotOciMetadata(wrongConfig.evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-DIGEST-MISMATCH",
      path: "$.manifest.config.digest",
    });
    const wrongSize = await createCppCuteAotOciFixture({ manifestConfigSizeDelta: 1 });
    await expect(verifyCppCuteAotOciMetadata(wrongSize.evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-DIGEST-MISMATCH",
      path: "$.manifest.config.size",
    });
  });

  it("strict-decodes both raw resources and rejects duplicate keys or trailing input", async () => {
    const invalidManifestCases = [
      new Uint8Array(0),
      textEncoder.encode("{}{}"),
      textEncoder.encode('{"schemaVersion":2,"schemaVersion":2}'),
      new Uint8Array([0xff]),
    ];
    for (const rawManifestBytes of invalidManifestCases) {
      const fixture = await createCppCuteAotOciFixture({ rawManifestBytes });
      await expect(verifyCppCuteAotOciMetadata(fixture.evidence)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.manifestBytes",
      });
    }
    for (const rawConfigBytes of [
      textEncoder.encode("{}{}"),
      textEncoder.encode('{"architecture":"amd64","architecture":"amd64"}'),
      new Uint8Array([0xff]),
    ]) {
      const fixture = await createCppCuteAotOciFixture({ rawConfigBytes });
      await expect(verifyCppCuteAotOciMetadata(fixture.evidence)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.configBytes",
      });
    }
  });

  it("accepts only a closed self-contained OCI platform manifest", async () => {
    const cases = [
      {
        mutateManifest: (manifest: Record<string, unknown>) => { manifest.mediaType = "application/vnd.oci.image.index.v1+json"; },
        path: "$.manifest.mediaType",
      },
      {
        mutateManifest: (manifest: Record<string, unknown>) => { manifest.artifactType = "application/example"; },
        path: "$.manifest",
      },
      {
        mutateManifest: (manifest: Record<string, unknown>) => {
          (manifest.config as Record<string, unknown>).mediaType = "application/vnd.docker.container.image.v1+json";
        },
        path: "$.manifest.config.mediaType",
      },
      {
        mutateManifest: (manifest: Record<string, unknown>) => {
          (manifest.config as Record<string, unknown>).urls = ["https://example.invalid/config"];
        },
        path: "$.manifest.config",
      },
      {
        layers: [defaultLayer({ mediaType: "application/vnd.oci.image.layer.nondistributable.v1.tar+gzip" })],
        path: "$.manifest.layers[0].mediaType",
      },
      {
        layers: [defaultLayer({ urls: ["https://example.invalid/layer"] })],
        path: "$.manifest.layers[0]",
      },
      {
        layers: [defaultLayer({ size: 0 })],
        path: "$.manifest.layers[0].size",
      },
      {
        layers: [defaultLayer({ digest: `sha256:${"A".repeat(64)}` })],
        path: "$.manifest.layers[0].digest",
      },
    ];
    for (const testCase of cases) {
      const fixture = await createCppCuteAotOciFixture(testCase);
      await expect(verifyCppCuteAotOciMetadata(fixture.evidence)).rejects.toMatchObject({
        code: expect.stringMatching(/OCI-(?:INVALID|RESOURCE-LIMIT)$/u),
        path: testCase.path,
      });
    }
  });

  it("bounds layer descriptors and manifest annotations under policy identity", async () => {
    const zeroLayers = await createCppCuteAotOciFixture({ layers: [] });
    await expect(verifyCppCuteAotOciMetadata(zeroLayers.evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT",
      path: "$.manifest.layers",
    });
    const zeroDiffIds = await createCppCuteAotOciFixture({ diffIds: [] });
    await expect(verifyCppCuteAotOciMetadata(zeroDiffIds.evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT",
      path: "$.config.rootfs.diff_ids",
    });
    const tooManyLayers = Array.from(
      { length: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.layers + 1 },
      (_, index) => defaultLayer({ digest: `sha256:${index.toString(16).padStart(64, "0")}` }),
    );
    const layerFixture = await createCppCuteAotOciFixture({ layers: tooManyLayers });
    await expect(verifyCppCuteAotOciMetadata(layerFixture.evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT",
      path: "$.manifest.layers",
    });
    const hugeLayer = await createCppCuteAotOciFixture({
      layers: [defaultLayer({ size: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.layerBytes + 1 })],
    });
    await expect(verifyCppCuteAotOciMetadata(hugeLayer.evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT",
      path: "$.manifest.layers[0].size",
    });
    const aggregateLayers = await createCppCuteAotOciFixture({
      layers: [
        defaultLayer({ size: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.layerBytes }),
        defaultLayer({ digest: `sha256:${"c".repeat(64)}`, size: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.layerBytes }),
        defaultLayer({ digest: `sha256:${"d".repeat(64)}`, size: 1 }),
      ],
    });
    await expect(verifyCppCuteAotOciMetadata(aggregateLayers.evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT",
      path: "$.manifest.layers",
    });
    const annotations = Object.fromEntries(Array.from(
      { length: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.annotations + 1 },
      (_, index) => [`key-${index}`, "value"],
    ));
    const annotationFixture = await createCppCuteAotOciFixture({
      mutateManifest: (manifest) => { manifest.annotations = annotations; },
    });
    await expect(verifyCppCuteAotOciMetadata(annotationFixture.evidence)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT",
      path: "$.manifest.annotations",
    });
  });

  it("requires linux/amd64 rootfs identity and a completely empty image execution config", async () => {
    const cases = [
      {
        mutateConfig: (config: Record<string, unknown>) => { config.architecture = "arm64"; },
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-PLATFORM-MISMATCH",
        path: "$.config",
      },
      {
        mutateConfig: (config: Record<string, unknown>) => { config.os = "windows"; },
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-PLATFORM-MISMATCH",
        path: "$.config",
      },
      {
        mutateConfig: (config: Record<string, unknown>) => { config.variant = "v8"; },
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.config",
      },
      {
        mutateConfig: (config: Record<string, unknown>) => {
          (config.rootfs as Record<string, unknown>).type = "unknown";
        },
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.config.rootfs.type",
      },
      {
        diffIds: [`sha256:${"b".repeat(64)}`, `sha256:${"c".repeat(64)}`],
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-IMAGE-MISMATCH",
        path: "$.config.rootfs.diff_ids",
      },
      {
        diffIds: [`sha256:${"B".repeat(64)}`],
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.config.rootfs.diff_ids[0]",
      },
      {
        imageConfig: { Env: [] },
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-IMAGE-MISMATCH",
        path: "$.config.config",
      },
      {
        imageConfig: { Volumes: {} },
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-IMAGE-MISMATCH",
        path: "$.config.config",
      },
      {
        imageConfig: { Healthcheck: { Test: ["NONE"] } },
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-IMAGE-MISMATCH",
        path: "$.config.config",
      },
    ] as const;
    for (const testCase of cases) {
      const fixture = await createCppCuteAotOciFixture(testCase);
      await expect(verifyCppCuteAotOciMetadata(fixture.evidence)).rejects.toMatchObject({
        code: testCase.code,
        path: testCase.path,
      });
    }

    for (const options of [{ imageConfig: null }, { omitImageConfig: true }, { imageConfig: {} }]) {
      const fixture = await createCppCuteAotOciFixture(options);
      await expect(verifyCppCuteAotOciMetadata(fixture.evidence)).resolves.toBeDefined();
    }
  });

  it("validates closed bounded history and its exact rootfs correspondence", async () => {
    const cases = [
      {
        history: [],
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-IMAGE-MISMATCH",
        path: "$.config.history",
      },
      {
        history: [{ empty_layer: "true" }],
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.config.history[0].empty_layer",
      },
      {
        history: [{ created: "not-a-timestamp" }],
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.config.history[0].created",
      },
      {
        history: [{ created: "2026-02-30T00:00:00Z" }],
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.config.history[0].created",
      },
      {
        history: [{ unknown: true }],
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
        path: "$.config.history[0]",
      },
      {
        history: Array.from(
          { length: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.historyEntries + 1 },
          () => ({ empty_layer: true }),
        ),
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT",
        path: "$.config.history",
      },
    ] as const;
    for (const testCase of cases) {
      const fixture = await createCppCuteAotOciFixture(testCase);
      await expect(verifyCppCuteAotOciMetadata(fixture.evidence)).rejects.toMatchObject({
        code: testCase.code,
        path: testCase.path,
      });
    }
  });

  it("snapshots synchronously and rejects hostile evidence shapes before hashing", async () => {
    const fixture = await createCppCuteAotOciFixture();
    const manifestBytes = new Uint8Array(fixture.evidence.manifestBytes);
    const offsetBacking = new Uint8Array(fixture.evidence.configBytes.byteLength + 2);
    const offsetConfig = offsetBacking.subarray(1, 1 + fixture.evidence.configBytes.byteLength);
    offsetConfig.set(fixture.evidence.configBytes);
    const pending = verifyCppCuteAotOciMetadata({ manifestBytes, configBytes: offsetConfig });
    manifestBytes.fill(0);
    offsetConfig.fill(0);
    await expect(pending).resolves.toMatchObject({ manifest: { digest: fixture.manifestDigest } });

    const accessor = { ...fixture.evidence } as Record<string, unknown>;
    Object.defineProperty(accessor, "configBytes", {
      enumerable: true,
      get: () => fixture.evidence.configBytes,
    });
    await expect(verifyCppCuteAotOciMetadata(accessor as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
      path: "$.configBytes",
    });
    for (const evidence of [
      { ...fixture.evidence, extra: true },
      Object.assign(Object.create(null), fixture.evidence),
      [fixture.evidence.manifestBytes, fixture.evidence.configBytes],
    ]) {
      await expect(verifyCppCuteAotOciMetadata(evidence as never)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID",
      });
    }
    const disguisedWords = new Uint16Array(Math.ceil(fixture.evidence.configBytes.byteLength / 2));
    Object.setPrototypeOf(disguisedWords, Uint8Array.prototype);
    await expect(verifyCppCuteAotOciMetadata({
      ...fixture.evidence,
      configBytes: disguisedWords as unknown as Uint8Array,
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID" });
    if (typeof SharedArrayBuffer !== "undefined") {
      await expect(verifyCppCuteAotOciMetadata({
        ...fixture.evidence,
        configBytes: new Uint8Array(new SharedArrayBuffer(1)),
      })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID" });
    }
  });

  it("enforces byte caps, opaque authorization, closed options, and cancellation", async () => {
    const fixture = await createCppCuteAotOciFixture();
    for (const [key, limit] of [
      ["manifestBytes", CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.manifestBytes],
      ["configBytes", CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.configBytes],
    ] as const) {
      await expect(verifyCppCuteAotOciMetadata({
        ...fixture.evidence,
        [key]: new Uint8Array(limit + 1),
      })).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-AOT-OCI-RESOURCE-LIMIT",
        path: `$.${key}`,
      });
    }
    const raw = await verifyCppCuteAotOciMetadata(fixture.evidence);
    expect(() => authorizeCppCuteAotOciMetadata({ ...fixture.plan } as never, raw)).toThrow();
    await expect(verifyCppCuteAotOciMetadata(
      fixture.evidence,
      { unknown: true } as never,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-OCI-INVALID" });
    const controller = new AbortController();
    controller.abort();
    let evidenceTouches = 0;
    const untouchedEvidence = new Proxy(fixture.evidence, {
      getPrototypeOf: () => { evidenceTouches += 1; return Object.prototype; },
      ownKeys: () => { evidenceTouches += 1; return ["manifestBytes", "configBytes"]; },
    });
    await expect(verifyCppCuteAotOciMetadata(
      untouchedEvidence,
      { signal: controller.signal },
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-OCI-CANCELLED" });
    expect(evidenceTouches).toBe(0);
    expect(CppCuteAotOciError).toBeDefined();
  });
});
