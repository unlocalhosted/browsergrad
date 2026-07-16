import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import type { CppCuteAotOciMetadataBytes } from "../../../src/cpp_cute_aot_oci.js";
import type { PreparedCppCuteAotOfflineRun } from "../../../src/cpp_cute_aot_runner_plan.js";
import {
  createCppCuteAotRunnerFixture,
  type CppCuteAotRunnerFixture,
} from "./cpp_cute_aot_runner_fixtures.js";

const textEncoder = new TextEncoder();
export const OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json";
export const OCI_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json";
export const OCI_LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip";
export const DEFAULT_LAYER_DIGEST = `sha256:${"a".repeat(64)}`;
export const DEFAULT_DIFF_ID = `sha256:${"b".repeat(64)}`;

type MutableJsonObject = Record<string, unknown>;

export interface CppCuteAotOciFixtureOptions {
  readonly manifestConfigDigest?: string;
  readonly manifestConfigSizeDelta?: number;
  readonly rawManifestBytes?: Uint8Array;
  readonly rawConfigBytes?: Uint8Array;
  readonly mutateManifest?: (manifest: MutableJsonObject) => void;
  readonly mutateConfig?: (config: MutableJsonObject) => void;
  readonly layers?: readonly MutableJsonObject[];
  readonly diffIds?: readonly string[];
  readonly history?: readonly MutableJsonObject[] | null;
  readonly imageConfig?: unknown;
  readonly omitImageConfig?: boolean;
  readonly environmentMatchesOciLayers?: boolean;
}

export interface CppCuteAotOciFixture {
  readonly plan: PreparedCppCuteAotOfflineRun;
  readonly runner: CppCuteAotRunnerFixture;
  readonly evidence: CppCuteAotOciMetadataBytes;
  readonly manifestDigest: string;
  readonly configDigest: string;
}

export async function createCppCuteAotOciFixture(
  options: CppCuteAotOciFixtureOptions = {},
): Promise<CppCuteAotOciFixture> {
  const layers = options.layers?.map((layer) => structuredClone(layer)) ?? [defaultLayer()];
  const diffIds = options.diffIds ?? [DEFAULT_DIFF_ID];
  const config: MutableJsonObject = {
    created: "2026-07-16T00:00:00Z",
    architecture: "amd64",
    os: "linux",
    rootfs: {
      type: "layers",
      diff_ids: [...diffIds],
    },
    history: options.history === null
      ? undefined
      : options.history?.map((entry) => structuredClone(entry))
        ?? diffIds.map(() => ({ created: "2026-07-16T00:00:00Z", created_by: "fixture" })),
  };
  if (!options.omitImageConfig) {
    config.config = options.imageConfig === undefined ? {} : options.imageConfig;
  }
  options.mutateConfig?.(config);
  if (config.history === undefined) delete config.history;
  const configBytes = options.rawConfigBytes === undefined
    ? textEncoder.encode(JSON.stringify(config))
    : new Uint8Array(options.rawConfigBytes);
  const configDigest = `sha256:${await sha256Hex(configBytes)}`;
  const manifest: MutableJsonObject = {
    schemaVersion: 2,
    mediaType: OCI_MANIFEST_MEDIA_TYPE,
    config: {
      mediaType: OCI_CONFIG_MEDIA_TYPE,
      digest: options.manifestConfigDigest ?? configDigest,
      size: configBytes.byteLength + (options.manifestConfigSizeDelta ?? 0),
    },
    layers,
    annotations: {
      "org.opencontainers.image.source": "https://github.com/unlocalhosted/browsergrad",
    },
  };
  options.mutateManifest?.(manifest);
  const manifestBytes = options.rawManifestBytes === undefined
    ? textEncoder.encode(JSON.stringify(manifest))
    : new Uint8Array(options.rawManifestBytes);
  const manifestDigest = `sha256:${await sha256Hex(manifestBytes)}`;
  const runner = await createCppCuteAotRunnerFixture(
    {
      containerManifestDigest: manifestDigest,
      containerConfigDigest: configDigest,
    },
    options.environmentMatchesOciLayers === true
      ? {
          environmentLayers: layers.map((layer, index) => ({
            mediaType: layer.mediaType as "application/vnd.oci.image.layer.v1.tar+gzip",
            digest: layer.digest as string,
            size: String(layer.size) as never,
            diffId: diffIds[index] as string,
          })),
        }
      : {},
  );
  return {
    plan: runner.plan,
    runner,
    evidence: Object.freeze({ manifestBytes, configBytes }),
    manifestDigest,
    configDigest,
  };
}

export function defaultLayer(overrides: MutableJsonObject = {}): MutableJsonObject {
  return {
    mediaType: OCI_LAYER_MEDIA_TYPE,
    digest: DEFAULT_LAYER_DIGEST,
    size: 123,
    ...overrides,
  };
}
