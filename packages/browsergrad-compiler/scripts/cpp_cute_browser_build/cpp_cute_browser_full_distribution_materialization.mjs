import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  open,
  realpath,
  unlink,
} from "node:fs/promises";
import {
  dirname,
  isAbsolute,
  join,
  normalize,
} from "node:path/posix";
import { pathToFileURL } from "node:url";

import {
  canonicalJsonBytes,
  decodeWireJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  verifyCppCuteBrowserBuildSignatureBinding,
  unwrapVerifiedCppCuteBrowserBuildSignatureBinding,
} from "../../dist/cpp_cute_browser_build_provenance.js";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
} from "../../dist/cpp_cute_browser_build_provenance_syntax.js";
import {
  copyPreparedCppCuteBrowserDistributionAssetManifestBytes,
  copyPreparedCppCuteBrowserDistributionProfileBytes,
  prepareCppCuteBrowserDistributionMetadata,
  unwrapPreparedCppCuteBrowserDistributionMetadata,
} from "../../dist/cpp_cute_browser_distribution_metadata.js";
import {
  cppCuteBrowserHeaderDistributionReproducibilityResourceBytes,
  requireVerifiedCppCuteBrowserHeaderDistributionReproducibility,
  verifyCppCuteBrowserHeaderDistributionReproducibilityResource,
} from "../../dist/cpp_cute_browser_header_distribution_reproducibility.js";
import {
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
  admitCppCuteBrowserProducerTrustPolicy,
} from "../../dist/cpp_cute_browser_producer_trust_policy.js";
import {
  unwrapVerifiedCppCuteBrowserBuildProducer,
  verifyCppCuteBrowserBuildProducer,
} from "../../dist/cpp_cute_browser_producer_trust.js";
import {
  cppCuteBrowserReproducibilityResourceBytes,
  requireVerifiedCppCuteBrowserReproducibility,
  verifyCppCuteBrowserReproducibilityResource,
} from "../../dist/cpp_cute_browser_reproducibility.js";
import {
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "../../dist/cpp_cute_browser_runtime_abi.js";
import {
  cppCuteDiagnosticNormalizationResourceBytes,
} from "../../dist/cpp_cute_diagnostic_normalization.js";
import {
  prepareCppCuteAttestationTrustStore,
} from "../../dist/cpp_cute_frontend_provenance.js";
import {
  cppCuteSemanticAdapterManifestResourceBytes,
} from "../../dist/cpp_cute_semantic_adapter_manifest.js";
import {
  inspectCppCuteBrowserVfsPack,
} from "../../dist/cpp_cute_browser_vfs_pack.js";
import {
  copyVerifiedCppCuteBrowserWorkerBundleBytes,
  inspectVerifiedCppCuteBrowserWorkerBundle,
  verifyCppCuteBrowserWorkerBundle,
} from "../../dist/cpp_cute_browser_worker_bundle.js";
import {
  materializeCppCuteBrowserDistributionOutputFiles,
  verifyCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";

export const
  CPP_CUTE_BROWSER_DETERMINISTIC_DISTRIBUTION_MATERIALIZATION_SCHEMA =
    "browsergrad.compiler.cpp-cute.browser-deterministic-distribution-materialization";
export const CPP_CUTE_BROWSER_FULL_DISTRIBUTION_MATERIALIZATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-full-distribution-materialization";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-MATERIALIZATION";
const DETERMINISTIC_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-deterministic-distribution-materialization.v1";
const FULL_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-full-distribution-materialization.v1";
const DETACHED_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/build-provenance.dsse.json";
const ASSET_MANIFEST_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/asset-manifest.json";
const BUILD_LOCK_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/build-input-lock.json";
const WASM_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/clang-extractor.wasm";
const WORKER_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/cpp-cute-browser-worker.mjs";
const DIAGNOSTIC_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/diagnostic-normalization.json";
const RUNTIME_ABI_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/runtime-abi-manifest.json";
const SEMANTIC_ADAPTER_OUTPUT_PATH =
  "assets/browsergrad-cpp-cute/semantic-adapter-manifest.json";
const PACK_BINDINGS = Object.freeze([
  Object.freeze({
    includeRootId: "clang-resource",
    outputPath:
      "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cuda",
    outputPath:
      "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cutlass",
    outputPath:
      "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "cxx-stdlib",
    outputPath:
      "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
  }),
  Object.freeze({
    includeRootId: "linux-sysroot",
    outputPath:
      "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
  }),
]);
const DETERMINISTIC_AUTHORITIES = new WeakMap();
const FULL_AUTHORITIES = new WeakMap();
const PROFILE_BYTE_LIMIT = 256 * 1024;
const TRUST_STORE_BYTE_LIMIT = 256 * 1024;
const MAX_ARGUMENT_COUNT = 8;
const MAX_ARGUMENT_BYTE_LENGTH = 16 * 1024;
const TRUST_STORE_DECODE_LIMITS = Object.freeze({
  maxDocumentBytes: TRUST_STORE_BYTE_LIMIT,
  maxDepth: 8,
  maxNodes: 2_048,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 256,
  maxObjectProperties: 16,
  maxRank: 1,
  maxIntegerBits: 32,
  maxArithmeticOperations: 4_096,
});

export class CppCuteBrowserFullDistributionMaterializationError
  extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserFullDistributionMaterializationError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Extends one exact private 17-file header distribution with the seven
 * deterministic current package outputs. The detached producer envelope is
 * deliberately excluded until a separately admitted producer signs it.
 */
export async function materializeCppCuteBrowserDeterministicDistribution(
  input,
) {
  const object = exactObject(
    input,
    ["outputRoot", "producerTrustPolicy", "wasmBytes"],
    "$.input",
  );
  const outputRoot = absolutePath(object.outputRoot, "$.input.outputRoot");
  const wasmBytes = snapshotBytes(
    object.wasmBytes,
    32 * 1024 * 1024,
    "$.input.wasmBytes",
  );
  const current = await prepareCurrentAuthorities(
    object.producerTrustPolicy,
  );
  assertWasmBytes(wasmBytes, current.wasmReproducibility);
  await verifyHeaderOnlyRoot(outputRoot, current.headerDistribution);
  const packs = await inspectDistributionPacks(
    outputRoot,
    current.headerDistribution,
  );
  const metadata = await prepareCppCuteBrowserDistributionMetadata({
    buildInputLock: current.buildInputLock,
    packs,
    producerTrustPolicy: current.producerTrustPolicy,
    wasmReproducibility: current.wasmReproducibility,
    workerBundle: current.workerBundle,
  });
  const outputs = deterministicNewOutputs(
    metadata,
    current,
    wasmBytes,
  );
  let materialization;
  try {
    materialization =
      await materializeCppCuteBrowserDistributionOutputFiles({
        outputRoot,
        existingOutputs: current.headerDistribution.outputs,
        outputs,
      });
  } catch (cause) {
    invalid(
      "$.input.outputRoot",
      "failed to add the exact deterministic distribution outputs",
      { cause },
    );
  }
  const expectedOutputs = combineOutputs(
    current.headerDistribution.outputs,
    materialization.outputs,
  );
  assertExactPlan(
    current.buildInputLock,
    expectedOutputs,
    "deterministic-subject",
  );
  const verification = await verifyExactRoot(outputRoot, expectedOutputs);
  return issueDeterministicAuthority({
    current,
    expectedOutputs,
    materializedOutputs: materialization.outputs,
    metadata,
    outputRoot,
    verification,
  });
}

/**
 * Reauthenticates an existing exact 24-file deterministic distribution and
 * its separate canonical profile handoff. No filesystem writes occur.
 */
export async function admitCppCuteBrowserDeterministicDistribution(input) {
  const object = exactObject(
    input,
    ["outputRoot", "producerTrustPolicy", "profileBytes"],
    "$.input",
  );
  const outputRoot = absolutePath(object.outputRoot, "$.input.outputRoot");
  const profileBytes = snapshotBytes(
    object.profileBytes,
    PROFILE_BYTE_LIMIT,
    "$.input.profileBytes",
  );
  const current = await prepareCurrentAuthorities(
    object.producerTrustPolicy,
  );
  const wasmOutput = expectedOutput(
    current.buildInputLock,
    WASM_OUTPUT_PATH,
  );
  const wasmBytes = await readExactImmutableOutput(
    outputRoot,
    {
      outputPath: WASM_OUTPUT_PATH,
      sha256: current.wasmReproducibility.wasmSha256,
      byteLength: String(current.wasmReproducibility.wasmByteLength),
    },
    current.wasmReproducibility.wasmByteLength,
  );
  if (wasmOutput.reproducibilityClass !== "deterministic-subject") {
    mismatch(
      "$.buildInputLock",
      "Wasm output is no longer one deterministic subject",
    );
  }
  assertWasmBytes(wasmBytes, current.wasmReproducibility);
  const packs = await inspectDistributionPacks(
    outputRoot,
    current.headerDistribution,
  );
  const metadata = await prepareCppCuteBrowserDistributionMetadata({
    buildInputLock: current.buildInputLock,
    packs,
    producerTrustPolicy: current.producerTrustPolicy,
    wasmReproducibility: current.wasmReproducibility,
    workerBundle: current.workerBundle,
  });
  if (!sameBytes(
    profileBytes,
    copyPreparedCppCuteBrowserDistributionProfileBytes(metadata),
  )) {
    mismatch(
      "$.input.profileBytes",
      "profile handoff differs from the exact reconstructed distribution metadata",
    );
  }
  const deterministicOutputs = deterministicNewOutputs(
    metadata,
    current,
    wasmBytes,
  ).map(outputIdentity);
  const expectedOutputs = combineOutputs(
    current.headerDistribution.outputs,
    deterministicOutputs,
  );
  assertExactPlan(
    current.buildInputLock,
    expectedOutputs,
    "deterministic-subject",
  );
  const verification = await verifyExactRoot(outputRoot, expectedOutputs);
  return issueDeterministicAuthority({
    current,
    expectedOutputs,
    materializedOutputs: Object.freeze([]),
    metadata,
    outputRoot,
    verification,
  });
}

/**
 * Adds the sole detached envelope only after an in-process independently
 * admitted producer authority proves that it binds this exact distribution.
 */
export async function finalizeCppCuteBrowserFullDistribution(input) {
  const object = exactObject(
    input,
    ["deterministicDistribution", "producer"],
    "$.input",
  );
  const deterministic = requireDeterministicAuthority(
    object.deterministicDistribution,
  );
  let producerRecord;
  try {
    producerRecord = unwrapVerifiedCppCuteBrowserBuildProducer(
      object.producer,
    );
  } catch (cause) {
    unverified(
      "$.input.producer",
      "expected one independently admitted producer authority",
      { cause },
    );
  }
  const signatureRecord =
    producerRecord.signatureBinding === undefined
      ? undefined
      : unwrapVerifiedCppCuteBrowserBuildSignatureBinding(
        producerRecord.signatureBinding,
      );
  const metadataRecord =
    unwrapPreparedCppCuteBrowserDistributionMetadata(
      deterministic.metadata,
    );
  if (signatureRecord === undefined ||
      signatureRecord.assetManifest !== metadataRecord.assetManifest ||
      signatureRecord.buildInputLock !== metadataRecord.buildInputLock ||
      signatureRecord.workerBundle !== metadataRecord.workerBundle ||
      producerRecord.trustPolicy !== metadataRecord.producerTrustPolicy ||
      object.producer.producerTrusted !== true ||
      object.producer.buildSubjectId !==
        deterministic.metadata.buildSubjectId ||
      object.producer.manifestId !==
        deterministic.metadata.assetManifestId ||
      object.producer.profileHash !== deterministic.metadata.profileHash ||
      object.producer.buildInputLockResourceSha256 !==
        deterministic.metadata.buildInputLockResourceSha256 ||
      object.producer.workerBundleSha256 !==
        deterministic.metadata.workerBundleSha256) {
    mismatch(
      "$.input.producer",
      "producer authority does not bind the exact deterministic distribution",
    );
  }
  const envelopeBytes = canonicalJsonBytes(signatureRecord.envelope);
  let materialization;
  try {
    materialization =
      await materializeCppCuteBrowserDistributionOutputFiles({
        outputRoot: deterministic.outputRoot,
        existingOutputs: deterministic.expectedOutputs,
        outputs: [{
          outputPath: DETACHED_OUTPUT_PATH,
          bytes: envelopeBytes,
        }],
      });
  } catch (cause) {
    invalid(
      "$.input.deterministicDistribution",
      "failed to add the exact detached producer envelope",
      { cause },
    );
  }
  const expectedOutputs = combineOutputs(
    deterministic.expectedOutputs,
    materialization.outputs,
  );
  assertExactPlan(
    deterministic.current.buildInputLock,
    expectedOutputs,
    "complete",
  );
  const verification = await verifyExactRoot(
    deterministic.outputRoot,
    expectedOutputs,
  );
  const fullHash = sha256(canonicalJsonBytes({
    domain: FULL_HASH_DOMAIN,
    deterministicMaterializationId:
      object.deterministicDistribution.materializationId,
    producerEvidenceId: object.producer.producerEvidenceId,
    outputVerificationId: verification.verificationId,
    outputs: expectedOutputs,
  }));
  const full = Object.freeze({
    schema: CPP_CUTE_BROWSER_FULL_DISTRIBUTION_MATERIALIZATION_SCHEMA,
    version: 1,
    materializationId:
      `bg.cpp.browser-full-distribution-materialization.sha256.${fullHash}`,
    authority:
      "producer-authenticated-exact-private-distribution-materialization-only",
    outputRoot: deterministic.outputRoot,
    deterministicMaterializationId:
      object.deterministicDistribution.materializationId,
    metadataId: deterministic.metadata.metadataId,
    producerEvidenceId: object.producer.producerEvidenceId,
    buildSubjectId: object.producer.buildSubjectId,
    buildSubjectSha256: object.producer.buildSubjectSha256,
    outputVerificationId: verification.verificationId,
    outputs: Object.freeze(expectedOutputs),
    totals: verification.totals,
    claims: Object.freeze({
      exactCurrentHeaderDistributionVerified: true,
      exactPackageWasmVerified: true,
      exactPackagePolicyAssetsVerified: true,
      exactBuildInputLockOutputPlanMatched: true,
      deterministicSubjectSetVerified: true,
      detachedEnvelopeMaterializedWithoutClobber: true,
      signatureVerified: true,
      producerTrusted: true,
      fullDistributedOutputSetMaterialized: true,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    }),
  });
  FULL_AUTHORITIES.set(full, Object.freeze({
    deterministicDistribution: object.deterministicDistribution,
    producer: object.producer,
    expectedOutputs: Object.freeze(expectedOutputs),
  }));
  return full;
}

export function requireCppCuteBrowserDeterministicDistributionAuthority(
  authority,
) {
  requireDeterministicAuthority(authority);
}

export function requireCppCuteBrowserFullDistributionMaterializationAuthority(
  authority,
) {
  if (typeof authority !== "object" || authority === null ||
      !FULL_AUTHORITIES.has(authority)) {
    unverified(
      "$.authority",
      "expected materializer-issued full-distribution authority",
    );
  }
}

export function copyCppCuteBrowserDeterministicDistributionProfileBytes(
  authority,
) {
  return copyPreparedCppCuteBrowserDistributionProfileBytes(
    requireDeterministicAuthority(authority).metadata,
  );
}

export async function runCppCuteBrowserFullDistributionMaterialization(
  argv,
) {
  const args = parseCppCuteBrowserFullDistributionMaterializationArguments(
    argv,
  );
  const producerPolicyBytes = await readImmutableInput(
    args["producer-policy"],
    CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
    "$.arguments.producer-policy",
  );
  const producerTrustPolicy =
    await admitCppCuteBrowserProducerTrustPolicy(producerPolicyBytes);
  if (args.operation === "materialize-deterministic") {
    const wasmBytes = await readImmutableInput(
      args.wasm,
      32 * 1024 * 1024,
      "$.arguments.wasm",
    );
    const deterministic =
      await materializeCppCuteBrowserDeterministicDistribution({
        outputRoot: args["output-root"],
        producerTrustPolicy,
        wasmBytes,
      });
    const profileBytes =
      copyCppCuteBrowserDeterministicDistributionProfileBytes(
        deterministic,
      );
    try {
      await writeExclusiveImmutableOutput(
        args["profile-output"],
        profileBytes,
        "$.arguments.profile-output",
      );
    } catch (cause) {
      await cleanupDeterministicFiles(deterministic);
      if (cause instanceof
          CppCuteBrowserFullDistributionMaterializationError) {
        throw cause;
      }
      invalid(
        "$.arguments.profile-output",
        "profile handoff failed after deterministic output materialization",
        { cause },
      );
    }
    return Object.freeze({
      operation: args.operation,
      profileOutputPath: args["profile-output"],
      profileSha256: deterministic.profileSha256,
      profileByteLength: String(profileBytes.byteLength),
      report: deterministic,
    });
  }

  const [profileBytes, trustStoreBytes, envelopeBytes] = await Promise.all([
    readImmutableInput(
      args.profile,
      PROFILE_BYTE_LIMIT,
      "$.arguments.profile",
    ),
    readImmutableInput(
      args["trust-store"],
      TRUST_STORE_BYTE_LIMIT,
      "$.arguments.trust-store",
    ),
    readImmutableInput(
      args.envelope,
      CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
      "$.arguments.envelope",
    ),
  ]);
  const deterministic = await admitCppCuteBrowserDeterministicDistribution({
    outputRoot: args["output-root"],
    producerTrustPolicy,
    profileBytes,
  });
  const deterministicRecord = requireDeterministicAuthority(deterministic);
  const metadata =
    unwrapPreparedCppCuteBrowserDistributionMetadata(
      deterministicRecord.metadata,
    );
  const trustStoreValue = decodeCanonicalJson(
    trustStoreBytes,
    TRUST_STORE_DECODE_LIMITS,
    "$.arguments.trust-store",
  );
  const trustStore = await prepareCppCuteAttestationTrustStore(
    trustStoreValue,
    { limits: TRUST_STORE_DECODE_LIMITS },
  );
  const envelope = decodeCanonicalJson(
    envelopeBytes,
    CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS,
    "$.arguments.envelope",
  );
  const signatureBinding = await verifyCppCuteBrowserBuildSignatureBinding(
    envelope,
    {
      assetManifest: metadata.assetManifest,
      buildInputLock: metadata.buildInputLock,
      workerBundle: metadata.workerBundle,
      trustStore,
    },
  );
  const producer = await verifyCppCuteBrowserBuildProducer(
    signatureBinding,
    metadata.producerTrustPolicy,
  );
  const full = await finalizeCppCuteBrowserFullDistribution({
    deterministicDistribution: deterministic,
    producer,
  });
  return Object.freeze({ operation: args.operation, report: full });
}

export function parseCppCuteBrowserFullDistributionMaterializationArguments(
  argv,
) {
  if (!Array.isArray(argv) || argv.length === 0 ||
      argv.length > MAX_ARGUMENT_COUNT) {
    invalid("$arguments", "expected one bounded nonempty argument array");
  }
  const values = {};
  for (const [index, argument] of argv.entries()) {
    if (typeof argument !== "string" || argument.length === 0 ||
        Buffer.byteLength(argument) > MAX_ARGUMENT_BYTE_LENGTH) {
      invalid(`$arguments[${index}]`, "argument is not one bounded string");
    }
    const match = /^--([a-z-]+)=(.+)$/u.exec(argument);
    if (match === null || match[1] === undefined ||
        match[2] === undefined) {
      invalid(`$arguments[${index}]`, "expected --name=value");
    }
    if (Object.hasOwn(values, match[1])) {
      invalid(`$arguments[${index}]`, `duplicate --${match[1]}`);
    }
    values[match[1]] = match[2];
  }
  const operation = values.operation;
  const keys = operation === "materialize-deterministic"
    ? [
        "operation",
        "output-root",
        "producer-policy",
        "profile-output",
        "wasm",
      ]
    : operation === "finalize"
      ? [
          "envelope",
          "operation",
          "output-root",
          "producer-policy",
          "profile",
          "trust-store",
        ]
      : undefined;
  if (keys === undefined) {
    invalid(
      "$arguments.operation",
      "operation must be materialize-deterministic or finalize",
    );
  }
  const actual = Object.keys(values).sort(compareText);
  if (actual.length !== keys.length ||
      actual.some((key, index) => key !== keys[index])) {
    invalid(
      "$arguments",
      `operation ${operation} requires exactly ${keys.join(", ")}`,
    );
  }
  for (const key of keys.filter((key) =>
    key !== "operation" && key !== "output-root")) {
    values[key] = absolutePath(values[key], `$arguments.${key}`);
  }
  values["output-root"] = absolutePath(
    values["output-root"],
    "$arguments.output-root",
  );
  return Object.freeze(values);
}

async function prepareCurrentAuthorities(producerTrustPolicy) {
  let buildInputLock;
  let headerDistribution;
  let wasmReproducibility;
  let workerBundle;
  try {
    [
      buildInputLock,
      headerDistribution,
      wasmReproducibility,
      workerBundle,
    ] = await Promise.all([
      decodeCppCuteBrowserBuildInputLock(
        cppCuteBrowserBuildInputLockResourceBytes(),
      ),
      verifyCppCuteBrowserHeaderDistributionReproducibilityResource(
        cppCuteBrowserHeaderDistributionReproducibilityResourceBytes(),
      ),
      verifyCppCuteBrowserReproducibilityResource(
        cppCuteBrowserReproducibilityResourceBytes(),
      ),
      verifyCppCuteBrowserWorkerBundle(),
    ]);
    requireVerifiedCppCuteBrowserHeaderDistributionReproducibility(
      headerDistribution,
    );
    requireVerifiedCppCuteBrowserReproducibility(wasmReproducibility);
  } catch (cause) {
    unverified(
      "$.package",
      "failed to reauthenticate current package distribution authorities",
      { cause },
    );
  }
  if (typeof producerTrustPolicy !== "object" ||
      producerTrustPolicy === null ||
      producerTrustPolicy.hostOnly !== true ||
      producerTrustPolicy.workerTransferable !== false ||
      producerTrustPolicy.predicateType !==
        "https://browsergrad.dev/provenance/cpp-cute-browser-build/v1") {
    unverified(
      "$.input.producerTrustPolicy",
      "expected one admitted host-only producer trust policy",
    );
  }
  const lock = unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock);
  if (headerDistribution.currentBuildInputLockId !== buildInputLock.lockId ||
      headerDistribution.currentBuildInputLockResourceSha256 !==
        buildInputLock.resourceSha256 ||
      wasmReproducibility.sourceSetSha256 !==
        buildInputLock.extractorSourceSetSha256) {
    mismatch(
      "$.package",
      "package evidence does not bind the current build-input lock",
    );
  }
  return Object.freeze({
    buildInputLock,
    buildInputLockRecord: lock,
    headerDistribution,
    producerTrustPolicy,
    wasmReproducibility,
    workerBundle,
  });
}

async function verifyHeaderOnlyRoot(outputRoot, headerDistribution) {
  try {
    await verifyCppCuteBrowserDistributionOutputFiles({
      outputRoot,
      expectedOutputs: headerDistribution.outputs,
    });
  } catch (cause) {
    invalid(
      "$.input.outputRoot",
      "output root is not the exact current 17-file header distribution",
      { cause },
    );
  }
}

async function inspectDistributionPacks(outputRoot, headerDistribution) {
  return Promise.all(PACK_BINDINGS.map(async (binding_) => {
    const expected = headerDistribution.outputs.find(
      (output) => output.outputPath === binding_.outputPath,
    );
    if (expected === undefined) {
      mismatch(
        "$.package.headerDistribution.outputs",
        `header evidence is missing ${binding_.outputPath}`,
      );
    }
    const bytes = await readExactImmutableOutput(
      outputRoot,
      expected,
      Number(expected.byteLength),
    );
    let pack;
    try {
      pack = await inspectCppCuteBrowserVfsPack(bytes);
    } catch (cause) {
      invalid(
        `$.outputs.${binding_.outputPath}`,
        "exact header output is not one canonical VFS pack",
        { cause },
      );
    }
    if (pack.packSha256 !== expected.sha256 ||
        pack.packByteLength !== expected.byteLength) {
      mismatch(
        `$.outputs.${binding_.outputPath}`,
        "VFS inspection differs from package-pinned output identity",
      );
    }
    return Object.freeze({
      includeRootId: binding_.includeRootId,
      outputPath: binding_.outputPath,
      pack,
    });
  }));
}

function deterministicNewOutputs(metadata, current, wasmBytes) {
  const workerBytes =
    copyVerifiedCppCuteBrowserWorkerBundleBytes(current.workerBundle);
  const worker =
    inspectVerifiedCppCuteBrowserWorkerBundle(current.workerBundle);
  if (sha256(workerBytes) !== worker.sha256 ||
      workerBytes.byteLength !== worker.byteLength) {
    mismatch(
      "$.package.workerBundle",
      "copied Worker bytes differ from their verified package identity",
    );
  }
  return Object.freeze([
    Object.freeze({
      outputPath: ASSET_MANIFEST_OUTPUT_PATH,
      bytes:
        copyPreparedCppCuteBrowserDistributionAssetManifestBytes(metadata),
    }),
    Object.freeze({
      outputPath: BUILD_LOCK_OUTPUT_PATH,
      bytes: cppCuteBrowserBuildInputLockResourceBytes(),
    }),
    Object.freeze({ outputPath: WASM_OUTPUT_PATH, bytes: wasmBytes }),
    Object.freeze({ outputPath: WORKER_OUTPUT_PATH, bytes: workerBytes }),
    Object.freeze({
      outputPath: DIAGNOSTIC_OUTPUT_PATH,
      bytes: cppCuteDiagnosticNormalizationResourceBytes(),
    }),
    Object.freeze({
      outputPath: RUNTIME_ABI_OUTPUT_PATH,
      bytes: cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    }),
    Object.freeze({
      outputPath: SEMANTIC_ADAPTER_OUTPUT_PATH,
      bytes: cppCuteSemanticAdapterManifestResourceBytes(),
    }),
  ]);
}

function issueDeterministicAuthority(input) {
  const deterministicHash = sha256(canonicalJsonBytes({
    domain: DETERMINISTIC_HASH_DOMAIN,
    metadataId: input.metadata.metadataId,
    outputVerificationId: input.verification.verificationId,
    outputs: input.expectedOutputs,
  }));
  const authority = Object.freeze({
    schema:
      CPP_CUTE_BROWSER_DETERMINISTIC_DISTRIBUTION_MATERIALIZATION_SCHEMA,
    version: 1,
    materializationId:
      `bg.cpp.browser-deterministic-distribution-materialization.sha256.${deterministicHash}`,
    authority:
      "exact-current-private-deterministic-distribution-materialization-only",
    outputRoot: input.outputRoot,
    metadataId: input.metadata.metadataId,
    profileId: input.metadata.profileId,
    profileHash: input.metadata.profileHash,
    compilationContractHash: input.metadata.compilationContractHash,
    profileSha256: input.metadata.profileSha256,
    profileByteLength: String(input.metadata.profileByteLength),
    assetManifestId: input.metadata.assetManifestId,
    assetManifestSha256: input.metadata.assetManifestSha256,
    assetManifestByteLength: input.metadata.assetManifestByteLength,
    assetSetSha256: input.metadata.assetSetSha256,
    buildSubjectId: input.metadata.buildSubjectId,
    buildSubjectSha256: input.metadata.buildSubjectSha256,
    buildInputLockId: input.metadata.buildInputLockId,
    buildInputLockResourceSha256:
      input.metadata.buildInputLockResourceSha256,
    workerBundleSha256: input.metadata.workerBundleSha256,
    wasmSha256: input.metadata.wasmSha256,
    wasmByteLength: String(input.metadata.wasmByteLength),
    headerDistributionReproducibilityId:
      input.current.headerDistribution.reproducibilityId,
    headerDistributionOutputVerificationId:
      input.current.headerDistribution.outputVerificationId,
    outputVerificationId: input.verification.verificationId,
    outputs: Object.freeze(input.expectedOutputs),
    totals: input.verification.totals,
    claims: Object.freeze({
      exactCurrentHeaderDistributionVerified: true,
      exactPackageWasmVerified: true,
      exactPackageWorkerVerified: true,
      exactPackagePolicyAssetsVerified: true,
      exactCurrentBuildInputLockVerified: true,
      exactBuildInputLockDeterministicOutputPlanMatched: true,
      exactDeterministicOutputTreeVerified: true,
      newFilesWrittenWithoutClobber:
        input.materializedOutputs.length === 7,
      detachedEnvelopePresent: false,
      signatureVerified: false,
      producerTrusted: false,
      fullDistributedOutputSetMaterialized: false,
      fullDistributedOutputSetReproducible: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      backendExecutionObserved: false,
      releaseReady: false,
    }),
  });
  DETERMINISTIC_AUTHORITIES.set(authority, Object.freeze({
    current: input.current,
    expectedOutputs: Object.freeze(input.expectedOutputs),
    metadata: input.metadata,
    outputRoot: input.outputRoot,
  }));
  return authority;
}

function requireDeterministicAuthority(authority) {
  if (typeof authority !== "object" || authority === null) {
    unverified(
      "$.authority",
      "expected materializer-issued deterministic distribution authority",
    );
  }
  const record = DETERMINISTIC_AUTHORITIES.get(authority);
  if (record === undefined) {
    unverified(
      "$.authority",
      "expected materializer-issued deterministic distribution authority",
    );
  }
  return record;
}

function assertWasmBytes(bytes, reproducibility) {
  if (bytes.byteLength !== reproducibility.wasmByteLength ||
      sha256(bytes) !== reproducibility.wasmSha256) {
    mismatch(
      "$.wasm",
      "Wasm bytes differ from the package-pinned two-build output",
    );
  }
}

function expectedOutput(buildInputLock, outputPath) {
  const output =
    unwrapPreparedCppCuteBrowserBuildInputLock(
      buildInputLock,
    ).lock.body.recipe.distributedOutputPlan.outputs.find(
      (candidate) => candidate.path === outputPath,
    );
  if (output === undefined) {
    mismatch("$.buildInputLock", `missing output ${outputPath}`);
  }
  return output;
}

function assertExactPlan(buildInputLock, outputs, scope) {
  const plan =
    unwrapPreparedCppCuteBrowserBuildInputLock(
      buildInputLock,
    ).lock.body.recipe.distributedOutputPlan.outputs;
  const expected = scope === "complete"
    ? plan
    : plan.filter(
      (output) => output.reproducibilityClass === scope,
    );
  if (outputs.length !== expected.length) {
    mismatch("$.outputs", "output count differs from the current build lock");
  }
  const paths = outputs.map((output) => output.outputPath).sort(compareText);
  const expectedPaths = expected.map((output) => output.path).sort(compareText);
  if (paths.some((path, index) => path !== expectedPaths[index])) {
    mismatch("$.outputs", "output paths differ from the current build lock");
  }
}

async function verifyExactRoot(outputRoot, expectedOutputs) {
  try {
    return await verifyCppCuteBrowserDistributionOutputFiles({
      outputRoot,
      expectedOutputs,
    });
  } catch (cause) {
    invalid(
      "$.outputRoot",
      "exact distribution output verification failed",
      { cause },
    );
  }
}

function combineOutputs(left, right) {
  const outputs = [...left, ...right].map((output) => Object.freeze({
    outputPath: output.outputPath,
    sha256: output.sha256,
    byteLength: output.byteLength,
  })).sort((first, second) =>
    compareText(first.outputPath, second.outputPath));
  const paths = new Set(outputs.map((output) => output.outputPath));
  if (paths.size !== outputs.length) {
    invalid("$.outputs", "combined output paths must be unique");
  }
  return outputs;
}

function outputIdentity(output) {
  return Object.freeze({
    outputPath: output.outputPath,
    sha256: sha256(output.bytes),
    byteLength: String(output.bytes.byteLength),
  });
}

async function readExactImmutableOutput(
  outputRoot,
  expected,
  maximumByteLength,
) {
  return readExactImmutableFile(
    join(outputRoot, expected.outputPath),
    expected,
    maximumByteLength,
    `$.outputs.${expected.outputPath}`,
  );
}

async function readExactImmutableFile(
  path,
  expected,
  maximumByteLength,
  diagnosticPath,
) {
  const byteLength = Number(expected.byteLength);
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0 ||
      byteLength > maximumByteLength) {
    resource(diagnosticPath, "file exceeds its fixed byte bound");
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!before.isFile() || before.isSymbolicLink() ||
        before.nlink !== 1n || before.size !== BigInt(byteLength) ||
        Number(before.mode & 0o222n) !== 0) {
      invalid(diagnosticPath, "expected one immutable single-link file");
    }
    const bytes = new Uint8Array(byteLength);
    let offset = 0;
    while (offset < byteLength) {
      const { bytesRead } = await handle.read(
        bytes,
        offset,
        byteLength - offset,
        offset,
      );
      if (bytesRead <= 0) {
        invalid(diagnosticPath, "file changed while read");
      }
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after) ||
        sha256(bytes) !== expected.sha256) {
      mismatch(diagnosticPath, "file identity changed while read");
    }
    return bytes;
  } catch (cause) {
    if (cause instanceof
        CppCuteBrowserFullDistributionMaterializationError) {
      throw cause;
    }
    invalid(diagnosticPath, "failed to read exact immutable file", {
      cause,
    });
  } finally {
    await handle?.close();
  }
}

async function readImmutableInput(path, maximumByteLength, diagnosticPath) {
  const canonical = absolutePath(path, diagnosticPath);
  let resolved;
  let stat;
  try {
    [resolved, stat] = await Promise.all([
      realpath(canonical),
      lstat(canonical, { bigint: true }),
    ]);
  } catch (cause) {
    invalid(diagnosticPath, "input file is unavailable", { cause });
  }
  if (resolved !== canonical || !stat.isFile() || stat.isSymbolicLink() ||
      stat.nlink !== 1n || stat.size <= 0n ||
      stat.size > BigInt(maximumByteLength) ||
      Number(stat.mode & 0o222n) !== 0) {
    invalid(
      diagnosticPath,
      "input must be one canonical immutable single-link file",
    );
  }
  return readExactImmutableFile(
    canonical,
    {
      outputPath: canonical,
      sha256: await hashImmutablePath(canonical, stat),
      byteLength: stat.size.toString(),
    },
    maximumByteLength,
    diagnosticPath,
  );
}

async function hashImmutablePath(path, expectedStat) {
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, expectedStat)) {
      invalid("$.input", "input identity changed before hashing");
    }
    const hash = createHash("sha256");
    const buffer = new Uint8Array(256 * 1024);
    let position = 0;
    while (position < Number(before.size)) {
      const length = Math.min(
        buffer.byteLength,
        Number(before.size) - position,
      );
      const { bytesRead } = await handle.read(
        buffer,
        0,
        length,
        position,
      );
      if (bytesRead <= 0) invalid("$.input", "input changed while hashing");
      hash.update(buffer.subarray(0, bytesRead));
      position += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) {
      invalid("$.input", "input identity changed while hashing");
    }
    return hash.digest("hex");
  } finally {
    await handle?.close();
  }
}

async function writeExclusiveImmutableOutput(path, bytes, diagnosticPath) {
  const outputPath = absolutePath(path, diagnosticPath);
  const parent = dirname(outputPath);
  let resolvedParent;
  let parentStat;
  try {
    [resolvedParent, parentStat] = await Promise.all([
      realpath(parent),
      lstat(parent, { bigint: true }),
    ]);
  } catch (cause) {
    invalid(diagnosticPath, "output parent is unavailable", { cause });
  }
  if (resolvedParent !== parent || !parentStat.isDirectory() ||
      parentStat.isSymbolicLink() ||
      Number(parentStat.mode & 0o077n) !== 0) {
    invalid(diagnosticPath, "output parent must be one canonical private directory");
  }
  let handle;
  try {
    handle = await open(
      outputPath,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL |
        constants.O_NOFOLLOW,
      0o400,
    );
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesWritten } = await handle.write(
        bytes,
        offset,
        bytes.byteLength - offset,
        offset,
      );
      if (bytesWritten <= 0) invalid(diagnosticPath, "output write stalled");
      offset += bytesWritten;
    }
    await handle.sync();
    await handle.chmod(0o400);
    const stat = await handle.stat({ bigint: true });
    if (!stat.isFile() || stat.nlink !== 1n ||
        stat.size !== BigInt(bytes.byteLength) ||
        Number(stat.mode & 0o222n) !== 0) {
      invalid(diagnosticPath, "persisted output is not immutable");
    }
    await handle.close();
    handle = undefined;
    await readExactImmutableFile(
      outputPath,
      {
        outputPath,
        sha256: sha256(bytes),
        byteLength: String(bytes.byteLength),
      },
      bytes.byteLength,
      diagnosticPath,
    );
    let parentHandle;
    try {
      parentHandle = await open(
        parent,
        constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW,
      );
      await parentHandle.sync();
    } finally {
      await parentHandle?.close();
    }
  } catch (cause) {
    if (handle !== undefined) {
      await handle.close().catch(() => undefined);
      handle = undefined;
      await unlink(outputPath).catch(() => undefined);
    }
    if (cause instanceof
        CppCuteBrowserFullDistributionMaterializationError) {
      throw cause;
    }
    invalid(diagnosticPath, "exclusive output creation failed", { cause });
  } finally {
    await handle?.close();
  }
}

async function cleanupDeterministicFiles(authority) {
  const record = requireDeterministicAuthority(authority);
  const headerPaths = new Set(
    record.current.headerDistribution.outputs.map(
      (output) => output.outputPath,
    ),
  );
  const created = record.expectedOutputs.filter(
    (output) => !headerPaths.has(output.outputPath),
  );
  for (const output of created) {
    const path = join(record.outputRoot, output.outputPath);
    try {
      await readExactImmutableFile(
        path,
        output,
        Number(output.byteLength),
        "$.cleanup",
      );
      await unlink(path);
    } catch (cause) {
      invalid(
        "$.cleanup",
        "failed to remove an exact file created before profile handoff failure",
        { cause },
      );
    }
  }
}

function decodeCanonicalJson(bytes, limits, diagnosticPath) {
  let value;
  try {
    value = decodeWireJson(bytes, { limits });
  } catch (cause) {
    invalid(diagnosticPath, "input is not bounded strict JSON", { cause });
  }
  let canonical;
  try {
    canonical = canonicalJsonBytes(value, { limits });
  } catch (cause) {
    invalid(diagnosticPath, "input cannot be canonically encoded", {
      cause,
    });
  }
  if (!sameBytes(bytes, canonical)) {
    mismatch(diagnosticPath, "input must use exact canonical JSON bytes");
  }
  return value;
}

function snapshotBytes(value, maximumByteLength, diagnosticPath) {
  if (!(value instanceof Uint8Array) ||
      Object.getPrototypeOf(value) !== Uint8Array.prototype ||
      value.buffer instanceof SharedArrayBuffer ||
      value.byteLength === 0 ||
      value.byteLength > maximumByteLength) {
    resource(
      diagnosticPath,
      "expected one bounded plain unshared Uint8Array",
    );
  }
  return new Uint8Array(value);
}

function exactObject(value, keys, diagnosticPath) {
  if (typeof value !== "object" || value === null ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length ||
      actual.some((key) =>
        typeof key !== "string" || !keys.includes(key))) {
    invalid(diagnosticPath, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor) ||
        descriptor.enumerable !== true) {
      invalid(`${diagnosticPath}.${key}`, "expected one data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function absolutePath(value, diagnosticPath) {
  if (typeof value !== "string" || !isAbsolute(value) ||
      value.includes("\0") || normalize(value) !== value) {
    invalid(
      diagnosticPath,
      "expected one canonical absolute NUL-free POSIX path",
    );
  }
  return value;
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev &&
    left.ino === right.ino &&
    left.mode === right.mode &&
    left.nlink === right.nlink &&
    left.size === right.size &&
    left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function sameBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function mismatch(path, message) {
  throw new CppCuteBrowserFullDistributionMaterializationError(
    path,
    message,
  );
}

function resource(path, message) {
  throw new CppCuteBrowserFullDistributionMaterializationError(
    path,
    message,
  );
}

function unverified(path, message, options) {
  throw new CppCuteBrowserFullDistributionMaterializationError(
    path,
    message,
    options,
  );
}

function invalid(path, message, options) {
  throw new CppCuteBrowserFullDistributionMaterializationError(
    path,
    message,
    options,
  );
}

async function main() {
  try {
    const result =
      await runCppCuteBrowserFullDistributionMaterialization(
        process.argv.slice(2),
      );
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (cause) {
    const error = cause instanceof Error
      ? cause
      : new Error("unknown full-distribution materialization failure");
    process.stderr.write(`${formatBoundedError(error)}\n`);
    process.exitCode = 1;
  }
}

function formatBoundedError(error) {
  const messages = [];
  let current = error;
  for (let depth = 0; depth < 4 && current instanceof Error; depth += 1) {
    messages.push(current.message.slice(0, 2_048));
    current = current.cause;
  }
  return messages.join("\ncaused by: ");
}

if (process.argv[1] !== undefined &&
    import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
