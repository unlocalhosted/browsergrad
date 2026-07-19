import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import { dirname, isAbsolute } from "node:path/posix";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import { prepareCppCuteBrowserHeaderSourcePlan } from "./cpp_cute_browser_header_source_plan.mjs";

export const CPP_CUTE_BROWSER_CUDA_REDISTRIBUTION_INDEX_SCHEMA =
  "browsergrad.compiler.cpp-cute.cuda-redistribution-index-admission";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-CUDA-REDISTRIBUTION-INDEX";
const INSPECTION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.cuda-redistribution-index-inspection.v1";
const ADMISSION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.cuda-redistribution-index-admission.v1";
const SHA256 = /^[0-9a-f]{64}$/u;
const IDENTIFIER = /^[a-z][a-z0-9-]*$/u;
const COMPONENT_KEY = /^[a-z][a-z0-9_]*$/u;
const WIRE_U64 = /^(0|[1-9][0-9]*)$/u;
const MAX_INDEX_BYTES = 256 * 1024;
const ADMISSIONS = new WeakSet();

export class CppCuteBrowserCudaRedistributionIndexError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserCudaRedistributionIndexError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Inspects one caller-expected CUDA redistribution index without granting
 * current-plan or distribution authority. This narrow seam is useful for
 * deterministic parser tests; release-shaped callers use the admission below.
 */
export function inspectCppCuteBrowserCudaRedistributionIndexBytes(input) {
  const object = exactObject(input, ["bytes", "expected"], "$.input");
  const expected = parseExpectedIndex(object.expected, "$.input.expected");
  const bytes = snapshotBytes(object.bytes, "$.input.bytes");
  if (String(bytes.byteLength) !== expected.byteLength || sha256(bytes) !== expected.sha256) {
    invalid("$.input.bytes", "CUDA redistribution index bytes differ from caller expectation");
  }
  const projection = inspectJson(bytes, expected);
  const inspectionHash = sha256(canonicalJsonBytes({
    domain: INSPECTION_HASH_DOMAIN,
    expected,
    projection,
  }));
  return Object.freeze({
    schema: CPP_CUTE_BROWSER_CUDA_REDISTRIBUTION_INDEX_SCHEMA,
    version: 1,
    indexId: `bg.cpp.cuda-redistribution-index-inspection.sha256.${inspectionHash}`,
    authority: "caller-expected-cuda-redistribution-index-inspection-only",
    sourceUrl: expected.url,
    releaseLabel: projection.releaseLabel,
    releaseProduct: projection.releaseProduct,
    releaseDate: projection.releaseDate,
    indexSha256: expected.sha256,
    indexByteLength: expected.byteLength,
    components: projection.components,
    claims: Object.freeze({
      exactIndexBytesVerified: true,
      selectedComponentMetadataVerified: true,
      exactCurrentHeaderSourcePlanBound: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    }),
  });
}

/**
 * Admits the exact CUDA 12.6.3 index already pinned by all current CUDA header
 * sources. The file must live in a canonical current-user private directory;
 * no network access or legal conclusion occurs here.
 */
export async function admitCppCuteBrowserCudaRedistributionIndex(input) {
  const object = exactObject(input, ["indexPath"], "$.input");
  const indexPath = absolutePath(object.indexPath, "$.input.indexPath");
  const expected = await currentExpectedIndex();
  const bytes = await readPrivateExactFile(indexPath, Number(expected.byteLength));
  const inspected = inspectCppCuteBrowserCudaRedistributionIndexBytes({ bytes, expected });
  const admissionHash = sha256(canonicalJsonBytes({
    domain: ADMISSION_HASH_DOMAIN,
    headerSourcePlanId: expected.headerSourcePlanId,
    inspected,
  }));
  const admission = Object.freeze({
    ...inspected,
    indexId: `bg.cpp.cuda-redistribution-index-admission.sha256.${admissionHash}`,
    authority: "exact-current-header-source-plan-cuda-index-admission-only",
    headerSourcePlanId: expected.headerSourcePlanId,
    claims: Object.freeze({
      ...inspected.claims,
      exactCurrentHeaderSourcePlanBound: true,
    }),
  });
  ADMISSIONS.add(admission);
  return admission;
}

export function requireCppCuteBrowserCudaRedistributionIndexAuthority(admission) {
  if (typeof admission !== "object" || admission === null || !ADMISSIONS.has(admission)) {
    invalid("$.cudaRedistributionIndex", "expected package-issued current CUDA index authority");
  }
}

async function currentExpectedIndex() {
  const plan = await prepareCppCuteBrowserHeaderSourcePlan();
  const cudaSources = plan.body.archives.filter(
    (source) => source.sourceKind === "nvidia-cuda-redist-component",
  );
  if (cudaSources.length !== 3) {
    invalid("$.headerSourcePlan", "current source plan must contain exactly three CUDA components");
  }
  const firstIndex = cudaSources[0]?.index;
  if (firstIndex === undefined || typeof firstIndex.url !== "string" ||
      typeof firstIndex.sha256 !== "string" || typeof firstIndex.byteLength !== "string" ||
      typeof firstIndex.releaseLabel !== "string") {
    invalid("$.headerSourcePlan", "current CUDA index policy is incomplete");
  }
  const components = cudaSources.map((source) => {
    const index = source.index;
    if (index === undefined || index.url !== firstIndex.url || index.sha256 !== firstIndex.sha256 ||
        index.byteLength !== firstIndex.byteLength || index.releaseLabel !== firstIndex.releaseLabel) {
      invalid("$.headerSourcePlan", "current CUDA sources do not share one exact index");
    }
    let acquisition;
    let indexUrl;
    try {
      acquisition = new URL(source.acquisitionUrl);
      indexUrl = new URL(index.url);
    } catch (cause) {
      invalid("$.headerSourcePlan", "current CUDA source URL is malformed", { cause });
    }
    const prefix = `${dirname(indexUrl.pathname)}/`;
    if (acquisition.origin !== indexUrl.origin || !acquisition.pathname.startsWith(prefix)) {
      invalid("$.headerSourcePlan", "CUDA component is outside its pinned redistribution index");
    }
    const relativePath = acquisition.pathname.slice(prefix.length);
    const componentKey = relativePath.split("/")[0];
    if (!COMPONENT_KEY.test(componentKey)) {
      invalid("$.headerSourcePlan", "CUDA component key is not portable");
    }
    return Object.freeze({
      sourceId: source.sourceId,
      componentKey,
      version: source.version,
      relativePath,
      sha256: source.archiveSha256,
      byteLength: source.archiveByteLength,
    });
  }).sort((left, right) => compareUtf8(left.componentKey, right.componentKey));
  return Object.freeze({
    headerSourcePlanId: plan.planId,
    url: firstIndex.url,
    releaseLabel: firstIndex.releaseLabel,
    sha256: firstIndex.sha256,
    byteLength: firstIndex.byteLength,
    components: Object.freeze(components),
  });
}

function parseExpectedIndex(value, diagnosticPath) {
  const object = exactObject(
    value,
    ["byteLength", "components", "headerSourcePlanId", "releaseLabel", "sha256", "url"],
    diagnosticPath,
    true,
  );
  if (typeof object.url !== "string" || !object.url.startsWith("https://") ||
      typeof object.releaseLabel !== "string" || object.releaseLabel === "" ||
      typeof object.sha256 !== "string" || !SHA256.test(object.sha256) ||
      typeof object.byteLength !== "string" || !WIRE_U64.test(object.byteLength) ||
      BigInt(object.byteLength) > BigInt(MAX_INDEX_BYTES)) {
    invalid(diagnosticPath, "expected one bounded HTTPS CUDA index identity");
  }
  if (object.headerSourcePlanId !== undefined &&
      (typeof object.headerSourcePlanId !== "string" || object.headerSourcePlanId === "")) {
    invalid(`${diagnosticPath}.headerSourcePlanId`, "expected one source-plan ID");
  }
  if (!Array.isArray(object.components) || object.components.length === 0 ||
      object.components.length > 16) {
    invalid(`${diagnosticPath}.components`, "expected one bounded component list");
  }
  const components = object.components.map((value_, index) => {
    const path = `${diagnosticPath}.components[${index}]`;
    const component = exactObject(
      value_,
      ["byteLength", "componentKey", "relativePath", "sha256", "sourceId", "version"],
      path,
    );
    if (typeof component.sourceId !== "string" || !IDENTIFIER.test(component.sourceId) ||
        typeof component.componentKey !== "string" || !COMPONENT_KEY.test(component.componentKey) ||
        typeof component.version !== "string" || component.version === "" ||
        typeof component.relativePath !== "string" || component.relativePath === "" ||
        component.relativePath.startsWith("/") || component.relativePath.includes("..") ||
        typeof component.sha256 !== "string" || !SHA256.test(component.sha256) ||
        typeof component.byteLength !== "string" || !WIRE_U64.test(component.byteLength)) {
      invalid(path, "CUDA component expectation is malformed");
    }
    return Object.freeze({
      sourceId: component.sourceId,
      componentKey: component.componentKey,
      version: component.version,
      relativePath: component.relativePath,
      sha256: component.sha256,
      byteLength: component.byteLength,
    });
  }).sort((left, right) => compareUtf8(left.componentKey, right.componentKey));
  for (let index = 1; index < components.length; index += 1) {
    if (components[index - 1].componentKey === components[index].componentKey ||
        components[index - 1].sourceId === components[index].sourceId) {
      invalid(`${diagnosticPath}.components`, "CUDA component keys and source IDs must be unique");
    }
  }
  return Object.freeze({
    ...(object.headerSourcePlanId === undefined ? {} : { headerSourcePlanId: object.headerSourcePlanId }),
    url: object.url,
    releaseLabel: object.releaseLabel,
    sha256: object.sha256,
    byteLength: object.byteLength,
    components: Object.freeze(components),
  });
}

function inspectJson(bytes, expected) {
  let parsed;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (cause) {
    invalid("$.input.bytes", "CUDA redistribution index is not strict UTF-8 JSON", { cause });
  }
  const root = plainObject(parsed, "$.index");
  if (root.release_label !== expected.releaseLabel || root.release_product !== "cuda" ||
      typeof root.release_date !== "string" || !/^\d{4}-\d{2}-\d{2}$/u.test(root.release_date)) {
    invalid("$.index", "CUDA release metadata differs from the expected release");
  }
  const components = expected.components.map((expectedComponent, index) => {
    const path = `$.index.${expectedComponent.componentKey}`;
    const component = plainObject(root[expectedComponent.componentKey], path);
    const platform = plainObject(component["linux-x86_64"], `${path}.linux-x86_64`);
    if (component.license !== "CUDA Toolkit" ||
        component.license_path !== `${expectedComponent.componentKey}/LICENSE.txt` ||
        component.version !== expectedComponent.version ||
        platform.relative_path !== expectedComponent.relativePath ||
        platform.sha256 !== expectedComponent.sha256 ||
        platform.size !== expectedComponent.byteLength) {
      invalid(path, "CUDA component metadata differs from the selected source archive");
    }
    return Object.freeze({
      ordinal: index,
      sourceId: expectedComponent.sourceId,
      componentKey: expectedComponent.componentKey,
      name: boundedString(component.name, `${path}.name`),
      version: expectedComponent.version,
      license: "CUDA Toolkit",
      licensePath: component.license_path,
      platform: "linux-x86_64",
      relativePath: expectedComponent.relativePath,
      archiveSha256: expectedComponent.sha256,
      archiveByteLength: expectedComponent.byteLength,
    });
  });
  return Object.freeze({
    releaseLabel: root.release_label,
    releaseProduct: root.release_product,
    releaseDate: root.release_date,
    components: Object.freeze(components),
  });
}

async function readPrivateExactFile(path, expectedByteLength) {
  const parent = dirname(path);
  let parentStat;
  let fileStat;
  try {
    if (await realpath(parent) !== parent) invalid("$.input.indexPath.parent", "parent must be canonical");
    parentStat = await lstat(parent, { bigint: true });
    fileStat = await lstat(path, { bigint: true });
  } catch (cause) {
    if (cause instanceof CppCuteBrowserCudaRedistributionIndexError) throw cause;
    invalid("$.input.indexPath", "CUDA redistribution index is unavailable", { cause });
  }
  const uid = typeof process.getuid === "function" ? BigInt(process.getuid()) : undefined;
  if (!parentStat.isDirectory() || parentStat.isSymbolicLink() ||
      Number(parentStat.mode & 0o077n) !== 0 || (uid !== undefined && parentStat.uid !== uid)) {
    invalid("$.input.indexPath.parent", "index parent must be one current-user private directory");
  }
  if (!fileStat.isFile() || fileStat.isSymbolicLink() || fileStat.nlink !== 1n ||
      fileStat.size !== BigInt(expectedByteLength) || (uid !== undefined && fileStat.uid !== uid) ||
      Number(fileStat.mode & 0o022n) !== 0 || await realpath(path) !== path) {
    invalid("$.input.indexPath", "index must be one exact current-user non-writable regular file");
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat({ bigint: true });
    if (!sameFileIdentity(fileStat, before)) invalid("$.input.indexPath", "index identity changed before read");
    const bytes = new Uint8Array(expectedByteLength);
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesRead } = await handle.read(bytes, offset, bytes.byteLength - offset, offset);
      if (bytesRead <= 0) invalid("$.input.indexPath", "index changed while read");
      offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    if (!sameFileIdentity(before, after)) invalid("$.input.indexPath", "index changed while read");
    return bytes;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserCudaRedistributionIndexError) throw cause;
    invalid("$.input.indexPath", "failed to read CUDA redistribution index", { cause });
  } finally {
    await handle?.close();
  }
}

function snapshotBytes(value, diagnosticPath) {
  if (!(value instanceof Uint8Array) || Object.getPrototypeOf(value) !== Uint8Array.prototype ||
      !(value.buffer instanceof ArrayBuffer)) {
    invalid(diagnosticPath, "expected one ordinary Uint8Array");
  }
  let bytes;
  try {
    bytes = new Uint8Array(value.byteLength);
    bytes.set(value);
  } catch (cause) {
    invalid(diagnosticPath, "failed to snapshot CUDA index bytes", { cause });
  }
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_INDEX_BYTES) {
    invalid(diagnosticPath, "CUDA index exceeds the byte bound");
  }
  return bytes;
}

function plainObject(value, diagnosticPath) {
  if (typeof value !== "object" || value === null || Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain JSON object");
  }
  return value;
}

function boundedString(value, diagnosticPath) {
  if (typeof value !== "string" || value === "" || value.length > 256) {
    invalid(diagnosticPath, "expected one bounded nonempty string");
  }
  return value;
}

function exactObject(value, keys, diagnosticPath, optional = false) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.some((key) => typeof key !== "string" || !keys.includes(key)) ||
      (!optional && actual.length !== keys.length)) {
    invalid(diagnosticPath, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined) continue;
    if (!("value" in descriptor)) invalid(`${diagnosticPath}.${key}`, "expected data property");
    result[key] = descriptor.value;
  }
  return result;
}

function absolutePath(value, diagnosticPath) {
  if (typeof value !== "string" || !isAbsolute(value) || value.includes("\0")) {
    invalid(diagnosticPath, "expected one absolute NUL-free POSIX path");
  }
  return value;
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino && left.mode === right.mode &&
    left.nlink === right.nlink && left.size === right.size && left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function invalid(path, message, options) {
  throw new CppCuteBrowserCudaRedistributionIndexError(path, message, options);
}
