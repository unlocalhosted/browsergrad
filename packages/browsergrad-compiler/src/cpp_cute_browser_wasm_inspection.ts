import {
  deepFreezeJson,
  hashCanonicalJson,
  sha256Hex,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  type PreparedCppCuteBrowserRuntimeAbiManifest,
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest,
} from "./cpp_cute_browser_runtime_abi.js";

/** Matches the closed browser profile's per-asset compressed-byte ceiling. */
export const CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH = 256 * 1024 * 1024;
export const CPP_CUTE_BROWSER_WASM_BASE_OPERATIONS = 8_000_000;
export const CPP_CUTE_BROWSER_WASM_MAX_OPERATIONS = CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH * 2;
export const CPP_CUTE_BROWSER_WASM_REPORT_MAX_ARRAY_LENGTH = 250_000;
export const CPP_CUTE_BROWSER_WASM_REPORT_MAX_BYTE_LENGTH = 64 * 1024 * 1024;
export const CPP_CUTE_BROWSER_WASM_REPORT_MAX_NODES = 1_000_000;

const MAX_SECTIONS = 32;
const MAX_CUSTOM_SECTIONS = 4;
const MAX_CUSTOM_SECTION_BYTES = 512 * 1024;
const MAX_CUSTOM_TOTAL_BYTES = 1024 * 1024;
const MAX_TYPES = 65_536;
const MAX_IMPORTS = 256;
const MAX_FUNCTIONS = 250_000;
const MAX_EXPORTS = 64;
const MAX_GLOBALS = 4_096;
const MAX_SEGMENTS = 65_536;
const MAX_PARAMETERS = 64;
const MAX_RESULTS = 1;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const MAX_LOCALS = 50_000;
// Emscripten's structured CFG output can exceed 1,024 nested controls. The
// parser is iterative, and this independent ceiling bounds its control array.
const MAX_CONTROL_DEPTH = 8_192;
const MAX_NAME_BYTES = 1_024;
const MAX_TOTAL_NAME_BYTES = 256 * 1024;

const TEXT_DECODER = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
const TEXT_ENCODER = new TextEncoder();
const WEBASSEMBLY_VALIDATE = typeof WebAssembly === "undefined" ? undefined : WebAssembly.validate;
const CONFORMANCES = new WeakMap<object, StoredConformance>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

const SECTION_RANK = new Map<number, number>([
  [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [13, 6], [6, 7],
  [7, 8], [8, 9], [9, 10], [12, 11], [10, 12], [11, 13],
]);
const SECTION_NAME = new Map<number, string>([
  [0, "custom"], [1, "type"], [2, "import"], [3, "function"],
  [4, "table"], [5, "memory"], [13, "tag"], [6, "global"],
  [7, "export"], [8, "start"], [9, "element"], [12, "data-count"],
  [10, "code"], [11, "data"],
]);

type ValueType = "f32" | "f64" | "i32" | "i64";
type ExternalKind = "function" | "global" | "memory" | "table" | "tag";
type ExtensionFeature =
  | "atomics"
  | "bulk-memory"
  | "bulk-memory-opt"
  | "exception-handling"
  | "memory64"
  | "multi-memory"
  | "mutable-globals"
  | "nontrapping-fptoint"
  | "sign-extension"
  | "simd128"
  | "threads";

export interface InspectCppCuteBrowserWasmOptions {
  readonly signal?: AbortSignal;
  readonly maxModuleByteLength?: number;
  readonly maxOperations?: number;
}

export type CppCuteBrowserWasmInspectionErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-ABI-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-UNVERIFIED";

export class CppCuteBrowserWasmInspectionError extends Error {
  constructor(
    readonly code: CppCuteBrowserWasmInspectionErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWasmInspectionError";
  }
}

export interface CppCuteBrowserWasmFunctionTypeProjection {
  readonly parameters: readonly ValueType[];
  readonly results: readonly ValueType[];
}

export interface CppCuteBrowserWasmImportProjection {
  readonly module: string;
  readonly name: string;
  readonly kind: ExternalKind;
  readonly functionType?: CppCuteBrowserWasmFunctionTypeProjection;
  readonly table?: CppCuteBrowserWasmTableProjection;
  readonly memory?: CppCuteBrowserWasmMemoryProjection;
  readonly global?: CppCuteBrowserWasmGlobalTypeProjection;
}

export interface CppCuteBrowserWasmExportProjection {
  readonly name: string;
  readonly kind: ExternalKind;
  readonly index: number;
  readonly functionType?: CppCuteBrowserWasmFunctionTypeProjection;
  readonly table?: CppCuteBrowserWasmTableProjection;
  readonly memory?: CppCuteBrowserWasmMemoryProjection;
  readonly global?: CppCuteBrowserWasmGlobalProjection;
}

export interface CppCuteBrowserWasmTableProjection {
  readonly elementType: "funcref";
  readonly minimum: number;
  readonly maximum: number | null;
}

export interface CppCuteBrowserWasmMemoryProjection {
  readonly addressType: "i32";
  readonly shared: false;
  readonly minimumPages: number;
  readonly maximumPages: number | null;
}

export interface CppCuteBrowserWasmGlobalTypeProjection {
  readonly valueType: ValueType;
  readonly mutable: boolean;
}

export interface CppCuteBrowserWasmGlobalProjection extends CppCuteBrowserWasmGlobalTypeProjection {
  readonly initializerOpcode: number | null;
}

export interface CppCuteBrowserWasmTargetFeatureProjection {
  readonly prefix: "+" | "-";
  readonly wireName: string;
  readonly normalizedName: string;
}

export interface CppCuteBrowserWasmCustomSectionProjection {
  readonly name: string;
  readonly payloadByteLength: number;
  readonly payloadSha256: string;
  readonly targetFeatures?: readonly CppCuteBrowserWasmTargetFeatureProjection[];
}

export interface CppCuteBrowserWasmInspectionProjection {
  readonly sectionOrder: readonly string[];
  readonly types: readonly CppCuteBrowserWasmFunctionTypeProjection[];
  readonly imports: readonly CppCuteBrowserWasmImportProjection[];
  readonly definedFunctionTypeIndices: readonly number[];
  readonly tables: readonly CppCuteBrowserWasmTableProjection[];
  readonly memories: readonly CppCuteBrowserWasmMemoryProjection[];
  readonly globals: readonly CppCuteBrowserWasmGlobalProjection[];
  readonly exports: readonly CppCuteBrowserWasmExportProjection[];
  readonly startFunctionIndex: number | null;
  readonly elementSegments: readonly CppCuteBrowserWasmElementSegmentProjection[];
  readonly codeBodies: readonly CppCuteBrowserWasmCodeBodyProjection[];
  readonly dataCount: number | null;
  readonly dataSegments: readonly CppCuteBrowserWasmDataSegmentProjection[];
  readonly tagCount: number;
  readonly customSections: readonly CppCuteBrowserWasmCustomSectionProjection[];
  readonly staticallyUsedExtensions: readonly ExtensionFeature[];
}

export interface CppCuteBrowserWasmElementSegmentProjection {
  readonly mode: "active" | "declarative" | "passive";
  readonly tableIndex: number | null;
  readonly elementType: "funcref";
  readonly elementCount: number;
}

export interface CppCuteBrowserWasmCodeBodyProjection {
  readonly bodyByteLength: number;
  readonly localCount: number;
}

export interface CppCuteBrowserWasmDataSegmentProjection {
  readonly mode: "active" | "passive";
  readonly memoryIndex: number | null;
  readonly byteLength: number;
}

/** Copy-safe review evidence. It is deliberately not an execution authority. */
export interface CppCuteBrowserWasmInspectionReport {
  readonly authority: "review-observation-only";
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly observedProjectionSha256: string;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiContractSha256: string;
  readonly exactInterfaceConformance: boolean;
  readonly mismatches: readonly string[];
  readonly projection: CppCuteBrowserWasmInspectionProjection;
  readonly rawWasmVerified: true;
  readonly workerExecutionReady: false;
  readonly releaseReady: false;
}

declare const preparedCppCuteBrowserWasmConformanceBrand: unique symbol;

/** Opaque exact-byte/raw-structure conformance gate. It is not Worker or release authority. */
export interface PreparedCppCuteBrowserWasmConformance {
  readonly [preparedCppCuteBrowserWasmConformanceBrand]: true;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly observedProjectionSha256: string;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiContractSha256: string;
  readonly exactInterfaceConformance: true;
  readonly rawWasmVerified: true;
  readonly workerExecutionReady: false;
  readonly releaseReady: false;
}

export interface PreparedCppCuteBrowserWasmConformanceRecord {
  readonly summary: CppCuteBrowserWasmInspectionReport & {
    readonly exactInterfaceConformance: true;
    readonly mismatches: readonly [];
  };
}

interface StoredConformance {
  readonly summary: PreparedCppCuteBrowserWasmConformanceRecord["summary"];
}

interface NormalizedOptions {
  readonly signal: AbortSignal | undefined;
  readonly maxModuleByteLength: number;
  readonly maxOperations: number | undefined;
}

interface ParserOptions extends NormalizedOptions {
  readonly maxOperations: number;
}

interface MutableFunctionType {
  readonly parameters: ValueType[];
  readonly results: ValueType[];
}

interface MutableImport {
  readonly module: string;
  readonly name: string;
  readonly kind: ExternalKind;
  readonly typeIndex?: number;
  readonly table?: CppCuteBrowserWasmTableProjection;
  readonly memory?: CppCuteBrowserWasmMemoryProjection;
  readonly global?: CppCuteBrowserWasmGlobalTypeProjection;
}

interface MutableExport {
  readonly name: string;
  readonly kind: ExternalKind;
  readonly index: number;
}

interface MutableCustomSection {
  readonly name: string;
  readonly payload: Uint8Array;
  readonly targetFeatures?: CppCuteBrowserWasmTargetFeatureProjection[];
}

interface ParsedModule {
  readonly sectionOrder: string[];
  readonly types: MutableFunctionType[];
  readonly imports: MutableImport[];
  readonly definedFunctionTypeIndices: number[];
  readonly tables: CppCuteBrowserWasmTableProjection[];
  readonly memories: CppCuteBrowserWasmMemoryProjection[];
  readonly globals: CppCuteBrowserWasmGlobalProjection[];
  readonly exports: MutableExport[];
  startFunctionIndex: number | null;
  readonly elementSegments: CppCuteBrowserWasmElementSegmentProjection[];
  readonly codeBodies: CppCuteBrowserWasmCodeBodyProjection[];
  dataCount: number | null;
  readonly dataSegments: CppCuteBrowserWasmDataSegmentProjection[];
  tagCount: number;
  readonly customSections: MutableCustomSection[];
  readonly features: Set<ExtensionFeature>;
}

interface ParserState {
  operations: number;
  totalNameBytes: number;
  readonly options: ParserOptions;
}

class Cursor {
  offset = 0;

  constructor(
    readonly bytes: Uint8Array,
    readonly path: string,
    readonly state: ParserState,
  ) {}

  get remaining(): number {
    return this.bytes.byteLength - this.offset;
  }

  readByte(path = this.path): number {
    tick(this.state);
    if (this.offset >= this.bytes.byteLength) invalid(path, "unexpected end of Wasm bytes");
    const value = this.bytes[this.offset];
    this.offset += 1;
    return value as number;
  }

  peekByte(path = this.path): number {
    if (this.offset >= this.bytes.byteLength) invalid(path, "unexpected end of Wasm bytes");
    return this.bytes[this.offset] as number;
  }

  readBytes(byteLength: number, path = this.path): Uint8Array {
    tick(this.state);
    if (!Number.isSafeInteger(byteLength) || byteLength < 0 || byteLength > this.remaining) {
      invalid(path, "declared byte length exceeds the enclosing Wasm boundary");
    }
    const start = this.offset;
    this.offset += byteLength;
    return this.bytes.subarray(start, this.offset);
  }

  child(byteLength: number, path: string): Cursor {
    return new Cursor(this.readBytes(byteLength, path), path, this.state);
  }

  expectEnd(path = this.path): void {
    if (this.remaining !== 0) invalid(path, `${this.remaining} trailing byte(s) remain`);
  }
}

/**
 * CPU-bounded review of exact module bytes. Parsing, engine validation, and
 * hashing are synchronous/non-preemptible phases; callers handling untrusted
 * or large assets must run this function in a disposable verifier Worker and
 * use Worker termination as the hard cancellation boundary.
 */
export async function inspectCppCuteBrowserWasmAgainstRuntimeAbi(
  bytes: Uint8Array,
  runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest,
  options: InspectCppCuteBrowserWasmOptions = {},
): Promise<CppCuteBrowserWasmInspectionReport> {
  const normalized = normalizeOptions(options);
  const abi = unwrapAbi(runtimeAbi);
  assertAbiInspectionCeilings(abi.manifest.body);
  throwIfAborted(normalized.signal);
  const snapshot = snapshotBytes(bytes, normalized.maxModuleByteLength);
  throwIfAborted(normalized.signal);
  const parsed = parseModule(snapshot, resolveParserOptions(normalized, snapshot.byteLength));
  validateWasmSemantics(snapshot);
  throwIfAborted(normalized.signal);
  const customSections = await Promise.all(parsed.customSections.map(async (section) => ({
    name: section.name,
    payloadByteLength: section.payload.byteLength,
    payloadSha256: await hashBytes(section.payload, `$.customSections.${section.name}`),
    ...(section.targetFeatures === undefined
      ? {}
      : { targetFeatures: section.targetFeatures }),
  })));
  throwIfAborted(normalized.signal);
  const projection = makeProjection(parsed, customSections);
  const mismatches = compareAgainstAbi(projection, abi.manifest.body);
  const [wasmSha256, observedProjectionSha256] = await Promise.all([
    hashBytes(snapshot, "$bytes"),
    hashProjection(projection),
  ]);
  throwIfAborted(normalized.signal);
  return freezeJson({
    authority: "review-observation-only",
    wasmSha256,
    wasmByteLength: snapshot.byteLength,
    observedProjectionSha256,
    runtimeAbiManifestId: runtimeAbi.manifestId,
    runtimeAbiContractSha256: runtimeAbi.contractSha256,
    exactInterfaceConformance: mismatches.length === 0,
    mismatches,
    projection,
    rawWasmVerified: true,
    workerExecutionReady: false,
    releaseReady: false,
  });
}

export async function verifyCppCuteBrowserWasmConformance(
  bytes: Uint8Array,
  runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest,
  options: InspectCppCuteBrowserWasmOptions = {},
): Promise<PreparedCppCuteBrowserWasmConformance> {
  const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, runtimeAbi, options);
  if (!report.exactInterfaceConformance) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-ABI-MISMATCH",
      "$bytes",
      `observed Wasm does not conform to the prepared runtime ABI: ${report.mismatches.join("; ")}`,
    );
  }
  const summary = report as PreparedCppCuteBrowserWasmConformanceRecord["summary"];
  const prepared = Object.freeze({
    wasmSha256: report.wasmSha256,
    wasmByteLength: report.wasmByteLength,
    observedProjectionSha256: report.observedProjectionSha256,
    runtimeAbiManifestId: report.runtimeAbiManifestId,
    runtimeAbiContractSha256: report.runtimeAbiContractSha256,
    exactInterfaceConformance: true,
    rawWasmVerified: true,
    workerExecutionReady: false,
    releaseReady: false,
  }) as PreparedCppCuteBrowserWasmConformance;
  CONFORMANCES.set(prepared, Object.freeze({ summary }));
  return prepared;
}

export function unwrapPreparedCppCuteBrowserWasmConformance(
  prepared: PreparedCppCuteBrowserWasmConformance,
): PreparedCppCuteBrowserWasmConformanceRecord {
  if (typeof prepared !== "object" || prepared === null) unverified("$prepared", "expected opaque raw-Wasm conformance authority");
  const stored = CONFORMANCES.get(prepared as object);
  if (stored === undefined ||
      prepared.wasmSha256 !== stored.summary.wasmSha256 ||
      prepared.wasmByteLength !== stored.summary.wasmByteLength ||
      prepared.observedProjectionSha256 !== stored.summary.observedProjectionSha256 ||
      prepared.runtimeAbiManifestId !== stored.summary.runtimeAbiManifestId ||
      prepared.runtimeAbiContractSha256 !== stored.summary.runtimeAbiContractSha256 ||
      prepared.exactInterfaceConformance !== true || prepared.rawWasmVerified !== true ||
      prepared.workerExecutionReady !== false || prepared.releaseReady !== false) {
    unverified("$prepared", "raw-Wasm conformance authority is forged or mutated");
  }
  return Object.freeze({ summary: cloneFrozen(stored.summary) });
}

function parseModule(bytes: Uint8Array, options: ParserOptions): ParsedModule {
  const state: ParserState = { operations: 0, totalNameBytes: 0, options };
  const cursor = new Cursor(bytes, "$bytes", state);
  const magic = cursor.readBytes(4, "$bytes.magic");
  if (!equalBytes(magic, new Uint8Array([0x00, 0x61, 0x73, 0x6d]))) invalid("$bytes.magic", "invalid WebAssembly magic");
  const version = cursor.readBytes(4, "$bytes.version");
  if (!equalBytes(version, new Uint8Array([0x01, 0x00, 0x00, 0x00]))) invalid("$bytes.version", "only WebAssembly binary version 1 is supported");

  const parsed: ParsedModule = {
    sectionOrder: [], types: [], imports: [], definedFunctionTypeIndices: [],
    tables: [], memories: [], globals: [], exports: [], startFunctionIndex: null,
    elementSegments: [], codeBodies: [], dataCount: null, dataSegments: [],
    tagCount: 0, customSections: [], features: new Set<ExtensionFeature>(),
  };
  const seen = new Set<number>();
  let lastRank = 0;
  let sectionCount = 0;
  let customBytes = 0;
  while (cursor.remaining > 0) {
    sectionCount += 1;
    if (sectionCount > MAX_SECTIONS) resource("$bytes.sections", `section count exceeds ${MAX_SECTIONS}`);
    const id = cursor.readByte("$bytes.sections.id");
    const name = SECTION_NAME.get(id);
    const rank = SECTION_RANK.get(id);
    if (name === undefined || (id !== 0 && rank === undefined)) invalid("$bytes.sections.id", `unknown standard section id ${id}`);
    const byteLength = readU32(cursor, `$bytes.sections.${name}.byteLength`);
    const section = cursor.child(byteLength, `$bytes.sections.${name}`);
    parsed.sectionOrder.push(name);
    if (id === 0) {
      if (parsed.customSections.length >= MAX_CUSTOM_SECTIONS) resource("$bytes.sections.custom", `custom section count exceeds ${MAX_CUSTOM_SECTIONS}`);
      if (byteLength > MAX_CUSTOM_SECTION_BYTES) resource("$bytes.sections.custom", `custom section exceeds ${MAX_CUSTOM_SECTION_BYTES} bytes`);
      customBytes += byteLength;
      if (customBytes > MAX_CUSTOM_TOTAL_BYTES) resource("$bytes.sections.custom", `custom sections exceed ${MAX_CUSTOM_TOTAL_BYTES} aggregate bytes`);
      parseCustomSection(section, parsed);
    } else {
      if (seen.has(id)) invalid(`$bytes.sections.${name}`, `${name} section occurs more than once`);
      if ((rank as number) <= lastRank) invalid(`$bytes.sections.${name}`, `${name} section is out of canonical order`);
      seen.add(id);
      lastRank = rank as number;
      parseStandardSection(id, section, parsed);
    }
    section.expectEnd();
  }
  finishModule(parsed);
  return parsed;
}

function parseStandardSection(id: number, cursor: Cursor, parsed: ParsedModule): void {
  switch (id) {
    case 1: parseTypeSection(cursor, parsed); return;
    case 2: parseImportSection(cursor, parsed); return;
    case 3: parseFunctionSection(cursor, parsed); return;
    case 4: parseTableSection(cursor, parsed); return;
    case 5: parseMemorySection(cursor, parsed); return;
    case 13: parseTagSection(cursor, parsed); return;
    case 6: parseGlobalSection(cursor, parsed); return;
    case 7: parseExportSection(cursor, parsed); return;
    case 8: parseStartSection(cursor, parsed); return;
    case 9: parseElementSection(cursor, parsed); return;
    case 12: parseDataCountSection(cursor, parsed); return;
    case 10: parseCodeSection(cursor, parsed); return;
    case 11: parseDataSection(cursor, parsed); return;
    default: invalid(cursor.path, `unsupported standard section id ${id}`);
  }
}

function parseTypeSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, MAX_TYPES, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) {
    if (cursor.readByte(`${cursor.path}[${index}].form`) !== 0x60) invalid(`${cursor.path}[${index}].form`, "only function types are supported");
    const parameterCount = boundedVectorCount(cursor, MAX_PARAMETERS, `${cursor.path}[${index}].parameters.count`);
    const parameters: ValueType[] = [];
    for (let parameter = 0; parameter < parameterCount; parameter += 1) parameters.push(readValueType(cursor, `${cursor.path}[${index}].parameters[${parameter}]`));
    const resultCount = boundedVectorCount(cursor, MAX_RESULTS, `${cursor.path}[${index}].results.count`);
    const results: ValueType[] = [];
    for (let result = 0; result < resultCount; result += 1) results.push(readValueType(cursor, `${cursor.path}[${index}].results[${result}]`));
    parsed.types.push({ parameters, results });
  }
}

function parseImportSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, MAX_IMPORTS, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) {
    const path = `${cursor.path}[${index}]`;
    const module = readName(cursor, `${path}.module`);
    const name = readName(cursor, `${path}.name`);
    const kindByte = cursor.readByte(`${path}.kind`);
    switch (kindByte) {
      case 0: {
        const typeIndex = readU32(cursor, `${path}.typeIndex`);
        requireIndex(typeIndex, parsed.types.length, `${path}.typeIndex`, "type");
        parsed.imports.push({ module, name, kind: "function", typeIndex });
        break;
      }
      case 1: parsed.imports.push({ module, name, kind: "table", table: readTableType(cursor, path) }); break;
      case 2: parsed.imports.push({ module, name, kind: "memory", memory: readMemoryType(cursor, parsed, path) }); break;
      case 3: parsed.imports.push({ module, name, kind: "global", global: readGlobalType(cursor, parsed, path) }); break;
      case 4:
        parsed.features.add("exception-handling");
        readTagType(cursor, parsed, path);
        parsed.imports.push({ module, name, kind: "tag" });
        break;
      default: invalid(`${path}.kind`, `unknown import kind ${kindByte}`);
    }
  }
}

function parseFunctionSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, MAX_FUNCTIONS, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) {
    const typeIndex = readU32(cursor, `${cursor.path}[${index}]`);
    requireIndex(typeIndex, parsed.types.length, `${cursor.path}[${index}]`, "type");
    parsed.definedFunctionTypeIndices.push(typeIndex);
  }
}

function parseTableSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, 1, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) parsed.tables.push(readTableType(cursor, `${cursor.path}[${index}]`));
}

function parseMemorySection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, 1, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) parsed.memories.push(readMemoryType(cursor, parsed, `${cursor.path}[${index}]`));
}

function parseTagSection(cursor: Cursor, parsed: ParsedModule): void {
  parsed.features.add("exception-handling");
  const count = boundedVectorCount(cursor, MAX_GLOBALS, `${cursor.path}.count`);
  parsed.tagCount = count;
  for (let index = 0; index < count; index += 1) readTagType(cursor, parsed, `${cursor.path}[${index}]`);
}

function parseGlobalSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, MAX_GLOBALS, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) {
    const path = `${cursor.path}[${index}]`;
    const type = readGlobalType(cursor, parsed, path);
    const initializerOpcode = parseConstExpression(cursor, parsed, `${path}.initializer`);
    parsed.globals.push({ ...type, initializerOpcode });
  }
}

function parseExportSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, MAX_EXPORTS, `${cursor.path}.count`);
  const names = new Set<string>();
  for (let index = 0; index < count; index += 1) {
    const path = `${cursor.path}[${index}]`;
    const name = readName(cursor, `${path}.name`);
    if (names.has(name)) invalid(`${path}.name`, `duplicate export name ${name}`);
    names.add(name);
    const kind = readExternalKind(cursor, `${path}.kind`);
    const targetIndex = readU32(cursor, `${path}.index`);
    parsed.exports.push({ name, kind, index: targetIndex });
  }
}

function parseStartSection(cursor: Cursor, parsed: ParsedModule): void {
  parsed.startFunctionIndex = readU32(cursor, cursor.path);
}

function parseElementSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, MAX_SEGMENTS, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) {
    const path = `${cursor.path}[${index}]`;
    const flags = readU32(cursor, `${path}.flags`);
    if (flags > 7) invalid(`${path}.flags`, `unsupported element segment flags ${flags}`);
    if (flags >= 4) invalid(`${path}.flags`, "element-expression forms require the unlisted reference-types extension");
    if (flags !== 0) parsed.features.add("bulk-memory");
    let mode: "active" | "declarative" | "passive";
    let tableIndex: number | null = null;
    if (flags === 0 || flags === 2 || flags === 4 || flags === 6) {
      mode = "active";
      tableIndex = flags === 2 || flags === 6 ? readU32(cursor, `${path}.tableIndex`) : 0;
      parseConstExpression(cursor, parsed, `${path}.offset`);
    } else {
      mode = flags === 3 || flags === 7 ? "declarative" : "passive";
    }
    const expressions = false;
    let elementType = "funcref" as const;
    if (flags === 1 || flags === 2 || flags === 3) {
      if (cursor.readByte(`${path}.elementKind`) !== 0x00) invalid(`${path}.elementKind`, "only funcref element kind is supported");
    }
    const elementCount = boundedVectorCount(cursor, MAX_FUNCTIONS, `${path}.elements.count`);
    for (let element = 0; element < elementCount; element += 1) {
      if (expressions) parseReferenceConstExpression(cursor, parsed, `${path}.elements[${element}]`);
      else requireIndex(
        readU32(cursor, `${path}.elements[${element}]`),
        totalFunctionCount(parsed),
        `${path}.elements[${element}]`,
        "function",
      );
    }
    parsed.elementSegments.push({ mode, tableIndex, elementType, elementCount });
  }
}

function parseDataCountSection(cursor: Cursor, parsed: ParsedModule): void {
  parsed.features.add("bulk-memory");
  parsed.dataCount = boundedCount(readU32(cursor, cursor.path), MAX_SEGMENTS, cursor.path);
}

function parseCodeSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, MAX_FUNCTIONS, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) {
    const path = `${cursor.path}[${index}]`;
    const bodyByteLength = boundedCount(readU32(cursor, `${path}.byteLength`), MAX_BODY_BYTES, `${path}.byteLength`);
    const body = cursor.child(bodyByteLength, path);
    const localGroupCount = boundedVectorCount(body, MAX_LOCALS, `${path}.locals.groupCount`);
    let localCount = 0;
    for (let group = 0; group < localGroupCount; group += 1) {
      const repetitions = readU32(body, `${path}.locals[${group}].count`);
      localCount += repetitions;
      if (!Number.isSafeInteger(localCount) || localCount > MAX_LOCALS) resource(`${path}.locals`, `local count exceeds ${MAX_LOCALS}`);
      readValueType(body, `${path}.locals[${group}].type`);
    }
    const typeIndex = parsed.definedFunctionTypeIndices[index];
    if (typeIndex === undefined) invalid(path, "code body has no matching function declaration");
    const parameters = parsed.types[typeIndex]?.parameters.length;
    if (parameters === undefined) invalid(path, "code body function type cannot be resolved");
    parseInstructions(body, parsed, path, parameters + localCount);
    body.expectEnd(path);
    parsed.codeBodies.push({ bodyByteLength, localCount });
  }
}

function parseDataSection(cursor: Cursor, parsed: ParsedModule): void {
  const count = boundedVectorCount(cursor, MAX_SEGMENTS, `${cursor.path}.count`);
  for (let index = 0; index < count; index += 1) {
    const path = `${cursor.path}[${index}]`;
    const flags = readU32(cursor, `${path}.flags`);
    let mode: "active" | "passive";
    let memoryIndex: number | null;
    if (flags === 0) {
      mode = "active"; memoryIndex = 0; parseConstExpression(cursor, parsed, `${path}.offset`);
    } else if (flags === 1) {
      parsed.features.add("bulk-memory"); mode = "passive"; memoryIndex = null;
    } else if (flags === 2) {
      parsed.features.add("bulk-memory"); mode = "active";
      memoryIndex = readU32(cursor, `${path}.memoryIndex`);
      if (memoryIndex !== 0) parsed.features.add("multi-memory");
      parseConstExpression(cursor, parsed, `${path}.offset`);
    } else invalid(`${path}.flags`, `unsupported data segment flags ${flags}`);
    const byteLength = readU32(cursor, `${path}.byteLength`);
    cursor.readBytes(byteLength, `${path}.bytes`);
    parsed.dataSegments.push({ mode, memoryIndex, byteLength });
  }
}

function parseCustomSection(cursor: Cursor, parsed: ParsedModule): void {
  const name = readName(cursor, `${cursor.path}.name`);
  if (parsed.customSections.some((section) => section.name === name)) invalid(`${cursor.path}.name`, `duplicate custom section name ${name}`);
  const payload = new Uint8Array(cursor.readBytes(cursor.remaining, `${cursor.path}.payload`));
  if (name === "dylink.0" || name === "producers" || name === "sourceMappingURL") invalid(`${cursor.path}.name`, `custom section ${name} is forbidden`);
  const targetFeatures = name === "target_features" ? parseTargetFeatures(payload, cursor.state) : undefined;
  parsed.customSections.push({ name, payload, ...(targetFeatures === undefined ? {} : { targetFeatures }) });
}

function parseTargetFeatures(payload: Uint8Array, state: ParserState): CppCuteBrowserWasmTargetFeatureProjection[] {
  const cursor = new Cursor(payload, "$bytes.sections.custom.target_features.payload", state);
  const count = boundedVectorCount(cursor, 64, `${cursor.path}.count`);
  const entries: CppCuteBrowserWasmTargetFeatureProjection[] = [];
  const seen = new Set<string>();
  for (let index = 0; index < count; index += 1) {
    const path = `${cursor.path}[${index}]`;
    const prefixByte = cursor.readByte(`${path}.prefix`);
    if (prefixByte !== 0x2b && prefixByte !== 0x2d) invalid(`${path}.prefix`, "target feature prefix must be + or -");
    const wireName = readName(cursor, `${path}.name`);
    if (!/^[A-Za-z0-9._+@=-]+$/u.test(wireName) || wireName.length === 0) invalid(`${path}.name`, "target feature name is outside the tool-conventions ASCII grammar");
    if (seen.has(wireName)) invalid(`${path}.name`, "target feature names must be unique");
    seen.add(wireName);
    const normalizedName = normalizeTargetFeatureName(wireName);
    entries.push({ prefix: prefixByte === 0x2b ? "+" : "-", wireName, normalizedName });
  }
  cursor.expectEnd();
  return entries;
}

function parseInstructions(cursor: Cursor, parsed: ParsedModule, path: string, localCount: number): void {
  const controls: Array<{ readonly kind: "block" | "function" | "if" | "loop"; elseSeen: boolean }> = [
    { kind: "function", elseSeen: false },
  ];
  while (controls.length > 0) {
    const opcodePath = `${path}.instructions@${cursor.offset}`;
    const opcode = cursor.readByte(opcodePath);
    if (opcode >= 0x45 && opcode <= 0xbf) continue;
    if (opcode >= 0xc0 && opcode <= 0xc4) { parsed.features.add("sign-extension"); continue; }
    switch (opcode) {
      case 0x00: case 0x01: case 0x0f: case 0x1a: case 0x1b: break;
      case 0x02: case 0x03: case 0x04:
        readBlockType(cursor, `${opcodePath}.blockType`);
        controls.push({ kind: opcode === 0x02 ? "block" : opcode === 0x03 ? "loop" : "if", elseSeen: false });
        if (controls.length > MAX_CONTROL_DEPTH) resource(opcodePath, `control depth exceeds ${MAX_CONTROL_DEPTH}`);
        break;
      case 0x05: {
        const top = controls.at(-1);
        if (top?.kind !== "if" || top.elseSeen) invalid(opcodePath, "else must occur once inside an if");
        top.elseSeen = true;
        break;
      }
      case 0x0b: controls.pop(); break;
      case 0x0c: case 0x0d: requireLabel(readU32(cursor, `${opcodePath}.label`), controls.length, opcodePath); break;
      case 0x0e: {
        const count = boundedVectorCount(cursor, MAX_CONTROL_DEPTH, `${opcodePath}.labels.count`);
        for (let index = 0; index < count; index += 1) requireLabel(readU32(cursor, `${opcodePath}.labels[${index}]`), controls.length, opcodePath);
        requireLabel(readU32(cursor, `${opcodePath}.defaultLabel`), controls.length, opcodePath);
        break;
      }
      case 0x10: requireIndex(readU32(cursor, `${opcodePath}.functionIndex`), totalFunctionCount(parsed), opcodePath, "function"); break;
      case 0x11: {
        requireIndex(readU32(cursor, `${opcodePath}.typeIndex`), parsed.types.length, opcodePath, "type");
        const tableIndexStart = cursor.offset;
        const tableIndex = readU32(cursor, `${opcodePath}.tableIndex`);
        if (cursor.offset - tableIndexStart !== 1) invalid(opcodePath, "padded call_indirect table index requires unlisted call-indirect-overlong support");
        requireIndex(tableIndex, totalTableCount(parsed), opcodePath, "table");
        break;
      }
      case 0x1c: invalid(opcodePath, "typed select requires an unlisted extension");
      case 0x20: case 0x21: case 0x22:
        requireIndex(readU32(cursor, `${opcodePath}.localIndex`), localCount, opcodePath, "local");
        break;
      case 0x23: case 0x24:
        requireIndex(readU32(cursor, `${opcodePath}.globalIndex`), totalGlobalCount(parsed), opcodePath, "global");
        break;
      case 0x25: case 0x26: invalid(opcodePath, "table instructions require an unlisted reference-types extension");
      case 0x28: case 0x29: case 0x2a: case 0x2b: case 0x2c: case 0x2d: case 0x2e: case 0x2f:
      case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35:
      case 0x36: case 0x37: case 0x38: case 0x39: case 0x3a: case 0x3b: case 0x3c: case 0x3d: case 0x3e:
        requireIndex(0, totalMemoryCount(parsed), opcodePath, "memory");
        parseMemarg(cursor, parsed, opcodePath);
        break;
      case 0x3f: case 0x40: {
        const memoryIndex = readU32(cursor, `${opcodePath}.memoryIndex`);
        if (memoryIndex !== 0) parsed.features.add("multi-memory");
        requireIndex(memoryIndex, totalMemoryCount(parsed), opcodePath, "memory");
        break;
      }
      case 0x41: readS32(cursor, `${opcodePath}.value`); break;
      case 0x42: readS64(cursor, `${opcodePath}.value`); break;
      case 0x43: cursor.readBytes(4, `${opcodePath}.value`); break;
      case 0x44: cursor.readBytes(8, `${opcodePath}.value`); break;
      case 0xd0: case 0xd1: case 0xd2: invalid(opcodePath, "reference instructions require an unlisted extension");
      case 0xfc: parseFcInstruction(cursor, parsed, opcodePath); break;
      case 0xfd: parsed.features.add("simd128"); invalid(opcodePath, "SIMD instructions are forbidden");
      case 0xfe: parsed.features.add("atomics"); parsed.features.add("threads"); invalid(opcodePath, "atomic/thread instructions are forbidden");
      case 0x06: case 0x07: case 0x08: case 0x09: case 0x0a: case 0x18: case 0x19:
        parsed.features.add("exception-handling"); invalid(opcodePath, "exception-handling instructions are forbidden");
      default: invalid(opcodePath, `unknown or unlisted opcode 0x${opcode.toString(16).padStart(2, "0")}`);
    }
  }
}

function parseFcInstruction(cursor: Cursor, parsed: ParsedModule, path: string): void {
  const subopcode = readU32(cursor, `${path}.subopcode`);
  if (subopcode <= 7) { parsed.features.add("nontrapping-fptoint"); return; }
  parsed.features.add("bulk-memory");
  switch (subopcode) {
    case 8:
      requireDataIndex(readU32(cursor, `${path}.dataIndex`), parsed, path);
      {
        const memoryIndex = readU32(cursor, `${path}.memoryIndex`);
        if (memoryIndex !== 0) parsed.features.add("multi-memory");
        requireIndex(memoryIndex, totalMemoryCount(parsed), path, "memory");
      }
      return;
    case 9: requireDataIndex(readU32(cursor, `${path}.dataIndex`), parsed, path); return;
    case 10: {
      parsed.features.add("bulk-memory-opt");
      const destination = readU32(cursor, `${path}.destinationMemoryIndex`);
      const source = readU32(cursor, `${path}.sourceMemoryIndex`);
      if (destination !== 0 || source !== 0) parsed.features.add("multi-memory");
      requireIndex(destination, totalMemoryCount(parsed), path, "memory");
      requireIndex(source, totalMemoryCount(parsed), path, "memory");
      return;
    }
    case 11: {
      parsed.features.add("bulk-memory-opt");
      const memoryIndex = readU32(cursor, `${path}.memoryIndex`);
      if (memoryIndex !== 0) parsed.features.add("multi-memory");
      requireIndex(memoryIndex, totalMemoryCount(parsed), path, "memory");
      return;
    }
    case 12:
      requireIndex(readU32(cursor, `${path}.elementIndex`), parsed.elementSegments.length, path, "element segment");
      requireIndex(readU32(cursor, `${path}.tableIndex`), totalTableCount(parsed), path, "table");
      return;
    case 13: requireIndex(readU32(cursor, `${path}.elementIndex`), parsed.elementSegments.length, path, "element segment"); return;
    case 14:
      requireIndex(readU32(cursor, `${path}.destinationTableIndex`), totalTableCount(parsed), path, "table");
      requireIndex(readU32(cursor, `${path}.sourceTableIndex`), totalTableCount(parsed), path, "table");
      return;
    case 15: case 16: case 17:
      invalid(`${path}.subopcode`, `0xfc subopcode ${subopcode} requires the unlisted reference-types extension`);
    default: invalid(`${path}.subopcode`, `unknown or unlisted 0xfc subopcode ${subopcode}`);
  }
}

function parseMemarg(cursor: Cursor, parsed: ParsedModule, path: string): void {
  const alignment = readU32(cursor, `${path}.alignment`);
  if ((alignment & 0x40) !== 0) {
    parsed.features.add("multi-memory");
    readU32(cursor, `${path}.memoryIndex`);
    readUnsignedLeb(cursor, 64, `${path}.offset`);
    invalid(path, "explicit-memory-index memarg requires the forbidden multi-memory extension");
  }
  readU32(cursor, `${path}.offset`);
}

function parseConstExpression(cursor: Cursor, parsed: ParsedModule, path: string): number {
  const opcode = cursor.readByte(`${path}.opcode`);
  switch (opcode) {
    case 0x23:
      requireIndex(
        readU32(cursor, `${path}.globalIndex`),
        parsed.imports.filter((entry) => entry.kind === "global").length,
        `${path}.globalIndex`,
        "imported global",
      );
      break;
    case 0x41: readS32(cursor, `${path}.value`); break;
    case 0x42: readS64(cursor, `${path}.value`); break;
    case 0x43: cursor.readBytes(4, `${path}.value`); break;
    case 0x44: cursor.readBytes(8, `${path}.value`); break;
    default: invalid(`${path}.opcode`, `unsupported constant-expression opcode 0x${opcode.toString(16)}`);
  }
  if (cursor.readByte(`${path}.end`) !== 0x0b) invalid(`${path}.end`, "constant expression is not exactly terminated");
  return opcode;
}

function parseReferenceConstExpression(cursor: Cursor, parsed: ParsedModule, path: string): void {
  const opcode = cursor.readByte(`${path}.opcode`);
  if (opcode === 0xd2) requireIndex(readU32(cursor, `${path}.functionIndex`), totalFunctionCount(parsed), path, "function");
  else if (opcode === 0xd0) readReferenceType(cursor, `${path}.referenceType`);
  else invalid(`${path}.opcode`, "element expression must be ref.func or ref.null");
  if (cursor.readByte(`${path}.end`) !== 0x0b) invalid(`${path}.end`, "element expression is not exactly terminated");
}

function finishModule(parsed: ParsedModule): void {
  const importedFunctions = parsed.imports.filter((entry) => entry.kind === "function").length;
  const importedTables = parsed.imports.filter((entry) => entry.kind === "table").length;
  const importedMemories = parsed.imports.filter((entry) => entry.kind === "memory").length;
  const importedGlobals = parsed.imports.filter((entry) => entry.kind === "global").length;
  const importedTags = parsed.imports.filter((entry) => entry.kind === "tag").length;
  if (parsed.definedFunctionTypeIndices.length !== parsed.codeBodies.length) invalid("$bytes.sections.code", "function and code section counts differ");
  if (parsed.dataCount !== null && parsed.dataCount !== parsed.dataSegments.length) invalid("$bytes.sections.data-count", "data-count does not equal the data segment count");
  if (importedTables + parsed.tables.length > 1) invalid("$bytes.sections.table", "module has more than one table");
  if (importedMemories + parsed.memories.length > 1) invalid("$bytes.sections.memory", "module has more than one memory");
  if (importedGlobals + parsed.globals.length > MAX_GLOBALS) resource("$bytes.sections.global", `global count exceeds ${MAX_GLOBALS}`);
  if (importedTags + parsed.tagCount > 0) invalid("$bytes.sections.tag", "runtime ABI forbids exception tags");
  for (let index = 0; index < parsed.exports.length; index += 1) {
    const entry = parsed.exports[index] as MutableExport;
    const maximum = entry.kind === "function" ? importedFunctions + parsed.definedFunctionTypeIndices.length
      : entry.kind === "table" ? importedTables + parsed.tables.length
      : entry.kind === "memory" ? importedMemories + parsed.memories.length
      : entry.kind === "global" ? importedGlobals + parsed.globals.length
      : importedTags + parsed.tagCount;
    requireIndex(entry.index, maximum, `$bytes.sections.export[${index}].index`, entry.kind);
  }
  if (parsed.startFunctionIndex !== null) requireIndex(parsed.startFunctionIndex, importedFunctions + parsed.definedFunctionTypeIndices.length, "$bytes.sections.start", "function");
  for (const segment of parsed.elementSegments) if (segment.tableIndex !== null) requireIndex(segment.tableIndex, importedTables + parsed.tables.length, "$bytes.sections.element.tableIndex", "table");
  for (const segment of parsed.dataSegments) if (segment.memoryIndex !== null) requireIndex(segment.memoryIndex, importedMemories + parsed.memories.length, "$bytes.sections.data.memoryIndex", "memory");

  const target = parsed.customSections.find((section) => section.name === "target_features")?.targetFeatures ?? [];
  for (const feature of parsed.features) {
    if (feature === "atomics" || feature === "exception-handling" || feature === "memory64" || feature === "multi-memory" || feature === "simd128" || feature === "threads") {
      invalid("$bytes.features", `forbidden feature ${feature} is present in decoded structure or opcodes`);
    }
  }
  for (const entry of target) {
    if (!isKnownFeature(entry.normalizedName)) invalid("$bytes.sections.custom.target_features", `unlisted target feature ${entry.wireName}`);
    if (entry.prefix === "+" && isForbiddenFeature(entry.normalizedName)) invalid("$bytes.sections.custom.target_features", `forbidden target feature ${entry.wireName} is enabled`);
    if (entry.prefix === "-" && parsed.features.has(entry.normalizedName as ExtensionFeature)) invalid("$bytes.sections.custom.target_features", `disabled target feature ${entry.wireName} is used`);
  }
}

function makeProjection(
  parsed: ParsedModule,
  customSections: CppCuteBrowserWasmCustomSectionProjection[],
): CppCuteBrowserWasmInspectionProjection {
  const importedFunctionTypes = parsed.imports.filter((entry) => entry.kind === "function").map((entry) => entry.typeIndex as number);
  const allTables = [
    ...parsed.imports.filter((entry) => entry.kind === "table").map((entry) => entry.table as CppCuteBrowserWasmTableProjection),
    ...parsed.tables,
  ];
  const allMemories = [
    ...parsed.imports.filter((entry) => entry.kind === "memory").map((entry) => entry.memory as CppCuteBrowserWasmMemoryProjection),
    ...parsed.memories,
  ];
  const allGlobals: CppCuteBrowserWasmGlobalProjection[] = [
    ...parsed.imports.filter((entry) => entry.kind === "global").map((entry) => ({
      ...(entry.global as CppCuteBrowserWasmGlobalTypeProjection),
      initializerOpcode: null,
    })),
    ...parsed.globals,
  ];
  const imports = parsed.imports.map((entry) => ({
    module: entry.module, name: entry.name, kind: entry.kind,
    ...(entry.typeIndex === undefined ? {} : { functionType: copyFunctionType(parsed.types[entry.typeIndex] as MutableFunctionType) }),
    ...(entry.table === undefined ? {} : { table: entry.table }),
    ...(entry.memory === undefined ? {} : { memory: entry.memory }),
    ...(entry.global === undefined ? {} : { global: entry.global }),
  }));
  const exports = parsed.exports.map((entry) => ({
    name: entry.name, kind: entry.kind, index: entry.index,
    ...(entry.kind === "function" ? { functionType: copyFunctionType(resolveFunctionType(parsed, importedFunctionTypes, entry.index)) } : {}),
    ...(entry.kind === "table" ? { table: { ...(allTables[entry.index] as CppCuteBrowserWasmTableProjection) } } : {}),
    ...(entry.kind === "memory" ? { memory: { ...(allMemories[entry.index] as CppCuteBrowserWasmMemoryProjection) } } : {}),
    ...(entry.kind === "global" ? { global: { ...(allGlobals[entry.index] as CppCuteBrowserWasmGlobalProjection) } } : {}),
  }));
  return freezeJson({
    sectionOrder: parsed.sectionOrder,
    types: parsed.types.map(copyFunctionType),
    imports,
    definedFunctionTypeIndices: parsed.definedFunctionTypeIndices,
    tables: parsed.tables,
    memories: parsed.memories,
    globals: parsed.globals,
    exports,
    startFunctionIndex: parsed.startFunctionIndex,
    elementSegments: parsed.elementSegments,
    codeBodies: parsed.codeBodies,
    dataCount: parsed.dataCount,
    dataSegments: parsed.dataSegments,
    tagCount: parsed.tagCount,
    customSections,
    staticallyUsedExtensions: [...parsed.features].sort(),
  });
}

function compareAgainstAbi(projection: CppCuteBrowserWasmInspectionProjection, rawBody: unknown): string[] {
  const body = rawBody as RuntimeAbiBody;
  const mismatches: string[] = [];
  const expectedImports: ExpectedFunction[] = body.hostImports.functions.map((entry) => ({
    module: body.hostImports.moduleName,
    name: entry.fieldName,
    parameters: [...entry.wasmParameters],
    results: [...entry.wasmResults],
  }));
  for (const generated of body.hostImports.generatedImportAllowlist.exactFunctions) {
    expectedImports.push({
      module: generated.moduleName,
      name: generated.fieldName,
      parameters: [...generated.wasmParameters],
      results: [...generated.wasmResults],
    });
  }
  const actualFunctionImports = projection.imports.filter((entry) => entry.kind === "function");
  const nonFunctionImports = projection.imports.filter((entry) => entry.kind !== "function");
  if (nonFunctionImports.length > 0) mismatches.push(`non-function imports are forbidden: ${nonFunctionImports.map(formatImport).join(",")}`);
  compareFunctionInventory(actualFunctionImports, expectedImports, "import", mismatches);

  const expectedExports: ExpectedFunction[] = body.cExports.map((entry) => ({
    module: "",
    name: entry.wasmExportName,
    parameters: [...entry.wasmParameters],
    results: [...entry.wasmResults],
  }));
  for (const support of body.wasm.supportExports.exactFunctionAllowlist) {
    expectedExports.push({ module: "", name: support.name, parameters: [...support.wasmParameters], results: [...support.wasmResults] });
  }
  const actualFunctionExports = projection.exports.filter((entry) => entry.kind === "function");
  compareFunctionInventory(actualFunctionExports, expectedExports, "export", mismatches);
  const expectedOtherExportNames = new Set<string>([
    body.wasm.memory.exportName,
    ...body.wasm.supportExports.exactGlobalAllowlist.map((entry) => entry.name),
    ...body.wasm.supportExports.exactTableAllowlist.map((entry) => entry.name),
  ]);
  const actualOtherExports = projection.exports.filter((entry) => entry.kind !== "function");
  if (actualOtherExports.some((entry) => !expectedOtherExportNames.has(entry.name))) mismatches.push("unlisted non-function export is present");
  const actualTableExportNames = projection.exports.filter((entry) => entry.kind === "table").map((entry) => entry.name).sort();
  const expectedTableExportNames = body.wasm.supportExports.exactTableAllowlist.map((entry) => entry.name).sort();
  if (!equalStrings(actualTableExportNames, expectedTableExportNames)) mismatches.push("table export inventory differs from the exact runtime ABI allowlist");
  const actualGlobalExportNames = projection.exports.filter((entry) => entry.kind === "global").map((entry) => entry.name).sort();
  const expectedGlobalExportNames = body.wasm.supportExports.exactGlobalAllowlist.map((entry) => entry.name).sort();
  if (!equalStrings(actualGlobalExportNames, expectedGlobalExportNames)) mismatches.push("global export inventory differs from the exact runtime ABI allowlist");

  if (projection.memories.length !== body.wasm.memory.count) mismatches.push(`memory count ${projection.memories.length} != ${body.wasm.memory.count}`);
  const memory = projection.memories[0];
  if (memory === undefined || memory.addressType !== "i32" || memory.shared ||
      memory.minimumPages !== body.wasm.memory.initialPages ||
      memory.maximumPages !== body.wasm.memory.maximumPages) mismatches.push("defined memory type/limits do not equal runtime ABI");
  const memoryExport = projection.exports.find((entry) => entry.kind === "memory" && entry.name === body.wasm.memory.exportName);
  if (memoryExport?.index !== 0) mismatches.push(`memory export ${body.wasm.memory.exportName} is missing or does not target memory 0`);
  if (projection.startFunctionIndex !== null) mismatches.push("start section is forbidden");
  if (projection.tagCount !== body.wasm.structuralPolicy.tags.exactCount) mismatches.push("tag count differs from runtime ABI");

  compareJsonProjection(projection.tables, body.wasm.structuralPolicy.tables.exactReviewedProjection, "table projection", mismatches);
  const exportedGlobals = projection.exports.filter((entry) => entry.kind === "global").map((entry) => ({
    name: entry.name,
    index: entry.index,
    global: entry.global,
  }));
  compareJsonProjection(exportedGlobals, body.wasm.structuralPolicy.globals.exactReviewedExports, "global export projection", mismatches);
  const customNames = projection.customSections.map((section) => section.name);
  const reviewedNames = body.wasm.structuralPolicy.customSections.exactReviewedNameAllowlist;
  for (const name of customNames) if (!reviewedNames.includes(name)) mismatches.push(`custom section ${name} is not independently allowlisted`);
  for (const name of reviewedNames) if (!customNames.includes(name)) mismatches.push(`reviewed custom section ${name} is missing`);
  const target = projection.customSections.find((section) => section.name === body.wasm.structuralPolicy.customSections.targetFeatures.sectionName);
  const requiredTarget = body.wasm.structuralPolicy.customSections.targetFeatures.requiredDeclarations;
  const targetMap = new Map((target?.targetFeatures ?? []).map((entry) => [entry.wireName, entry.prefix]));
  for (const required of requiredTarget) if (targetMap.get(required) !== "+") mismatches.push(`required target feature +${required} is missing`);
  for (const forbidden of body.wasm.structuralPolicy.customSections.targetFeatures.forbiddenDeclarations) if (targetMap.get(forbidden) === "+") mismatches.push(`forbidden target feature +${forbidden} is declared`);
  const rawProjection = target?.targetFeatures?.map((entry) => `${entry.prefix}${entry.wireName}`) ?? [];
  compareJsonProjection(rawProjection, body.wasm.structuralPolicy.customSections.targetFeatures.exactRawSectionProjection, "target_features raw projection", mismatches);
  const allowedTargetFeatures = new Set(body.wasm.featurePolicy.allowedExtensions);
  for (const entry of target?.targetFeatures ?? []) {
    if (entry.prefix === "+" && !allowedTargetFeatures.has(entry.normalizedName)) {
      mismatches.push(`target feature +${entry.wireName} is outside the runtime ABI extension allowlist`);
    }
  }
  for (const feature of projection.staticallyUsedExtensions) if (!body.wasm.featurePolicy.allowedExtensions.includes(feature)) mismatches.push(`decoded extension ${feature} is not allowed`);
  return [...new Set(mismatches)].sort();
}

interface RuntimeAbiBody {
  readonly wasm: {
    readonly memory: { readonly count: number; readonly exportName: string; readonly initialPages: number; readonly maximumPages: number };
    readonly supportExports: {
      readonly exactFunctionAllowlist: ReadonlyArray<{ readonly name: string; readonly wasmParameters: readonly ValueType[]; readonly wasmResults: readonly ValueType[] }>;
      readonly exactGlobalAllowlist: ReadonlyArray<{ readonly name: string }>;
      readonly exactTableAllowlist: ReadonlyArray<{ readonly name: string }>;
    };
    readonly structuralPolicy: {
      readonly tables: { readonly maximumCount: number; readonly exactReviewedProjection: readonly unknown[] };
      readonly globals: { readonly maximumCount: number; readonly exactReviewedExports: readonly unknown[] };
      readonly tags: { readonly exactCount: number };
      readonly customSections: {
        readonly maximumCount: number;
        readonly maximumSectionByteLength: number;
        readonly maximumTotalByteLength: number;
        readonly exactReviewedNameAllowlist: readonly string[];
        readonly targetFeatures: {
          readonly sectionName: string;
          readonly requiredDeclarations: readonly string[];
          readonly forbiddenDeclarations: readonly string[];
          readonly exactRawSectionProjection: readonly unknown[];
        };
      };
    };
    readonly featurePolicy: { readonly allowedExtensions: readonly string[] };
  };
  readonly cExports: ReadonlyArray<{ readonly wasmExportName: string; readonly wasmParameters: readonly ValueType[]; readonly wasmResults: readonly ValueType[] }>;
  readonly hostImports: {
    readonly moduleName: string;
    readonly functions: ReadonlyArray<{ readonly fieldName: string; readonly wasmParameters: readonly ValueType[]; readonly wasmResults: readonly ValueType[] }>;
    readonly generatedImportAllowlist: {
      readonly exactFunctions: ReadonlyArray<{ readonly moduleName: string; readonly fieldName: string; readonly wasmParameters: readonly ValueType[]; readonly wasmResults: readonly ValueType[] }>;
    };
  };
}

interface ExpectedFunction {
  readonly module: string;
  readonly name: string;
  readonly parameters: readonly ValueType[];
  readonly results: readonly ValueType[];
}

function compareFunctionInventory(
  actual: ReadonlyArray<CppCuteBrowserWasmImportProjection | CppCuteBrowserWasmExportProjection>,
  expected: readonly ExpectedFunction[],
  kind: "export" | "import",
  mismatches: string[],
): void {
  const actualKeys = actual.map((entry) => functionKey(
    "module" in entry ? entry.module : "",
    entry.name,
    entry.functionType,
  )).sort();
  const expectedKeys = expected.map((entry) => functionKey(entry.module, entry.name, { parameters: entry.parameters, results: entry.results })).sort();
  if (!equalStrings(actualKeys, expectedKeys)) mismatches.push(`${kind} function names/signatures differ from the exact runtime ABI inventory`);
}

function functionKey(module: string, name: string, type: CppCuteBrowserWasmFunctionTypeProjection | undefined): string {
  return JSON.stringify([
    module,
    name,
    type?.parameters ?? null,
    type?.results ?? null,
  ]);
}

function compareJsonProjection(actual: unknown, expected: unknown, label: string, mismatches: string[]): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) mismatches.push(`${label} differs from independently reviewed ABI projection`);
}

function readTableType(cursor: Cursor, path: string): CppCuteBrowserWasmTableProjection {
  const elementType = readReferenceType(cursor, `${path}.elementType`);
  const limits = readLimits(cursor, `${path}.limits`, false);
  return { elementType, minimum: limits.minimum, maximum: limits.maximum };
}

function readMemoryType(cursor: Cursor, parsed: ParsedModule, path: string): CppCuteBrowserWasmMemoryProjection {
  const flags = peekU32(cursor, `${path}.limits.flags`);
  if ((flags & 0x04) !== 0) parsed.features.add("memory64");
  if ((flags & 0x02) !== 0) { parsed.features.add("threads"); parsed.features.add("atomics"); }
  const limits = readLimits(cursor, `${path}.limits`, true);
  return { addressType: "i32", shared: false, minimumPages: limits.minimum, maximumPages: limits.maximum };
}

function readLimits(cursor: Cursor, path: string, memory: boolean): { readonly minimum: number; readonly maximum: number | null } {
  const flags = readU32(cursor, `${path}.flags`);
  if (flags !== 0 && flags !== 1) invalid(`${path}.flags`, `${memory ? "memory" : "table"} limits must be wasm32, unshared, and use only min or min/max`);
  const minimum = readU32(cursor, `${path}.minimum`);
  const maximum = flags === 1 ? readU32(cursor, `${path}.maximum`) : null;
  if (maximum !== null && maximum < minimum) invalid(`${path}.maximum`, "maximum is less than minimum");
  return { minimum, maximum };
}

function readGlobalType(cursor: Cursor, parsed: ParsedModule, path: string): CppCuteBrowserWasmGlobalTypeProjection {
  const valueType = readValueType(cursor, `${path}.valueType`);
  const mutability = cursor.readByte(`${path}.mutability`);
  if (mutability !== 0 && mutability !== 1) invalid(`${path}.mutability`, "global mutability must be 0 or 1");
  if (mutability === 1) parsed.features.add("mutable-globals");
  return { valueType, mutable: mutability === 1 };
}

function readTagType(cursor: Cursor, parsed: ParsedModule, path: string): void {
  if (cursor.readByte(`${path}.attribute`) !== 0) invalid(`${path}.attribute`, "unknown tag attribute");
  requireIndex(readU32(cursor, `${path}.typeIndex`), parsed.types.length, `${path}.typeIndex`, "type");
}

function readValueType(cursor: Cursor, path: string): ValueType {
  const byte = cursor.readByte(path);
  if (byte === 0x7f) return "i32";
  if (byte === 0x7e) return "i64";
  if (byte === 0x7d) return "f32";
  if (byte === 0x7c) return "f64";
  if (byte === 0x7b) invalid(path, "v128 requires forbidden SIMD");
  invalid(path, `unsupported value type 0x${byte.toString(16)}`);
}

function readReferenceType(cursor: Cursor, path: string): "funcref" {
  if (cursor.readByte(path) !== 0x70) invalid(path, "only MVP funcref tables are supported");
  return "funcref";
}

function readExternalKind(cursor: Cursor, path: string): ExternalKind {
  const byte = cursor.readByte(path);
  if (byte === 0) return "function";
  if (byte === 1) return "table";
  if (byte === 2) return "memory";
  if (byte === 3) return "global";
  if (byte === 4) return "tag";
  invalid(path, `unknown external kind ${byte}`);
}

function readBlockType(cursor: Cursor, path: string): void {
  const byte = cursor.peekByte(path);
  if (byte === 0x40 || byte === 0x7f || byte === 0x7e || byte === 0x7d || byte === 0x7c) { cursor.readByte(path); return; }
  readSignedLeb(cursor, 33, path);
  invalid(path, "type-index block signatures require the unlisted multi-value extension");
}

function readName(cursor: Cursor, path: string): string {
  const byteLength = boundedCount(readU32(cursor, `${path}.byteLength`), MAX_NAME_BYTES, `${path}.byteLength`);
  cursor.state.totalNameBytes += byteLength;
  if (cursor.state.totalNameBytes > MAX_TOTAL_NAME_BYTES) resource(path, `aggregate name bytes exceed ${MAX_TOTAL_NAME_BYTES}`);
  const bytes = cursor.readBytes(byteLength, path);
  let value: string;
  try { value = TEXT_DECODER.decode(bytes); } catch (cause) { invalid(path, "name is not fatal UTF-8", { cause }); }
  if (!equalBytes(TEXT_ENCODER.encode(value), bytes)) invalid(path, "name does not round-trip to its exact UTF-8 bytes");
  return value;
}

function readU32(cursor: Cursor, path: string): number {
  const value = readUnsignedLeb(cursor, 32, path);
  return Number(value);
}

function peekU32(cursor: Cursor, path: string): number {
  const saved = cursor.offset;
  const value = readU32(cursor, path);
  cursor.offset = saved;
  return value;
}

function readS32(cursor: Cursor, path: string): number {
  return Number(readSignedLeb(cursor, 32, path));
}

function readS64(cursor: Cursor, path: string): bigint {
  return readSignedLeb(cursor, 64, path);
}

function readUnsignedLeb(cursor: Cursor, bits: number, path: string): bigint {
  let value = 0n;
  const maximumBytes = Math.ceil(bits / 7);
  for (let index = 0; index < maximumBytes; index += 1) {
    const byte = cursor.readByte(path);
    value |= BigInt(byte & 0x7f) << BigInt(index * 7);
    if ((byte & 0x80) === 0) {
      if (value >= (1n << BigInt(bits))) invalid(path, `unsigned LEB exceeds u${bits}`);
      return value;
    }
  }
  invalid(path, `unterminated or overflowing unsigned u${bits} LEB`);
}

function readSignedLeb(cursor: Cursor, bits: number, path: string): bigint {
  let value = 0n;
  let shift = 0n;
  const maximumBytes = Math.ceil(bits / 7);
  for (let index = 0; index < maximumBytes; index += 1) {
    const byte = cursor.readByte(path);
    value |= BigInt(byte & 0x7f) << shift;
    shift += 7n;
    if ((byte & 0x80) === 0) {
      if ((byte & 0x40) !== 0) value |= -1n << shift;
      const minimum = -(1n << BigInt(bits - 1));
      const maximum = (1n << BigInt(bits - 1)) - 1n;
      if (value < minimum || value > maximum) invalid(path, `signed LEB exceeds s${bits}`);
      return value;
    }
  }
  invalid(path, `unterminated or overflowing signed s${bits} LEB`);
}

function boundedVectorCount(cursor: Cursor, maximum: number, path: string): number {
  return boundedCount(readU32(cursor, path), maximum, path);
}

function boundedCount(value: number, maximum: number, path: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > maximum) resource(path, `count or length ${value} exceeds ${maximum}`);
  return value;
}

function requireIndex(value: number, count: number, path: string, kind: string): void {
  if (value >= count) invalid(path, `${kind} index ${value} is outside count ${count}`);
}

function requireLabel(value: number, depth: number, path: string): void {
  if (value >= depth) invalid(path, `label index ${value} is outside control depth ${depth}`);
}

function resolveFunctionType(parsed: ParsedModule, importedTypeIndices: number[], functionIndex: number): MutableFunctionType {
  const typeIndex = functionIndex < importedTypeIndices.length
    ? importedTypeIndices[functionIndex]
    : parsed.definedFunctionTypeIndices[functionIndex - importedTypeIndices.length];
  if (typeIndex === undefined || parsed.types[typeIndex] === undefined) invalid("$bytes.functions", "function type index cannot be resolved");
  return parsed.types[typeIndex] as MutableFunctionType;
}

function totalFunctionCount(parsed: ParsedModule): number {
  return parsed.imports.filter((entry) => entry.kind === "function").length + parsed.definedFunctionTypeIndices.length;
}

function totalTableCount(parsed: ParsedModule): number {
  return parsed.imports.filter((entry) => entry.kind === "table").length + parsed.tables.length;
}

function totalMemoryCount(parsed: ParsedModule): number {
  return parsed.imports.filter((entry) => entry.kind === "memory").length + parsed.memories.length;
}

function totalGlobalCount(parsed: ParsedModule): number {
  return parsed.imports.filter((entry) => entry.kind === "global").length + parsed.globals.length;
}

function requireDataIndex(value: number, parsed: ParsedModule, path: string): void {
  if (parsed.dataCount === null) invalid(path, "memory.init/data.drop require a preceding data-count section");
  requireIndex(value, parsed.dataCount, path, "data segment");
}

function copyFunctionType(type: MutableFunctionType): CppCuteBrowserWasmFunctionTypeProjection {
  return { parameters: [...type.parameters], results: [...type.results] };
}

function normalizeTargetFeatureName(wireName: string): string {
  if (wireName === "sign-ext") return "sign-extension";
  if (wireName === "multimemory") return "multi-memory";
  return wireName;
}

function isKnownFeature(value: string): boolean {
  return value === "atomics" || value === "bulk-memory" || value === "exception-handling" ||
    value === "memory64" || value === "multi-memory" || value === "mutable-globals" ||
    value === "nontrapping-fptoint" || value === "sign-extension" || value === "simd128" || value === "threads" ||
    value === "bulk-memory-opt" || value === "call-indirect-overlong" || value === "extended-const" ||
    value === "multivalue" || value === "reference-types" || value === "relaxed-simd" || value === "tail-call";
}

function isForbiddenFeature(value: string): boolean {
  return value === "atomics" || value === "exception-handling" || value === "memory64" ||
    value === "multi-memory" || value === "simd128" || value === "threads";
}

function normalizeOptions(options: InspectCppCuteBrowserWasmOptions): NormalizedOptions {
  let descriptors: PropertyDescriptorMap;
  try {
    if (typeof options !== "object" || options === null || Object.getPrototypeOf(options) !== Object.prototype) invalid("$options", "options must be a plain object");
    const keys = Reflect.ownKeys(options);
    for (const key of keys) if (typeof key !== "string" || !["signal", "maxModuleByteLength", "maxOperations"].includes(key)) invalid("$options", `unknown option ${String(key)}`);
    descriptors = Object.getOwnPropertyDescriptors(options);
  } catch (cause) {
    if (cause instanceof CppCuteBrowserWasmInspectionError) throw cause;
    invalid("$options", "options cannot be inspected safely", { cause });
  }
  for (const [key, descriptor] of Object.entries(descriptors)) {
    if (!("value" in descriptor)) invalid(`$options.${key}`, "option accessors are forbidden");
  }
  const signal = descriptors.signal?.value as AbortSignal | undefined;
  if (signal !== undefined) readSignalAborted(signal);
  const moduleLimitValue = descriptors.maxModuleByteLength?.value as unknown;
  const operationLimitValue = descriptors.maxOperations?.value as unknown;
  const maxModuleByteLength = moduleLimitValue === undefined
    ? CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH
    : narrowLimit(moduleLimitValue as number, CPP_CUTE_BROWSER_WASM_MAX_BYTE_LENGTH, "$options.maxModuleByteLength");
  const maxOperations = operationLimitValue === undefined
    ? undefined
    : narrowLimit(operationLimitValue as number, CPP_CUTE_BROWSER_WASM_MAX_OPERATIONS, "$options.maxOperations");
  return { signal, maxModuleByteLength, maxOperations };
}

function resolveParserOptions(options: NormalizedOptions, byteLength: number): ParserOptions {
  // Every tick either consumes bytes or advances one bounded structural item.
  // Two operations per admitted byte is a conservative parser ceiling, while
  // the fixed floor keeps small structurally dense modules inexpensive to
  // configure. The caller may still narrow this budget explicitly.
  const maxOperations = options.maxOperations ?? Math.min(
    CPP_CUTE_BROWSER_WASM_MAX_OPERATIONS,
    Math.max(CPP_CUTE_BROWSER_WASM_BASE_OPERATIONS, byteLength * 2),
  );
  return { ...options, maxOperations };
}

function narrowLimit(value: number, ceiling: number, path: string): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > ceiling) invalid(path, `limit must be an integer in [1, ${ceiling}]`);
  return value;
}

function snapshotBytes(value: unknown, maximum: number): Uint8Array {
  try {
    const inspection = inspectUnsharedPlainUint8Array(value);
    if (inspection.byteLength > maximum) resource("$bytes", `Wasm bytes exceed ${maximum}`);
    return copyInspectedUnsharedUint8Array(value, inspection);
  } catch (cause) {
    if (cause instanceof CppCuteBrowserWasmInspectionError) throw cause;
    invalid("$bytes", "Wasm input must be one plain unshared Uint8Array snapshot", { cause });
  }
}

function unwrapAbi(runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest): RuntimeAbiBodyRecord {
  try {
    return unwrapPreparedCppCuteBrowserRuntimeAbiManifest(runtimeAbi) as unknown as RuntimeAbiBodyRecord;
  } catch (cause) {
    unverified("$runtimeAbi", "expected opaque prepared runtime-ABI authority", { cause });
  }
}

interface RuntimeAbiBodyRecord { readonly manifest: { readonly body: RuntimeAbiBody } }

function assertAbiInspectionCeilings(body: RuntimeAbiBody): void {
  const structural = body.wasm.structuralPolicy;
  if (structural.tables.maximumCount !== 1 || structural.globals.maximumCount !== MAX_GLOBALS ||
      structural.customSections.maximumCount !== MAX_CUSTOM_SECTIONS ||
      structural.customSections.maximumSectionByteLength !== MAX_CUSTOM_SECTION_BYTES ||
      structural.customSections.maximumTotalByteLength !== MAX_CUSTOM_TOTAL_BYTES) {
    unverified(
      "$runtimeAbi",
      "runtime ABI structural ceilings differ from the inspector's closed safety profile",
    );
  }
}

function validateWasmSemantics(bytes: Uint8Array): void {
  if (WEBASSEMBLY_VALIDATE === undefined) unverified("$bytes", "intrinsic WebAssembly.validate is unavailable");
  let valid: boolean;
  try {
    // This bounded native call is synchronous. Production must invoke the
    // complete inspector in its disposable verifier Worker so termination is
    // the hard cancellation boundary; this authority never grants Worker use.
    valid = WEBASSEMBLY_VALIDATE(bytes as Uint8Array<ArrayBuffer>);
  } catch (cause) {
    invalid("$bytes", "intrinsic WebAssembly validation failed", { cause });
  }
  if (!valid) invalid("$bytes", "decoded module fails complete WebAssembly type and index validation");
}

function tick(state: ParserState): void {
  state.operations += 1;
  if (state.operations > state.options.maxOperations) resource("$bytes", `inspection operations exceed ${state.options.maxOperations}`);
  if ((state.operations & 0x3ff) === 0) throwIfAborted(state.options.signal);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal !== undefined && readSignalAborted(signal)) fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-CANCELLED", "$options.signal", "raw-Wasm inspection was cancelled");
}

function readSignalAborted(signal: AbortSignal): boolean {
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined) invalid("$options.signal", "AbortSignal intrinsic is unavailable");
  try { return ABORT_SIGNAL_ABORTED_GETTER.call(signal) as boolean; }
  catch (cause) { invalid("$options.signal", "signal is not a genuine AbortSignal", { cause }); }
}

async function hashBytes(bytes: Uint8Array, path: string): Promise<string> {
  try { return await sha256Hex(bytes); }
  catch (cause) { fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-HASH-UNAVAILABLE", path, "SHA-256 is unavailable", { cause }); }
}

async function hashProjection(projection: CppCuteBrowserWasmInspectionProjection): Promise<string> {
  try {
    return await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.browser-wasm-observed-projection.v1",
      projection: projection as unknown as JsonValue,
    } as JsonObject, {
      limits: {
        maxArrayLength: CPP_CUTE_BROWSER_WASM_REPORT_MAX_ARRAY_LENGTH,
        maxDocumentBytes: CPP_CUTE_BROWSER_WASM_REPORT_MAX_BYTE_LENGTH,
        maxNodes: CPP_CUTE_BROWSER_WASM_REPORT_MAX_NODES,
      },
    });
  } catch (cause) {
    fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-HASH-UNAVAILABLE", "$.projection", "projection SHA-256 is unavailable", { cause });
  }
}

function freezeJson<T>(value: T): T {
  return deepFreezeJson(value as unknown as JsonValue) as unknown as T;
}

function cloneFrozen<T>(value: T): T {
  return freezeJson(JSON.parse(JSON.stringify(value)) as T);
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index += 1) if (left[index] !== right[index]) return false;
  return true;
}

function equalStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function formatImport(entry: CppCuteBrowserWasmImportProjection): string {
  return `${entry.module}.${entry.name}:${entry.kind}`;
}

function fail(code: CppCuteBrowserWasmInspectionErrorCode, path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteBrowserWasmInspectionError(code, path, message, options);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID", path, message, options);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-RESOURCE-LIMIT", path, message);
}

function unverified(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-UNVERIFIED", path, message, options);
}
