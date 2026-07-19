import {
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  bindCppCuteBrowserVfsMount,
  closeCppCuteBrowserVfsSession,
  createCppCuteBrowserVfsMountHostImports,
  discardCppCuteBrowserVfsMount,
  observeCppCuteBrowserVfsMount,
  type CppCuteBrowserVfsHostImports,
  type PreparedCppCuteBrowserVfsMount,
  type PreparedCppCuteBrowserVfsSession,
} from "./cpp_cute_browser_vfs_session.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE,
} from "./resources/cpp_cute_browser_runtime_abi_v1.js";

export const CPP_CUTE_BROWSER_EMSCRIPTEN_FACTORY_PROTOCOL =
  "browsergrad.compiler.cpp-cute.emscripten-factory-binding@1";

const RUNTIME_ABI = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body;
const EXPECTED_C_ABI_VERSION = RUNTIME_ABI.wasm.cAbiVersion;
const VFS_MODULE_NAME = RUNTIME_ABI.hostImports.moduleName;
const REQUIRED_FACADE_EXPORTS = Object.freeze([
  "_bg_cpp_cute_abi_version",
  "_bg_cpp_cute_alloc",
  "_bg_cpp_cute_allocator_metrics_pointer",
  "_bg_cpp_cute_compile",
  "_bg_cpp_cute_free",
  "_bg_cpp_cute_reset",
  "_bg_cpp_cute_result_length",
  "_bg_cpp_cute_result_pointer",
  "_bg_cpp_cute_status",
] as const);
const REQUIRED_WASM_EXPORTS = Object.freeze(
  REQUIRED_FACADE_EXPORTS.map((name) => name.slice(1)),
);
const REQUIRED_VFS_IMPORTS = Object.freeze(
  RUNTIME_ABI.hostImports.functions.map((entry) => entry.fieldName).sort(),
);
const REQUIRED_GENERATED_IMPORTS = Object.freeze(
  RUNTIME_ABI.hostImports.generatedImportAllowlist.exactFunctions.map((entry) =>
    Object.freeze({ module: entry.moduleName, name: entry.fieldName })
  ).sort(compareImport),
);
const REQUIRED_MODULE_IMPORTS = Object.freeze([
  ...REQUIRED_GENERATED_IMPORTS,
  ...REQUIRED_VFS_IMPORTS.map((name) => Object.freeze({ module: VFS_MODULE_NAME, name })),
].sort(compareImport));

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const MAX_FACTORY_LOG_UTF8_BYTE_LENGTH = 64 * 1024;
const NATIVE_OBJECT_PROTOTYPE = Object.prototype;
const NATIVE_ASYNC_FUNCTION_PROTOTYPE = Object.getPrototypeOf(async function () {
  // Captured only to authenticate the package-generated async factory shape.
});
const NATIVE_PROMISE_PROTOTYPE = Promise.prototype;
const NATIVE_PROMISE_THEN = Promise.prototype.then;
const NATIVE_WEBASSEMBLY_INSTANCE_PROTOTYPE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.Instance.prototype;
const NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.Memory.prototype;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_OBJECT_CREATE = Object.create;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_ARRAY_PUSH = Array.prototype.push;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const NATIVE_TEXT_ENCODE = TextEncoder.prototype.encode;
const NATIVE_TEXT_ENCODER = new TextEncoder();
const NATIVE_SUBTLE = globalThis.crypto?.subtle;
const NATIVE_SUBTLE_DIGEST = NATIVE_SUBTLE?.digest;
const NATIVE_WEBASSEMBLY_COMPILE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.compile;
const NATIVE_WEBASSEMBLY_INSTANTIATE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.instantiate;
const NATIVE_WEBASSEMBLY_MODULE_IMPORTS = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.Module.imports;
const NATIVE_AGGREGATE_ERROR = AggregateError;
const EMPTY_INSTANTIATE_RESULT = NATIVE_OBJECT_FREEZE(NATIVE_OBJECT_CREATE(null)) as
  Readonly<Record<string, never>>;

type RequiredFacadeExportName = typeof REQUIRED_FACADE_EXPORTS[number];

export interface CppCuteBrowserEmscriptenFactoryModuleArgument {
  readonly instantiateWasm: (
    imports: WebAssembly.Imports,
    receiveInstance: (instance: WebAssembly.Instance) => void,
  ) => Readonly<Record<string, never>>;
  readonly onAbort: (reason: unknown) => void;
  readonly print: (line: unknown) => void;
  readonly printErr: (line: unknown) => void;
}

export type CppCuteBrowserGeneratedEmscriptenFactory = (
  moduleArgument: CppCuteBrowserEmscriptenFactoryModuleArgument,
) => Promise<unknown>;

declare const preparedFactoryBrand: unique symbol;

/**
 * Worker-local ownership of one generated factory plus the exact compiled
 * host-verified module. It is not Worker execution or lowering evidence.
 */
export interface PreparedCppCuteBrowserEmscriptenFactory {
  readonly [preparedFactoryBrand]: true;
  readonly authority: "package-generated-factory-instantiation-only";
  readonly protocol: typeof CPP_CUTE_BROWSER_EMSCRIPTEN_FACTORY_PROTOCOL;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly cAbiVersion: number;
  readonly allocatorMetricsPointer: number;
  readonly generatedImportCount: number;
  readonly vfsImportCount: number;
  readonly networkAuthorityGranted: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface PrepareCppCuteBrowserEmscriptenFactoryInput {
  /** Must be the statically imported, package-pinned generated factory. */
  readonly factory: CppCuteBrowserGeneratedEmscriptenFactory;
  /** Already host-verified bytes transferred into the dedicated Worker. */
  readonly clangWasmBytes: Uint8Array;
  readonly expectedWasmSha256: string;
  readonly expectedWasmByteLength: number;
  readonly vfsMount: PreparedCppCuteBrowserVfsMount;
}

export interface CppCuteBrowserEmscriptenFactoryInspection {
  readonly state: "prepared" | "taken" | "discarded";
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly cAbiVersion: number;
  readonly allocatorMetricsPointer: number;
  readonly generatedImportCount: number;
  readonly vfsImportCount: number;
  readonly factoryInvoked: true;
  readonly factoryFacadeMatchedInstance: true;
  readonly moduleImportProjectionVerified: true;
  readonly networkAuthorityGranted: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface TakenCppCuteBrowserEmscriptenFactory {
  readonly instance: WebAssembly.Instance;
  readonly memory: WebAssembly.Memory;
  readonly vfsSession: PreparedCppCuteBrowserVfsSession;
  readonly moduleFacade: Readonly<Record<RequiredFacadeExportName, Function>>;
  readonly stdout: readonly string[];
  readonly stderr: readonly string[];
}

export type CppCuteBrowserEmscriptenFactoryErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-CAPABILITY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-STATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-UNVERIFIED";

export class CppCuteBrowserEmscriptenFactoryError extends Error {
  constructor(
    readonly code: CppCuteBrowserEmscriptenFactoryErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserEmscriptenFactoryError";
  }
}

interface ActiveFactoryBinding {
  readonly instance: WebAssembly.Instance;
  readonly memory: WebAssembly.Memory;
  readonly vfsSession: PreparedCppCuteBrowserVfsSession;
  readonly moduleFacade: Readonly<Record<RequiredFacadeExportName, Function>>;
  readonly stdout: readonly string[];
  readonly stderr: readonly string[];
}

interface StoredFactoryBinding {
  state: "prepared" | "taken" | "discarded";
  active: ActiveFactoryBinding | null;
  readonly inspection: Omit<CppCuteBrowserEmscriptenFactoryInspection, "state">;
}

const FACTORY_BINDINGS = new WeakMap<object, StoredFactoryBinding>();

export async function prepareCppCuteBrowserEmscriptenFactory(
  input: PrepareCppCuteBrowserEmscriptenFactoryInput,
): Promise<PreparedCppCuteBrowserEmscriptenFactory> {
  const values = exactDataRecord(input, "$.input", [
    "factory", "clangWasmBytes", "expectedWasmSha256", "expectedWasmByteLength", "vfsMount",
  ]);
  const factory = exactFactory(values["factory"], "$.input.factory");
  const clangWasmBytes = exactBytes(values["clangWasmBytes"], "$.input.clangWasmBytes");
  const expectedWasmSha256 = sha256(values["expectedWasmSha256"], "$.input.expectedWasmSha256");
  const expectedWasmByteLength = boundedPositiveInteger(
    values["expectedWasmByteLength"],
    clangWasmBytes.byteLength,
    "$.input.expectedWasmByteLength",
  );
  const vfsMount = values["vfsMount"] as PreparedCppCuteBrowserVfsMount;
  requireConstructionIntrinsics();

  let vfsSession: PreparedCppCuteBrowserVfsSession | undefined;
  let pendingInstantiation: Promise<WebAssembly.Instance> | undefined;
  let factoryHookOpen = true;
  try {
    const digestPromise = nativeSha256Hex(clangWasmBytes);
    const compilePromise = NATIVE_REFLECT_APPLY(
      NATIVE_WEBASSEMBLY_COMPILE!,
      WebAssembly,
      [clangWasmBytes],
    ) as Promise<WebAssembly.Module>;
    const [actualWasmSha256, compiledModule] = await Promise.all([
      digestPromise,
      compilePromise,
    ]);
    if (actualWasmSha256 !== expectedWasmSha256) {
      mismatch("$.input.clangWasmBytes", "Wasm bytes differ from the host-verified digest");
    }
    verifyModuleImportProjection(compiledModule);

    const vfsImports = createCppCuteBrowserVfsMountHostImports(vfsMount);
    const stdout: string[] = [];
    const stderr: string[] = [];
    const stdoutSink = boundedLogSink(stdout, "$.factory.stdout");
    const stderrSink = boundedLogSink(stderr, "$.factory.stderr");
    let abortReason: unknown;
    let hookCalls = 0;
    let callbackCalls = 0;
    let instance: WebAssembly.Instance | undefined;
    let memory: WebAssembly.Memory | undefined;
    let instantiationPromise: Promise<WebAssembly.Instance> | undefined;

    const moduleArgument: CppCuteBrowserEmscriptenFactoryModuleArgument = {
      instantiateWasm: (factoryImports, receiveInstance) => {
        if (!factoryHookOpen) {
          state("$.factory.instantiateWasm", "generated factory used the instantiation hook after terminalization");
        }
        hookCalls += 1;
        if (hookCalls !== 1) {
          state("$.factory.instantiateWasm", "generated factory invoked instantiateWasm more than once");
        }
        if (typeof receiveInstance !== "function") {
          invalid("$.factory.instantiateWasm.receiveInstance", "factory callback must be a function");
        }
        const imports = projectFactoryImports(factoryImports, vfsImports);
        instantiationPromise = Promise.resolve(NATIVE_REFLECT_APPLY(
          NATIVE_WEBASSEMBLY_INSTANTIATE!,
          WebAssembly,
          [compiledModule, imports],
        ) as Promise<WebAssembly.Instance>).then((created) => {
          if (NATIVE_GET_PROTOTYPE_OF(created) !== NATIVE_WEBASSEMBLY_INSTANCE_PROTOTYPE) {
            invalid("$.factory.instance", "instantiateWasm did not produce an exact WebAssembly.Instance");
          }
          const observedMemory = exactExportedMemory(created);
          vfsSession = bindCppCuteBrowserVfsMount({ mount: vfsMount, memory: observedMemory });
          instance = created;
          memory = observedMemory;
          callbackCalls += 1;
          if (callbackCalls !== 1) {
            state("$.factory.instantiateWasm.receiveInstance", "factory callback was already delivered");
          }
          NATIVE_REFLECT_APPLY(receiveInstance, undefined, [created]);
          return created;
        });
        pendingInstantiation = instantiationPromise;
        return EMPTY_INSTANTIATE_RESULT;
      },
      onAbort: (reason) => {
        abortReason = reason;
      },
      print: stdoutSink,
      printErr: stderrSink,
    };

    let factoryPromise: Promise<unknown>;
    try {
      factoryPromise = NATIVE_REFLECT_APPLY(factory, undefined, [moduleArgument]) as Promise<unknown>;
    } catch (cause) {
      invalid("$.factory", "generated factory threw before returning its native Promise", { cause });
    }
    if (NATIVE_GET_PROTOTYPE_OF(factoryPromise) !== NATIVE_PROMISE_PROTOTYPE) {
      invalid("$.factory", "generated factory must return one exact native Promise");
    }
    if (hookCalls !== 1 || instantiationPromise === undefined) {
      if (hookCalls === 1) {
        await factoryPromise;
      } else {
        NATIVE_REFLECT_APPLY(NATIVE_PROMISE_THEN, factoryPromise, [
          undefined,
          () => undefined,
        ]);
      }
      mismatch("$.factory.instantiateWasm", "generated factory did not synchronously use the host instantiation hook");
    }

    const failOnInstantiation = instantiationPromise.then(
      () => new Promise<never>(() => undefined),
      (cause) => Promise.reject(cause),
    );
    let moduleFacadeValue: unknown;
    try {
      moduleFacadeValue = await Promise.race([factoryPromise, failOnInstantiation]);
    } catch (cause) {
      if (abortReason !== undefined) {
        invalid("$.factory.abort", "generated factory aborted during initialization", { cause });
      }
      invalid("$.factory", "generated factory initialization failed", { cause });
    }
    await instantiationPromise;
    if (instance === undefined || memory === undefined || vfsSession === undefined || callbackCalls !== 1) {
      mismatch("$.factory", "generated factory resolved without one bound Wasm instance");
    }
    const moduleFacade = verifyFactoryFacade(moduleFacadeValue, instance);
    const cAbiVersion = callI32Export(
      moduleFacade._bg_cpp_cute_abi_version,
      "$.factory.module._bg_cpp_cute_abi_version",
    );
    if (cAbiVersion !== EXPECTED_C_ABI_VERSION) {
      mismatch("$.factory.module._bg_cpp_cute_abi_version", "executed C ABI version differs from the runtime manifest");
    }
    const allocatorMetricsPointer = callI32Export(
      moduleFacade._bg_cpp_cute_allocator_metrics_pointer,
      "$.factory.module._bg_cpp_cute_allocator_metrics_pointer",
    );
    if (allocatorMetricsPointer <= 0 || allocatorMetricsPointer % 8 !== 0) {
      mismatch("$.factory.module._bg_cpp_cute_allocator_metrics_pointer", "allocator metrics pointer must be nonzero and 8-byte aligned");
    }
    factoryHookOpen = false;

    const prepared = NATIVE_OBJECT_FREEZE({
      authority: "package-generated-factory-instantiation-only",
      protocol: CPP_CUTE_BROWSER_EMSCRIPTEN_FACTORY_PROTOCOL,
      wasmSha256: actualWasmSha256,
      wasmByteLength: expectedWasmByteLength,
      cAbiVersion,
      allocatorMetricsPointer,
      generatedImportCount: REQUIRED_GENERATED_IMPORTS.length,
      vfsImportCount: REQUIRED_VFS_IMPORTS.length,
      networkAuthorityGranted: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    }) as PreparedCppCuteBrowserEmscriptenFactory;
    weakMapSet(FACTORY_BINDINGS, prepared, {
      state: "prepared",
      active: NATIVE_OBJECT_FREEZE({
        instance,
        memory,
        vfsSession,
        moduleFacade,
        stdout: NATIVE_OBJECT_FREEZE([...stdout]),
        stderr: NATIVE_OBJECT_FREEZE([...stderr]),
      }),
      inspection: NATIVE_OBJECT_FREEZE({
        wasmSha256: actualWasmSha256,
        wasmByteLength: expectedWasmByteLength,
        cAbiVersion,
        allocatorMetricsPointer,
        generatedImportCount: REQUIRED_GENERATED_IMPORTS.length,
        vfsImportCount: REQUIRED_VFS_IMPORTS.length,
        factoryInvoked: true,
        factoryFacadeMatchedInstance: true,
        moduleImportProjectionVerified: true,
        networkAuthorityGranted: false,
        workerExecutionObserved: false,
        loweringAuthorityMinted: false,
      }),
    });
    return prepared;
  } catch (cause) {
    factoryHookOpen = false;
    if (pendingInstantiation !== undefined) {
      try {
        await pendingInstantiation;
      } catch {
        // The primary failure remains authoritative; cleanup follows below.
      }
    }
    cleanupFailedPreparation(vfsMount, vfsSession, cause);
  }
}

export function inspectCppCuteBrowserEmscriptenFactory(
  prepared: PreparedCppCuteBrowserEmscriptenFactory,
): CppCuteBrowserEmscriptenFactoryInspection {
  const stored = storedFactory(prepared);
  return NATIVE_OBJECT_FREEZE({ state: stored.state, ...stored.inspection });
}

export function takeCppCuteBrowserEmscriptenFactory(
  prepared: PreparedCppCuteBrowserEmscriptenFactory,
): TakenCppCuteBrowserEmscriptenFactory {
  const stored = storedFactory(prepared);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.factory", "only a prepared generated factory may be taken");
  }
  const active = stored.active;
  stored.state = "taken";
  stored.active = null;
  return active;
}

export function discardCppCuteBrowserEmscriptenFactory(
  prepared: PreparedCppCuteBrowserEmscriptenFactory,
): void {
  const stored = storedFactory(prepared);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.factory", "only a prepared generated factory may be discarded");
  }
  const active = stored.active;
  stored.state = "discarded";
  stored.active = null;
  try {
    closeCppCuteBrowserVfsSession(active.vfsSession, "failed");
  } catch (cause) {
    cleanup("$.factory.vfsSession", "failed to close the discarded factory VFS session", cause);
  }
}

function verifyModuleImportProjection(module: WebAssembly.Module): void {
  const imports = NATIVE_REFLECT_APPLY(
    NATIVE_WEBASSEMBLY_MODULE_IMPORTS!,
    WebAssembly.Module,
    [module],
  ) as WebAssembly.ModuleImportDescriptor[];
  const projected = imports.map((entry, index) => {
    if (entry.kind !== "function") {
      mismatch(`$.wasm.imports[${index}]`, "factory runtime admits function imports only");
    }
    return NATIVE_OBJECT_FREEZE({ module: entry.module, name: entry.name });
  }).sort(compareImport);
  if (projected.length !== REQUIRED_MODULE_IMPORTS.length) {
    mismatch("$.wasm.imports", "Wasm import count differs from the runtime ABI");
  }
  for (let index = 0; index < REQUIRED_MODULE_IMPORTS.length; index += 1) {
    const actual = projected[index];
    const expected = REQUIRED_MODULE_IMPORTS[index];
    if (actual?.module !== expected?.module || actual?.name !== expected?.name) {
      mismatch(`$.wasm.imports[${index}]`, "Wasm import identity differs from the runtime ABI");
    }
  }
}

function projectFactoryImports(
  value: unknown,
  vfsImports: CppCuteBrowserVfsHostImports,
): WebAssembly.Imports {
  const root = exactDataRecord(value, "$.factory.imports", ["env", "wasi_snapshot_preview1"]);
  const env = projectGeneratedModule(root["env"], "env");
  const wasi = projectGeneratedModule(root["wasi_snapshot_preview1"], "wasi_snapshot_preview1");
  const vfs = NATIVE_OBJECT_CREATE(null) as Record<string, Function>;
  const descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(vfsImports);
  for (const name of REQUIRED_VFS_IMPORTS) {
    const candidate = descriptors[name];
    if (candidate === undefined || !("value" in candidate) || typeof candidate.value !== "function") {
      unverified(`$.vfsImports.${name}`, "stable VFS import is unavailable");
    }
    vfs[name] = candidate.value as Function;
  }
  const imports = NATIVE_OBJECT_CREATE(null) as WebAssembly.Imports;
  imports["env"] = env;
  imports["wasi_snapshot_preview1"] = wasi;
  imports[VFS_MODULE_NAME] = NATIVE_OBJECT_FREEZE(vfs);
  return NATIVE_OBJECT_FREEZE(imports);
}

function projectGeneratedModule(value: unknown, moduleName: string): WebAssembly.ModuleImports {
  const object = exactObject(value, `$.factory.imports.${moduleName}`);
  const descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(object);
  const projected = NATIVE_OBJECT_CREATE(null) as Record<string, Function>;
  for (const expected of REQUIRED_GENERATED_IMPORTS) {
    if (expected.module !== moduleName) continue;
    const candidate = descriptors[expected.name];
    if (candidate === undefined || !("value" in candidate) || typeof candidate.value !== "function") {
      mismatch(
        `$.factory.imports.${moduleName}.${expected.name}`,
        "generated factory did not provide one required import function",
      );
    }
    projected[expected.name] = candidate.value as Function;
  }
  return NATIVE_OBJECT_FREEZE(projected);
}

function exactExportedMemory(instance: WebAssembly.Instance): WebAssembly.Memory {
  const descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(instance.exports);
  const candidate = descriptors["memory"];
  if (candidate === undefined || !("value" in candidate) ||
      NATIVE_GET_PROTOTYPE_OF(candidate.value) !== NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE) {
    mismatch("$.factory.instance.exports.memory", "Wasm instance must export one exact WebAssembly.Memory");
  }
  return candidate.value as WebAssembly.Memory;
}

function verifyFactoryFacade(
  value: unknown,
  instance: WebAssembly.Instance,
): Readonly<Record<RequiredFacadeExportName, Function>> {
  const object = exactObject(value, "$.factory.module");
  const facadeDescriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(object);
  const exportDescriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(instance.exports);
  const facade = NATIVE_OBJECT_CREATE(null) as Record<RequiredFacadeExportName, Function>;
  for (let index = 0; index < REQUIRED_FACADE_EXPORTS.length; index += 1) {
    const facadeName = REQUIRED_FACADE_EXPORTS[index]!;
    const exportName = REQUIRED_WASM_EXPORTS[index]!;
    const candidate = facadeDescriptors[facadeName];
    const exported = exportDescriptors[exportName];
    if (candidate === undefined || !("value" in candidate) || typeof candidate.value !== "function" ||
        exported === undefined || !("value" in exported) || candidate.value !== exported.value) {
      mismatch(`$.factory.module.${facadeName}`, "factory facade does not reference the exact Wasm export");
    }
    facade[facadeName] = candidate.value as Function;
  }
  return NATIVE_OBJECT_FREEZE(facade);
}

function boundedLogSink(target: string[], path: string): (value: unknown) => void {
  let byteLength = 0;
  return (value) => {
    if (typeof value !== "string") invalid(path, "factory log entry must be a string");
    const bytes = NATIVE_REFLECT_APPLY(
      NATIVE_TEXT_ENCODE,
      NATIVE_TEXT_ENCODER,
      [value],
    ) as Uint8Array;
    if (byteLength + bytes.byteLength > MAX_FACTORY_LOG_UTF8_BYTE_LENGTH) {
      resource(path, "factory log exceeded the bounded UTF-8 byte ceiling");
    }
    byteLength += bytes.byteLength;
    NATIVE_REFLECT_APPLY(NATIVE_ARRAY_PUSH, target, [value]);
  };
}

function callI32Export(operation: Function, path: string): number {
  let value: unknown;
  try {
    value = NATIVE_REFLECT_APPLY(operation, undefined, []);
  } catch (cause) {
    mismatch(path, "Wasm export trapped during factory verification", { cause });
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
    mismatch(path, "Wasm export did not return one unsigned i32 value");
  }
  return value;
}

function cleanupFailedPreparation(
  mount: PreparedCppCuteBrowserVfsMount,
  session: PreparedCppCuteBrowserVfsSession | undefined,
  primaryCause: unknown,
): never {
  const cleanupCauses: unknown[] = [];
  try {
    if (session !== undefined) {
      closeCppCuteBrowserVfsSession(session, "failed");
    } else if (observeCppCuteBrowserVfsMount(mount).state === "prepared") {
      discardCppCuteBrowserVfsMount(mount);
    }
  } catch (cause) {
    cleanupCauses.push(cause);
  }
  if (cleanupCauses.length !== 0) {
    cleanup(
      "$.factory.cleanup",
      "factory preparation failed and VFS authority cleanup also failed",
      new NATIVE_AGGREGATE_ERROR([primaryCause, ...cleanupCauses], "factory preparation cleanup failures"),
    );
  }
  throw primaryCause;
}

function storedFactory(
  prepared: PreparedCppCuteBrowserEmscriptenFactory,
): StoredFactoryBinding {
  if ((typeof prepared !== "object" && typeof prepared !== "function") || prepared === null) {
    unverified("$.factory", "generated factory authority is not authentic");
  }
  const stored = NATIVE_REFLECT_APPLY(
    NATIVE_WEAK_MAP_GET,
    FACTORY_BINDINGS,
    [prepared],
  ) as StoredFactoryBinding | undefined;
  if (stored === undefined) unverified("$.factory", "generated factory authority is not authentic");
  return stored;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function exactFactory(value: unknown, path: string): CppCuteBrowserGeneratedEmscriptenFactory {
  if (typeof value !== "function" ||
      NATIVE_GET_PROTOTYPE_OF(value) !== NATIVE_ASYNC_FUNCTION_PROTOTYPE) {
    invalid(path, "expected one exact package-generated factory function");
  }
  return value as CppCuteBrowserGeneratedEmscriptenFactory;
}

function exactBytes(value: unknown, path: string): Uint8Array {
  try {
    inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid(path, "expected one exact unshared Uint8Array", { cause });
  }
  return value as Uint8Array;
}

function exactDataRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
): Record<string, unknown> {
  const object = exactObject(value, path);
  const descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(object);
  const ownKeys = NATIVE_REFLECT_OWN_KEYS(descriptors);
  if (ownKeys.length !== keys.length || ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(path, `expected exact keys ${keys.join(", ")}`);
  }
  const result: Record<string, unknown> = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) invalid(`${path}.${key}`, "expected data property");
    result[key] = descriptor.value;
  }
  return result;
}

function exactObject(value: unknown, path: string): object & Record<string, unknown> {
  if (typeof value !== "object" || value === null) invalid(path, "expected object");
  let prototype: unknown;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(value);
  } catch (cause) {
    invalid(path, "object prototype is not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_OBJECT_PROTOTYPE && prototype !== null) {
    invalid(path, "expected plain or null-prototype object");
  }
  return value as object & Record<string, unknown>;
}

function sha256(value: unknown, path: string): string {
  if (typeof value !== "string" || !SHA256_HEX.test(value)) invalid(path, "expected lowercase SHA-256");
  return value;
}

function boundedPositiveInteger(value: unknown, expected: number, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    invalid(path, "expected positive safe integer");
  }
  if (value !== expected) mismatch(path, "declared Wasm byte length differs from transferred bytes");
  return value;
}

async function nativeSha256Hex(bytes: Uint8Array): Promise<string> {
  if (NATIVE_SUBTLE_DIGEST === undefined) {
    capability("$.crypto", "captured native SubtleCrypto SHA-256 is unavailable");
  }
  let digest: ArrayBuffer;
  try {
    digest = await NATIVE_REFLECT_APPLY(
      NATIVE_SUBTLE_DIGEST,
      NATIVE_SUBTLE,
      ["SHA-256", bytes],
    ) as ArrayBuffer;
  } catch (cause) {
    invalid("$.input.clangWasmBytes", "failed to hash transferred Wasm bytes", { cause });
  }
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function requireConstructionIntrinsics(): void {
  if (NATIVE_WEBASSEMBLY_COMPILE === undefined || NATIVE_WEBASSEMBLY_INSTANTIATE === undefined ||
      NATIVE_WEBASSEMBLY_MODULE_IMPORTS === undefined ||
      NATIVE_WEBASSEMBLY_INSTANCE_PROTOTYPE === undefined ||
      NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE === undefined) {
    capability("$.webAssembly", "captured native WebAssembly construction intrinsics are unavailable");
  }
}

function compareImport(
  left: { readonly module: string; readonly name: string },
  right: { readonly module: string; readonly name: string },
): number {
  return left.module === right.module
    ? left.name.localeCompare(right.name)
    : left.module.localeCompare(right.module);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-INVALID", path, message, options);
}

function mismatch(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-MISMATCH", path, message, options);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-RESOURCE-LIMIT", path, message);
}

function state(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-STATE", path, message);
}

function capability(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-CAPABILITY", path, message);
}

function cleanup(path: string, message: string, cause: unknown): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-CLEANUP", path, message, { cause });
}

function unverified(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-UNVERIFIED", path, message);
}

function fail(
  code: CppCuteBrowserEmscriptenFactoryErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserEmscriptenFactoryError(code, path, message, options);
}
