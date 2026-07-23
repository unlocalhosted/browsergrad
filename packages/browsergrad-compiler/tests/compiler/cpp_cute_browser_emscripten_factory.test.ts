import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { beforeEach, describe, expect, it, vi } from "vitest";

interface MockMountState {
  state: "prepared" | "bound" | "discarded";
  readonly session: object;
  createImportsCalls: number;
  bindCalls: number;
  discardCalls: number;
  closeCalls: number;
}

const vfsAuthorities = vi.hoisted(() => ({
  mounts: new WeakMap<object, MockMountState>(),
  sessions: new WeakMap<object, MockMountState>(),
  imports: Object.freeze({
    bg_vfs_status: () => 9,
    bg_vfs_open: () => 9,
    bg_vfs_read: () => 9,
    bg_vfs_close: () => 9,
    bg_vfs_directory_count: () => 9,
    bg_vfs_directory_entry: () => 9,
  }),
}));

vi.mock("../../src/cpp_cute_browser_vfs_session.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_vfs_session.js")>();
  const mount = (value: object): MockMountState => {
    const stored = vfsAuthorities.mounts.get(value);
    if (stored === undefined) {
      throw new actual.CppCuteBrowserVfsSessionError(
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED",
        "$.mount",
        "unregistered test mount",
      );
    }
    return stored;
  };
  const session = (value: object): MockMountState => {
    const stored = vfsAuthorities.sessions.get(value);
    if (stored === undefined) {
      throw new actual.CppCuteBrowserVfsSessionError(
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED",
        "$.session",
        "unregistered test session",
      );
    }
    return stored;
  };
  return {
    ...actual,
    createCppCuteBrowserVfsMountHostImports: (value: object) => {
      const stored = mount(value);
      if (stored.state !== "prepared") throw new Error("test mount is terminal");
      stored.createImportsCalls += 1;
      return vfsAuthorities.imports;
    },
    bindCppCuteBrowserVfsMount: ({ mount: value }: { readonly mount: object }) => {
      const stored = mount(value);
      if (stored.state !== "prepared") throw new Error("test mount is not bindable");
      stored.bindCalls += 1;
      stored.state = "bound";
      return stored.session;
    },
    observeCppCuteBrowserVfsMount: (value: object) => ({ state: mount(value).state }),
    discardCppCuteBrowserVfsMount: (value: object) => {
      const stored = mount(value);
      if (stored.state !== "prepared") throw new Error("test mount is not discardable");
      stored.discardCalls += 1;
      stored.state = "discarded";
    },
    closeCppCuteBrowserVfsSession: (value: object) => {
      const stored = session(value);
      if (stored.state !== "bound") throw new Error("test session is not closable");
      stored.closeCalls += 1;
      stored.state = "discarded";
      return Object.freeze({});
    },
  };
});

import {
  CppCuteBrowserEmscriptenFactoryError,
  discardCppCuteBrowserEmscriptenFactory,
  inspectCppCuteBrowserEmscriptenFactory,
  prepareCppCuteBrowserEmscriptenFactory,
  takeCppCuteBrowserEmscriptenFactory,
  type CppCuteBrowserEmscriptenFactoryModuleArgument,
  type CppCuteBrowserGeneratedEmscriptenFactory,
} from "../../src/cpp_cute_browser_emscripten_factory.js";
import type { PreparedCppCuteBrowserVfsMount } from "../../src/cpp_cute_browser_vfs_session.js";
import { CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE } from "../../src/resources/cpp_cute_browser_runtime_abi_v1.js";

type WasmValueType = "f32" | "f64" | "i32" | "i64";
type Bytes = readonly number[];

const RUNTIME_ABI = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body;
const WASM_VALUE_TYPE = Object.freeze({ f32: 0x7d, f64: 0x7c, i32: 0x7f, i64: 0x7e });
const HEADER = Object.freeze([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
let nextMockOrdinal = 1;

interface FactoryOptions {
  readonly omitGeneratedImport?: string;
  readonly extraImportModule?: boolean;
  readonly mismatchFacade?: boolean;
  readonly logLine?: string;
}

interface FactoryFixture {
  readonly bytes: Uint8Array;
  readonly factory: CppCuteBrowserGeneratedEmscriptenFactory;
  readonly calls: { count: number };
  readonly emitStdout: (line: string) => void;
  readonly emitStderr: (line: string) => void;
}

interface MountFixture {
  readonly mount: PreparedCppCuteBrowserVfsMount;
  readonly state: MockMountState;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("package-generated Emscripten factory binding", () => {
  it("instantiates only the host-verified module through exact generated and VFS imports", async () => {
    const fixture = factoryFixture({ logLine: "factory initialized" });
    const mount = mountFixture();
    const prepared = await prepareCppCuteBrowserEmscriptenFactory({
      factory: fixture.factory,
      clangWasmBytes: fixture.bytes,
      expectedWasmSha256: await sha256Hex(fixture.bytes),
      expectedWasmByteLength: fixture.bytes.byteLength,
      vfsMount: mount.mount,
    });

    expect(inspectCppCuteBrowserEmscriptenFactory(prepared)).toMatchObject({
      state: "prepared",
      cAbiVersion: 65_540,
      allocatorMetricsPointer: 8,
      frontendWorkMetricsPointer: 16,
      generatedImportCount: 66,
      vfsImportCount: 6,
      factoryInvoked: true,
      factoryFacadeMatchedInstance: true,
      moduleImportProjectionVerified: true,
      networkAuthorityGranted: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(fixture.calls.count).toBe(1);
    expect(mount.state).toMatchObject({
      state: "bound",
      createImportsCalls: 1,
      bindCalls: 1,
      discardCalls: 0,
      closeCalls: 0,
    });

    const taken = takeCppCuteBrowserEmscriptenFactory(prepared);
    expect(taken.memory).toBe(taken.instance.exports.memory);
    expect(taken.vfsSession).toBe(mount.state.session);
    expect(taken.snapshotStdout()).toEqual(["factory initialized"]);
    expect(taken.snapshotStderr()).toEqual([]);
    fixture.emitStderr("native compile diagnostic");
    expect(taken.snapshotStderr()).toEqual(["native compile diagnostic"]);
    expect(Object.isFrozen(taken.snapshotStderr())).toBe(true);
    expect(taken.moduleFacade._bg_cpp_cute_compile).toBe(
      taken.instance.exports.bg_cpp_cute_compile,
    );
    expect(inspectCppCuteBrowserEmscriptenFactory(prepared).state).toBe("taken");
    expectSyncCode(
      () => takeCppCuteBrowserEmscriptenFactory(prepared),
      "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-STATE",
    );
  });

  it("rejects hash and length drift before invoking the generated factory", async () => {
    const fixture = factoryFixture();
    for (const input of [
      { hash: "0".repeat(64), length: fixture.bytes.byteLength },
      { hash: await sha256Hex(fixture.bytes), length: fixture.bytes.byteLength + 1 },
    ]) {
      const mount = mountFixture();
      await expect(prepareCppCuteBrowserEmscriptenFactory({
        factory: fixture.factory,
        clangWasmBytes: fixture.bytes,
        expectedWasmSha256: input.hash,
        expectedWasmByteLength: input.length,
        vfsMount: mount.mount,
      })).rejects.toSatisfy(expectCode(
        "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-MISMATCH",
      ));
      expect(mount.state.state).toBe(
        input.length === fixture.bytes.byteLength ? "discarded" : "prepared",
      );
      expect(mount.state.bindCalls).toBe(0);
    }
    expect(fixture.calls.count).toBe(0);
  });

  it("rejects missing generated imports and any ambient import module", async () => {
    for (const options of [
      { omitGeneratedImport: "invoke_iii" },
      { extraImportModule: true },
    ]) {
      const fixture = factoryFixture(options);
      const mount = mountFixture();
      await expect(prepareCppCuteBrowserEmscriptenFactory({
        factory: fixture.factory,
        clangWasmBytes: fixture.bytes,
        expectedWasmSha256: await sha256Hex(fixture.bytes),
        expectedWasmByteLength: fixture.bytes.byteLength,
        vfsMount: mount.mount,
      })).rejects.toSatisfy(expectCode(
        options.omitGeneratedImport === undefined
          ? "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-INVALID"
          : "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-MISMATCH",
      ));
      expect(mount.state.state).toBe("discarded");
    }
  });

  it("closes the bound VFS session when the generated facade differs from the instance", async () => {
    const fixture = factoryFixture({ mismatchFacade: true });
    const mount = mountFixture();
    await expect(prepareCppCuteBrowserEmscriptenFactory({
      factory: fixture.factory,
      clangWasmBytes: fixture.bytes,
      expectedWasmSha256: await sha256Hex(fixture.bytes),
      expectedWasmByteLength: fixture.bytes.byteLength,
      vfsMount: mount.mount,
    })).rejects.toSatisfy(expectCode(
      "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-MISMATCH",
    ));
    expect(mount.state).toMatchObject({ state: "discarded", bindCalls: 1, closeCalls: 1 });
  });

  it("discards a prepared binding by terminalizing its bound VFS session", async () => {
    const fixture = factoryFixture();
    const mount = mountFixture();
    const prepared = await prepareCppCuteBrowserEmscriptenFactory({
      factory: fixture.factory,
      clangWasmBytes: fixture.bytes,
      expectedWasmSha256: await sha256Hex(fixture.bytes),
      expectedWasmByteLength: fixture.bytes.byteLength,
      vfsMount: mount.mount,
    });
    discardCppCuteBrowserEmscriptenFactory(prepared);
    expect(inspectCppCuteBrowserEmscriptenFactory(prepared).state).toBe("discarded");
    expect(mount.state).toMatchObject({ state: "discarded", closeCalls: 1 });
    expectSyncCode(
      () => discardCppCuteBrowserEmscriptenFactory(prepared),
      "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-STATE",
    );
  });

  it("rejects forged factories, byte views, input accessors, and authorities", async () => {
    const fixture = factoryFixture();
    const mount = mountFixture();
    const digest = await sha256Hex(fixture.bytes);
    const cases: unknown[] = [
      { ...baseInput(fixture, mount, digest), factory: () => Promise.resolve({}) },
      { ...baseInput(fixture, mount, digest), clangWasmBytes: new Proxy(fixture.bytes, {}) },
      Object.defineProperty({}, "factory", { get: () => fixture.factory }),
      { ...baseInput(fixture, mount, digest), extra: true },
    ];
    for (const value of cases) {
      const freshMount = mountFixture();
      const candidate = value === cases[0] || value === cases[1] || value === cases[3]
        ? { ...(value as Record<string, unknown>), vfsMount: freshMount.mount }
        : value;
      await expect(prepareCppCuteBrowserEmscriptenFactory(
        candidate as never,
      )).rejects.toSatisfy(expectCode(
        "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-INVALID",
      ));
    }
    expectSyncCode(
      () => inspectCppCuteBrowserEmscriptenFactory(Object.freeze({}) as never),
      "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-UNVERIFIED",
    );
  });
});

function baseInput(
  fixture: FactoryFixture,
  mount: MountFixture,
  digest: string,
): Record<string, unknown> {
  return {
    factory: fixture.factory,
    clangWasmBytes: fixture.bytes,
    expectedWasmSha256: digest,
    expectedWasmByteLength: fixture.bytes.byteLength,
    vfsMount: mount.mount,
  };
}

function mountFixture(): MountFixture {
  const ordinal = nextMockOrdinal++;
  const mount = Object.freeze({ mountOrdinal: ordinal }) as PreparedCppCuteBrowserVfsMount;
  const session = Object.freeze({ sessionOrdinal: ordinal });
  const state: MockMountState = {
    state: "prepared",
    session,
    createImportsCalls: 0,
    bindCalls: 0,
    discardCalls: 0,
    closeCalls: 0,
  };
  vfsAuthorities.mounts.set(mount, state);
  vfsAuthorities.sessions.set(session, state);
  return { mount, state };
}

function factoryFixture(options: FactoryOptions = {}): FactoryFixture {
  const bytes = abiFactoryModule();
  const calls = { count: 0 };
  let moduleArgument: CppCuteBrowserEmscriptenFactoryModuleArgument | undefined;
  const generatedImports: Record<string, Function> = {};
  for (const entry of RUNTIME_ABI.hostImports.generatedImportAllowlist.exactFunctions) {
    if (entry.fieldName !== options.omitGeneratedImport) generatedImports[entry.fieldName] = () => 0;
  }
  Object.freeze(generatedImports);
  const factory = async function (
    nextModuleArgument: CppCuteBrowserEmscriptenFactoryModuleArgument,
  ): Promise<unknown> {
    moduleArgument = nextModuleArgument;
    calls.count += 1;
    if (options.logLine !== undefined) nextModuleArgument.print(options.logLine);
    return new Promise((resolve) => {
      const imports: Record<string, WebAssembly.ModuleImports> = {
        env: generatedImports,
        wasi_snapshot_preview1: generatedImports,
      };
      if (options.extraImportModule === true) imports["ambient_network"] = Object.freeze({});
      nextModuleArgument.instantiateWasm(imports, (instance) => {
        const facade: Record<string, unknown> = {};
        for (const entry of RUNTIME_ABI.cExports) {
          facade[`_${entry.wasmExportName}`] = instance.exports[entry.wasmExportName];
        }
        if (options.mismatchFacade === true) {
          facade["_bg_cpp_cute_compile"] = instance.exports.bg_cpp_cute_status;
        }
        resolve(Object.freeze(facade));
      });
    });
  };
  const emit = (stream: "print" | "printErr", line: string): void => {
    if (moduleArgument === undefined) throw new Error("test factory has not been invoked");
    moduleArgument[stream](line);
  };
  return {
    bytes,
    factory,
    calls,
    emitStdout: (line) => emit("print", line),
    emitStderr: (line) => emit("printErr", line),
  };
}

function abiFactoryModule(): Uint8Array {
  const importEntries = [
    ...RUNTIME_ABI.hostImports.generatedImportAllowlist.exactFunctions.map((entry) => ({
      module: entry.moduleName,
      name: entry.fieldName,
      parameters: entry.wasmParameters,
      results: entry.wasmResults,
    })),
    ...RUNTIME_ABI.hostImports.functions.map((entry) => ({
      module: RUNTIME_ABI.hostImports.moduleName,
      name: entry.fieldName,
      parameters: entry.wasmParameters,
      results: entry.wasmResults,
    })),
  ];
  const typeIndices = new Map<string, number>();
  const types: number[][] = [];
  const typeIndex = (parameters: readonly WasmValueType[], results: readonly WasmValueType[]): number => {
    const key = `${parameters.join(",")}=>${results.join(",")}`;
    const existing = typeIndices.get(key);
    if (existing !== undefined) return existing;
    const next = types.length;
    typeIndices.set(key, next);
    types.push(functionType(parameters, results));
    return next;
  };
  const imports = importEntries.map((entry) => [
    ...name(entry.module), ...name(entry.name), 0x00,
    ...u32(typeIndex(entry.parameters, entry.results)),
  ]);
  const definedTypeIndices = RUNTIME_ABI.cExports.map((entry) =>
    u32(typeIndex(entry.wasmParameters, entry.wasmResults))
  );
  const exports = [
    ...RUNTIME_ABI.cExports.map((entry, index) => [
      ...name(entry.wasmExportName), 0x00, ...u32(imports.length + index),
    ]),
    [...name("memory"), 0x02, 0x00],
  ];
  const bodies = RUNTIME_ABI.cExports.map((entry) => {
    const result = entry.wasmExportName === "bg_cpp_cute_abi_version"
      ? RUNTIME_ABI.wasm.cAbiVersion
      : entry.wasmExportName === "bg_cpp_cute_allocator_metrics_pointer" ? 8
        : entry.wasmExportName === "bg_cpp_cute_frontend_work_metrics_pointer" ? 16
          : 0;
    const instructions = entry.wasmResults.length === 0 ? [0x0b] : [0x41, ...u32(result), 0x0b];
    const body = [0x00, ...instructions];
    return [...u32(body.length), ...body];
  });
  return new Uint8Array([
    ...HEADER,
    ...section(1, vector(types)),
    ...section(2, vector(imports)),
    ...section(3, vector(definedTypeIndices)),
    ...section(5, [0x01, 0x01, 0x01, 0x02]),
    ...section(7, vector(exports)),
    ...section(10, vector(bodies)),
  ]);
}

function functionType(
  parameters: readonly WasmValueType[],
  results: readonly WasmValueType[],
): number[] {
  return [
    0x60,
    ...u32(parameters.length),
    ...parameters.map((value) => WASM_VALUE_TYPE[value]),
    ...u32(results.length),
    ...results.map((value) => WASM_VALUE_TYPE[value]),
  ];
}

function u32(value: number): number[] {
  const result: number[] = [];
  do {
    let byte = value & 0x7f;
    value >>>= 7;
    if (value !== 0) byte |= 0x80;
    result.push(byte);
  } while (value !== 0);
  return result;
}

function name(value: string): number[] {
  const bytes = [...new TextEncoder().encode(value)];
  return [...u32(bytes.length), ...bytes];
}

function section(id: number, payload: Bytes): number[] {
  return [id, ...u32(payload.length), ...payload];
}

function vector(entries: readonly Bytes[]): number[] {
  return [...u32(entries.length), ...entries.flatMap((entry) => [...entry])];
}

function expectCode(code: CppCuteBrowserEmscriptenFactoryError["code"]): (error: unknown) => boolean {
  return (error) => error instanceof CppCuteBrowserEmscriptenFactoryError && error.code === code;
}

function expectSyncCode(
  operation: () => unknown,
  code: CppCuteBrowserEmscriptenFactoryError["code"],
): void {
  try {
    operation();
  } catch (error) {
    expect(expectCode(code)(error)).toBe(true);
    return;
  }
  throw new Error(`expected synchronous ${code}`);
}
