import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_WASM_BASE_OPERATIONS,
  CppCuteBrowserWasmInspectionError,
  inspectCppCuteBrowserWasmAgainstRuntimeAbi,
  unwrapPreparedCppCuteBrowserWasmConformance,
  verifyCppCuteBrowserWasmConformance,
} from "../../src/cpp_cute_browser_wasm_inspection.js";
import {
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
} from "../../src/cpp_cute_browser_runtime_abi.js";

type Bytes = readonly number[];

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

function functionType(parameters: Bytes, results: Bytes): number[] {
  return [0x60, ...u32(parameters.length), ...parameters, ...u32(results.length), ...results];
}

const I32 = 0x7f;
const HEADER = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00] as const;
const VFS_IMPORTS = [
  ["bg_vfs_status", 5],
  ["bg_vfs_open", 5],
  ["bg_vfs_read", 6],
  ["bg_vfs_close", 1],
  ["bg_vfs_directory_count", 5],
  ["bg_vfs_directory_entry", 7],
] as const;
const C_EXPORTS = [
  ["bg_cpp_cute_abi_version", 0],
  ["bg_cpp_cute_alloc", 1],
  ["bg_cpp_cute_allocator_metrics_pointer", 0],
  ["bg_cpp_cute_compile", 2],
  ["bg_cpp_cute_free", 3],
  ["bg_cpp_cute_reset", 4],
  ["bg_cpp_cute_result_length", 0],
  ["bg_cpp_cute_result_pointer", 0],
  ["bg_cpp_cute_status", 0],
  ["bg_cpp_cute_last_diagnostic_code", 0],
] as const;

function targetFeatures(entries: readonly string[]): number[] {
  const featureEntries = entries.map((entry) => {
    const prefix = entry[0] === "+" ? 0x2b : 0x2d;
    return [prefix, ...name(entry.slice(1))];
  });
  return section(0, [...name("target_features"), ...vector(featureEntries)]);
}

interface ModuleOptions {
  readonly includeTargetFeatures?: boolean;
  readonly memoryMinimum?: number;
  readonly memoryMaximum?: number;
  readonly start?: boolean;
  readonly firstResultOpcode?: number;
  readonly customSections?: readonly number[][];
}

function abiShapedModule(options: ModuleOptions = {}): Uint8Array {
  const types = [
    functionType([], [I32]),
    functionType([I32], [I32]),
    functionType([I32, I32], [I32]),
    functionType([I32, I32], []),
    functionType([], []),
    functionType([I32, I32, I32], [I32]),
    functionType([I32, I32, I32, I32, I32], [I32]),
    functionType([I32, I32, I32, I32, I32, I32], [I32]),
  ];
  const imports = VFS_IMPORTS.map(([fieldName, typeIndex]) => [
    ...name("browsergrad_vfs_v1"), ...name(fieldName), 0x00, ...u32(typeIndex),
  ]);
  const definedTypeIndices = C_EXPORTS.map(([, typeIndex]) => u32(typeIndex));
  const exports = [
    ...C_EXPORTS.map(([exportName, _typeIndex], index) => [...name(exportName), 0x00, ...u32(VFS_IMPORTS.length + index)]),
    [...name("memory"), 0x02, 0x00],
  ];
  const resultBody = (opcode = 0x41): number[] => {
    const instructions = opcode === 0x41 ? [0x41, 0x00, 0x0b] : [opcode, 0x00, 0x0b];
    return [...u32(1 + instructions.length), 0x00, ...instructions];
  };
  const voidBody = (): number[] => [...u32(2), 0x00, 0x0b];
  const bodies = [
    resultBody(options.firstResultOpcode), resultBody(), resultBody(), resultBody(),
    voidBody(), voidBody(), resultBody(), resultBody(), resultBody(),
    resultBody(),
  ];
  const standard = [
    section(1, vector(types)),
    section(2, vector(imports)),
    section(3, vector(definedTypeIndices)),
    section(5, [0x01, 0x01, ...u32(options.memoryMinimum ?? 4_096), ...u32(options.memoryMaximum ?? 16_384)]),
    section(7, vector(exports)),
    ...(options.start === true ? [section(8, u32(VFS_IMPORTS.length + 5))] : []),
    section(10, vector(bodies)),
  ];
  return new Uint8Array([
    ...HEADER,
    ...(options.includeTargetFeatures === true
      ? targetFeatures(["+bulk-memory", "+mutable-globals", "+nontrapping-fptoint", "+sign-ext"])
      : []),
    ...(options.customSections ?? []).flat(),
    ...standard.flat(),
  ]);
}

function oneFunctionModule(
  instructions: Bytes,
  options: { readonly parameters?: Bytes; readonly result?: boolean; readonly extraSections?: readonly number[][] } = {},
): Uint8Array {
  const parameters = options.parameters ?? [];
  const results = options.result === true ? [I32] : [];
  const body = [0x00, ...instructions, 0x0b];
  return new Uint8Array([
    ...HEADER,
    ...section(1, vector([functionType(parameters, results)])),
    ...section(3, vector([[0x00]])),
    ...(options.extraSections ?? []).flat(),
    ...section(10, vector([[...u32(body.length), ...body]])),
  ]);
}

function concatenateBytes(chunks: readonly Uint8Array[]): Uint8Array {
  const result = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.byteLength, 0));
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function operationBudgetRegressionModule(): Uint8Array {
  const nopCount = Math.floor(CPP_CUTE_BROWSER_WASM_BASE_OPERATIONS / 2) + 1;
  const body = new Uint8Array(nopCount + 2);
  body[0] = 0x00;
  body.fill(0x01, 1, body.byteLength - 1);
  body[body.byteLength - 1] = 0x0b;
  const typeSection = new Uint8Array(section(1, vector([functionType([], [])])));
  const functionSection = new Uint8Array(section(3, vector([[0x00], [0x00]])));
  const bodyLength = new Uint8Array(u32(body.byteLength));
  const codePayloadByteLength = 1 + (bodyLength.byteLength + body.byteLength) * 2;
  const codeSectionHeader = new Uint8Array([0x0a, ...u32(codePayloadByteLength)]);
  const chunks = [
    new Uint8Array(HEADER), typeSection, functionSection, codeSectionHeader,
    new Uint8Array([0x02]), bodyLength, body, bodyLength, body,
  ];
  return concatenateBytes(chunks);
}

function projectionBudgetRegressionModule(): Uint8Array {
  const functionCount = 30_000;
  const typeSection = new Uint8Array(section(1, vector([functionType([], [])])));
  const functionCountBytes = new Uint8Array(u32(functionCount));
  const functionPayload = new Uint8Array(functionCountBytes.byteLength + functionCount);
  functionPayload.set(functionCountBytes);
  const functionSection = concatenateBytes([
    new Uint8Array([0x03, ...u32(functionPayload.byteLength)]),
    functionPayload,
  ]);
  const codePayload = new Uint8Array(functionCountBytes.byteLength + functionCount * 3);
  codePayload.set(functionCountBytes);
  for (let offset = functionCountBytes.byteLength; offset < codePayload.byteLength; offset += 3) {
    codePayload.set([0x02, 0x00, 0x0b], offset);
  }
  const codeSection = concatenateBytes([
    new Uint8Array([0x0a, ...u32(codePayload.byteLength)]),
    codePayload,
  ]);
  return concatenateBytes([new Uint8Array(HEADER), typeSection, functionSection, codeSection]);
}

function structurallyRichModule(): Uint8Array {
  const custom = targetFeatures([
    "+bulk-memory", "+mutable-globals", "+nontrapping-fptoint", "+sign-ext",
  ]);
  const type = section(1, vector([functionType([], [])]));
  const functions = section(3, vector([[0x00]]));
  const table = section(4, [0x01, 0x70, 0x01, 0x01, 0x01]);
  const memory = section(5, [0x01, 0x01, 0x01, 0x02]);
  const global = section(6, [0x01, I32, 0x01, 0x41, 0x00, 0x0b]);
  const exports = section(7, vector([
    [...name("f"), 0x00, 0x00],
    [...name("t"), 0x01, 0x00],
    [...name("m"), 0x02, 0x00],
    [...name("g"), 0x03, 0x00],
  ]));
  const element = section(9, [0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x00]);
  const dataCount = section(12, [0x01]);
  const code = section(10, vector([[0x02, 0x00, 0x0b]]));
  const data = section(11, [0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0xaa]);
  return new Uint8Array([...HEADER, ...custom, ...type, ...functions, ...table, ...memory, ...global, ...exports, ...element, ...dataCount, ...code, ...data]);
}

async function runtimeAbi() {
  return decodeCppCuteBrowserRuntimeAbiManifest(cppCuteBrowserRuntimeAbiManifestResourceBytes());
}

function expectCode(code: CppCuteBrowserWasmInspectionError["code"]): (error: unknown) => boolean {
  return (error) => error instanceof CppCuteBrowserWasmInspectionError && error.code === code;
}

describe("bounded raw-Wasm inspection", () => {
  it("projects exact signatures, memory limits, framing, and current review blockers", async () => {
    const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(abiShapedModule(), await runtimeAbi());

    expect(report.authority).toBe("review-observation-only");
    expect(report.rawWasmVerified).toBe(true);
    expect(report.workerExecutionReady).toBe(false);
    expect(report.releaseReady).toBe(false);
    expect(report.exactInterfaceConformance).toBe(false);
    expect(report.projection.sectionOrder).toEqual(["type", "import", "function", "memory", "export", "code"]);
    expect(report.projection.imports).toHaveLength(6);
    expect(report.projection.imports[2]?.functionType).toEqual({
      parameters: ["i32", "i32", "i32", "i32", "i32"], results: ["i32"],
    });
    expect(report.projection.exports.find((entry) => entry.name === "bg_cpp_cute_free")?.functionType).toEqual({
      parameters: ["i32", "i32"], results: [],
    });
    expect(report.projection.memories).toEqual([{ addressType: "i32", shared: false, minimumPages: 4_096, maximumPages: 16_384 }]);
    expect(report.mismatches.some((entry) => entry.startsWith("required target feature +"))).toBe(false);
    expect(Object.isFrozen(report.projection.imports)).toBe(true);
    expect(report.wasmSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(report.observedProjectionSha256).toMatch(/^[0-9a-f]{64}$/u);
  });

  it("parses exact target_features wire names without treating metadata as conformance", async () => {
    const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule({ includeTargetFeatures: true }),
      await runtimeAbi(),
    );
    const target = report.projection.customSections[0];
    expect(target?.name).toBe("target_features");
    expect(target?.targetFeatures).toEqual([
      { prefix: "+", wireName: "bulk-memory", normalizedName: "bulk-memory" },
      { prefix: "+", wireName: "mutable-globals", normalizedName: "mutable-globals" },
      { prefix: "+", wireName: "nontrapping-fptoint", normalizedName: "nontrapping-fptoint" },
      { prefix: "+", wireName: "sign-ext", normalizedName: "sign-extension" },
    ]);
    expect(report.mismatches).toContain("custom section target_features is not independently allowlisted");
    expect(report.exactInterfaceConformance).toBe(false);
  });

  it("does not let observed bytes mint or widen the current empty allowlists", async () => {
    const abi = await runtimeAbi();
    await expect(verifyCppCuteBrowserWasmConformance(
      abiShapedModule({ includeTargetFeatures: true }),
      abi,
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-ABI-MISMATCH"));
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule(),
      { manifestId: abi.manifestId, contractSha256: abi.contractSha256 } as never,
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-UNVERIFIED"));
  });

  it("reports ABI memory/start drift without granting authority", async () => {
    const abi = await runtimeAbi();
    const memory = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule({ memoryMinimum: 1 }), abi,
    );
    expect(memory.mismatches).toContain("defined memory type/limits do not equal runtime ABI");
    const start = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule({ start: true }), abi,
    );
    expect(start.mismatches).toContain("start section is forbidden");
  });

  it("rejects duplicate, out-of-order, unknown, truncated, and nonminimal sections", async () => {
    const abi = await runtimeAbi();
    const cases = [
      new Uint8Array([...HEADER, ...section(1, [0]), ...section(1, [0])]),
      new Uint8Array([...HEADER, ...section(3, [0]), ...section(1, [0])]),
      new Uint8Array([...HEADER, 14, 0]),
      new Uint8Array([...HEADER, 1, 2, 1]),
      new Uint8Array([...HEADER, 1, 0x80, 0x00]),
    ];
    for (const bytes of cases) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, abi)).rejects.toSatisfy(
        expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
      );
    }
  });

  it("rejects function/code count drift and code-body trailing bytes", async () => {
    const abi = await runtimeAbi();
    const mismatch = new Uint8Array([
      ...HEADER,
      ...section(1, vector([functionType([], [])])),
      ...section(3, vector([[0]])),
      ...section(10, vector([])),
    ]);
    const trailingBody = new Uint8Array([
      ...HEADER,
      ...section(1, vector([functionType([], [])])),
      ...section(3, vector([[0]])),
      ...section(10, vector([[3, 0, 0x0b, 0x00]])),
    ]);
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(mismatch, abi)).rejects.toSatisfy(
      expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
    );
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(trailingBody, abi)).rejects.toSatisfy(
      expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
    );
  });

  it("rejects forbidden opcode families and target-feature declarations", async () => {
    const abi = await runtimeAbi();
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule({ firstResultOpcode: 0xfd }), abi,
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
    const forbidden = targetFeatures(["+simd128"]);
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule({ customSections: [forbidden] }), abi,
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
  });

  it("rejects malformed target_features and forbidden custom metadata", async () => {
    const abi = await runtimeAbi();
    const malformed = section(0, [...name("target_features"), 1, 0x2b, 0x80]);
    const invalidUtf8 = section(0, [1, 0xff]);
    const producers = section(0, [...name("producers")]);
    for (const custom of [malformed, invalidUtf8, producers]) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
        new Uint8Array([...HEADER, ...custom]), abi,
      )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
    }
  });

  it("enforces caller-narrowed byte/operation budgets and cancellation", async () => {
    const abi = await runtimeAbi();
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule(), abi, { maxModuleByteLength: 8 },
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-RESOURCE-LIMIT"));
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule(), abi, { maxOperations: 8 },
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-RESOURCE-LIMIT"));
    const controller = new AbortController();
    controller.abort();
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule(), abi, { signal: controller.signal },
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-CANCELLED"));
  });

  it("scales the default operation budget to an admitted module snapshot", async () => {
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      operationBudgetRegressionModule(),
      await runtimeAbi(),
    )).resolves.toMatchObject({ rawWasmVerified: true });
  });

  it("hashes a bounded production-scale structural projection", async () => {
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      projectionBudgetRegressionModule(),
      await runtimeAbi(),
    )).resolves.toMatchObject({
      rawWasmVerified: true,
      observedProjectionSha256: expect.stringMatching(/^[0-9a-f]{64}$/u),
    });
  });

  it("admits valid generated control nesting beyond the old diagnostic ceiling", async () => {
    const depth = 1_025;
    const instructions = [
      ...Array.from({ length: depth }, () => [0x02, 0x40]).flat(),
      ...Array.from({ length: depth }, () => 0x0b),
    ];
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      oneFunctionModule(instructions),
      await runtimeAbi(),
    )).resolves.toMatchObject({ rawWasmVerified: true });
  });

  it("trusts decoded features without requiring advisory target metadata", async () => {
    const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      oneFunctionModule([0x41, 0x00, 0xc0], { result: true }),
      await runtimeAbi(),
    );
    expect(report.projection.staticallyUsedExtensions).toContain("sign-extension");
    expect(report.mismatches).not.toContain("required target feature +sign-ext is missing");
  });

  it("rejects subclass, proxy, and shared byte views before parsing", async () => {
    const abi = await runtimeAbi();
    class BytesSubclass extends Uint8Array {}
    const values: unknown[] = [new BytesSubclass(HEADER), new Proxy(new Uint8Array(HEADER), {})];
    if (typeof SharedArrayBuffer !== "undefined") values.push(new Uint8Array(new SharedArrayBuffer(8)));
    for (const value of values) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(value as Uint8Array, abi)).rejects.toSatisfy(
        expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
      );
    }
  });

  it("accepts spec-valid padded LEB encodings while retaining exact framing", async () => {
    const abi = await runtimeAbi();
    const paddedEmptyTypeSection = new Uint8Array([
      ...HEADER,
      0x01, 0x82, 0x00, // section byte length = 2
      0x80, 0x00, // type vector count = 0
    ]);
    const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(paddedEmptyTypeSection, abi);
    expect(report.projection.types).toEqual([]);

    const paddedSignedConstant = oneFunctionModule([0x41, 0x80, 0x00], { result: true });
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(paddedSignedConstant, abi)).resolves.toMatchObject({
      rawWasmVerified: true,
    });
  });

  it("rejects overflowing and unterminated LEB encodings at each width boundary", async () => {
    const abi = await runtimeAbi();
    const cases = [
      new Uint8Array([...HEADER, 0x01, 0x80, 0x80, 0x80, 0x80, 0x10]),
      new Uint8Array([...HEADER, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80]),
      oneFunctionModule([0x41, 0xff, 0xff, 0xff, 0xff, 0x0f], { result: true }),
      oneFunctionModule([0x42, ...Array.from({ length: 10 }, () => 0x80)], { result: true }),
    ];
    for (const bytes of cases) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, abi)).rejects.toSatisfy(
        expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
      );
    }
  });

  it("uses intrinsic WebAssembly validation to reject type-stack-invalid bodies", async () => {
    const stackInvalid = oneFunctionModule([0x41, 0x00]);
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      stackInvalid, await runtimeAbi(),
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
  });

  it("accepts generic fatal UTF-8 Wasm names and leaves ABI vocabulary to comparison", async () => {
    const unicodeExport = new Uint8Array([
      ...HEADER,
      ...section(1, vector([functionType([], [])])),
      ...section(3, vector([[0x00]])),
      ...section(7, vector([[...name("lambda-λ"), 0x00, 0x00]])),
      ...section(10, vector([[0x02, 0x00, 0x0b]])),
    ]);
    const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(unicodeExport, await runtimeAbi());
    expect(report.projection.exports[0]?.name).toBe("lambda-λ");
    expect(report.exactInterfaceConformance).toBe(false);
  });

  it("projects table/global/element/data/code/custom surfaces completely", async () => {
    const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(structurallyRichModule(), await runtimeAbi());
    expect(report.projection.tables).toEqual([{ elementType: "funcref", minimum: 1, maximum: 1 }]);
    expect(report.projection.globals).toEqual([{ valueType: "i32", mutable: true, initializerOpcode: 0x41 }]);
    expect(report.projection.exports.find((entry) => entry.name === "t")?.table).toEqual({
      elementType: "funcref", minimum: 1, maximum: 1,
    });
    expect(report.projection.exports.find((entry) => entry.name === "g")?.global).toEqual({
      valueType: "i32", mutable: true, initializerOpcode: 0x41,
    });
    expect(report.projection.elementSegments).toEqual([{ mode: "active", tableIndex: 0, elementType: "funcref", elementCount: 1 }]);
    expect(report.projection.dataCount).toBe(1);
    expect(report.projection.dataSegments).toEqual([{ mode: "active", memoryIndex: 0, byteLength: 1 }]);
    expect(report.projection.codeBodies).toEqual([{ bodyByteLength: 2, localCount: 0 }]);
    expect(report.projection.staticallyUsedExtensions).toEqual(["bulk-memory", "mutable-globals"]);
    expect(report.mismatches).toContain("table export inventory differs from the exact runtime ABI allowlist");
    expect(report.mismatches).toContain("global export inventory differs from the exact runtime ABI allowlist");
    expect(report.mismatches).toContain("global export projection differs from independently reviewed ABI projection");
    expect(report.mismatches).not.toContain("table projection differs from independently reviewed ABI projection");
    expect(report.exactInterfaceConformance).toBe(false);
  });

  it("rejects data-count mismatch and bulk-memory data indices without data-count", async () => {
    const abi = await runtimeAbi();
    const mismatchedDataCount = new Uint8Array([
      ...HEADER,
      ...section(12, [0x01]),
      ...section(11, [0x00]),
    ]);
    const memory = section(5, [0x01, 0x01, 0x01, 0x02]);
    const memoryInitWithoutDataCount = oneFunctionModule(
      [0xfc, 0x08, 0x00, 0x00],
      { extraSections: [memory] },
    );
    for (const bytes of [mismatchedDataCount, memoryInitWithoutDataCount]) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, abi)).rejects.toSatisfy(
        expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
      );
    }
  });

  it("rejects out-of-range function, export, local, global, table, memory, and element indices", async () => {
    const abi = await runtimeAbi();
    const badExport = new Uint8Array([
      ...HEADER,
      ...section(7, vector([[...name("f"), 0x00, 0x00]])),
    ]);
    const badElement = new Uint8Array([
      ...HEADER,
      ...section(4, [0x01, 0x70, 0x01, 0x01, 0x01]),
      ...section(9, [0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x00]),
    ]);
    const badLocal = oneFunctionModule([0x20, 0x00]);
    const badGlobal = oneFunctionModule([0x23, 0x00]);
    const badCall = oneFunctionModule([0x10, 0x01]);
    const badTable = oneFunctionModule([0x11, 0x00, 0x00]);
    const badMemory = oneFunctionModule([0x3f, 0x00]);
    for (const bytes of [badExport, badElement, badLocal, badGlobal, badCall, badTable, badMemory]) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, abi)).rejects.toSatisfy(
        expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
      );
    }
  });

  it("rejects shared, memory64, inverted, duplicate-memory, and imported-memory shapes", async () => {
    const abi = await runtimeAbi();
    const types = section(1, [0x00]);
    const shared = new Uint8Array([...HEADER, ...types, ...section(5, [0x01, 0x03, 0x01, 0x02])]);
    const memory64 = new Uint8Array([...HEADER, ...types, ...section(5, [0x01, 0x05, 0x01, 0x02])]);
    const inverted = new Uint8Array([...HEADER, ...types, ...section(5, [0x01, 0x01, 0x02, 0x01])]);
    const duplicate = new Uint8Array([...HEADER, ...section(5, [0x02, 0x00, 0x01, 0x00, 0x01])]);
    const imported = new Uint8Array([
      ...HEADER,
      ...section(2, vector([[...name("env"), ...name("memory"), 0x02, 0x01, 0x01, 0x02]])),
    ]);
    for (const bytes of [shared, memory64, inverted]) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, abi)).rejects.toSatisfy(
        expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
      );
    }
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(duplicate, abi)).rejects.toSatisfy(
      expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-RESOURCE-LIMIT"),
    );
    const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(imported, abi);
    expect(report.mismatches.some((entry) => entry.includes("non-function imports"))).toBe(true);
  });

  it("rejects tags and every forbidden/reserved opcode family", async () => {
    const abi = await runtimeAbi();
    const tag = new Uint8Array([
      ...HEADER,
      ...section(1, vector([functionType([], [])])),
      ...section(13, [0x01, 0x00, 0x00]),
    ]);
    const opcodeCases = [
      oneFunctionModule([0xfe, 0x00]),
      oneFunctionModule([0x06, 0x40]),
      oneFunctionModule([0xfc, 0x12]),
      oneFunctionModule([0xff]),
      oneFunctionModule([0xd0, 0x70]),
    ];
    for (const bytes of [tag, ...opcodeCases]) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, abi)).rejects.toSatisfy(
        expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
      );
    }
  });

  it("decodes table.copy as two table indices rather than an element index", async () => {
    const table = section(4, [0x01, 0x70, 0x01, 0x01, 0x01]);
    const metadata = targetFeatures([
      "+bulk-memory", "+mutable-globals", "+nontrapping-fptoint", "+sign-ext",
    ]);
    const tableCopy = oneFunctionModule(
      [0x41, 0x00, 0x41, 0x00, 0x41, 0x00, 0xfc, 0x0e, 0x00, 0x00],
      { extraSections: [metadata, table] },
    );
    const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(tableCopy, await runtimeAbi());
    expect(report.projection.staticallyUsedExtensions).toContain("bulk-memory");
    expect(report.exactInterfaceConformance).toBe(false);
  });

  it("rejects padded call_indirect table indices as undeclared call-indirect-overlong", async () => {
    const table = section(4, [0x01, 0x70, 0x01, 0x01, 0x01]);
    const paddedTableIndex = oneFunctionModule(
      [0x41, 0x00, 0x11, 0x00, 0x80, 0x00],
      { extraSections: [table] },
    );
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      paddedTableIndex, await runtimeAbi(),
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
  });

  it("rejects explicit-memory-index memargs even when the encoded memory index is zero", async () => {
    const memory = section(5, [0x01, 0x01, 0x01, 0x02]);
    const explicitMemoryZero = oneFunctionModule(
      [0x41, 0x00, 0x28, 0x40, 0x00, 0x00],
      { result: true, extraSections: [memory] },
    );
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      explicitMemoryZero, await runtimeAbi(),
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
  });

  it("admits reviewed single-memory copy/fill while rejecting nonzero memory indices", async () => {
    const memory = section(5, [0x01, 0x01, 0x01, 0x02]);
    const memoryCopy = oneFunctionModule(
      [0x41, 0x00, 0x41, 0x00, 0x41, 0x00, 0xfc, 0x0a, 0x00, 0x00],
      { extraSections: [memory] },
    );
    const memoryFill = oneFunctionModule(
      [0x41, 0x00, 0x41, 0x00, 0x41, 0x00, 0xfc, 0x0b, 0x00],
      { extraSections: [memory] },
    );
    const abi = await runtimeAbi();
    for (const bytes of [memoryCopy, memoryFill]) {
      const report = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, abi);
      expect(report.projection.staticallyUsedExtensions).toEqual(["bulk-memory", "bulk-memory-opt"]);
      expect(report.mismatches).not.toContain("decoded extension bulk-memory-opt is not allowed");
      expect(report.exactInterfaceConformance).toBe(false);
    }
    const nonzeroMemoryIndex = oneFunctionModule(
      [0x41, 0x00, 0x41, 0x00, 0x41, 0x00, 0xfc, 0x0a, 0x01, 0x00],
      { extraSections: [memory] },
    );
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(nonzeroMemoryIndex, abi))
      .rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
  });

  it("rejects duplicate/negative-used target features and classifies valid unsorted/unlisted metadata", async () => {
    const abi = await runtimeAbi();
    const duplicate = targetFeatures(["+bulk-memory", "+bulk-memory"]);
    const unsorted = targetFeatures(["+sign-ext", "+bulk-memory"]);
    const unlisted = targetFeatures(["+tail-call"]);
    const negativeUsed = oneFunctionModule(
      [0x20, 0x00, 0xc0],
      { parameters: [I32], result: true, extraSections: [targetFeatures(["-sign-ext"])] },
    );
    for (const bytes of [
      new Uint8Array([...HEADER, ...duplicate]),
      negativeUsed,
    ]) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, abi)).rejects.toSatisfy(
        expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"),
      );
    }
    const unsortedReport = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      new Uint8Array([...HEADER, ...unsorted]), abi,
    );
    expect(unsortedReport.projection.customSections[0]?.targetFeatures?.map((entry) => entry.wireName)).toEqual([
      "sign-ext", "bulk-memory",
    ]);
    const unlistedReport = await inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      new Uint8Array([...HEADER, ...unlisted]), abi,
    );
    expect(unlistedReport.mismatches).toContain("target feature +tail-call is outside the runtime ABI extension allowlist");
  });

  it("rejects hostile option records and fake signals", async () => {
    const abi = await runtimeAbi();
    const hostileOptions: unknown[] = [
      Object.create(null),
      { unknown: true },
      { maxModuleByteLength: 0 },
      { maxOperations: Number.MAX_SAFE_INTEGER },
      { signal: {} },
    ];
    for (const options of hostileOptions) {
      await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
        abiShapedModule(), abi, options as never,
      )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
    }
    const throwing = Object.defineProperty({}, "signal", { get: () => { throw new Error("boom"); } });
    await expect(inspectCppCuteBrowserWasmAgainstRuntimeAbi(
      abiShapedModule(), abi, throwing,
    )).rejects.toSatisfy(expectCode("BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID"));
  });

  it("snapshots caller bytes before any async hashing and keeps reports deeply immutable", async () => {
    const bytes = abiShapedModule();
    const originalFirst = bytes[0];
    const pending = inspectCppCuteBrowserWasmAgainstRuntimeAbi(bytes, await runtimeAbi());
    bytes.fill(0xff);
    const report = await pending;
    expect(originalFirst).toBe(0x00);
    expect(report.rawWasmVerified).toBe(true);
    expect(Object.isFrozen(report)).toBe(true);
    expect(Object.isFrozen(report.projection.customSections)).toBe(true);
    expect(() => (report.projection.imports as unknown as unknown[]).push("forged")).toThrow();
  });

  it("rejects forged opaque conformance summaries", () => {
    expect(() => unwrapPreparedCppCuteBrowserWasmConformance({
      wasmSha256: "0".repeat(64),
      wasmByteLength: 8,
      observedProjectionSha256: "0".repeat(64),
      runtimeAbiManifestId: "forged",
      runtimeAbiContractSha256: "0".repeat(64),
      exactInterfaceConformance: true,
      rawWasmVerified: true,
      workerExecutionReady: false,
      releaseReady: false,
    } as never)).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-UNVERIFIED",
    }));
  });
});
