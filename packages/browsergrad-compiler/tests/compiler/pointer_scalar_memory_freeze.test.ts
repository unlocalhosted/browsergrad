import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  canEmitSemanticKernelIrWgsl,
  compileCudaLiteKernelForWebGpu,
  runCompiledKernelSemanticReference,
  type CompiledKernelInput,
  type CudaLiteSemanticSymbol,
  type SemanticExpression,
  type SemanticKernelIrOperation,
  type SemanticMemoryRef,
} from "../../src/index.js";

type BufferType = "Float32Array" | "Int32Array" | "Uint32Array";

interface BufferFixture {
  readonly type: BufferType;
  readonly values: readonly number[];
}

interface ScenarioFixture {
  readonly buffers: Readonly<Record<string, BufferFixture>>;
  readonly scalars?: Readonly<Record<string, number>>;
  readonly expectedBuffers: Readonly<Record<string, readonly number[]>>;
  readonly referenceTiers: readonly string[];
}

interface PointerCaseFixture {
  readonly id: string;
  readonly source: string;
  readonly workgroupSize: readonly [number, number, number];
  readonly operationKinds: readonly string[];
  readonly requiredSymbolFacts?: readonly Readonly<Record<string, unknown>>[];
  readonly requiredMemoryFacts?: readonly Readonly<Record<string, unknown>>[];
  readonly scenarios: readonly ScenarioFixture[];
  readonly wgslEligible: boolean;
}

interface PointerFixture {
  readonly schemaVersion: 1;
  readonly adapterId: "compiler.pointer-scalar-memory.v0";
  readonly cases: readonly PointerCaseFixture[];
}

const fixtureFile = join(dirname(fileURLToPath(import.meta.url)), "../fixtures/pointer-scalar-memory.v0.json");
const fixture = JSON.parse(readFileSync(fixtureFile, "utf8")) as PointerFixture;

describe("frozen compiler pointer/scalar memory behavior", () => {
  for (const testCase of fixture.cases) {
    it(testCase.id, () => {
      const compiled = compileCudaLiteKernelForWebGpu(testCase.source, { workgroupSize: testCase.workgroupSize });
      const operations = allOperations(compiled.kernelIr.operations);

      expect(operations.map((operation) => operation.kind)).toEqual(testCase.operationKinds);
      for (const expected of testCase.requiredSymbolFacts ?? []) {
        const allSymbols = [...compiled.kernelIr.symbols, ...compiled.kernelIr.memory, ...compiled.kernelIr.params];
        const symbol = allSymbols.find((candidate) => candidate.name === expected.name);
        expect(
          symbolProjection(symbol),
          `missing symbol fact ${String(expected.name)}; symbols are ${allSymbols.map((candidate) => candidate.name).join(", ")}`,
        ).toEqual(expect.objectContaining(expected));
      }
      const memoryFacts = operations.flatMap(operationMemoryRefs).map(memoryRefProjection);
      for (const expected of testCase.requiredMemoryFacts ?? []) {
        expect(memoryFacts, `missing memory fact ${JSON.stringify(expected)}`).toContainEqual(expect.objectContaining(expected));
      }

      expect(canEmitSemanticKernelIrWgsl(compiled.wgslLegalizedKernelIr)).toBe(testCase.wgslEligible);
      for (const scenario of testCase.scenarios) {
        expect(scenario.referenceTiers).toEqual(["semantic-cpu-reference"]);
        const result = runCompiledKernelSemanticReference(
          compiled,
          {
            buffers: instantiateBuffers(scenario.buffers),
            ...(scenario.scalars === undefined ? {} : { scalars: scenario.scalars }),
          },
          { gridDim: [1, 1, 1], blockDim: testCase.workgroupSize },
        );
        for (const [name, expected] of Object.entries(scenario.expectedBuffers)) {
          expect(Array.from(result.buffers[name] ?? []), `${name} output`).toEqual(expected);
        }
      }
    });
  }
});

function instantiateBuffers(fixtures: Readonly<Record<string, BufferFixture>>): CompiledKernelInput["buffers"] {
  return Object.fromEntries(Object.entries(fixtures).map(([name, buffer]) => [name, instantiateBuffer(buffer)]));
}

function instantiateBuffer(buffer: BufferFixture): CompiledKernelInput["buffers"][string] {
  switch (buffer.type) {
    case "Float32Array": return new Float32Array(buffer.values);
    case "Int32Array": return new Int32Array(buffer.values);
    case "Uint32Array": return new Uint32Array(buffer.values);
  }
}

function allOperations(roots: readonly SemanticKernelIrOperation[]): SemanticKernelIrOperation[] {
  const operations: SemanticKernelIrOperation[] = [];
  const visit = (operation: SemanticKernelIrOperation): void => {
    operations.push(operation);
    if (operation.kind === "branch") {
      operation.consequent.forEach(visit);
      operation.alternate.forEach(visit);
    } else if (operation.kind === "loop") {
      if (operation.init !== undefined && isKernelOperation(operation.init)) visit(operation.init);
      operation.body.forEach(visit);
      operation.continuing?.forEach(visit);
    } else if (operation.kind === "block") {
      operation.body.forEach(visit);
    }
  };
  roots.forEach(visit);
  return operations;
}

const kernelOperationKinds: ReadonlySet<string> = new Set([
  "declare", "dim3-declare", "cooperative-group-declare", "load", "store", "copy", "copy-fence",
  "matrix-fill", "matrix-load", "matrix-mma", "matrix-store", "surface-write", "surface-read-store",
  "atomic", "runtime-copy", "pool-allocate", "pointer-rebind", "pointer-array-rebind", "expression",
  "branch", "loop", "barrier", "fence", "device-launch", "inline-asm", "return", "continue", "break", "block",
]);

function isKernelOperation(value: SemanticExpression | SemanticKernelIrOperation): value is SemanticKernelIrOperation {
  return kernelOperationKinds.has(value.kind);
}

function symbolProjection(symbol: CudaLiteSemanticSymbol | undefined): Readonly<Record<string, unknown>> {
  if (symbol === undefined) return {};
  return {
    name: symbol.name,
    pointer: symbol.pointer === true,
    pointerRuntimeState: symbol.pointerRuntimeState === true,
    pointerAddressSpace: symbol.pointerAddressSpace,
    pointerSelection: symbol.pointerSelection !== undefined,
    pointerArrayAliasCount: symbol.pointerArrayAliases?.length,
  };
}

function operationMemoryRefs(operation: SemanticKernelIrOperation): readonly SemanticMemoryRef[] {
  switch (operation.kind) {
    case "load": return [operation.source];
    case "store": return [operation.target, ...operation.reads];
    case "copy": return [operation.source, operation.target];
    case "matrix-load": return [operation.source];
    case "matrix-store": return [operation.target];
    case "atomic": return operation.target === undefined ? [] : [operation.target];
    case "call": return operation.reads;
    case "pointer-rebind": return [operation.source];
    case "pointer-array-rebind": return [operation.source];
    default: return [];
  }
}

function memoryRefProjection(ref: SemanticMemoryRef): Readonly<Record<string, unknown>> {
  return {
    base: ref.base,
    addressSpace: ref.addressSpace,
    valueType: ref.valueType,
    containerValueType: ref.containerValueType,
    pointerBaseIsScalarLane: ref.pointerBaseIsScalarLane === true,
    pointerBaseUnitBytes: ref.pointerBaseUnitBytes,
    packedByteLanes: ref.packedByteLanes,
    indexRank: ref.indices.length,
    fields: ref.fields,
  };
}
