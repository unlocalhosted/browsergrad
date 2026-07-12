import { describe, expect, it } from "vitest";
import {
  analyzeCudaLite,
  validateSemanticKernelIr,
  typeCheckSemanticKernelIr,
  legalizeSemanticKernelIrForWgsl,
  createCudaLiteSemanticModel,
  createSemanticEnvironment,
  emitSemanticKernelIrWgsl,
  lowerSemanticCudaRuntime,
  lowerSemanticModelToKernelIr,
  parseCudaLite,
  type CudaLiteModule,
  type SemanticKernelIrModule,
  type SemanticExpression,
  type SemanticMemoryRef,
} from "../../src/index.js";
import { semanticIdKey, semanticMemoryIdFromSymbol } from "../../src/semantic_ids.js";
import {
  completeAnalysis,
  completeRuntimeLowering,
} from "../../src/compiler_phases.js";

describe("compiler phase contracts", () => {
  it("requires ordered, verified compiler stages", () => {
    const ast = parseCudaLite(`__global__ void phase_contract(float *out) { out[0] = 1.0f; }`);
    const analysis = analyzeCudaLite(ast);
    const semantic = createCudaLiteSemanticModel(analysis);
    const out = semantic.params[0]!;

    expect(semantic.environment.symbols.get(out.id)).toBe(out);
    expect(semantic.environment.symbolsByName.get("out")).toEqual([out.id]);
    if (false) {
      // @ts-expect-error semantic identity is not a source/backend name
      const _name: string = out.id;
    }
    const canonical = lowerSemanticModelToKernelIr(analysis, semantic, { workgroupSize: [1, 1, 1] });
    const runtimeLowered = lowerSemanticCudaRuntime(canonical);
    const verified = validateSemanticKernelIr(runtimeLowered);
    const typeChecked = typeCheckSemanticKernelIr(verified);
    const legalized = legalizeSemanticKernelIrForWgsl(typeChecked);
    const store = runtimeLowered.operations.find((operation) => operation.kind === "store");
    expect(store?.kind === "store" ? store.target.baseId : undefined).toBe(out.id);
    if (false) {
      const value: SemanticExpression = { kind: "literal", literalKind: "number", value: 1, valueType: "int", span: out.span };
      // @ts-expect-error computed semantic expressions require a result type
      const _expression: SemanticExpression = {
        kind: "binary",
        operator: "+",
        left: value,
        right: value,
        span: out.span,
      };
      // @ts-expect-error semantic memory references require a non-void element type
      const _ref: SemanticMemoryRef = {
        baseId: semanticMemoryIdFromSymbol(out.id),
        base: out.name,
        addressSpace: out.addressSpace,
        indices: [],
        fields: [],
        span: out.span,
      };
    }
    const emitted = emitSemanticKernelIrWgsl(legalized);

    expect(emitted.wgsl).toContain("browsergrad-semantic-wgsl");
  });

  it("rejects duplicate semantic declaration identities", () => {
    const ast = parseCudaLite(`__global__ void duplicate_identity(float *out) { out[0] = 1.0f; }`);
    const semantic = createCudaLiteSemanticModel(analyzeCudaLite(ast));
    const out = semantic.params[0]!;

    expect(() => createSemanticEnvironment([out, out], [])).toThrow(
      `duplicate semantic symbol id '${semanticIdKey(out.id)}'`,
    );
  });

  it("rejects phase bypasses during TypeScript checking", () => {
    if (false) {
      // @ts-expect-error raw AST has not passed parser phase
      analyzeCudaLite({} as CudaLiteModule);
      // @ts-expect-error analysis transition requires parsed source proof
      completeAnalysis({}, {} as CudaLiteModule);
      // @ts-expect-error runtime lowering transition requires canonical IR proof
      completeRuntimeLowering({} as SemanticKernelIrModule);
      // @ts-expect-error raw IR has not passed mandatory verifier
      emitSemanticKernelIrWgsl({} as SemanticKernelIrModule);
      const raw = {} as SemanticKernelIrModule;
      // @ts-expect-error raw IR cannot enter semantic type checking
      typeCheckSemanticKernelIr(raw);
      const verified = validateSemanticKernelIr(raw);
      // @ts-expect-error verified IR has not passed mandatory semantic type checking
      emitSemanticKernelIrWgsl(verified);
      // @ts-expect-error verified IR cannot enter WGSL legalization
      legalizeSemanticKernelIrForWgsl(verified);
      const typeChecked = typeCheckSemanticKernelIr(verified);
      // @ts-expect-error type-checked IR has not passed mandatory WGSL legalization
      emitSemanticKernelIrWgsl(typeChecked);
    }
    expect(true).toBe(true);
  });
});
