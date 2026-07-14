import { describe, expect, it } from "vitest";
import {
  analyzeCudaLite,
  canEmitSemanticKernelIrWgsl,
  compileCudaLiteKernel,
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

    const compiled = compileCudaLiteKernel(`__global__ void compiled_contract(float *out) { out[0] = 1.0f; }`);
    expect(compiled.verifiedKernelIr.ir).toBe(compiled.kernelIr);
    expect(compiled.typeCheckedKernelIr.verified).toBe(compiled.verifiedKernelIr);
    expect(compiled.wgslLegalizedKernelIr.typeChecked).toBe(compiled.typeCheckedKernelIr);
  });

  it("rejects duplicate semantic declaration identities", () => {
    const ast = parseCudaLite(`__global__ void duplicate_identity(float *out) { out[0] = 1.0f; }`);
    const semantic = createCudaLiteSemanticModel(analyzeCudaLite(ast));
    const out = semantic.params[0]!;

    expect(() => createSemanticEnvironment([out, out], [])).toThrow(
      `duplicate semantic symbol id '${semanticIdKey(out.id)}'`,
    );
  });

  it("uses the module environment while retaining lexical helper scopes", () => {
    const analysis = analyzeCudaLite(parseCudaLite(`
__device__ int add_local(int value) {
  int result = value;
  {
    int value = 2;
    result += value;
  }
  return result;
}

__global__ void environment_scope(int *out) {
  out[0] = add_local(4);
}
`));
    const semantic = createCudaLiteSemanticModel(analysis);
    const out = semantic.params[0]!;
    const helper = semantic.functions[0]!;
    const moduleHelper = semantic.environment.resolveSymbol("add_local");
    const resultDeclaration = helper.body.find((operation) =>
      operation.kind === "declare" && operation.target.name === "result",
    );
    const innerBlock = helper.body.find((operation) => operation.kind === "block");
    const innerValueDeclaration = innerBlock?.kind === "block"
      ? innerBlock.body.find((operation) => operation.kind === "declare" && operation.target.name === "value")
      : undefined;

    expect(semantic.environment.resolveSymbol("out")).toBe(out);
    expect(semantic.environment.resolveMemorySymbol(semanticMemoryIdFromSymbol(out.id))).toBe(out);
    expect(semantic.environment.resolveFunction("add_local")).toBe(helper);
    expect(semantic.environment.resolveFunctionCandidates("add_local")).toEqual([helper]);
    expect(moduleHelper?.kind).toBe("function");
    expect(resultDeclaration?.kind === "declare" && resultDeclaration.init).toMatchObject({
      kind: "symbol",
      id: helper.params[0]!.id,
    });
    expect(innerValueDeclaration?.kind === "declare" ? innerValueDeclaration.target.id : undefined)
      .not.toBe(helper.params[0]!.id);

    const canonical = lowerSemanticModelToKernelIr(analysis, semantic, { workgroupSize: [1, 1, 1] });
    const store = canonical.operations.find((operation) => operation.kind === "store");
    const callCalleeId = store?.kind === "store" && store.value.kind === "call" && store.value.callee.kind === "symbol"
      ? semanticIdKey(store.value.callee.id)
      : undefined;
    expect(callCalleeId)
      .toBe(moduleHelper === undefined ? undefined : semanticIdKey(moduleHelper.id));
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
      // @ts-expect-error WGSL readiness requires legalized IR proof
      canEmitSemanticKernelIrWgsl(raw);
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
