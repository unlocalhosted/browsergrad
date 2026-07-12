import { describe, expect, it } from "vitest";
import {
  analyzeCudaLite,
  assertValidSemanticKernelIr,
  assertTypeCheckedSemanticKernelIr,
  assertWgslLegalizedSemanticKernelIr,
  createCudaLiteSemanticModel,
  createSemanticEnvironment,
  emitSemanticKernelIrWgsl,
  lowerSemanticCudaRuntime,
  lowerSemanticModelToKernelIr,
  parseCudaLite,
  type CudaLiteModule,
  type SemanticKernelIrModule,
  type VerifiedSemanticKernelIr,
} from "../../src/index.js";
import { semanticIdKey } from "../../src/semantic_ids.js";

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
    assertValidSemanticKernelIr(runtimeLowered);
    assertTypeCheckedSemanticKernelIr(runtimeLowered);
    assertWgslLegalizedSemanticKernelIr(runtimeLowered);
    const store = runtimeLowered.operations.find((operation) => operation.kind === "store");
    expect(store?.kind === "store" ? store.target.baseId : undefined).toBe(out.id);
    const emitted = emitSemanticKernelIrWgsl(runtimeLowered);

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
      // @ts-expect-error raw IR has not passed mandatory verifier
      emitSemanticKernelIrWgsl({} as SemanticKernelIrModule);
      const raw = {} as SemanticKernelIrModule;
      assertValidSemanticKernelIr(raw);
      const verified = raw as VerifiedSemanticKernelIr;
      // @ts-expect-error verified IR has not passed mandatory semantic type checking
      emitSemanticKernelIrWgsl(verified);
      assertTypeCheckedSemanticKernelIr(verified);
      // @ts-expect-error type-checked IR has not passed mandatory WGSL legalization
      emitSemanticKernelIrWgsl(verified);
    }
    expect(true).toBe(true);
  });
});
