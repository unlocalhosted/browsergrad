import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import ts from "typescript";
import { describe, expect, it } from "vitest";

import {
  checkFrozenCompilerPointerScalarMemorySource,
  checkFrozenCuteStaticLayoutSource,
  checkFrozenTensorGpuPlanSource,
  checkWorkspaceImportSpecifier,
  countPythonCustomConstructors,
  extractModuleSpecifiers,
  extractPythonCustomLabels,
  runSemanticArchitectureCheck,
  validateCompilerPointerBehaviorFixture,
  validateSemanticFreezeManifest,
} from "../../../scripts/semantic-architecture-check.mjs";

const repoRoot = dirname(dirname(dirname(dirname(fileURLToPath(import.meta.url)))));
const freezeManifest = JSON.parse(
  readFileSync(join(repoRoot, "architecture/semantic-freeze.json"), "utf8"),
) as {
  adapters: Array<{ freeze?: { kind?: string } }>;
};

function freeze(kind: string): Record<string, unknown> {
  const value = freezeManifest.adapters.find((adapter) => adapter.freeze?.kind === kind)?.freeze;
  if (value === undefined) throw new Error(`missing ${kind} freeze fixture`);
  return value as Record<string, unknown>;
}

describe("semantic architecture guardrails", () => {
  it("accepts the repository baseline", () => {
    expect(runSemanticArchitectureCheck(repoRoot)).toEqual([]);
  });

  it("rejects cross-package implementation imports", () => {
    expect(
      checkWorkspaceImportSpecifier(
        "@unlocalhosted/browsergrad-kernels",
        "packages/browsergrad-kernels/src/new.ts",
        "@unlocalhosted/browsergrad-compiler/src/semantic_ir",
      ),
    ).toEqual([
      "packages/browsergrad-kernels/src/new.ts deep-imports implementation path @unlocalhosted/browsergrad-compiler/src/semantic_ir",
      "packages/browsergrad-kernels/src/new.ts imports compiler from kernels",
    ]);
  });

  it("rejects a new TensorGpuPlan operation", () => {
    const filename = join(repoRoot, "packages/browsergrad-kernels/src/tensor_plan.ts");
    const source = readFileSync(filename, "utf8").replace(
      'export type TensorPlanOp =\n  | "BUFFER"',
      'export type TensorPlanOp =\n  | "NEW_VIEW_OP"\n  | "BUFFER"',
    );
    expect(checkFrozenTensorGpuPlanSource(ts, source, freeze("tensor-gpu-plan")))
      .toContainEqual(expect.stringContaining("TensorPlanOp operations changed"));
  });

  it("rejects compiler pointer/scalar schema widening and readonly loss", () => {
    const definitionFile = join(repoRoot, "packages/browsergrad-compiler/src/semantic_ir_types.ts");
    const publicBarrelFile = join(repoRoot, "packages/browsergrad-compiler/src/index.ts");
    const baseline = readFileSync(definitionFile, "utf8");
    const publicBarrel = readFileSync(publicBarrelFile, "utf8");
    const pointerFreeze = freeze("compiler-pointer-scalar-memory");

    expect(checkFrozenCompilerPointerScalarMemorySource(
      ts,
      baseline.replace('  | "unknown";', '  | "unknown"\n  | "new-space";'),
      publicBarrel,
      pointerFreeze,
    )).toContainEqual(expect.stringContaining("SemanticAddressSpace values changed"));
    expect(checkFrozenCompilerPointerScalarMemorySource(
      ts,
      baseline.replace("export interface SemanticPointerAlias {", "export interface SemanticPointerAlias {\n  readonly sourceShapedOffset?: number;"),
      publicBarrel,
      pointerFreeze,
    )).toContainEqual(expect.stringContaining("SemanticPointerAlias changed"));
    expect(checkFrozenCompilerPointerScalarMemorySource(
      ts,
      baseline.replace("  readonly pointerRoot?: SemanticMemoryId;", "  pointerRoot?: SemanticMemoryId;"),
      publicBarrel,
      pointerFreeze,
    )).toContainEqual(expect.stringContaining("pointerRoot must remain readonly"));
    expect(checkFrozenCompilerPointerScalarMemorySource(
      ts,
      baseline.replace('readonly kind: "pointer-rebind"; readonly target:', 'readonly kind: "pointer-rebind"; readonly hidden?: number; readonly target:'),
      publicBarrel,
      pointerFreeze,
    )).toContainEqual(expect.stringContaining("operation pointer-rebind changed"));
  });

  it("rejects compiler pointer/scalar public-export and behavior-fixture drift", () => {
    const definitionFile = join(repoRoot, "packages/browsergrad-compiler/src/semantic_ir_types.ts");
    const publicBarrelFile = join(repoRoot, "packages/browsergrad-compiler/src/index.ts");
    const fixtureFile = join(repoRoot, "packages/browsergrad-compiler/tests/fixtures/pointer-scalar-memory.v0.json");
    const pointerFreeze = freeze("compiler-pointer-scalar-memory");
    expect(checkFrozenCompilerPointerScalarMemorySource(
      ts,
      readFileSync(definitionFile, "utf8"),
      readFileSync(publicBarrelFile, "utf8").replace("  type SemanticMemoryRef,\n", ""),
      pointerFreeze,
    )).toContainEqual(expect.stringContaining("public export SemanticMemoryRef is missing"));

    const fixture = JSON.parse(readFileSync(fixtureFile, "utf8")) as { cases: unknown[] };
    fixture.cases.pop();
    expect(validateCompilerPointerBehaviorFixture(fixture, pointerFreeze))
      .toContainEqual(expect.stringContaining("behavior fixture IDs changed"));
  });

  it("rejects union widening, inheritance, and readonly loss", () => {
    const filename = join(repoRoot, "packages/browsergrad-kernels/src/tensor_plan.ts");
    const baseline = readFileSync(filename, "utf8");
    const planFreeze = freeze("tensor-gpu-plan");
    expect(
      checkFrozenTensorGpuPlanSource(
        ts,
        baseline.replace('export type TensorPlanOp =\n  | "BUFFER"', 'export type TensorPlanOp =\n  | string\n  | "BUFFER"'),
        planFreeze,
      ),
    ).toContainEqual(expect.stringContaining("closed string-literal union"));
    expect(
      checkFrozenTensorGpuPlanSource(
        ts,
        baseline.replace("export interface TensorPlanStep {", "interface HiddenSemantics { hidden: string }\nexport interface TensorPlanStep extends HiddenSemantics {"),
        planFreeze,
      ),
    ).toContainEqual(expect.stringContaining("must not extend"));
    expect(
      checkFrozenTensorGpuPlanSource(
        ts,
        baseline.replace("  readonly step: number;", "  step: number;"),
        planFreeze,
      ),
    ).toContainEqual(expect.stringContaining("must remain readonly"));
  });

  it("rejects a new static CuTe query", () => {
    const filename = join(repoRoot, "packages/browsergrad-compiler/src/cute_static_layout.ts");
    const source = readFileSync(filename, "utf8").replace(
      '"size" | "rank" | "cosize"',
      '"size" | "rank" | "cosize" | "depth"',
    );
    expect(checkFrozenCuteStaticLayoutSource(ts, source, freeze("cute-static-layout")))
      .toContainEqual(expect.stringContaining("cute_static_layout queries changed"));
  });

  it("rejects CuTe surface widening", () => {
    const filename = join(repoRoot, "packages/browsergrad-compiler/src/cute_static_layout.ts");
    const baseline = readFileSync(filename, "utf8");
    const cuteFreeze = freeze("cute-static-layout");
    expect(
      checkFrozenCuteStaticLayoutSource(
        ts,
        baseline.replace('"size" | "rank" | "cosize"', 'string | "size" | "rank" | "cosize"'),
        cuteFreeze,
      ),
    ).toContainEqual(expect.stringContaining("closed string-literal union"));
    expect(
      checkFrozenCuteStaticLayoutSource(
        ts,
        `${baseline}\nexport const NEW_CUTE_HANDLER = true;\n`,
        cuteFreeze,
      ),
    ).toContainEqual(expect.stringContaining("exports changed"));
  });

  it("counts executable custom constructors but ignores comments and strings", () => {
    const source = `
      """UOp(op=OP_CUSTOM, arg={"name": "documentation"})"""
      # UOp(op=OP_CUSTOM)
      node = UOp(op=OP_CUSTOM, inputs=(), shape=(), dtype="float32")
    `;
    expect(countPythonCustomConstructors(source)).toBe(1);
  });

  it("detects positional and aliased custom constructors", () => {
    expect(countPythonCustomConstructors("node = UOp(OP_CUSTOM, (), (), 'float32')")).toBe(1);
    expect(countPythonCustomConstructors("Custom = OP_CUSTOM\nnode = UOp(op=Custom, inputs=())")).toBe(1);
  });

  it("does not infer custom labels from comments or docstrings", () => {
    const source = `
      """UOp(op=OP_CUSTOM, arg={"name": "docs_only"})"""
      # {"op": "comment_only"}
      node = UOp(op=OP_CUSTOM, arg={"name": "real_label"})
    `;
    expect(extractPythonCustomLabels(source)).toEqual(["real_label"]);
  });

  it("discovers dynamic, require, and import-equals dependencies", () => {
    const source = `
      import x = require("pkg-a");
      const y = require("pkg-b");
      const z = import("pkg-c");
    `;
    expect(extractModuleSpecifiers(ts, source).sort()).toEqual(["pkg-a", "pkg-b", "pkg-c"]);
  });

  it("requires all baseline freezes and their accepted decision", () => {
    const mutated = structuredClone(freezeManifest) as {
      adapters: Array<{ id?: string; freeze?: { kind?: string } }>;
      policy?: unknown;
    };
    const adapter = mutated.adapters.find((entry) => entry.id === "jit.core-custom-ops.v0");
    if (adapter !== undefined) delete adapter.freeze;
    expect(validateSemanticFreezeManifest(repoRoot, mutated))
      .toContainEqual(expect.stringContaining("required freeze jit.core-custom-ops.v0"));
  });
});
