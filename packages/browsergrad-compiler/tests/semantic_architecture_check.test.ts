import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import ts from "typescript";
import { describe, expect, it } from "vitest";

import {
  checkFrozenCompilerPointerScalarMemorySource,
  checkFrozenGradCompatibilitySources,
  checkFrozenRuntimeAssignmentRequirementsSource,
  checkRuntimeAssignmentResolutionConsumerSources,
  checkFrozenCuteSourceNormalizerFiles,
  checkFrozenCuteStaticLayoutSource,
  checkFrozenTensorGpuPlanSource,
  checkWorkspaceImportSpecifier,
  countPythonCustomConstructors,
  extractModuleSpecifiers,
  extractPythonCustomLabelFields,
  extractPythonCustomLabels,
  extractPythonDefinitionTokenDigests,
  runSemanticArchitectureCheck,
  validateCompilerPointerBehaviorFixture,
  validateGradCompatibilityInventory,
  validateJitOpaqueOperationInventory,
  validateAssignmentRequirementRegistrySource,
  validateGradFrameworkPlatformSupportSource,
  validateImplementationCheckpoint,
  validatePlatformVocabularySnapshot,
  validateProgramCapabilityRegistrySource,
  validateSemanticFreezeManifest,
  validateSemanticArchitectureDeclarationParity,
  validateSharedSemanticFixtureContracts,
} from "../../../scripts/semantic-architecture-check.mjs";

const repoRoot = dirname(dirname(dirname(dirname(fileURLToPath(import.meta.url)))));
const freezeManifest = JSON.parse(
  readFileSync(join(repoRoot, "architecture/semantic-freeze.json"), "utf8"),
) as {
  adapters: Array<{ freeze?: { kind?: string } }>;
};
const fixtureContracts = JSON.parse(
  readFileSync(join(repoRoot, "architecture/semantic-fixture-contracts.json"), "utf8"),
) as {
  contracts: Array<{
    caseIds: string[];
    contentSha256: string;
    excludedRoutingFields: string[];
  }>;
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

  it("rejects architecture implementation/declaration drift", () => {
    const implementation = readFileSync(
      join(repoRoot, "scripts/semantic-architecture-check.mjs"),
      "utf8",
    );
    const declaration = readFileSync(
      join(repoRoot, "scripts/semantic-architecture-check.d.mts"),
      "utf8",
    );
    expect(validateSemanticArchitectureDeclarationParity(
      ts,
      implementation,
      declaration,
    )).toEqual([]);
    expect(validateSemanticArchitectureDeclarationParity(
      ts,
      implementation,
      declaration.replace(
        "  torchCompatLimitedSource: string,\n",
        "",
      ),
    )).toContainEqual(
      expect.stringContaining(
        "declaration signature checkFrozenGradCompatibilitySources changed",
      ),
    );
  });

  it("rejects shared semantic fixture content, coverage, and routing drift", () => {
    expect(validateSharedSemanticFixtureContracts(repoRoot, fixtureContracts)).toEqual([]);

    const wrongHash = structuredClone(fixtureContracts);
    wrongHash.contracts[0]!.contentSha256 = "0".repeat(64);
    expect(validateSharedSemanticFixtureContracts(repoRoot, wrongHash))
      .toContainEqual(expect.stringContaining("content changed"));

    const missingCase = structuredClone(fixtureContracts);
    missingCase.contracts[0]!.caseIds.pop();
    expect(validateSharedSemanticFixtureContracts(repoRoot, missingCase))
      .toContainEqual(expect.stringContaining("case order/coverage changed"));

    const semanticRoutingCollision = structuredClone(fixtureContracts);
    semanticRoutingCollision.contracts[0]!.excludedRoutingFields.push("axes");
    expect(validateSharedSemanticFixtureContracts(repoRoot, semanticRoutingCollision))
      .toContainEqual(expect.stringContaining("contains excluded routing field axes"));
  });

  it("keeps implementation chronology out of the normative requirements checkpoint", () => {
    const source = readFileSync(
      join(repoRoot, "docs/platform/package-requirements-lld.md"),
      "utf8",
    );
    expect(validateImplementationCheckpoint(source)).toEqual([]);
    expect(validateImplementationCheckpoint(
      source.replace(
        "## Purpose",
        `${"historical evidence\n".repeat(181)}\n## Purpose`,
      ),
    )).toContainEqual(expect.stringContaining("move chronology to the implementation ledger"));
    expect(validateImplementationCheckpoint(
      `${source}\n## Implementation Checkpoint — Active duplicate\n`,
    )).toContainEqual(expect.stringContaining("exactly one active implementation checkpoint"));
    expect(validateImplementationCheckpoint(
      source.replace("This checkpoint is informational.", "Current status follows."),
    )).toContainEqual(expect.stringContaining("must state that it is informational"));
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
    expect(
      checkWorkspaceImportSpecifier(
        "@unlocalhosted/browsergrad-kernels",
        "packages/browsergrad-kernels/src/new.ts",
        "@unlocalhosted/browsergrad-semantic-core/host",
      ),
    ).toEqual([
      "packages/browsergrad-kernels/src/new.ts imports @unlocalhosted/browsergrad-semantic-core/host; kernels may import semantic-core schema/layout/kernel/schedule/graph protocols only",
    ]);
    expect(
      checkWorkspaceImportSpecifier(
        "@unlocalhosted/browsergrad-kernels",
        "packages/browsergrad-kernels/src/schedule.ts",
        "@unlocalhosted/browsergrad-semantic-core/schedule",
      ),
    ).toEqual([]);
    expect(
      checkWorkspaceImportSpecifier(
        "@unlocalhosted/browsergrad-kernels",
        "packages/browsergrad-kernels/src/semantic_host_graph.ts",
        "@unlocalhosted/browsergrad-semantic-core/graph",
      ),
    ).toEqual([]);
  });

  it("keeps Node effects out of compiler production source", () => {
    for (const specifier of ["node:child_process", "child_process", "node:fs/promises"]) {
      expect(checkWorkspaceImportSpecifier(
        "@unlocalhosted/browsergrad-compiler",
        "packages/browsergrad-compiler/src/escape.ts",
        specifier,
      )).toEqual([
        `packages/browsergrad-compiler/src/escape.ts imports Node built-in ${specifier}; compiler production source must remain browser-safe`,
      ]);
    }
    expect(checkWorkspaceImportSpecifier(
      "@unlocalhosted/browsergrad-compiler",
      "packages/browsergrad-compiler/scripts/intentional-node.mjs",
      "node:child_process",
    )).toEqual([]);
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

  it("rejects legacy assignment requirement mapping and status widening", () => {
    const capabilityFile = join(repoRoot, "packages/browsergrad-runtime/src/assignment-capabilities.ts");
    const typesFile = join(repoRoot, "packages/browsergrad-runtime/src/assignment-types.ts");
    const capabilitySource = readFileSync(capabilityFile, "utf8");
    const typesSource = readFileSync(typesFile, "utf8");
    const runtimeFreeze = freeze("runtime-assignment-requirements");

    expect(checkFrozenRuntimeAssignmentRequirementsSource(
      ts,
      capabilitySource.replace("export interface BrowserGpuCapabilityInput {", "export interface BrowserGpuCapabilityInput {\n  readonly newBackend?: boolean;"),
      typesSource,
      runtimeFreeze,
    )).toContainEqual(expect.stringContaining("BrowserGpuCapabilityInput fields changed"));
    expect(checkFrozenRuntimeAssignmentRequirementsSource(
      ts,
      capabilitySource.replace("input.webgpu && input.cudaLiteCompiler", "input.cudaLiteCompiler"),
      typesSource,
      runtimeFreeze,
    )).toContainEqual(expect.stringContaining("browserGpuCapabilities mappings changed"));
    expect(checkFrozenRuntimeAssignmentRequirementsSource(
      ts,
      capabilitySource.replace("  return uniqueSorted(capabilities);", '  capabilities.push("new-route");\n  return uniqueSorted(capabilities);'),
      typesSource,
      runtimeFreeze,
    )).toContainEqual(expect.stringContaining("push calls"));
    expect(checkFrozenRuntimeAssignmentRequirementsSource(
      ts,
      capabilitySource,
      typesSource.replace('export type AssignmentCapabilityMode = "browser" | "simulated" | "external";', 'export type AssignmentCapabilityMode = "browser" | "simulated" | "external" | "native";'),
      runtimeFreeze,
    )).toContainEqual(expect.stringContaining("AssignmentCapabilityMode values changed"));
  });

  it("rejects generated assignment requirement registry drift", () => {
    const registrySource = readFileSync(
      join(
        repoRoot,
        "packages/browsergrad-runtime/src/assignment-requirement-registry.generated.ts",
      ),
      "utf8",
    );
    const vocabulary = JSON.parse(
      readFileSync(
        join(repoRoot, "architecture/platform-vocabulary.json"),
        "utf8",
      ),
    ) as { legacyAssignmentRequirements: unknown[] };

    expect(
      validateAssignmentRequirementRegistrySource(
        registrySource,
        vocabulary,
      ),
    ).toEqual([]);
    expect(
      validateAssignmentRequirementRegistrySource(
        registrySource.replace(
          '"requirementId": "webgpu"',
          '"requirementId": "webgpu-mutated"',
        ),
        vocabulary,
      ),
    ).toContainEqual(expect.stringContaining("registry is stale"));

    const duplicated = structuredClone(vocabulary);
    duplicated.legacyAssignmentRequirements.push(
      duplicated.legacyAssignmentRequirements[0],
    );
    expect(
      validateAssignmentRequirementRegistrySource(registrySource, duplicated),
    ).toContainEqual(expect.stringContaining("duplicate assignment requirement"));
  });

  it("rejects generated program capability registry drift", () => {
    const registrySource = readFileSync(
      join(
        repoRoot,
        "packages/browsergrad-runtime/src/program-capability-registry.generated.ts",
      ),
      "utf8",
    );
    const vocabulary = JSON.parse(
      readFileSync(
        join(repoRoot, "architecture/platform-vocabulary.json"),
        "utf8",
      ),
    ) as {
      semanticCapabilities: unknown[];
      backends: unknown[];
    };

    expect(
      validateProgramCapabilityRegistrySource(registrySource, vocabulary),
    ).toEqual([]);
    expect(
      validateProgramCapabilityRegistrySource(
        registrySource.replace(
          '"backendId": "browsergrad.kernels.webgpu"',
          '"backendId": "browsergrad.kernels.mutated"',
        ),
        vocabulary,
      ),
    ).toContainEqual(expect.stringContaining("registry is stale"));

    const duplicated = structuredClone(vocabulary);
    duplicated.backends.push(duplicated.backends[0]);
    expect(
      validateProgramCapabilityRegistrySource(registrySource, duplicated),
    ).toContainEqual(
      expect.stringContaining("duplicate platform vocabulary ID"),
    );
  });

  it("rejects generated Grad framework platform support drift", () => {
    const registrySource = readFileSync(
      join(
        repoRoot,
        "packages/browsergrad-grad/src/framework-platform-support.generated.ts",
      ),
      "utf8",
    );
    const inventory = JSON.parse(
      readFileSync(
        join(repoRoot, "architecture/grad-compatibility-inventory.json"),
        "utf8",
      ),
    ) as { behaviors: unknown[] };

    expect(
      validateGradFrameworkPlatformSupportSource(registrySource, inventory),
    ).toEqual([]);
    expect(
      validateGradFrameworkPlatformSupportSource(
        registrySource.replace(
          '"operationId": "browsergrad.grad.view.expand.v1"',
          '"operationId": "browsergrad.grad.view.mutated.v1"',
        ),
        inventory,
      ),
    ).toContainEqual(expect.stringContaining("is stale"));

    const duplicated = structuredClone(inventory);
    duplicated.behaviors.push(duplicated.behaviors[0]);
    expect(
      validateGradFrameworkPlatformSupportSource(registrySource, duplicated),
    ).toContainEqual(expect.stringContaining("is duplicated"));
  });

  it("rejects runtime readiness consumers that drop resolution records", () => {
    const runtimeFreeze = freeze("runtime-assignment-requirements");
    const typesFile = join(
      repoRoot,
      "packages/browsergrad-runtime/src/assignment-types.ts",
    );
    const capabilityFile = join(
      repoRoot,
      "packages/browsergrad-runtime/src/assignment-capabilities.ts",
    );
    const typesSource = readFileSync(typesFile, "utf8");
    expect(checkFrozenRuntimeAssignmentRequirementsSource(
      ts,
      readFileSync(capabilityFile, "utf8"),
      typesSource.replace(
        "readonly requirementResolutions?: readonly AssignmentRequirementResolution[];",
        "readonly requirementResolutions?: readonly string[];",
      ),
      runtimeFreeze,
    )).toContainEqual(
      expect.stringContaining(
        "requirementResolutions must remain one optional readonly",
      ),
    );

    const consumers = Object.fromEntries(
      (runtimeFreeze.resolutionConsumers as Array<{
        file: string;
      }>).map(({ file }) => [
        file,
        readFileSync(join(repoRoot, file), "utf8"),
      ]),
    );
    consumers["packages/browsergrad-runtime/src/assignment-run-plan.ts"] =
      consumers["packages/browsergrad-runtime/src/assignment-run-plan.ts"]!
        .replace(
          "environment: AssignmentReadinessEnvironment",
          "environment: AssignmentCapabilityEnvironment",
        );
    expect(checkRuntimeAssignmentResolutionConsumerSources(
      ts,
      consumers,
      runtimeFreeze,
    )).toContainEqual(
      expect.stringContaining(
        "createAssignmentRunPlan must consume AssignmentReadinessEnvironment directly",
      ),
    );
  });

  it("rejects Grad dtype/view source drift", () => {
    const tensorSource = readFileSync(join(repoRoot, "packages/browsergrad-grad/src/python/tensor.py"), "utf8");
    const torchCompatSource = readFileSync(join(repoRoot, "packages/browsergrad-grad/src/python/_torch_compat_real.py"), "utf8");
    const torchCompatLimitedSource = readFileSync(join(repoRoot, "packages/browsergrad-grad/src/python/_torch_compat_limited.py"), "utf8");
    const gradFreeze = freeze("grad-view-bf16");

    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        'if isinstance(spec, str) and spec in ("bfloat16", "bf16"):',
        'if isinstance(spec, str) and spec == "bf16":',
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("tensor.py:_resolve_dtype changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "        if spec in aliases:",
        "        if True:",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("tensor.py:_resolve_dtype changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "    if resolved.name not in supported:",
        "    if False:",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("tensor.py:_resolve_dtype changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        '"uint8", "uint16", "uint32", "uint64", "bool",',
        '"uint8", "uint16", "uint32", "complex64", "bool",',
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Grad eager storage dtypes changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource,
      torchCompatSource.replace('torch_mod.bfloat16 = "bfloat16"', 'torch_mod.bfloat16 = "float32"'),
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Grad torch dtype tokens changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "        if self.data.flags.c_contiguous:",
        "        if True:",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Tensor.contiguous changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        '"""Return self for C-contiguous storage, otherwise an owning C-order copy."""',
        '"""Equivalent compatibility wording."""',
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toEqual([]);
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace("    def contiguous(self)", "    @staticmethod\n    def contiguous(self)"),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Tensor.contiguous changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "return Tensor(self.data, dtype=self.data.dtype, requires_grad=False)",
        "return Tensor(self.data.copy(), dtype=self.data.dtype, requires_grad=False)",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Tensor.detach changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "writeable=reshaped.flags.writeable,",
        "writeable=False,",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("tensor.py:_expand changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "0 if source_dim == 1 and target_dim != 1 else stride",
        "stride",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("tensor.py:_expand changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "and out.dtype in _VARIADIC_FLOATING_DTYPES",
        "and False",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Tensor.to changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        '    def cpu(self) -> "Tensor":\n        return self',
        '    def cpu(self) -> "Tensor":\n        return self.detach()',
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Tensor.cpu changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "CPU/Pyodide-backed and no CUDA transfer occurred",
        "CPU/Pyodide-backed",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Tensor.cuda changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        'return self.data.astype(target_dtype, order="K", copy=True)',
        'return self.data.astype(target_dtype, order="K", copy=False)',
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Tensor._numpy_snapshot changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "if copy is False:",
        "if False:",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("Tensor.__array__ changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace(
        "if arr.dtype.name not in supported:",
        "if False:",
      ),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("tensor.py:from_numpy changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource.replace("if _GRAD_ENABLED and any(p.requires_grad for p in parents):", "if True and any(p.requires_grad for p in parents):"),
      torchCompatSource,
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("tensor.py:_build_ctx changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource,
      torchCompatSource.replace(
        'if device != "cpu":',
        "if False:",
      ),
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("install_real._tensor_factory changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource,
      torchCompatSource.replace(
        "return data.data, True",
        "return data.data, False",
      ),
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("install_real._tensor_factory_source changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource,
      torchCompatSource.replace(
        'order="K", copy=True',
        'order="K", copy=False',
      ),
      torchCompatLimitedSource,
      gradFreeze,
    )).toContainEqual(expect.stringContaining("install_real._tensor_factory changed"));
    expect(checkFrozenGradCompatibilitySources(
      tensorSource,
      torchCompatSource,
      torchCompatLimitedSource.replace(
        'if device_spec != "cpu":',
        "if False:",
      ),
      gradFreeze,
    )).toContainEqual(expect.stringContaining("install_limited._module_to_shim changed"));
  });

  it("rejects Grad compatibility inventory and behavior-fixture drift", () => {
    const inventory = JSON.parse(readFileSync(join(repoRoot, "architecture/grad-compatibility-inventory.json"), "utf8")) as {
      dtypeResolution: {
        aliases: Record<string, string>;
        unsupportedDtypes: Record<string, string>;
      };
      behaviors: Array<Record<string, unknown>>;
    };
    const fixture = JSON.parse(readFileSync(join(repoRoot, "packages/browsergrad-grad/tests-integration/fixtures/grad-view-bf16.v0.json"), "utf8")) as {
      cases: Array<Record<string, unknown>>;
    };
    const gradFreeze = freeze("grad-view-bf16");

    const widened = structuredClone(inventory);
    widened.behaviors[0]!.backendHint = "numpy";
    expect(validateGradCompatibilityInventory(widened, fixture, gradFreeze))
      .toContainEqual(expect.stringContaining("behaviors[0] keys changed"));

    const changedAlias = structuredClone(inventory);
    changedAlias.dtypeResolution.aliases.float16 = "float32";
    expect(validateGradCompatibilityInventory(changedAlias, fixture, gradFreeze))
      .toContainEqual(expect.stringContaining("differs from the frozen source alias map"));
    const changedUnsupported = structuredClone(inventory);
    changedUnsupported.dtypeResolution.unsupportedDtypes.bf16 = "float32";
    expect(validateGradCompatibilityInventory(changedUnsupported, fixture, gradFreeze))
      .toContainEqual(expect.stringContaining("unsupportedDtypes changed"));
    const changedFixture = structuredClone(fixture);
    changedFixture.cases.splice(3, 1);
    expect(validateGradCompatibilityInventory(inventory, changedFixture, gradFreeze))
      .toContainEqual(expect.stringContaining("behavior fixture IDs changed"));
  });

  it("rejects static capability outcomes and unregistered assignment requirements", () => {
    const vocabulary = JSON.parse(readFileSync(join(repoRoot, "architecture/platform-vocabulary.json"), "utf8")) as {
      semanticCapabilities: Array<Record<string, unknown>>;
      legacyAssignmentRequirements: Array<Record<string, unknown>>;
    };
    const usage = JSON.parse(readFileSync(join(repoRoot, "architecture/assignment-requirement-usage.generated.json"), "utf8")) as {
      requirements: Array<{ requirementId: string }>;
    };
    const runtimeFreeze = freeze("runtime-assignment-requirements") as {
      browserMappings: readonly unknown[];
    };
    const profileIds = usage.requirements.map((entry) => entry.requirementId);

    const capabilityOutcome = structuredClone(vocabulary);
    capabilityOutcome.semanticCapabilities[0]!.outcome = "passed";
    expect(validatePlatformVocabularySnapshot(repoRoot, capabilityOutcome, profileIds, runtimeFreeze.browserMappings))
      .toContainEqual(expect.stringContaining("contains runtime/evidence state"));

    const disguisedOutcome = structuredClone(vocabulary);
    disguisedOutcome.semanticCapabilities[0]!.support = "passed";
    expect(validatePlatformVocabularySnapshot(repoRoot, disguisedOutcome, profileIds, runtimeFreeze.browserMappings))
      .toContainEqual(expect.stringContaining("record keys changed"));

    const danglingCapability = structuredClone(vocabulary);
    const semanticRequirement = danglingCapability.legacyAssignmentRequirements.find((entry) => entry.kind === "semantic-feature");
    if (semanticRequirement === undefined) throw new Error("missing semantic-feature requirement fixture");
    semanticRequirement.capabilityId = "browsergrad.missing.capability";
    expect(validatePlatformVocabularySnapshot(repoRoot, danglingCapability, profileIds, runtimeFreeze.browserMappings))
      .toContainEqual(expect.stringContaining("links unknown capability"));

    const oracleCapability = structuredClone(vocabulary);
    const oracleRequirement = oracleCapability.legacyAssignmentRequirements.find((entry) => entry.kind === "oracle");
    if (oracleRequirement === undefined) throw new Error("missing oracle requirement fixture");
    oracleRequirement.capabilityId = "browsergrad.layout.index-map";
    expect(validatePlatformVocabularySnapshot(repoRoot, oracleCapability, profileIds, runtimeFreeze.browserMappings))
      .toContainEqual(expect.stringContaining("only when kind is semantic-feature"));

    const missingRequirement = structuredClone(vocabulary);
    missingRequirement.legacyAssignmentRequirements.pop();
    expect(validatePlatformVocabularySnapshot(repoRoot, missingRequirement, profileIds, runtimeFreeze.browserMappings))
      .toContainEqual(expect.stringContaining("registered legacy assignment requirements changed"));

    const widenedPolicy = structuredClone(vocabulary) as typeof vocabulary & {
      identifierPolicies: { canonical: string };
    };
    widenedPolicy.identifierPolicies.canonical = ".*";
    expect(validatePlatformVocabularySnapshot(repoRoot, widenedPolicy, profileIds, runtimeFreeze.browserMappings))
      .toContainEqual(expect.stringContaining("canonical identifier policy changed"));
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

  it("rejects CuTe source-normalizer drift", () => {
    const normalizerFreeze = freeze("cute-source-normalizers");
    const files = normalizerFreeze.files as Record<string, string>;
    const sources = Object.fromEntries(
      Object.keys(files).map((file) => [file, readFileSync(join(repoRoot, file), "utf8")]),
    );
    expect(checkFrozenCuteSourceNormalizerFiles(sources, normalizerFreeze)).toEqual([]);
    const target = "scripts/cuda-lite-source-normalizer-cute-la.mjs";
    expect(
      checkFrozenCuteSourceNormalizerFiles(
        { ...sources, [target]: `${sources[target]}\nexport function normalizeCuteNewMotif(source) { return source; }\n` },
        normalizerFreeze,
      ),
    ).toContainEqual(expect.stringContaining(`${target} CuTe source normalizer changed`));
  });

  it("rejects missing or newly split CuTe source-normalizer files", () => {
    const normalizerFreeze = freeze("cute-source-normalizers");
    const files = normalizerFreeze.files as Record<string, string>;
    const sources = Object.fromEntries(
      Object.keys(files).map((file) => [file, readFileSync(join(repoRoot, file), "utf8")]),
    );
    const missing = { ...sources };
    delete missing[Object.keys(files)[0] as string];
    expect(checkFrozenCuteSourceNormalizerFiles(missing, normalizerFreeze))
      .toContainEqual(expect.stringContaining("CuTe source-normalizer files changed"));
    expect(checkFrozenCuteSourceNormalizerFiles(
      { ...sources, "scripts/cuda-lite-source-normalizer-cute-new.mjs": "export const replacement = true;" },
      normalizerFreeze,
    )).toContainEqual(expect.stringContaining("CuTe source-normalizer files changed"));
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

  it("separates CUSTOM name/op dispatch fields and excludes non-CUSTOM arg strings", () => {
    const source = `
      helper = UOp(op=OP_CMP, arg={"op": "gt"})
      reduced = UOp(op=OP_REDUCE, arg={"op": "sum"})
      name_node = UOp(op=OP_CUSTOM, arg={"name": "abs"})
      arg = {"op": "flash_attention"}
      op_node = UOp(op=OP_CUSTOM, arg=arg)
      dynamic_node = UOp(op=OP_CUSTOM, arg={"name": op_name})
    `;
    expect(extractPythonCustomLabels(source)).toEqual(["abs", "flash_attention"]);
    expect(extractPythonCustomLabelFields(source)).toEqual({
      name: ["abs"],
      op: ["flash_attention"],
      dynamicName: ["op_name"],
      dynamicOp: [],
    });
  });

  it("fingerprints same-count CUSTOM relabels and constructor decision changes", () => {
    const baseline = `
def op(x):
    return UOp(op=OP_CUSTOM, inputs=(x,), shape=x.shape, dtype=x.dtype,
               arg={"fn": callback, "name": "abs"})
`;
    const relabeled = baseline.replace('"abs"', '"sin"');
    const changedDtype = baseline.replace("dtype=x.dtype", 'dtype="float32"');
    expect(extractPythonDefinitionTokenDigests(relabeled).op)
      .not.toEqual(extractPythonDefinitionTokenDigests(baseline).op);
    expect(extractPythonDefinitionTokenDigests(changedDtype).op)
      .not.toEqual(extractPythonDefinitionTokenDigests(baseline).op);
  });

  it("rejects JIT opaque-operation inventory widening, missing coverage, and field drift", () => {
    const inventory = JSON.parse(readFileSync(join(repoRoot, "architecture/jit-opaque-operation-inventory.json"), "utf8")) as {
      operations: Array<Record<string, unknown>>;
      constructorSites: Array<Record<string, unknown>>;
    };
    const fixture = JSON.parse(readFileSync(join(repoRoot, "packages/browsergrad-jit/tests-integration/fixtures/jit-opaque-operation.v0.json"), "utf8"));
    const jitFreeze = freeze("jit-op-custom");

    const widened = structuredClone(inventory);
    widened.operations[0]!.backendHint = "webgpu";
    expect(validateJitOpaqueOperationInventory(widened, fixture, jitFreeze))
      .toContainEqual(expect.stringContaining("operations[0] keys changed"));

    const missing = structuredClone(inventory);
    missing.operations.pop();
    expect(validateJitOpaqueOperationInventory(missing, fixture, jitFreeze))
      .toContainEqual(expect.stringContaining("opaque-operation IDs changed"));

    const changedField = structuredClone(inventory);
    changedField.operations[0]!.labelField =
      changedField.operations[0]!.labelField === "op" ? "name" : "op";
    expect(validateJitOpaqueOperationInventory(changedField, fixture, jitFreeze))
      .toContainEqual(expect.stringContaining("labelField disagrees"));

    const extraOperationIdentity = structuredClone(inventory);
    const customKernelSite = extraOperationIdentity.constructorSites.find(
      (site) => site.id === "custom-kernel.builder",
    );
    if (
      customKernelSite === undefined
      || !Array.isArray(customKernelSite.operationIds)
    ) {
      throw new Error("missing custom-kernel constructor fixture");
    }
    customKernelSite.operationIds.push("jit.custom.unreviewed-kernel.v0");
    expect(validateJitOpaqueOperationInventory(extraOperationIdentity, fixture, jitFreeze))
      .toContainEqual(expect.stringContaining("constructor-site operation coverage changed"));

    const frozenConstructorCount = Object.values(
      jitFreeze.constructorCounts as Record<string, number>,
    ).reduce((total, count) => total + count, 0);

    const missingConstructorSite = structuredClone(inventory);
    missingConstructorSite.constructorSites.pop();
    expect(validateJitOpaqueOperationInventory(missingConstructorSite, fixture, jitFreeze)).toContainEqual(
      expect.stringContaining(`${frozenConstructorCount} exact frozen CUSTOM constructor calls`),
    );

    const groupedCalls = structuredClone(inventory);
    groupedCalls.constructorSites[0]!.constructorCount = 2;
    expect(validateJitOpaqueOperationInventory(groupedCalls, fixture, jitFreeze))
      .toContainEqual(expect.stringContaining("constructorCount must be exactly 1"));

    const collapsedPlanDecision = structuredClone(inventory) as typeof inventory & {
      executionContext: Record<string, unknown>;
    };
    delete collapsedPlanDecision.executionContext.tensorGpuPlanExecution;
    expect(validateJitOpaqueOperationInventory(collapsedPlanDecision, fixture, jitFreeze))
      .toContainEqual(expect.stringContaining("executionContext keys changed"));

    const missingRealizedDtype = structuredClone(inventory);
    delete missingRealizedDtype.operations[0]!.realizedDtypeRule;
    expect(validateJitOpaqueOperationInventory(missingRealizedDtype, fixture, jitFreeze))
      .toContainEqual(expect.stringContaining("operations[0] keys changed"));

    const missingReachability = structuredClone(inventory);
    missingReachability.operations[0]!.constructorReachability = "";
    expect(validateJitOpaqueOperationInventory(missingReachability, fixture, jitFreeze))
      .toContainEqual(expect.stringContaining("constructorReachability must be a non-empty string"));
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

    const removedPartitionMutated = structuredClone(freezeManifest) as {
      adapters: Array<{
        id?: string;
        freeze?: { removedUnsupportedSurfaceOperationIds?: string[] };
      }>;
    };
    const removedPartitionAdapter = removedPartitionMutated.adapters.find(
      (entry) => entry.id === "jit.core-custom-ops.v0",
    );
    if (removedPartitionAdapter?.freeze !== undefined) {
      delete removedPartitionAdapter.freeze.removedUnsupportedSurfaceOperationIds;
    }
    expect(validateSemanticFreezeManifest(repoRoot, removedPartitionMutated))
      .toContainEqual(
        expect.stringContaining("removedUnsupportedSurfaceOperationIds"),
      );

    const overlappingPartitionMutated = structuredClone(freezeManifest) as {
      adapters: Array<{
        id?: string;
        freeze?: { removedUnsupportedSurfaceOperationIds?: string[] };
      }>;
    };
    const overlappingPartitionAdapter = overlappingPartitionMutated.adapters.find(
      (entry) => entry.id === "jit.core-custom-ops.v0",
    );
    if (overlappingPartitionAdapter?.freeze !== undefined) {
      overlappingPartitionAdapter.freeze.removedUnsupportedSurfaceOperationIds = [
        "jit.custom.user.v0",
      ];
    }
    expect(validateSemanticFreezeManifest(repoRoot, overlappingPartitionMutated))
      .toContainEqual(expect.stringContaining("remains current opaque"));

    const runtimeMutated = structuredClone(freezeManifest) as {
      adapters: Array<{ id?: string; freeze?: { kind?: string } }>;
    };
    const runtimeAdapter = runtimeMutated.adapters.find((entry) => entry.id === "runtime.generic-backend-labels.v0");
    if (runtimeAdapter !== undefined) delete runtimeAdapter.freeze;
    expect(validateSemanticFreezeManifest(repoRoot, runtimeMutated))
      .toContainEqual(expect.stringContaining("required freeze runtime.generic-backend-labels.v0"));

    const gradMutated = structuredClone(freezeManifest) as {
      adapters: Array<{ id?: string; freeze?: { kind?: string } }>;
    };
    const gradAdapter = gradMutated.adapters.find((entry) => entry.id === "grad.view-bf16-compat.v0");
    if (gradAdapter !== undefined) delete gradAdapter.freeze;
    expect(validateSemanticFreezeManifest(repoRoot, gradMutated))
      .toContainEqual(expect.stringContaining("required freeze grad.view-bf16-compat.v0"));
  });
});
