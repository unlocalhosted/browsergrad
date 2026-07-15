import type ts from "typescript";

export function runSemanticArchitectureCheck(root?: string): string[];

export function checkWorkspaceImportSpecifier(
  packageName: string,
  file: string,
  specifier: string,
): string[];

export function countPythonCustomConstructors(source: string): number;

export function extractPythonCustomLabels(source: string): string[];

export function extractPythonCustomLabelFields(source: string): {
  readonly name: string[];
  readonly op: string[];
  readonly dynamicName: string[];
  readonly dynamicOp: string[];
};

export function extractPythonDefinitionTokenDigests(
  source: string,
): Record<string, readonly string[]>;

export function checkFrozenGradCompatibilitySources(
  tensorSource: string,
  torchCompatSource: string,
  freeze: Record<string, unknown>,
): string[];

export function validateGradCompatibilityInventory(
  inventory: unknown,
  fixture: unknown,
  freeze: Record<string, unknown>,
  filename?: string,
): string[];

export function validateJitOpaqueOperationInventory(
  inventory: unknown,
  fixture: unknown,
  freeze: Record<string, unknown>,
  filename?: string,
): string[];

export function validateSemanticFreezeManifest(
  root: string,
  manifest: unknown,
): string[];

export function validateSharedSemanticFixtureContracts(
  root: string,
  manifest: unknown,
): string[];

export function extractModuleSpecifiers(
  typescript: typeof ts,
  source: string,
  filename?: string,
): string[];

export function buildAssignmentRequirementUsage(
  root: string,
  profileDirectory?: string,
  profileSuffix?: string,
): unknown;

export function checkFrozenCompilerPointerScalarMemorySource(
  typescript: typeof ts,
  source: string,
  publicBarrelSource: string,
  freeze: Record<string, unknown>,
  filename?: string,
  publicBarrelFilename?: string,
): string[];

export function validateCompilerPointerBehaviorFixture(
  fixture: unknown,
  freeze: Record<string, unknown>,
  filename?: string,
): string[];

export function checkFrozenRuntimeAssignmentRequirementsSource(
  typescript: typeof ts,
  capabilitySource: string,
  typesSource: string,
  freeze: Record<string, unknown>,
  capabilityFilename?: string,
  typesFilename?: string,
): string[];

export function validatePlatformVocabularySnapshot(
  root: string,
  vocabulary: unknown,
  profileIds: readonly string[],
  browserMappings: readonly unknown[],
): string[];

export function checkFrozenCuteSourceNormalizerFiles(
  sources: Readonly<Record<string, string>>,
  freeze: Record<string, unknown>,
): string[];

export function checkFrozenCuteStaticLayoutSource(
  typescript: typeof ts,
  source: string,
  freeze: Record<string, unknown>,
  filename?: string,
): string[];

export function checkFrozenTensorGpuPlanSource(
  typescript: typeof ts,
  source: string,
  freeze: Record<string, unknown>,
  filename?: string,
): string[];
