import type ts from "typescript";

export function runSemanticArchitectureCheck(root?: string): string[];

export function checkWorkspaceImportSpecifier(
  packageName: string,
  file: string,
  specifier: string,
): string[];

export function countPythonCustomConstructors(source: string): number;

export function extractPythonCustomLabels(source: string): string[];

export function validateSemanticFreezeManifest(
  root: string,
  manifest: unknown,
): string[];

export function extractModuleSpecifiers(
  typescript: typeof ts,
  source: string,
  filename?: string,
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
