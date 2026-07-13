import type { WgslKernelBindingInput } from "@unlocalhosted/browsergrad-kernels";
import type { SemanticKernelIrModule, SemanticKernelIrOperation } from "./semantic_ir.js";
import { collectSemanticPoolAllocations } from "./semantic_ir.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import {
  emitPoolHelper,
  emitRawPoolHelper,
  rawPoolHelperName,
} from "./wgsl_support_helpers.js";

type SemanticPoolAllocation = Extract<SemanticKernelIrOperation, { readonly kind: "pool-allocate" }>;

export interface SemanticWgslPoolResources {
  readonly allocations: readonly SemanticPoolAllocation[];
  readonly devicePoolNames: readonly string[];
}

export function collectSemanticWgslPoolResources(ir: SemanticKernelIrModule): SemanticWgslPoolResources {
  const allocations = collectSemanticPoolAllocations(ir.operations);
  return {
    allocations,
    devicePoolNames: [...new Set(allocations.flatMap((operation) =>
      operation.pool.kind === "device-pool" ? [operation.pool.name] : []
    ))],
  };
}

export function semanticWgslPoolBindingNames(devicePoolNames: readonly string[]): readonly string[] {
  return devicePoolNames.flatMap((name) => [poolDataName(name), poolOffsetName(name)]);
}

export function semanticWgslPoolBindings(
  devicePoolNames: readonly string[],
  firstBinding: number,
): readonly WgslKernelBindingInput[] {
  return devicePoolNames.flatMap((name, index): readonly WgslKernelBindingInput[] => [
    { kind: "storage", name: poolDataName(name), valueType: "u32", access: "read_write", binding: firstBinding + index * 2 },
    { kind: "storage", name: poolOffsetName(name), valueType: "u32", access: "read_write", binding: firstBinding + index * 2 + 1 },
  ]);
}

export function emitSemanticWgslPoolDeclarations(
  devicePoolNames: readonly string[],
  bindingIndex: (name: string) => number,
  nameFor: (name: string) => string,
): readonly string[] {
  return devicePoolNames.flatMap((name) => {
    const dataName = poolDataName(name);
    const offsetName = poolOffsetName(name);
    return [
      `@group(0) @binding(${bindingIndex(dataName)}) var<storage, read_write> ${nameFor(dataName)}: array<u32>;`,
      `@group(0) @binding(${bindingIndex(offsetName)}) var<storage, read_write> ${nameFor(offsetName)}: atomic<u32>;`,
    ];
  });
}

export function emitSemanticWgslPoolHelpers(
  resources: SemanticWgslPoolResources,
  nameFor: (name: string) => string,
): readonly string[] {
  const lines: string[] = [];
  for (const poolName of resources.devicePoolNames) {
    lines.push("", ...emitPoolHelper(poolName, { nameFor }));
  }
  const rawAllocators = new Map<string, { readonly baseName: string; readonly offsetName: string }>();
  for (const allocation of resources.allocations) {
    if (allocation.pool.kind !== "raw-pool") continue;
    const baseName = nameFor(allocation.pool.data.base);
    const offsetName = nameFor(allocation.pool.offset.base);
    rawAllocators.set(rawPoolHelperName(baseName, offsetName), { baseName, offsetName });
  }
  for (const allocator of rawAllocators.values()) lines.push("", ...emitRawPoolHelper(allocator));
  return lines;
}
