import type { CudaLiteTextureDescriptor } from "./types.js";
import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import {
  isSemanticKernelIrOperation,
  semanticExpressionChildren,
  semanticOperationExpressions,
} from "./semantic_ir_walk.js";
import { safeWgslIdentifier } from "./wgsl_names.js";

export interface SemanticTextureDescriptorOptions {
  readonly f16Mode?: "native" | "f32";
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
  readonly textureDescriptors?: Readonly<Record<string, CudaLiteTextureDescriptor>>;
}

export interface SemanticTextureDescriptorSignature {
  readonly key: string;
  readonly descriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>;
}

export type SemanticTextureDescriptorSpecializations = ReadonlyMap<string, ReadonlyMap<string, SemanticTextureDescriptorSignature>>;

export interface SemanticTextureDescriptorHelper {
  readonly textureName: string;
  readonly descriptor: CudaLiteTextureDescriptor;
}

export function collectSemanticTextureDescriptorSpecializations(
  ir: SemanticKernelIrModule,
  options: SemanticTextureDescriptorOptions,
): SemanticTextureDescriptorSpecializations {
  if (options.textureDescriptors === undefined) return new Map();
  const out = new Map<string, Map<string, SemanticTextureDescriptorSignature>>();
  let changed = true;
  while (changed) {
    changed = false;
    changed = collectSemanticTextureDescriptorSpecializationsFromOperations(ir.operations, options.textureDescriptors, ir, out) || changed;
    for (const fn of ir.functions) {
      for (const signature of out.get(fn.name)?.values() ?? []) {
        const scope = { ...options.textureDescriptors, ...signature.descriptors };
        changed = collectSemanticTextureDescriptorSpecializationsFromOperations(fn.body, scope, ir, out) || changed;
      }
    }
  }
  return out;
}

export function semanticFunctionCallName(
  callee: string,
  fn: SemanticKernelIrModule["functions"][number],
  args: readonly SemanticExpression[],
  options: SemanticTextureDescriptorOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
): string {
  const signature = semanticTextureDescriptorSignatureForCall(fn, args, options.textureDescriptors ?? {});
  if (!signature || !textureSpecializations.get(callee)?.has(signature.key)) return callee;
  return semanticSpecializedFunctionName(callee, signature.key);
}

export function semanticSpecializedFunctionName(name: string, key: string): string {
  return `${name}__bg_tex_${semanticStableHash(key)}`;
}

export function semanticOptionsWithTextureDescriptors(
  options: SemanticTextureDescriptorOptions,
  descriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>,
): SemanticTextureDescriptorOptions {
  const next: SemanticTextureDescriptorOptions = {
    ...(options.f16Mode === undefined ? {} : { f16Mode: options.f16Mode }),
    textureDescriptors: Object.assign({}, options.textureDescriptors, descriptors),
  };
  if (options.pointerBaseOffsets !== undefined) return { ...next, pointerBaseOffsets: options.pointerBaseOffsets };
  return next;
}

export function semanticTextureDescriptorHelpers(
  options: SemanticTextureDescriptorOptions,
  textureSpecializations: SemanticTextureDescriptorSpecializations,
  names: ReadonlyMap<string, string>,
): readonly SemanticTextureDescriptorHelper[] {
  const helpers = new Map<string, SemanticTextureDescriptorHelper>();
  const add = (textureName: string, descriptor: CudaLiteTextureDescriptor): void => {
    helpers.set(semanticTextureDescriptorHelperName(textureName, names, descriptor), { textureName, descriptor });
  };
  for (const [textureName, descriptor] of Object.entries(options.textureDescriptors ?? {})) add(textureName, descriptor);
  for (const specializations of textureSpecializations.values()) {
    for (const signature of specializations.values()) {
      for (const [textureName, descriptor] of Object.entries(signature.descriptors)) add(textureName, descriptor);
    }
  }
  return [...helpers.values()];
}

export function semanticTextureDescriptorHelperName(
  textureName: string,
  names: ReadonlyMap<string, string>,
  descriptor: CudaLiteTextureDescriptor,
): string {
  return `bg_sem_tex2d_${safeWgslIdentifier(textureWgslNameFor(textureName, names))}_${semanticStableHash(semanticTextureDescriptorKey(descriptor))}`;
}

export function semanticTextureScaledCoord(
  value: string,
  extent: string,
  descriptor: CudaLiteTextureDescriptor,
): string {
  return descriptor.normalizedCoords ? `(${value} * f32(${extent}))` : value;
}

export function semanticTextureIndex(
  value: string,
  extent: string,
  descriptor: CudaLiteTextureDescriptor,
  axis: "x" | "y",
): string {
  const mode = descriptor.addressMode?.[axis === "x" ? 0 : 1] ?? "clamp";
  const signedExtent = `i32(${extent})`;
  if (mode === "wrap") return `(((${value}) % ${signedExtent}) + ${signedExtent}) % ${signedExtent}`;
  return `clamp(${value}, 0, (${signedExtent} - 1))`;
}

function collectSemanticTextureDescriptorSpecializationsFromOperations(
  operations: readonly SemanticKernelIrOperation[],
  scope: Readonly<Record<string, CudaLiteTextureDescriptor>>,
  ir: SemanticKernelIrModule,
  out: Map<string, Map<string, SemanticTextureDescriptorSignature>>,
): boolean {
  let changed = false;
  for (const operation of operations) {
    for (const expression of semanticOperationExpressions(operation)) {
      changed = collectSemanticTextureDescriptorSpecializationsFromExpression(expression, scope, ir, out) || changed;
    }
    if (operation.kind === "call") {
      changed = addSemanticTextureDescriptorSignature(operation.callee, operation.args, scope, ir, out) || changed;
    }
    if (operation.kind === "branch") {
      changed = collectSemanticTextureDescriptorSpecializationsFromOperations(operation.consequent, scope, ir, out) || changed;
      changed = collectSemanticTextureDescriptorSpecializationsFromOperations(operation.alternate, scope, ir, out) || changed;
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        changed = collectSemanticTextureDescriptorSpecializationsFromOperations([operation.init], scope, ir, out) || changed;
      }
      changed = collectSemanticTextureDescriptorSpecializationsFromOperations(operation.body, scope, ir, out) || changed;
    }
    if (operation.kind === "block") {
      changed = collectSemanticTextureDescriptorSpecializationsFromOperations(operation.body, scope, ir, out) || changed;
    }
  }
  return changed;
}

function collectSemanticTextureDescriptorSpecializationsFromExpression(
  expression: SemanticExpression,
  scope: Readonly<Record<string, CudaLiteTextureDescriptor>>,
  ir: SemanticKernelIrModule,
  out: Map<string, Map<string, SemanticTextureDescriptorSignature>>,
): boolean {
  let changed = false;
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    changed = addSemanticTextureDescriptorSignature(expression.callee.name, expression.args, scope, ir, out);
  }
  for (const child of semanticExpressionChildren(expression)) {
    changed = collectSemanticTextureDescriptorSpecializationsFromExpression(child, scope, ir, out) || changed;
  }
  return changed;
}

function addSemanticTextureDescriptorSignature(
  callee: string,
  args: readonly SemanticExpression[],
  scope: Readonly<Record<string, CudaLiteTextureDescriptor>>,
  ir: SemanticKernelIrModule,
  out: Map<string, Map<string, SemanticTextureDescriptorSignature>>,
): boolean {
  const fn = ir.functions.find((item) => item.name === callee);
  if (!fn) return false;
  const signature = semanticTextureDescriptorSignatureForCall(fn, args, scope);
  if (!signature) return false;
  let signatures = out.get(fn.name);
  if (!signatures) {
    signatures = new Map();
    out.set(fn.name, signatures);
  }
  if (signatures.has(signature.key)) return false;
  signatures.set(signature.key, signature);
  return true;
}

function semanticTextureDescriptorSignatureForCall(
  fn: SemanticKernelIrModule["functions"][number],
  args: readonly SemanticExpression[],
  scope: Readonly<Record<string, CudaLiteTextureDescriptor>>,
): SemanticTextureDescriptorSignature | undefined {
  const descriptors: Record<string, CudaLiteTextureDescriptor> = {};
  for (const [index, param] of fn.params.entries()) {
    if (param.addressSpace !== "texture") continue;
    const arg = args[index];
    if (arg?.kind !== "symbol" || arg.addressSpace !== "texture") continue;
    const descriptor = scope[arg.name];
    if (descriptor !== undefined) descriptors[param.name] = descriptor;
  }
  if (Object.keys(descriptors).length === 0) return undefined;
  const key = semanticTextureDescriptorSignatureKey(descriptors);
  return { key, descriptors };
}

function semanticTextureDescriptorSignatureKey(
  descriptors: Readonly<Record<string, CudaLiteTextureDescriptor>>,
): string {
  return Object.entries(descriptors)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, descriptor]) => `${name}=${semanticTextureDescriptorKey(descriptor)}`)
    .join(",");
}

function semanticTextureDescriptorKey(descriptor: CudaLiteTextureDescriptor): string {
  return JSON.stringify({
    normalizedCoords: descriptor.normalizedCoords ?? false,
    addressMode: descriptor.addressMode ?? ["clamp", "clamp"],
    filterMode: descriptor.filterMode ?? "point",
  });
}

function semanticStableHash(value: string): string {
  let hash = 5381;
  for (let i = 0; i < value.length; i++) hash = ((hash * 33) ^ value.charCodeAt(i)) >>> 0;
  return hash.toString(36);
}

function textureWgslNameFor(name: string, names: ReadonlyMap<string, string>): string {
  return names.get(name) ?? safeWgslIdentifier(name);
}
