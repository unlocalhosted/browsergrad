import { asImpl } from "./device.js";
import { KernelError, type KernelDevice } from "./types.js";

export type WgslValueType = "f16" | "f32" | "i32" | "u32";
export type WgslStorageAccess = "read" | "read_write";
export interface WgslFloat16Array extends Uint16Array {}
export type WgslTypedArray = WgslFloat16Array | Float32Array | Int32Array | Uint32Array;

export interface KernelFeatureSet {
  readonly webgpu: boolean;
  readonly shaderF16: boolean;
  readonly float16Array: boolean;
  readonly subgroups: boolean;
  readonly compatibilityMode: boolean;
  readonly features: readonly string[];
}

export interface WgslStorageBindingInput {
  readonly kind: "storage";
  readonly name: string;
  readonly valueType: WgslValueType;
  readonly access?: WgslStorageAccess;
  readonly binding?: number;
}

export interface WgslUniformBindingInput {
  readonly kind: "uniform";
  readonly name: string;
  readonly byteLength?: number;
  readonly binding?: number;
}

export type WgslKernelBindingInput =
  | WgslStorageBindingInput
  | WgslUniformBindingInput;

export type WgslKernelBinding =
  | (WgslStorageBindingInput & { readonly binding: number; readonly access: WgslStorageAccess })
  | (WgslUniformBindingInput & { readonly binding: number });
type WgslStorageBinding = Extract<WgslKernelBinding, { readonly kind: "storage" }>;

export interface WgslKernelProgramInput {
  readonly name: string;
  readonly wgsl: string;
  readonly bindings: readonly WgslKernelBindingInput[];
  readonly workgroupSize: readonly [number, number, number];
}

export interface WgslKernelProgram {
  readonly name: string;
  readonly wgsl: string;
  readonly bindings: readonly WgslKernelBinding[];
  readonly workgroupSize: readonly [number, number, number];
}

export interface WgslKernelRunInput {
  readonly buffers: Readonly<Record<string, WgslTypedArray>>;
  readonly uniforms?: Readonly<Record<string, ArrayBuffer | ArrayBufferView>>;
  readonly readback?: readonly string[];
}

export interface WgslKernelLaunch {
  readonly dispatchCount: readonly [number, number, number];
}

export interface WgslKernelRunResult {
  readonly buffers: Readonly<Record<string, WgslTypedArray>>;
}

interface CachedWgslPipeline {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

const WGSL_PIPELINE_CACHE_LIMIT = 128;
const WGSL_PIPELINE_CACHE = new WeakMap<GPUDevice, Map<string, Promise<CachedWgslPipeline>>>();

export async function detectKernelFeatures(
  adapterOrDevice?: GPUAdapter | GPUDevice | KernelDevice,
): Promise<KernelFeatureSet> {
  const features = featureStrings(adapterOrDevice);
  if (features) return createFeatureSet(true, features, false);

  if (typeof navigator === "undefined" || !navigator.gpu) {
    return createFeatureSet(false, [], false);
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return createFeatureSet(false, [], false);
  return createFeatureSet(true, [...adapter.features].map(String), false);
}

export function defineWgslKernelProgram(
  input: WgslKernelProgramInput,
): WgslKernelProgram {
  validateIdentifier(input.name, "name");
  if (input.wgsl.trim().length === 0) {
    throw new KernelError("WGSL program source must not be empty");
  }
  const workgroupSize = validateWorkgroupSize(input.workgroupSize);
  if (input.bindings.length === 0) {
    throw new KernelError("WGSL program must declare at least one binding");
  }

  const seenNames = new Set<string>();
  const seenBindings = new Set<number>();
  const bindings = input.bindings.map((binding, index): WgslKernelBinding => {
    validateIdentifier(binding.name, `binding[${index}].name`);
    if (seenNames.has(binding.name)) {
      throw new KernelError(`duplicate WGSL binding name: ${binding.name}`);
    }
    seenNames.add(binding.name);
    const bindingIndex = binding.binding ?? index;
    validateNonNegativeInteger(bindingIndex, `binding[${index}].binding`);
    if (seenBindings.has(bindingIndex)) {
      throw new KernelError(`duplicate WGSL binding index: ${bindingIndex}`);
    }
    seenBindings.add(bindingIndex);

    if (binding.kind === "storage") {
      validateValueType(binding.valueType, binding.name);
      return {
        kind: "storage",
        name: binding.name,
        valueType: binding.valueType,
        access: binding.access ?? "read_write",
        binding: bindingIndex,
      };
    }
    if (binding.byteLength !== undefined) {
      validatePositiveInteger(binding.byteLength, `${binding.name}.byteLength`);
    }
    return {
      kind: "uniform",
      name: binding.name,
      ...(binding.byteLength === undefined ? {} : { byteLength: binding.byteLength }),
      binding: bindingIndex,
    };
  });

  return {
    name: input.name,
    wgsl: input.wgsl,
    bindings,
    workgroupSize,
  };
}

export async function runWgslKernelProgram(
  device: KernelDevice,
  program: WgslKernelProgram,
  input: WgslKernelRunInput,
  launch: WgslKernelLaunch,
): Promise<WgslKernelRunResult> {
  const impl = asImpl(device);
  const gpu = impl.gpu;
  const dispatchCount = validateDispatchCount(launch.dispatchCount);
  const readbackNames = new Set(
    input.readback ??
      program.bindings
        .filter((binding) => binding.kind === "storage" && binding.access === "read_write")
        .map((binding) => binding.name),
  );

  const bindGroupEntries: GPUBindGroupEntry[] = [];
  const layoutEntries: GPUBindGroupLayoutEntry[] = [];
  const storageBuffers = new Map<string, { binding: WgslStorageBinding; buffer: GPUBuffer; byteLength: number }>();
  const uniformBuffers: GPUBuffer[] = [];
  const readBuffers = new Set<GPUBuffer>();

  try {
    for (const binding of program.bindings) {
      if (binding.kind === "storage") {
        const data = input.buffers[binding.name];
        if (!data) throw new KernelError(`missing storage buffer input: ${binding.name}`);
        validateTypedArray(data, binding);
        const byteLength = validateStorageByteLength(data.byteLength, binding.name);
        const buffer = gpu.createBuffer({
          size: byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        gpu.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
        storageBuffers.set(binding.name, { binding, buffer, byteLength: data.byteLength });
        layoutEntries.push({
          binding: binding.binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: binding.access === "read" ? "read-only-storage" : "storage",
          },
        });
        bindGroupEntries.push({ binding: binding.binding, resource: { buffer } });
      } else {
        const data = input.uniforms?.[binding.name];
        if (!data) throw new KernelError(`missing uniform input: ${binding.name}`);
        const bytes = bytesFromBufferSource(data);
        if (binding.byteLength !== undefined && bytes.byteLength > binding.byteLength) {
          throw new KernelError(
            `uniform ${binding.name} has ${bytes.byteLength} bytes; expected at most ${binding.byteLength}`,
          );
        }
        if (binding.byteLength === undefined && bytes.byteLength === 0) {
          throw new KernelError(`uniform ${binding.name} must not be empty`);
        }
        const buffer = gpu.createBuffer({
          size: alignTo(binding.byteLength ?? bytes.byteLength, 16),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        gpu.queue.writeBuffer(buffer, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
        uniformBuffers.push(buffer);
        layoutEntries.push({
          binding: binding.binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        });
        bindGroupEntries.push({ binding: binding.binding, resource: { buffer } });
      }
    }

    const { pipeline, bindGroupLayout } = await getOrCreatePipeline(gpu, program, layoutEntries);
    const bindGroup = gpu.createBindGroup({
      layout: bindGroupLayout,
      entries: bindGroupEntries,
    });
    const encoder = gpu.createCommandEncoder({ label: `bg-wgsl-${program.name}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.max(Math.ceil(dispatchCount[0] / program.workgroupSize[0]), 1),
      Math.max(Math.ceil(dispatchCount[1] / program.workgroupSize[1]), 1),
      Math.max(Math.ceil(dispatchCount[2] / program.workgroupSize[2]), 1),
    );
    pass.end();

    const reads: Array<{ name: string; binding: WgslStorageBinding; readBuffer: GPUBuffer; byteLength: number }> = [];
    for (const name of readbackNames) {
      const entry = storageBuffers.get(name);
      if (!entry) throw new KernelError(`readback buffer is not a storage binding: ${name}`);
      const readBuffer = gpu.createBuffer({
        size: alignTo(entry.byteLength, 4),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      readBuffers.add(readBuffer);
      encoder.copyBufferToBuffer(entry.buffer, 0, readBuffer, 0, alignTo(entry.byteLength, 4));
      reads.push({ name, binding: entry.binding, readBuffer, byteLength: entry.byteLength });
    }

    gpu.queue.submit([encoder.finish()]);
    const output: Record<string, WgslTypedArray> = {};
    for (const read of reads) {
      await read.readBuffer.mapAsync(GPUMapMode.READ);
      try {
        const bytes = read.readBuffer.getMappedRange(0, read.byteLength).slice(0);
        output[read.name] = typedArrayFromBytes(read.binding.valueType, bytes);
      } finally {
        read.readBuffer.unmap();
      }
      read.readBuffer.destroy();
      readBuffers.delete(read.readBuffer);
    }
    impl.recordInvocation();
    return { buffers: output };
  } finally {
    for (const readBuffer of readBuffers) readBuffer.destroy();
    for (const entry of storageBuffers.values()) entry.buffer.destroy();
    for (const buffer of uniformBuffers) buffer.destroy();
  }
}

async function getOrCreatePipeline(
  gpu: GPUDevice,
  program: WgslKernelProgram,
  layoutEntries: readonly GPUBindGroupLayoutEntry[],
): Promise<CachedWgslPipeline> {
  const cacheKey = [
    program.name,
    hashString(program.wgsl),
    program.workgroupSize.join(","),
    layoutSignature(layoutEntries),
  ].join("::");
  let cache = WGSL_PIPELINE_CACHE.get(gpu);
  if (!cache) {
    cache = new Map();
    WGSL_PIPELINE_CACHE.set(gpu, cache);
  }
  const existing = cache.get(cacheKey);
  if (existing) return existing;
  if (cache.size >= WGSL_PIPELINE_CACHE_LIMIT) {
    const firstKey = cache.keys().next().value;
    if (firstKey !== undefined) cache.delete(firstKey);
  }
  const promise = createPipeline(gpu, program, layoutEntries).catch((error: unknown) => {
    cache.delete(cacheKey);
    throw error;
  });
  cache.set(cacheKey, promise);
  return promise;
}

async function createPipeline(
  gpu: GPUDevice,
  program: WgslKernelProgram,
  layoutEntries: readonly GPUBindGroupLayoutEntry[],
): Promise<CachedWgslPipeline> {
  const bindGroupLayout = gpu.createBindGroupLayout({ entries: [...layoutEntries] });
  const shaderModule = gpu.createShaderModule({ code: program.wgsl });
  const compilationInfo = "getCompilationInfo" in shaderModule
    ? await shaderModule.getCompilationInfo()
    : undefined;
  const messages = compilationInfo?.messages ?? [];
  const errorMessages = messages.filter((message) => message.type === "error");
  if (errorMessages.length > 0) {
    throw new KernelError(
      `WGSL compile failed for ${program.name}: ${formatCompilationMessages(errorMessages)}`,
    );
  }
  try {
    const pipeline = await gpu.createComputePipelineAsync({
      layout: gpu.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shaderModule, entryPoint: "main" },
    });
    return { pipeline, bindGroupLayout };
  } catch (error) {
    const detail = messages.length > 0
      ? formatCompilationMessages(messages)
      : error instanceof Error
        ? error.message
        : String(error);
    throw new KernelError(`WGSL pipeline creation failed for ${program.name}: ${detail}`);
  }
}

function featureStrings(input: GPUAdapter | GPUDevice | KernelDevice | undefined): string[] | undefined {
  if (!input) return undefined;
  const maybeKernelDevice = input as { readonly gpu?: { readonly features?: GPUSupportedFeatures } };
  if (maybeKernelDevice.gpu?.features) return [...maybeKernelDevice.gpu.features].map(String);
  const maybeFeatureOwner = input as { readonly features?: GPUSupportedFeatures };
  if (maybeFeatureOwner.features) return [...maybeFeatureOwner.features].map(String);
  return undefined;
}

function createFeatureSet(
  webgpu: boolean,
  features: readonly string[],
  compatibilityMode: boolean,
): KernelFeatureSet {
  return {
    webgpu,
    shaderF16: features.includes("shader-f16"),
    float16Array: getFloat16ArrayConstructor() !== undefined,
    subgroups: features.includes("subgroups"),
    compatibilityMode,
    features: [...features].sort(),
  };
}

function bytesFromBufferSource(source: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
}

function typedArrayFromBytes(type: WgslValueType, bytes: ArrayBuffer): WgslTypedArray {
  switch (type) {
    case "f16":
      return new (requireFloat16ArrayConstructor("f16 readback"))(bytes);
    case "f32":
      return new Float32Array(bytes);
    case "i32":
      return new Int32Array(bytes);
    case "u32":
      return new Uint32Array(bytes);
  }
}

function validateTypedArray(data: WgslTypedArray, binding: Extract<WgslKernelBinding, { kind: "storage" }>): void {
  if (binding.valueType === "f16" && !isFloat16Array(data)) {
    throw new KernelError(`storage buffer ${binding.name} expects Float16Array`);
  }
  if (binding.valueType === "f32" && !(data instanceof Float32Array)) {
    throw new KernelError(`storage buffer ${binding.name} expects Float32Array`);
  }
  if (binding.valueType === "i32" && !(data instanceof Int32Array)) {
    throw new KernelError(`storage buffer ${binding.name} expects Int32Array`);
  }
  if (binding.valueType === "u32" && !(data instanceof Uint32Array)) {
    throw new KernelError(`storage buffer ${binding.name} expects Uint32Array`);
  }
}

function validateStorageByteLength(byteLength: number, name: string): number {
  if (!Number.isInteger(byteLength) || byteLength <= 0) {
    throw new KernelError(`storage buffer ${name} must not be empty`);
  }
  return alignTo(byteLength, 4);
}

function validateWorkgroupSize(
  value: readonly [number, number, number],
): [number, number, number] {
  return [
    validatePositiveInteger(value[0], "workgroupSize[0]"),
    validatePositiveInteger(value[1], "workgroupSize[1]"),
    validatePositiveInteger(value[2], "workgroupSize[2]"),
  ];
}

function validateDispatchCount(
  value: readonly [number, number, number],
): [number, number, number] {
  return [
    validatePositiveInteger(value[0], "dispatchCount[0]"),
    validatePositiveInteger(value[1], "dispatchCount[1]"),
    validatePositiveInteger(value[2], "dispatchCount[2]"),
  ];
}

function validateValueType(type: WgslValueType, name: string): void {
  if (type !== "f16" && type !== "f32" && type !== "i32" && type !== "u32") {
    throw new KernelError(`${name} has unsupported WGSL value type: ${String(type)}`);
  }
}

function isFloat16Array(data: WgslTypedArray): boolean {
  const ctor = getFloat16ArrayConstructor();
  return ctor !== undefined && data instanceof ctor;
}

function requireFloat16ArrayConstructor(context: string): Float16ArrayConstructor {
  const ctor = getFloat16ArrayConstructor();
  if (!ctor) throw new KernelError(`Float16Array runtime support is required for ${context}`);
  return ctor;
}

function getFloat16ArrayConstructor(): Float16ArrayConstructor | undefined {
  return (globalThis as typeof globalThis & { readonly Float16Array?: Float16ArrayConstructor }).Float16Array;
}

function validateIdentifier(value: string, name: string): void {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new KernelError(`${name} must be a valid WGSL identifier`);
  }
}

function validatePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new KernelError(`${name} must be a positive integer`);
  }
  return value;
}

function validateNonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new KernelError(`${name} must be a non-negative integer`);
  }
  return value;
}

function formatCompilationMessages(messages: readonly GPUCompilationMessage[]): string {
  return messages
    .map((message) => {
      const loc = `${message.lineNum}:${message.linePos}`;
      return `${message.type} ${loc} ${message.message}`;
    })
    .join("; ");
}

function layoutSignature(entries: readonly GPUBindGroupLayoutEntry[]): string {
  return entries
    .map((entry) => {
      const type = entry.buffer?.type ?? "none";
      return `${entry.binding}:${type}`;
    })
    .join("|");
}

function hashString(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function alignTo(value: number, align: number): number {
  return Math.ceil(value / align) * align;
}
