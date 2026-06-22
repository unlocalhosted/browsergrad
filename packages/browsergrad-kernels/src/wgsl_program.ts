import { asImpl } from "./device.js";
import { KernelError, type KernelDevice } from "./types.js";

export type WgslValueType = "f16" | "f32" | "i32" | "u32";
export type WgslStorageAccess = "read" | "read_write";
export interface WgslFloat16Array extends ArrayBufferView {
  readonly BYTES_PER_ELEMENT: 2;
  readonly length: number;
  [index: number]: number;
  [Symbol.iterator](): IterableIterator<number>;
  slice(start?: number, end?: number): WgslFloat16Array;
}
export type WgslTypedArray = WgslFloat16Array | Float32Array | Int32Array | Uint32Array;

export interface WgslResidentBuffer {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
  readonly valueType: WgslValueType;
}

export interface WgslStorageBufferDescriptor {
  readonly valueType: WgslValueType;
  readonly byteLength?: number;
  readonly data?: WgslTypedArray;
  readonly usage?: GPUBufferUsageFlags;
  readonly label?: string;
}

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

export interface WgslTexture2DBindingInput {
  readonly kind: "texture2d";
  readonly name: string;
  readonly valueType: "f32";
  readonly binding?: number;
}

export type WgslKernelBindingInput =
  | WgslStorageBindingInput
  | WgslUniformBindingInput
  | WgslTexture2DBindingInput;

export type WgslKernelBinding =
  | (WgslStorageBindingInput & { readonly binding: number; readonly access: WgslStorageAccess })
  | (WgslUniformBindingInput & { readonly binding: number })
  | (WgslTexture2DBindingInput & { readonly binding: number });
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
  readonly residentBuffers?: Readonly<Record<string, WgslResidentBuffer>>;
  readonly textures?: Readonly<Record<string, WgslTexture2DInput>>;
  readonly uniforms?: Readonly<Record<string, ArrayBuffer | ArrayBufferView>>;
  readonly readback?: readonly string[];
}

export interface WgslTexture2DInput {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
}

export interface WgslKernelLaunch {
  readonly dispatchCount: readonly [number, number, number];
}

export interface WgslKernelSequenceStep {
  readonly program: WgslKernelProgram;
  readonly launch: WgslKernelLaunch;
  readonly storageAliases?: Readonly<Record<string, string>>;
  readonly uniforms?: Readonly<Record<string, ArrayBuffer | ArrayBufferView>>;
}

export interface WgslKernelRunResult {
  readonly buffers: Readonly<Record<string, WgslTypedArray>>;
}

export interface WgslPreparedKernelSequenceRunOptions {
  readonly readback?: readonly string[];
  readonly awaitCompletion?: boolean;
  readonly uniforms?: Readonly<Record<string, ArrayBuffer | ArrayBufferView>>;
  readonly stepUniforms?: Readonly<Record<number, Readonly<Record<string, ArrayBuffer | ArrayBufferView>>>>;
}

export interface WgslPreparedKernelSequence {
  readonly stepCount: number;
  run(options?: WgslPreparedKernelSequenceRunOptions): Promise<WgslKernelRunResult>;
  destroy(): void;
}

interface CachedWgslPipeline {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

type KernelDeviceImpl = ReturnType<typeof asImpl>;
type PreparedStorageBuffer = { binding: WgslStorageBinding; buffer: GPUBuffer; byteLength: number; owned: boolean };
type PreparedUniformBuffer = { buffer: GPUBuffer; byteLength: number };
type PreparedExecutableStep = { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; program: WgslKernelProgram };

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
    if (binding.kind === "texture2d") {
      return {
        kind: "texture2d",
        name: binding.name,
        valueType: binding.valueType,
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
  return runWgslKernelProgramSequence(device, [{ program, launch }], input);
}

export function createWgslStorageBuffer(
  device: KernelDevice,
  descriptor: WgslStorageBufferDescriptor,
): WgslResidentBuffer {
  const impl = asImpl(device);
  const gpu = impl.gpu;
  const byteLength = descriptor.byteLength ?? descriptor.data?.byteLength;
  if (byteLength === undefined) {
    throw new KernelError("storage buffer descriptor requires byteLength or data");
  }
  const logicalByteLength = validateResidentByteLength(byteLength, descriptor.label ?? "resident");
  validateResidentValueByteLength(logicalByteLength, descriptor.valueType, descriptor.label ?? "resident");
  if (descriptor.data) {
    validateTypedArray(descriptor.data, {
      kind: "storage",
      name: descriptor.label ?? "resident",
      valueType: descriptor.valueType,
      access: "read_write",
      binding: 0,
    });
    if (descriptor.data.byteLength > logicalByteLength) {
      throw new KernelError("storage buffer data exceeds descriptor byteLength");
    }
  }
  const buffer = gpu.createBuffer({
    ...(descriptor.label === undefined ? {} : { label: descriptor.label }),
    size: alignTo(logicalByteLength, 4),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC |
      (descriptor.usage ?? 0),
  });
  if (descriptor.data) {
    const upload = bytesForGpuWrite(descriptor.data, alignTo(logicalByteLength, 4));
    gpu.queue.writeBuffer(buffer, 0, upload.buffer, upload.byteOffset, upload.byteLength);
  }
  return {
    buffer,
    byteLength: logicalByteLength,
    valueType: descriptor.valueType,
  };
}

export async function readWgslStorageBuffer(
  device: KernelDevice,
  resident: WgslResidentBuffer,
): Promise<WgslTypedArray> {
  validateResidentBuffer(resident, "resident");
  const impl = asImpl(device);
  const gpu = impl.gpu;
  const readBuffer = gpu.createBuffer({
    size: alignTo(resident.byteLength, 4),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
    const encoder = gpu.createCommandEncoder({ label: "bg-wgsl-read-resident" });
    encoder.copyBufferToBuffer(resident.buffer, 0, readBuffer, 0, alignTo(resident.byteLength, 4));
    gpu.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    try {
      const bytes = readBuffer
        .getMappedRange(0, alignTo(resident.byteLength, 4))
        .slice(0, resident.byteLength);
      return typedArrayFromBytes(resident.valueType, bytes);
    } finally {
      readBuffer.unmap();
    }
  } finally {
    readBuffer.destroy();
  }
}

export function writeWgslStorageBuffer(
  device: KernelDevice,
  resident: WgslResidentBuffer,
  data: WgslTypedArray,
  offsetBytes = 0,
): void {
  validateResidentBuffer(resident, "resident");
  validateTypedArray(data, {
    kind: "storage",
    name: "resident",
    valueType: resident.valueType,
    access: "read_write",
    binding: 0,
  });
  if (!Number.isInteger(offsetBytes) || offsetBytes < 0) {
    throw new KernelError("resident buffer write offset must be a non-negative integer");
  }
  if (offsetBytes + data.byteLength > resident.byteLength) {
    throw new KernelError("resident buffer write exceeds buffer byteLength");
  }
  const impl = asImpl(device);
  impl.gpu.queue.writeBuffer(resident.buffer, offsetBytes, data.buffer, data.byteOffset, data.byteLength);
}

export function destroyWgslStorageBuffer(resident: WgslResidentBuffer): void {
  resident.buffer.destroy();
}

export async function runWgslKernelProgramSequence(
  device: KernelDevice,
  steps: readonly WgslKernelSequenceStep[],
  input: WgslKernelRunInput,
): Promise<WgslKernelRunResult> {
  const prepared = await prepareWgslKernelProgramSequence(device, steps, input);
  try {
    return await prepared.run();
  } finally {
    prepared.destroy();
  }
}

export async function prepareWgslKernelProgramSequence(
  device: KernelDevice,
  steps: readonly WgslKernelSequenceStep[],
  input: WgslKernelRunInput,
): Promise<WgslPreparedKernelSequence> {
  if (steps.length === 0) throw new KernelError("WGSL program sequence must contain at least one step");
  const impl = asImpl(device);
  const gpu = impl.gpu;
  const dispatchCounts = steps.map((step, index) => validateDispatchCountWithName(step.launch.dispatchCount, `steps[${index}].launch.dispatchCount`));
  const programs = steps.map((step) => step.program);
  const readbackNames = new Set(
    input.readback ??
      steps
        .flatMap((step) => step.program.bindings.map((binding) => ({ step, binding })))
        .filter(({ binding }) => binding.kind === "storage" && binding.access === "read_write")
        .map(({ step, binding }) => storageBufferNameForStep(step, binding.name)),
  );

  const storageBuffers = new Map<string, PreparedStorageBuffer>();
  const uniformBuffers: GPUBuffer[] = [];
  const uniformBuffersByStep = new Map<string, PreparedUniformBuffer>();
  const textures: GPUTexture[] = [];
  const textureViewsByName = new Map<string, GPUTextureView>();

  try {
    for (const binding of collectStorageBindings(steps)) {
      const resident = input.residentBuffers?.[binding.name];
      const data = input.buffers[binding.name];
      if (resident && data) {
        throw new KernelError(`storage buffer ${binding.name} provided as both typed array and resident GPU buffer`);
      }
      if (resident) {
        validateResidentStorageBuffer(resident, binding);
        storageBuffers.set(binding.name, { binding, buffer: resident.buffer, byteLength: resident.byteLength, owned: false });
        continue;
      }
      if (!data) throw new KernelError(`missing storage buffer input: ${binding.name}`);
      validateTypedArray(data, binding);
      const byteLength = validateStorageByteLength(data.byteLength, binding.name);
      const buffer = gpu.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const upload = bytesForGpuWrite(data, byteLength);
      gpu.queue.writeBuffer(buffer, 0, upload.buffer, upload.byteOffset, upload.byteLength);
      storageBuffers.set(binding.name, { binding, buffer, byteLength: data.byteLength, owned: true });
    }

    for (let stepIndex = 0; stepIndex < steps.length; stepIndex++) {
      const step = steps[stepIndex]!;
      for (const binding of step.program.bindings) {
        if (binding.kind !== "uniform") continue;
        const data = step.uniforms?.[binding.name] ?? input.uniforms?.[binding.name];
        if (!data) throw new KernelError(`missing uniform input for step ${stepIndex}: ${binding.name}`);
        const bytes = bytesFromBufferSource(data);
        if (binding.byteLength !== undefined && bytes.byteLength > binding.byteLength) {
          throw new KernelError(
            `uniform ${binding.name} for step ${stepIndex} has ${bytes.byteLength} bytes; expected at most ${binding.byteLength}`,
          );
        }
        if (binding.byteLength === undefined && bytes.byteLength === 0) {
          throw new KernelError(`uniform ${binding.name} for step ${stepIndex} must not be empty`);
        }
        const byteLength = alignTo(binding.byteLength ?? bytes.byteLength, 16);
        const buffer = gpu.createBuffer({
          size: byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        gpu.queue.writeBuffer(buffer, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
        uniformBuffers.push(buffer);
        uniformBuffersByStep.set(sequenceUniformKey(stepIndex, binding.name), { buffer, byteLength });
      }
    }

    for (const binding of collectTextureBindings(programs)) {
        const textureInput = input.textures?.[binding.name];
        if (!textureInput) throw new KernelError(`missing texture input: ${binding.name}`);
        validateTexture2DInput(textureInput, binding.name);
        const texture = gpu.createTexture({
          size: { width: textureInput.width, height: textureInput.height },
          format: "r32float",
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const bytes = paddedTextureRows(textureInput);
        gpu.queue.writeTexture(
          { texture },
          bytes.data,
          { bytesPerRow: bytes.bytesPerRow, rowsPerImage: textureInput.height },
          { width: textureInput.width, height: textureInput.height },
        );
        const view = texture.createView();
        textures.push(texture);
        textureViewsByName.set(binding.name, view);
    }

    const executableSteps: PreparedExecutableStep[] = [];
    for (let stepIndex = 0; stepIndex < programs.length; stepIndex++) {
      const program = programs[stepIndex]!;
      const { layoutEntries, bindGroupEntries } = bindGroupInputsForProgram(
        steps[stepIndex]!,
        storageBuffers,
        uniformBuffersForStep(uniformBuffersByStep, stepIndex),
        textureViewsByName,
      );
      const { pipeline, bindGroupLayout } = await getOrCreatePipeline(gpu, program, layoutEntries);
      const bindGroup = gpu.createBindGroup({
        layout: bindGroupLayout,
        entries: [...bindGroupEntries],
      });
      executableSteps.push({ pipeline, bindGroup, program });
    }

    return new PreparedWgslKernelSequenceImpl(
      impl,
      dispatchCounts,
      storageBuffers,
      uniformBuffers,
      uniformBuffersByStep,
      textures,
      executableSteps,
      readbackNames,
      `bg-wgsl-sequence-${programs.map((program) => program.name).join("-")}`,
    );
  } catch (error) {
    destroyPreparedResources(storageBuffers, uniformBuffers, textures);
    throw error;
  }
}

class PreparedWgslKernelSequenceImpl implements WgslPreparedKernelSequence {
  readonly stepCount: number;
  private destroyed = false;
  private lastSubmission: Promise<void> = Promise.resolve();

  constructor(
    private readonly impl: KernelDeviceImpl,
    private readonly dispatchCounts: readonly (readonly [number, number, number])[],
    private readonly storageBuffers: Map<string, PreparedStorageBuffer>,
    private readonly uniformBuffers: readonly GPUBuffer[],
    private readonly uniformBuffersByStep: ReadonlyMap<string, PreparedUniformBuffer>,
    private readonly textures: readonly GPUTexture[],
    private readonly executableSteps: readonly PreparedExecutableStep[],
    private readonly defaultReadbackNames: ReadonlySet<string>,
    private readonly label: string,
  ) {
    this.stepCount = executableSteps.length;
  }

  async run(options: WgslPreparedKernelSequenceRunOptions = {}): Promise<WgslKernelRunResult> {
    if (this.destroyed) throw new KernelError("prepared WGSL sequence has been destroyed");
    const gpu = this.impl.gpu;
    if (hasPreparedUniformUpdates(options)) await this.lastSubmission;
    updatePreparedUniformBuffers(gpu, this.uniformBuffersByStep, options);
    const encoder = gpu.createCommandEncoder({ label: this.label });
    for (let i = 0; i < this.executableSteps.length; i++) {
      const step = this.executableSteps[i]!;
      const dispatchCount = this.dispatchCounts[i]!;
      const pass = encoder.beginComputePass({ label: `bg-wgsl-${step.program.name}-step-${i}` });
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.bindGroup);
      pass.dispatchWorkgroups(
        Math.max(Math.ceil(dispatchCount[0] / step.program.workgroupSize[0]), 1),
        Math.max(Math.ceil(dispatchCount[1] / step.program.workgroupSize[1]), 1),
        Math.max(Math.ceil(dispatchCount[2] / step.program.workgroupSize[2]), 1),
      );
      pass.end();
    }

    const readbackNames = new Set(options.readback ?? this.defaultReadbackNames);
    const readBuffers = new Set<GPUBuffer>();
    const reads: Array<{ name: string; binding: WgslStorageBinding; readBuffer: GPUBuffer; byteLength: number }> = [];
    try {
      for (const name of readbackNames) {
        const entry = this.storageBuffers.get(name);
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
      const completion = gpu.queue.onSubmittedWorkDone();
      this.lastSubmission = completion;
      const output = await collectReadbacks(reads);
      if (options.awaitCompletion === true) await completion;
      for (let i = 0; i < this.stepCount; i++) this.impl.recordInvocation();
      return { buffers: output };
    } finally {
      for (const readBuffer of readBuffers) readBuffer.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    destroyPreparedResources(this.storageBuffers, this.uniformBuffers, this.textures);
  }
}

function hasPreparedUniformUpdates(options: WgslPreparedKernelSequenceRunOptions): boolean {
  return Object.keys(options.uniforms ?? {}).length > 0 || Object.keys(options.stepUniforms ?? {}).length > 0;
}

async function collectReadbacks(
  reads: readonly { readonly name: string; readonly binding: WgslStorageBinding; readonly readBuffer: GPUBuffer; readonly byteLength: number }[],
): Promise<Record<string, WgslTypedArray>> {
  const output: Record<string, WgslTypedArray> = {};
  const mapResults = await Promise.allSettled(reads.map((read) => read.readBuffer.mapAsync(GPUMapMode.READ)));
  const failedMap = mapResults.find((result): result is PromiseRejectedResult => result.status === "rejected");
  if (failedMap) {
    for (let i = 0; i < mapResults.length; i++) {
      if (mapResults[i]?.status === "fulfilled") reads[i]?.readBuffer.unmap();
    }
    throw failedMap.reason;
  }
  for (const read of reads) {
    try {
      const bytes = read.readBuffer.getMappedRange(0, read.byteLength).slice(0);
      output[read.name] = typedArrayFromBytes(read.binding.valueType, bytes);
    } finally {
      read.readBuffer.unmap();
    }
  }
  return output;
}

function destroyPreparedResources(
  storageBuffers: ReadonlyMap<string, PreparedStorageBuffer>,
  uniformBuffers: readonly GPUBuffer[],
  textures: readonly GPUTexture[],
): void {
  for (const entry of storageBuffers.values()) {
    if (entry.owned) entry.buffer.destroy();
  }
  for (const buffer of uniformBuffers) buffer.destroy();
  for (const texture of textures) texture.destroy();
}

function updatePreparedUniformBuffers(
  gpu: GPUDevice,
  uniformBuffersByStep: ReadonlyMap<string, PreparedUniformBuffer>,
  options: WgslPreparedKernelSequenceRunOptions,
): void {
  if (options.uniforms) {
    for (const [name, data] of Object.entries(options.uniforms)) {
      let matched = false;
      for (const [key, entry] of uniformBuffersByStep) {
        if (key.slice(key.indexOf("\0") + 1) !== name) continue;
        matched = true;
        writeUniformBuffer(gpu, entry, data, `uniform ${name}`);
      }
      if (!matched) throw new KernelError(`prepared uniform input is not a uniform binding: ${name}`);
    }
  }

  if (options.stepUniforms) {
    for (const [stepIndexText, uniforms] of Object.entries(options.stepUniforms)) {
      const stepIndex = Number(stepIndexText);
      if (!Number.isInteger(stepIndex) || stepIndex < 0) {
        throw new KernelError(`prepared step uniform index must be a non-negative integer: ${stepIndexText}`);
      }
      for (const [name, data] of Object.entries(uniforms)) {
        const key = sequenceUniformKey(stepIndex, name);
        const entry = uniformBuffersByStep.get(key);
        if (!entry) throw new KernelError(`prepared uniform input is not a uniform binding for step ${stepIndex}: ${name}`);
        writeUniformBuffer(gpu, entry, data, `uniform ${name} for step ${stepIndex}`);
      }
    }
  }
}

function writeUniformBuffer(
  gpu: GPUDevice,
  entry: PreparedUniformBuffer,
  data: ArrayBuffer | ArrayBufferView,
  name: string,
): void {
  const bytes = paddedBytesForUniformWrite(data, entry.byteLength, name);
  gpu.queue.writeBuffer(entry.buffer, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function paddedBytesForUniformWrite(
  source: ArrayBuffer | ArrayBufferView,
  byteLength: number,
  name: string,
): Uint8Array {
  const bytes = bytesFromBufferSource(source);
  if (bytes.byteLength > byteLength) {
    throw new KernelError(`${name} has ${bytes.byteLength} bytes; expected at most ${byteLength}`);
  }
  if (bytes.byteLength === byteLength) return bytes;
  const padded = new Uint8Array(byteLength);
  padded.set(bytes);
  return padded;
}

function collectStorageBindings(steps: readonly WgslKernelSequenceStep[]): readonly WgslStorageBinding[] {
  const byName = new Map<string, WgslStorageBinding>();
  for (const step of steps) {
    for (const binding of step.program.bindings) {
      if (binding.kind !== "storage") continue;
      const storageName = storageBufferNameForStep(step, binding.name);
      const existing = byName.get(storageName);
      if (!existing) {
        byName.set(storageName, { ...binding, name: storageName });
        continue;
      }
      byName.set(storageName, {
        ...existing,
        access: existing.access === "read_write" || binding.access === "read_write" ? "read_write" : "read",
      });
    }
  }
  return [...byName.values()];
}

function storageBufferNameForStep(step: WgslKernelSequenceStep, bindingName: string): string {
  return step.storageAliases?.[bindingName] ?? bindingName;
}

function collectTextureBindings(programs: readonly WgslKernelProgram[]): readonly Extract<WgslKernelBinding, { kind: "texture2d" }>[] {
  const byName = new Map<string, Extract<WgslKernelBinding, { kind: "texture2d" }>>();
  for (const program of programs) {
    for (const binding of program.bindings) {
      if (binding.kind !== "texture2d") continue;
      byName.set(binding.name, binding);
    }
  }
  return [...byName.values()];
}

function sequenceUniformKey(stepIndex: number, name: string): string {
  return `${stepIndex}\0${name}`;
}

function uniformBuffersForStep(
  uniformBuffersByStep: ReadonlyMap<string, PreparedUniformBuffer>,
  stepIndex: number,
): ReadonlyMap<string, GPUBuffer> {
  const buffers = new Map<string, GPUBuffer>();
  const prefix = `${stepIndex}\0`;
  for (const [key, entry] of uniformBuffersByStep) {
    if (key.startsWith(prefix)) buffers.set(key.slice(prefix.length), entry.buffer);
  }
  return buffers;
}

function bindGroupInputsForProgram(
  step: WgslKernelSequenceStep,
  storageBuffers: ReadonlyMap<string, { readonly buffer: GPUBuffer }>,
  uniformBuffers: ReadonlyMap<string, GPUBuffer>,
  textureViews: ReadonlyMap<string, GPUTextureView>,
): { readonly layoutEntries: readonly GPUBindGroupLayoutEntry[]; readonly bindGroupEntries: readonly GPUBindGroupEntry[] } {
  const layoutEntries: GPUBindGroupLayoutEntry[] = [];
  const bindGroupEntries: GPUBindGroupEntry[] = [];
  for (const binding of step.program.bindings) {
    if (binding.kind === "storage") {
      const storageName = storageBufferNameForStep(step, binding.name);
      const entry = storageBuffers.get(storageName);
      if (!entry) throw new KernelError(`missing storage buffer input: ${storageName}`);
      layoutEntries.push({
        binding: binding.binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding.access === "read" ? "read-only-storage" : "storage" },
      });
      bindGroupEntries.push({ binding: binding.binding, resource: { buffer: entry.buffer } });
    } else if (binding.kind === "uniform") {
      const buffer = uniformBuffers.get(binding.name);
      if (!buffer) throw new KernelError(`missing uniform input: ${binding.name}`);
      layoutEntries.push({
        binding: binding.binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      });
      bindGroupEntries.push({ binding: binding.binding, resource: { buffer } });
    } else {
      const view = textureViews.get(binding.name);
      if (!view) throw new KernelError(`missing texture input: ${binding.name}`);
      layoutEntries.push({
        binding: binding.binding,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: "unfilterable-float",
          viewDimension: "2d",
          multisampled: false,
        },
      });
      bindGroupEntries.push({ binding: binding.binding, resource: view });
    }
  }
  return { layoutEntries, bindGroupEntries };
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

function bytesForGpuWrite(source: ArrayBufferView, alignedByteLength: number): Uint8Array {
  const bytes = bytesFromBufferSource(source);
  if (bytes.byteLength === alignedByteLength) return bytes;
  const padded = new Uint8Array(alignedByteLength);
  padded.set(bytes);
  return padded;
}

function typedArrayFromBytes(type: WgslValueType, bytes: ArrayBuffer): WgslTypedArray {
  switch (type) {
    case "f16":
      return new (requireFloat16ArrayConstructor("f16 readback"))(bytes) as WgslFloat16Array;
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

function validateResidentStorageBuffer(resident: WgslResidentBuffer, binding: WgslStorageBinding): void {
  validateResidentBuffer(resident, binding.name);
  if (resident.valueType !== binding.valueType) {
    throw new KernelError(`storage buffer ${binding.name} expects ${binding.valueType}, got ${resident.valueType}`);
  }
}

function validateResidentBuffer(resident: WgslResidentBuffer, name: string): void {
  validateValueType(resident.valueType, name);
  validateResidentByteLength(resident.byteLength, name);
  validateResidentValueByteLength(resident.byteLength, resident.valueType, name);
  if (!resident.buffer) {
    throw new KernelError(`storage buffer ${name} requires a GPUBuffer`);
  }
}

function validateResidentByteLength(byteLength: number, name: string): number {
  if (!Number.isInteger(byteLength) || byteLength <= 0) {
    throw new KernelError(`storage buffer ${name} must not be empty`);
  }
  return byteLength;
}

function validateResidentValueByteLength(byteLength: number, type: WgslValueType, name: string): void {
  const divisor = type === "f16" ? 2 : 4;
  if (byteLength % divisor !== 0) {
    throw new KernelError(`storage buffer ${name} byteLength must be a multiple of ${divisor} for ${type}`);
  }
}

function validateTexture2DInput(input: WgslTexture2DInput, name: string): void {
  validatePositiveInteger(input.width, `${name}.width`);
  validatePositiveInteger(input.height, `${name}.height`);
  if (!(input.data instanceof Float32Array)) {
    throw new KernelError(`texture ${name} expects Float32Array data`);
  }
  const expected = input.width * input.height;
  if (input.data.length < expected) {
    throw new KernelError(`texture ${name} expects at least ${expected} texels`);
  }
}

function paddedTextureRows(input: WgslTexture2DInput): { readonly data: Uint8Array; readonly bytesPerRow: number } {
  const sourceRowBytes = input.width * Float32Array.BYTES_PER_ELEMENT;
  const bytesPerRow = alignTo(sourceRowBytes, 256);
  const out = new Uint8Array(bytesPerRow * input.height);
  const source = new Uint8Array(input.data.buffer, input.data.byteOffset, input.width * input.height * Float32Array.BYTES_PER_ELEMENT);
  for (let row = 0; row < input.height; row++) {
    const sourceStart = row * sourceRowBytes;
    out.set(source.subarray(sourceStart, sourceStart + sourceRowBytes), row * bytesPerRow);
  }
  return { data: out, bytesPerRow };
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

function validateDispatchCountWithName(
  value: readonly [number, number, number],
  name: string,
): [number, number, number] {
  return [
    validatePositiveInteger(value[0], `${name}[0]`),
    validatePositiveInteger(value[1], `${name}[1]`),
    validatePositiveInteger(value[2], `${name}[2]`),
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
      const type = entry.buffer?.type ?? entry.texture?.sampleType ?? "none";
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
