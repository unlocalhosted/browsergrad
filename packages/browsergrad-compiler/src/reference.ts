import {
  createWgslFloat16Array,
  float16BitsToFloat32,
  float32ToFloat16Bits,
  isWgslFloat16Array,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames, collectKernelLaunchCallees } from "./ast_queries.js";
import { expressionName } from "./analyzer.js";
import { CUDA_INTRINSICS_BY_NAME } from "./intrinsics.js";
import { validateCudaKernelLaunch } from "./launch.js";
import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import { alignofCudaType, sizeofCudaType } from "./type_layout.js";
import {
  cudaVectorConstructorType,
  cudaVectorFieldIndex,
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
  type CudaLiteVectorType,
} from "./vector_types.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteDeviceFunction,
  type CudaLiteExpression,
  type CudaLiteGlobalConstant,
  type CudaLiteKernel,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelLaunch,
  type KernelMemoryAccess,
  type KernelThreadTrace,
  type ReferenceKernelResult,
} from "./types.js";

type EvalValue = number | AddressValue | CooperativeGroupValue | ComplexValue | CudaVectorValue | PoolPointerValue;
type LocalValue = number | Vector3 | AddressValue | CooperativeGroupValue | ComplexValue | CudaVectorValue | PoolPointerValue | LocalArrayValue;
interface Vector3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface AddressValue {
  readonly kind: "address";
  readonly target: LValue;
}

interface CooperativeGroupValue {
  readonly kind: "cooperative-group";
  readonly groupKind: CudaLiteCooperativeGroupDecl["groupKind"];
  readonly tileSize?: number;
}

interface ComplexValue {
  readonly kind: "complex64";
  readonly x: number;
  readonly y: number;
}

interface CudaVectorValue {
  readonly kind: "cuda-vector";
  readonly valueType: CudaLiteVectorType;
  readonly lanes: readonly number[];
}

interface PoolPointerValue {
  readonly kind: "pool-pointer";
  readonly poolName: string;
  readonly byteOffset: number;
  readonly rawBuffer?: boolean;
  readonly valueType?: CudaLiteScalarType;
}

interface LValue {
  readonly name: string;
  readonly space: "local" | "buffer" | "shared" | "constant" | "pool";
  readonly index?: number;
  readonly field?: "x" | "y" | "z" | "w";
  readonly fieldIndex?: number;
  readonly valueType?: CudaLiteScalarType;
  readonly rawStorageIndex?: boolean;
}

interface ThreadContext {
  readonly blockIdx: Vector3;
  readonly threadIdx: Vector3;
  readonly blockDim: Vector3;
  readonly gridDim: Vector3;
  readonly buffers: Map<string, WgslTypedArray>;
  readonly constants: Map<string, number | WgslTypedArray>;
  readonly constantDimensions: Map<string, readonly number[]>;
  readonly textures: NonNullable<CompiledKernelInput["textures"]>;
  readonly surfaces: NonNullable<CompiledKernelInput["surfaces"]>;
  readonly memoryPools: Map<string, MemoryPoolValue>;
  readonly functions: ReadonlyMap<string, CudaLiteDeviceFunction>;
  readonly kernels: ReadonlyMap<string, CudaLiteKernel>;
  readonly scalars: Readonly<Record<string, number>>;
  readonly valueTypes: ReadonlyMap<string, CudaLiteScalarType>;
  readonly locals: Map<string, LocalValue>;
  readonly shared: Map<string, SharedArrayValue>;
  readonly trace: MutableTrace;
}

interface MemoryPoolValue {
  readonly data: Uint32Array;
  offset: number;
}

interface SharedArrayValue {
  readonly dimensions: readonly number[];
  readonly valueType: CudaLiteScalarType;
  readonly data: WgslTypedArray;
}

interface LocalArrayValue {
  readonly kind: "local-array";
  readonly dimensions: readonly number[];
  readonly valueType: CudaLiteScalarType;
  readonly data: WgslTypedArray;
}

interface MutableTrace {
  readonly blockIdx: readonly [number, number, number];
  readonly threadIdx: readonly [number, number, number];
  readonly reads: KernelMemoryAccess[];
  readonly writes: KernelMemoryAccess[];
  readonly sharedReads: KernelMemoryAccess[];
  readonly sharedWrites: KernelMemoryAccess[];
}

type ExecControl = { readonly kind: "return"; readonly value?: EvalValue } | { readonly kind: "continue" };
type BarrierKind = "barrier" | "grid-barrier";
type BarrierGenerator = Generator<BarrierKind, ExecControl | void, void>;

export function runCompiledKernelReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): ReferenceKernelResult {
  validateCudaKernelLaunch(launch, compiled.ir.workgroupSize);
  validateInputs(compiled, input);
  const buffers = cloneBuffers(input.buffers);
  const constants = cloneConstants(input.constants ?? {});
  for (const constant of compiled.ir.constants) {
    if (constant.init !== undefined && !constants.has(constant.name)) {
      constants.set(constant.name, constantInitialValue(constant));
    }
  }
  const constantDimensions = new Map(compiled.ir.constants.map((constant) => [constant.name, constant.dimensions]));
  const textures = input.textures ?? {};
  const surfaces = cloneSurfaces(input.surfaces ?? {});
  const memoryPools = cloneMemoryPools(input.memoryPools ?? {});
  const functions = new Map(compiled.ir.functions.map((fn) => [fn.name, fn]));
  const kernels = collectReferenceKernels(compiled);
  const scalars = input.scalars ?? {};
  const valueTypes = new Map<string, CudaLiteScalarType>([
    ...compiled.ir.params.map((param) => [param.name, param.valueType] as const),
    ...compiled.ir.constants.map((constant) => [constant.name, constant.valueType] as const),
  ]);
  const traces: MutableTrace[] = [];

  if (usesGridSync(compiled.ir.body)) {
    runGrid(compiled.ir.body, compiled, buffers, constants, constantDimensions, textures, surfaces, memoryPools, functions, kernels, scalars, valueTypes, vectorFromTuple(launch.blockDim), vectorFromTuple(launch.gridDim), traces);
  } else {
    for (let bz = 0; bz < launch.gridDim[2]; bz++) {
      for (let by = 0; by < launch.gridDim[1]; by++) {
        for (let bx = 0; bx < launch.gridDim[0]; bx++) {
          runBlock(compiled.ir.body, compiled, buffers, constants, constantDimensions, textures, surfaces, memoryPools, functions, kernels, scalars, valueTypes, {
            x: bx,
            y: by,
            z: bz,
          }, vectorFromTuple(launch.blockDim), vectorFromTuple(launch.gridDim), traces);
        }
      }
    }
  }

  const readback = input.readback ??
    [
      ...compiled.ir.params.filter((param) => (param.pointer && !param.constant) || param.valueType === "surface2d").map((param) => param.name),
      ...collectExternalDevicePoolNames(compiled.ir.body),
    ];
  const result: Record<string, WgslTypedArray> = {};
  for (const name of readback) {
    const buffer = buffers.get(name) ?? surfaces[name]?.data ?? memoryPools.get(name)?.data;
    if (!buffer) throw compilerFailure(`missing readback buffer '${name}'`);
    result[name] = cloneTypedArray(buffer);
  }
  return { buffers: result, trace: traces.map(freezeTrace) };
}

function collectReferenceKernels(compiled: CompiledCudaLiteKernel): Map<string, CudaLiteKernel> {
  const kernels = new Map(compiled.ast.kernels.map((kernel) => [kernel.name, kernel] as const));
  const launched = new Set<string>();
  for (const kernel of compiled.ast.kernels) {
    for (const name of collectKernelLaunchCallees(kernel.body)) launched.add(name);
  }
  for (const fn of compiled.ast.functions) {
    for (const name of collectKernelLaunchCallees(fn.body)) launched.add(name);
  }
  for (const fn of compiled.ast.functions) {
    if (launched.has(fn.name)) kernels.set(fn.name, deviceFunctionAsKernel(fn));
  }
  return kernels;
}

function deviceFunctionAsKernel(fn: CudaLiteDeviceFunction): CudaLiteKernel {
  return {
    kind: "kernel",
    name: fn.name,
    params: fn.params,
    body: fn.body,
    span: fn.span,
  };
}

function runBlock(
  body: readonly CudaLiteStatement[],
  compiled: CompiledCudaLiteKernel,
  buffers: Map<string, WgslTypedArray>,
  constants: Map<string, number | WgslTypedArray>,
  constantDimensions: Map<string, readonly number[]>,
  textures: NonNullable<CompiledKernelInput["textures"]>,
  surfaces: NonNullable<CompiledKernelInput["surfaces"]>,
  memoryPools: Map<string, MemoryPoolValue>,
  functions: ReadonlyMap<string, CudaLiteDeviceFunction>,
  kernels: ReadonlyMap<string, CudaLiteKernel>,
  scalars: Readonly<Record<string, number>>,
  valueTypes: ReadonlyMap<string, CudaLiteScalarType>,
  blockIdx: Vector3,
  blockDim: Vector3,
  gridDim: Vector3,
  traces: MutableTrace[],
): void {
  const shared = allocateShared(sharedDeclarationsFor(body, compiled.ir.sharedDeclarations));
  const generators: BarrierGenerator[] = [];
  const active: boolean[] = [];

  for (let tz = 0; tz < blockDim.z; tz++) {
    for (let ty = 0; ty < blockDim.y; ty++) {
      for (let tx = 0; tx < blockDim.x; tx++) {
        const trace: MutableTrace = {
          blockIdx: [blockIdx.x, blockIdx.y, blockIdx.z],
          threadIdx: [tx, ty, tz],
          reads: [],
          writes: [],
          sharedReads: [],
          sharedWrites: [],
        };
        traces.push(trace);
        const context: ThreadContext = {
          blockIdx,
          threadIdx: { x: tx, y: ty, z: tz },
          blockDim,
          gridDim,
          buffers,
          constants,
          constantDimensions,
          textures,
          surfaces,
          memoryPools,
          functions,
          kernels,
          scalars,
          valueTypes,
          locals: new Map(),
          shared,
          trace,
        };
        generators.push(execStatements(body, context));
        active.push(true);
      }
    }
  }

  while (active.some(Boolean)) {
    let activeBefore = 0;
    let barriers = 0;
    for (let i = 0; i < generators.length; i++) {
      if (!active[i]) continue;
      activeBefore++;
      const next = generators[i]!.next();
      if (next.done) {
        active[i] = false;
      } else {
        barriers++;
      }
    }
    if (barriers > 0 && barriers !== activeBefore) {
      throw compilerFailure("barrier mismatch: not every active thread reached __syncthreads()");
    }
  }
}

function runGrid(
  body: readonly CudaLiteStatement[],
  compiled: CompiledCudaLiteKernel,
  buffers: Map<string, WgslTypedArray>,
  constants: Map<string, number | WgslTypedArray>,
  constantDimensions: Map<string, readonly number[]>,
  textures: NonNullable<CompiledKernelInput["textures"]>,
  surfaces: NonNullable<CompiledKernelInput["surfaces"]>,
  memoryPools: Map<string, MemoryPoolValue>,
  functions: ReadonlyMap<string, CudaLiteDeviceFunction>,
  kernels: ReadonlyMap<string, CudaLiteKernel>,
  scalars: Readonly<Record<string, number>>,
  valueTypes: ReadonlyMap<string, CudaLiteScalarType>,
  blockDim: Vector3,
  gridDim: Vector3,
  traces: MutableTrace[],
): void {
  const generators: BarrierGenerator[] = [];
  const active: boolean[] = [];
  const blockKeys: string[] = [];
  const sharedByBlock = new Map<string, Map<string, SharedArrayValue>>();
  for (let bz = 0; bz < gridDim.z; bz++) {
    for (let by = 0; by < gridDim.y; by++) {
      for (let bx = 0; bx < gridDim.x; bx++) {
        const blockIdx = { x: bx, y: by, z: bz };
        const blockKey = `${bx},${by},${bz}`;
        const shared = sharedByBlock.get(blockKey) ?? allocateShared(sharedDeclarationsFor(body, compiled.ir.sharedDeclarations));
        sharedByBlock.set(blockKey, shared);
        for (let tz = 0; tz < blockDim.z; tz++) {
          for (let ty = 0; ty < blockDim.y; ty++) {
            for (let tx = 0; tx < blockDim.x; tx++) {
              const trace: MutableTrace = {
                blockIdx: [bx, by, bz],
                threadIdx: [tx, ty, tz],
                reads: [],
                writes: [],
                sharedReads: [],
                sharedWrites: [],
              };
              traces.push(trace);
              generators.push(execStatements(body, {
                blockIdx,
                threadIdx: { x: tx, y: ty, z: tz },
                blockDim,
                gridDim,
                buffers,
                constants,
                constantDimensions,
                textures,
                surfaces,
                memoryPools,
                functions,
                kernels,
                scalars,
                valueTypes,
                locals: new Map(),
                shared,
                trace,
              }));
              active.push(true);
              blockKeys.push(blockKey);
            }
          }
        }
      }
    }
  }
  while (active.some(Boolean)) {
    let activeBefore = 0;
    let gridBarriers = 0;
    const activeByBlock = new Map<string, number>();
    const barriersByBlock = new Map<string, number>();
    for (let i = 0; i < generators.length; i++) {
      if (!active[i]) continue;
      activeBefore++;
      const blockKey = blockKeys[i]!;
      activeByBlock.set(blockKey, (activeByBlock.get(blockKey) ?? 0) + 1);
      const next = generators[i]!.next();
      if (next.done) {
        active[i] = false;
      } else if (next.value === "grid-barrier") {
        gridBarriers++;
      } else {
        barriersByBlock.set(blockKey, (barriersByBlock.get(blockKey) ?? 0) + 1);
      }
    }
    if (gridBarriers > 0 && gridBarriers !== activeBefore) {
      throw compilerFailure("grid barrier mismatch: not every active thread reached grid.sync()");
    }
    if (gridBarriers > 0) continue;
    for (const [blockKey, barriers] of barriersByBlock) {
      if (barriers !== activeByBlock.get(blockKey)) {
        throw compilerFailure("barrier mismatch: not every active thread reached __syncthreads()");
      }
    }
  }
}

function* execStatements(
  statements: readonly CudaLiteStatement[],
  context: ThreadContext,
): BarrierGenerator {
  for (const statement of statements) {
    switch (statement.kind) {
      case "block": {
        const outerNames = new Set(context.locals.keys());
        const blockNames = blockScopedNames(statement.body);
        const locals = new Map(context.locals);
        const result = yield* execStatements(statement.body, { ...context, locals });
        for (const name of outerNames) {
          if (blockNames.has(name)) continue;
          if (locals.has(name)) context.locals.set(name, locals.get(name)!);
        }
        if (result) return result;
        break;
      }
      case "var":
        execVar(statement, context);
        break;
      case "dim3":
        context.locals.set(statement.name, vectorFromExpressions(statement.args, context));
        break;
      case "cooperative-group":
        context.locals.set(statement.name, {
          kind: "cooperative-group",
          groupKind: statement.groupKind,
          ...(statement.tileSize === undefined ? {} : { tileSize: statement.tileSize }),
        });
        break;
      case "kernel-launch":
        execKernelLaunch(statement, context);
        break;
      case "asm":
        execInlineAsm(statement, context);
        break;
      case "expr":
        if (execCpAsyncStatement(statement.expression, context)) {
          break;
        } else if (isBarrier(statement.expression)) {
          yield "barrier";
        } else if (cooperativeSyncKind(statement.expression, context) === "grid") {
          yield "grid-barrier";
        } else if (cooperativeSyncKind(statement.expression, context) === "block") {
          yield "barrier";
        } else {
          evalExpression(statement.expression, context);
        }
        break;
      case "if":
        if (truthy(evalNumber(statement.condition, context))) {
          const control = yield* execStatements(statement.consequent, context);
          if (control) return control;
        } else if (statement.alternate) {
          const control = yield* execStatements(statement.alternate, context);
          if (control) return control;
        }
        break;
      case "for":
        if (statement.init) {
          if (statement.init.kind === "var") execVar(statement.init, context);
          else evalExpression(statement.init, context);
        }
        while (statement.condition ? truthy(evalNumber(statement.condition, context)) : true) {
          const control = yield* execStatements(statement.body, context);
          if (control?.kind === "return") return control;
          if (statement.update) evalExpression(statement.update, context);
        }
        break;
      case "while":
        while (truthy(evalNumber(statement.condition, context))) {
          const control = yield* execStatements(statement.body, context);
          if (control?.kind === "return") return control;
        }
        break;
      case "return":
        return {
          kind: "return",
          ...(statement.value === undefined ? {} : { value: evalExpression(statement.value, context) }),
        };
      case "continue":
        return { kind: "continue" };
    }
  }
}

function blockScopedNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const names = new Set<string>();
  for (const statement of statements) {
    if ((statement.kind === "var" && statement.storage === "local") || statement.kind === "dim3" || statement.kind === "cooperative-group") {
      names.add(statement.name);
    }
    if (statement.kind === "for" && statement.init?.kind === "var" && statement.init.storage === "local") {
      names.add(statement.init.name);
    }
  }
  return names;
}

function execInlineAsm(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  context: ThreadContext,
): void {
  if (/\bmov\.u32\b/u.test(statement.template) && /%%laneid\b/u.test(statement.template)) {
    if (statement.inputs.length !== 0) throw compilerFailure("laneid inline asm expects no inputs");
    if (statement.output === undefined) throw compilerFailure("laneid inline asm expects an output operand");
    const target = resolveLValue(statement.output, context);
    writeLValue(target, localLinearRank(context) % 32, context);
    return;
  }
  if (/\bbfind\.u32\b/u.test(statement.template)) {
    if (statement.inputs.length !== 1) throw compilerFailure("bfind.u32 inline asm expects one input");
    if (statement.output === undefined) throw compilerFailure("bfind.u32 inline asm expects an output operand");
    const value = evalExpression(statement.inputs[0]!, context);
    const bits = valueAsNumber(value, "bfind.u32") >>> 0;
    const found = bits === 0 ? 0xffffffff : 31 - Math.clz32(bits);
    writeLValue(resolveLValue(statement.output, context), found, context);
    return;
  }
  if (!/\bfma\.rn\.f32\b/u.test(statement.template)) {
    throw compilerFailure("unsupported inline asm template");
  }
  if (statement.inputs.length !== 2) {
    throw compilerFailure("fma.rn.f32 inline asm expects two inputs");
  }
  if (statement.output === undefined) throw compilerFailure("fma.rn.f32 inline asm expects an output operand");
  const target = resolveLValue(statement.output, context);
  const current = valueAsNumber(readLValue(target, context), target.name);
  const a = evalNumber(statement.inputs[0]!, context);
  const b = evalNumber(statement.inputs[1]!, context);
  writeLValue(target, current + a * b, context);
}

function execCpAsyncStatement(expression: CudaLiteExpression, context: ThreadContext): boolean {
  if (expression.kind !== "call") return false;
  const name = expressionNameForReference(expression.callee);
  if (isCpAsyncFenceCall(name)) return true;
  if (!isCpAsyncCopyCall(name)) return false;
  const [dst, src, bytes] = expression.args;
  if (!dst || !src) return true;
  let dstLvalue: LValue;
  let srcLvalue: LValue;
  try {
    dstLvalue = resolvePointerArgument(dst, context);
    srcLvalue = resolvePointerArgument(src, context);
  } catch {
    return true;
  }
  const valueType = pointerValueTypeForExpression(src, context);
  const count = cpAsyncElementCount(bytes, valueType, context);
  for (let index = 0; index < count; index++) {
    writeLValue(offsetLValue(dstLvalue, index, valueType), readLValue(offsetLValue(srcLvalue, index, valueType), context), context);
  }
  return true;
}

function isCpAsyncCopyCall(name: string | undefined): boolean {
  return name === "CP_ASYNC_CA" || name === "CP_ASYNC_CG" || name === "CP_ASYNC_BULK";
}

function isCpAsyncFenceCall(name: string | undefined): boolean {
  return name === "CP_ASYNC_COMMIT_GROUP" ||
    name === "CP_ASYNC_WAIT_ALL" ||
    name === "CP_ASYNC_WAIT_GROUP" ||
    name === "CP_ASYNC_BULK_COMMIT_GROUP" ||
    name === "CP_ASYNC_BULK_WAIT_ALL" ||
    name === "CP_ASYNC_BULK_WAIT_GROUP";
}

function isPointerIdentityCall(name: string | undefined): boolean {
  return name === "__builtin_assume_aligned" || name === "ct::assume_aligned";
}

function cpAsyncElementCount(bytes: CudaLiteExpression | undefined, valueType: CudaLiteScalarType, context: ThreadContext): number {
  const byteCount = bytes === undefined ? elementByteSize(valueType) : Math.trunc(evalNumber(bytes, context));
  const elementBytes = elementByteSize(valueType);
  if (elementBytes <= 0) return 1;
  return Math.max(1, Math.min(16, Math.floor(byteCount / elementBytes)));
}

function offsetLValue(lvalue: LValue, elementOffset: number, valueType: CudaLiteScalarType): LValue {
  const stride = lvalue.rawStorageIndex ? valueStorageWidth(valueType) : 1;
  return {
    ...lvalue,
    index: (lvalue.index ?? 0) + elementOffset * stride,
    valueType,
  };
}

function execVar(statement: CudaLiteVarDecl, context: ThreadContext): void {
  if (statement.storage === "shared") return;
  if (statement.pointer) {
    context.locals.set(statement.name, resolvePointerInitializer(statement, context));
    return;
  }
  if (statement.dimensions.length > 0) {
    const localArray = allocateLocalArray(statement);
    if (statement.init) initializeLocalArray(localArray, statement.init, context);
    context.locals.set(statement.name, localArray);
    return;
  }
  context.locals.set(statement.name, statement.init ? evalExpression(statement.init, context) : zeroLocalValue(statement.valueType));
}

function execKernelLaunch(
  statement: Extract<CudaLiteStatement, { kind: "kernel-launch" }>,
  context: ThreadContext,
): void {
  const kernel = context.kernels.get(statement.callee);
  if (!kernel) throw compilerFailure(`unknown dynamic kernel '${statement.callee}'`);
  const gridDim = vectorFromLaunchExpressions(statement.grid, context);
  const blockDim = vectorFromLaunchExpressions(statement.block, context);
  const locals = new Map<string, LocalValue>();
  const childValueTypes = new Map(context.valueTypes);
  for (const [index, param] of kernel.params.entries()) {
    const arg = statement.args[index];
    childValueTypes.set(param.name, param.valueType);
    if (!arg) {
      locals.set(param.name, zeroParamLocalValue(param));
      continue;
    }
    if (param.cooperativeGroupKind !== undefined) {
      locals.set(param.name, cooperativeGroupArgumentValue(arg, param, context));
    } else if (param.pointer) {
      locals.set(param.name, pointerArgumentValue(arg, param.valueType, context));
    } else {
      locals.set(param.name, evalNumber(arg, context));
    }
  }
  for (let bz = 0; bz < gridDim.z; bz++) {
    for (let by = 0; by < gridDim.y; by++) {
      for (let bx = 0; bx < gridDim.x; bx++) {
        const childContextSeed = new Map(locals);
        runChildBlock(kernel.body, context, childContextSeed, childValueTypes, { x: bx, y: by, z: bz }, blockDim, gridDim);
      }
    }
  }
}

function runChildBlock(
  body: readonly CudaLiteStatement[],
  parent: ThreadContext,
  seedLocals: Map<string, LocalValue>,
  valueTypes: ReadonlyMap<string, CudaLiteScalarType>,
  blockIdx: Vector3,
  blockDim: Vector3,
  gridDim: Vector3,
): void {
  const shared = allocateShared(sharedDeclarationsFor(body, []));
  const generators: BarrierGenerator[] = [];
  const active: boolean[] = [];
  const trace = parent.trace;
  for (let tz = 0; tz < blockDim.z; tz++) {
    for (let ty = 0; ty < blockDim.y; ty++) {
      for (let tx = 0; tx < blockDim.x; tx++) {
        generators.push(execStatements(body, {
          ...parent,
          blockIdx,
          threadIdx: { x: tx, y: ty, z: tz },
          blockDim,
          gridDim,
          valueTypes,
          locals: new Map(seedLocals),
          shared,
          trace,
        }));
        active.push(true);
      }
    }
  }
  while (active.some(Boolean)) {
    let activeBefore = 0;
    let barriers = 0;
    for (let i = 0; i < generators.length; i++) {
      if (!active[i]) continue;
      activeBefore++;
      const next = generators[i]!.next();
      if (next.done) active[i] = false;
      else barriers++;
    }
    if (barriers > 0 && barriers !== activeBefore) {
      throw compilerFailure("barrier mismatch: not every active dynamic-launch thread reached __syncthreads()");
    }
  }
}

function pointerArgumentValue(
  arg: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: ThreadContext,
): LocalValue {
  if (arg.kind === "call" && isPointerIdentityCall(expressionName(arg.callee))) {
    const pointer = arg.args[0];
    if (!pointer) throw compilerFailure("pointer identity call expects pointer argument");
    return pointerArgumentValue(pointer, valueType, context);
  }
  if (arg.kind === "cast" && arg.pointer) {
    const pointer = pointerArgumentValue(arg.expression, arg.valueType, context);
    if (isPoolPointer(pointer)) return { ...pointer, valueType: arg.valueType };
    if (isAddress(pointer)) {
      return {
        kind: "address",
        target: {
          ...pointer.target,
          index: lvalueStorageIndex(pointer.target, context),
          valueType: arg.valueType,
          rawStorageIndex: true,
        },
      };
    }
  }
  const offset = pointerOffsetArgumentValue(arg, valueType, context);
  if (offset) return offset;
  if (arg.kind === "unary" && arg.operator === "&") {
    return { kind: "address", target: resolveLValue(arg.argument, context) };
  }
  if (arg.kind === "identifier") {
    const local = context.locals.get(arg.name);
    if (isPoolPointer(local)) return { ...local, valueType };
    if (isAddress(local)) return local;
    if (context.buffers.has(arg.name)) return { kind: "address", target: { name: arg.name, space: "buffer", index: 0, valueType } };
    if (context.shared.has(arg.name)) return { kind: "address", target: { name: arg.name, space: "shared", index: 0, valueType } };
    if (context.memoryPools.has(arg.name)) return { kind: "pool-pointer", poolName: arg.name, byteOffset: 0, valueType };
    if (local && typeof local !== "number") return local;
  }
  const value = evalExpression(arg, context);
  if (isPoolPointer(value)) return { ...value, valueType };
  if (isAddress(value)) return value;
  throw compilerFailure("unsupported dynamic kernel pointer argument");
}

function pointerOffsetArgumentValue(
  arg: CudaLiteExpression,
  valueType: CudaLiteScalarType,
  context: ThreadContext,
): LocalValue | undefined {
  if (arg.kind !== "binary" || (arg.operator !== "+" && arg.operator !== "-")) return undefined;
  const left = pointerArgumentValue(arg.left, valueType, context);
  const delta = evalNumber(arg.right, context) * (arg.operator === "-" ? -1 : 1);
  if (isPoolPointer(left)) {
    return { ...left, byteOffset: left.byteOffset + delta * elementByteSize(valueType), valueType };
  }
  if (typeof left !== "number" && "kind" in left && left.kind === "address") {
    return {
      kind: "address",
      target: {
        ...left.target,
        index: (left.target.index ?? 0) + Math.trunc(delta),
      },
    };
  }
  return undefined;
}

function execCudaRuntimeCopy(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  context: ThreadContext,
): void {
  const shape = cudaRuntimeCopyShape(expression);
  if (!shape) throw compilerFailure("unsupported CUDA runtime copy call");
  const dst = expression.args[0];
  const src = expression.args[shape.srcIndex];
  const count = expression.args[shape.countIndex];
  if (!dst || !src || !count) throw compilerFailure("CUDA runtime copy expects dst, src, and byte count");
  const dstView = pointerBytesForCopy(dst, context);
  const srcView = pointerBytesForCopy(src, context);
  const byteCount = Math.max(0, Math.trunc(evalNumber(count, context)));
  if (srcView.byteOffset < 0 || dstView.byteOffset < 0) return;
  const readable = Math.max(0, Math.min(byteCount, srcView.bytes.byteLength - srcView.byteOffset));
  const writable = Math.max(0, Math.min(readable, dstView.bytes.byteLength - dstView.byteOffset));
  if (writable <= 0) return;
  const copied = srcView.bytes.slice(srcView.byteOffset, srcView.byteOffset + writable);
  dstView.bytes.set(copied, dstView.byteOffset);
  context.trace.reads.push({ name: srcView.name, index: srcView.byteOffset, value: writable, ok: writable === byteCount });
  context.trace.writes.push({ name: dstView.name, index: dstView.byteOffset, value: writable, ok: writable === byteCount });
}

function cudaRuntimeCopyShape(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
): { readonly srcIndex: number; readonly countIndex: number } | undefined {
  const name = expression.callee.kind === "identifier" ? expression.callee.name : undefined;
  if (name === "cudaMemcpy" || name === "cudaMemcpyAsync") return { srcIndex: 1, countIndex: 2 };
  if (name === "cudaMemcpyPeerAsync") return { srcIndex: 2, countIndex: 4 };
  return undefined;
}

interface PointerByteView {
  readonly name: string;
  readonly bytes: Uint8Array;
  readonly byteOffset: number;
}

function pointerBytesForCopy(expression: CudaLiteExpression, context: ThreadContext): PointerByteView {
  const valueType = pointerValueTypeForExpression(expression, context);
  const pointer = pointerArgumentValue(expression, valueType, context);
  if (isPoolPointer(pointer)) {
    if (pointer.rawBuffer) {
      const buffer = context.buffers.get(pointer.poolName);
      if (!buffer) throw compilerFailure(`missing raw pool buffer '${pointer.poolName}'`);
      return { name: pointer.poolName, bytes: byteView(buffer), byteOffset: pointer.byteOffset };
    }
    const pool = context.memoryPools.get(pointer.poolName);
    if (!pool) throw compilerFailure(`missing memory pool '${pointer.poolName}'`);
    return { name: pointer.poolName, bytes: byteView(pool.data), byteOffset: pointer.byteOffset };
  }
  if (typeof pointer === "number" || !("kind" in pointer) || pointer.kind !== "address") {
    throw compilerFailure("CUDA runtime copy expects pointer arguments");
  }
  return lvalueByteView(pointer.target, valueType, context);
}

function lvalueByteView(
  lvalue: LValue,
  valueType: CudaLiteScalarType,
  context: ThreadContext,
): PointerByteView {
  const index = lvalue.index ?? 0;
  if (lvalue.space === "buffer") {
    const buffer = context.buffers.get(lvalue.name);
    if (!buffer) throw compilerFailure(`missing buffer '${lvalue.name}'`);
    return { name: lvalue.name, bytes: byteView(buffer), byteOffset: index * elementByteSize(valueType) };
  }
  if (lvalue.space === "shared") {
    const shared = context.shared.get(lvalue.name);
    if (!shared) throw compilerFailure(`missing shared array '${lvalue.name}'`);
    return { name: lvalue.name, bytes: byteView(shared.data), byteOffset: index * elementByteSize(shared.valueType) };
  }
  if (lvalue.space === "pool") {
    const pool = context.memoryPools.get(lvalue.name);
    const buffer = context.buffers.get(lvalue.name);
    if (!pool && !buffer) throw compilerFailure(`missing memory pool '${lvalue.name}'`);
    return { name: lvalue.name, bytes: byteView(pool ? pool.data : buffer!), byteOffset: index * 4 };
  }
  throw compilerFailure(`CUDA runtime copy cannot copy from ${lvalue.space} '${lvalue.name}'`);
}

function pointerValueTypeForExpression(
  expression: CudaLiteExpression,
  context: ThreadContext,
): CudaLiteScalarType {
  if (expression.kind === "cast" && expression.pointer) return expression.valueType;
  if (expression.kind === "call" && isPointerIdentityCall(expressionName(expression.callee))) {
    const pointer = expression.args[0];
    return pointer ? pointerValueTypeForExpression(pointer, context) : "float";
  }
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return pointerValueTypeForExpression(expression.left, context);
  }
  if (expression.kind === "unary" && expression.operator === "&") {
    const root = rootIdentifierFromExpression(expression.argument);
    return root ? context.valueTypes.get(root) ?? "uint" : "uint";
  }
  if (expression.kind === "identifier") {
    const local = context.locals.get(expression.name);
    if (isPoolPointer(local) && local.valueType) return local.valueType;
    if (isAddress(local) && local.target.valueType) return local.target.valueType;
    return context.valueTypes.get(expression.name) ?? "uint";
  }
  const root = rootIdentifierFromExpression(expression);
  return root ? context.valueTypes.get(root) ?? "uint" : "uint";
}

function expressionValueType(expression: CudaLiteExpression, context: ThreadContext): CudaLiteScalarType | undefined {
  if (expression.kind === "identifier") {
    const local = context.locals.get(expression.name);
    if (isCudaVectorValue(local)) return local.valueType;
    if (isComplex(local)) return "complex64";
    return context.valueTypes.get(expression.name);
  }
  if (expression.kind === "index") return pointerValueTypeForExpression(expression.target, context);
  if (expression.kind === "unary" && expression.operator === "*") return pointerValueTypeForExpression(expression.argument, context);
  if (expression.kind === "call") {
    const name = expressionNameForReference(expression.callee);
    return name ? cudaVectorConstructorType(name) : undefined;
  }
  if (expression.kind === "member") return expressionValueType(expression.object, context);
  return undefined;
}

function byteView(buffer: WgslTypedArray): Uint8Array {
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

function elementByteSize(valueType: CudaLiteScalarType): number {
  const vector = cudaVectorLaneCount(valueType);
  if (vector > 1) return vector * scalarByteSize(cudaVectorScalarType(valueType));
  return scalarByteSize(valueType);
}

function scalarByteSize(valueType: CudaLiteScalarType | undefined): number {
  if (valueType === "half") return 2;
  if (valueType === "complex64") return 8;
  return 4;
}

function rootIdentifierFromExpression(expression: CudaLiteExpression): string | undefined {
  if (expression.kind === "identifier") return expression.name;
  if (expression.kind === "index") return rootIdentifierFromExpression(expression.target);
  if (expression.kind === "member") return rootIdentifierFromExpression(expression.object);
  if (expression.kind === "cast") return rootIdentifierFromExpression(expression.expression);
  if (expression.kind === "unary") return rootIdentifierFromExpression(expression.argument);
  if (expression.kind === "binary" && (expression.operator === "+" || expression.operator === "-")) {
    return rootIdentifierFromExpression(expression.left);
  }
  return undefined;
}

function evalExpression(expression: CudaLiteExpression, context: ThreadContext): EvalValue {
  switch (expression.kind) {
    case "number":
      return expression.value;
    case "string":
      return 0;
    case "initializer":
      throw compilerFailure("braced initializer is only valid in a declaration");
    case "identifier": {
      const value = readIdentifier(expression.name, context);
      if (isComplex(value)) return value;
      if (isCudaVectorValue(value)) return value;
      if (isPoolPointer(value)) return value;
      return valueAsNumber(value, expression.name);
    }
    case "cast":
      if (expression.pointer) {
        const value = evalExpression(expression.expression, context);
        return isPoolPointer(value) ? { ...value, valueType: expression.valueType } : value;
      }
      return castNumber(expression.valueType, evalNumber(expression.expression, context));
    case "member": {
      const object = readMemberObject(expression.object, context);
      if (isComplex(object)) {
        if (expression.property === "x") return object.x;
        if (expression.property === "y") return object.y;
        throw compilerFailure(`unsupported complex member '${expression.property}'`);
      }
      if (isCudaVectorValue(object)) {
        if (expression.property === "size") return cudaVectorLaneCount(object.valueType);
        const index = cudaVectorFieldIndex(object.valueType, expression.property);
        if (index !== undefined) return object.lanes[index] ?? 0;
        throw compilerFailure(`unsupported ${object.valueType} member '${expression.property}'`);
      }
      if (expression.property === "x") return object.x;
      if (expression.property === "y") return object.y;
      if (expression.property === "z") return object.z;
      throw compilerFailure(`unsupported vector member '${expression.property}'`);
    }
    case "index": {
      const lvalue = resolveLValue(expression, context);
      return readLValue(lvalue, context);
    }
    case "unary": {
      if (expression.operator === "&") return { kind: "address", target: resolveLValue(expression.argument, context) };
      if (expression.operator === "*") return evalDeref(expression.argument, context);
      const value = evalNumber(expression.argument, context);
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      return truthy(value) ? 0 : 1;
    }
    case "binary":
      return evalVectorBinary(expression.operator, expression.left, expression.right, context) ??
        evalBinary(expression.operator, expression.left, expression.right, context);
    case "conditional":
      return truthy(evalNumber(expression.condition, context))
        ? evalExpression(expression.consequent, context)
        : evalExpression(expression.alternate, context);
    case "assignment":
      return evalAssignment(expression.operator, expression.left, expression.right, context);
    case "update": {
      const pointerRebase = evalPointerRebaseUpdate(expression, context);
      if (pointerRebase) return pointerRebase;
      const lvalue = resolveLValue(expression.argument, context);
      const current = readLValue(lvalue, context);
      const currentNumber = valueAsNumber(current, lvalue.name);
      const next = expression.operator === "++" ? currentNumber + 1 : currentNumber - 1;
      writeLValue(lvalue, next, context);
      return expression.prefix ? next : currentNumber;
    }
    case "sequence": {
      let value: EvalValue = 0;
      for (const item of expression.expressions) value = evalExpression(item, context);
      return value;
    }
    case "call":
      return evalCall(expression, context);
  }
}

function evalDeref(expression: CudaLiteExpression, context: ThreadContext): EvalValue {
  return readLValue(resolvePointerArgument(expression, context), context);
}

function evalNumber(expression: CudaLiteExpression, context: ThreadContext): number {
  return valueAsNumber(evalExpression(expression, context), expression.kind);
}

function evalVectorBinary(
  operator: string,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  context: ThreadContext,
): CudaVectorValue | undefined {
  if (operator !== "+" && operator !== "-" && operator !== "*" && operator !== "/") return undefined;
  const leftType = vectorExpressionType(leftExpression, context);
  const rightType = vectorExpressionType(rightExpression, context);
  if (!leftType && !rightType) return undefined;
  if (leftType && rightType && leftType !== rightType) return undefined;
  const valueType = leftType ?? rightType!;
  const left = leftType
    ? valueAsCudaVector(evalExpression(leftExpression, context), valueType)
    : vectorSplat(valueType, evalNumber(leftExpression, context));
  const right = rightType
    ? valueAsCudaVector(evalExpression(rightExpression, context), valueType)
    : vectorSplat(valueType, evalNumber(rightExpression, context));
  return {
    kind: "cuda-vector",
    valueType,
    lanes: left.lanes.map((value, index) => roundVectorLane(
      valueType,
      operator === "+"
        ? (value ?? 0) + (right.lanes[index] ?? 0)
        : operator === "-"
          ? (value ?? 0) - (right.lanes[index] ?? 0)
          : operator === "*"
            ? (value ?? 0) * (right.lanes[index] ?? 0)
            : (value ?? 0) / (right.lanes[index] ?? 1),
    )),
  };
}

function vectorExpressionType(
  expression: CudaLiteExpression,
  context: ThreadContext,
): CudaLiteVectorType | undefined {
  if (expression.kind === "identifier") {
    const local = context.locals.get(expression.name);
    if (isCudaVectorValue(local)) return local.valueType;
    const type = context.valueTypes.get(expression.name);
    return isCudaVectorType(type) ? type : undefined;
  }
  if (expression.kind === "index") {
    const type = pointerValueTypeForExpression(expression.target, context);
    return isCudaVectorType(type) ? type : undefined;
  }
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    const constructor = name ? cudaVectorConstructorType(name) : undefined;
    if (constructor) return constructor;
    if ((name === "tex2D" || name === "tex2DLod" || name === "tex1Dfetch") && isCudaVectorType(expression.templateValueType)) {
      return expression.templateValueType;
    }
    if (isHalf2Intrinsic(name) || name === "__float22half2_rn" || name === "__float2half2_rn" || name === "__floats2half2_rn") return "half2";
    if (name === "__half22float2") return "float2";
  }
  if (expression.kind === "binary") {
    if (expression.operator !== "+" && expression.operator !== "-" && expression.operator !== "*" && expression.operator !== "/") return undefined;
    const left = vectorExpressionType(expression.left, context);
    const right = vectorExpressionType(expression.right, context);
    if (left && right) return left === right ? left : undefined;
    return left ?? right;
  }
  return undefined;
}

function vectorSplat(valueType: CudaLiteVectorType, value: number): CudaVectorValue {
  return {
    kind: "cuda-vector",
    valueType,
    lanes: Array.from({ length: cudaVectorLaneCount(valueType) }, () => roundVectorLane(valueType, value)),
  };
}

function roundVectorLane(valueType: CudaLiteVectorType, value: number): number {
  const scalar = cudaVectorScalarType(valueType);
  if (scalar === "half") return roundHalf(value);
  if (scalar === "int") return Math.trunc(value) | 0;
  if (scalar === "uint") return Math.trunc(value) >>> 0;
  return value;
}

function castNumber(type: Exclude<CudaLiteScalarType, "void">, value: number): EvalValue {
  if (type === "int") return Math.trunc(value) | 0;
  if (type === "uint") return Math.trunc(value) >>> 0;
  if (type === "bool") return truthy(value) ? 1 : 0;
  if (type === "complex64") return { kind: "complex64", x: value, y: 0 };
  return value;
}

function readMemberObject(expression: CudaLiteExpression, context: ThreadContext): Vector3 | ComplexValue | CudaVectorValue {
  if (expression.kind === "identifier") {
    const value = readIdentifier(expression.name, context);
    if (isComplex(value)) return value;
    if (isCudaVectorValue(value)) return value;
    if (typeof value === "number" || "kind" in value) throw compilerFailure(`'${expression.name}' is not a vector`);
    return value;
  }
  const value = evalExpression(expression, context);
  if (isComplex(value)) return value;
  if (isCudaVectorValue(value)) return value;
  throw compilerFailure("member access only supports CUDA-lite builtin vectors, CUDA vector values, and complex values");
}

function readIdentifier(name: string, context: ThreadContext): LocalValue {
  if (name === "nullptr") return { kind: "pool-pointer", poolName: "", byteOffset: -1 };
  const namedConstant = CUDA_NAMED_CONSTANTS.get(name);
  if (namedConstant) return namedConstant.value;
  if (name === "threadIdx") return context.threadIdx;
  if (name === "blockIdx") return context.blockIdx;
  if (name === "blockDim") return context.blockDim;
  if (name === "gridDim") return context.gridDim;
  if (context.locals.has(name)) return context.locals.get(name)!;
  if (context.shared.has(name)) return readLValue({ name, space: "shared", index: 0 }, context);
  if (context.constants.has(name)) {
    const value = context.constants.get(name)!;
    if (typeof value === "number") return value;
  }
  if (Object.prototype.hasOwnProperty.call(context.scalars, name)) return context.scalars[name]!;
  throw compilerFailure(`unknown identifier '${name}'`);
}

function evalBinary(
  operator: string,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  context: ThreadContext,
): number {
  const pointerEquality = evalPointerEquality(operator, leftExpression, rightExpression, context);
  if (pointerEquality !== undefined) return pointerEquality;
  if (operator === "&&") {
    return truthy(evalNumber(leftExpression, context)) && truthy(evalNumber(rightExpression, context)) ? 1 : 0;
  }
  if (operator === "||") {
    return truthy(evalNumber(leftExpression, context)) || truthy(evalNumber(rightExpression, context)) ? 1 : 0;
  }
  const left = evalNumber(leftExpression, context);
  const right = evalNumber(rightExpression, context);
  switch (operator) {
    case "+":
      return left + right;
    case "-":
      return left - right;
    case "*":
      return left * right;
    case "/":
      return left / right;
    case "%":
      return left % right;
    case "<<":
      return Math.trunc(left) << Math.trunc(right);
    case ">>":
      return Math.trunc(left) >> Math.trunc(right);
    case "&":
      return Math.trunc(left) & Math.trunc(right);
    case "^":
      return Math.trunc(left) ^ Math.trunc(right);
    case "|":
      return Math.trunc(left) | Math.trunc(right);
    case "<":
      return left < right ? 1 : 0;
    case "<=":
      return left <= right ? 1 : 0;
    case ">":
      return left > right ? 1 : 0;
    case ">=":
      return left >= right ? 1 : 0;
    case "==":
      return left === right ? 1 : 0;
    case "!=":
      return left !== right ? 1 : 0;
    default:
      throw compilerFailure(`unsupported binary operator '${operator}'`);
  }
}

function evalPointerEquality(
  operator: string,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  context: ThreadContext,
): number | undefined {
  if (operator !== "==" && operator !== "!=") return undefined;
  const left = pointerValueForEquality(leftExpression, context);
  const right = pointerValueForEquality(rightExpression, context);
  if (left === undefined || right === undefined) return undefined;
  const equal = pointerIdentity(left) === pointerIdentity(right);
  return operator === "==" ? (equal ? 1 : 0) : (equal ? 0 : 1);
}

function pointerValueForEquality(expression: CudaLiteExpression, context: ThreadContext): EvalValue | undefined {
  if (expression.kind === "number") return expression.value === 0 ? 0 : undefined;
  if (expression.kind === "identifier") {
    if (expression.name === "nullptr") return { kind: "pool-pointer", poolName: "", byteOffset: -1 };
    const namedConstant = CUDA_NAMED_CONSTANTS.get(expression.name);
    if (namedConstant?.valueType === "voidptr" && namedConstant.value === 0) return 0;
    const pointer = mutablePointerValue(expression.name, context);
    if (pointer) return pointer;
    return undefined;
  }
  if (
    (expression.kind === "cast" && expression.pointer) ||
    expression.kind === "binary" ||
    (expression.kind === "unary" && expression.operator === "&")
  ) {
    try {
      const pointer = pointerArgumentValue(expression, pointerValueTypeForExpression(expression, context), context);
      if (isAddress(pointer) || isPoolPointer(pointer) || typeof pointer === "number") return pointer;
      return undefined;
    } catch {
      return undefined;
    }
  }
  return undefined;
}

function pointerIdentity(value: EvalValue): string {
  if (typeof value === "number") return Math.trunc(value) === 0 ? "null" : `scalar:${Math.trunc(value)}`;
  if (isAddress(value)) return `${value.target.space}:${value.target.name}:${value.target.index ?? 0}`;
  if (isPoolPointer(value)) {
    return value.byteOffset < 0 || value.poolName.length === 0
      ? "null"
      : `pool:${value.poolName}:${value.byteOffset}`;
  }
  return "unknown";
}

function evalAssignment(
  operator: string,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  context: ThreadContext,
): EvalValue {
  const pointerRebase = evalPointerRebaseAssignment(operator, leftExpression, rightExpression, context);
  if (pointerRebase) return pointerRebase;
  const lvalue = resolveLValue(leftExpression, context);
  const right = evalExpression(rightExpression, context);
  if (isPoolPointer(right)) {
    if (operator !== "=") throw compilerFailure("pool pointers only support assignment");
    writeLValue(lvalue, right, context);
    return right;
  }
  if (isComplex(right)) {
    if (operator !== "=") throw compilerFailure("complex values only support assignment");
    writeLValue(lvalue, right, context);
    return right;
  }
  if (isCudaVectorValue(right)) {
    if (operator !== "=") {
      const current = readLValue(lvalue, context);
      if (!isCudaVectorValue(current)) throw compilerFailure("CUDA vector compound assignment expects a CUDA vector target");
      const value = applyVectorOperator(operator, current, right);
      writeLValue(lvalue, value, context);
      return value;
    }
    writeLValue(lvalue, right, context);
    return right;
  }
  if (isAddress(right)) {
    if (operator !== "=") throw compilerFailure("addresses only support assignment");
    writeLValue(lvalue, right, context);
    return right;
  }
  const rightNumber = valueAsNumber(right, lvalue.name);
  if (operator !== "=") {
    const currentValue = readLValue(lvalue, context);
    if (isCudaVectorValue(currentValue)) {
      const value = applyVectorOperator(operator, currentValue, vectorSplat(currentValue.valueType, rightNumber));
      writeLValue(lvalue, value, context);
      return value;
    }
  }
  const current = operator === "=" ? 0 : valueAsNumber(readLValue(lvalue, context), lvalue.name);
  const value: number = operator === "="
    ? rightNumber
    : operator === "+="
      ? current + rightNumber
    : operator === "-="
      ? current - rightNumber
      : operator === "*="
        ? current * rightNumber
        : operator === "/="
          ? current / rightNumber
          : operator === "<<="
            ? Math.trunc(current) << Math.trunc(rightNumber)
            : Math.trunc(current) >> Math.trunc(rightNumber);
  writeLValue(lvalue, value, context);
  return value;
}

function applyVectorOperator(operator: string, current: CudaVectorValue, right: CudaVectorValue): CudaVectorValue {
  if (current.valueType !== right.valueType) throw compilerFailure("CUDA vector compound assignment expects matching vector types");
  if (operator !== "+=" && operator !== "-=" && operator !== "*=" && operator !== "/=") {
    throw compilerFailure("CUDA vector compound assignment supports +=, -=, *=, and /=");
  }
  const op = operator.slice(0, -1);
  return {
    kind: "cuda-vector",
    valueType: current.valueType,
    lanes: current.lanes.map((value, index) => roundVectorLane(
      current.valueType,
      op === "+"
        ? (value ?? 0) + (right.lanes[index] ?? 0)
        : op === "-"
          ? (value ?? 0) - (right.lanes[index] ?? 0)
          : op === "*"
            ? (value ?? 0) * (right.lanes[index] ?? 0)
            : (value ?? 0) / (right.lanes[index] ?? 1),
    )),
  };
}

function evalPointerRebaseAssignment(
  operator: string,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  context: ThreadContext,
): AddressValue | PoolPointerValue | undefined {
  if (leftExpression.kind !== "identifier" || (operator !== "+=" && operator !== "-=")) return undefined;
  const current = mutablePointerValue(leftExpression.name, context);
  if (!current) return undefined;
  const valueType = pointerValueTypeForExpression(leftExpression, context);
  const delta = Math.trunc(evalNumber(rightExpression, context)) * (operator === "+=" ? 1 : -1);
  const next = offsetPointerValue(current, delta, valueType);
  context.locals.set(leftExpression.name, next);
  return next;
}

function evalPointerRebaseUpdate(
  expression: Extract<CudaLiteExpression, { kind: "update" }>,
  context: ThreadContext,
): AddressValue | PoolPointerValue | undefined {
  if (expression.argument.kind !== "identifier") return undefined;
  const current = mutablePointerValue(expression.argument.name, context);
  if (!current) return undefined;
  const valueType = pointerValueTypeForExpression(expression.argument, context);
  const next = offsetPointerValue(current, expression.operator === "++" ? 1 : -1, valueType);
  context.locals.set(expression.argument.name, next);
  return expression.prefix ? next : current;
}

function mutablePointerValue(name: string, context: ThreadContext): AddressValue | PoolPointerValue | undefined {
  const local = context.locals.get(name);
  if (isAddress(local) || isPoolPointer(local)) return local;
  const valueType = context.valueTypes.get(name);
  if (context.buffers.has(name)) return { kind: "address", target: { name, space: "buffer", index: 0, ...(valueType ? { valueType } : {}) } };
  if (context.shared.has(name)) return { kind: "address", target: { name, space: "shared", index: 0, ...(valueType ? { valueType } : {}) } };
  if (context.memoryPools.has(name)) return { kind: "pool-pointer", poolName: name, byteOffset: 0, ...(valueType ? { valueType } : {}) };
  return undefined;
}

function offsetPointerValue(
  value: AddressValue | PoolPointerValue,
  delta: number,
  valueType: CudaLiteScalarType,
): AddressValue | PoolPointerValue {
  if (isPoolPointer(value)) return { ...value, byteOffset: value.byteOffset + delta * elementByteSize(valueType), valueType };
  const storageDelta = value.target.rawStorageIndex ? delta * valueStorageWidth(valueType) : delta;
  return {
    kind: "address",
    target: {
      ...value.target,
      index: (value.target.index ?? 0) + storageDelta,
      valueType: value.target.valueType ?? valueType,
    },
  };
}

function fillLocalArray(expression: Extract<CudaLiteExpression, { kind: "call" }>, context: ThreadContext): void {
  const target = expression.args[0];
  const value = expression.args[1];
  if (target?.kind !== "identifier" || !value) throw compilerFailure("fill_*D_regs expects local array and value");
  const local = context.locals.get(target.name);
  if (!isLocalArray(local)) throw compilerFailure(`'${target.name}' is not a local array`);
  const fill = evalExpression(value, context);
  for (let index = 0; index < local.data.length; index += valueStorageWidth(local.valueType)) {
    writeBufferValue(local.data, index, local.valueType, undefined, fill);
  }
}

function evalCall(expression: Extract<CudaLiteExpression, { kind: "call" }>, context: ThreadContext): EvalValue {
  const name = expression.kind === "call" && expression.callee.kind === "identifier"
    ? expression.callee.name
    : undefined;
  const cooperativeGroupCall = evalCooperativeGroupCall(expression, context);
  if (cooperativeGroupCall !== undefined) return cooperativeGroupCall;
  if (name === "printf") return 0;
  if (name === "div_ceil") {
    const numerator = evalNumber(expression.args[0]!, context);
    const denominator = evalNumber(expression.args[1]!, context);
    return denominator === 0 ? 0 : Math.trunc((numerator + denominator - 1) / denominator);
  }
  if (name === "fill_1D_regs" || name === "fill_2D_regs" || name === "fill_3D_regs") {
    fillLocalArray(expression, context);
    return 0;
  }
  if (name !== undefined && isHostManagedRuntimeNoopCall(name)) return 0;
  if (name === "cudaMemcpy" || name === "cudaMemcpyAsync" || name === "cudaMemcpyPeerAsync") {
    execCudaRuntimeCopy(expression, context);
    return 0;
  }
  if (name === "deviceAllocate" || name === "streamOrderedAllocate") {
    if (expression.args.length === 4) {
      const baseRef = expression.args[0];
      if (baseRef?.kind !== "identifier") throw compilerFailure(`${name} expects raw pool base`);
      const offsetRef = expression.args[1];
      if (!offsetRef) throw compilerFailure(`${name} expects raw pool offset`);
      const offset = resolveLValue(offsetRef, context);
      const oldOffset = valueAsNumber(readLValue(offset, context), offset.name);
      const poolSize = Math.max(0, Math.trunc(evalNumber(expression.args[2]!, context)));
      const sizeBytes = Math.max(0, Math.trunc(evalNumber(expression.args[3]!, context)));
      writeLValue(offset, oldOffset + sizeBytes, context);
      if (oldOffset + sizeBytes > poolSize) {
        return { kind: "pool-pointer", poolName: baseRef.name, byteOffset: -1, rawBuffer: true };
      }
      return { kind: "pool-pointer", poolName: baseRef.name, byteOffset: oldOffset, rawBuffer: true };
    }
    const poolName = poolNameFromAllocatorArg(expression.args[0], context);
    if (!poolName) throw compilerFailure(`${name} expects DevicePool* argument`);
    const pool = context.memoryPools.get(poolName);
    if (!pool) throw compilerFailure(`missing memory pool '${poolName}'`);
    const sizeBytes = Math.max(0, Math.trunc(evalNumber(expression.args[1]!, context)));
    const oldOffset = pool.offset;
    pool.offset += sizeBytes;
    if (oldOffset + sizeBytes > pool.data.length * 4) {
      return { kind: "pool-pointer", poolName, byteOffset: -1 };
    }
    return { kind: "pool-pointer", poolName, byteOffset: oldOffset };
  }
  if (name === "atomicAdd" || name === "atomicAdd_system") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => current + value);
  }
  if (name === "atomicSub") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => current - value);
  }
  if (name === "atomicMin" || name === "atomicMin_system") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => Math.min(current, value));
  }
  if (name === "atomicMax" || name === "atomicMax_system" || name === "atomicMaxFloat") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => Math.max(current, value));
  }
  if (name === "atomicAnd" || name === "atomicAnd_system") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => Math.trunc(current) & Math.trunc(value));
  }
  if (name === "atomicOr" || name === "atomicOr_system") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => Math.trunc(current) | Math.trunc(value));
  }
  if (name === "atomicXor" || name === "atomicXor_system") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => Math.trunc(current) ^ Math.trunc(value));
  }
  if (name === "atomicInc" || name === "atomicInc_system") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => {
      const old = current >>> 0;
      const limit = value >>> 0;
      return old >= limit ? 0 : (old + 1) >>> 0;
    });
  }
  if (name === "atomicDec" || name === "atomicDec_system") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => {
      const old = current >>> 0;
      const limit = value >>> 0;
      return old === 0 || old > limit ? limit : (old - 1) >>> 0;
    });
  }
  if (name === "atomicExch" || name === "atomicExch_system") {
    return evalAtomicReadModifyWrite(expression, context, (_current, value) => value);
  }
  if (name === "atomicCAS" || name === "atomicCAS_system") {
    const first = expression.args[0];
    if (!first) throw compilerFailure("atomicCAS expects target");
    const lvalue = resolveAtomicTarget(first, context);
    const current = valueAsNumber(readLValue(lvalue, context), lvalue.name);
    const compare = evalNumber(expression.args[1]!, context);
    const value = evalNumber(expression.args[2]!, context);
    if (Math.trunc(current) === Math.trunc(compare)) writeLValue(lvalue, value, context);
    return current;
  }
  if (name === "tex2D" || name === "tex2DLod" || name === "tex1Dfetch") {
    const textureRef = expression.args[0];
    if (textureRef?.kind !== "identifier") throw compilerFailure(`${name} expects texture reference`);
    const texture = context.textures[textureRef.name];
    if (!texture) throw compilerFailure(`missing texture input '${textureRef.name}'`);
    const x = Math.max(0, Math.min(texture.width - 1, Math.floor(evalNumber(expression.args[1]!, context))));
    const y = name === "tex1Dfetch"
      ? 0
      : Math.max(0, Math.min(texture.height - 1, Math.floor(evalNumber(expression.args[2]!, context))));
    const valueType = expression.templateValueType ?? "float";
    if (isCudaVectorType(valueType)) return readTextureVector(texture, x, y, valueType);
    return texture.data[(y * texture.width + x) * textureChannels(texture)] ?? 0;
  }
  if (name === "surf2Dread") {
    const target = expression.args[0];
    const surfaceRef = expression.args[1];
    if (!target) throw compilerFailure("surf2Dread expects output target");
    if (surfaceRef?.kind !== "identifier") throw compilerFailure("surf2Dread expects surface reference");
    const surface = context.surfaces[surfaceRef.name];
    if (!surface) throw compilerFailure(`missing surface input '${surfaceRef.name}'`);
    const x = Math.trunc(evalNumber(expression.args[2]!, context) / 4);
    const y = Math.trunc(evalNumber(expression.args[3]!, context));
    const index = y * surface.width + x;
    const ok = x >= 0 && y >= 0 && x < surface.width && y < surface.height && index < surface.data.length;
    const value = ok ? surface.data[index] ?? 0 : 0;
    const lvalue = target.kind === "unary" && target.operator === "&"
      ? resolveLValue(target.argument, context)
      : resolveLValue(target, context);
    writeLValue(lvalue, value, context);
    context.trace.reads.push({ name: surfaceRef.name, index, value, ok });
    return 0;
  }
  if (name === "surf2Dwrite") {
    const surfaceRef = expression.args[1];
    if (surfaceRef?.kind !== "identifier") throw compilerFailure("surf2Dwrite expects surface reference");
    const surface = context.surfaces[surfaceRef.name];
    if (!surface) throw compilerFailure(`missing surface input '${surfaceRef.name}'`);
    const value = evalNumber(expression.args[0]!, context);
    const x = Math.trunc(evalNumber(expression.args[2]!, context) / 4);
    const y = Math.trunc(evalNumber(expression.args[3]!, context));
    const index = y * surface.width + x;
    const ok = x >= 0 && y >= 0 && x < surface.width && y < surface.height && index < surface.data.length;
    if (ok) surface.data[index] = value;
    context.trace.writes.push({ name: surfaceRef.name, index, value, ok });
    return 0;
  }
  if (name === "sizeof") {
    const target = expression.args[0];
    return target?.kind === "identifier" ? sizeofCudaType(target.name) ?? 4 : 4;
  }
  if (name === "alignof") {
    const target = expression.args[0];
    return target?.kind === "identifier" ? alignofCudaType(target.name) ?? 4 : 4;
  }
  if (name === "vec_at") {
    const vector = expression.args[0];
    const index = expression.args[1];
    if (!vector || !index) throw compilerFailure("vec_at expects vector and index arguments");
    const value = evalExpression(vector, context);
    if (!isCudaVectorValue(value)) throw compilerFailure("vec_at expects a CUDA vector value");
    const lane = Math.max(0, Math.min(value.lanes.length - 1, Math.trunc(evalNumber(index, context))));
    return value.lanes[lane] ?? 0;
  }
  if (name === "curand_init") {
    const state = expression.args[3];
    if (!state) throw compilerFailure("curand_init expects state address");
    const seed = evalNumber(expression.args[0]!, context) >>> 0;
    const sequence = evalNumber(expression.args[1]!, context) >>> 0;
    const offset = evalNumber(expression.args[2]!, context) >>> 0;
    const initialized = curandNext((seed ^ Math.imul(sequence, 747796405) ^ offset ^ 2891336453) >>> 0);
    writeLValue(resolveAddressArgument(state, context), initialized, context);
    return 0;
  }
  if (name === "curand_uniform") {
    const state = expression.args[0];
    if (!state) throw compilerFailure("curand_uniform expects state address");
    const lvalue = resolveAddressArgument(state, context);
    const next = curandNext(valueAsNumber(readLValue(lvalue, context), lvalue.name) >>> 0);
    writeLValue(lvalue, next, context);
    return (next + 1) * 2.3283064365386963e-10;
  }
  if (name === "__ldcs") {
    const target = expression.args[0];
    if (!target) throw compilerFailure("__ldcs expects pointer argument");
    return readLValue(resolvePointerArgument(target, context), context);
  }
  if (name === "__stcs") {
    const target = expression.args[0];
    const value = expression.args[1];
    if (!target || !value) throw compilerFailure("__stcs expects pointer and value arguments");
    writeLValue(resolvePointerArgument(target, context), evalExpression(value, context), context);
    return 0;
  }
  if (name === "__cvta_generic_to_shared") {
    const target = expression.args[0];
    if (!target) throw compilerFailure("__cvta_generic_to_shared expects pointer argument");
    return lvalueStorageIndex(resolvePointerArgument(target, context), context);
  }
  if (isPointerIdentityCall(name)) {
    const pointer = expression.args[0];
    if (!pointer) throw compilerFailure("pointer identity call expects pointer argument");
    const value = pointerArgumentValue(pointer, pointerValueTypeForExpression(pointer, context), context);
    if (typeof value === "number" || isAddress(value) || isPoolPointer(value)) return value;
    throw compilerFailure("pointer identity call expects pointer argument");
  }
  if (isCpAsyncCopyCall(name) || isCpAsyncFenceCall(name)) return 0;
  const vectorConstructor = name ? cudaVectorConstructorType(name) : undefined;
  if (vectorConstructor) {
    const lanes = expression.args.map((arg) => evalNumber(arg, context));
    const scalar = lanes[0] ?? 0;
    return {
      kind: "cuda-vector",
      valueType: vectorConstructor,
      lanes: lanes.length === 1 ? Array(cudaVectorLaneCount(vectorConstructor)).fill(scalar) : lanes,
    };
  }
  if (isHalf2Intrinsic(name)) {
    const left = valueAsCudaVector(evalExpression(expression.args[0]!, context), "half2");
    const right = valueAsCudaVector(evalExpression(expression.args[1]!, context), "half2");
    const addend = name === "__hfma2"
      ? valueAsCudaVector(evalExpression(expression.args[2]!, context), "half2")
      : undefined;
    const op = half2IntrinsicOperator(name);
    return {
      kind: "cuda-vector",
      valueType: "half2",
      lanes: left.lanes.map((value, index) => roundHalf(op(value ?? 0, right.lanes[index] ?? 0) + (addend?.lanes[index] ?? 0))),
    };
  }
  if (name === "__half22float2") {
    const value = valueAsCudaVector(evalExpression(expression.args[0]!, context), "half2");
    return { kind: "cuda-vector", valueType: "float2", lanes: value.lanes };
  }
  if (name === "__low2float" || name === "__high2float") {
    const value = valueAsCudaVector(evalExpression(expression.args[0]!, context), "half2");
    return value.lanes[name === "__low2float" ? 0 : 1] ?? 0;
  }
  if (name === "__float22half2_rn") {
    const value = valueAsCudaVector(evalExpression(expression.args[0]!, context), "float2");
    return { kind: "cuda-vector", valueType: "half2", lanes: value.lanes.map((lane) => roundHalf(lane ?? 0)) };
  }
  if (name === "__float2half2_rn") {
    const value = roundHalf(evalNumber(expression.args[0]!, context));
    return { kind: "cuda-vector", valueType: "half2", lanes: [value, value] };
  }
  if (name === "__floats2half2_rn") {
    return {
      kind: "cuda-vector",
      valueType: "half2",
      lanes: [
        roundHalf(evalNumber(expression.args[0]!, context)),
        roundHalf(evalNumber(expression.args[1]!, context)),
      ],
    };
  }
  const deviceFunction = name ? context.functions.get(name) : undefined;
  if (deviceFunction) return evalDeviceFunction(
    deviceFunction,
    deviceFunctionArgs(deviceFunction, expression.args, context),
    context,
  );
  const args = expression.args.map((arg) => evalNumber(arg, context));
  const intrinsic = name ? CUDA_INTRINSICS_BY_NAME.get(name) : undefined;
  if (intrinsic?.evaluate) return intrinsic.evaluate(args);
  switch (name) {
    case "__syncthreads":
    case "__syncwarp":
      return 0;
    case "__threadfence":
      return 0;
    case "clock":
      return context.blockIdx.x * 104729 +
        context.blockIdx.y * 1009 +
        context.blockIdx.z * 97 +
        context.threadIdx.x +
        context.threadIdx.y * 31 +
        context.threadIdx.z * 7;
    case "clock64":
      return context.blockIdx.x * 104729 +
        context.blockIdx.y * 1009 +
        context.blockIdx.z * 97 +
        context.threadIdx.x +
        context.threadIdx.y * 31 +
        context.threadIdx.z * 7;
    case "min":
      return Math.min(...args);
    case "max":
      return Math.max(...args);
    case "bg_subgroup_add":
      return args[0] ?? 0;
    case "__shfl_sync":
    case "__shfl_down_sync":
    case "__shfl_up_sync":
    case "__shfl_xor_sync":
      return args[1] ?? 0;
    case "__any_sync":
    case "__all_sync":
      return truthy(args[1] ?? 0) ? 1 : 0;
    case "warpReduceSum":
    case "warpReduceMax":
    case "warpReduceMin":
    case "warp_reduce_sum":
    case "warp_reduce_max":
    case "warp_reduce_min":
    case "warp_reduce_sum_f32":
    case "warp_reduce_max_f32":
    case "warp_reduce_sum_f16":
    case "warp_reduce_sum_f16_f16":
    case "warp_reduce_sum_f16_f32":
    case "blockReduce":
      return args[0] ?? 0;
    default:
      throw compilerFailure(`unsupported call '${name ?? "<expr>"}'`);
  }
}

function isHalf2Intrinsic(name: string | undefined): boolean {
  return name === "__hadd2" ||
    name === "__hsub2" ||
    name === "__hmul2" ||
    name === "__hfma2" ||
    name === "__hmin2" ||
    name === "__hmax2";
}

function half2IntrinsicOperator(name: string | undefined): (left: number, right: number) => number {
  switch (name) {
    case "__hadd2":
      return (left, right) => left + right;
    case "__hsub2":
      return (left, right) => left - right;
    case "__hmul2":
    case "__hfma2":
      return (left, right) => left * right;
    case "__hmin2":
      return Math.min;
    case "__hmax2":
      return Math.max;
    default:
      throw compilerFailure(`unsupported half2 intrinsic '${name ?? "<expr>"}'`);
  }
}

function roundHalf(value: number): number {
  return float16BitsToFloat32(float32ToFloat16Bits(value));
}

function isHostManagedRuntimeNoopCall(name: string): boolean {
  return name === "cudaDeviceSynchronize" ||
    name === "cudaStreamCreate" ||
    name === "cudaStreamCreateWithFlags" ||
    name === "cudaStreamDestroy" ||
    name === "cudaStreamSynchronize" ||
    name === "cudaEventCreate" ||
    name === "cudaEventCreateWithFlags" ||
    name === "cudaEventDestroy" ||
    name === "cudaEventRecord" ||
    name === "cudaEventSynchronize";
}

function readTextureVector(
  texture: { readonly width: number; readonly data: Float32Array; readonly channels?: number },
  x: number,
  y: number,
  valueType: CudaLiteVectorType,
): CudaVectorValue {
  const channels = textureChannels(texture);
  const base = (y * texture.width + x) * channels;
  const lanes = Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) => {
    if (lane < channels) return texture.data[base + lane] ?? 0;
    return lane === 3 ? 1 : 0;
  });
  return { kind: "cuda-vector", valueType, lanes };
}

function textureChannels(texture: { readonly width: number; readonly data: Float32Array; readonly channels?: number }): number {
  return texture.channels === 2 || texture.channels === 4 ? texture.channels : 1;
}

function poolNameFromAllocatorArg(expression: CudaLiteExpression | undefined, context: ThreadContext): string | undefined {
  if (expression?.kind === "identifier") {
    const local = context.locals.get(expression.name);
    if (isPoolPointer(local)) return local.poolName;
    return expression.name;
  }
  if (expression?.kind === "unary" && expression.operator === "&" && expression.argument.kind === "identifier") {
    return expression.argument.name;
  }
  return undefined;
}

function evalCooperativeGroupCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  context: ThreadContext,
): number | undefined {
  const namespaceCall = evalCooperativeNamespaceCall(expression, context);
  if (namespaceCall !== undefined) return namespaceCall;
  const callee = expression.callee;
  if (callee.kind !== "member" || callee.object.kind !== "identifier") return undefined;
  const value = context.locals.get(callee.object.name);
  if (!isCooperativeGroup(value)) return undefined;
  const args = expression.args.map((arg) => evalNumber(arg, context));
  if (callee.property === "sync") return 0;
  if (callee.property === "size") {
    if (value.groupKind === "tile") return value.tileSize ?? 32;
    return context.blockDim.x * context.blockDim.y * context.blockDim.z;
  }
  if (callee.property === "thread_rank") {
    const localRank = localLinearRank(context);
    if (value.groupKind === "tile") return localRank % (value.tileSize ?? 32);
    return localRank;
  }
  if (callee.property === "meta_group_size") {
    if (value.groupKind !== "tile") return 1;
    const blockSize = context.blockDim.x * context.blockDim.y * context.blockDim.z;
    return Math.ceil(blockSize / (value.tileSize ?? 32));
  }
  if (callee.property === "meta_group_rank") {
    if (value.groupKind !== "tile") return 0;
    return Math.floor(localLinearRank(context) / (value.tileSize ?? 32));
  }
  if (callee.property === "shfl_down" || callee.property === "shfl_up" || callee.property === "shfl_xor") return valueAsNumber(args[0] ?? 0, "shuffle value");
  return undefined;
}

function evalCooperativeNamespaceCall(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  context: ThreadContext,
): number | undefined {
  const name = expressionNameForReference(expression.callee);
  if (!name?.endsWith("::sync") && !name?.endsWith("::reduce")) return undefined;
  const groupArg = expression.args[0];
  if (groupArg?.kind !== "identifier") return undefined;
  const value = context.locals.get(groupArg.name);
  if (!isCooperativeGroup(value)) return undefined;
  if (name.endsWith("::sync")) return 0;
  const reduced = expression.args[1];
  const item = reduced ? evalNumber(reduced, context) : 0;
  const op = expression.args[2] ? expressionNameForReference(expression.args[2]) : undefined;
  if (op?.endsWith("::plus")) {
    const size = value.groupKind === "tile"
      ? value.tileSize ?? 32
      : context.blockDim.x * context.blockDim.y * context.blockDim.z;
    return item * size;
  }
  return item;
}

function localLinearRank(context: ThreadContext): number {
  return context.threadIdx.x +
    context.threadIdx.y * context.blockDim.x +
    context.threadIdx.z * context.blockDim.x * context.blockDim.y;
}

function deviceFunctionArgs(
  fn: CudaLiteDeviceFunction,
  args: readonly CudaLiteExpression[],
  context: ThreadContext,
): readonly EvalValue[] {
  return fn.params.map((param, index) => {
    const arg = args[index];
    if (!arg) return zeroParamLocalValue(param);
    if (param.cooperativeGroupKind !== undefined) return cooperativeGroupArgumentValue(arg, param, context);
    if (!param.pointer) {
      const value = evalExpression(arg, context);
      if (isCudaVectorType(param.valueType)) return value;
      return castNumber(param.valueType, valueAsNumber(value, param.name));
    }
    const value = pointerArgumentValue(arg, param.valueType, context);
    if (isAddress(value) || isPoolPointer(value)) return value;
    throw compilerFailure(`device pointer parameter '${param.name}' expects a pointer argument`);
  });
}

function evalDeviceFunction(fn: CudaLiteDeviceFunction, args: readonly EvalValue[], context: ThreadContext): EvalValue {
  const locals = new Map<string, LocalValue>();
  for (const [index, param] of fn.params.entries()) locals.set(param.name, args[index] ?? zeroParamLocalValue(param));
  const fnContext: ThreadContext = {
    ...context,
    locals,
  };
  const generator = execStatements(fn.body, fnContext);
  while (true) {
    const next = generator.next();
    if (next.done) {
      const control = next.value;
      if (control?.kind === "return") return control.value ?? zeroLocalValue(fn.returnType);
      return zeroLocalValue(fn.returnType);
    }
    throw compilerFailure(`device function '${fn.name}' cannot contain __syncthreads()`);
  }
}

function cooperativeGroupArgumentValue(
  arg: CudaLiteExpression,
  param: CudaLiteParam,
  context: ThreadContext,
): CooperativeGroupValue {
  const value = arg.kind === "identifier" ? readIdentifier(arg.name, context) : evalExpression(arg, context);
  if (isCooperativeGroup(value)) return value;
  return {
    kind: "cooperative-group",
    groupKind: param.cooperativeGroupKind ?? "block",
    ...(param.tileSize === undefined ? {} : { tileSize: param.tileSize }),
  };
}

function evalAtomicReadModifyWrite(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  context: ThreadContext,
  op: (current: number, value: number) => number,
): number {
  const first = expression.args[0];
  if (!first) throw compilerFailure("atomic operation expects target");
  const lvalue = resolveAtomicTarget(first, context);
  const current = valueAsNumber(readLValue(lvalue, context), lvalue.name);
  const value = evalNumber(expression.args[1]!, context);
  writeLValue(lvalue, op(current, value), context);
  return current;
}

function resolveAtomicTarget(expression: CudaLiteExpression, context: ThreadContext): LValue {
  if (expression.kind === "cast" && expression.pointer) {
    const pointer = pointerArgumentValue(expression, expression.valueType, context);
    if (isAddress(pointer)) return pointer.target;
  }
  if (expression.kind === "unary" && expression.operator === "&") return resolveLValue(expression.argument, context);
  const pointer = pointerArgumentValue(expression, pointerValueTypeForExpression(expression, context), context);
  if (isAddress(pointer)) return pointer.target;
  throw compilerFailure("atomic operation expects storage or shared address");
}

function resolveAddressArgument(expression: CudaLiteExpression, context: ThreadContext): LValue {
  if (expression.kind !== "unary" || expression.operator !== "&") {
    throw compilerFailure("expected address argument");
  }
  return resolveLValue(expression.argument, context);
}

function resolvePointerArgument(expression: CudaLiteExpression, context: ThreadContext): LValue {
  if (expression.kind === "unary" && expression.operator === "&") return resolveLValue(expression.argument, context);
  const valueType = pointerValueTypeForExpression(expression, context);
  const pointer = pointerArgumentValue(expression, valueType, context);
  if (isPoolPointer(pointer)) {
    return {
      name: pointer.poolName,
      space: "pool",
      index: Math.trunc(pointer.byteOffset / elementByteSize(valueType)),
      valueType,
    };
  }
  if (typeof pointer !== "number" && "kind" in pointer && pointer.kind === "address") return pointer.target;
  throw compilerFailure("expected pointer argument");
}

function curandNext(state: number): number {
  return (Math.imul(state >>> 0, 1664525) + 1013904223) >>> 0;
}

function resolveLValue(expression: CudaLiteExpression, context: ThreadContext): LValue {
  const pointerCast = resolvePointerCastLValue(expression, context);
  if (pointerCast) return pointerCast;
  const pool = resolvePoolLValue(expression, context);
  if (pool) return pool;
  if (expression.kind === "unary" && expression.operator === "*") {
    return resolvePointerArgument(expression.argument, context);
  }
  if (expression.kind === "member") {
    if (expression.property !== "x" && expression.property !== "y") {
      const info = expressionValueType(expression.object, context);
      if (!isCudaVectorType(info) || cudaVectorFieldIndex(info, expression.property) === undefined) {
        throw compilerFailure(`unsupported lvalue member '${expression.property}'`);
      }
    }
    const field = expression.property as NonNullable<LValue["field"]>;
    return { ...resolveLValue(expression.object, context), field };
  }
  if (expression.kind === "identifier") {
    if (context.buffers.has(expression.name)) return { name: expression.name, space: "buffer", index: 0 };
    if (context.shared.has(expression.name)) return { name: expression.name, space: "shared", index: 0 };
    return { name: expression.name, space: "local" };
  }
  const chain: number[] = [];
  let cursor: CudaLiteExpression = expression;
  while (cursor.kind === "index") {
    chain.unshift(Math.trunc(evalNumber(cursor.index, context)));
    cursor = cursor.target;
  }
  if (cursor.kind !== "identifier") throw compilerFailure("unsupported lvalue");
  const alias = context.locals.get(cursor.name);
  if (alias && typeof alias !== "number" && "kind" in alias && alias.kind === "address") {
    if (chain.length !== 1) throw compilerFailure(`pointer alias '${cursor.name}' expects one-dimensional indexing`);
    const index = (alias.target.index ?? 0) + chain[0]! * valueStorageWidth(alias.target.valueType);
    return {
      name: alias.target.name,
      space: alias.target.space,
      index,
      ...(alias.target.valueType === undefined ? {} : { valueType: alias.target.valueType }),
      rawStorageIndex: true,
    };
  }
  if (isPoolPointer(alias)) {
    if (chain.length !== 1) throw compilerFailure(`pool pointer alias '${cursor.name}' expects one-dimensional indexing`);
    return {
      name: alias.poolName,
      space: "pool",
      index: alias.byteOffset < 0 ? -1 : Math.trunc(alias.byteOffset / 4) + chain[0]!,
      ...(alias.valueType === undefined ? {} : { valueType: alias.valueType }),
    };
  }
  if (isCudaVectorValue(alias)) {
    if (chain.length !== 1) throw compilerFailure(`CUDA vector '${cursor.name}' expects one-dimensional indexing`);
    return { name: cursor.name, space: "local", fieldIndex: chain[0]!, valueType: alias.valueType };
  }
  if (isLocalArray(alias)) {
    return { name: cursor.name, space: "local", index: flattenIndex(alias.dimensions, chain) };
  }
  const shared = context.shared.get(cursor.name);
  if (shared) {
    return { name: cursor.name, space: "shared", index: flattenIndex(shared.dimensions, chain) };
  }
  const constant = context.constants.get(cursor.name);
  if (constant && typeof constant !== "number") {
    const dimensions = contextConstantDimensions(cursor.name, context);
    return { name: cursor.name, space: "constant", index: flattenIndex(dimensions, chain) };
  }
  if (context.buffers.has(cursor.name)) {
    if (chain.length !== 1) throw compilerFailure(`buffer '${cursor.name}' expects one-dimensional indexing`);
    return { name: cursor.name, space: "buffer", index: chain[0]! };
  }
  throw compilerFailure(`unknown lvalue '${cursor.name}'`);
}

function resolvePointerCastLValue(expression: CudaLiteExpression, context: ThreadContext): LValue | undefined {
  if (expression.kind !== "index") return undefined;
  const target = expression.target;
  if (target.kind !== "cast" || !target.pointer) return undefined;
  const pointer = pointerArgumentValue(target.expression, target.valueType, context);
  const index = Math.trunc(evalNumber(expression.index, context)) * valueStorageWidth(target.valueType);
  if (isPoolPointer(pointer)) {
    return {
      name: pointer.poolName,
      space: "pool",
      index: Math.trunc(pointer.byteOffset / 4) + index,
      valueType: target.valueType,
    };
  }
  if (typeof pointer !== "number" && "kind" in pointer && pointer.kind === "address") {
    return {
      name: pointer.target.name,
      space: pointer.target.space,
      index: lvalueStorageIndex(pointer.target, context) + index,
      valueType: target.valueType,
      rawStorageIndex: true,
    };
  }
  throw compilerFailure("pointer cast expects a pointer argument");
}

function lvalueStorageIndex(lvalue: LValue, context: ThreadContext): number {
  if (lvalue.index === undefined) return 0;
  if (lvalue.rawStorageIndex) return lvalue.index;
  if (lvalue.valueType) return lvalue.index * valueStorageWidth(lvalue.valueType);
  if (lvalue.space === "shared") return lvalue.index * valueStorageWidth(context.shared.get(lvalue.name)?.valueType);
  if (lvalue.space === "buffer" || lvalue.space === "constant") return lvalue.index * valueStorageWidth(context.valueTypes.get(lvalue.name));
  if (lvalue.space === "local") {
    const local = context.locals.get(lvalue.name);
    return lvalue.index * valueStorageWidth(isLocalArray(local) ? local.valueType : undefined);
  }
  return lvalue.index;
}

function resolvePoolLValue(expression: CudaLiteExpression, context: ThreadContext): LValue | undefined {
  if (expression.kind !== "index") return undefined;
  const target = expression.target;
  if (target.kind !== "cast" || !target.pointer) return undefined;
  const pointer = valueAsPoolPointer(evalExpression(target.expression, context), "pool pointer");
  if (pointer.byteOffset < 0) return {
    name: pointer.poolName,
    space: "pool",
    index: -1,
    valueType: target.valueType,
  };
  const index = Math.trunc(evalNumber(expression.index, context));
  const stride = target.valueType === "complex64" ? 2 : 1;
  return {
    name: pointer.poolName,
    space: "pool",
    index: Math.trunc(pointer.byteOffset / 4) + index * stride,
    valueType: target.valueType,
  };
}

function resolvePointerInitializer(statement: CudaLiteVarDecl, context: ThreadContext): AddressValue | PoolPointerValue {
  const init = statement.init;
  if (!init) return { kind: "pool-pointer", poolName: "", byteOffset: -1 };
  if (init?.kind === "call" || (init?.kind === "cast" && init.pointer) || init?.kind === "identifier" || init?.kind === "binary") {
    const value = pointerArgumentValue(init, statement.valueType, context);
    if (isPoolPointer(value)) return value;
    if (typeof value !== "number" && "kind" in value && value.kind === "address") {
      return { ...value, target: { ...value.target, valueType: value.target.valueType ?? statement.valueType } };
    }
  }
  if (init?.kind !== "unary" || init.operator !== "&") {
    throw compilerFailure(`pointer '${statement.name}' must initialize from an address`);
  }
  const target = resolveLValue(init.argument, context);
  if (target.space !== "shared" && target.space !== "buffer") {
    throw compilerFailure(`pointer '${statement.name}' can only alias storage or shared memory in CUDA-lite v0`);
  }
  return { kind: "address", target: { ...target, valueType: target.valueType ?? statement.valueType } };
}

function readLValue(lvalue: LValue, context: ThreadContext): EvalValue {
  if (lvalue.space === "local") {
    if (lvalue.index === undefined) return projectField(readIdentifier(lvalue.name, context), lvalue);
    const local = context.locals.get(lvalue.name);
    if (!isLocalArray(local)) throw compilerFailure(`'${lvalue.name}' is not a local array`);
    const valueType = lvalue.valueType ?? local.valueType;
    const storageIndex = lvalue.rawStorageIndex ? lvalue.index : lvalue.index * valueStorageWidth(valueType);
    const width = valueStorageWidth(valueType);
    const ok = storageIndex >= 0 && storageIndex + width - 1 < local.data.length;
    return ok ? readBufferValue(local.data, storageIndex, valueType, lvalue.field) : 0;
  }
  if (lvalue.index === undefined) throw compilerFailure(`missing index for '${lvalue.name}'`);
  if (lvalue.space === "buffer") {
    const buffer = context.buffers.get(lvalue.name);
    if (!buffer) throw compilerFailure(`missing buffer '${lvalue.name}'`);
    const valueType = lvalue.valueType ?? context.valueTypes.get(lvalue.name);
    const storageIndex = lvalue.rawStorageIndex ? lvalue.index : lvalue.index * valueStorageWidth(valueType);
    const width = valueStorageWidth(valueType);
    const ok = storageIndex >= 0 && storageIndex + width - 1 < buffer.length;
    const value = ok ? readBufferValue(buffer, storageIndex, valueType, lvalue.field) : 0;
    context.trace.reads.push({ name: lvalue.name, index: storageIndex, value: traceValue(value), ok });
    return value;
  }
  if (lvalue.space === "constant") {
    const value = context.constants.get(lvalue.name);
    if (!value || typeof value === "number") throw compilerFailure(`missing constant buffer '${lvalue.name}'`);
    const valueType = lvalue.valueType ?? context.valueTypes.get(lvalue.name);
    const storageIndex = lvalue.rawStorageIndex ? lvalue.index : lvalue.index * valueStorageWidth(valueType);
    const width = valueStorageWidth(valueType);
    const ok = storageIndex >= 0 && storageIndex + width - 1 < value.length;
    const read = ok ? readBufferValue(value, storageIndex, valueType, lvalue.field) : 0;
    context.trace.reads.push({ name: lvalue.name, index: storageIndex, value: traceValue(read), ok });
    return read;
  }
  if (lvalue.space === "pool") {
    const pool = context.memoryPools.get(lvalue.name);
    const buffer = context.buffers.get(lvalue.name);
    if (!pool && !buffer) throw compilerFailure(`missing memory pool '${lvalue.name}'`);
    const length = pool ? pool.data.length : buffer!.length;
    const ok = lvalue.index >= 0 && lvalue.index < length;
    const value = ok
      ? pool
        ? readPoolValue(pool.data, lvalue.index, lvalue.valueType)
        : readBufferValue(buffer!, lvalue.index, lvalue.valueType, lvalue.field)
      : 0;
    context.trace.reads.push({ name: lvalue.name, index: lvalue.index, value: traceValue(value), ok });
    return value;
  }
  const shared = context.shared.get(lvalue.name);
  if (!shared) throw compilerFailure(`missing shared array '${lvalue.name}'`);
  const valueType = lvalue.valueType ?? shared.valueType;
  const storageIndex = lvalue.rawStorageIndex ? lvalue.index : lvalue.index * valueStorageWidth(valueType);
  const width = valueStorageWidth(valueType);
  const ok = storageIndex >= 0 && storageIndex + width - 1 < shared.data.length;
  const value = ok ? readBufferValue(shared.data, storageIndex, valueType, lvalue.field) : 0;
  context.trace.sharedReads.push({ name: lvalue.name, index: storageIndex, value: traceValue(value), ok });
  return value;
}

function writeLValue(lvalue: LValue, value: EvalValue, context: ThreadContext): void {
  if (lvalue.space === "local") {
    if (lvalue.index !== undefined) {
      const local = context.locals.get(lvalue.name);
      if (!isLocalArray(local)) throw compilerFailure(`'${lvalue.name}' is not a local array`);
      const valueType = lvalue.valueType ?? local.valueType;
      const storageIndex = lvalue.rawStorageIndex ? lvalue.index : lvalue.index * valueStorageWidth(valueType);
      const width = valueStorageWidth(valueType);
      const ok = storageIndex >= 0 && storageIndex + width - 1 < local.data.length;
      if (ok) writeBufferValue(local.data, storageIndex, valueType, lvalue.field, value);
      return;
    }
    if (lvalue.field || lvalue.fieldIndex !== undefined) {
      const current = readIdentifier(lvalue.name, context);
      if (isComplex(current)) {
        if (lvalue.fieldIndex !== undefined) throw compilerFailure(`'${lvalue.name}' is not a CUDA vector`);
        context.locals.set(lvalue.name, { ...current, [lvalue.field!]: valueAsNumber(value, lvalue.name) });
      } else if (isCudaVectorValue(current)) {
        const field = lvalue.fieldIndex ?? cudaVectorFieldIndex(current.valueType, lvalue.field!);
        if (field === undefined) throw compilerFailure(`unsupported ${current.valueType} member '${lvalue.field}'`);
        const lanes = [...current.lanes];
        lanes[field] = valueAsNumber(value, lvalue.name);
        context.locals.set(lvalue.name, { ...current, lanes });
      } else {
        throw compilerFailure(`'${lvalue.name}' is not complex or CUDA vector`);
      }
    } else {
      context.locals.set(lvalue.name, value);
    }
    return;
  }
  if (lvalue.index === undefined) throw compilerFailure(`missing index for '${lvalue.name}'`);
  if (lvalue.space === "buffer") {
    const buffer = context.buffers.get(lvalue.name);
    if (!buffer) throw compilerFailure(`missing buffer '${lvalue.name}'`);
    const valueType = lvalue.valueType ?? context.valueTypes.get(lvalue.name);
    const storageIndex = lvalue.rawStorageIndex ? lvalue.index : lvalue.index * valueStorageWidth(valueType);
    const width = valueStorageWidth(valueType);
    const ok = storageIndex >= 0 && storageIndex + width - 1 < buffer.length;
    if (ok) writeBufferValue(buffer, storageIndex, valueType, lvalue.field, value);
    context.trace.writes.push({ name: lvalue.name, index: storageIndex, value: traceValue(value), ok });
    return;
  }
  if (lvalue.space === "constant") throw compilerFailure(`cannot write constant memory '${lvalue.name}'`);
  if (lvalue.space === "pool") {
    const pool = context.memoryPools.get(lvalue.name);
    const buffer = context.buffers.get(lvalue.name);
    if (!pool && !buffer) throw compilerFailure(`missing memory pool '${lvalue.name}'`);
    const length = pool ? pool.data.length : buffer!.length;
    const ok = lvalue.index >= 0 && lvalue.index < length;
    if (ok) {
      if (pool) writePoolValue(pool.data, lvalue.index, lvalue.valueType, value);
      else writeBufferValue(buffer!, lvalue.index, lvalue.valueType, lvalue.field, value);
    }
    context.trace.writes.push({ name: lvalue.name, index: lvalue.index, value: traceValue(value), ok });
    return;
  }
  const shared = context.shared.get(lvalue.name);
  if (!shared) throw compilerFailure(`missing shared array '${lvalue.name}'`);
  const valueType = lvalue.valueType ?? shared.valueType;
  const storageIndex = lvalue.rawStorageIndex ? lvalue.index : lvalue.index * valueStorageWidth(valueType);
  const width = valueStorageWidth(valueType);
  const ok = storageIndex >= 0 && storageIndex + width - 1 < shared.data.length;
  if (ok) writeBufferValue(shared.data, storageIndex, valueType, lvalue.field, value);
  context.trace.sharedWrites.push({ name: lvalue.name, index: storageIndex, value: traceValue(value), ok });
}

function allocateShared(declarations: readonly CudaLiteVarDecl[]): Map<string, SharedArrayValue> {
  const shared = new Map<string, SharedArrayValue>();
  for (const declaration of declarations) {
    const elements = declaration.dimensions.reduce((product, item) => product * item, 1);
    const length = elements * valueStorageWidth(declaration.valueType);
    const scalarType = cudaVectorScalarType(declaration.valueType);
    const data = declaration.valueType === "int" || scalarType === "int"
      ? new Int32Array(length)
      : declaration.valueType === "uint" || scalarType === "uint"
        ? new Uint32Array(length)
      : declaration.valueType === "bool"
          ? new Uint32Array(length)
          : declaration.valueType === "half" || scalarType === "half"
            ? createWgslFloat16Array(length)
            : new Float32Array(length);
    shared.set(declaration.name, { dimensions: declaration.dimensions, valueType: declaration.valueType, data });
  }
  return shared;
}

function allocateLocalArray(declaration: CudaLiteVarDecl): LocalArrayValue {
  return {
    kind: "local-array",
    dimensions: declaration.dimensions,
    valueType: declaration.valueType,
    data: allocateTypedArray(declaration.valueType, declaration.dimensions),
  };
}

function initializeLocalArray(
  array: LocalArrayValue,
  initializer: CudaLiteExpression,
  context: ThreadContext,
): void {
  const elements = flattenInitializerExpressions(initializer);
  const width = valueStorageWidth(array.valueType);
  const capacity = array.data.length;
  for (let elementIndex = 0; elementIndex < elements.length; elementIndex++) {
    const storageIndex = elementIndex * width;
    if (storageIndex + width > capacity) break;
    writeBufferValue(array.data, storageIndex, array.valueType, undefined, evalExpression(elements[elementIndex]!, context));
  }
}

function flattenInitializerExpressions(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenInitializerExpressions(element));
}

function allocateTypedArray(valueType: CudaLiteScalarType, dimensions: readonly number[]): WgslTypedArray {
  const elements = dimensions.reduce((product, item) => product * item, 1);
  const length = elements * valueStorageWidth(valueType);
  const scalarType = cudaVectorScalarType(valueType);
  return valueType === "int" || scalarType === "int"
    ? new Int32Array(length)
    : valueType === "uint" || scalarType === "uint"
      ? new Uint32Array(length)
      : valueType === "bool"
        ? new Uint32Array(length)
        : valueType === "half" || scalarType === "half"
          ? createWgslFloat16Array(length)
          : new Float32Array(length);
}

function readBufferValue(
  buffer: WgslTypedArray,
  storageIndex: number,
  valueType: CudaLiteScalarType | undefined,
  field: "x" | "y" | "z" | "w" | undefined,
): EvalValue {
  if (valueType === "complex64") {
    const value = {
      kind: "complex64" as const,
      x: Number(buffer[storageIndex]!),
      y: Number(buffer[storageIndex + 1]!),
    };
    if (field) {
      if (field !== "x" && field !== "y") throw compilerFailure(`unsupported complex member '${field}'`);
      return value[field];
    }
    return value;
  }
  if (isCudaVectorType(valueType)) {
    const lanes = Array.from({ length: cudaVectorLaneCount(valueType) }, (_, lane) => Number(buffer[storageIndex + lane]!));
    if (field) {
      const index = cudaVectorFieldIndex(valueType, field);
      return index === undefined ? 0 : lanes[index] ?? 0;
    }
    return { kind: "cuda-vector", valueType, lanes };
  }
  return Number(buffer[storageIndex]!);
}

function writeBufferValue(
  buffer: WgslTypedArray,
  storageIndex: number,
  valueType: CudaLiteScalarType | undefined,
  field: "x" | "y" | "z" | "w" | undefined,
  value: EvalValue,
): void {
  if (valueType === "complex64") {
    if (field) {
      buffer[storageIndex + (field === "x" ? 0 : 1)] = valueAsNumber(value, field);
      return;
    }
    const complex = valueAsComplex(value, "complex write");
    buffer[storageIndex] = complex.x;
    buffer[storageIndex + 1] = complex.y;
    return;
  }
  if (isCudaVectorType(valueType)) {
    if (field) {
      const index = cudaVectorFieldIndex(valueType, field);
      if (index !== undefined) buffer[storageIndex + index] = valueAsNumber(value, field);
      return;
    }
    const vector = valueAsCudaVector(value, valueType);
    for (let lane = 0; lane < cudaVectorLaneCount(valueType); lane++) buffer[storageIndex + lane] = vector.lanes[lane] ?? 0;
    return;
  }
  buffer[storageIndex] = valueAsNumber(value, "write value");
}

function readPoolValue(
  data: Uint32Array,
  storageIndex: number,
  valueType: CudaLiteScalarType | undefined,
): EvalValue {
  const raw = data[storageIndex] ?? 0;
  if (valueType === "float") return floatFromBits(raw);
  if (valueType === "int") return intFromBits(raw);
  if (valueType === "bool") return raw === 0 ? 0 : 1;
  return raw;
}

function writePoolValue(
  data: Uint32Array,
  storageIndex: number,
  valueType: CudaLiteScalarType | undefined,
  value: EvalValue,
): void {
  if (valueType === "float") {
    data[storageIndex] = bitsFromFloat(valueAsNumber(value, "pool write"));
    return;
  }
  if (valueType === "int") {
    data[storageIndex] = valueAsNumber(value, "pool write") >>> 0;
    return;
  }
  data[storageIndex] = valueAsNumber(value, "pool write") >>> 0;
}

function flattenIndex(dimensions: readonly number[], indices: readonly number[]): number {
  if (dimensions.length !== indices.length) {
    throw compilerFailure(`expected ${dimensions.length} indices, got ${indices.length}`);
  }
  let flat = 0;
  for (let i = 0; i < dimensions.length; i++) {
    flat = flat * dimensions[i]! + indices[i]!;
  }
  return flat;
}

function isBarrier(expression: CudaLiteExpression): boolean {
  return expression.kind === "call" &&
    expression.callee.kind === "identifier" &&
    (expression.callee.name === "__syncthreads" || expression.callee.name === "__syncwarp");
}

function cooperativeSyncKind(expression: CudaLiteExpression, context: ThreadContext): "block" | "grid" | undefined {
  if (expression.kind === "call" && expressionNameForReference(expression.callee)?.endsWith("::sync")) {
    const groupArg = expression.args[0];
    if (groupArg?.kind !== "identifier") return undefined;
    const group = context.locals.get(groupArg.name);
    if (!isCooperativeGroup(group)) return undefined;
    return group.groupKind === "grid" ? "grid" : "block";
  }
  if (expression.kind !== "call" || expression.callee.kind !== "member" || expression.callee.property !== "sync") return undefined;
  if (expression.callee.object.kind !== "identifier") return undefined;
  const value = context.locals.get(expression.callee.object.name);
  if (!isCooperativeGroup(value)) return undefined;
  return value.groupKind === "grid" ? "grid" : "block";
}

function cloneBuffers(
  buffers: Readonly<Record<string, WgslTypedArray>>,
): Map<string, WgslTypedArray> {
  const out = new Map<string, WgslTypedArray>();
  for (const [name, buffer] of Object.entries(buffers)) {
    out.set(name, cloneTypedArray(buffer));
  }
  return out;
}

function cloneSurfaces(
  surfaces: NonNullable<CompiledKernelInput["surfaces"]>,
): Record<string, { readonly width: number; readonly height: number; readonly data: Float32Array }> {
  const out: Record<string, { readonly width: number; readonly height: number; readonly data: Float32Array }> = {};
  for (const [name, surface] of Object.entries(surfaces)) {
    out[name] = {
      width: surface.width,
      height: surface.height,
      data: new Float32Array(surface.data),
    };
  }
  return out;
}

function cloneMemoryPools(
  pools: NonNullable<CompiledKernelInput["memoryPools"]>,
): Map<string, MemoryPoolValue> {
  const out = new Map<string, MemoryPoolValue>();
  for (const [name, pool] of Object.entries(pools)) {
    out.set(name, {
      data: new Uint32Array(pool.data),
      offset: pool.offset?.[0] ?? 0,
    });
  }
  return out;
}

function cloneConstants(
  constants: Readonly<Record<string, number | WgslTypedArray>>,
): Map<string, number | WgslTypedArray> {
  const out = new Map<string, number | WgslTypedArray>();
  for (const [name, value] of Object.entries(constants)) {
    out.set(name, typeof value === "number" ? value : cloneTypedArray(value));
  }
  return out;
}

function constantInitialValue(constant: CudaLiteGlobalConstant): number | WgslTypedArray {
  if (!constant.init) throw compilerFailure(`constant '${constant.name}' has no initializer`);
  const values = flattenConstantInitializer(constant.init).map(evaluateConstantNumber);
  if (constant.dimensions.length === 0) return values[0] ?? 0;
  const total = constant.dimensions.reduce((product, dimension) => product * dimension, 1);
  const padded = Array.from({ length: total }, (_, index) => values[index] ?? 0);
  if (constant.valueType === "int") return Int32Array.from(padded.map((value) => Math.trunc(value)));
  if (constant.valueType === "uint" || constant.valueType === "bool" || constant.valueType === "voidptr") {
    return Uint32Array.from(padded.map((value) => Math.trunc(value) >>> 0));
  }
  if (constant.valueType === "half") return createWgslFloat16Array(padded);
  if (constant.valueType === "float") return Float32Array.from(padded);
  if (constant.valueType === "complex64") return Float32Array.from(padded);
  return Float32Array.from(padded);
}

function flattenConstantInitializer(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  if (expression.kind !== "initializer") return [expression];
  return expression.elements.flatMap((element) => flattenConstantInitializer(element));
}

function evaluateConstantNumber(expression: CudaLiteExpression): number {
  switch (expression.kind) {
    case "number":
      return expression.value;
    case "identifier": {
      const named = CUDA_NAMED_CONSTANTS.get(expression.name);
      if (named) return named.value;
      throw compilerFailure(`constant initializer unknown symbol '${expression.name}'`);
    }
    case "cast":
      return valueAsNumber(castNumber(expression.valueType, evaluateConstantNumber(expression.expression)), "constant initializer");
    case "unary": {
      const value = evaluateConstantNumber(expression.argument);
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      return truthy(value) ? 0 : 1;
    }
    case "binary":
      return evalConstantBinary(expression.operator, evaluateConstantNumber(expression.left), evaluateConstantNumber(expression.right));
    case "conditional":
      return truthy(evaluateConstantNumber(expression.condition))
        ? evaluateConstantNumber(expression.consequent)
        : evaluateConstantNumber(expression.alternate);
    default:
      throw compilerFailure("constant initializer must be a numeric constant expression");
  }
}

function evalConstantBinary(operator: string, left: number, right: number): number {
  switch (operator) {
    case "+":
      return left + right;
    case "-":
      return left - right;
    case "*":
      return left * right;
    case "/":
      return right === 0 ? 0 : left / right;
    case "%":
      return right === 0 ? 0 : left % right;
    case "<<":
      return Math.trunc(left) << Math.trunc(right);
    case ">>":
      return Math.trunc(left) >> Math.trunc(right);
    case "&":
      return Math.trunc(left) & Math.trunc(right);
    case "^":
      return Math.trunc(left) ^ Math.trunc(right);
    case "|":
      return Math.trunc(left) | Math.trunc(right);
    case "<":
      return left < right ? 1 : 0;
    case "<=":
      return left <= right ? 1 : 0;
    case ">":
      return left > right ? 1 : 0;
    case ">=":
      return left >= right ? 1 : 0;
    case "==":
      return left === right ? 1 : 0;
    case "!=":
      return left !== right ? 1 : 0;
    case "&&":
      return truthy(left) && truthy(right) ? 1 : 0;
    case "||":
      return truthy(left) || truthy(right) ? 1 : 0;
    default:
      throw compilerFailure(`unsupported constant initializer operator '${operator}'`);
  }
}

function cloneTypedArray<T extends WgslTypedArray>(value: T): T {
  return value.slice() as T;
}

const BITCAST_BUFFER = new ArrayBuffer(4);
const BITCAST_FLOAT = new Float32Array(BITCAST_BUFFER);
const BITCAST_UINT = new Uint32Array(BITCAST_BUFFER);
const BITCAST_INT = new Int32Array(BITCAST_BUFFER);

function bitsFromFloat(value: number): number {
  BITCAST_FLOAT[0] = value;
  return BITCAST_UINT[0] ?? 0;
}

function floatFromBits(value: number): number {
  BITCAST_UINT[0] = value >>> 0;
  return BITCAST_FLOAT[0] ?? 0;
}

function intFromBits(value: number): number {
  BITCAST_UINT[0] = value >>> 0;
  return BITCAST_INT[0] ?? 0;
}

function vectorFromTuple(value: readonly [number, number, number]): Vector3 {
  return { x: value[0], y: value[1], z: value[2] };
}

function vectorFromExpressions(expressions: readonly CudaLiteExpression[], context: ThreadContext): Vector3 {
  return {
    x: Math.trunc(expressions[0] ? evalNumber(expressions[0], context) : 1),
    y: Math.trunc(expressions[1] ? evalNumber(expressions[1], context) : 1),
    z: Math.trunc(expressions[2] ? evalNumber(expressions[2], context) : 1),
  };
}

function vectorFromLaunchExpressions(expressions: readonly CudaLiteExpression[], context: ThreadContext): Vector3 {
  if (expressions.length === 1 && expressions[0]?.kind === "identifier") {
    const local = context.locals.get(expressions[0].name);
    if (isVector3(local)) return local;
  }
  return vectorFromExpressions(expressions, context);
}

function isVector3(value: LocalValue | undefined): value is Vector3 {
  return value !== undefined &&
    typeof value !== "number" &&
    !("kind" in value) &&
    typeof value.x === "number" &&
    typeof value.y === "number" &&
    typeof value.z === "number";
}

function sharedDeclarationsFor(
  statements: readonly CudaLiteStatement[],
  fallback: readonly CudaLiteVarDecl[],
): readonly CudaLiteVarDecl[] {
  if (fallback.length > 0) return fallback;
  const out: CudaLiteVarDecl[] = [];
  const visit = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "shared") out.push(item);
      if (item.kind === "if") {
        visit(item.consequent);
        if (item.alternate) visit(item.alternate);
      }
      if (item.kind === "for") visit(item.body);
    }
  };
  visit(statements);
  return out;
}

function usesGridSync(statements: readonly CudaLiteStatement[]): boolean {
  const gridGroups = new Set<string>();
  const visitExpression = (expression: CudaLiteExpression): boolean => {
    if (expression.kind === "call") {
      if (
        expression.callee.kind === "member" &&
        expression.callee.property === "sync" &&
        expression.callee.object.kind === "identifier" &&
        gridGroups.has(expression.callee.object.name)
      ) return true;
      if (visitExpression(expression.callee)) return true;
      return expression.args.some(visitExpression);
    }
    if (expression.kind === "cast") return visitExpression(expression.expression);
    if (expression.kind === "member") return visitExpression(expression.object);
    if (expression.kind === "index") return visitExpression(expression.target) || visitExpression(expression.index);
    if (expression.kind === "unary" || expression.kind === "update") return visitExpression(expression.argument);
    if (expression.kind === "binary") return visitExpression(expression.left) || visitExpression(expression.right);
    if (expression.kind === "conditional") return visitExpression(expression.condition) || visitExpression(expression.consequent) || visitExpression(expression.alternate);
    if (expression.kind === "assignment") return visitExpression(expression.left) || visitExpression(expression.right);
    if (expression.kind === "sequence") return expression.expressions.some(visitExpression);
    return false;
  };
  const walk = (items: readonly CudaLiteStatement[]): boolean => {
    for (const item of items) {
      if (item.kind === "cooperative-group" && item.groupKind === "grid") gridGroups.add(item.name);
      if (item.kind === "expr" && visitExpression(item.expression)) return true;
      if (item.kind === "if") {
        if (visitExpression(item.condition) || walk(item.consequent) || (item.alternate ? walk(item.alternate) : false)) return true;
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.init && visitExpression(item.init.init)) return true;
        if (item.init && item.init.kind !== "var" && visitExpression(item.init)) return true;
        if (item.condition && visitExpression(item.condition)) return true;
        if (item.update && visitExpression(item.update)) return true;
        if (walk(item.body)) return true;
      }
      if (item.kind === "return" && item.value && visitExpression(item.value)) return true;
    }
    return false;
  };
  return walk(statements);
}

function valueAsNumber(value: LocalValue, name: string): number {
  if (typeof value === "number") return value;
  if (isPoolPointer(value)) return value.byteOffset < 0 ? 0 : value.byteOffset + 1;
  throw compilerFailure(`'${name}' is not a scalar`);
}

function valueAsPoolPointer(value: EvalValue, name: string): PoolPointerValue {
  if (isPoolPointer(value)) return value;
  if (typeof value === "number" && value === 0) return { kind: "pool-pointer", poolName: "", byteOffset: -1 };
  throw compilerFailure(`'${name}' is not a pool pointer`);
}

function valueAsComplex(value: LocalValue, name: string): ComplexValue {
  if (isComplex(value)) return value;
  throw compilerFailure(`'${name}' is not complex`);
}

function valueAsCudaVector(value: LocalValue, type: CudaLiteVectorType): CudaVectorValue {
  if (isCudaVectorValue(value) && value.valueType === type) return value;
  throw compilerFailure(`value is not ${type}`);
}

function isComplex(value: LocalValue | EvalValue | undefined): value is ComplexValue {
  return value !== undefined &&
    typeof value !== "number" &&
    "kind" in value &&
    value.kind === "complex64";
}

function isCudaVectorValue(value: LocalValue | EvalValue | undefined): value is CudaVectorValue {
  return value !== undefined &&
    typeof value !== "number" &&
    "kind" in value &&
    value.kind === "cuda-vector";
}

function isLocalArray(value: LocalValue | undefined): value is LocalArrayValue {
  return value !== undefined &&
    typeof value !== "number" &&
    "kind" in value &&
    value.kind === "local-array";
}

function isPoolPointer(value: LocalValue | EvalValue | undefined): value is PoolPointerValue {
  return value !== undefined &&
    typeof value !== "number" &&
    "kind" in value &&
    value.kind === "pool-pointer";
}

function isAddress(value: LocalValue | EvalValue | undefined): value is AddressValue {
  return value !== undefined &&
    typeof value !== "number" &&
    "kind" in value &&
    value.kind === "address";
}

function projectField(value: LocalValue, lvalue: LValue): EvalValue {
  if (lvalue.fieldIndex !== undefined) {
    if (!isCudaVectorValue(value)) throw compilerFailure(`'${lvalue.name}' is not a CUDA vector`);
    return value.lanes[lvalue.fieldIndex] ?? 0;
  }
  if (!lvalue.field) {
    if (isComplex(value)) return value;
    if (isCudaVectorValue(value)) return value;
    return valueAsNumber(value, lvalue.name);
  }
  if (isCudaVectorValue(value)) {
    const field = cudaVectorFieldIndex(value.valueType, lvalue.field);
    if (field !== undefined) return value.lanes[field] ?? 0;
    throw compilerFailure(`unsupported ${value.valueType} member '${lvalue.field}'`);
  }
  const complex = valueAsComplex(value, lvalue.name);
  if (lvalue.field !== "x" && lvalue.field !== "y") throw compilerFailure(`unsupported complex member '${lvalue.field}'`);
  return complex[lvalue.field];
}

function zeroLocalValue(type: CudaLiteScalarType): EvalValue {
  if (type === "complex64") return { kind: "complex64", x: 0, y: 0 };
  if (isCudaVectorType(type)) return { kind: "cuda-vector", valueType: type, lanes: Array.from({ length: cudaVectorLaneCount(type) }, () => 0) };
  return 0;
}

function zeroParamLocalValue(param: CudaLiteParam): EvalValue {
  if (param.cooperativeGroupKind !== undefined) {
    return {
      kind: "cooperative-group",
      groupKind: param.cooperativeGroupKind,
      ...(param.tileSize === undefined ? {} : { tileSize: param.tileSize }),
    };
  }
  return zeroLocalValue(param.valueType);
}

function valueStorageWidth(type: CudaLiteScalarType | undefined): number {
  if (isCudaVectorType(type)) return cudaVectorLaneCount(type);
  return type === "complex64" ? 2 : 1;
}

function traceValue(value: EvalValue): number {
  if (isPoolPointer(value)) return valueAsNumber(value, "pool pointer");
  if (isAddress(value)) return value.target.index ?? 0;
  if (isCudaVectorValue(value)) return value.lanes[0] ?? 0;
  if (isCooperativeGroup(value)) return 0;
  return isComplex(value) ? value.x : value;
}

function expressionNameForReference(expression: CudaLiteExpression): string | undefined {
  return expression.kind === "identifier" ? expression.name : undefined;
}

function isCooperativeGroup(value: LocalValue | undefined): value is CooperativeGroupValue {
  return value !== undefined &&
    typeof value !== "number" &&
    "kind" in value &&
    value.kind === "cooperative-group";
}

function truthy(value: number): boolean {
  return value !== 0 && !Number.isNaN(value);
}

function freezeTrace(trace: MutableTrace): KernelThreadTrace {
  return {
    blockIdx: trace.blockIdx,
    threadIdx: trace.threadIdx,
    reads: trace.reads.map((item) => ({ ...item })),
    writes: trace.writes.map((item) => ({ ...item })),
    sharedReads: trace.sharedReads.map((item) => ({ ...item })),
    sharedWrites: trace.sharedWrites.map((item) => ({ ...item })),
  };
}

function validateInputs(compiled: CompiledCudaLiteKernel, input: CompiledKernelInput): void {
  for (const param of compiled.ir.params) {
    if (param.valueType === "texture2d") {
      const texture = input.textures?.[param.name];
      if (!texture) throw compilerFailure(`missing texture input '${param.name}'`);
      validateSurfaceInput(`texture ${param.name}`, texture);
    } else if (param.valueType === "surface2d") {
      const surface = input.surfaces?.[param.name];
      if (!surface) throw compilerFailure(`missing surface input '${param.name}'`);
      validateSurfaceInput(param.name, surface);
    } else if (param.valueType === "devicepool") {
      const pool = input.memoryPools?.[param.name];
      if (!pool) throw compilerFailure(`missing memory pool input '${param.name}'`);
      validateMemoryPoolInput(param.name, pool);
    } else if (param.pointer) {
      const buffer = input.buffers[param.name];
      if (!buffer) throw compilerFailure(`missing buffer input '${param.name}'`);
      const scalarType = cudaVectorScalarType(param.valueType);
      if ((param.valueType === "int" || scalarType === "int") && !(buffer instanceof Int32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Int32Array`);
      }
      if ((param.valueType === "uint" || scalarType === "uint") && !(buffer instanceof Uint32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Uint32Array`);
      }
      if ((param.valueType === "float" || scalarType === "float") && !(buffer instanceof Float32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Float32Array`);
      }
      if ((param.valueType === "half" || scalarType === "half") && !isWgslFloat16Array(buffer)) {
        throw compilerFailure(`buffer '${param.name}' expects Float16Array`);
      }
      if (param.valueType === "bool" && !(buffer instanceof Uint32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Uint32Array`);
      }
      if (param.valueType === "complex64" && !(buffer instanceof Float32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects interleaved Float32Array`);
      }
    } else if (input.scalars?.[param.name] === undefined) {
      throw compilerFailure(`missing scalar input '${param.name}'`);
    }
  }
  for (const constant of compiled.ir.constants) {
    if (constant.init !== undefined) continue;
    const value = input.constants?.[constant.name];
    if (value === undefined) throw compilerFailure(`missing constant input '${constant.name}'`);
    if (constant.dimensions.length === 0) {
      if (typeof value !== "number") throw compilerFailure(`constant '${constant.name}' expects number`);
    } else {
      if (typeof value === "number") throw compilerFailure(`constant '${constant.name}' expects typed array`);
      validateTypedConstant(constant.name, constant.valueType, value);
      const expected = constant.dimensions.reduce((product, dimension) => product * dimension, 1);
      if (value.length < expected) throw compilerFailure(`constant '${constant.name}' expects at least ${expected} elements`);
    }
  }
  for (const texture of compiled.ir.textures) {
    const value = input.textures?.[texture.name];
    if (!value) throw compilerFailure(`missing texture input '${texture.name}'`);
    validateSurfaceInput(`texture ${texture.name}`, value);
  }
  for (const poolName of collectExternalDevicePoolNames(compiled.ir.body)) {
    const pool = input.memoryPools?.[poolName];
    if (!pool) throw compilerFailure(`missing memory pool input '${poolName}'`);
    validateMemoryPoolInput(poolName, pool);
  }
}

function validateSurfaceInput(name: string, value: { readonly width: number; readonly height: number; readonly data: Float32Array }): void {
  if (!(value.data instanceof Float32Array)) throw compilerFailure(`${name} expects Float32Array data`);
  if (!Number.isInteger(value.width) || value.width <= 0) throw compilerFailure(`${name} width must be positive`);
  if (!Number.isInteger(value.height) || value.height <= 0) throw compilerFailure(`${name} height must be positive`);
  const expected = value.width * value.height;
  if (value.data.length < expected) throw compilerFailure(`${name} expects at least ${expected} elements`);
}

function validateMemoryPoolInput(name: string, value: { readonly data: Uint32Array; readonly offset?: Uint32Array }): void {
  if (!(value.data instanceof Uint32Array)) throw compilerFailure(`${name} memory pool expects Uint32Array data`);
  if (value.offset !== undefined && (!(value.offset instanceof Uint32Array) || value.offset.length < 1)) {
    throw compilerFailure(`${name} memory pool offset expects Uint32Array length >= 1`);
  }
}

function contextConstantDimensions(name: string, context: ThreadContext): readonly number[] {
  const dimensions = context.constantDimensions.get(name);
  if (!dimensions) throw compilerFailure(`constant dimensions unavailable for '${name}'`);
  return dimensions;
}

function validateTypedConstant(name: string, valueType: string, value: WgslTypedArray): void {
  const scalarType = cudaVectorScalarType(valueType as CudaLiteScalarType);
  if ((valueType === "int" || scalarType === "int") && !(value instanceof Int32Array)) {
    throw compilerFailure(`constant '${name}' expects Int32Array`);
  }
  if ((valueType === "uint" || scalarType === "uint") && !(value instanceof Uint32Array)) {
    throw compilerFailure(`constant '${name}' expects Uint32Array`);
  }
  if ((valueType === "float" || scalarType === "float") && !(value instanceof Float32Array)) {
    throw compilerFailure(`constant '${name}' expects Float32Array`);
  }
  if ((valueType === "half" || scalarType === "half") && !isWgslFloat16Array(value)) {
    throw compilerFailure(`constant '${name}' expects Float16Array`);
  }
  if (valueType === "bool" && !(value instanceof Uint32Array)) {
    throw compilerFailure(`constant '${name}' expects Uint32Array`);
  }
  if (valueType === "complex64" && !(value instanceof Float32Array)) {
    throw compilerFailure(`constant '${name}' expects interleaved Float32Array`);
  }
}

function compilerFailure(message: string): CudaLiteCompilerError {
  return new CudaLiteCompilerError(message, [{
    code: "reference-runtime-error",
    severity: "error",
    message,
    span: { start: 0, end: 0, line: 1, column: 1 },
  }]);
}
