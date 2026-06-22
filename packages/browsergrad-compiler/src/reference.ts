import type { WgslTypedArray } from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames } from "./ast_queries.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteDeviceFunction,
  type CudaLiteExpression,
  type CudaLiteKernel,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type CudaLiteVarDecl,
  type KernelLaunch,
  type KernelMemoryAccess,
  type KernelThreadTrace,
  type ReferenceKernelResult,
} from "./types.js";

type EvalValue = number | ComplexValue | PoolPointerValue;
type LocalValue = number | Vector3 | AddressValue | CooperativeGroupValue | ComplexValue | PoolPointerValue;
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
  readonly field?: "x" | "y";
  readonly valueType?: CudaLiteScalarType;
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

interface MutableTrace {
  readonly blockIdx: readonly [number, number, number];
  readonly threadIdx: readonly [number, number, number];
  readonly reads: KernelMemoryAccess[];
  readonly writes: KernelMemoryAccess[];
  readonly sharedReads: KernelMemoryAccess[];
  readonly sharedWrites: KernelMemoryAccess[];
}

type ExecControl = { readonly kind: "return"; readonly value?: number } | { readonly kind: "continue" };
type BarrierKind = "barrier" | "grid-barrier";
type BarrierGenerator = Generator<BarrierKind, ExecControl | void, void>;

export function runCompiledKernelReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): ReferenceKernelResult {
  validateLaunch(launch, compiled.ir.workgroupSize);
  validateInputs(compiled, input);
  const buffers = cloneBuffers(input.buffers);
  const constants = cloneConstants(input.constants ?? {});
  const constantDimensions = new Map(compiled.ir.constants.map((constant) => [constant.name, constant.dimensions]));
  const textures = input.textures ?? {};
  const surfaces = cloneSurfaces(input.surfaces ?? {});
  const memoryPools = cloneMemoryPools(input.memoryPools ?? {});
  const functions = new Map(compiled.ir.functions.map((fn) => [fn.name, fn]));
  const kernels = new Map(compiled.ast.kernels.map((kernel) => [kernel.name, kernel]));
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
        if (isBarrier(statement.expression)) {
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
      case "return":
        return {
          kind: "return",
          ...(statement.value === undefined ? {} : { value: evalNumber(statement.value, context) }),
        };
      case "continue":
        return { kind: "continue" };
    }
  }
}

function execInlineAsm(
  statement: Extract<CudaLiteStatement, { kind: "asm" }>,
  context: ThreadContext,
): void {
  if (!/\bfma\.rn\.f32\b/u.test(statement.template)) {
    throw compilerFailure("unsupported inline asm template");
  }
  if (statement.inputs.length !== 2) {
    throw compilerFailure("fma.rn.f32 inline asm expects two inputs");
  }
  const target = resolveLValue(statement.output, context);
  const current = valueAsNumber(readLValue(target, context), target.name);
  const a = evalNumber(statement.inputs[0]!, context);
  const b = evalNumber(statement.inputs[1]!, context);
  writeLValue(target, current + a * b, context);
}

function execVar(statement: CudaLiteVarDecl, context: ThreadContext): void {
  if (statement.storage === "shared") return;
  if (statement.pointer) {
    context.locals.set(statement.name, resolvePointerInitializer(statement, context));
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
      locals.set(param.name, zeroLocalValue(param.valueType));
      continue;
    }
    if (param.pointer) {
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
  const offset = pointerOffsetArgumentValue(arg, valueType, context);
  if (offset) return offset;
  if (arg.kind === "unary" && arg.operator === "&") {
    return { kind: "address", target: resolveLValue(arg.argument, context) };
  }
  if (arg.kind === "identifier") {
    if (context.buffers.has(arg.name)) return { kind: "address", target: { name: arg.name, space: "buffer", index: 0 } };
    if (context.memoryPools.has(arg.name)) return { kind: "pool-pointer", poolName: arg.name, byteOffset: 0, valueType };
    const local = context.locals.get(arg.name);
    if (isPoolPointer(local)) return { ...local, valueType };
    if (local && typeof local !== "number") return local;
  }
  const value = evalExpression(arg, context);
  if (isPoolPointer(value)) return { ...value, valueType };
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
    return { ...left, byteOffset: left.byteOffset + delta * 4, valueType };
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

function execCudaMemcpyPeerAsync(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  context: ThreadContext,
): void {
  const dst = expression.args[0];
  const src = expression.args[2];
  const count = expression.args[4];
  if (!dst || !src || !count) throw compilerFailure("cudaMemcpyPeerAsync expects dst, src, and byte count");
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
    throw compilerFailure("cudaMemcpyPeerAsync expects pointer arguments");
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
  throw compilerFailure(`cudaMemcpyPeerAsync cannot copy from ${lvalue.space} '${lvalue.name}'`);
}

function pointerValueTypeForExpression(
  expression: CudaLiteExpression,
  context: ThreadContext,
): CudaLiteScalarType {
  if (expression.kind === "cast" && expression.pointer) return expression.valueType;
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
    return context.valueTypes.get(expression.name) ?? "uint";
  }
  const root = rootIdentifierFromExpression(expression);
  return root ? context.valueTypes.get(root) ?? "uint" : "uint";
}

function byteView(buffer: WgslTypedArray): Uint8Array {
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

function elementByteSize(valueType: CudaLiteScalarType): number {
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
    case "identifier": {
      const value = readIdentifier(expression.name, context);
      if (isComplex(value)) return value;
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
      if (expression.operator === "&") return 0;
      if (expression.operator === "*") return evalDeref(expression.argument, context);
      const value = evalNumber(expression.argument, context);
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      return truthy(value) ? 0 : 1;
    }
    case "binary":
      return evalBinary(expression.operator, expression.left, expression.right, context);
    case "conditional":
      return truthy(evalNumber(expression.condition, context))
        ? evalExpression(expression.consequent, context)
        : evalExpression(expression.alternate, context);
    case "assignment":
      return evalAssignment(expression.operator, expression.left, expression.right, context);
    case "update": {
      const lvalue = resolveLValue(expression.argument, context);
      const current = readLValue(lvalue, context);
      const currentNumber = valueAsNumber(current, lvalue.name);
      const next = expression.operator === "++" ? currentNumber + 1 : currentNumber - 1;
      writeLValue(lvalue, next, context);
      return expression.prefix ? next : currentNumber;
    }
    case "call":
      return evalCall(expression, context);
  }
}

function evalDeref(expression: CudaLiteExpression, context: ThreadContext): EvalValue {
  if (expression.kind === "identifier") {
    if (context.buffers.has(expression.name)) return readLValue({ name: expression.name, space: "buffer", index: 0 }, context);
    const local = context.locals.get(expression.name);
    if (isPoolPointer(local)) return readLValue({ name: local.poolName, space: "pool", index: Math.trunc(local.byteOffset / 4), valueType: "float" }, context);
  }
  if (expression.kind === "cast" && expression.pointer) {
    const pointer = valueAsPoolPointer(evalExpression(expression.expression, context), "pool pointer");
    return readLValue({ name: pointer.poolName, space: "pool", index: Math.trunc(pointer.byteOffset / 4), valueType: expression.valueType }, context);
  }
  throw compilerFailure("unsupported pointer dereference");
}

function evalNumber(expression: CudaLiteExpression, context: ThreadContext): number {
  return valueAsNumber(evalExpression(expression, context), expression.kind);
}

function castNumber(type: Exclude<CudaLiteScalarType, "void">, value: number): EvalValue {
  if (type === "int") return Math.trunc(value) | 0;
  if (type === "uint") return Math.trunc(value) >>> 0;
  if (type === "bool") return truthy(value) ? 1 : 0;
  if (type === "complex64") return { kind: "complex64", x: value, y: 0 };
  return value;
}

function readMemberObject(expression: CudaLiteExpression, context: ThreadContext): Vector3 | ComplexValue {
  if (expression.kind === "identifier") {
    const value = readIdentifier(expression.name, context);
    if (isComplex(value)) return value;
    if (typeof value === "number" || "kind" in value) throw compilerFailure(`'${expression.name}' is not a vector`);
    return value;
  }
  const value = evalExpression(expression, context);
  if (isComplex(value)) return value;
  throw compilerFailure("member access only supports CUDA-lite builtin vectors and complex values");
}

function readIdentifier(name: string, context: ThreadContext): LocalValue {
  if (name === "nullptr") return { kind: "pool-pointer", poolName: "", byteOffset: -1 };
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

function evalAssignment(
  operator: string,
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  context: ThreadContext,
): EvalValue {
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
  const current = operator === "=" ? 0 : valueAsNumber(readLValue(lvalue, context), lvalue.name);
  const value: number = operator === "="
    ? right
    : operator === "+="
      ? current + right
    : operator === "-="
      ? current - right
      : operator === "*="
        ? current * right
        : operator === "/="
          ? current / right
          : operator === "<<="
            ? Math.trunc(current) << Math.trunc(right)
            : Math.trunc(current) >> Math.trunc(right);
  writeLValue(lvalue, value, context);
  return value;
}

function evalCall(expression: Extract<CudaLiteExpression, { kind: "call" }>, context: ThreadContext): EvalValue {
  const name = expression.kind === "call" && expression.callee.kind === "identifier"
    ? expression.callee.name
    : undefined;
  const cooperativeGroupCall = evalCooperativeGroupCall(expression, context);
  if (cooperativeGroupCall !== undefined) return cooperativeGroupCall;
  if (name === "printf") return 0;
  if (name === "cudaDeviceSynchronize") return 0;
  if (name === "cudaMemcpyPeerAsync") {
    execCudaMemcpyPeerAsync(expression, context);
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
  if (name === "atomicAdd") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => current + value);
  }
  if (name === "atomicSub") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => current - value);
  }
  if (name === "atomicMin") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => Math.min(current, value));
  }
  if (name === "atomicMax" || name === "atomicMaxFloat") {
    return evalAtomicReadModifyWrite(expression, context, (current, value) => Math.max(current, value));
  }
  if (name === "atomicExch") {
    return evalAtomicReadModifyWrite(expression, context, (_current, value) => value);
  }
  if (name === "atomicCAS") {
    const first = expression.args[0];
    if (!first) throw compilerFailure("atomicCAS expects target");
    const target = first.kind === "unary" && first.operator === "&" ? first.argument : first;
    const lvalue = resolveLValue(target, context);
    const current = valueAsNumber(readLValue(lvalue, context), lvalue.name);
    const compare = evalNumber(expression.args[1]!, context);
    const value = evalNumber(expression.args[2]!, context);
    if (Math.trunc(current) === Math.trunc(compare)) writeLValue(lvalue, value, context);
    return current;
  }
  if (name === "tex2D") {
    const textureRef = expression.args[0];
    if (textureRef?.kind !== "identifier") throw compilerFailure("tex2D expects texture reference");
    const texture = context.textures[textureRef.name];
    if (!texture) throw compilerFailure(`missing texture input '${textureRef.name}'`);
    const x = Math.max(0, Math.min(texture.width - 1, Math.floor(evalNumber(expression.args[1]!, context))));
    const y = Math.max(0, Math.min(texture.height - 1, Math.floor(evalNumber(expression.args[2]!, context))));
    return texture.data[y * texture.width + x] ?? 0;
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
    return target?.kind === "identifier" ? sizeofType(target.name) : 4;
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
  const args = expression.args.map((arg) => evalNumber(arg, context));
  const deviceFunction = name ? context.functions.get(name) : undefined;
  if (deviceFunction) return evalDeviceFunction(deviceFunction, args, context);
  switch (name) {
    case "__syncthreads":
      return 0;
    case "min":
      return Math.min(...args);
    case "max":
      return Math.max(...args);
    case "sqrtf":
      return Math.sqrt(args[0] ?? 0);
    case "expf":
      return Math.exp(args[0] ?? 0);
    case "logf":
      return Math.log(args[0] ?? 0);
    case "__half2float":
    case "__float2half":
      return args[0] ?? 0;
    case "bg_subgroup_add":
      return args[0] ?? 0;
    case "__shfl_down_sync":
    case "__shfl_up_sync":
    case "__shfl_xor_sync":
      return args[1] ?? 0;
    default:
      throw compilerFailure(`unsupported call '${name ?? "<expr>"}'`);
  }
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
  if (callee.property === "shfl_down" || callee.property === "shfl_up" || callee.property === "shfl_xor") return valueAsNumber(args[0] ?? 0, "shuffle value");
  return undefined;
}

function localLinearRank(context: ThreadContext): number {
  return context.threadIdx.x +
    context.threadIdx.y * context.blockDim.x +
    context.threadIdx.z * context.blockDim.x * context.blockDim.y;
}

function evalDeviceFunction(fn: CudaLiteDeviceFunction, args: readonly number[], context: ThreadContext): number {
  const locals = new Map<string, LocalValue>();
  for (const [index, param] of fn.params.entries()) locals.set(param.name, args[index] ?? 0);
  const fnContext: ThreadContext = {
    ...context,
    locals,
  };
  const generator = execStatements(fn.body, fnContext);
  while (true) {
    const next = generator.next();
    if (next.done) {
      const control = next.value;
      if (control?.kind === "return") return control.value ?? 0;
      return 0;
    }
    throw compilerFailure(`device function '${fn.name}' cannot contain __syncthreads()`);
  }
}

function evalAtomicReadModifyWrite(
  expression: Extract<CudaLiteExpression, { kind: "call" }>,
  context: ThreadContext,
  op: (current: number, value: number) => number,
): number {
  const first = expression.args[0];
  if (!first) throw compilerFailure("atomic operation expects target");
  const target = first.kind === "unary" && first.operator === "&" ? first.argument : first;
  const lvalue = resolveLValue(target, context);
  const current = valueAsNumber(readLValue(lvalue, context), lvalue.name);
  const value = evalNumber(expression.args[1]!, context);
  writeLValue(lvalue, op(current, value), context);
  return current;
}

function resolveAddressArgument(expression: CudaLiteExpression, context: ThreadContext): LValue {
  if (expression.kind !== "unary" || expression.operator !== "&") {
    throw compilerFailure("expected address argument");
  }
  return resolveLValue(expression.argument, context);
}

function curandNext(state: number): number {
  return (Math.imul(state >>> 0, 1664525) + 1013904223) >>> 0;
}

function resolveLValue(expression: CudaLiteExpression, context: ThreadContext): LValue {
  const pool = resolvePoolLValue(expression, context);
  if (pool) return pool;
  if (expression.kind === "member") {
    if (expression.property !== "x" && expression.property !== "y") {
      throw compilerFailure(`unsupported lvalue member '${expression.property}'`);
    }
    return { ...resolveLValue(expression.object, context), field: expression.property };
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
    return {
      name: alias.target.name,
      space: alias.target.space,
      index: (alias.target.index ?? 0) + chain[0]!,
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
  if (init?.kind === "call" || (init?.kind === "cast" && init.pointer) || init?.kind === "identifier") {
    const value = evalExpression(init, context);
    if (isPoolPointer(value)) return value;
  }
  if (init?.kind !== "unary" || init.operator !== "&") {
    throw compilerFailure(`pointer '${statement.name}' must initialize from an address`);
  }
  const target = resolveLValue(init.argument, context);
  if (target.space !== "shared") {
    throw compilerFailure(`pointer '${statement.name}' can only alias shared memory in CUDA-lite v0`);
  }
  return { kind: "address", target };
}

function readLValue(lvalue: LValue, context: ThreadContext): EvalValue {
  if (lvalue.space === "local") return projectField(readIdentifier(lvalue.name, context), lvalue);
  if (lvalue.index === undefined) throw compilerFailure(`missing index for '${lvalue.name}'`);
  if (lvalue.space === "buffer") {
    const buffer = context.buffers.get(lvalue.name);
    if (!buffer) throw compilerFailure(`missing buffer '${lvalue.name}'`);
    const valueType = context.valueTypes.get(lvalue.name);
    const storageIndex = valueType === "complex64" ? lvalue.index * 2 : lvalue.index;
    const ok = storageIndex >= 0 && storageIndex < buffer.length && (valueType !== "complex64" || storageIndex + 1 < buffer.length);
    const value = ok ? readBufferValue(buffer, storageIndex, valueType, lvalue.field) : 0;
    context.trace.reads.push({ name: lvalue.name, index: storageIndex, value: traceValue(value), ok });
    return value;
  }
  if (lvalue.space === "constant") {
    const value = context.constants.get(lvalue.name);
    if (!value || typeof value === "number") throw compilerFailure(`missing constant buffer '${lvalue.name}'`);
    const valueType = context.valueTypes.get(lvalue.name);
    const storageIndex = valueType === "complex64" ? lvalue.index * 2 : lvalue.index;
    const ok = storageIndex >= 0 && storageIndex < value.length && (valueType !== "complex64" || storageIndex + 1 < value.length);
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
  const storageIndex = shared.valueType === "complex64" ? lvalue.index * 2 : lvalue.index;
  const ok = storageIndex >= 0 && storageIndex < shared.data.length && (shared.valueType !== "complex64" || storageIndex + 1 < shared.data.length);
  const value = ok ? readBufferValue(shared.data, storageIndex, shared.valueType, lvalue.field) : 0;
  context.trace.sharedReads.push({ name: lvalue.name, index: storageIndex, value: traceValue(value), ok });
  return value;
}

function writeLValue(lvalue: LValue, value: EvalValue, context: ThreadContext): void {
  if (lvalue.space === "local") {
    if (lvalue.field) {
      const current = readIdentifier(lvalue.name, context);
      if (!isComplex(current)) throw compilerFailure(`'${lvalue.name}' is not complex`);
      context.locals.set(lvalue.name, { ...current, [lvalue.field]: valueAsNumber(value, lvalue.name) });
    } else {
      context.locals.set(lvalue.name, value);
    }
    return;
  }
  if (lvalue.index === undefined) throw compilerFailure(`missing index for '${lvalue.name}'`);
  if (lvalue.space === "buffer") {
    const buffer = context.buffers.get(lvalue.name);
    if (!buffer) throw compilerFailure(`missing buffer '${lvalue.name}'`);
    const valueType = context.valueTypes.get(lvalue.name);
    const storageIndex = valueType === "complex64" ? lvalue.index * 2 : lvalue.index;
    const ok = storageIndex >= 0 && storageIndex < buffer.length && (valueType !== "complex64" || storageIndex + 1 < buffer.length);
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
  const storageIndex = shared.valueType === "complex64" ? lvalue.index * 2 : lvalue.index;
  const ok = storageIndex >= 0 && storageIndex < shared.data.length && (shared.valueType !== "complex64" || storageIndex + 1 < shared.data.length);
  if (ok) writeBufferValue(shared.data, storageIndex, shared.valueType, lvalue.field, value);
  context.trace.sharedWrites.push({ name: lvalue.name, index: storageIndex, value: traceValue(value), ok });
}

function allocateShared(declarations: readonly CudaLiteVarDecl[]): Map<string, SharedArrayValue> {
  const shared = new Map<string, SharedArrayValue>();
  for (const declaration of declarations) {
    const elements = declaration.dimensions.reduce((product, item) => product * item, 1);
    const length = declaration.valueType === "complex64" ? elements * 2 : elements;
    const data = declaration.valueType === "int"
      ? new Int32Array(length)
      : declaration.valueType === "uint"
        ? new Uint32Array(length)
        : declaration.valueType === "bool"
          ? new Uint32Array(length)
          : declaration.valueType === "half"
            ? new Float16Array(length)
            : new Float32Array(length);
    shared.set(declaration.name, { dimensions: declaration.dimensions, valueType: declaration.valueType, data });
  }
  return shared;
}

function readBufferValue(
  buffer: WgslTypedArray,
  storageIndex: number,
  valueType: CudaLiteScalarType | undefined,
  field: "x" | "y" | undefined,
): EvalValue {
  if (valueType === "complex64") {
    const value = {
      kind: "complex64" as const,
      x: Number(buffer[storageIndex]!),
      y: Number(buffer[storageIndex + 1]!),
    };
    return field ? value[field] : value;
  }
  return Number(buffer[storageIndex]!);
}

function writeBufferValue(
  buffer: WgslTypedArray,
  storageIndex: number,
  valueType: CudaLiteScalarType | undefined,
  field: "x" | "y" | undefined,
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
    expression.callee.name === "__syncthreads";
}

function cooperativeSyncKind(expression: CudaLiteExpression, context: ThreadContext): "block" | "grid" | undefined {
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

function isComplex(value: LocalValue | EvalValue | undefined): value is ComplexValue {
  return value !== undefined &&
    typeof value !== "number" &&
    "kind" in value &&
    value.kind === "complex64";
}

function isPoolPointer(value: LocalValue | EvalValue | undefined): value is PoolPointerValue {
  return value !== undefined &&
    typeof value !== "number" &&
    "kind" in value &&
    value.kind === "pool-pointer";
}

function projectField(value: LocalValue, lvalue: LValue): EvalValue {
  if (!lvalue.field) {
    if (isComplex(value)) return value;
    return valueAsNumber(value, lvalue.name);
  }
  const complex = valueAsComplex(value, lvalue.name);
  return complex[lvalue.field];
}

function zeroLocalValue(type: CudaLiteScalarType): EvalValue {
  return type === "complex64" ? { kind: "complex64", x: 0, y: 0 } : 0;
}

function sizeofType(typeName: string): number {
  switch (typeName) {
    case "half":
    case "__half":
      return 2;
    case "cufftComplex":
      return 8;
    default:
      return 4;
  }
}

function traceValue(value: EvalValue): number {
  if (isPoolPointer(value)) return valueAsNumber(value, "pool pointer");
  return isComplex(value) ? value.x : value;
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

function validateLaunch(launch: KernelLaunch, workgroupSize: readonly [number, number, number]): void {
  for (let axis = 0; axis < 3; axis++) {
    const block = launch.blockDim[axis];
    const grid = launch.gridDim[axis];
    const expected = workgroupSize[axis];
    if (block !== expected) {
      throw compilerFailure("launch.blockDim must match compiled workgroupSize for reference/WebGPU parity");
    }
    if (grid === undefined || !Number.isInteger(grid) || grid <= 0) {
      throw compilerFailure("launch.gridDim values must be positive integers");
    }
  }
}

function validateInputs(compiled: CompiledCudaLiteKernel, input: CompiledKernelInput): void {
  for (const param of compiled.ir.params) {
    if (param.valueType === "surface2d") {
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
      if (param.valueType === "int" && !(buffer instanceof Int32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Int32Array`);
      }
      if (param.valueType === "uint" && !(buffer instanceof Uint32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Uint32Array`);
      }
      if (param.valueType === "float" && !(buffer instanceof Float32Array)) {
        throw compilerFailure(`buffer '${param.name}' expects Float32Array`);
      }
      if (param.valueType === "half" && !(buffer instanceof Float16Array)) {
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
  if (valueType === "int" && !(value instanceof Int32Array)) {
    throw compilerFailure(`constant '${name}' expects Int32Array`);
  }
  if (valueType === "uint" && !(value instanceof Uint32Array)) {
    throw compilerFailure(`constant '${name}' expects Uint32Array`);
  }
  if (valueType === "float" && !(value instanceof Float32Array)) {
    throw compilerFailure(`constant '${name}' expects Float32Array`);
  }
  if (valueType === "half" && !(value instanceof Float16Array)) {
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
