import { describe, expect, it } from "vitest";
import {
  emitFloatAtomicAddHelper,
  emitFloatAtomicMaxHelper,
  emitIntegerAtomicLoopHelpers,
  floatAtomicHelperName,
  integerAtomicLoopHelperName,
  isAtomicCasCallName,
  isAtomicExchangeCallName,
  isAtomicReturnCallName,
} from "../src/wgsl_atomic_helpers";
import {
  collectAssignedNames,
  constantBooleanExpression,
  constantNumberExpression,
  countReturns,
  expressionContainsSubgroupCall,
  isBarrierCall,
  isCooperativeGroupSubgroupMember,
  isSubgroupCallName,
  splitIfTrailingBreak,
  splitIfTrailingVoidReturn,
  statementContainsVoidReturn,
  statementContainsBarrier,
  statementContainsSubgroupCall,
} from "../src/wgsl_control_analysis";
import {
  bitcastStorageViewType,
  castExpressionToVectorScalar,
  emitStorageCarrierAsU32,
  emitU32AsStorageCarrier,
  emitSharedFlatAccess,
  emitVectorConstructor,
  emitVectorSplat,
  emitVectorStorageFieldWrite,
  emitVectorStorageFieldWriteAt,
  emitVectorStorageRead,
  emitVectorStorageReadAt,
  emitVectorStorageWrite,
  emitVectorStorageWriteAt,
  vectorFieldName,
  vectorStorageBase,
  wgslElementByteSize,
  wgslScalar,
  zeroValue,
} from "../src/wgsl_storage";
import {
  resolveStorageView,
  resolveStorageViewForPointerExpression,
  type WgslStorageViewCallbacks,
  type WgslStorageViewContext,
} from "../src/wgsl_storage_views";
import {
  emitExpressionAsWgslScalarText,
  emitNumberLiteral,
  emitNumberLiteralAsU32,
  emitVectorLaneSetExpression,
  isAbstractIntegerLiteral,
  isBitwiseOperator,
  isComparisonOperator,
  isIntegerNumberLiteral,
  isShiftOperator,
  isVectorArithmeticOperator,
  numberLiteralHasFloatSyntax,
  numberLiteralHasIntegerSuffix,
  numberLiteralHasUnsignedSuffix,
  promotedCudaScalarType,
  updateDeltaForValueType,
} from "../src/wgsl_value_conversion";
import type {
  CudaLiteCallExpression,
  CudaLiteExpression,
  CudaLiteNumberLiteral,
  CudaLiteStatement,
  CudaLiteVarDecl,
} from "../src/types";

const span = { start: 0, end: 0, line: 1, column: 1 };

describe("WGSL storage helpers", () => {
  it("maps CUDA scalar and vector types to WGSL storage/value types", () => {
    expect(wgslScalar("float")).toBe("f32");
    expect(wgslScalar("double")).toBe("f32");
    expect(wgslScalar("half2")).toBe("vec2<f16>");
    expect(wgslScalar("float4")).toBe("vec4<f32>");
    expect(zeroValue("float4")).toBe("vec4<f32>(0.0, 0.0, 0.0, 0.0)");
    expect(zeroValue("complex64")).toBe("vec2<f32>(0.0, 0.0)");
    expect(wgslElementByteSize("uchar")).toBe(1);
    expect(wgslElementByteSize("half2")).toBe(4);
    expect(wgslElementByteSize("float4")).toBe(16);
  });

  it("centralizes storage carrier casts and bitcast policy", () => {
    expect(bitcastStorageViewType("float", "uint")).toBe("u32");
    expect(bitcastStorageViewType("uint", "float")).toBe("f32");
    expect(bitcastStorageViewType("bool", "uint")).toBeUndefined();
    expect(emitStorageCarrierAsU32("x", "float")).toBe("bitcast<u32>(x)");
    expect(emitStorageCarrierAsU32("x", "uint")).toBe("x");
    expect(emitU32AsStorageCarrier("bits", "int")).toBe("bitcast<i32>(bits)");
    expect(emitU32AsStorageCarrier("bits", "uchar")).toBe("bits");
  });

  it("emits vector lane storage reads and writes from one path", () => {
    expect(vectorStorageBase("i", 4)).toBe("(u32(i) * 4u)");
    expect(vectorFieldName(2)).toBe("z");
    expect(emitVectorStorageRead("data", "float4", "i")).toBe("vec4<f32>(data[(u32(i) * 4u) + 0u], data[(u32(i) * 4u) + 1u], data[(u32(i) * 4u) + 2u], data[(u32(i) * 4u) + 3u])");
    expect(emitVectorStorageWrite("data", "float2", "i", "value")).toBe("data[(u32(i) * 2u) + 0u] = value.x; data[(u32(i) * 2u) + 1u] = value.y");
    expect(emitVectorStorageFieldWrite("data", "float4", "i", "w", "v")).toBe("data[(u32(i) * 4u) + 3u] = v");
    expect(emitVectorStorageFieldWrite("data", "float4", "i", "q", "v")).toBeUndefined();
    expect(emitVectorStorageReadAt("data", "half2", "base")).toBe("vec2<f16>(f16(data[base + 0u]), f16(data[base + 1u]))");
    expect(emitVectorStorageWriteAt("data", "uint2", "base", "value")).toBe("data[base + 0u] = value.x; data[base + 1u] = value.y");
    expect(emitVectorStorageFieldWriteAt("data", "base", 1, "v")).toBe("data[base + 1u] = v");
  });

  it("emits vector constructors, splats, scalar casts, and flattened shared access", () => {
    expect(emitVectorConstructor("float4", ["x"])).toBe("vec4<f32>(x, x, x, x)");
    expect(emitVectorConstructor("float2", ["x", "y"])).toBe("vec2<f32>(x, y)");
    expect(castExpressionToVectorScalar("x", "uint4")).toBe("u32(x)");
    expect(emitVectorSplat("int3", "v")).toBe("vec3<i32>(v, v, v)");
    expect(emitSharedFlatAccess("tile", [2, 4], "idx")).toBe("tile[min((((idx) / 4u) % 2u), 1u)][min((idx % 4u), 3u)]");
  });

  it("parses storage-view index units instead of making callers rediscover them", () => {
    const context = storageContext({
      params: [{ name: "out", valueType: "float3", pointer: true }],
    });

    const parsed = resolveStorageViewForPointerExpression(id("p"), id("i"), context, storageCallbacks);

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.view.root).toEqual({ kind: "param", name: "out", valueType: "float3" });
    expect(parsed.view.rootBytes).toBe(12);
    expect(parsed.view.viewBytes).toBe(4);
    expect(parsed.view.flatScalarVectorStorage).toBe(true);
    expect(parsed.view.indexUnit).toBe("flat-scalar-lane");
    expect(parsed.view.subElementLane).toBeUndefined();
    expect(parsed.view.index).toBe("(u32((1 * 3)) + u32(i))");
  });

  it("keeps packed shared vector scalar views distinct from flat atomic lanes", () => {
    const packed = storageContext({
      shared: [varDecl("tile", "float3", "shared")],
    });
    const packedView = resolveStorageView("tile", number("0", 0), id("i"), "float", packed, storageCallbacks);

    expect(packedView.ok).toBe(true);
    if (!packedView.ok) return;
    expect(packedView.view.root.kind).toBe("shared");
    expect(packedView.view.flatScalarVectorStorage).toBe(false);
    expect(packedView.view.indexUnit).toBe("storage-element");
    expect(packedView.view.index).toBe("(u32(0) + (u32(i) / 3u))");
    expect(packedView.view.subElementLane).toBe("(u32(i) % 3u)");

    const atomicFlat = storageContext({
      shared: [varDecl("tile", "float3", "shared")],
      atomicShared: ["tile"],
    });
    const flatView = resolveStorageView("tile", number("0", 0), id("i"), "float", atomicFlat, storageCallbacks);

    expect(flatView.ok).toBe(true);
    if (!flatView.ok) return;
    expect(flatView.view.flatScalarVectorStorage).toBe(true);
    expect(flatView.view.indexUnit).toBe("flat-scalar-lane");
    expect(flatView.view.index).toBe("(u32(0) + u32(i))");
    expect(flatView.view.subElementLane).toBeUndefined();
  });
});

describe("WGSL value conversion helpers", () => {
  it("normalizes CUDA number literals without losing integer/floating intent", () => {
    expect(emitNumberLiteral("1.")).toBe("1.0");
    expect(emitNumberLiteral(".5f")).toBe("0.5");
    expect(emitNumberLiteral("42UL")).toBe("42u");
    expect(emitNumberLiteral("0xffu")).toBe("0xffu");
    expect(emitNumberLiteralAsU32("7")).toBe("7u");
    expect(numberLiteralHasFloatSyntax("1e-3")).toBe(true);
    expect(numberLiteralHasFloatSyntax("0xff")).toBe(false);
    expect(numberLiteralHasUnsignedSuffix("42UL")).toBe(true);
    expect(numberLiteralHasIntegerSuffix("42LL")).toBe(true);
  });

  it("classifies operators and scalar promotions for assignment/cast paths", () => {
    expect(isComparisonOperator("<=")).toBe(true);
    expect(isShiftOperator("<<")).toBe(true);
    expect(isBitwiseOperator("^")).toBe(true);
    expect(isVectorArithmeticOperator("*")).toBe(true);
    expect(promotedCudaScalarType("int", "float")).toBe("float");
    expect(promotedCudaScalarType("uint", "int")).toBe("uint");
    expect(promotedCudaScalarType("half", "int")).toBe("half");
    expect(promotedCudaScalarType("float2", "float")).toBeUndefined();
    expect(emitExpressionAsWgslScalarText("x", "uint")).toBe("u32(x)");
    expect(updateDeltaForValueType("float")).toBe("1.0");
    expect(updateDeltaForValueType("uint")).toBe("1u");
  });

  it("detects abstract integer literals and emits vector lane updates", () => {
    const plain = number("15", 15);
    const suffixed = number("15u", 15);
    const floating = number("1.5", 1.5);
    expect(isIntegerNumberLiteral(plain)).toBe(true);
    expect(isIntegerNumberLiteral(floating)).toBe(false);
    expect(isAbstractIntegerLiteral(plain)).toBe(true);
    expect(isAbstractIntegerLiteral(suffixed)).toBe(false);
    expect(emitVectorLaneSetExpression("v", "float4", 2, "x")).toBe("vec4<f32>(select((v).x, f32(x), 2u == 0u), select((v).y, f32(x), 2u == 1u), select((v).z, f32(x), 2u == 2u), select((v).w, f32(x), 2u == 3u))");
    expect(emitVectorLaneSetExpression("bits", "uint2", "lane", "x")).toBe("vec2<u32>(select((bits).x, u32(x), lane == 0u), select((bits).y, u32(x), lane == 1u))");
  });
});

function number(raw: string, value: number): CudaLiteNumberLiteral {
  return { kind: "number", raw, value, span };
}

function emitTestExpression(expression: CudaLiteExpression): string {
  if (expression.kind === "number") return expression.raw;
  if (expression.kind === "identifier") return expression.name;
  if (expression.kind === "binary") return `(${emitTestExpression(expression.left)} ${expression.operator} ${emitTestExpression(expression.right)})`;
  throw new Error(`unsupported test expression ${expression.kind}`);
}

function varDecl(name: string, valueType: CudaLiteVarDecl["valueType"], storage: CudaLiteVarDecl["storage"]): CudaLiteVarDecl {
  return {
    kind: "var",
    name,
    valueType,
    storage,
    pointer: false,
    dimensions: [2],
    span,
  };
}

function storageContext(options: {
  readonly params?: readonly { readonly name: string; readonly valueType: "float" | "float3"; readonly pointer: boolean }[];
  readonly shared?: readonly CudaLiteVarDecl[];
  readonly atomicShared?: readonly string[];
}): WgslStorageViewContext {
  const params = new Map((options.params ?? []).map((param) => [param.name, {
    name: param.name,
    valueType: param.valueType,
    pointer: param.pointer,
    constant: false,
    span,
  }]));
  const shared = new Map((options.shared ?? []).map((decl) => [decl.name, decl]));
  return {
    ir: {
      name: "test",
      params: [...params.values()],
      constants: [],
      textures: [],
      body: [],
      functions: [],
      deviceGlobals: [],
      sharedDeclarations: [...shared.values()],
      atomicParams: [],
      atomicDeviceGlobals: [],
      atomicShared: options.atomicShared ?? [],
      requiredFeatures: [],
      workgroupSize: [1, 1, 1],
    },
    nameFor: (name) => name,
    paramFor: (name) => params.get(name),
    deviceGlobalFor: () => undefined,
    devicePointerParamFor: () => undefined,
    pointerAliasFor: (name) => name === "p"
      ? { rootName: "out", baseIndex: number("1", 1), valueType: "float" }
      : undefined,
    localArrayFor: () => undefined,
  };
}

const storageCallbacks: WgslStorageViewCallbacks = {
  emitExpression: (expression) => emitTestExpression(expression),
  emitExpressionAsWgslScalar: (expression) => emitTestExpression(expression),
  pointerBaseExpression: () => undefined,
  sharedDeclarationFor: (name, context) => context.ir.sharedDeclarations.find((decl) => decl.name === name),
  devicePointerValueTypeForExpression: () => "float",
};

describe("WGSL atomic helpers", () => {
  it("names atomic helper families by value and address space", () => {
    expect(floatAtomicHelperName("Add", "storage")).toBe("bg_atomicAdd_f32");
    expect(floatAtomicHelperName("Max", "workgroup")).toBe("bg_atomicMax_f32_workgroup");
    expect(integerAtomicLoopHelperName("Inc", {
      valueType: "uint",
      storageValueType: "float",
      storageScalar: "u32",
      addressSpace: "storage",
    })).toBe("bg_atomicInc_storage_f32_as_u32");
    expect(integerAtomicLoopHelperName("Dec", {
      valueType: "int",
      storageValueType: "int",
      storageScalar: "i32",
      addressSpace: "workgroup",
    })).toBe("bg_atomicDec_workgroup_i32");
  });

  it("classifies CUDA atomic call names without conflating CAS/exchange/returning ops", () => {
    expect(isAtomicCasCallName("atomicCAS_system")).toBe(true);
    expect(isAtomicCasCallName("atomicAdd")).toBe(false);
    expect(isAtomicExchangeCallName("atomicExch")).toBe(true);
    expect(isAtomicReturnCallName("atomicAdd_system")).toBe(true);
    expect(isAtomicReturnCallName("atomicExch_system")).toBe(true);
    expect(isAtomicReturnCallName("atomicCAS")).toBe(false);
  });

  it("emits CAS-loop helpers for float and integer atomic paths", () => {
    const add = emitFloatAtomicAddHelper("storage").join("\n");
    expect(add).toContain("fn bg_atomicAdd_f32(ptr_value: ptr<storage, atomic<u32>, read_write>, value: f32) -> f32");
    expect(add).toContain("atomicCompareExchangeWeak(ptr_value, old_bits, new_bits)");
    const max = emitFloatAtomicMaxHelper("workgroup").join("\n");
    expect(max).toContain("fn bg_atomicMax_f32_workgroup(ptr_value: ptr<workgroup, atomic<u32>>, value: f32) -> f32");
    expect(max).toContain("let new_value = max(old_value, value);");
    const integerHelpers = emitIntegerAtomicLoopHelpers().join("\n");
    expect(integerHelpers).toContain("fn bg_atomicInc_storage_u32");
    expect(integerHelpers).toContain("fn bg_atomicDec_workgroup_i32");
    expect(integerHelpers).toContain("fn bg_atomicInc_workgroup_f32_as_u32");
  });
});

describe("WGSL control analysis helpers", () => {
  const context = { ir: { workgroupSize: [1, 8, 1] as const } };

  it("folds simple constant numbers and boolean predicates", () => {
    expect(constantNumberExpression(member("threadIdx", "x"), context)).toBe(0);
    expect(constantNumberExpression(member("threadIdx", "y"), context)).toBeUndefined();
    expect(constantBooleanExpression(binary("==", member("threadIdx", "x"), number("0", 0)), context)).toBe(true);
    expect(constantBooleanExpression({ kind: "unary", operator: "!", argument: number("0", 0), span }, context)).toBe(true);
  });

  it("detects barrier and subgroup calls through statements and expressions", () => {
    const barrier = exprStmt(call("__syncthreads"));
    const subgroupCall = call("__shfl_sync");
    const cooperative = exprStmt({
      kind: "call",
      callee: member("tile", "shfl_down"),
      args: [],
      span,
    });
    expect(isBarrierCall(call("__syncwarp"))).toBe(true);
    expect(statementContainsBarrier({ kind: "block", body: [barrier], span })).toBe(true);
    expect(isSubgroupCallName("warp_reduce_sum")).toBe(true);
    expect(isCooperativeGroupSubgroupMember("ballot")).toBe(true);
    expect(expressionContainsSubgroupCall(subgroupCall)).toBe(true);
    expect(statementContainsSubgroupCall({ kind: "block", body: [cooperative], span })).toBe(true);
  });

  it("splits trailing return/break only when safe before barriers", () => {
    const condition = id("done");
    const assign = exprStmt({
      kind: "assignment",
      operator: "=",
      left: id("x"),
      right: number("1", 1),
      span,
    });
    const trailingReturn: CudaLiteStatement = {
      kind: "if",
      condition,
      consequent: [assign, { kind: "return", span }],
      span,
    };
    expect(splitIfTrailingVoidReturn(trailingReturn)?.beforeReturn).toHaveLength(1);
    const alternateReturn: CudaLiteStatement = {
      kind: "if",
      condition,
      consequent: [assign],
      alternate: [{ kind: "return", span }],
      span,
    };
    const alternateSplit = splitIfTrailingVoidReturn(alternateReturn);
    expect(alternateSplit?.activeBranch).toHaveLength(1);
    expect(alternateSplit?.condition.kind).toBe("unary");
    const nestedReturn: CudaLiteStatement = {
      kind: "if",
      condition,
      consequent: [alternateReturn],
      span,
    };
    expect(splitIfTrailingVoidReturn(nestedReturn)).toBeUndefined();
    expect(statementContainsVoidReturn(nestedReturn)).toBe(true);
    const trailingBreak: CudaLiteStatement = {
      kind: "if",
      condition,
      consequent: [assign, { kind: "break", span }],
      span,
    };
    expect(splitIfTrailingBreak(trailingBreak)?.beforeBreak).toHaveLength(1);
    const unsafe: CudaLiteStatement = {
      kind: "if",
      condition,
      consequent: [exprStmt(call("__syncthreads")), { kind: "return", span }],
      span,
    };
    expect(splitIfTrailingVoidReturn(unsafe)).toBeUndefined();
  });

  it("counts returns and assigned roots across nested statements", () => {
    const statements: CudaLiteStatement[] = [
      {
        kind: "if",
        condition: id("p"),
        consequent: [{ kind: "return", value: number("1", 1), span }],
        alternate: [{ kind: "return", value: number("2", 2), span }],
        span,
      },
      exprStmt({ kind: "assignment", operator: "=", left: id("x"), right: number("3", 3), span }),
      exprStmt({ kind: "update", operator: "++", argument: id("y"), prefix: false, span }),
    ];
    expect(countReturns(statements)).toBe(2);
    expect([...collectAssignedNames(statements)].sort()).toEqual(["x", "y"]);
  });
});

function id(name: string): CudaLiteExpression {
  return { kind: "identifier", name, span };
}

function member(object: string, property: string): CudaLiteExpression {
  return { kind: "member", object: id(object), property, span };
}

function call(name: string): CudaLiteCallExpression {
  return { kind: "call", callee: id(name), args: [], span };
}

function binary(
  operator: Extract<CudaLiteExpression, { kind: "binary" }>["operator"],
  left: CudaLiteExpression,
  right: CudaLiteExpression,
): CudaLiteExpression {
  return { kind: "binary", operator, left, right, span };
}

function exprStmt(expression: CudaLiteExpression): CudaLiteStatement {
  return { kind: "expr", expression, span };
}
