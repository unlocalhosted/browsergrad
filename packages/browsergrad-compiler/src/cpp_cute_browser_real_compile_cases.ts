export const CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS = Object.freeze([
  "rank2",
  "rank3",
  "rank1",
  "rank4",
  "strided-slice",
  "broadcast",
  "i32-rank2",
  "u32-broadcast",
] as const);

export type CppCuteBrowserRealCompileCaseId =
  (typeof CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS)[number];
export type CppCuteBrowserRealCompileDType = "f32" | "i32" | "u32";

export interface CppCuteBrowserRealCompileLayoutProjection {
  readonly shape: readonly string[];
  readonly strides: readonly string[];
}

export interface CppCuteBrowserRealCompileCase {
  readonly caseId: CppCuteBrowserRealCompileCaseId;
  readonly virtualPath: string;
  readonly sourceSha256: string;
  readonly source: string;
  readonly coordinateRank: number;
  readonly dtype: CppCuteBrowserRealCompileDType;
  readonly sourceSpanElements: bigint;
  readonly destinationSpanElements: bigint;
  readonly sourceLayout: CppCuteBrowserRealCompileLayoutProjection;
  readonly destinationLayout: CppCuteBrowserRealCompileLayoutProjection;
}

const CASES = {
  rank2: defineCase({
    caseId: "rank2",
    virtualPath: "/workspace/src/real-view-copy-rank2.cu",
    sourceSha256:
      "4134804a9892ed1f0a2778fae305e957b5a981afccf2a096f1585f3b1d4e6f06",
    coordinateRank: 2,
    dtype: "f32",
    sourceSpanElements: 6n,
    destinationSpanElements: 6n,
    sourceLayout: { shape: ["3", "2"], strides: ["1", "3"] },
    destinationLayout: { shape: ["3", "2"], strides: ["2", "1"] },
    source: source([
      "#include <cute/tensor.hpp>",
      "using SourceLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<1>, cute::Int<3>>>;",
      "using DestinationLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<2>, cute::Int<1>>>;",
      "__device__ void copy_views(const float* source, float* destination) {",
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});",
      "  auto destination_tensor = cute::make_tensor(destination, DestinationLayout{});",
      "  cute::copy(source_tensor, destination_tensor);",
      "}",
    ]),
  }),
  rank3: defineCase({
    caseId: "rank3",
    virtualPath: "/workspace/src/real-view-copy-rank3.cu",
    sourceSha256:
      "6a7beae44e88d7fe8749cb5b485dc7d51d30ed285d33314895be461d428550dd",
    coordinateRank: 3,
    dtype: "f32",
    sourceSpanElements: 24n,
    destinationSpanElements: 24n,
    sourceLayout: { shape: ["2", "3", "4"], strides: ["1", "2", "6"] },
    destinationLayout: {
      shape: ["2", "3", "4"],
      strides: ["12", "4", "1"],
    },
    source: source([
      "#include <cute/tensor.hpp>",
      "using SourceLayout = cute::Layout<",
      "  cute::Shape<cute::Int<2>, cute::Int<3>, cute::Int<4>>,",
      "  cute::Stride<cute::Int<1>, cute::Int<2>, cute::Int<6>>>;",
      "using DestinationLayout = cute::Layout<",
      "  cute::Shape<cute::Int<2>, cute::Int<3>, cute::Int<4>>,",
      "  cute::Stride<cute::Int<12>, cute::Int<4>, cute::Int<1>>>;",
      "__device__ void copy_views(const float* source, float* destination) {",
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});",
      "  auto destination_tensor = cute::make_tensor(destination, DestinationLayout{});",
      "  cute::copy(source_tensor, destination_tensor);",
      "}",
    ]),
  }),
  rank1: defineCase({
    caseId: "rank1",
    virtualPath: "/workspace/src/real-view-copy-rank1.cu",
    sourceSha256:
      "7c8fc9f261fab7181e9c25d124f2604d31a48a5c9bc49e043c42f9371a27c1c7",
    coordinateRank: 1,
    dtype: "f32",
    sourceSpanElements: 7n,
    destinationSpanElements: 4n,
    sourceLayout: { shape: ["4"], strides: ["2"] },
    destinationLayout: { shape: ["4"], strides: ["1"] },
    source: source([
      "#include <cute/tensor.hpp>",
      "using SourceLayout = cute::Layout<",
      "  cute::Shape<cute::Int<4>>,",
      "  cute::Stride<cute::Int<2>>>;",
      "using DestinationLayout = cute::Layout<",
      "  cute::Shape<cute::Int<4>>,",
      "  cute::Stride<cute::Int<1>>>;",
      "__device__ void copy_views(const float* source, float* destination) {",
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});",
      "  auto destination_tensor = cute::make_tensor(destination, DestinationLayout{});",
      "  cute::copy(source_tensor, destination_tensor);",
      "}",
    ]),
  }),
  rank4: defineCase({
    caseId: "rank4",
    virtualPath: "/workspace/src/real-view-copy-rank4.cu",
    sourceSha256:
      "28d6094af10254112f25ea717739c836c2c74a4c3f35c7b88dcc45ad60e5a05a",
    coordinateRank: 4,
    dtype: "f32",
    sourceSpanElements: 16n,
    destinationSpanElements: 16n,
    sourceLayout: {
      shape: ["2", "2", "2", "2"],
      strides: ["1", "2", "4", "8"],
    },
    destinationLayout: {
      shape: ["2", "2", "2", "2"],
      strides: ["8", "4", "2", "1"],
    },
    source: source([
      "#include <cute/tensor.hpp>",
      "using SourceLayout = cute::Layout<",
      "  cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<2>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<1>, cute::Int<2>, cute::Int<4>, cute::Int<8>>>;",
      "using DestinationLayout = cute::Layout<",
      "  cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<2>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<8>, cute::Int<4>, cute::Int<2>, cute::Int<1>>>;",
      "__device__ void copy_views(const float* source, float* destination) {",
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});",
      "  auto destination_tensor = cute::make_tensor(destination, DestinationLayout{});",
      "  cute::copy(source_tensor, destination_tensor);",
      "}",
    ]),
  }),
  "strided-slice": defineCase({
    caseId: "strided-slice",
    virtualPath: "/workspace/src/real-view-copy-strided-slice.cu",
    sourceSha256:
      "55f4f5fcf55093a05cb977e3b83479098f6ddc42b830ec63f44b97f27fe3264a",
    coordinateRank: 2,
    dtype: "f32",
    sourceSpanElements: 12n,
    destinationSpanElements: 6n,
    sourceLayout: { shape: ["3", "2"], strides: ["2", "7"] },
    destinationLayout: { shape: ["3", "2"], strides: ["2", "1"] },
    source: source([
      "#include <cute/tensor.hpp>",
      "using SourceLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<2>, cute::Int<7>>>;",
      "using DestinationLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<2>, cute::Int<1>>>;",
      "__device__ void copy_views(const float* source, float* destination) {",
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});",
      "  auto destination_tensor = cute::make_tensor(destination, DestinationLayout{});",
      "  cute::copy(source_tensor, destination_tensor);",
      "}",
    ]),
  }),
  broadcast: defineCase({
    caseId: "broadcast",
    virtualPath: "/workspace/src/real-view-copy-broadcast.cu",
    sourceSha256:
      "bfd91bdaac57ef7314570a8de56f26165a7b263593f319d728c53c13ef7c6376",
    coordinateRank: 2,
    dtype: "f32",
    sourceSpanElements: 2n,
    destinationSpanElements: 6n,
    sourceLayout: { shape: ["3", "2"], strides: ["0", "1"] },
    destinationLayout: { shape: ["3", "2"], strides: ["2", "1"] },
    source: source([
      "#include <cute/tensor.hpp>",
      "using SourceLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<0>, cute::Int<1>>>;",
      "using DestinationLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<2>, cute::Int<1>>>;",
      "__device__ void copy_views(const float* source, float* destination) {",
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});",
      "  auto destination_tensor = cute::make_tensor(destination, DestinationLayout{});",
      "  cute::copy(source_tensor, destination_tensor);",
      "}",
    ]),
  }),
  "i32-rank2": defineCase({
    caseId: "i32-rank2",
    virtualPath: "/workspace/src/real-view-copy-i32-rank2.cu",
    sourceSha256:
      "88a083b141a5b7a85a9a1f2420873029f891cc1738e74ababe849a58bb839577",
    coordinateRank: 2,
    dtype: "i32",
    sourceSpanElements: 6n,
    destinationSpanElements: 6n,
    sourceLayout: { shape: ["3", "2"], strides: ["1", "3"] },
    destinationLayout: { shape: ["3", "2"], strides: ["2", "1"] },
    source: source([
      "#include <cute/tensor.hpp>",
      "using SourceLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<1>, cute::Int<3>>>;",
      "using DestinationLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<2>, cute::Int<1>>>;",
      "__device__ void copy_views(const int* source, int* destination) {",
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});",
      "  auto destination_tensor = cute::make_tensor(destination, DestinationLayout{});",
      "  cute::copy(source_tensor, destination_tensor);",
      "}",
    ]),
  }),
  "u32-broadcast": defineCase({
    caseId: "u32-broadcast",
    virtualPath: "/workspace/src/real-view-copy-u32-broadcast.cu",
    sourceSha256:
      "c21158f1cb394377b5b8057435bfaad64b515689a8b04391a1bcae643a567536",
    coordinateRank: 2,
    dtype: "u32",
    sourceSpanElements: 2n,
    destinationSpanElements: 6n,
    sourceLayout: { shape: ["3", "2"], strides: ["0", "1"] },
    destinationLayout: { shape: ["3", "2"], strides: ["2", "1"] },
    source: source([
      "#include <cute/tensor.hpp>",
      "using SourceLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<0>, cute::Int<1>>>;",
      "using DestinationLayout = cute::Layout<",
      "  cute::Shape<cute::Int<3>, cute::Int<2>>,",
      "  cute::Stride<cute::Int<2>, cute::Int<1>>>;",
      "__device__ void copy_views(" +
        "const unsigned int* source, unsigned int* destination) {",
      "  auto source_tensor = cute::make_tensor(source, SourceLayout{});",
      "  auto destination_tensor = cute::make_tensor(destination, DestinationLayout{});",
      "  cute::copy(source_tensor, destination_tensor);",
      "}",
    ]),
  }),
} as const satisfies Record<
  CppCuteBrowserRealCompileCaseId,
  CppCuteBrowserRealCompileCase
>;

export const CPP_CUTE_BROWSER_REAL_COMPILE_CASES:
Readonly<Record<CppCuteBrowserRealCompileCaseId, CppCuteBrowserRealCompileCase>> =
  Object.freeze(CASES);

export function cppCuteBrowserRealCompileCase(
  caseId: CppCuteBrowserRealCompileCaseId,
): CppCuteBrowserRealCompileCase {
  return CPP_CUTE_BROWSER_REAL_COMPILE_CASES[caseId];
}

function source(lines: readonly string[]): string {
  return [...lines, ""].join("\n");
}

function defineCase<const T extends CppCuteBrowserRealCompileCase>(
  value: T,
): Readonly<T> {
  return Object.freeze({
    ...value,
    sourceLayout: freezeLayout(value.sourceLayout),
    destinationLayout: freezeLayout(value.destinationLayout),
  });
}

function freezeLayout(
  value: CppCuteBrowserRealCompileLayoutProjection,
): CppCuteBrowserRealCompileLayoutProjection {
  return Object.freeze({
    shape: Object.freeze([...value.shape]),
    strides: Object.freeze([...value.strides]),
  });
}
