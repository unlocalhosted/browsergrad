import type {
  CudaLiteExpression,
  CudaLiteMatrixTileMetadata,
  CudaLiteScalarType,
  CudaLiteVarDecl,
} from "./types.js";

export type MatrixTileRole = "matrix_a" | "matrix_b" | "accumulator";
export type MatrixTileLayout = "row_major" | "col_major" | "mem_row_major" | "mem_col_major";
export type WmmaBuiltin = "fill_fragment" | "load_matrix_sync" | "mma_sync" | "store_matrix_sync";

export interface MatrixTileResolvedSpec {
  readonly role: MatrixTileRole;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly layout?: MatrixTileLayout;
}

export interface MatrixTileReference {
  readonly root: string;
  readonly indices: readonly CudaLiteExpression[];
}

export function wmmaBuiltinName(name: string | undefined): WmmaBuiltin | undefined {
  if (name === undefined) return undefined;
  const short = stripCudaNamespace(name);
  switch (short) {
    case "wmma::fill_fragment":
      return "fill_fragment";
    case "wmma::load_matrix_sync":
      return "load_matrix_sync";
    case "wmma::mma_sync":
      return "mma_sync";
    case "wmma::store_matrix_sync":
      return "store_matrix_sync";
    default:
      return undefined;
  }
}

export function stripCudaNamespace(name: string): string {
  return name.startsWith("nvcuda::") ? name.slice("nvcuda::".length) : name;
}

export function normalizeMatrixTileRole(role: string): MatrixTileRole | undefined {
  const short = stripCudaNamespace(role);
  if (short === "wmma::matrix_a") return "matrix_a";
  if (short === "wmma::matrix_b") return "matrix_b";
  if (short === "wmma::accumulator") return "accumulator";
  return undefined;
}

export function normalizeMatrixTileLayout(layout: string | undefined): MatrixTileLayout | undefined {
  if (layout === undefined) return undefined;
  const short = stripCudaNamespace(layout);
  if (short === "wmma::row_major") return "row_major";
  if (short === "wmma::col_major") return "col_major";
  if (short === "wmma::mem_row_major") return "mem_row_major";
  if (short === "wmma::mem_col_major") return "mem_col_major";
  return undefined;
}

export function resolveMatrixTileSpec(tile: CudaLiteMatrixTileMetadata): MatrixTileResolvedSpec | undefined {
  const role = normalizeMatrixTileRole(tile.role);
  const layout = normalizeMatrixTileLayout(tile.layout);
  if (
    role === undefined ||
    tile.m.value === undefined ||
    tile.n.value === undefined ||
    tile.k.value === undefined ||
    tile.valueType === undefined ||
    (tile.layout !== undefined && layout === undefined)
  ) {
    return undefined;
  }
  return {
    role,
    m: tile.m.value,
    n: tile.n.value,
    k: tile.k.value,
    valueType: tile.valueType,
    ...(layout === undefined ? {} : { layout }),
  };
}

export function matrixTileElementCount(tile: MatrixTileResolvedSpec): number {
  if (tile.role === "matrix_a") return tile.m * tile.k;
  if (tile.role === "matrix_b") return tile.k * tile.n;
  return tile.m * tile.n;
}

export function matrixTileStorageDimensions(declaration: CudaLiteVarDecl): readonly number[] {
  const spec = declaration.matrixTile ? resolveMatrixTileSpec(declaration.matrixTile) : undefined;
  if (!spec) return declaration.dimensions;
  const leading = declaration.dimensions.reduce((product, dimension) => product * dimension, 1);
  return [Math.max(1, leading) * matrixTileElementCount(spec)];
}

export function matrixTileReference(expression: CudaLiteExpression): MatrixTileReference | undefined {
  const indices: CudaLiteExpression[] = [];
  let cursor = expression;
  while (cursor.kind === "index") {
    indices.unshift(cursor.index);
    cursor = cursor.target;
  }
  return cursor.kind === "identifier" ? { root: cursor.name, indices } : undefined;
}

export function flattenMatrixTileLeadingIndex(dimensions: readonly number[], indices: readonly number[]): number {
  if (dimensions.length !== indices.length) return -1;
  let flat = 0;
  for (let i = 0; i < dimensions.length; i++) {
    flat = flat * dimensions[i]! + indices[i]!;
  }
  return flat;
}
