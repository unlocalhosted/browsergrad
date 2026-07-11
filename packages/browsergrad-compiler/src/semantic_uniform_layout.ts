import type { CudaLiteScalarType } from "./types.js";
import { cudaVectorLaneCount, cudaVectorScalarType, isCudaVectorType } from "./vector_types.js";

export interface SemanticUniformField {
  readonly name: string;
  readonly valueType: CudaLiteScalarType;
}

export interface SemanticUniformFieldLayout extends SemanticUniformField {
  readonly offset: number;
  readonly align: number;
  readonly size: number;
}

export interface SemanticUniformLayout {
  readonly fields: readonly SemanticUniformFieldLayout[];
  readonly byteLength: number;
}

export function semanticUniformLayout(fields: readonly SemanticUniformField[]): SemanticUniformLayout {
  const laidOut: SemanticUniformFieldLayout[] = [];
  let offset = 0;
  let structAlign = 1;
  for (const field of fields) {
    const shape = semanticUniformShape(field.valueType);
    structAlign = Math.max(structAlign, shape.align);
    offset = alignTo(offset, shape.align);
    laidOut.push({ ...field, offset, ...shape });
    offset += shape.size;
  }
  return { fields: laidOut, byteLength: fields.length === 0 ? 0 : Math.max(16, alignTo(offset, structAlign)) };
}

function semanticUniformShape(valueType: CudaLiteScalarType): { readonly align: number; readonly size: number } {
  if (!isCudaVectorType(valueType)) return { align: 4, size: 4 };
  const lanes = cudaVectorLaneCount(valueType);
  const scalar = cudaVectorScalarType(valueType);
  const laneBytes = scalar === "half" ? 2 : 4;
  if (lanes === 2) return { align: laneBytes * 2, size: laneBytes * 2 };
  return { align: laneBytes * 4, size: laneBytes * lanes };
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}
