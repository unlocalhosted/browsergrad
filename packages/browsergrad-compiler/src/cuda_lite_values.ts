export function cudaLiteTruthy(value: number): boolean {
  return value !== 0 && !Number.isNaN(value);
}

export function cudaLiteTotalElements(dimensions: readonly number[]): number {
  return dimensions.length === 0 ? 1 : dimensions.reduce((product, dimension) => product * dimension, 1);
}

export function cudaLiteDimensionStride(dimensions: readonly number[], offset: number): number {
  return dimensions.slice(offset + 1).reduce((product, dimension) => product * dimension, 1);
}

export function cudaLiteFlatIndexForDimensions(dimensions: readonly number[], indices: readonly number[]): number {
  return indices.reduce((sum, index, offset) => sum + index * cudaLiteDimensionStride(dimensions, offset), 0);
}

export function cudaLiteFlatIndicesForDimensions(dimensions: readonly number[], flatIndex: number): readonly number[] {
  return dimensions.map((dimension, offset) => {
    const stride = cudaLiteDimensionStride(dimensions, offset);
    return Math.floor(flatIndex / stride) % Math.max(1, dimension);
  });
}
