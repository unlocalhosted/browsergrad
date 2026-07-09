export function cudaLiteTruthy(value: number): boolean {
  return value !== 0 && !Number.isNaN(value);
}

export function cudaLiteTotalElements(dimensions: readonly number[]): number {
  return dimensions.length === 0 ? 1 : dimensions.reduce((product, dimension) => product * dimension, 1);
}
