export const CUDA_BARRIER_CALL_NAMES = ["__syncthreads", "__syncwarp"] as const;
export const CUDA_FENCE_CALL_NAMES = ["__threadfence", "__threadfence_block", "__threadfence_system"] as const;
export const CUDA_SYNCTHREADS_PREDICATE_CALL_NAMES = ["__syncthreads_count", "__syncthreads_and", "__syncthreads_or"] as const;

export type CudaBarrierCallName = typeof CUDA_BARRIER_CALL_NAMES[number];
export type CudaFenceCallName = typeof CUDA_FENCE_CALL_NAMES[number];
export type CudaSyncthreadsPredicateCallName = typeof CUDA_SYNCTHREADS_PREDICATE_CALL_NAMES[number];
export type CudaSyncthreadsPredicateReduction = "count" | "and" | "or";
export type CudaSyncthreadsCollectiveOp = "sum" | "all" | "any";

export function isCudaBarrierCallName(name: string | undefined): name is CudaBarrierCallName {
  return name === "__syncthreads" || name === "__syncwarp";
}

export function isCudaFenceCallName(name: string | undefined): name is CudaFenceCallName {
  return name === "__threadfence" || name === "__threadfence_block" || name === "__threadfence_system";
}

export function isCudaSyncthreadsPredicateCallName(name: string | undefined): name is CudaSyncthreadsPredicateCallName {
  return name === "__syncthreads_count" || name === "__syncthreads_and" || name === "__syncthreads_or";
}

export function cudaSyncthreadsPredicateReduction(name: string | undefined): CudaSyncthreadsPredicateReduction | undefined {
  if (name === "__syncthreads_count") return "count";
  if (name === "__syncthreads_and") return "and";
  if (name === "__syncthreads_or") return "or";
  return undefined;
}

export function cudaSyncthreadsCollectiveOp(name: string | undefined): CudaSyncthreadsCollectiveOp | undefined {
  if (name === "__syncthreads_count") return "sum";
  if (name === "__syncthreads_and") return "all";
  if (name === "__syncthreads_or") return "any";
  return undefined;
}
