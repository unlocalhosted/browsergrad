export const CUDA_LEGACY_VOTE_CALL_NAMES = ["__any", "__all", "__ballot"] as const;
export const CUDA_SYNC_VOTE_CALL_NAMES = ["__any_sync", "__all_sync", "__ballot_sync", "__match_any_sync"] as const;
export const CUDA_LEGACY_SHUFFLE_CALL_NAMES = ["__shfl", "__shfl_down", "__shfl_up", "__shfl_xor"] as const;
export const CUDA_SYNC_SHUFFLE_CALL_NAMES = ["__shfl_sync", "__shfl_down_sync", "__shfl_up_sync", "__shfl_xor_sync"] as const;
export const CUDA_ARITHMETIC_REDUCE_CALL_NAMES = ["__reduce_add_sync", "__reduce_min_sync", "__reduce_max_sync"] as const;
export const CUDA_BITWISE_REDUCE_CALL_NAMES = ["__reduce_and_sync", "__reduce_or_sync", "__reduce_xor_sync"] as const;
export const CUDA_WARP_SUM_CALL_NAMES = [
  "warpReduceSum",
  "warp_reduce_sum",
  "warp_reduce_sum_f32",
  "warp_reduce_sum_f16",
  "warp_reduce_sum_f16_f16",
  "warp_reduce_sum_f16_f32",
  "warp_reduce_sum_i8_i32",
  "warp_reduce_sum_i32_i32",
] as const;
export const CUDA_WARP_MIN_CALL_NAMES = ["warp_reduce_min"] as const;
export const CUDA_WARP_MAX_CALL_NAMES = ["warpReduceMax", "warp_reduce_max", "warp_reduce_max_f32"] as const;
export const CUDA_COMPAT_SUBGROUP_CALL_NAMES = ["bg_subgroup_add"] as const;
export const CUDA_SUBGROUP_CALL_NAMES = [
  "__activemask",
  ...CUDA_LEGACY_VOTE_CALL_NAMES,
  ...CUDA_SYNC_VOTE_CALL_NAMES,
  ...CUDA_ARITHMETIC_REDUCE_CALL_NAMES,
  ...CUDA_BITWISE_REDUCE_CALL_NAMES,
  ...CUDA_WARP_SUM_CALL_NAMES,
  ...CUDA_WARP_MIN_CALL_NAMES,
  ...CUDA_WARP_MAX_CALL_NAMES,
  ...CUDA_COMPAT_SUBGROUP_CALL_NAMES,
  ...CUDA_LEGACY_SHUFFLE_CALL_NAMES,
  ...CUDA_SYNC_SHUFFLE_CALL_NAMES,
] as const;

export type CudaShuffleOp = "sync" | "down" | "up" | "xor";
export type CudaVoteOp = "any" | "all" | "ballot" | "match-any";
export type CudaArithmeticReduceOp = "add" | "min" | "max";
export type CudaBitwiseReduceOp = "and" | "or" | "xor";

export function isCudaLegacyVoteCallName(name: string | undefined): boolean {
  return name === "__any" || name === "__all" || name === "__ballot";
}

export function isCudaVoteCallName(name: string | undefined): boolean {
  return isCudaLegacyVoteCallName(name) ||
    name === "__any_sync" ||
    name === "__all_sync" ||
    name === "__ballot_sync" ||
    name === "__match_any_sync";
}

export function cudaVoteOpForCall(name: string | undefined): CudaVoteOp | undefined {
  if (name === "__any" || name === "__any_sync") return "any";
  if (name === "__all" || name === "__all_sync") return "all";
  if (name === "__ballot" || name === "__ballot_sync") return "ballot";
  if (name === "__match_any_sync") return "match-any";
  return undefined;
}

export function isCudaLegacyShuffleCallName(name: string | undefined): boolean {
  return name === "__shfl" || name === "__shfl_down" || name === "__shfl_up" || name === "__shfl_xor";
}

export function isCudaShuffleCallName(name: string | undefined): boolean {
  return isCudaLegacyShuffleCallName(name) ||
    name === "__shfl_sync" ||
    name === "__shfl_down_sync" ||
    name === "__shfl_up_sync" ||
    name === "__shfl_xor_sync";
}

export function cudaShuffleOpForCall(name: string | undefined): CudaShuffleOp | undefined {
  if (name === "__shfl" || name === "__shfl_sync") return "sync";
  if (name === "__shfl_down" || name === "__shfl_down_sync") return "down";
  if (name === "__shfl_up" || name === "__shfl_up_sync") return "up";
  if (name === "__shfl_xor" || name === "__shfl_xor_sync") return "xor";
  return undefined;
}

export function cudaArithmeticReduceOpForCall(name: string | undefined): CudaArithmeticReduceOp | undefined {
  if (name === "__reduce_add_sync" || name === "bg_subgroup_add" || isCudaWarpSumCallName(name)) return "add";
  if (name === "__reduce_min_sync" || isCudaWarpMinCallName(name)) return "min";
  if (name === "__reduce_max_sync" || isCudaWarpMaxCallName(name)) return "max";
  return undefined;
}

export function isCudaWarpSumCallName(name: string | undefined): boolean {
  return CUDA_WARP_SUM_CALL_NAMES.some((candidate) => candidate === name);
}

export function isCudaWarpMaxCallName(name: string | undefined): boolean {
  return CUDA_WARP_MAX_CALL_NAMES.some((candidate) => candidate === name);
}

export function isCudaWarpMinCallName(name: string | undefined): boolean {
  return CUDA_WARP_MIN_CALL_NAMES.some((candidate) => candidate === name);
}

export function isCudaWarpReduceCallName(name: string | undefined): boolean {
  return isCudaWarpSumCallName(name) || isCudaWarpMinCallName(name) || isCudaWarpMaxCallName(name);
}

export function cudaBitwiseReduceOpForCall(name: string | undefined): CudaBitwiseReduceOp | undefined {
  if (name === "__reduce_and_sync") return "and";
  if (name === "__reduce_or_sync") return "or";
  if (name === "__reduce_xor_sync") return "xor";
  return undefined;
}

export function isCudaArithmeticReduceCallName(name: string | undefined): boolean {
  return cudaArithmeticReduceOpForCall(name) !== undefined;
}

export function isCudaBitwiseReduceCallName(name: string | undefined): boolean {
  return cudaBitwiseReduceOpForCall(name) !== undefined;
}
