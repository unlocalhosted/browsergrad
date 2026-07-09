export const CUDA_LEGACY_VOTE_CALL_NAMES = ["__any", "__all", "__ballot"] as const;
export const CUDA_SYNC_VOTE_CALL_NAMES = ["__any_sync", "__all_sync", "__ballot_sync", "__match_any_sync"] as const;
export const CUDA_LEGACY_SHUFFLE_CALL_NAMES = ["__shfl", "__shfl_down", "__shfl_up", "__shfl_xor"] as const;
export const CUDA_SYNC_SHUFFLE_CALL_NAMES = ["__shfl_sync", "__shfl_down_sync", "__shfl_up_sync", "__shfl_xor_sync"] as const;

export type CudaShuffleOp = "sync" | "down" | "up" | "xor";
export type CudaVoteOp = "any" | "all" | "ballot" | "match-any";

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
