export const CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES = [
  "__isGlobal",
  "__isShared",
  "__isConstant",
  "__isLocal",
] as const;

export type CudaAddressSpacePredicateCallName = typeof CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES[number];

export type CudaAddressSpacePredicateKind = "global" | "shared" | "constant" | "local";

export function isCudaAddressSpacePredicateCallName(name: string | undefined): name is CudaAddressSpacePredicateCallName {
  return name === "__isGlobal" || name === "__isShared" || name === "__isConstant" || name === "__isLocal";
}

export function cudaAddressSpacePredicateKind(name: string | undefined): CudaAddressSpacePredicateKind | undefined {
  if (name === "__isGlobal") return "global";
  if (name === "__isShared") return "shared";
  if (name === "__isConstant") return "constant";
  if (name === "__isLocal") return "local";
  return undefined;
}

export function isCudaPointerIdentityCallName(name: string | undefined): boolean {
  return name === "__builtin_assume_aligned" || name === "ct::assume_aligned";
}
