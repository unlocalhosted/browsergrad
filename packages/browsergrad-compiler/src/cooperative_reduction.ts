export type CooperativeReductionOperation = "add" | "min" | "max";

export function cooperativeReductionOperationForName(
  name: string | undefined,
): CooperativeReductionOperation | undefined {
  if (name?.endsWith("::plus")) return "add";
  if (name?.endsWith("::less")) return "min";
  if (name?.endsWith("::greater")) return "max";
  return undefined;
}

export function isCooperativeReductionObjectName(name: string): boolean {
  return cooperativeReductionOperationForName(name) !== undefined;
}
