import type { SemanticKernelIrModule, SemanticKernelIrOperation } from "./semantic_ir_types.js";

export interface SemanticCooperativeGroupInfo {
  readonly kind: "thread" | "block" | "grid" | "tile" | "coalesced" | "binary";
  readonly tileSize?: number;
  readonly partitioned?: boolean;
  readonly partitionParent?: string;
  readonly partitionPredicate?: import("./semantic_ir.js").SemanticExpression;
}

export function semanticCooperativeGroupInfo(
  ir: SemanticKernelIrModule,
  name: string,
): SemanticCooperativeGroupInfo | undefined {
  const declaration = [...ir.operations, ...ir.functions.flatMap((fn) => fn.body)]
    .flatMap(semanticOperationsDeep)
    .find((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }> =>
      operation.kind === "cooperative-group-declare" && operation.declaration.name === name,
    );
  if (declaration) {
    return {
      kind: declaration.declaration.groupKind,
      ...(declaration.declaration.tileSize === undefined ? {} : { tileSize: declaration.declaration.tileSize }),
      ...(declaration.declaration.partitionParent === undefined ? {} : { partitioned: true }),
      ...(declaration.declaration.partitionParent === undefined ? {} : { partitionParent: declaration.declaration.partitionParent }),
      ...(declaration.declaration.partitionPredicate === undefined ? {} : { partitionPredicate: declaration.declaration.partitionPredicate }),
    };
  }
  const param = ir.functions.flatMap((fn) => fn.params)
    .find((item) => item.name === name && item.cooperativeGroupKind !== undefined);
  return param?.cooperativeGroupKind === undefined
    ? undefined
    : { kind: param.cooperativeGroupKind, ...(param.tileSize === undefined ? {} : { tileSize: param.tileSize }) };
}

export function semanticCooperativeGroupRankParamName(name: string): string {
  return `${name}__bg_group_rank`;
}

export function semanticCooperativeGroupSizeParamName(name: string): string {
  return `${name}__bg_group_size`;
}

function semanticOperationsDeep(operation: SemanticKernelIrOperation): readonly SemanticKernelIrOperation[] {
  if (operation.kind === "branch") return [operation, ...operation.consequent.flatMap(semanticOperationsDeep), ...operation.alternate.flatMap(semanticOperationsDeep)];
  if (operation.kind === "loop" || operation.kind === "block") return [operation, ...operation.body.flatMap(semanticOperationsDeep)];
  return [operation];
}
