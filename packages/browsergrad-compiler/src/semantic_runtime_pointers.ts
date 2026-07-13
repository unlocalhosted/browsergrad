import type {
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { semanticPointerSymbolNeedsRuntimeState } from "./semantic_ir.js";
import { semanticIdsEqual } from "./semantic_ids.js";
import { isSemanticKernelIrOperation } from "./semantic_ir_walk.js";

export type SemanticRuntimePointerDeclaration = Extract<
  SemanticKernelIrOperation,
  { readonly kind: "declare" }
>;

export function semanticRuntimePointerDeclarations(
  ir: SemanticKernelIrModule,
): readonly SemanticRuntimePointerDeclaration[] {
  return [
    ...collectSemanticRuntimePointerDeclarations(ir.operations),
    ...ir.functions.flatMap((fn) => collectSemanticRuntimePointerDeclarations(fn.body)),
  ];
}

export function semanticPointerDeclarationNeedsRuntimeState(
  declaration: SemanticRuntimePointerDeclaration,
): boolean {
  return semanticPointerSymbolNeedsRuntimeState(declaration.target);
}

export function semanticRuntimePointerDeclarationForRef(
  ir: SemanticKernelIrModule,
  ref: SemanticMemoryRef,
): SemanticRuntimePointerDeclaration | undefined {
  if (ref.addressSpace !== "local") return undefined;
  return semanticRuntimePointerDeclarations(ir).find((operation) =>
    semanticIdsEqual(operation.target.id, ref.baseId) &&
    operation.target.dimensions.length === 0 &&
    semanticPointerDeclarationNeedsRuntimeState(operation)
  );
}

function collectSemanticRuntimePointerDeclarations(
  operations: readonly SemanticKernelIrOperation[],
): readonly SemanticRuntimePointerDeclaration[] {
  const declarations: SemanticRuntimePointerDeclaration[] = [];
  for (const operation of operations) {
    if (operation.kind === "declare" && semanticPointerDeclarationNeedsRuntimeState(operation)) {
      declarations.push(operation);
    }
    if (operation.kind === "block") {
      declarations.push(...collectSemanticRuntimePointerDeclarations(operation.body));
    }
    if (operation.kind === "branch") {
      declarations.push(...collectSemanticRuntimePointerDeclarations(operation.consequent));
      declarations.push(...collectSemanticRuntimePointerDeclarations(operation.alternate));
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        declarations.push(...collectSemanticRuntimePointerDeclarations([operation.init]));
      }
      declarations.push(...collectSemanticRuntimePointerDeclarations(operation.body));
      if (operation.continuing) {
        declarations.push(...collectSemanticRuntimePointerDeclarations(operation.continuing));
      }
    }
  }
  return declarations;
}
