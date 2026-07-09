import {
  statementsUseCall,
  statementsUseIdentifier,
} from "./ir_usage.js";
import type {
  CudaLiteStatement,
  KernelIrModule,
} from "./types.js";

export function kernelIrUsesCall(ir: KernelIrModule, names: ReadonlySet<string>): boolean {
  return statementsUseCall(ir.body, names) ||
    ir.functions.some((fn) => statementsUseCall(fn.body, names));
}

export function kernelIrUsesIdentifier(ir: KernelIrModule, names: ReadonlySet<string>): boolean {
  return statementsUseIdentifier(ir.body, names) ||
    ir.functions.some((fn) => statementsUseIdentifier(fn.body, names));
}

export function kernelIrStatements(ir: KernelIrModule): readonly (readonly CudaLiteStatement[])[] {
  return [
    ir.body,
    ...ir.functions.map((fn) => fn.body),
  ];
}
