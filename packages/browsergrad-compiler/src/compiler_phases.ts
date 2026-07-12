declare const cudaLiteCompilerPhase: unique symbol;

export type CompilerPhase<Name extends string> = {
  readonly [cudaLiteCompilerPhase]: Readonly<Record<Name, true>>;
};

export type Parsed<T> = T & CompilerPhase<"parsed">;
export type Analyzed<T> = T & CompilerPhase<"analyzed">;
export type TypedSemantic<T> = T & CompilerPhase<"typed-semantic">;
export type CanonicalIr<T> = T & CompilerPhase<"canonical-ir">;
export type RuntimeLoweredIr<T> = T & CompilerPhase<"runtime-lowered-ir">;
export type VerifiedIr<T> = T & CompilerPhase<"verified-ir">;
export type TypeCheckedIr<T> = T & CompilerPhase<"type-checked-ir">;
export type WgslLegalizedIr<T> = T & CompilerPhase<"wgsl-legalized-ir">;

function completePhase<T, Name extends string>(value: T): T & CompilerPhase<Name> {
  return value as T & CompilerPhase<Name>;
}

export function completeParsing<T>(value: T): Parsed<T> {
  return completePhase(value);
}

export function completeAnalysis<T>(value: T, _parsed: Parsed<unknown>): Analyzed<T> {
  return completePhase(value);
}

export function completeSemanticTyping<T>(value: T, _analysis: Analyzed<unknown>): TypedSemantic<T> {
  return completePhase(value);
}

export function completeCanonicalLowering<T>(value: T, _semantic: TypedSemantic<unknown>): CanonicalIr<T> {
  return completePhase(value);
}

export function completeRuntimeLowering<T extends CanonicalIr<unknown>>(value: T): RuntimeLoweredIr<T> {
  return completePhase(value);
}
