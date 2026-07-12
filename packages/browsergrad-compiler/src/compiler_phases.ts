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
export type WgslLegalizedIr<T> = T & CompilerPhase<"wgsl-legalized-ir">;

export function markCompilerPhase<T, Name extends string>(
  value: T,
  _phase: Name,
): T & CompilerPhase<Name> {
  return value as T & CompilerPhase<Name>;
}
