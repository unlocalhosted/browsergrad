declare const cudaLiteCompilerPhase: unique symbol;

export type CompilerPhase<Name extends string> = {
  readonly [cudaLiteCompilerPhase]: Readonly<Record<Name, true>>;
};

export type Parsed<T> = T & CompilerPhase<"parsed">;
export type Analyzed<T> = T & CompilerPhase<"analyzed">;
export type TypedSemantic<T> = T & CompilerPhase<"typed-semantic">;
export type CanonicalIr<T> = T & CompilerPhase<"canonical-ir">;
export type RuntimeLoweredIr<T> = T & CompilerPhase<"runtime-lowered-ir">;

export interface VerifiedIrArtifact<T> extends CompilerPhase<"verified-ir"> {
  readonly kind: "verified-semantic-kernel-ir";
  readonly ir: T;
}

export interface TypeCheckedIrArtifact<T> extends CompilerPhase<"type-checked-ir"> {
  readonly kind: "type-checked-semantic-kernel-ir";
  readonly ir: T;
  readonly verified: VerifiedIrArtifact<T>;
}

export interface WgslLegalizedIrArtifact<T> extends CompilerPhase<"wgsl-legalized-ir"> {
  readonly kind: "wgsl-legalized-semantic-kernel-ir";
  readonly ir: T;
  readonly typeChecked: TypeCheckedIrArtifact<T>;
}

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

export function completeIrVerification<T>(value: T): T & CompilerPhase<"verified-ir"> {
  return completePhase(value);
}

export function completeIrTypeChecking<T>(value: T): T & CompilerPhase<"type-checked-ir"> {
  return completePhase(value);
}

export function completeWgslLegalization<T>(value: T): T & CompilerPhase<"wgsl-legalized-ir"> {
  return completePhase(value);
}
