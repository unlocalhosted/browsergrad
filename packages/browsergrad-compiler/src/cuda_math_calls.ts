export function isCudaSincosCallName(name: string | undefined): boolean {
  return name === "sincos" || name === "sincosf" || name === "__sincosf" || name === "sincospi" || name === "sincospif";
}

export function isCudaSincosPiCallName(name: string | undefined): boolean {
  return name === "sincospi" || name === "sincospif";
}

export function isCudaNanPayloadCallName(name: string | undefined): boolean {
  return name === "nan" || name === "nanf" || name === "__builtin_nan" || name === "__builtin_nanf";
}

export function isCudaFrexpCallName(name: string | undefined): boolean {
  return name === "frexp" || name === "frexpf";
}

export function isCudaModfCallName(name: string | undefined): boolean {
  return name === "modf" || name === "modff";
}

export function isCudaRemquoCallName(name: string | undefined): boolean {
  return name === "remquo" || name === "remquof";
}

export function isCudaFloatDecomposeCallName(name: string | undefined): boolean {
  return isCudaFrexpCallName(name) || isCudaModfCallName(name) || isCudaRemquoCallName(name);
}
