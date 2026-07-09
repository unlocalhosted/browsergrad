export function isCudaSincosCallName(name: string | undefined): boolean {
  return name === "sincos" || name === "sincosf" || name === "__sincosf" || name === "sincospi" || name === "sincospif";
}

export function isCudaSincosPiCallName(name: string | undefined): boolean {
  return name === "sincospi" || name === "sincospif";
}

export function isCudaNanPayloadCallName(name: string | undefined): boolean {
  return name === "nan" || name === "nanf" || name === "__builtin_nan" || name === "__builtin_nanf";
}
