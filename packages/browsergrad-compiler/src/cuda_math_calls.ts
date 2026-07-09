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

export interface CudaVibMinMaxInfo {
  readonly choose: "max" | "min";
  readonly signed: boolean;
  readonly packed: boolean;
  readonly valueType: "int" | "uint";
}

export function cudaVibMinMaxInfo(name: string | undefined): CudaVibMinMaxInfo | undefined {
  switch (name) {
    case "__vibmax_s32":
      return { choose: "max", signed: true, packed: false, valueType: "int" };
    case "__vibmin_s32":
      return { choose: "min", signed: true, packed: false, valueType: "int" };
    case "__vibmax_u32":
      return { choose: "max", signed: false, packed: false, valueType: "uint" };
    case "__vibmin_u32":
      return { choose: "min", signed: false, packed: false, valueType: "uint" };
    case "__vibmax_s16x2":
      return { choose: "max", signed: true, packed: true, valueType: "uint" };
    case "__vibmin_s16x2":
      return { choose: "min", signed: true, packed: true, valueType: "uint" };
    case "__vibmax_u16x2":
      return { choose: "max", signed: false, packed: true, valueType: "uint" };
    case "__vibmin_u16x2":
      return { choose: "min", signed: false, packed: true, valueType: "uint" };
    default:
      return undefined;
  }
}

export function isCudaVibMinMaxCallName(name: string | undefined): boolean {
  return cudaVibMinMaxInfo(name) !== undefined;
}
