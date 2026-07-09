export interface ReferenceVector3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export function referenceVectorFromTuple(value: readonly [number, number, number]): ReferenceVector3 {
  return { x: value[0], y: value[1], z: value[2] };
}

export function isReferenceVector3(value: unknown): value is ReferenceVector3 {
  return typeof value === "object" &&
    value !== null &&
    !("kind" in value) &&
    typeof (value as ReferenceVector3).x === "number" &&
    typeof (value as ReferenceVector3).y === "number" &&
    typeof (value as ReferenceVector3).z === "number";
}
