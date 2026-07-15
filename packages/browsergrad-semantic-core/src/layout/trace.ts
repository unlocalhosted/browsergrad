import { deepFreezeJson, type JsonValue } from "../schema/json.js";
import { encodeWireI64, encodeWireU64, parseWireI64, wireIntegerToBigInt, type WireI64, type WireU64 } from "../schema/integers.js";
import type { DecodeLimits } from "../schema/limits.js";
import { unwrapLayoutArtifact, type LayoutArtifactPayloadV1, type VerifiedLayoutArtifact } from "./artifact.js";
import { prepareViewAccessor } from "./prepare.js";

export interface LayoutCoordinateRequest {
  readonly viewId: string;
  readonly coordinates: readonly WireI64[];
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly limits?: Partial<DecodeLimits>;
}

export interface LayoutCoordinateTrace {
  readonly viewId: string;
  readonly allocationId: string;
  readonly aliasSetId: string;
  readonly logicalCoordinates: readonly WireI64[];
  readonly logicalShape: readonly WireU64[];
  readonly indexMapId: string;
  readonly mapLocation: { readonly unit: "element" | "byte"; readonly value: WireI64 | WireU64 };
  readonly viewByteOffset: WireU64;
  readonly rootByteStart: WireI64 | WireU64;
  readonly rootByteEndExclusive: WireI64 | WireU64;
  readonly allocationByteLength: WireU64;
  readonly logicalInBounds: boolean;
  readonly predicateInBounds: boolean;
  readonly allocationInBounds: boolean;
  readonly accessInBounds: boolean;
}

export interface LayoutAliasTraceRequest {
  readonly left: Omit<LayoutCoordinateRequest, "bindings" | "limits">;
  readonly right: Omit<LayoutCoordinateRequest, "bindings" | "limits">;
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly limits?: Partial<DecodeLimits>;
}

export interface LayoutAliasTrace {
  readonly left: LayoutCoordinateTrace;
  readonly right: LayoutCoordinateTrace;
  readonly sameAllocation: boolean;
  readonly sameAliasSet: boolean;
  readonly byteRangesOverlap: boolean | null;
  readonly relation: "same-allocation" | "may-alias" | "disjoint";
}

export function layoutArtifactPayload(artifact: VerifiedLayoutArtifact): LayoutArtifactPayloadV1 {
  return unwrapLayoutArtifact(artifact);
}

export function traceViewCoordinate(
  artifact: VerifiedLayoutArtifact,
  request: LayoutCoordinateRequest,
): LayoutCoordinateTrace {
  const coordinates = request.coordinates.map((value, axis) => (
    wireIntegerToBigInt(parseWireI64(value, `$.coordinates[${axis}]`))
  ));
  const accessor = prepareViewAccessor(artifact, {
    viewId: request.viewId,
    ...(request.bindings === undefined ? {} : { bindings: request.bindings }),
    ...(request.limits === undefined ? {} : { limits: request.limits }),
  });
  const access = accessor.access(coordinates);
  return deepFreezeJson({
    viewId: accessor.viewId,
    allocationId: accessor.allocationId,
    aliasSetId: accessor.aliasSetId,
    logicalCoordinates: request.coordinates.map((value, axis) => parseWireI64(value, `$.coordinates[${axis}]`)),
    logicalShape: accessor.logicalShape.map((value) => encodeWireU64(value)),
    indexMapId: accessor.indexMapId,
    mapLocation: Object.freeze({ unit: accessor.locationUnit, value: encodeWireAddress(access.mapLocation) }),
    viewByteOffset: encodeWireU64(accessor.viewByteOffset),
    rootByteStart: encodeWireAddress(access.rootByteStart),
    rootByteEndExclusive: encodeWireAddress(access.rootByteEndExclusive),
    allocationByteLength: encodeWireU64(accessor.allocationByteLength),
    logicalInBounds: access.logicalInBounds,
    predicateInBounds: access.predicateInBounds,
    allocationInBounds: access.allocationInBounds,
    accessInBounds: access.accessInBounds,
  } as unknown as JsonValue) as unknown as LayoutCoordinateTrace;
}

export function traceViewAlias(
  artifact: VerifiedLayoutArtifact,
  request: LayoutAliasTraceRequest,
): LayoutAliasTrace {
  const shared = {
    ...(request.bindings === undefined ? {} : { bindings: request.bindings }),
    ...(request.limits === undefined ? {} : { limits: request.limits }),
  };
  const left = traceViewCoordinate(artifact, { ...request.left, ...shared });
  const right = traceViewCoordinate(artifact, { ...request.right, ...shared });
  const sameAllocation = left.allocationId === right.allocationId;
  const sameAliasSet = left.aliasSetId === right.aliasSetId;
  const byteRangesOverlap = sameAllocation && left.accessInBounds && right.accessInBounds
    ? BigInt(left.rootByteStart) < BigInt(right.rootByteEndExclusive) && BigInt(right.rootByteStart) < BigInt(left.rootByteEndExclusive)
    : null;
  return deepFreezeJson({
    left,
    right,
    sameAllocation,
    sameAliasSet,
    byteRangesOverlap,
    relation: sameAllocation ? "same-allocation" : sameAliasSet ? "may-alias" : "disjoint",
  } as unknown as JsonValue) as unknown as LayoutAliasTrace;
}

function encodeWireAddress(value: bigint): WireI64 | WireU64 {
  return value < 0n ? encodeWireI64(value) : encodeWireU64(value);
}
