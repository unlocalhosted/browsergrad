import { unwrapLayoutArtifact, type VerifiedLayoutArtifact } from "../layout/artifact.js";
import type { PreparedViewAccessor } from "../layout/prepare.js";
import type { IndexExpr, PredicateExpr } from "../layout/model.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import type { ViewCopyOperation } from "./model.js";

export const PORTABLE_F32_VIEW_COPY_PROFILE = "browsergrad.view-copy.positive-affine-f32@1";
/** @deprecated Use PORTABLE_F32_VIEW_COPY_PROFILE. */
export const INITIAL_PORTABLE_VIEW_COPY_PROFILE = PORTABLE_F32_VIEW_COPY_PROFILE;
export const PORTABLE_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-word32@1";
export const PORTABLE_EDGE_RANK_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-rank1-rank4-word32@1";
export const PORTABLE_RANK5_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-rank5-word32@1";
export const PORTABLE_RANK6_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-rank6-word32@1";
export const PORTABLE_RANK7_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-rank7-word32@1";
export const PORTABLE_PACKED8_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-rank2-rank3-packed8@1";
export const PORTABLE_PACKED16_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-rank2-rank3-packed16@1";
export const PORTABLE_WORD64_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-rank2-rank3-word64@1";
export const PORTABLE_SIGNED_AFFINE_RANK1_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.signed-affine-rank1-word32@1";
export const PORTABLE_SIGNED_AFFINE_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.signed-affine-rank2-rank3-word32@1";
export const PORTABLE_SIGNED_AFFINE_HIGH_RANK_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.signed-affine-rank4-rank5-word32@1";
export const PORTABLE_SIGNED_AFFINE_RANK6_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.signed-affine-rank6-word32@1";
export const PORTABLE_SIGNED_AFFINE_RANK7_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.signed-affine-rank7-word32@1";

export type PortableViewCopyDType =
  | "bool"
  | "i8"
  | "u8"
  | "i16"
  | "u16"
  | "f16"
  | "bf16"
  | "f32"
  | "i32"
  | "u32"
  | "f64"
  | "i64"
  | "u64";

export interface PortableViewCopyProfile {
  readonly profileId:
    | typeof PORTABLE_F32_VIEW_COPY_PROFILE
    | typeof PORTABLE_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_EDGE_RANK_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_RANK5_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_RANK6_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_RANK7_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_PACKED8_VIEW_COPY_PROFILE
    | typeof PORTABLE_PACKED16_VIEW_COPY_PROFILE
    | typeof PORTABLE_WORD64_VIEW_COPY_PROFILE
    | typeof PORTABLE_SIGNED_AFFINE_RANK1_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_SIGNED_AFFINE_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_SIGNED_AFFINE_HIGH_RANK_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_SIGNED_AFFINE_RANK6_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_SIGNED_AFFINE_RANK7_WORD32_VIEW_COPY_PROFILE;
  readonly rank: 1 | 2 | 3 | 4 | 5 | 6 | 7;
  readonly dtype: PortableViewCopyDType;
}

type PortableViewCopyStorageClass =
  | "packed8"
  | "packed16"
  | "word32"
  | "word64";

/**
 * Shared frontend/backend legalization floor. Backends add their own integer,
 * buffer, device-limit, and schedule proofs without redefining view meaning.
 */
export function verifyPortableViewCopyProfile(
  layoutArtifact: VerifiedLayoutArtifact,
  operation: ViewCopyOperation,
  source: PreparedViewAccessor,
  destination: PreparedViewAccessor,
): PortableViewCopyProfile {
  const dtypeProfile = classifyPortableViewCopyDType(operation.dtype);
  const nonWord32 = dtypeProfile.storageClass !== "word32";
  if ((nonWord32 || dtypeProfile.dtype !== "f32") &&
      operation.source.invalidSource.kind !== "reject") {
    unsupported(
      "$.operation.source.invalidSource",
      "portable integer and non-word32 view-copy requires reject-on-invalid-source",
    );
  }
  const rank = source.logicalShape.length;
  if ((rank < (nonWord32 ? 2 : 1) || rank > (nonWord32 ? 3 : 7)) ||
      destination.logicalShape.length !== rank) {
    unsupported(
      "$.operation",
      nonWord32
        ? "portable non-word32 view-copy requires equal source and destination ranks in [2, 3]"
        : "portable word32 view-copy requires equal source and destination ranks in [1, 7]",
    );
  }
  if (source.memorySpace.kind !== "global" || destination.memorySpace.kind !== "global") {
    unsupported("$.operation", "portable view-copy requires global-memory allocations");
  }
  if (!source.fullySpecialized || !destination.fullySpecialized) {
    unsupported("$.bindings", "portable view-copy specialization requires bindings for every index-map dimension reference");
  }
  const layout = unwrapLayoutArtifact(layoutArtifact);
  const indexMaps = new Map(layout.indexMaps.map((entry) => [entry.indexMapId, entry]));
  const sourceMap = indexMaps.get(source.indexMapId);
  const destinationMap = indexMaps.get(destination.indexMapId);
  if (sourceMap === undefined || destinationMap === undefined) throw new Error("internal: prepared index map disappeared");
  const symbolMinima = new Map(layout.symbols.map((symbol) => [symbol.id, BigInt(symbol.domain.min)]));
  const sourceLocationProfile = analyzeAffine(
    sourceMap.location,
    symbolMinima,
    "$.operation.source.indexMap.location",
  );
  const sourcePredicateSignedCoordinateScale = analyzePortablePredicate(
    sourceMap.inBounds,
    symbolMinima,
    "$.operation.source.indexMap.inBounds",
  );
  const sourceSignedCoordinateScale =
    sourceLocationProfile.signedCoordinateScale ||
    sourcePredicateSignedCoordinateScale;
  const destinationLocationProfile = analyzeAffine(
    destinationMap.location,
    symbolMinima,
    "$.operation.destination.indexMap.location",
  );
  const destinationPredicateSignedCoordinateScale = analyzePortablePredicate(
    destinationMap.inBounds,
    symbolMinima,
    "$.operation.destination.indexMap.inBounds",
  );
  const destinationSignedCoordinateScale =
    destinationLocationProfile.signedCoordinateScale ||
    destinationPredicateSignedCoordinateScale;
  if (destinationSignedCoordinateScale) {
    unsupported(
      "$.operation.destination",
      "portable view-copy requires a positive-affine dense destination",
    );
  }
  const signedCoordinateScale = sourceSignedCoordinateScale;
  if (nonWord32 && signedCoordinateScale) {
    unsupported(
      "$.operation.source",
      "portable non-word32 view-copy requires a positive-affine source",
    );
  }
  return Object.freeze({
    profileId: selectPortableViewCopyProfile(
      dtypeProfile,
      rank as 1 | 2 | 3 | 4 | 5 | 6 | 7,
      signedCoordinateScale,
    ),
    rank: rank as 1 | 2 | 3 | 4 | 5 | 6 | 7,
    dtype: dtypeProfile.dtype,
  });
}

/** @deprecated Use verifyPortableViewCopyProfile. */
export const verifyInitialPortableViewCopyProfile = verifyPortableViewCopyProfile;

function classifyPortableViewCopyDType(
  dtype: ViewCopyOperation["dtype"],
): Readonly<{
  dtype: PortableViewCopyDType;
  storageClass: PortableViewCopyStorageClass;
}> {
  switch (dtype) {
    case "bool":
    case "i8":
    case "u8":
      return { dtype, storageClass: "packed8" };
    case "i16":
    case "u16":
    case "f16":
    case "bf16":
      return { dtype, storageClass: "packed16" };
    case "f32":
    case "i32":
    case "u32":
      return { dtype, storageClass: "word32" };
    case "f64":
    case "i64":
    case "u64":
      return { dtype, storageClass: "word64" };
    default:
      return unsupported(
        "$.operation.dtype",
        "portable view-copy supports exact 8-bit bool, i8, or u8 storage, exact 16-bit i16, u16, f16, or bf16 storage, exact 32-bit f32, i32, or u32 storage, and exact 64-bit f64, i64, or u64 storage only",
      );
  }
}

function selectPortableViewCopyProfile(
  dtypeProfile: Readonly<{
    dtype: PortableViewCopyDType;
    storageClass: PortableViewCopyStorageClass;
  }>,
  rank: 1 | 2 | 3 | 4 | 5 | 6 | 7,
  signedCoordinateScale: boolean,
): PortableViewCopyProfile["profileId"] {
  if (dtypeProfile.storageClass === "packed8") {
    return PORTABLE_PACKED8_VIEW_COPY_PROFILE;
  }
  if (dtypeProfile.storageClass === "packed16") {
    return PORTABLE_PACKED16_VIEW_COPY_PROFILE;
  }
  if (dtypeProfile.storageClass === "word64") {
    return PORTABLE_WORD64_VIEW_COPY_PROFILE;
  }
  if (signedCoordinateScale) {
    if (rank === 1) return PORTABLE_SIGNED_AFFINE_RANK1_WORD32_VIEW_COPY_PROFILE;
    if (rank === 2 || rank === 3) return PORTABLE_SIGNED_AFFINE_WORD32_VIEW_COPY_PROFILE;
    if (rank === 6) return PORTABLE_SIGNED_AFFINE_RANK6_WORD32_VIEW_COPY_PROFILE;
    if (rank === 7) return PORTABLE_SIGNED_AFFINE_RANK7_WORD32_VIEW_COPY_PROFILE;
    return PORTABLE_SIGNED_AFFINE_HIGH_RANK_WORD32_VIEW_COPY_PROFILE;
  }
  if (rank === 5) return PORTABLE_RANK5_WORD32_VIEW_COPY_PROFILE;
  if (rank === 6) return PORTABLE_RANK6_WORD32_VIEW_COPY_PROFILE;
  if (rank === 7) return PORTABLE_RANK7_WORD32_VIEW_COPY_PROFILE;
  if (rank === 1 || rank === 4) return PORTABLE_EDGE_RANK_WORD32_VIEW_COPY_PROFILE;
  return dtypeProfile.dtype === "f32"
    ? PORTABLE_F32_VIEW_COPY_PROFILE
    : PORTABLE_WORD32_VIEW_COPY_PROFILE;
}

interface AffineProfile {
  readonly coordinateDependent: boolean;
  readonly provablyNonnegative: boolean;
  readonly signedCoordinateScale: boolean;
}

function analyzeAffine(
  expression: IndexExpr,
  symbolMinima: ReadonlyMap<string, bigint>,
  path: string,
): AffineProfile {
  switch (expression.kind) {
    case "const": {
      return {
        coordinateDependent: false,
        provablyNonnegative: BigInt(expression.value) >= 0n,
        signedCoordinateScale: false,
      };
    }
    case "dimension": {
      return {
        coordinateDependent: false,
        provablyNonnegative:
          (symbolMinima.get(expression.symbolId) ?? -1n) >= 0n,
        signedCoordinateScale: false,
      };
    }
    case "coordinate": {
      return {
        coordinateDependent: true,
        provablyNonnegative: true,
        signedCoordinateScale: false,
      };
    }
    case "add": {
      const terms = expression.terms.map((term, index) =>
        analyzeAffine(term, symbolMinima, `${path}.terms[${index}]`));
      return {
        coordinateDependent: terms.some((term) => term.coordinateDependent),
        provablyNonnegative: terms.every((term) => term.provablyNonnegative),
        signedCoordinateScale: terms.some(
          (term) => term.signedCoordinateScale,
        ),
      };
    }
    case "mul": {
      const lhs = analyzeAffine(
        expression.lhs,
        symbolMinima,
        `${path}.lhs`,
      );
      const rhs = analyzeAffine(
        expression.rhs,
        symbolMinima,
        `${path}.rhs`,
      );
      if (lhs.coordinateDependent && rhs.coordinateDependent) {
        unsupported(
          path,
          "non-affine coordinate multiplication is outside the portable affine profiles",
        );
      }
      return {
        coordinateDependent: lhs.coordinateDependent || rhs.coordinateDependent,
        provablyNonnegative: lhs.provablyNonnegative && rhs.provablyNonnegative,
        signedCoordinateScale:
          lhs.signedCoordinateScale ||
          rhs.signedCoordinateScale ||
          (lhs.coordinateDependent && !rhs.provablyNonnegative) ||
          (rhs.coordinateDependent && !lhs.provablyNonnegative),
      };
    }
    case "floorDiv":
    case "ceilDiv":
    case "mod":
    case "min":
    case "max": unsupported(path, `${expression.kind} requires a separately proved backend integer profile`);
  }
}

function analyzePortablePredicate(
  expression: PredicateExpr,
  symbolMinima: ReadonlyMap<string, bigint>,
  path: string,
): boolean {
  switch (expression.kind) {
    case "bool": return false;
    case "equal":
    case "lessEqual": {
      const lhs = analyzeAffine(expression.lhs, symbolMinima, `${path}.lhs`);
      const rhs = analyzeAffine(expression.rhs, symbolMinima, `${path}.rhs`);
      return lhs.signedCoordinateScale || rhs.signedCoordinateScale;
    }
    case "and":
    case "or": {
      const values = expression.values.map((value, index) =>
        analyzePortablePredicate(
          value,
          symbolMinima,
          `${path}.values[${index}]`,
        ));
      return values.some(Boolean);
    }
    case "not": {
      return analyzePortablePredicate(
        expression.value,
        symbolMinima,
        `${path}.value`,
      );
    }
  }
}

function unsupported(path: string, message: string): never {
  throw new SemanticSchemaError({
    code: KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}
