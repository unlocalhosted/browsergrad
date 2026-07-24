import { unwrapLayoutArtifact, type VerifiedLayoutArtifact } from "../layout/artifact.js";
import type { PreparedViewAccessor } from "../layout/prepare.js";
import type { IndexExpr, PredicateExpr } from "../layout/model.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import type { ViewCopyOperation } from "./model.js";

export const INITIAL_PORTABLE_VIEW_COPY_PROFILE = "browsergrad.view-copy.positive-affine-f32@1";
export const PORTABLE_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-word32@1";
export const PORTABLE_EDGE_RANK_WORD32_VIEW_COPY_PROFILE =
  "browsergrad.view-copy.positive-affine-rank1-rank4-word32@1";

export type PortableViewCopyDType = "f32" | "i32" | "u32";

export interface PortableViewCopyProfile {
  readonly profileId:
    | typeof INITIAL_PORTABLE_VIEW_COPY_PROFILE
    | typeof PORTABLE_WORD32_VIEW_COPY_PROFILE
    | typeof PORTABLE_EDGE_RANK_WORD32_VIEW_COPY_PROFILE;
  readonly rank: 1 | 2 | 3 | 4;
  readonly dtype: PortableViewCopyDType;
}

/**
 * Shared frontend/backend legalization floor. Backends add their own integer,
 * buffer, device-limit, and schedule proofs without redefining view meaning.
 */
export function verifyInitialPortableViewCopyProfile(
  layoutArtifact: VerifiedLayoutArtifact,
  operation: ViewCopyOperation,
  source: PreparedViewAccessor,
  destination: PreparedViewAccessor,
): PortableViewCopyProfile {
  if (operation.dtype !== "f32" &&
      operation.dtype !== "i32" &&
      operation.dtype !== "u32") {
    unsupported(
      "$.operation.dtype",
      "portable view-copy supports exact 32-bit f32, i32, or u32 storage only",
    );
  }
  if (operation.dtype !== "f32" &&
      operation.source.invalidSource.kind !== "reject") {
    unsupported(
      "$.operation.source.invalidSource",
      "portable integer view-copy requires reject-on-invalid-source",
    );
  }
  const rank = source.logicalShape.length;
  if ((rank < 1 || rank > 4) ||
      destination.logicalShape.length !== rank) {
    unsupported(
      "$.operation",
      "portable view-copy requires equal source and destination ranks in [1, 4]",
    );
  }
  if (source.memorySpace.kind !== "global" || destination.memorySpace.kind !== "global") {
    unsupported("$.operation", "initial portable view-copy profile requires global-memory allocations");
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
  requirePositiveAffine(sourceMap.location, symbolMinima, "$.operation.source.indexMap.location");
  requirePortablePredicate(sourceMap.inBounds, symbolMinima, "$.operation.source.indexMap.inBounds");
  requirePositiveAffine(destinationMap.location, symbolMinima, "$.operation.destination.indexMap.location");
  requirePortablePredicate(destinationMap.inBounds, symbolMinima, "$.operation.destination.indexMap.inBounds");
  return Object.freeze({
    profileId: rank === 1 || rank === 4
      ? PORTABLE_EDGE_RANK_WORD32_VIEW_COPY_PROFILE
      : operation.dtype === "f32"
        ? INITIAL_PORTABLE_VIEW_COPY_PROFILE
        : PORTABLE_WORD32_VIEW_COPY_PROFILE,
    rank: rank as 1 | 2 | 3 | 4,
    dtype: operation.dtype,
  });
}

interface AffineProfile {
  readonly coordinateDependent: boolean;
  readonly provablyNonnegative: boolean;
}

function requirePositiveAffine(
  expression: IndexExpr,
  symbolMinima: ReadonlyMap<string, bigint>,
  path: string,
): AffineProfile {
  switch (expression.kind) {
    case "const": return { coordinateDependent: false, provablyNonnegative: BigInt(expression.value) >= 0n };
    case "dimension": return { coordinateDependent: false, provablyNonnegative: (symbolMinima.get(expression.symbolId) ?? -1n) >= 0n };
    case "coordinate": return { coordinateDependent: true, provablyNonnegative: true };
    case "add": {
      const terms = expression.terms.map((term, index) => requirePositiveAffine(term, symbolMinima, `${path}.terms[${index}]`));
      return {
        coordinateDependent: terms.some((term) => term.coordinateDependent),
        provablyNonnegative: terms.every((term) => term.provablyNonnegative),
      };
    }
    case "mul": {
      const lhs = requirePositiveAffine(expression.lhs, symbolMinima, `${path}.lhs`);
      const rhs = requirePositiveAffine(expression.rhs, symbolMinima, `${path}.rhs`);
      if (lhs.coordinateDependent && rhs.coordinateDependent) unsupported(path, "non-affine coordinate multiplication is outside the initial portable profile");
      if (lhs.coordinateDependent && !rhs.provablyNonnegative) unsupported(`${path}.rhs`, "negative or unproved coordinate stride is outside the initial portable profile");
      if (rhs.coordinateDependent && !lhs.provablyNonnegative) unsupported(`${path}.lhs`, "negative or unproved coordinate stride is outside the initial portable profile");
      return {
        coordinateDependent: lhs.coordinateDependent || rhs.coordinateDependent,
        provablyNonnegative: lhs.provablyNonnegative && rhs.provablyNonnegative,
      };
    }
    case "floorDiv":
    case "ceilDiv":
    case "mod":
    case "min":
    case "max": unsupported(path, `${expression.kind} requires a separately proved backend integer profile`);
  }
}

function requirePortablePredicate(
  expression: PredicateExpr,
  symbolMinima: ReadonlyMap<string, bigint>,
  path: string,
): void {
  switch (expression.kind) {
    case "bool": return;
    case "equal":
    case "lessEqual":
      requirePositiveAffine(expression.lhs, symbolMinima, `${path}.lhs`);
      requirePositiveAffine(expression.rhs, symbolMinima, `${path}.rhs`);
      return;
    case "and":
    case "or":
      expression.values.forEach((value, index) => requirePortablePredicate(value, symbolMinima, `${path}.values[${index}]`));
      return;
    case "not": requirePortablePredicate(expression.value, symbolMinima, `${path}.value`); return;
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
