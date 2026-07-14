import type { CudaLiteDiagnostic } from "./types.js";

/**
 * An externally supplied source fragment that will appear verbatim in a
 * CUDA-lite compilation unit. Keep normalization policy outside the compiler:
 * callers may transform a fragment before handing it to this API, then retain
 * the transform record and source provenance alongside the assembled unit.
 */
export interface CudaLiteSourceFragment {
  /** Source text to emit verbatim. */
  readonly source: string;
  /** Optional caller-defined category such as "device-function" or "kernel". */
  readonly kind?: string;
  /** Optional stable caller-defined fragment name. */
  readonly label?: string;
  /** Where this fragment originated before caller-owned normalization. */
  readonly provenance?: CudaLiteSourceProvenance;
}

/**
 * Caller-owned source provenance. The compiler does not interpret source
 * names or rewrite locations, so this stays useful for files, virtual cells,
 * generated snippets, and corpus records alike.
 */
export interface CudaLiteSourceProvenance {
  readonly sourceName: string;
  readonly sourceOffset?: number;
  readonly sourceLine?: number;
  readonly sourceColumn?: number;
}

/** A transform already applied by the caller before assembling fragments. */
export interface CudaLiteAppliedSourceTransform {
  readonly name: string;
  readonly detail?: string;
}

/** A one-based position in the assembled CUDA-lite source. */
export interface CudaLiteCompilationUnitPosition {
  readonly offset: number;
  readonly line: number;
  readonly column: number;
}

/**
 * A verbatim fragment range in the assembled source. Ranges are half-open:
 * `outputStart` is included and `outputEnd` is excluded.
 */
export interface CudaLiteCompilationUnitSegment {
  readonly fragmentIndex: number;
  readonly outputStart: CudaLiteCompilationUnitPosition;
  readonly outputEnd: CudaLiteCompilationUnitPosition;
  readonly kind?: string;
  readonly label?: string;
  readonly provenance?: CudaLiteSourceProvenance;
}

/** A position in the source that supplied a prepared compilation-unit fragment. */
export interface CudaLiteSourceProvenancePosition {
  readonly sourceName: string;
  readonly offset?: number;
  readonly line?: number;
  readonly column?: number;
}

/**
 * The portion of one compiler diagnostic attributable to a caller-provided
 * fragment. A diagnostic can map to several fragments when its span crosses a
 * compilation-unit boundary. Separators intentionally produce no mapping.
 */
export interface CudaLiteProvenanceDiagnosticSegment {
  readonly fragmentIndex: number;
  readonly outputStart: CudaLiteCompilationUnitPosition;
  readonly outputEnd: CudaLiteCompilationUnitPosition;
  readonly kind?: string;
  readonly label?: string;
  readonly sourceStart?: CudaLiteSourceProvenancePosition;
  readonly sourceEnd?: CudaLiteSourceProvenancePosition;
}

export interface PrepareCudaLiteCompilationUnitOptions {
  /** Fragments in their exact desired output order. */
  readonly fragments: readonly CudaLiteSourceFragment[];
  /** Text inserted between every adjacent fragment. Defaults to a line feed. */
  readonly separator?: string;
  /** Caller-owned record of transforms applied before assembly. */
  readonly appliedTransforms?: readonly CudaLiteAppliedSourceTransform[];
}

/**
 * A deterministic, browser-safe CUDA-lite source assembly result.
 *
 * `source` is ready to pass to `compileCudaLiteKernel()`. This utility does
 * not parse, normalize, or otherwise transform source; `segments` and
 * `appliedTransforms` make externally normalized input inspectable instead.
 */
export interface PreparedCudaLiteCompilationUnit {
  readonly source: string;
  readonly segments: readonly CudaLiteCompilationUnitSegment[];
  readonly appliedTransforms: readonly CudaLiteAppliedSourceTransform[];
}

/**
 * Assemble explicit CUDA-lite source/context fragments without importing any
 * corpus-normalization policy into the browser compiler package.
 */
export function prepareCudaLiteCompilationUnit(
  options: PrepareCudaLiteCompilationUnitOptions,
): PreparedCudaLiteCompilationUnit {
  const separator = options.separator ?? "\n";
  if (typeof separator !== "string") throw new TypeError("CUDA-lite compilation-unit separator must be a string");

  const sourceParts: string[] = [];
  const segments: CudaLiteCompilationUnitSegment[] = [];
  let position: CudaLiteCompilationUnitPosition = { offset: 0, line: 1, column: 1 };

  for (const [fragmentIndex, fragment] of options.fragments.entries()) {
    if (typeof fragment.source !== "string") {
      throw new TypeError(`CUDA-lite source fragment ${fragmentIndex} must contain a string source`);
    }
    if (fragmentIndex > 0) {
      sourceParts.push(separator);
      position = advancePosition(position, separator);
    }

    const outputStart = position;
    sourceParts.push(fragment.source);
    position = advancePosition(position, fragment.source);
    const outputEnd = position;
    segments.push({
      fragmentIndex,
      outputStart,
      outputEnd,
      ...(fragment.kind === undefined ? {} : { kind: fragment.kind }),
      ...(fragment.label === undefined ? {} : { label: fragment.label }),
      ...(fragment.provenance === undefined ? {} : { provenance: fragment.provenance }),
    });
  }

  return {
    source: sourceParts.join(""),
    segments,
    appliedTransforms: options.appliedTransforms?.map((transform) => ({
      name: transform.name,
      ...(transform.detail === undefined ? {} : { detail: transform.detail }),
    })) ?? [],
  };
}

/**
 * Relate a compiler diagnostic from assembled source back to the verbatim
 * caller fragments that produced it. This deliberately maps only fragment
 * provenance: caller-owned transforms remain responsible for their own
 * source-map policy.
 */
export function mapCudaLiteDiagnosticToSourceProvenance(
  unit: PreparedCudaLiteCompilationUnit,
  diagnostic: CudaLiteDiagnostic,
): readonly CudaLiteProvenanceDiagnosticSegment[] {
  const { start, end } = diagnostic.span;
  if (end < start) return [];

  return unit.segments.flatMap((segment) => {
    const segmentStart = segment.outputStart.offset;
    const segmentEnd = segment.outputEnd.offset;
    const overlapStart = Math.max(start, segmentStart);
    const overlapEnd = Math.min(end, segmentEnd);
    const pointInsideSegment = start === end && start >= segmentStart && start < segmentEnd;
    if (overlapStart > overlapEnd || overlapStart === overlapEnd && !pointInsideSegment) return [];

    const outputStart = positionAtOffset(unit.source, segment, overlapStart);
    const outputEnd = positionAtOffset(unit.source, segment, overlapEnd);
    return [{
      fragmentIndex: segment.fragmentIndex,
      outputStart,
      outputEnd,
      ...(segment.kind === undefined ? {} : { kind: segment.kind }),
      ...(segment.label === undefined ? {} : { label: segment.label }),
      ...(segment.provenance === undefined ? {} : {
        sourceStart: provenancePositionAtOffset(unit.source, segment, overlapStart),
        sourceEnd: provenancePositionAtOffset(unit.source, segment, overlapEnd),
      }),
    }];
  });
}

function positionAtOffset(
  source: string,
  segment: CudaLiteCompilationUnitSegment,
  offset: number,
): CudaLiteCompilationUnitPosition {
  return advancePosition(
    segment.outputStart,
    source.slice(segment.outputStart.offset, offset),
  );
}

function provenancePositionAtOffset(
  source: string,
  segment: CudaLiteCompilationUnitSegment,
  offset: number,
): CudaLiteSourceProvenancePosition {
  const provenance = segment.provenance;
  if (provenance === undefined) throw new Error("expected source provenance for prepared source segment");
  const relative = advancePosition(
    { offset: 0, line: 1, column: 1 },
    source.slice(segment.outputStart.offset, offset),
  );
  return {
    sourceName: provenance.sourceName,
    ...(provenance.sourceOffset === undefined ? {} : { offset: provenance.sourceOffset + relative.offset }),
    ...(provenance.sourceLine === undefined ? {} : { line: provenance.sourceLine + relative.line - 1 }),
    ...(provenance.sourceColumn === undefined ? {} : {
      column: relative.line === 1
        ? provenance.sourceColumn + relative.column - 1
        : relative.column,
    }),
  };
}

function advancePosition(
  position: CudaLiteCompilationUnitPosition,
  source: string,
): CudaLiteCompilationUnitPosition {
  let line = position.line;
  let column = position.column;
  for (let index = 0; index < source.length; index++) {
    const character = source[index];
    if (character === "\r") {
      if (source[index + 1] === "\n") index++;
      line++;
      column = 1;
    } else if (character === "\n") {
      line++;
      column = 1;
    } else {
      column++;
    }
  }
  return { offset: position.offset + source.length, line, column };
}
