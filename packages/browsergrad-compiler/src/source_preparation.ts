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
