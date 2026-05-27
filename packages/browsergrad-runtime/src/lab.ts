/**
 * Lab manifest + semver gate (PRD-013 spike).
 *
 * Two pieces:
 *   1. `parseManifest(json)` — validates a lab.json against the v0 schema.
 *      No ajv dependency; hand-written validator (8 fields, mechanical).
 *      Returns `{ ok: true, manifest }` or `{ ok: false, errors }`.
 *   2. `assertCompatibleRuntime(manifest, runtimeVersion)` — refuses to
 *      boot when the lab's `requires_browsergrad` pin doesn't satisfy
 *      the live runtime semver. Hard fail with a clear pointer.
 *
 * Per the DL/GPU review: legacy CDN + GH Action mirror + dataset
 * registry + sandbox policy are deferred. The contract surface (semver
 * pin + manifest shape) is what makes alignment real; everything else
 * waits until 5+ labs exist and pain points are observable.
 *
 * v0 manifest schema (8 required fields):
 *   id: string                         — lab identifier, kebab-case
 *   version: string                    — semver lab version
 *   requires_browsergrad: string       — semver range (e.g. "^0.7.0")
 *   required_ops: string[]             — IR opcodes the lab uses
 *   rubric_path: string                — relative path to rubric.py
 *   starter_path: string               — relative path to starter.py
 *   reference_path: string             — relative path to reference.py
 *   datasets: Array<{ name, url, hash? }>  — bounded array
 *
 * Refused fields (added at decoration time):
 *   policy, telemetry, memory_budget_mb, expected_wall_clock_ms,
 *   learning_goals, required_prds, curriculum_chapter — defer until
 *   an author asks for them.
 */

export interface LabManifest {
  readonly id: string;
  readonly version: string;
  readonly requires_browsergrad: string;
  readonly required_ops: readonly string[];
  readonly rubric_path: string;
  readonly starter_path: string;
  readonly reference_path: string;
  readonly datasets: readonly LabDataset[];
}

export interface LabDataset {
  readonly name: string;
  readonly url: string;
  readonly hash?: string;
}

export type ManifestParseResult =
  | { ok: true; manifest: LabManifest }
  | { ok: false; errors: readonly string[] };

const SEMVER_RE = /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/;
const SEMVER_RANGE_RE = /^[\^~]?\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/;
const ID_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

export function parseManifest(input: unknown): ManifestParseResult {
  const errors: string[] = [];
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return { ok: false, errors: ["manifest must be a JSON object"] };
  }
  const o = input as Record<string, unknown>;

  const requireString = (key: string, re?: RegExp): string | undefined => {
    const v = o[key];
    if (typeof v !== "string") {
      errors.push(`${key}: required string field missing or not a string`);
      return undefined;
    }
    if (re && !re.test(v)) {
      errors.push(`${key}: value ${JSON.stringify(v)} does not match ${re.source}`);
    }
    return v;
  };

  const id = requireString("id", ID_RE);
  const version = requireString("version", SEMVER_RE);
  const requires_browsergrad = requireString("requires_browsergrad", SEMVER_RANGE_RE);

  const required_ops = (() => {
    const v = o.required_ops;
    if (!Array.isArray(v)) {
      errors.push("required_ops: required string[] missing or not an array");
      return [] as string[];
    }
    if (v.length > 64) {
      errors.push(`required_ops: at most 64 entries (got ${v.length})`);
    }
    for (let i = 0; i < v.length; i++) {
      if (typeof v[i] !== "string") {
        errors.push(`required_ops[${i}]: must be a string`);
      }
    }
    return v as string[];
  })();

  const rubric_path = requireString("rubric_path");
  const starter_path = requireString("starter_path");
  const reference_path = requireString("reference_path");

  const datasets: LabDataset[] = (() => {
    const v = o.datasets;
    if (v === undefined) return [];
    if (!Array.isArray(v)) {
      errors.push("datasets: must be an array (omit for no datasets)");
      return [];
    }
    if (v.length > 32) {
      errors.push(`datasets: at most 32 entries (got ${v.length})`);
    }
    const out: LabDataset[] = [];
    for (let i = 0; i < v.length; i++) {
      const d = v[i];
      if (typeof d !== "object" || d === null) {
        errors.push(`datasets[${i}]: must be an object`);
        continue;
      }
      const dObj = d as Record<string, unknown>;
      if (typeof dObj.name !== "string") {
        errors.push(`datasets[${i}].name: required string field`);
      }
      if (typeof dObj.url !== "string") {
        errors.push(`datasets[${i}].url: required string field`);
      }
      if (dObj.hash !== undefined && typeof dObj.hash !== "string") {
        errors.push(`datasets[${i}].hash: must be a string when present`);
      }
      if (typeof dObj.name === "string" && typeof dObj.url === "string") {
        out.push({
          name: dObj.name,
          url: dObj.url,
          ...(typeof dObj.hash === "string" ? { hash: dObj.hash } : {}),
        });
      }
    }
    return out;
  })();

  if (errors.length > 0) return { ok: false, errors };

  return {
    ok: true,
    manifest: {
      id: id!,
      version: version!,
      requires_browsergrad: requires_browsergrad!,
      required_ops,
      rubric_path: rubric_path!,
      starter_path: starter_path!,
      reference_path: reference_path!,
      datasets,
    },
  };
}

/**
 * Semver compatibility check.
 *
 * Supports a small subset of npm semver — enough for the lab pin use case:
 *   `^A.B.C` — compatible with the same major, version ≥ A.B.C
 *   `~A.B.C` — compatible with the same major+minor, version ≥ A.B.C
 *   `A.B.C` — exact match (no pre-release matching)
 *
 * Pre-release tags are stripped from the comparison. If the runtime
 * version doesn't parse, we fail-open (return true) — the caller has
 * bigger problems.
 */
export function isSemverCompatible(
  range: string,
  runtimeVersion: string,
): boolean {
  const stripPre = (s: string): string => s.split("-")[0]!;
  const rVer = stripPre(runtimeVersion);
  const [rMaj, rMin, rPat] = rVer.split(".").map(Number);
  if (
    rMaj === undefined ||
    rMin === undefined ||
    rPat === undefined ||
    Number.isNaN(rMaj) ||
    Number.isNaN(rMin) ||
    Number.isNaN(rPat)
  ) {
    return true; // fail-open
  }

  const prefix = range[0] === "^" || range[0] === "~" ? range[0] : "";
  const body = prefix ? range.slice(1) : range;
  const [maj, min, pat] = stripPre(body).split(".").map(Number);
  if (
    maj === undefined ||
    min === undefined ||
    pat === undefined ||
    Number.isNaN(maj) ||
    Number.isNaN(min) ||
    Number.isNaN(pat)
  ) {
    return false;
  }

  const numericGreaterEqual = (
    a: readonly [number, number, number],
    b: readonly [number, number, number],
  ): boolean => {
    if (a[0] !== b[0]) return a[0] > b[0];
    if (a[1] !== b[1]) return a[1] > b[1];
    return a[2] >= b[2];
  };

  if (prefix === "^") {
    return rMaj === maj && numericGreaterEqual([rMaj, rMin, rPat], [maj, min, pat]);
  }
  if (prefix === "~") {
    return rMaj === maj && rMin === min && numericGreaterEqual([rMaj, rMin, rPat], [maj, min, pat]);
  }
  // Exact match.
  return rMaj === maj && rMin === min && rPat === pat;
}

/**
 * Hard-fail compatibility check. Throws LabRuntimeMismatch if the
 * manifest's pin doesn't satisfy the live runtime version.
 */
export class LabRuntimeMismatch extends Error {
  constructor(public readonly manifestPin: string, public readonly runtimeVersion: string) {
    super(
      `Lab manifest requires browsergrad ${manifestPin}, but the active runtime ` +
        `is ${runtimeVersion}. Update the lab's requires_browsergrad pin or ` +
        `pin the runtime to a compatible version. (No legacy-CDN fallback ` +
        `in v0; hard-fail is the contract.)`,
    );
    this.name = "LabRuntimeMismatch";
  }
}

export function assertCompatibleRuntime(
  manifest: LabManifest,
  runtimeVersion: string,
): void {
  if (!isSemverCompatible(manifest.requires_browsergrad, runtimeVersion)) {
    throw new LabRuntimeMismatch(manifest.requires_browsergrad, runtimeVersion);
  }
}
