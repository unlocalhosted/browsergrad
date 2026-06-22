import { BrowsergradError, type SessionFS } from "./types.js";
import { datasetFileName, joinAssignmentPath } from "./assignment-profile.js";
import type {
  AssignmentDatasetCacheEntry,
  AssignmentDatasetCachePlan,
  AssignmentDatasetCacheStrategy,
  AssignmentDatasetMount,
  AssignmentMaterializeResult,
  AssignmentMountContent,
  AssignmentMountContentEvaluation,
  AssignmentMountContents,
  AssignmentMountHashCheck,
  AssignmentMountHashVerification,
  AssignmentMountPlan,
  AssignmentMountPreflightReport,
  AssignmentRunPlan,
} from "./assignment-types.js";

export function createAssignmentMountPlan(
  plan: AssignmentRunPlan,
): AssignmentMountPlan {
  return {
    root: plan.files.root,
    files: [
      {
        role: "rubric",
        path: plan.files.rubricPath,
        required: true,
      },
      ...(plan.files.starterPath
        ? [{
            role: "starter" as const,
            path: plan.files.starterPath,
            required: false,
          }]
        : []),
      ...(plan.files.referencePath
        ? [{
            role: "reference" as const,
            path: plan.files.referencePath,
            required: false,
          }]
        : []),
    ],
    datasets: plan.datasets.map((dataset) => ({
      name: dataset.name,
      url: dataset.url,
      ...(dataset.hash ? { hash: dataset.hash } : {}),
      mountPath: joinAssignmentPath(
        plan.files.fixturesPath ?? joinAssignmentPath(plan.files.root, "fixtures"),
        `datasets/${datasetFileName(dataset)}`,
      ),
    })),
  };
}

export function createAssignmentDatasetCachePlan(
  plan: AssignmentMountPlan,
): AssignmentDatasetCachePlan {
  return {
    root: plan.root,
    datasets: plan.datasets.map((dataset) => datasetCacheEntry(dataset)),
  };
}

export async function materializeAssignmentMountPlan(
  fs: SessionFS,
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): Promise<AssignmentMaterializeResult> {
  const entries = collectAssignmentMountEntries(plan, contents);

  const writtenPaths: string[] = [];
  for (const entry of entries.writes) {
    await fs.write(entry.path, entry.content);
    writtenPaths.push(entry.path);
  }

  return { writtenPaths, skippedOptionalPaths: entries.skippedOptionalPaths };
}

function datasetCacheEntry(
  dataset: AssignmentDatasetMount,
): AssignmentDatasetCacheEntry {
  const hash = dataset.hash;
  const parsed = hash
    ? parseAssignmentContentHash(hash)
    : undefined;
  if (hash && parsed?.status === "ok") {
    const digest = parsed.expected.slice("sha256:".length);
    return {
      name: dataset.name,
      url: dataset.url,
      hash,
      mountPath: dataset.mountPath,
      strategy: "content-addressed",
      cacheKey: parsed.expected,
      cachePath: `datasets/sha256/${digest}`,
    };
  }

  const strategy: AssignmentDatasetCacheStrategy = parsed?.status === "invalid"
    ? "invalid-hash"
    : parsed?.status === "unsupported"
      ? "unsupported-hash"
      : "source-addressed";

  return {
    name: dataset.name,
    url: dataset.url,
    ...(hash ? { hash } : {}),
    mountPath: dataset.mountPath,
    strategy,
    cacheKey: `url:${dataset.url}`,
    cachePath: `datasets/url/${encodeURIComponent(dataset.url)}`,
  };
}

export function evaluateAssignmentMountContents(
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): AssignmentMountContentEvaluation {
  const evaluation = inspectAssignmentMountContents(plan, contents);
  return {
    ok: evaluation.missingRequiredFiles.length === 0 &&
      evaluation.missingDatasets.length === 0,
    writablePaths: evaluation.writes.map((entry) => entry.path),
    missingRequiredFiles: evaluation.missingRequiredFiles,
    missingDatasets: evaluation.missingDatasets,
    skippedOptionalPaths: evaluation.skippedOptionalPaths,
  };
}

export async function verifyAssignmentMountContentHashes(
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): Promise<AssignmentMountHashVerification> {
  const checks: AssignmentMountHashCheck[] = [];
  for (const dataset of plan.datasets) {
    if (!dataset.hash) continue;
    checks.push(await verifyDatasetMountHash(dataset, contents));
  }
  return {
    ok: checks.every((check) => check.ok),
    checks,
  };
}

export async function createAssignmentMountPreflightReport(
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): Promise<AssignmentMountPreflightReport> {
  const content = evaluateAssignmentMountContents(plan, contents);
  const hashes = await verifyAssignmentMountContentHashes(plan, contents);
  return {
    ok: content.ok && hashes.ok,
    content,
    hashes,
  };
}

function parseAssignmentContentHash(hash: string):
  | { readonly status: "ok"; readonly algorithm: "sha256"; readonly expected: string }
  | { readonly status: "invalid" | "unsupported"; readonly algorithm: string } {
  const [algorithm, digest, ...extra] = hash.split(":");
  if (!algorithm || digest === undefined || extra.length > 0) {
    return { status: "invalid", algorithm: algorithm ?? "" };
  }
  const normalizedAlgorithm = algorithm.toLowerCase();
  if (normalizedAlgorithm !== "sha256") {
    return { status: "unsupported", algorithm: normalizedAlgorithm };
  }
  if (!/^[0-9a-f]{64}$/i.test(digest)) {
    return { status: "invalid", algorithm: normalizedAlgorithm };
  }
  return {
    status: "ok",
    algorithm: "sha256",
    expected: `sha256:${digest.toLowerCase()}`,
  };
}

async function verifyDatasetMountHash(
  dataset: AssignmentDatasetMount,
  contents: AssignmentMountContents,
): Promise<AssignmentMountHashCheck> {
  const parsed = parseAssignmentContentHash(dataset.hash!);
  if (parsed.status !== "ok") {
    return {
      name: dataset.name,
      path: dataset.mountPath,
      algorithm: parsed.algorithm,
      expected: dataset.hash!,
      ok: false,
      status: parsed.status,
    };
  }

  const content = contents.datasets?.[dataset.name];
  if (content === undefined) {
    return {
      name: dataset.name,
      path: dataset.mountPath,
      algorithm: parsed.algorithm,
      expected: parsed.expected,
      ok: false,
      status: "missing",
    };
  }

  const actual = `sha256:${await sha256Hex(content)}`;
  const ok = actual === parsed.expected;
  return {
    name: dataset.name,
    path: dataset.mountPath,
    algorithm: parsed.algorithm,
    expected: parsed.expected,
    actual,
    ok,
    status: ok ? "match" : "mismatch",
  };
}

async function sha256Hex(content: AssignmentMountContent): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new BrowsergradError("Web Crypto SHA-256 is not available");
  }
  const bytes = typeof content === "string"
    ? new TextEncoder().encode(content)
    : content;
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await subtle.digest("SHA-256", copy.buffer);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function collectAssignmentMountEntries(
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): {
  readonly writes: readonly { readonly path: string; readonly content: AssignmentMountContent }[];
  readonly skippedOptionalPaths: readonly string[];
} {
  const evaluation = inspectAssignmentMountContents(plan, contents);
  if (evaluation.missingRequiredFiles.length > 0) {
    throw new BrowsergradError(
      `missing required assignment file content: ${evaluation.missingRequiredFiles[0]}`,
    );
  }
  if (evaluation.missingDatasets.length > 0) {
    throw new BrowsergradError(
      `missing assignment dataset content: ${evaluation.missingDatasets[0]}`,
    );
  }
  return {
    writes: evaluation.writes,
    skippedOptionalPaths: evaluation.skippedOptionalPaths,
  };
}

function inspectAssignmentMountContents(
  plan: AssignmentMountPlan,
  contents: AssignmentMountContents,
): {
  readonly writes: readonly { readonly path: string; readonly content: AssignmentMountContent }[];
  readonly missingRequiredFiles: readonly string[];
  readonly missingDatasets: readonly string[];
  readonly skippedOptionalPaths: readonly string[];
} {
  const skippedOptionalPaths: string[] = [];
  const missingRequiredFiles: string[] = [];
  const missingDatasets: string[] = [];
  const fileWrites: Array<{ path: string; content: AssignmentMountContent }> = [];
  const datasetWrites: Array<{ path: string; content: AssignmentMountContent }> = [];

  for (const file of plan.files) {
    const content = contents.files[file.path];
    if (content === undefined) {
      if (file.required) {
        missingRequiredFiles.push(file.path);
        continue;
      }
      skippedOptionalPaths.push(file.path);
      continue;
    }
    fileWrites.push({ path: file.path, content });
  }

  for (const dataset of plan.datasets) {
    const content = contents.datasets?.[dataset.name];
    if (content === undefined) {
      missingDatasets.push(dataset.name);
      continue;
    }
    datasetWrites.push({ path: dataset.mountPath, content });
  }

  return {
    writes: [...fileWrites, ...datasetWrites],
    missingRequiredFiles,
    missingDatasets,
    skippedOptionalPaths,
  };
}
