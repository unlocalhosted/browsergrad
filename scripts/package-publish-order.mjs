export const WORKSPACE_RUNTIME_DEPENDENCY_FIELDS = Object.freeze([
  "dependencies",
  "optionalDependencies",
  "peerDependencies",
]);

/**
 * Return public workspace packages in deterministic dependency-first order.
 *
 * A workspace-owned package name establishes an edge regardless of range.
 * Packed manifests contain registry ranges while source manifests commonly use
 * workspace ranges; release order must remain identical for both forms.
 * Development-only dependencies intentionally do not affect publication.
 *
 * @param {ReadonlyArray<{ dir: string, manifest: Record<string, unknown> }>} packages
 * @param {{ workspacePackageNames?: Iterable<string> }} [options]
 */
export function sortWorkspacePackages(packages, options = {}) {
  if (!Array.isArray(packages)) {
    throw new TypeError("Workspace packages must be an array");
  }

  const byName = new Map();
  for (const entry of packages) {
    if (entry === null || typeof entry !== "object") {
      throw new TypeError("Workspace package entries must be objects");
    }
    const { dir, manifest } = entry;
    if (typeof dir !== "string" || dir.length === 0) {
      throw new TypeError("Workspace package directory must be a non-empty string");
    }
    if (manifest === null || typeof manifest !== "object" || Array.isArray(manifest)) {
      throw new TypeError(`Workspace package at ${dir} must have an object manifest`);
    }
    const name = manifest.name;
    if (typeof name !== "string" || name.length === 0) {
      throw new TypeError(`Workspace package at ${dir} must have a non-empty name`);
    }
    if (byName.has(name)) {
      throw new Error(`Duplicate workspace package name: ${name}`);
    }
    byName.set(name, entry);
  }
  const workspacePackageNames = options.workspacePackageNames === undefined
    ? new Set(byName.keys())
    : new Set(options.workspacePackageNames);
  for (const name of workspacePackageNames) {
    if (typeof name !== "string" || name.length === 0) {
      throw new TypeError("Known workspace package names must be non-empty strings");
    }
  }

  const ordered = [];
  const state = new Map();
  const stack = [];

  for (const name of [...byName.keys()].sort()) {
    visit(name);
  }
  return ordered;

  function visit(name) {
    const currentState = state.get(name);
    if (currentState === "visited") {
      return;
    }
    if (currentState === "visiting") {
      const cycleStart = stack.indexOf(name);
      const cycle = [...stack.slice(cycleStart), name];
      throw new Error(`Circular workspace dependency: ${cycle.join(" -> ")}`);
    }

    state.set(name, "visiting");
    stack.push(name);
    const entry = byName.get(name);
    const dependencyNames = new Set();
    for (const field of WORKSPACE_RUNTIME_DEPENDENCY_FIELDS) {
      const dependencies = entry.manifest[field];
      if (dependencies === undefined) {
        continue;
      }
      if (dependencies === null || typeof dependencies !== "object" || Array.isArray(dependencies)) {
        throw new TypeError(`${name} ${field} must be an object`);
      }
      for (const [dependencyName, dependencyRange] of Object.entries(dependencies)) {
        if (typeof dependencyRange !== "string") {
          throw new TypeError(`${name} ${field}.${dependencyName} must be a string`);
        }
        if (byName.has(dependencyName)) {
          dependencyNames.add(dependencyName);
        } else if (workspacePackageNames.has(dependencyName)) {
          throw new Error(
            `${name} ${field}.${dependencyName} targets a non-public workspace package`,
          );
        } else if (dependencyRange.startsWith("workspace:")) {
          throw new Error(
            `${name} ${field}.${dependencyName} uses ${dependencyRange}, but target is not a public workspace package`,
          );
        }
      }
    }
    for (const dependencyName of [...dependencyNames].sort()) {
      visit(dependencyName);
    }
    stack.pop();
    state.set(name, "visited");
    ordered.push(entry);
  }
}

/**
 * Classify staged-release states before any registry mutation.
 *
 * A tag release may resume its selected target only with the exact current
 * workflow/ref/commit identity. A batch release cannot make that claim for
 * versions published by earlier runs, so it accepts those only after approved
 * workflow provenance and protected-main reachability are proved. Missing
 * targets are always published and then checked against the current identity.
 */
export function classifyStagedPublication(
  orderedPackages,
  targetNames,
  closureNames,
  statusByName,
  selectedPackageName,
) {
  if (!Array.isArray(orderedPackages)) {
    throw new TypeError("Ordered packages must be an array");
  }
  const targets = new Set(targetNames);
  const closure = new Set(closureNames);
  if (selectedPackageName !== null && !targets.has(selectedPackageName)) {
    throw new Error(`Selected package is not a publication target: ${selectedPackageName}`);
  }

  const approvedExisting = [];
  const strictExisting = [];
  const publish = [];
  for (const entry of orderedPackages) {
    const name = entry?.manifest?.name;
    if (!closure.has(name)) continue;
    const status = statusByName.get(name);
    if (status === "published") {
      if (selectedPackageName !== null && targets.has(name)) {
        strictExisting.push(name);
      } else {
        approvedExisting.push(name);
      }
      continue;
    }
    if (status !== "missing") {
      throw new Error(`Unknown registry status for ${String(name)}: ${String(status)}`);
    }
    if (!targets.has(name)) {
      throw new Error(`Unpublished dependency-closure package is not a target: ${name}`);
    }
    publish.push(name);
  }

  return Object.freeze({
    approvedExisting: Object.freeze(approvedExisting),
    strictExisting: Object.freeze(strictExisting),
    publish: Object.freeze(publish),
  });
}
