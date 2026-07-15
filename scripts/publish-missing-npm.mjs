import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  closeSync,
  constants,
  existsSync,
  fstatSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  writeSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import {
  classifyStagedPublication,
  sortWorkspacePackages,
  WORKSPACE_RUNTIME_DEPENDENCY_FIELDS,
} from "./package-publish-order.mjs";
import { createReadOnlyNpmEnvironment } from "./npm-read-only-environment.mjs";
import {
  inspectPackedPackageTarball,
  verifyPublishedPackageEquivalence,
  verifyPublishedTarballEquivalence,
} from "./verify-published-package-equivalence.mjs";
import {
  verifyPublishedNpmDependencyProvenance,
  verifyPublishedNpmProvenance,
} from "./verify-npm-provenance.mjs";

const REGISTRY = "https://registry.npmjs.org/";
const STAGED_MANIFEST = "browsergrad-release-manifest.json";
const STAGED_SCHEMA = "browsergrad.staged-npm-release@1";
const MAX_STAGED_MANIFEST_BYTES = 1024 * 1024;
const MAX_STAGED_TARBALL_BYTES = 64 * 1024 * 1024;
const COPY_BUFFER_BYTES = 1024 * 1024;
const APPROVED_RELEASE_WORKFLOWS = Object.freeze([
  ".github/workflows/release.yml",
  ".github/workflows/publish-npm.yml",
]);
const root = fileURLToPath(new URL("..", import.meta.url));
const packagesRoot = join(root, "packages");
const options = parseArguments(process.argv.slice(2));

const workspacePackages = readdirSync(packagesRoot, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => join(packagesRoot, entry.name))
  .filter((dir) => existsSync(join(dir, "package.json")))
  .map((dir) => ({ dir, manifest: readPackageJson(dir) }));
const publicPackages = workspacePackages
  .filter(({ manifest }) => manifest.private !== true && manifest.name?.startsWith("@unlocalhosted/"));
const orderedPackages = sortWorkspacePackages(publicPackages, {
  workspacePackageNames: workspacePackages.map(({ manifest }) => manifest.name),
});
const byName = new Map(orderedPackages.map((entry) => [entry.manifest.name, entry]));

let publicationTargets;
let stagedRelease;
if (options.publishStaged !== undefined) {
  stagedRelease = readStagedRelease(options.publishStaged, byName, options.selectedPackageName);
  publicationTargets = stagedRelease.targetSpecs.map((spec) => byName.get(splitExactSpec(spec).name));
} else {
  const selectedPackages = orderedPackages.filter(
    ({ manifest }) => options.selectedPackageName === undefined
      || manifest.name === options.selectedPackageName,
  );
  if (selectedPackages.length === 0) {
    throw new Error(options.selectedPackageName === undefined
      ? "No public @unlocalhosted packages found."
      : `Public workspace package not found: ${options.selectedPackageName}`);
  }
  publicationTargets = selectedPackages;
}

const registryStatus = new Map();
for (const entry of orderedPackages) {
  registryStatus.set(entry.manifest.name, npmVersionStatus(entry));
}
const requiredNames = workspaceDependencyClosure(publicationTargets, orderedPackages);
if (stagedRelease !== undefined) {
  validateStagedReleasePlan(
    stagedRelease,
    publicationTargets,
    requiredNames,
    orderedPackages,
    options.selectedPackageName,
  );
}

if (options.selectedPackageName !== undefined) {
  for (const dependencyName of requiredNames) {
    if (
      dependencyName !== options.selectedPackageName
      && registryStatus.get(dependencyName) !== "published"
    ) {
      throw new Error(
        `${options.selectedPackageName} requires unpublished workspace dependency ${dependencyName}`,
      );
    }
  }
}

if (stagedRelease === undefined) {
  for (const entry of orderedPackages) {
    const { dir, manifest } = entry;
    if (!requiredNames.has(manifest.name) || registryStatus.get(manifest.name) !== "published") {
      continue;
    }
    const equivalence = verifyPublishedPackageEquivalence(dir);
    console.log(`verified ${manifest.name}@${manifest.version}: registry-equivalent tree ${equivalence.treeHash}`);
  }
}

if (options.preflight) {
  console.log(`preflight ok: ${publicationTargets.length} target(s), ${requiredNames.size} package(s) in dependency closure`);
} else if (options.dryRun) {
  for (const { manifest } of publicationTargets) {
    const action = registryStatus.get(manifest.name) === "missing" ? "publish" : "resume";
    console.log(`would ${action} ${exactSpec(manifest)}`);
  }
} else if (options.stageDir !== undefined) {
  stageRelease(options.stageDir, publicationTargets, requiredNames, orderedPackages, options.selectedPackageName);
} else if (stagedRelease !== undefined) {
  try {
    await publishStagedRelease(stagedRelease, publicationTargets, requiredNames, registryStatus, options.provenance);
  } finally {
    stagedRelease.cleanup();
  }
}

function stageRelease(directory, targets, closure, allPackages, selectedPackageName) {
  assertTrackedWorkspaceClean();
  const targetNames = new Set(targets.map(({ manifest }) => manifest.name));
  const requestedOutputDirectory = resolve(root, directory);
  mkdirSync(requestedOutputDirectory, { recursive: true });
  const outputDirectory = realpathSync(requestedOutputDirectory);
  if (readdirSync(outputDirectory).length !== 0) {
    throw new Error(`Staging directory must be empty: ${outputDirectory}`);
  }
  const artifacts = [];
  const baselineIdentityByName = new Map();
  for (const { dir, manifest } of allPackages) {
    if (!closure.has(manifest.name) || !targetNames.has(manifest.name)) {
      continue;
    }
    assertFrozenArtifactLifecycle(manifest);
    const baselineDirectory = mkdtempSync(join(tmpdir(), "browsergrad-prepublish-baseline-"));
    try {
      const baselineTarball = packPackage(dir, manifest, baselineDirectory);
      baselineIdentityByName.set(
        manifest.name,
        inspectPackedPackageTarball(baselineTarball),
      );
    } finally {
      rmSync(baselineDirectory, { recursive: true, force: true });
    }
  }
  for (const { dir, manifest } of allPackages) {
    if (!closure.has(manifest.name)) {
      continue;
    }
    assertFrozenArtifactLifecycle(manifest);
    const baselineIdentity = baselineIdentityByName.get(manifest.name);
    if (targetNames.has(manifest.name)) {
      if (baselineIdentity === undefined) {
        throw new Error(`Missing prepublish baseline for ${exactSpec(manifest)}`);
      }
      console.log(`prepare ${exactSpec(manifest)}`);
      requireSuccess(run("pnpm", ["run", "prepublishOnly"], {
        cwd: dir,
        stdio: "inherit",
        timeoutMs: 20 * 60_000,
      }), `prepublishOnly failed for ${exactSpec(manifest)}`);
      assertTrackedWorkspaceClean(outputDirectory);
    }

    const tarball = packPackage(dir, manifest, outputDirectory);
    assertDirectChild(outputDirectory, tarball, "staged tarball");
    const packedIdentity = inspectPackedPackageTarball(tarball);
    if (`${packedIdentity.name}@${packedIdentity.version}` !== exactSpec(manifest)) {
      throw new Error(`Packed tarball identity does not match ${exactSpec(manifest)}`);
    }
    if (
      baselineIdentity !== undefined
      && (
        baselineIdentity.fileCount !== packedIdentity.fileCount
        || baselineIdentity.treeHash !== packedIdentity.treeHash
      )
    ) {
      throw new Error(
        `${exactSpec(manifest)} prepublishOnly changed its validated packed artifact tree`,
      );
    }
    const integrity = tarballIntegrity(tarball);
    artifacts.push(Object.freeze({
      spec: exactSpec(manifest),
      file: basename(tarball),
      integrity,
    }));
    console.log(`staged ${exactSpec(manifest)}: ${integrity}`);
  }
  const sourceRevision = requireSuccess(
    run("git", ["rev-parse", "HEAD"]),
    "Could not resolve staged release revision",
  ).stdout.trim();
  const manifest = {
    schema: STAGED_SCHEMA,
    sourceRevision,
    selectedPackageName: selectedPackageName ?? null,
    targetSpecs: targets.map(({ manifest: targetManifest }) => exactSpec(targetManifest)),
    artifacts,
  };
  writeFileSync(join(outputDirectory, STAGED_MANIFEST), `${JSON.stringify(manifest, null, 2)}\n`, {
    encoding: "utf8",
    flag: "wx",
  });
  console.log(`staged release manifest: ${join(outputDirectory, STAGED_MANIFEST)}`);
}

function packPackage(packageDirectory, manifest, destination) {
  const pack = run("pnpm", [
    "pack",
    "--json",
    "--config.ignore-scripts=true",
    "--pack-destination",
    destination,
  ], { cwd: packageDirectory, timeoutMs: 5 * 60_000 });
  requireSuccess(pack, `pnpm pack failed for ${exactSpec(manifest)}`);
  const packed = parsePackResult(pack.stdout, manifest);
  return resolve(packed.filename);
}

async function publishStagedRelease(release, targets, closure, statuses, requireProvenance) {
  if (!requireProvenance) {
    throw new Error("Publication requires --provenance");
  }
  assertPublishAuthority();
  assertStagedRevision(release.sourceRevision);
  const provenanceIdentity = githubProvenanceIdentity(release, targets);
  const artifactByName = new Map(
    release.artifacts.map((artifact) => [splitExactSpec(artifact.spec).name, artifact]),
  );
  const targetNames = new Set(targets.map(({ manifest }) => manifest.name));
  const preverifiedProvenance = new Set();
  const publicationState = classifyStagedPublication(
    orderedPackages,
    targetNames,
    closure,
    statuses,
    release.selectedPackageName,
  );
  const approvedExistingNames = new Set(publicationState.approvedExisting);
  const strictExistingNames = new Set(publicationState.strictExisting);

  // Complete every immutable check possible before the first registry mutation.
  for (const { manifest } of orderedPackages) {
    if (!closure.has(manifest.name)) {
      continue;
    }
    const artifact = artifactByName.get(manifest.name);
    if (artifact === undefined) {
      throw new Error(`Staged release is missing dependency-closure artifact ${exactSpec(manifest)}`);
    }
    if (statuses.get(manifest.name) === "published") {
      const equivalence = retry(
        () => verifyPublishedTarballEquivalence(artifact.path, artifact.spec),
        `registry equivalence for ${artifact.spec}`,
      );
      console.log(`verified ${artifact.spec}: staged/registry tree ${equivalence.treeHash}`);
      if (approvedExistingNames.has(manifest.name)) {
        const attested = await retryAsync(
          () => verifyPublishedNpmDependencyProvenance(artifact.spec, {
            expectedIntegrity: artifact.integrity,
            repository: provenanceIdentity.repository,
            allowedWorkflowPaths: APPROVED_RELEASE_WORKFLOWS,
          }),
          `dependency provenance for ${artifact.spec}`,
        );
        assertAttestedCommitReachableFromMain(attested.commit, artifact.spec);
        console.log(
          `verified ${artifact.spec}: approved ${attested.workflowPath} provenance at ${attested.commit}`,
        );
        preverifiedProvenance.add(manifest.name);
      }
    }
  }
  for (const { manifest } of targets) {
    if (!strictExistingNames.has(manifest.name)) {
      continue;
    }
    const artifact = artifactByName.get(manifest.name);
    await retryAsync(
      () => verifyPublishedNpmProvenance(artifact.spec, {
        expectedIntegrity: artifact.integrity,
        ...provenanceIdentity,
      }),
      `pre-publication resume provenance for ${artifact.spec}`,
    );
    preverifiedProvenance.add(manifest.name);
    console.log(`verified ${artifact.spec}: resumable registry signature and provenance identity`);
  }

  for (const { manifest } of targets) {
    const spec = exactSpec(manifest);
    const artifact = artifactByName.get(manifest.name);
    if (artifact === undefined) {
      throw new Error(`Staged release is missing publication target ${spec}`);
    }
    if (statuses.get(manifest.name) === "missing") {
      assertWorkspaceDependenciesStillPublished(manifest);
      assertPrivateArtifactUnchanged(artifact);
      console.log(`publish ${spec}: ${artifact.integrity}`);
      const args = [
        "publish",
        artifact.path,
        "--access", "public",
        "--ignore-scripts",
        "--registry", REGISTRY,
      ];
      args.push("--provenance");
      const result = run("npm", args, {
        stdio: "inherit",
        timeoutMs: 10 * 60_000,
        allowPublishAuthority: true,
      });
      if (result.status !== 0) {
        console.warn(`npm publish returned ${String(result.status)} for ${spec}; checking registry before failing`);
      }
      waitForPublished(manifest);
      statuses.set(manifest.name, "published");
    } else {
      console.log(`resume ${spec}: immutable version already exists`);
    }

    const equivalence = retry(
      () => verifyPublishedTarballEquivalence(artifact.path, spec),
      `post-publish registry equivalence for ${spec}`,
    );
    console.log(`verified ${spec}: immutable tree ${equivalence.treeHash}`);
    if (!preverifiedProvenance.has(manifest.name)) {
      await retryAsync(
        () => verifyPublishedNpmProvenance(spec, {
          expectedIntegrity: artifact.integrity,
          ...provenanceIdentity,
        }),
        `provenance for ${spec}`,
      );
    }
    console.log(`verified ${spec}: registry signature, provenance, workflow identity`);
  }
}

function readStagedRelease(directory, packageByName, selectedPackageName) {
  const resolvedDirectory = realpathSync(resolve(root, directory));
  const manifestPath = join(resolvedDirectory, STAGED_MANIFEST);
  const manifestText = readBoundedSingleLinkFile(
    manifestPath,
    MAX_STAGED_MANIFEST_BYTES,
    "Staged release manifest",
  ).toString("utf8");
  const parsed = JSON.parse(manifestText);
  if (
    parsed?.schema !== STAGED_SCHEMA
    || !hasExactKeys(parsed, ["schema", "sourceRevision", "selectedPackageName", "targetSpecs", "artifacts"])
    || !/^[0-9a-f]{40}$/u.test(parsed.sourceRevision)
    || (parsed.selectedPackageName !== null && typeof parsed.selectedPackageName !== "string")
    || !Array.isArray(parsed.targetSpecs)
    || !Array.isArray(parsed.artifacts)
  ) {
    throw new Error(`Invalid staged release manifest in ${resolvedDirectory}`);
  }
  if (manifestText !== `${JSON.stringify(parsed, null, 2)}\n`) {
    throw new Error(`Staged release manifest must use the canonical JSON encoding: ${manifestPath}`);
  }
  if ((parsed.selectedPackageName ?? undefined) !== selectedPackageName) {
    throw new Error("Staged release package selection does not match --package");
  }
  const targetSpecs = validateUniqueSpecs(parsed.targetSpecs, packageByName, "target");
  if (
    selectedPackageName !== undefined
    && (targetSpecs.length !== 1 || splitExactSpec(targetSpecs[0]).name !== selectedPackageName)
  ) {
    throw new Error("Selected-package staged release must target exactly the selected package");
  }
  const artifactSpecs = validateUniqueSpecs(
    parsed.artifacts.map((artifact) => artifact?.spec),
    packageByName,
    "artifact",
  );
  const artifactFiles = parsed.artifacts.map((artifact) => artifact?.file);
  if (
    artifactFiles.some((file) => typeof file !== "string")
    || new Set(artifactFiles).size !== artifactFiles.length
  ) {
    throw new Error("Staged release artifact filenames must be unique strings");
  }
  const expectedDirectoryEntries = [STAGED_MANIFEST, ...artifactFiles].sort();
  const actualDirectoryEntries = readdirSync(resolvedDirectory).sort();
  if (
    expectedDirectoryEntries.length !== actualDirectoryEntries.length
    || expectedDirectoryEntries.some((entry, index) => entry !== actualDirectoryEntries[index])
  ) {
    throw new Error("Staged release directory must contain only its manifest and declared artifacts");
  }
  const publishDirectory = mkdtempSync(join(tmpdir(), "browsergrad-publish-artifacts-"));
  chmodSync(publishDirectory, 0o700);
  try {
    const artifacts = parsed.artifacts.map((artifact, index) => {
      if (
        artifact === null
        || typeof artifact !== "object"
        || Array.isArray(artifact)
        || !hasExactKeys(artifact, ["spec", "file", "integrity"])
        || typeof artifact.file !== "string"
        || basename(artifact.file) !== artifact.file
        || !artifact.file.endsWith(".tgz")
        || !/^sha512-[A-Za-z0-9+/]+={0,2}$/u.test(artifact.integrity)
      ) {
        throw new Error(`Invalid staged artifact at index ${index}`);
      }
      const candidatePath = join(resolvedDirectory, artifact.file);
      const candidateStat = lstatSync(candidatePath);
      if (
        !candidateStat.isFile()
        || candidateStat.isSymbolicLink()
        || candidateStat.nlink !== 1
        || candidateStat.size > MAX_STAGED_TARBALL_BYTES
      ) {
        throw new Error(`Staged artifact must be a bounded single-link regular file: ${artifact.spec}`);
      }
      const sourcePath = realpathSync(candidatePath);
      assertDirectChild(resolvedDirectory, sourcePath, "staged artifact");
      const path = join(publishDirectory, artifact.file);
      const copied = copyVerifiedArtifact(sourcePath, path, artifact.integrity);
      const packedIdentity = inspectPackedPackageTarball(path);
      if (`${packedIdentity.name}@${packedIdentity.version}` !== artifact.spec) {
        throw new Error(`Staged tarball identity does not match manifest: ${artifact.spec}`);
      }
      return Object.freeze({
        spec: artifact.spec,
        file: artifact.file,
        integrity: artifact.integrity,
        path,
        fingerprint: copied.fingerprint,
      });
    });
    if (artifactSpecs.length !== artifacts.length) {
      throw new Error("Staged release artifact validation mismatch");
    }
    let cleaned = false;
    const cleanup = () => {
      if (!cleaned) {
        cleaned = true;
        rmSync(publishDirectory, { recursive: true, force: true });
      }
    };
    process.once("exit", cleanup);
    return Object.freeze({
      sourceRevision: parsed.sourceRevision,
      selectedPackageName: parsed.selectedPackageName,
      targetSpecs: Object.freeze(targetSpecs),
      artifacts: Object.freeze(artifacts),
      cleanup: () => {
        process.removeListener("exit", cleanup);
        cleanup();
      },
    });
  } catch (error) {
    rmSync(publishDirectory, { recursive: true, force: true });
    throw error;
  }
}

function validateStagedReleasePlan(release, targets, closure, allPackages, selectedPackageName) {
  const targetNames = new Set(targets.map(({ manifest }) => manifest.name));
  const targetSpecs = allPackages
    .filter(({ manifest }) => targetNames.has(manifest.name))
    .map(({ manifest }) => exactSpec(manifest));
  if (
    release.targetSpecs.length !== targetSpecs.length
    || release.targetSpecs.some((spec, index) => spec !== targetSpecs[index])
  ) {
    throw new Error("Staged release target order does not match the dependency-first workspace plan");
  }
  if (
    selectedPackageName !== undefined
    && (targetSpecs.length !== 1 || splitExactSpec(targetSpecs[0]).name !== selectedPackageName)
  ) {
    throw new Error("Selected-package staged release must target exactly the selected package");
  }
  const expectedArtifacts = allPackages
    .filter(({ manifest }) => closure.has(manifest.name))
    .map(({ manifest }) => exactSpec(manifest));
  const actualArtifacts = release.artifacts.map(({ spec }) => spec);
  if (
    expectedArtifacts.length !== actualArtifacts.length
    || expectedArtifacts.some((spec, index) => spec !== actualArtifacts[index])
  ) {
    throw new Error("Staged release artifacts must exactly match the dependency closure");
  }
}

function validateUniqueSpecs(specs, packageByName, label) {
  const seen = new Set();
  return specs.map((spec, index) => {
    const { name, version } = splitExactSpec(spec);
    const entry = packageByName.get(name);
    if (entry === undefined || entry.manifest.version !== version) {
      throw new Error(`Staged ${label} spec does not match workspace: ${String(spec)}`);
    }
    if (seen.has(name)) {
      throw new Error(`Duplicate staged ${label} package: ${name}`);
    }
    seen.add(name);
    return spec;
  });
}

function hasExactKeys(value, expectedKeys) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const actualKeys = Object.keys(value).sort();
  const sortedExpected = [...expectedKeys].sort();
  return actualKeys.length === sortedExpected.length
    && actualKeys.every((key, index) => key === sortedExpected[index]);
}

function npmVersionStatus({ manifest }) {
  const spec = exactSpec(manifest);
  let lastOutput = "";
  let missingObservations = 0;
  const readDirectory = mkdtempSync(join(tmpdir(), "browsergrad-npm-view-"));
  chmodSync(readDirectory, 0o700);
  try {
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      const result = run("npm", [
        "view", spec, "version", "--json",
        "--registry", REGISTRY,
        "--fetch-retries", "2",
        "--fetch-retry-mintimeout", "1000",
        "--fetch-retry-maxtimeout", "5000",
      ], { cwd: readDirectory });
      if (result.status === 0) {
        return "published";
      }
      lastOutput = `${result.stdout ?? ""}${result.stderr ?? ""}`;
      if (/(?:^|\W)E404(?:\W|$)/mu.test(lastOutput)) {
        missingObservations += 1;
        if (missingObservations >= 2) {
          return "missing";
        }
        sleep(500);
        continue;
      }
      if (attempt < 3) {
        sleep(1000 * attempt);
      }
    }
  } finally {
    rmSync(readDirectory, { recursive: true, force: true });
  }
  throw new Error(`Could not classify npm version for ${spec} after 3 attempts\n${lastOutput}`);
}

function waitForPublished(manifest) {
  for (let attempt = 1; attempt <= 8; attempt += 1) {
    if (npmVersionStatus({ manifest }) === "published") {
      return;
    }
    if (attempt < 8) {
      sleep(Math.min(1000 * (2 ** (attempt - 1)), 5000));
    }
  }
  throw new Error(`${exactSpec(manifest)} did not become visible after publication`);
}

function assertWorkspaceDependenciesStillPublished(manifest) {
  const dependencyNames = new Set();
  for (const field of WORKSPACE_RUNTIME_DEPENDENCY_FIELDS) {
    for (const dependencyName of Object.keys(manifest[field] ?? {})) {
      if (byName.has(dependencyName)) dependencyNames.add(dependencyName);
    }
  }
  for (const dependencyName of [...dependencyNames].sort()) {
    const dependency = byName.get(dependencyName);
    if (npmVersionStatus(dependency) !== "published") {
      throw new Error(
        `${exactSpec(manifest)} dependency disappeared immediately before publication: ${exactSpec(dependency.manifest)}`,
      );
    }
  }
}

function retry(operation, label) {
  let lastError;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      return operation();
    } catch (error) {
      lastError = error;
      if (attempt < 5) {
        sleep(Math.min(1000 * (2 ** (attempt - 1)), 5000));
      }
    }
  }
  throw new Error(`${label} failed after 5 attempts`, { cause: lastError });
}

async function retryAsync(operation, label) {
  let lastError;
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt < 5) {
        await new Promise((resolvePromise) => {
          setTimeout(resolvePromise, Math.min(1000 * (2 ** (attempt - 1)), 5000));
        });
      }
    }
  }
  throw new Error(`${label} failed after 5 attempts`, { cause: lastError });
}

function workspaceDependencyClosure(targets, allPackages) {
  const packageByName = new Map(allPackages.map((entry) => [entry.manifest.name, entry]));
  const required = new Set(targets.map(({ manifest }) => manifest.name));
  const queue = [...required];
  while (queue.length > 0) {
    const name = queue.shift();
    const entry = packageByName.get(name);
    for (const field of WORKSPACE_RUNTIME_DEPENDENCY_FIELDS) {
      for (const dependencyName of Object.keys(entry.manifest[field] ?? {})) {
        if (packageByName.has(dependencyName) && !required.has(dependencyName)) {
          required.add(dependencyName);
          queue.push(dependencyName);
        }
      }
    }
  }
  return required;
}

function assertFrozenArtifactLifecycle(manifest) {
  if (typeof manifest.scripts?.prepublishOnly !== "string" || manifest.scripts.prepublishOnly.length === 0) {
    throw new Error(`${manifest.name} must define the complete prepublishOnly validation gate`);
  }
  for (const lifecycle of ["prepublish", "prepare", "prepack", "postpack", "publish", "postpublish", "dependencies"]) {
    if (manifest.scripts?.[lifecycle] !== undefined) {
      throw new Error(`${manifest.name} cannot define ${lifecycle}; release artifacts are packed once after prepublishOnly`);
    }
  }
}

function assertTrackedWorkspaceClean(allowedUntrackedDirectory) {
  const result = requireSuccess(
    run("git", ["status", "--porcelain=v1", "--untracked-files=no"]),
    "Could not inspect tracked workspace state",
  );
  if (result.stdout.trim() !== "") {
    throw new Error(`Tracked workspace must remain clean while staging release artifacts:\n${result.stdout}`);
  }
  const untracked = requireSuccess(
    run("git", ["ls-files", "--others", "--exclude-standard", "-z"]),
    "Could not inspect untracked workspace state",
  ).stdout.split("\0").filter(Boolean);
  const allowedRelative = allowedUntrackedDirectory === undefined
    ? undefined
    : relative(root, allowedUntrackedDirectory);
  const allowedPrefix = allowedRelative !== undefined
    && allowedRelative !== ""
    && allowedRelative !== ".."
    && !allowedRelative.startsWith(`..${sep}`)
    ? `${allowedRelative}${sep}`
    : undefined;
  const unexpected = untracked.filter((path) =>
    allowedPrefix === undefined
      || (path !== allowedRelative && !path.startsWith(allowedPrefix)));
  if (unexpected.length !== 0) {
    throw new Error(
      `Untracked source files cannot enter staged release artifacts:\n${unexpected.join("\n")}`,
    );
  }
}

function assertPublishAuthority() {
  if (!process.env.NODE_AUTH_TOKEN && !process.env.ACTIONS_ID_TOKEN_REQUEST_URL) {
    throw new Error("npm granular token or GitHub Actions OIDC is required to publish");
  }
}

function assertStagedRevision(sourceRevision) {
  const actual = process.env.GITHUB_SHA
    ?? requireSuccess(run("git", ["rev-parse", "HEAD"]), "Could not resolve publish revision").stdout.trim();
  if (actual !== sourceRevision) {
    throw new Error(`Staged release revision ${sourceRevision} does not match publish revision ${actual}`);
  }
}

function assertAttestedCommitReachableFromMain(commit, spec) {
  requireSuccess(
    run("git", ["cat-file", "-e", `${commit}^{commit}`]),
    `Attested dependency commit is not available locally for ${spec}: ${commit}`,
  );
  requireSuccess(
    run("git", ["merge-base", "--is-ancestor", commit, "origin/main"]),
    `Attested dependency commit is not reachable from origin/main for ${spec}: ${commit}`,
  );
}

function githubProvenanceIdentity(release, targets) {
  const repositorySlug = requiredEnvironment("GITHUB_REPOSITORY");
  const ref = requiredEnvironment("GITHUB_REF");
  const commit = requiredEnvironment("GITHUB_SHA");
  const workflowRef = requiredEnvironment("GITHUB_WORKFLOW_REF");
  const prefix = `${repositorySlug}/`;
  const suffix = `@${ref}`;
  if (!workflowRef.startsWith(prefix) || !workflowRef.endsWith(suffix)) {
    throw new Error(`GITHUB_WORKFLOW_REF does not match repository/ref: ${workflowRef}`);
  }
  const workflowPath = workflowRef.slice(prefix.length, -suffix.length);
  if (!APPROVED_RELEASE_WORKFLOWS.includes(workflowPath)) {
    throw new Error(`Unexpected provenance workflow path: ${workflowPath}`);
  }
  if (release.selectedPackageName === null) {
    if (workflowPath !== ".github/workflows/publish-npm.yml" || ref !== "refs/heads/main") {
      throw new Error("Batch publication provenance must come from publish-npm.yml on refs/heads/main");
    }
  } else {
    if (targets.length !== 1 || targets[0].manifest.name !== release.selectedPackageName) {
      throw new Error("Selected publication provenance requires exactly one matching target");
    }
    const { name, version } = targets[0].manifest;
    const shortname = name.slice("@unlocalhosted/browsergrad-".length);
    const expectedRef = `refs/tags/${shortname}-v${version}`;
    if (workflowPath !== ".github/workflows/release.yml" || ref !== expectedRef) {
      throw new Error(
        `Selected publication provenance must come from release.yml at ${expectedRef}`,
      );
    }
  }
  return Object.freeze({
    repository: `https://github.com/${repositorySlug}`,
    workflowPath,
    ref,
    commit,
  });
}

function requiredEnvironment(name) {
  const value = process.env[name];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} is required for provenance verification`);
  }
  return value;
}

function parsePackResult(output, expectedManifest) {
  let parsed;
  try {
    parsed = JSON.parse(output);
  } catch (error) {
    throw new Error(`pnpm pack emitted invalid JSON for ${exactSpec(expectedManifest)}`, { cause: error });
  }
  if (
    parsed?.name !== expectedManifest.name
    || parsed?.version !== expectedManifest.version
    || typeof parsed.filename !== "string"
    || !Array.isArray(parsed.files)
    || parsed.files.length === 0
  ) {
    throw new Error(`pnpm pack emitted invalid metadata for ${exactSpec(expectedManifest)}`);
  }
  return parsed;
}

function copyVerifiedArtifact(sourcePath, destinationPath, expectedIntegrity) {
  let sourceFd;
  let destinationFd;
  try {
    sourceFd = openSync(sourcePath, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
    const sourceBefore = fstatSync(sourceFd);
    assertBoundedRegularFile(sourceBefore, sourcePath);
    destinationFd = openSync(
      destinationPath,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL,
      0o600,
    );
    const hash = createHash("sha512");
    const buffer = Buffer.allocUnsafe(COPY_BUFFER_BYTES);
    let copiedBytes = 0;
    while (copiedBytes < sourceBefore.size) {
      const bytesRead = readSync(
        sourceFd,
        buffer,
        0,
        Math.min(buffer.byteLength, sourceBefore.size - copiedBytes),
        copiedBytes,
      );
      if (bytesRead === 0) {
        throw new Error(`Staged artifact changed while being copied: ${sourcePath}`);
      }
      hash.update(buffer.subarray(0, bytesRead));
      let written = 0;
      while (written < bytesRead) {
        written += writeSync(destinationFd, buffer, written, bytesRead - written);
      }
      copiedBytes += bytesRead;
    }
    const sourceAfter = fstatSync(sourceFd);
    if (!sameFileFingerprint(sourceBefore, sourceAfter)) {
      throw new Error(`Staged artifact changed while being copied: ${sourcePath}`);
    }
    const destinationStat = fstatSync(destinationFd);
    assertBoundedRegularFile(destinationStat, destinationPath);
    if (destinationStat.size !== sourceBefore.size) {
      throw new Error(`Private artifact copy size mismatch: ${destinationPath}`);
    }
    const integrity = `sha512-${hash.digest("base64")}`;
    if (integrity !== expectedIntegrity) {
      throw new Error(`Staged artifact integrity mismatch: ${sourcePath}`);
    }
    return Object.freeze({
      fingerprint: Object.freeze(fileFingerprint(destinationStat)),
      integrity,
    });
  } finally {
    if (destinationFd !== undefined) closeSync(destinationFd);
    if (sourceFd !== undefined) closeSync(sourceFd);
  }
}

function readBoundedSingleLinkFile(path, maximumBytes, label) {
  const pathStat = lstatSync(path);
  if (
    !pathStat.isFile()
    || pathStat.isSymbolicLink()
    || pathStat.nlink !== 1
    || pathStat.size > maximumBytes
  ) {
    throw new Error(`${label} must be single-link and at most ${maximumBytes} bytes`);
  }
  let fd;
  try {
    fd = openSync(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
    const before = fstatSync(fd);
    if (
      !before.isFile()
      || before.nlink !== 1
      || before.size > maximumBytes
      || !sameFileFingerprint(pathStat, before)
    ) {
      throw new Error(`${label} changed before descriptor validation: ${path}`);
    }
    const content = Buffer.alloc(before.size);
    let offset = 0;
    while (offset < content.byteLength) {
      const bytesRead = readSync(
        fd,
        content,
        offset,
        content.byteLength - offset,
        offset,
      );
      if (bytesRead === 0) {
        throw new Error(`${label} changed while being read: ${path}`);
      }
      offset += bytesRead;
    }
    const after = fstatSync(fd);
    if (!sameFileFingerprint(before, after)) {
      throw new Error(`${label} changed while being read: ${path}`);
    }
    return content;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

function assertPrivateArtifactUnchanged(artifact) {
  const current = lstatSync(artifact.path);
  assertBoundedRegularFile(current, artifact.path);
  if (!sameFileFingerprint(current, artifact.fingerprint)) {
    throw new Error(`Private staged artifact changed before publication: ${artifact.spec}`);
  }
  if (tarballIntegrity(artifact.path) !== artifact.integrity) {
    throw new Error(`Private staged artifact integrity changed before publication: ${artifact.spec}`);
  }
  const packedIdentity = inspectPackedPackageTarball(artifact.path);
  if (`${packedIdentity.name}@${packedIdentity.version}` !== artifact.spec) {
    throw new Error(`Private staged artifact identity changed before publication: ${artifact.spec}`);
  }
}

function tarballIntegrity(path) {
  let fd;
  try {
    fd = openSync(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
    const before = fstatSync(fd);
    assertBoundedRegularFile(before, path);
    const hash = createHash("sha512");
    const buffer = Buffer.allocUnsafe(COPY_BUFFER_BYTES);
    let offset = 0;
    while (offset < before.size) {
      const bytesRead = readSync(fd, buffer, 0, Math.min(buffer.byteLength, before.size - offset), offset);
      if (bytesRead === 0) {
        throw new Error(`Tarball changed while hashing: ${path}`);
      }
      hash.update(buffer.subarray(0, bytesRead));
      offset += bytesRead;
    }
    const after = fstatSync(fd);
    if (!sameFileFingerprint(before, after)) {
      throw new Error(`Tarball changed while hashing: ${path}`);
    }
    return `sha512-${hash.digest("base64")}`;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

function assertBoundedRegularFile(stat, path) {
  if (!stat.isFile() || stat.nlink !== 1 || stat.size > MAX_STAGED_TARBALL_BYTES) {
    throw new Error(`Artifact must be a bounded single-link regular file: ${path}`);
  }
}

function fileFingerprint(stat) {
  return {
    dev: stat.dev,
    ino: stat.ino,
    mode: stat.mode,
    nlink: stat.nlink,
    size: stat.size,
    mtimeMs: stat.mtimeMs,
    ctimeMs: stat.ctimeMs,
  };
}

function sameFileFingerprint(left, right) {
  return ["dev", "ino", "mode", "nlink", "size", "mtimeMs", "ctimeMs"]
    .every((key) => left[key] === right[key]);
}

function assertDirectChild(parent, child, label) {
  const childRelative = relative(realpathSync(parent), realpathSync(child));
  if (childRelative === "" || childRelative.startsWith(`..${sep}`) || childRelative === ".." || childRelative.includes(sep)) {
    throw new Error(`${label} must be a direct child of ${parent}: ${child}`);
  }
}

function exactSpec(manifest) {
  return `${manifest.name}@${manifest.version}`;
}

function splitExactSpec(spec) {
  if (typeof spec !== "string") {
    throw new Error(`Expected exact package spec, got ${String(spec)}`);
  }
  const separator = spec.lastIndexOf("@");
  const name = spec.slice(0, separator);
  const version = spec.slice(separator + 1);
  if (!/^@unlocalhosted\/browsergrad-[a-z0-9-]+$/u.test(name) || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/u.test(version)) {
    throw new Error(`Expected exact public BrowserGrad package spec, got ${spec}`);
  }
  return Object.freeze({ name, version });
}

function readPackageJson(dir) {
  return JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
}

function requireSuccess(result, message) {
  if (result.status !== 0) {
    throw new Error(`${message}\n${result.stdout ?? ""}\n${result.stderr ?? ""}`);
  }
  return result;
}

function run(cmd, args, options = {}) {
  if (options.allowPublishAuthority === true && (cmd !== "npm" || args[0] !== "publish")) {
    throw new Error("Publish authority can be passed only to the exact npm publish command");
  }
  const result = spawnSync(cmd, args, {
    cwd: options.cwd ?? root,
    encoding: "utf8",
    stdio: options.stdio ?? "pipe",
    env: options.allowPublishAuthority === true
      ? process.env
      : createReadOnlyNpmEnvironment(),
    timeout: options.timeoutMs ?? 120_000,
    maxBuffer: options.maxBufferBytes ?? 16 * 1024 * 1024,
  });
  if (result.error !== undefined) {
    throw new Error(`${cmd} ${args.join(" ")} failed to execute`, { cause: result.error });
  }
  return result;
}

function sleep(milliseconds) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
}

function parseArguments(args) {
  const parsed = {
    dryRun: false,
    preflight: false,
    provenance: false,
    selectedPackageName: undefined,
    stageDir: undefined,
    publishStaged: undefined,
  };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--dry-run") {
      parsed.dryRun = true;
    } else if (argument === "--preflight") {
      parsed.preflight = true;
    } else if (argument === "--provenance") {
      parsed.provenance = true;
    } else if (["--package", "--stage-dir", "--publish-staged"].includes(argument)) {
      const value = args[index + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error(`${argument} requires a value`);
      }
      index += 1;
      if (argument === "--package") parsed.selectedPackageName = value;
      if (argument === "--stage-dir") parsed.stageDir = value;
      if (argument === "--publish-staged") parsed.publishStaged = value;
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }
  const modes = [parsed.dryRun, parsed.preflight, parsed.stageDir !== undefined, parsed.publishStaged !== undefined]
    .filter(Boolean).length;
  if (modes !== 1) {
    throw new Error("exactly one of --dry-run, --preflight, --stage-dir, or --publish-staged is required");
  }
  if (parsed.provenance && parsed.publishStaged === undefined) {
    throw new Error("--provenance is valid only with --publish-staged");
  }
  return Object.freeze(parsed);
}
