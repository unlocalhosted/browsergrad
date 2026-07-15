import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  readlinkSync,
  realpathSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { createReadOnlyNpmEnvironment } from "./npm-read-only-environment.mjs";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const packagesRoot = realpathSync(join(root, "packages"));
const tarSnapshotScript = join(root, "scripts/snapshot-package-tar.py");

export function verifyPublishedPackageEquivalence(packageDir) {
  const resolvedPackageDir = resolvePublicWorkspacePackage(packageDir);
  const manifest = JSON.parse(readFileSync(join(resolvedPackageDir, "package.json"), "utf8"));
  if (typeof manifest.name !== "string" || typeof manifest.version !== "string") {
    throw new Error(`${resolvedPackageDir}/package.json must contain name and version`);
  }
  const spec = `${manifest.name}@${manifest.version}`;
  const temp = mkdtempSync(join(tmpdir(), "browsergrad-registry-equivalence-"));
  try {
    const localTarballs = join(temp, "local-tarballs");
    mkdirSync(localTarballs, { recursive: true });

    run("pnpm", ["pack", "--config.ignore-scripts=true", "--pack-destination", localTarballs], resolvedPackageDir);
    const localTarball = onlyTarball(localTarballs, `local ${spec}`);
    return verifyPublishedTarballEquivalence(localTarball, spec);
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
}

export function verifyPublishedTarballEquivalence(localTarball, spec) {
  if (typeof spec !== "string" || !/^@unlocalhosted\/browsergrad-[a-z0-9-]+@\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/u.test(spec)) {
    throw new Error(`Expected an exact public BrowserGrad package spec, got ${String(spec)}`);
  }
  const localTarballPath = resolve(localTarball);
  const localTarballStat = lstatSync(localTarballPath);
  if (!localTarballStat.isFile() || localTarballStat.isSymbolicLink()) {
    throw new Error(`Local tarball must be a regular non-symlink file: ${localTarball}`);
  }
  const resolvedLocalTarball = realpathSync(localTarballPath);
  const temp = mkdtempSync(join(tmpdir(), "browsergrad-registry-tarball-equivalence-"));
  try {
    const registryTarballs = join(temp, "registry-tarballs");
    mkdirSync(registryTarballs, { recursive: true });
    run("npm", [
      "pack",
      spec,
      "--pack-destination", registryTarballs,
      "--ignore-scripts",
      "--registry=https://registry.npmjs.org/",
      "--fetch-retries=2",
      "--fetch-retry-mintimeout=1000",
      "--fetch-retry-maxtimeout=5000",
    ], registryTarballs);
    const registryTarball = onlyTarball(registryTarballs, `registry ${spec}`);
    const localSnapshot = snapshotPackageTarball(resolvedLocalTarball);
    const registrySnapshot = snapshotPackageTarball(registryTarball);
    const { name: expectedName, version: expectedVersion } = splitExactSpec(spec);
    for (const [label, snapshot] of [["staged", localSnapshot], ["registry", registrySnapshot]]) {
      if (snapshot.name !== expectedName || snapshot.version !== expectedVersion) {
        throw new Error(
          `${label} tarball identity ${snapshot.name}@${snapshot.version} does not match ${spec}`,
        );
      }
    }
    const comparison = comparePackageSnapshots(
      localSnapshot.entries,
      registrySnapshot.entries,
    );
    if (!comparison.equal) {
      throw new Error(
        `${spec} staged artifact differs from immutable registry artifact:\n${comparison.differences.join("\n")}`,
      );
    }
    return Object.freeze({
      spec,
      fileCount: comparison.fileCount,
      treeHash: comparison.treeHash,
    });
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
}

export function inspectPackedPackageTarball(tarball) {
  const tarballPath = resolve(tarball);
  const stat = lstatSync(tarballPath);
  if (!stat.isFile() || stat.isSymbolicLink()) {
    throw new Error(`Packed tarball must be a regular non-symlink file: ${tarball}`);
  }
  const snapshot = snapshotPackageTarball(realpathSync(tarballPath));
  return Object.freeze({
    name: snapshot.name,
    version: snapshot.version,
    fileCount: snapshot.entries.size,
    treeHash: hashSnapshot(snapshot.entries),
  });
}

export function comparePackageTrees(leftRoot, rightRoot) {
  return comparePackageSnapshots(snapshotTree(leftRoot), snapshotTree(rightRoot));
}

export function comparePackageSnapshots(left, right) {
  const paths = [...new Set([...left.keys(), ...right.keys()])].sort();
  const differences = [];
  for (const path of paths) {
    const leftEntry = left.get(path);
    const rightEntry = right.get(path);
    if (leftEntry === undefined) {
      differences.push(`registry-only ${path}`);
    } else if (rightEntry === undefined) {
      differences.push(`local-only ${path}`);
    } else if (leftEntry !== rightEntry) {
      differences.push(`content-or-mode ${path}`);
    }
    if (differences.length === 20) {
      differences.push("additional differences omitted");
      break;
    }
  }
  const equal = differences.length === 0;
  const treeHash = equal ? hashSnapshot(left) : undefined;
  return Object.freeze({ equal, differences: Object.freeze(differences), fileCount: paths.length, treeHash });
}

function snapshotPackageTarball(tarball) {
  const result = run("python3", [tarSnapshotScript, tarball], root, {
    timeoutMs: 30_000,
    maxBufferBytes: 16 * 1024 * 1024,
  });
  let parsed;
  try {
    parsed = JSON.parse(result.stdout);
  } catch (error) {
    throw new Error(`Package snapshot helper emitted invalid JSON for ${tarball}`, { cause: error });
  }
  if (
    parsed?.schema !== "browsergrad.packed-package-snapshot@1"
    || typeof parsed.packageName !== "string"
    || typeof parsed.packageVersion !== "string"
    || !Array.isArray(parsed.entries)
  ) {
    throw new Error(`Package snapshot helper emitted an invalid envelope for ${tarball}`);
  }
  const snapshot = new Map();
  for (const entry of parsed.entries) {
    if (
      !Array.isArray(entry)
      || entry.length !== 2
      || typeof entry[0] !== "string"
      || typeof entry[1] !== "string"
      || snapshot.has(entry[0])
    ) {
      throw new Error(`Package snapshot helper emitted an invalid entry for ${tarball}`);
    }
    snapshot.set(entry[0], entry[1]);
  }
  if (snapshot.size === 0) {
    throw new Error(`Package snapshot helper emitted no files for ${tarball}`);
  }
  return Object.freeze({
    name: parsed.packageName,
    version: parsed.packageVersion,
    entries: snapshot,
  });
}

function snapshotTree(treeRoot) {
  const snapshot = new Map();
  visit(treeRoot, "");
  return snapshot;

  function visit(directory, prefix) {
    const entries = readdirSync(directory, { withFileTypes: true })
      .sort((left, right) => compareText(left.name, right.name));
    for (const entry of entries) {
      const relativePath = prefix === "" ? entry.name : `${prefix}/${entry.name}`;
      const absolutePath = join(directory, entry.name);
      if (entry.isDirectory()) {
        visit(absolutePath, relativePath);
        continue;
      }
      const stat = lstatSync(absolutePath);
      const mode = (stat.mode & 0o777).toString(8).padStart(3, "0");
      if (entry.isFile()) {
        const digest = createHash("sha512").update(readFileSync(absolutePath)).digest("hex");
        snapshot.set(relativePath, `file:${mode}:${stat.size}:${digest}`);
      } else if (entry.isSymbolicLink()) {
        snapshot.set(relativePath, `link:${mode}:${readlinkSync(absolutePath)}`);
      } else {
        throw new Error(`Unsupported packed entry type: ${relativePath}`);
      }
    }
  }
}

function hashSnapshot(snapshot) {
  const hash = createHash("sha256");
  hash.update("browsergrad.packed-package-tree@1\0");
  for (const [path, identity] of [...snapshot.entries()].sort(([left], [right]) => compareText(left, right))) {
    hash.update(path);
    hash.update("\0");
    hash.update(identity);
    hash.update("\0");
  }
  return hash.digest("hex");
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function onlyTarball(directory, label) {
  const tarballs = readdirSync(directory)
    .filter((name) => name.endsWith(".tgz"))
    .map((name) => join(directory, name));
  if (tarballs.length !== 1) {
    throw new Error(`Expected exactly one ${label} tarball, found ${tarballs.length}`);
  }
  return tarballs[0];
}

function publishedVersionExists(packageDir) {
  const manifest = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"));
  const spec = `${manifest.name}@${manifest.version}`;
  const readDirectory = mkdtempSync(join(tmpdir(), "browsergrad-npm-view-"));
  let result;
  try {
    result = spawnSync("npm", [
      "view", spec, "version",
      "--registry=https://registry.npmjs.org/",
      "--fetch-retries=2",
      "--fetch-retry-mintimeout=1000",
      "--fetch-retry-maxtimeout=5000",
    ], {
      cwd: readDirectory,
      encoding: "utf8",
      stdio: "pipe",
      env: createReadOnlyNpmEnvironment(),
    });
  } finally {
    rmSync(readDirectory, { recursive: true, force: true });
  }
  if (result.status === 0) {
    return true;
  }
  const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
  if (output.includes("E404")) {
    return false;
  }
  throw new Error(`Could not query npm version for ${spec}\n${output}`);
}

function splitExactSpec(spec) {
  const separator = spec.lastIndexOf("@");
  return Object.freeze({
    name: spec.slice(0, separator),
    version: spec.slice(separator + 1),
  });
}

function resolvePublicWorkspacePackage(packageDir) {
  const resolved = realpathSync(resolve(root, packageDir));
  const workspaceRelative = relative(packagesRoot, resolved);
  if (
    workspaceRelative === ""
    || workspaceRelative.startsWith(`..${sep}`)
    || workspaceRelative === ".."
    || workspaceRelative.includes(sep)
  ) {
    throw new Error(`Package directory must be a direct child of ${packagesRoot}: ${packageDir}`);
  }
  const manifest = JSON.parse(readFileSync(join(resolved, "package.json"), "utf8"));
  if (manifest.private === true || !manifest.name?.startsWith("@unlocalhosted/")) {
    throw new Error(`Package directory is not a public @unlocalhosted workspace package: ${packageDir}`);
  }
  return resolved;
}

function run(cmd, args, cwd, options = {}) {
  const result = spawnSync(cmd, args, {
    cwd,
    encoding: "utf8",
    stdio: "pipe",
    env: createReadOnlyNpmEnvironment(),
    timeout: options.timeoutMs ?? 120_000,
    maxBuffer: options.maxBufferBytes ?? 8 * 1024 * 1024,
  });
  if (result.error !== undefined) {
    throw new Error(`${cmd} ${args.join(" ")} failed to execute`, { cause: result.error });
  }
  if (result.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} failed\n${result.stdout ?? ""}\n${result.stderr ?? ""}`);
  }
  return result;
}

const invokedPath = process.argv[1] === undefined ? undefined : resolve(process.argv[1]);
if (invokedPath === fileURLToPath(import.meta.url)) {
  const allowMissing = process.argv.includes("--allow-missing");
  const packageDirs = process.argv.slice(2).filter((arg) => arg !== "--allow-missing");
  if (packageDirs.length === 0) {
    throw new Error(`Usage: node ${basename(process.argv[1])} [--allow-missing] <package-dir>...`);
  }
  for (const packageDir of packageDirs) {
    const resolvedPackageDir = resolve(root, packageDir);
    if (!publishedVersionExists(resolvedPackageDir)) {
      if (allowMissing) {
        const manifest = JSON.parse(readFileSync(join(resolvedPackageDir, "package.json"), "utf8"));
        console.log(`skip ${manifest.name}@${manifest.version}: not published yet`);
        continue;
      }
      throw new Error(`${packageDir} workspace version is not published`);
    }
    const result = verifyPublishedPackageEquivalence(resolvedPackageDir);
    console.log(`verified ${result.spec}: ${result.fileCount} files, tree ${result.treeHash}`);
  }
}
