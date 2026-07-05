import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

const root = new URL("..", import.meta.url).pathname;
const packagesRoot = join(root, "packages");
const dryRun = process.argv.includes("--dry-run");

function run(cmd, args, options = {}) {
  const result = spawnSync(cmd, args, {
    cwd: options.cwd ?? root,
    encoding: "utf8",
    stdio: options.stdio ?? "pipe",
    env: process.env,
  });
  return result;
}

function packageJson(dir) {
  return JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
}

const packageDirs = sortByWorkspaceDependencies(readdirSync(packagesRoot, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => join(packagesRoot, entry.name))
  .filter((dir) => existsSync(join(dir, "package.json")))
  .filter((dir) => {
    const pkg = packageJson(dir);
    return pkg.private !== true && pkg.name?.startsWith("@unlocalhosted/");
  }));

if (packageDirs.length === 0) {
  throw new Error("No public @unlocalhosted packages found.");
}

if (!dryRun && !process.env.NODE_AUTH_TOKEN && !process.env.NPM_TOKEN) {
  throw new Error("NODE_AUTH_TOKEN or NPM_TOKEN is required to publish.");
}

for (const dir of packageDirs) {
  const pkg = packageJson(dir);
  const spec = `${pkg.name}@${pkg.version}`;
  const view = run("npm", ["view", spec, "version"]);

  if (view.status === 0) {
    console.log(`skip ${spec}: already published`);
    continue;
  }

  const output = `${view.stdout ?? ""}${view.stderr ?? ""}`;
  if (!output.includes("E404")) {
    console.error(output);
    throw new Error(`Could not check npm version for ${spec}`);
  }

  if (dryRun) {
    console.log(`would publish ${spec}`);
    continue;
  }

  console.log(`publish ${spec}`);
  const publish = run("pnpm", ["publish", "--access", "public", "--no-git-checks"], {
    cwd: dir,
    stdio: "inherit",
  });
  if (publish.status !== 0) {
    throw new Error(`pnpm publish failed for ${spec}`);
  }
}

function sortByWorkspaceDependencies(dirs) {
  const byName = new Map();
  const ordered = [];
  const visiting = new Set();
  const visited = new Set();

  for (const dir of dirs) {
    byName.set(packageJson(dir).name, dir);
  }

  for (const dir of dirs.sort()) {
    visit(dir);
  }

  return ordered;

  function visit(dir) {
    const pkg = packageJson(dir);
    if (visited.has(pkg.name)) {
      return;
    }
    if (visiting.has(pkg.name)) {
      throw new Error(`Circular workspace dependency involving ${pkg.name}`);
    }

    visiting.add(pkg.name);
    for (const [depName, depRange] of Object.entries(pkg.dependencies ?? {})) {
      if (depRange === "workspace:*" && byName.has(depName)) {
        visit(byName.get(depName));
      }
    }
    visiting.delete(pkg.name);
    visited.add(pkg.name);
    ordered.push(dir);
  }
}
