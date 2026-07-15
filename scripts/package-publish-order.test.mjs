import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import test from "node:test";

import {
  classifyStagedPublication,
  sortWorkspacePackages,
  WORKSPACE_RUNTIME_DEPENDENCY_FIELDS,
} from "./package-publish-order.mjs";

const root = resolve(new URL("..", import.meta.url).pathname);

function entry(name, fields = {}) {
  return { dir: `/workspace/${name}`, manifest: { name, ...fields } };
}

test("orders workspace runtime, optional, and peer dependencies before consumers", () => {
  const packages = [
    entry("@scope/app", {
      dependencies: { "@scope/runtime": "workspace:*" },
      optionalDependencies: { "@scope/optional": "workspace:~" },
      peerDependencies: { "@scope/peer": "workspace:^" },
    }),
    entry("@scope/peer"),
    entry("@scope/optional"),
    entry("@scope/runtime"),
  ];

  const ordered = sortWorkspacePackages(packages).map(({ manifest }) => manifest.name);
  assert.equal(ordered.at(-1), "@scope/app");
  for (const dependency of ["@scope/runtime", "@scope/optional", "@scope/peer"]) {
    assert.ok(ordered.indexOf(dependency) < ordered.indexOf("@scope/app"));
  }
  assert.deepEqual(WORKSPACE_RUNTIME_DEPENDENCY_FIELDS, [
    "dependencies",
    "optionalDependencies",
    "peerDependencies",
  ]);
});

test("ignores external and development-only dependencies", () => {
  const ordered = sortWorkspacePackages([
    entry("@scope/a", { devDependencies: { "@scope/z": "workspace:*" } }),
    entry("@scope/z", { dependencies: { external: "1.0.0" } }),
  ]).map(({ manifest }) => manifest.name);
  assert.deepEqual(ordered, ["@scope/a", "@scope/z"]);
});

test("uses deterministic lexical traversal without mutating input", () => {
  const packages = [entry("@scope/z"), entry("@scope/a"), entry("@scope/m")];
  assert.deepEqual(
    sortWorkspacePackages(packages).map(({ manifest }) => manifest.name),
    ["@scope/a", "@scope/m", "@scope/z"],
  );
  assert.deepEqual(packages.map(({ manifest }) => manifest.name), ["@scope/z", "@scope/a", "@scope/m"]);
});

test("rejects cycles across dependency classes with complete cycle path", () => {
  assert.throws(
    () => sortWorkspacePackages([
      entry("@scope/a", { optionalDependencies: { "@scope/b": "1" } }),
      entry("@scope/b", { peerDependencies: { "@scope/c": "1" } }),
      entry("@scope/c", { dependencies: { "@scope/a": "1" } }),
    ]),
    /Circular workspace dependency: @scope\/a -> @scope\/b -> @scope\/c -> @scope\/a/u,
  );
});

test("rejects duplicate names and malformed dependency maps", () => {
  assert.throws(
    () => sortWorkspacePackages([entry("@scope/a"), entry("@scope/a")]),
    /Duplicate workspace package name/u,
  );
  assert.throws(
    () => sortWorkspacePackages([entry("@scope/a", { peerDependencies: [] })]),
    /@scope\/a peerDependencies must be an object/u,
  );
  assert.throws(
    () => sortWorkspacePackages([
      entry("@scope/a", { optionalDependencies: { "@scope/missing": "workspace:~" } }),
    ]),
    /target is not a public workspace package/u,
  );
  assert.throws(
    () => sortWorkspacePackages([
      entry("@scope/public", { dependencies: { "@scope/private": "^1.0.0" } }),
    ], {
      workspacePackageNames: ["@scope/public", "@scope/private"],
    }),
    /targets a non-public workspace package/u,
  );
});

test("current public workspace has dependency-first release order", () => {
  const packagesRoot = join(root, "packages");
  const workspacePackages = readdirSync(packagesRoot, { withFileTypes: true })
    .filter((candidate) => candidate.isDirectory())
    .map((candidate) => join(packagesRoot, candidate.name))
    .filter((dir) => existsSync(join(dir, "package.json")))
    .map((dir) => ({
      dir,
      manifest: JSON.parse(readFileSync(join(dir, "package.json"), "utf8")),
    }));
  const packages = workspacePackages
    .filter(({ manifest }) => manifest.private !== true && manifest.name?.startsWith("@unlocalhosted/"));

  const ordered = sortWorkspacePackages(packages, {
    workspacePackageNames: workspacePackages.map(({ manifest }) => manifest.name),
  }).map(({ manifest }) => manifest.name);
  const before = (dependency, consumer) => {
    assert.ok(
      ordered.indexOf(dependency) < ordered.indexOf(consumer),
      `${dependency} must publish before ${consumer}: ${ordered.join(", ")}`,
    );
  };
  before("@unlocalhosted/browsergrad-semantic-core", "@unlocalhosted/browsergrad-kernels");
  before("@unlocalhosted/browsergrad-semantic-core", "@unlocalhosted/browsergrad-compiler");
  before("@unlocalhosted/browsergrad-kernels", "@unlocalhosted/browsergrad-compiler");
  before("@unlocalhosted/browsergrad-kernels", "@unlocalhosted/browsergrad-jit");
});

test("batch resume audits every existing target and publishes only missing targets", () => {
  const packages = [entry("@scope/base"), entry("@scope/consumer")];
  const result = classifyStagedPublication(
    packages,
    ["@scope/base", "@scope/consumer"],
    ["@scope/base", "@scope/consumer"],
    new Map([
      ["@scope/base", "published"],
      ["@scope/consumer", "missing"],
    ]),
    null,
  );
  assert.deepEqual(result, {
    approvedExisting: ["@scope/base"],
    strictExisting: [],
    publish: ["@scope/consumer"],
  });
});

test("selected resume uses strict current identity while dependencies use approved provenance", () => {
  const packages = [entry("@scope/base"), entry("@scope/consumer")];
  const result = classifyStagedPublication(
    packages,
    ["@scope/consumer"],
    ["@scope/base", "@scope/consumer"],
    new Map([
      ["@scope/base", "published"],
      ["@scope/consumer", "published"],
    ]),
    "@scope/consumer",
  );
  assert.deepEqual(result, {
    approvedExisting: ["@scope/base"],
    strictExisting: ["@scope/consumer"],
    publish: [],
  });
});

test("selected missing target publishes only after existing dependencies are approved", () => {
  const packages = [entry("@scope/base"), entry("@scope/consumer")];
  const result = classifyStagedPublication(
    packages,
    ["@scope/consumer"],
    ["@scope/base", "@scope/consumer"],
    new Map([
      ["@scope/base", "published"],
      ["@scope/consumer", "missing"],
    ]),
    "@scope/consumer",
  );
  assert.deepEqual(result, {
    approvedExisting: ["@scope/base"],
    strictExisting: [],
    publish: ["@scope/consumer"],
  });
  assert.throws(
    () => classifyStagedPublication(
      packages,
      ["@scope/consumer"],
      ["@scope/base", "@scope/consumer"],
      new Map([
        ["@scope/base", "missing"],
        ["@scope/consumer", "missing"],
      ]),
      "@scope/consumer",
    ),
    /Unpublished dependency-closure package is not a target/u,
  );
});
