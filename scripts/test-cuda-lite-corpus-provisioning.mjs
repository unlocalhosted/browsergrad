#!/usr/bin/env node
import assert from "node:assert/strict";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  CudaLiteCorpusProvisioningError,
  probeCudaLiteCorpusHostToolchain,
  provisionCudaLiteCorpusCheckout,
  verifyCudaLiteCorpusCheckout,
  withCudaLiteCorpusCheckoutLease,
} from "./cuda-lite-corpus-provisioning.mjs";

const testRoot = mkdtempSync(path.join(os.tmpdir(), "browsergrad-corpus-provisioning-test-"));
const scriptsDir = path.dirname(fileURLToPath(import.meta.url));
const provisioningModuleUrl = new URL("./cuda-lite-corpus-provisioning.mjs", import.meta.url).href;

try {
  const hostToolchain = probeCudaLiteCorpusHostToolchain();
  assert.equal(hostToolchain.kind, "browsergrad-cuda-lite-corpus-host-toolchain-capability");
  assert.equal(hostToolchain.version, 1);
  assert.equal(hostToolchain.platform, process.platform);
  assert.equal(path.isAbsolute(hostToolchain.git.executable), true);
  assert.equal(path.isAbsolute(hostToolchain.python.executable), true);
  assert.match(hostToolchain.git.version, /^git version \S+/u);
  assert.match(hostToolchain.python.version, /^\d+\.\d+\.\d+/u);
  assert.equal(hostToolchain.descriptorRelativeIo, true);
  assert.equal(hostToolchain.noFollowOpen, true);
  assert.equal(
    hostToolchain.atomicNoReplacePrimitive,
    process.platform === "darwin" ? "renamex_np" : "renameat2",
  );

  const source = path.join(testRoot, "local-source");
  initRepository(source);
  const firstCommit = commitFile(source, "kernel.cu", "__global__ void first() {}\n", "first");

  const toolBin = path.join(testRoot, "host-toolchain-path");
  mkdirSync(toolBin);
  symlinkSync(hostToolchain.git.executable, path.join(toolBin, "git"));
  symlinkSync(hostToolchain.python.executable, path.join(toolBin, "python3"));
  const pathSelectedCheckout = path.join(testRoot, "path-selected-checkout");
  const pathSelected = runHostToolchainChild(
    hostToolchainEnvironment(toolBin),
    {
      root: testRoot,
      corpus: corpus({ path: pathSelectedCheckout, repo: source, commit: firstCommit }),
      gitStdio: "pipe",
    },
  );
  assert.equal(pathSelected.ok, true, pathSelected.message);
  assert.equal(pathSelected.capability.git.selection, "absolute-path-entry");
  assert.equal(pathSelected.capability.python.selection, "absolute-path-entry");
  assert.equal(pathSelected.capability.git.executable, hostToolchain.git.executable);
  assert.equal(pathSelected.capability.python.executable, hostToolchain.python.executable);
  assert.equal(pathSelected.admission.action, "created");
  assert.deepEqual(pathSelected.admission.hostToolchainCapability, pathSelected.capability);
  assert.equal(git(pathSelectedCheckout, ["rev-parse", "HEAD"]).trim(), firstCommit);

  const emptyToolBin = path.join(testRoot, "empty-host-toolchain-path");
  mkdirSync(emptyToolBin);
  const explicitlySelected = runHostToolchainChild(hostToolchainEnvironment(emptyToolBin, {
    BROWSERGRAD_CUDA_GIT_EXECUTABLE: hostToolchain.git.executable,
    BROWSERGRAD_CUDA_PYTHON_EXECUTABLE: hostToolchain.python.executable,
  }));
  assert.equal(explicitlySelected.ok, true, explicitlySelected.message);
  assert.equal(
    explicitlySelected.capability.git.selection,
    "BROWSERGRAD_CUDA_GIT_EXECUTABLE",
  );
  assert.equal(
    explicitlySelected.capability.python.selection,
    "BROWSERGRAD_CUDA_PYTHON_EXECUTABLE",
  );

  const absentTools = runHostToolchainChild(hostToolchainEnvironment(emptyToolBin));
  assert.equal(absentTools.ok, false);
  assert.equal(absentTools.code, "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN");
  assert.match(absentTools.message, /BROWSERGRAD_CUDA_GIT_EXECUTABLE/u);

  const invalidGit = runHostToolchainChild(hostToolchainEnvironment(emptyToolBin, {
    BROWSERGRAD_CUDA_GIT_EXECUTABLE: hostToolchain.python.executable,
    BROWSERGRAD_CUDA_PYTHON_EXECUTABLE: hostToolchain.python.executable,
  }));
  assert.equal(invalidGit.ok, false);
  assert.equal(invalidGit.code, "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN");
  assert.match(invalidGit.message, /Git capability probe failed/u);

  const invalidPython = runHostToolchainChild(hostToolchainEnvironment(emptyToolBin, {
    BROWSERGRAD_CUDA_GIT_EXECUTABLE: hostToolchain.git.executable,
    BROWSERGRAD_CUDA_PYTHON_EXECUTABLE: hostToolchain.git.executable,
  }));
  assert.equal(invalidPython.ok, false);
  assert.equal(invalidPython.code, "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN");
  assert.match(invalidPython.message, /Python host-helper capability probe failed/u);

  const checkoutPath = path.join(testRoot, "admitted-corpus");
  const firstCorpus = corpus({ path: checkoutPath, repo: source, commit: firstCommit });

  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: firstCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-ABSENT",
  );

  const createdStarted = process.hrtime.bigint();
  const created = provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: firstCorpus,
    gitStdio: "pipe",
  });
  const createdMs = Number(process.hrtime.bigint() - createdStarted) / 1_000_000;
  assert.equal(created.action, "created");
  assert.deepEqual(created.hostToolchainCapability, hostToolchain);
  assert.equal(created.corpus.repo, realpathSync(source));
  assert.equal(created.corpus.commit, firstCommit);
  assert.equal(created.configuredOriginUrlMatched, true);
  assert.equal(created.exactCommitObserved, true);
  assert.equal(created.exactPhysicalHeadTreeObserved, true);
  assert.equal(created.auditConsumerSnapshotMaterialized, false);
  assert.match(created.physicalTreeSha256, /^[0-9a-f]{64}$/u);
  assert.equal(created.corpusAuditExecuted, false);
  assert.equal(created.browserExecutionObserved, false);
  assert.equal(created.webgpuExecutionObserved, false);
  assert.equal(created.productionConformanceAuthorityMinted, false);
  assert.equal(created.releaseReady, false);
  assert.deepEqual(structuredClone(created), created);
  assert.equal(git(checkoutPath, ["rev-parse", "HEAD"]).trim(), firstCommit);
  assert.equal(
    git(checkoutPath, ["config", "--get", "remote.origin.url"]).trim(),
    realpathSync(source),
  );
  assert.equal(git(checkoutPath, ["status", "--porcelain=v1"]).trim(), "");
  assert.equal(
    existsSync(path.join(checkoutPath, ".git", ".browsergrad-owned-corpus-reservation")),
    false,
  );

  const exact = verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: firstCorpus });
  assert.equal(exact.action, "verified");
  let confirmedMutationHookCalled = false;
  const confirmed = provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: firstCorpus,
    gitStdio: "pipe",
    testOnlyAfterExistingCheckoutPinned: () => {
      confirmedMutationHookCalled = true;
      throw new Error("confirmed checkout must not enter the Git mutation/network seam");
    },
  });
  assert.equal(confirmed.action, "confirmed");
  assert.equal(confirmedMutationHookCalled, false);

  const secondCommit = commitFile(
    source,
    "kernel.cu",
    "__global__ void refreshed() {}\n",
    "second",
  );
  const refreshedCorpus = corpus({ path: checkoutPath, repo: source, commit: secondCommit });
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-WRONG-COMMIT",
  );
  const refreshed = provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: refreshedCorpus,
    gitStdio: "pipe",
  });
  assert.equal(refreshed.action, "refreshed");
  assert.equal(git(checkoutPath, ["rev-parse", "HEAD"]).trim(), secondCommit);
  assert.equal(readFileSync(path.join(checkoutPath, "kernel.cu"), "utf8"),
    "__global__ void refreshed() {}\n");

  const movedExistingCheckout = path.join(testRoot, "moved-existing-checkout");
  assertProvisioningError(
    () => provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: firstCorpus,
      gitStdio: "pipe",
      testOnlyAfterExistingCheckoutPinned: (physicalTarget) => {
        renameSync(physicalTarget, movedExistingCheckout);
        mkdirSync(physicalTarget);
        writeFileSync(path.join(physicalTarget, "FOREIGN"), "must receive zero writes\n");
      },
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT",
  );
  assert.deepEqual(readdirSync(checkoutPath), ["FOREIGN"]);
  assert.equal(readFileSync(path.join(checkoutPath, "FOREIGN"), "utf8"),
    "must receive zero writes\n");
  rmSync(checkoutPath, { recursive: true });
  renameSync(movedExistingCheckout, checkoutPath);
  assert.equal(git(checkoutPath, ["rev-parse", "HEAD"]).trim(), firstCommit);
  const restoredAfterReplacementRace = provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: refreshedCorpus,
    gitStdio: "pipe",
  });
  assert.equal(restoredAfterReplacementRace.action, "refreshed");
  assert.equal(git(checkoutPath, ["rev-parse", "HEAD"]).trim(), secondCommit);

  git(checkoutPath, ["update-index", "--skip-worktree", "kernel.cu"]);
  writeFileSync(path.join(checkoutPath, "kernel.cu"), "__global__ void hidden_skip() {}\n");
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-HIDDEN-INDEX-STATE",
  );
  writeFileSync(path.join(checkoutPath, "kernel.cu"), "__global__ void refreshed() {}\n");
  git(checkoutPath, ["update-index", "--no-skip-worktree", "kernel.cu"]);

  git(checkoutPath, ["update-index", "--assume-unchanged", "kernel.cu"]);
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-HIDDEN-INDEX-STATE",
  );
  git(checkoutPath, ["update-index", "--no-assume-unchanged", "kernel.cu"]);

  git(checkoutPath, ["config", "user.email", "browsergrad-tests@example.invalid"]);
  git(checkoutPath, ["config", "user.name", "BrowserGrad Tests"]);
  writeFileSync(path.join(checkoutPath, "kernel.cu"), "__global__ void replaced_head() {}\n");
  git(checkoutPath, ["add", "--", "kernel.cu"]);
  git(checkoutPath, ["commit", "--quiet", "-m", "malicious replacement"]);
  const replacementCommit = git(checkoutPath, ["rev-parse", "HEAD"]).trim();
  git(checkoutPath, ["reset", "--hard", "--quiet", secondCommit]);
  git(checkoutPath, ["replace", secondCommit, replacementCommit]);
  writeFileSync(path.join(checkoutPath, "kernel.cu"), "__global__ void replaced_head() {}\n");
  git(checkoutPath, ["add", "--", "kernel.cu"]);
  assert.equal(git(checkoutPath, ["rev-parse", "HEAD"]).trim(), secondCommit);
  assert.equal(git(checkoutPath, ["status", "--porcelain=v1"]).trim(), "");
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-REPLACE-REF",
  );
  git(checkoutPath, ["replace", "-d", secondCommit]);
  rmSync(path.join(checkoutPath, ".git", "refs", "replace"), { recursive: true });
  git(checkoutPath, ["reset", "--hard", "--quiet", secondCommit]);

  const infoExclude = path.join(checkoutPath, ".git", "info", "exclude");
  const originalExclude = readFileSync(infoExclude, "utf8");
  writeFileSync(infoExclude, `${originalExclude}\nignored-hidden.cu\n`);
  const ignoredPath = path.join(checkoutPath, "ignored-hidden.cu");
  writeFileSync(ignoredPath, "__global__ void ignored_hidden() {}\n");
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-DIRTY",
  );
  rmSync(ignoredPath);
  writeFileSync(infoExclude, originalExclude);

  const dirtyPath = path.join(checkoutPath, "untracked-dirty.cu");
  writeFileSync(dirtyPath, "// must be preserved\n");
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-DIRTY",
  );
  assertProvisioningError(
    () => provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: refreshedCorpus,
      gitStdio: "pipe",
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-DIRTY",
  );
  assert.equal(readFileSync(dirtyPath, "utf8"), "// must be preserved\n");
  rmSync(dirtyPath);

  const wrongOrigin = path.join(testRoot, "wrong-origin");
  initRepository(wrongOrigin);
  commitFile(wrongOrigin, "other.cu", "__global__ void other() {}\n", "other");
  git(checkoutPath, ["remote", "set-url", "origin", wrongOrigin]);
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-WRONG-ORIGIN",
  );
  assertProvisioningError(
    () => provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: refreshedCorpus,
      gitStdio: "pipe",
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-WRONG-ORIGIN",
  );
  assert.equal(git(checkoutPath, ["config", "--get", "remote.origin.url"]).trim(), wrongOrigin);
  git(checkoutPath, ["remote", "set-url", "origin", realpathSync(source)]);

  const submoduleSource = path.join(testRoot, "submodule-source");
  initRepository(submoduleSource);
  commitFile(submoduleSource, "root.cu", "__global__ void root() {}\n", "root");
  git(submoduleSource, [
    "-c", "protocol.file.allow=always", "submodule", "add", "--quiet",
    realpathSync(wrongOrigin), "nested",
  ]);
  git(submoduleSource, ["commit", "--quiet", "-am", "submodule"]);
  const submoduleCommit = git(submoduleSource, ["rev-parse", "HEAD"]).trim();
  const submodulePath = path.join(testRoot, "submodule-checkout");
  const submoduleCorpus = corpus({
    path: submodulePath,
    repo: submoduleSource,
    commit: submoduleCommit,
  });
  const submoduleAdmission = provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: submoduleCorpus,
    gitStdio: "pipe",
  });
  assert.equal(submoduleAdmission.pinnedSubmoduleCount, 1);
  assert.match(submoduleAdmission.pinnedSubmoduleBindingSha256, /^[0-9a-f]{64}$/u);
  assert.equal(
    submoduleAdmission.exactHeadTreeEntryCount,
    submoduleAdmission.exactHeadFileCount + 1,
  );
  const verifiedSubmodule = verifyCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: submoduleCorpus,
  });
  assert.equal(
    verifiedSubmodule.pinnedSubmoduleBindingSha256,
    submoduleAdmission.pinnedSubmoduleBindingSha256,
  );
  assert.equal(verifiedSubmodule.physicalTreeSha256, submoduleAdmission.physicalTreeSha256);

  const nestedGitlink = path.join(submodulePath, "nested");
  const nestedInitiallyPresent = existsSync(nestedGitlink);
  if (!nestedInitiallyPresent) mkdirSync(nestedGitlink);
  writeFileSync(path.join(nestedGitlink, "FOREIGN.cu"), "must be rejected\n");
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: submoduleCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-DIRTY",
  );
  rmSync(path.join(nestedGitlink, "FOREIGN.cu"));
  if (!nestedInitiallyPresent) rmSync(nestedGitlink, { recursive: true });

  if (existsSync(nestedGitlink)) rmSync(nestedGitlink, { recursive: true });
  symlinkSync(wrongOrigin, nestedGitlink, "dir");
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: submoduleCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-DIRTY",
  );
  rmSync(nestedGitlink);
  if (nestedInitiallyPresent) mkdirSync(nestedGitlink);

  assertProvisioningError(
    () => withCudaLiteCorpusCheckoutLease({
      root: testRoot,
      corpus: submoduleCorpus,
      skipFetch: true,
      gitStdio: "pipe",
    }, () => {
      if (nestedInitiallyPresent) rmSync(nestedGitlink, { recursive: true });
      else mkdirSync(nestedGitlink);
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-DIRTY",
  );
  if (nestedInitiallyPresent) mkdirSync(nestedGitlink);
  else rmSync(nestedGitlink, { recursive: true });

  const environmentKeys = [
    "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "LD_PRELOAD", "LD_LIBRARY_PATH",
  ];
  const previousEnvironment = new Map(environmentKeys.map((key) => [key, process.env[key]]));
  const ambientPath = path.join(testRoot, "ambient-config-checkout");
  const ambientCorpus = corpus({ path: ambientPath, repo: source, commit: secondCommit });
  try {
    process.env.GIT_CONFIG_COUNT = "1";
    process.env.GIT_CONFIG_KEY_0 = `url.${realpathSync(wrongOrigin)}.insteadOf`;
    process.env.GIT_CONFIG_VALUE_0 = realpathSync(source);
    assert.equal(
      git(testRoot, ["ls-remote", realpathSync(source), secondCommit]).trim(),
      "",
      "ambient url.*.insteadOf repro must redirect the unsanitized Git invocation",
    );
    process.env.DYLD_INSERT_LIBRARIES = "/definitely/not/a/browsergrad-test-library.dylib";
    process.env.DYLD_LIBRARY_PATH = "/definitely/not/a/browsergrad-test-library-path";
    process.env.LD_PRELOAD = "/definitely/not/a/browsergrad-test-library.so";
    process.env.LD_LIBRARY_PATH = "/definitely/not/a/browsergrad-test-library-path";
    const ambientAdmission = provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: ambientCorpus,
      gitStdio: "pipe",
    });
    assert.equal(ambientAdmission.action, "created");
    assert.equal(ambientAdmission.corpus.commit, secondCommit);
    assert.equal(readFileSync(path.join(ambientPath, "kernel.cu"), "utf8"),
      "__global__ void refreshed() {}\n");
  } finally {
    for (const [key, value] of previousEnvironment) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }

  const mergeHead = path.join(checkoutPath, ".git", "MERGE_HEAD");
  writeFileSync(mergeHead, `${firstCommit}\n`);
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
    "BG-CUDA-LITE-CORPUS-PROVISION-HIDDEN-GIT-STATE",
  );
  rmSync(mergeHead);

  const nonGitPath = path.join(testRoot, "not-a-checkout");
  mkdirSync(nonGitPath);
  const sentinel = path.join(nonGitPath, "preserve.txt");
  writeFileSync(sentinel, "preserve me\n");
  const nonGitCorpus = corpus({ path: nonGitPath, repo: source, commit: secondCommit });
  assertProvisioningError(
    () => provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: nonGitCorpus,
      gitStdio: "pipe",
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-NON-GIT",
  );
  assert.equal(readFileSync(sentinel, "utf8"), "preserve me\n");

  const fakeGitPath = path.join(testRoot, "fake-git-checkout");
  mkdirSync(path.join(fakeGitPath, ".git"), { recursive: true });
  const fakeGitCorpus = corpus({ path: fakeGitPath, repo: source, commit: secondCommit });
  assertProvisioningError(
    () => provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: fakeGitCorpus,
      gitStdio: "pipe",
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-NON-GIT",
  );

  const symlinkPath = path.join(testRoot, "symlink-checkout");
  symlinkSync(checkoutPath, symlinkPath, "dir");
  const symlinkCorpus = corpus({ path: symlinkPath, repo: source, commit: secondCommit });
  assertProvisioningError(
    () => provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: symlinkCorpus,
      gitStdio: "pipe",
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-SYMLINK",
  );
  assert.equal(git(checkoutPath, ["rev-parse", "HEAD"]).trim(), secondCommit);

  const competing = withCudaLiteCorpusCheckoutLease({
    root: testRoot,
    corpus: refreshedCorpus,
    skipFetch: true,
    gitStdio: "pipe",
  }, () => {
    assertProvisioningError(
      () => verifyCudaLiteCorpusCheckout({ root: testRoot, corpus: refreshedCorpus }),
      "BG-CUDA-LITE-CORPUS-PROVISION-BUSY",
    );
    const moduleUrl = new URL("./cuda-lite-corpus-provisioning.mjs", import.meta.url).href;
    const childScript = `
      import { verifyCudaLiteCorpusCheckout } from ${JSON.stringify(moduleUrl)};
      const options = JSON.parse(process.argv[1]);
      try {
        verifyCudaLiteCorpusCheckout(options);
        process.exit(9);
      } catch (error) {
        process.stdout.write(String(error?.code ?? "unknown"));
      }
    `;
    for (const childCorpus of [
      refreshedCorpus,
      { ...refreshedCorpus, id: "different-id-same-physical-checkout" },
    ]) {
      const child = spawnSync(process.execPath, [
        "--input-type=module", "-e", childScript,
        JSON.stringify({ root: testRoot, corpus: childCorpus }),
      ], { encoding: "utf8", shell: false });
      assert.equal(child.status, 0, child.stderr);
      assert.equal(child.stdout, "BG-CUDA-LITE-CORPUS-PROVISION-BUSY");
    }
    return "lease-held";
  });
  assert.equal(competing.result, "lease-held");

  const snapshotResiduesBeforeCrash = readdirSync(testRoot)
    .filter((entry) => entry.startsWith(".browsergrad-corpus-snapshot-"));
  const crashModuleUrl = new URL("./cuda-lite-corpus-provisioning.mjs", import.meta.url).href;
  const abruptLeaseOwner = spawnSync(process.execPath, [
    "--input-type=module", "-e", `
      import { withCudaLiteCorpusCheckoutLease } from ${JSON.stringify(crashModuleUrl)};
      const options = JSON.parse(process.argv[1]);
      withCudaLiteCorpusCheckoutLease(options, () => process.kill(process.pid, "SIGKILL"));
    `,
    JSON.stringify({
      root: testRoot,
      corpus: refreshedCorpus,
      skipFetch: true,
      gitStdio: "pipe",
    }),
  ], { encoding: "utf8", shell: false });
  assert.equal(abruptLeaseOwner.status, null);
  assert.equal(abruptLeaseOwner.signal, "SIGKILL");
  const snapshotResiduesAfterCrash = readdirSync(testRoot)
    .filter((entry) => entry.startsWith(".browsergrad-corpus-snapshot-"));
  assert.equal(snapshotResiduesAfterCrash.length, snapshotResiduesBeforeCrash.length + 1);
  const crashedSnapshotName = snapshotResiduesAfterCrash.find((entry) =>
    !snapshotResiduesBeforeCrash.includes(entry));
  assert.ok(crashedSnapshotName);
  const crashedSnapshotPath = path.join(testRoot, crashedSnapshotName);
  const crashedSnapshotMarker = JSON.parse(readFileSync(
    path.join(crashedSnapshotPath, ".browsergrad-owned-audit-snapshot"),
    "utf8",
  ));
  for (let index = 1; index <= 5; index += 1) {
    const basename = `.browsergrad-corpus-snapshot-` +
      `${crashedSnapshotMarker.targetPathSha256.slice(0, 16)}-` +
      index.toString(16).padStart(32, "0");
    const duplicatePath = path.join(testRoot, basename);
    mkdirSync(duplicatePath, { mode: 0o700 });
    writeFileSync(
      path.join(duplicatePath, ".browsergrad-owned-audit-snapshot"),
      `${JSON.stringify({
        ...crashedSnapshotMarker,
        token: index.toString(16).padStart(64, "0"),
        basename,
      })}\n`,
      { mode: 0o600 },
    );
    writeFileSync(
      path.join(duplicatePath, "kernel.cu"),
      readFileSync(path.join(crashedSnapshotPath, "kernel.cu")),
      { mode: 0o600 },
    );
  }
  const boundedResiduesBeforeRecovery = readdirSync(testRoot)
    .filter((entry) => entry.startsWith(".browsergrad-corpus-snapshot-"));
  assert.equal(boundedResiduesBeforeRecovery.length, snapshotResiduesBeforeCrash.length + 6);
  const recoveredAfterCrash = verifyCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: refreshedCorpus,
  });
  assert.equal(recoveredAfterCrash.action, "verified");
  assert.equal(recoveredAfterCrash.residueReclamation.reclaimLimit, 4);
  assert.equal(recoveredAfterCrash.residueReclamation.reclaimedCount, 4);
  const boundedResiduesAfterFirstRecovery = readdirSync(testRoot)
    .filter((entry) => entry.startsWith(".browsergrad-corpus-snapshot-"));
  assert.equal(
    boundedResiduesAfterFirstRecovery.length,
    boundedResiduesBeforeRecovery.length - 4,
    "one lease must reclaim no more than its declared residue bound",
  );
  const drainedAfterCrash = verifyCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: refreshedCorpus,
  });
  assert.equal(drainedAfterCrash.residueReclamation.reclaimedCount, 2);
  assert.deepEqual(
    readdirSync(testRoot).filter((entry) =>
      entry.startsWith(".browsergrad-corpus-snapshot-")),
    snapshotResiduesBeforeCrash,
  );

  const interruptedReservationPath = path.join(testRoot, "interrupted-reservation-checkout");
  const interruptedReservationCorpus = corpus({
    path: interruptedReservationPath,
    repo: source,
    commit: secondCommit,
  });
  const reservationResiduesBeforeCrash = reservationResidues(testRoot);
  const abruptReservationOwner = spawnSync(process.execPath, [
    "--input-type=module", "-e", `
      import { provisionCudaLiteCorpusCheckout } from ${JSON.stringify(crashModuleUrl)};
      const options = JSON.parse(process.argv[1]);
      provisionCudaLiteCorpusCheckout({
        ...options,
        testOnlyAfterDestinationReservation: () => process.kill(process.pid, "SIGKILL"),
      });
    `,
    JSON.stringify({
      root: testRoot,
      corpus: interruptedReservationCorpus,
      gitStdio: "pipe",
    }),
  ], { encoding: "utf8", shell: false });
  assert.equal(abruptReservationOwner.status, null);
  assert.equal(abruptReservationOwner.signal, "SIGKILL");
  const reservationResiduesAfterCrash = reservationResidues(testRoot);
  assert.equal(reservationResiduesAfterCrash.length, reservationResiduesBeforeCrash.length + 1);
  const interruptedReservationResidue = reservationResiduesAfterCrash.find((entry) =>
    !reservationResiduesBeforeCrash.includes(entry));
  assert.ok(interruptedReservationResidue);
  const recoveredReservation = provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: interruptedReservationCorpus,
    gitStdio: "pipe",
  });
  assert.equal(recoveredReservation.action, "created");
  assert.equal(recoveredReservation.residueReclamation.reclaimedCount, 1);
  assert.equal(existsSync(path.join(testRoot, interruptedReservationResidue)), false);

  const failedReservationPath = path.join(testRoot, "failed-owned-reservation-checkout");
  const failedOwnedReservationCorpus = corpus({
    path: failedReservationPath,
    repo: source,
    commit: secondCommit,
  });
  const reservationResiduesBeforeFailure = reservationResidues(testRoot);
  assert.throws(() => provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: failedOwnedReservationCorpus,
    gitStdio: "pipe",
    testOnlyAfterDestinationReservation: () => {
      throw new Error("fixture failure before Git initialization");
    },
  }));
  const reservationResiduesAfterFailure = reservationResidues(testRoot);
  assert.equal(reservationResiduesAfterFailure.length, reservationResiduesBeforeFailure.length + 1);
  const recoveredFailedReservation = provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: failedOwnedReservationCorpus,
    gitStdio: "pipe",
  });
  assert.equal(recoveredFailedReservation.action, "created");
  assert.equal(recoveredFailedReservation.residueReclamation.reclaimedCount, 1);
  assert.deepEqual(reservationResidues(testRoot), reservationResiduesBeforeFailure);

  const kernelPath = path.join(checkoutPath, "kernel.cu");
  assertProvisioningError(
    () => withCudaLiteCorpusCheckoutLease({
      root: testRoot,
      corpus: refreshedCorpus,
      skipFetch: true,
      gitStdio: "pipe",
    }, () => {
      const mutation = spawnSync(process.execPath, [
        "-e",
        "require('node:fs').writeFileSync(process.argv[1], '__global__ void raced() {}\\n')",
        kernelPath,
      ], { encoding: "utf8", shell: false });
      assert.equal(mutation.status, 0, mutation.stderr);
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-DIRTY",
  );
  writeFileSync(kernelPath, "__global__ void refreshed() {}\n");

  let consumedSnapshotPath;
  const transientRestore = withCudaLiteCorpusCheckoutLease({
    root: testRoot,
    corpus: refreshedCorpus,
    skipFetch: true,
    gitStdio: "pipe",
  }, (snapshotAdmission) => {
    consumedSnapshotPath = snapshotAdmission.corpus.physicalPath;
    assert.equal(snapshotAdmission.corpus.physicalPathKind,
      "private-owned-verified-audit-snapshot");
    assert.notEqual(consumedSnapshotPath, realpathSync(checkoutPath));
    writeFileSync(kernelPath, "__global__ void transient_malicious() {}\n");
    assert.equal(
      readFileSync(path.join(consumedSnapshotPath, "kernel.cu"), "utf8"),
      "__global__ void refreshed() {}\n",
    );
    writeFileSync(kernelPath, "__global__ void refreshed() {}\n");
    return "exact-snapshot-only";
  });
  assert.equal(transientRestore.result, "exact-snapshot-only");
  assert.equal(transientRestore.admission.auditConsumerSnapshotMaterialized, true);
  assert.equal(transientRestore.admission.corpus.physicalPath, realpathSync(checkoutPath));
  assert.equal(existsSync(consumedSnapshotPath), false);

  let foreignSnapshotPath;
  const movedOwnedSnapshot = path.join(testRoot, "moved-owned-audit-snapshot");
  assertProvisioningError(
    () => withCudaLiteCorpusCheckoutLease({
      root: testRoot,
      corpus: refreshedCorpus,
      skipFetch: true,
      gitStdio: "pipe",
      testOnlyBeforeAuditSnapshotCleanup: (snapshotPath) => {
        foreignSnapshotPath = snapshotPath;
        renameSync(snapshotPath, movedOwnedSnapshot);
        mkdirSync(snapshotPath);
        writeFileSync(path.join(snapshotPath, "FOREIGN"), "cleanup must not touch me\n");
      },
    }, () => "snapshot-cleanup-race"),
    "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-CLEANUP",
  );
  assert.deepEqual(readdirSync(foreignSnapshotPath), ["FOREIGN"]);
  assert.equal(readFileSync(path.join(foreignSnapshotPath, "FOREIGN"), "utf8"),
    "cleanup must not touch me\n");
  assert.equal(readFileSync(path.join(movedOwnedSnapshot, "kernel.cu"), "utf8"),
    "__global__ void refreshed() {}\n");
  assert.equal(
    existsSync(path.join(movedOwnedSnapshot, ".browsergrad-owned-audit-snapshot")),
    true,
  );

  const symlinkResidueTarget = path.join(testRoot, "foreign-symlink-residue-target");
  mkdirSync(symlinkResidueTarget);
  writeFileSync(path.join(symlinkResidueTarget, "FOREIGN"), "symlink target must survive\n");
  const symlinkResiduePath = path.join(
    testRoot,
    `.browsergrad-corpus-snapshot-` +
      `${crashedSnapshotMarker.targetPathSha256.slice(0, 16)}-${"a".repeat(32)}`,
  );
  symlinkSync(symlinkResidueTarget, symlinkResiduePath, "dir");

  const malformedResiduePath = path.join(
    testRoot,
    `.browsergrad-corpus-snapshot-` +
      `${crashedSnapshotMarker.targetPathSha256.slice(0, 16)}-${"b".repeat(32)}`,
  );
  mkdirSync(malformedResiduePath, { mode: 0o700 });
  writeFileSync(
    path.join(malformedResiduePath, ".browsergrad-owned-audit-snapshot"),
    "not-an-ownership-record\n",
    { mode: 0o600 },
  );

  const ambiguousMarkerResiduePath = writeSyntheticSnapshotResidue(
    testRoot,
    crashedSnapshotMarker,
    "c".repeat(32),
  );
  writeFileSync(
    path.join(ambiguousMarkerResiduePath, ".browsergrad-failed-corpus-reservation"),
    "ambiguous second marker\n",
    { mode: 0o600 },
  );
  const foreignUidResiduePath = writeSyntheticSnapshotResidue(
    testRoot,
    crashedSnapshotMarker,
    "d".repeat(32),
    { ownerUid: process.getuid() + 1 },
  );
  const replacementResiduePath = writeSyntheticSnapshotResidue(
    testRoot,
    crashedSnapshotMarker,
    "e".repeat(32),
  );
  const liveOwnerResiduePath = writeSyntheticSnapshotResidue(
    testRoot,
    crashedSnapshotMarker,
    "f".repeat(32),
    { ownerPid: process.pid },
  );
  const movedReplacementResidue = path.join(testRoot, "moved-owned-replacement-residue");
  let replacementHookCalled = false;
  const adversarialResidueAdmission = verifyCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: refreshedCorpus,
    testOnlyBeforeResidueReclamation: (candidatePath) => {
      if (path.basename(candidatePath) !== path.basename(replacementResiduePath)) return;
      replacementHookCalled = true;
      renameSync(candidatePath, movedReplacementResidue);
      mkdirSync(candidatePath, { mode: 0o700 });
      writeFileSync(path.join(candidatePath, "FOREIGN"), "replacement must survive\n");
    },
  });
  assert.equal(replacementHookCalled, true);
  assert.equal(adversarialResidueAdmission.residueReclamation.reclaimedCount, 0);
  assert.ok(adversarialResidueAdmission.residueReclamation.retainedAmbiguousCount >= 6);
  assert.equal(readFileSync(path.join(symlinkResidueTarget, "FOREIGN"), "utf8"),
    "symlink target must survive\n");
  assert.equal(existsSync(symlinkResiduePath), true);
  assert.equal(existsSync(malformedResiduePath), true);
  assert.equal(existsSync(ambiguousMarkerResiduePath), true);
  assert.equal(existsSync(foreignUidResiduePath), true);
  assert.equal(existsSync(liveOwnerResiduePath), true);
  assert.equal(readFileSync(path.join(replacementResiduePath, "FOREIGN"), "utf8"),
    "replacement must survive\n");
  assert.equal(
    existsSync(path.join(movedReplacementResidue, ".browsergrad-owned-audit-snapshot")),
    true,
  );

  const appearedPath = path.join(testRoot, "appearing-target");
  const appearedSentinel = path.join(appearedPath, "competitor.txt");
  const appearedCorpus = corpus({ path: appearedPath, repo: source, commit: secondCommit });
  assertProvisioningError(
    () => provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: appearedCorpus,
      gitStdio: "pipe",
      testOnlyBeforeDestinationReservation: (physicalTarget) => {
        mkdirSync(physicalTarget);
        writeFileSync(path.join(physicalTarget, "competitor.txt"), "must survive\n");
      },
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-TARGET-APPEARED",
  );
  assert.equal(readFileSync(appearedSentinel, "utf8"), "must survive\n");

  const hijackedPath = path.join(testRoot, "post-marker-hijack");
  const stolenReservation = path.join(testRoot, "stolen-owned-reservation");
  const hijackedCorpus = corpus({ path: hijackedPath, repo: source, commit: secondCommit });
  assertProvisioningError(
    () => provisionCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: hijackedCorpus,
      gitStdio: "pipe",
      testOnlyAfterDestinationReservation: (physicalTarget, stagingPath) => {
        renameSync(stagingPath, stolenReservation);
        mkdirSync(physicalTarget);
        writeFileSync(path.join(physicalTarget, "FOREIGN"), "must receive zero writes\n");
      },
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-TARGET-APPEARED",
  );
  assert.deepEqual(readdirSync(hijackedPath), ["FOREIGN"]);
  assert.equal(readFileSync(path.join(hijackedPath, "FOREIGN"), "utf8"),
    "must receive zero writes\n");
  assert.equal(existsSync(path.join(stolenReservation, ".git")), true);
  assert.equal(
    existsSync(path.join(stolenReservation, ".git", ".browsergrad-failed-corpus-reservation")),
    true,
  );

  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: corpus({ path: testRoot, repo: source, commit: secondCommit }),
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-PATH",
  );
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({
      root: path.parse(testRoot).root,
      corpus: corpus({ path: checkoutPath, repo: source, commit: secondCommit }),
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-PATH",
  );
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: corpus({
        path: path.join(path.dirname(testRoot), "outside-corpus"),
        repo: source,
        commit: secondCommit,
      }),
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-PATH",
  );
  assertProvisioningError(
    () => verifyCudaLiteCorpusCheckout({
      root: testRoot,
      corpus: corpus({ path: checkoutPath, repo: source, commit: secondCommit.slice(0, 12) }),
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-INVALID",
  );

  const failedPath = path.join(testRoot, "failed-provision");
  const failedCorpus = corpus({ path: failedPath, repo: source, commit: "f".repeat(40) });
  assert.throws(() => provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: failedCorpus,
    gitStdio: "pipe",
    testOnlyAfterDestinationReservation: (_physicalTarget, stagingPath) => {
      writeFileSync(path.join(stagingPath, "FOREIGN"), "watcher-owned file must survive\n");
    },
  }));
  assert.equal(lstatExists(failedPath), false);
  const preservedFailedReservation = readdirSync(testRoot)
    .filter((entry) => entry.startsWith(".browsergrad-corpus-reservation-"))
    .map((entry) => path.join(testRoot, entry))
    .find((entry) => existsSync(path.join(entry, "FOREIGN")));
  assert.ok(preservedFailedReservation);
  assert.equal(
    readFileSync(path.join(preservedFailedReservation, "FOREIGN"), "utf8"),
    "watcher-owned file must survive\n",
  );
  assert.equal(
    existsSync(path.join(
      preservedFailedReservation,
      ".git",
      ".browsergrad-failed-corpus-reservation",
    )),
    true,
  );
  assert.deepEqual(
    readdirSync(testRoot).filter((entry) => entry.startsWith(".browsergrad-corpus-lease-")),
    [],
  );
  const retriedFailedReservation = provisionCudaLiteCorpusCheckout({
    root: testRoot,
    corpus: corpus({ path: failedPath, repo: source, commit: secondCommit }),
    gitStdio: "pipe",
  });
  assert.equal(retriedFailedReservation.action, "created");
  assert.equal(retriedFailedReservation.residueReclamation.reclaimedCount, 0);
  assert.equal(retriedFailedReservation.residueReclamation.retainedAmbiguousCount, 1);
  assert.equal(
    readFileSync(path.join(preservedFailedReservation, "FOREIGN"), "utf8"),
    "watcher-owned file must survive\n",
  );

  const physicalRootA = path.join(testRoot, "physical-root-a");
  const physicalRootB = path.join(testRoot, "physical-root-b");
  const logicalRoot = path.join(testRoot, "logical-root");
  mkdirSync(physicalRootA);
  mkdirSync(physicalRootB);
  symlinkSync(physicalRootA, logicalRoot, "dir");
  const pinnedDisplayPath = path.join(logicalRoot, "pinned-checkout");
  const pinnedCorpus = corpus({ path: pinnedDisplayPath, repo: source, commit: secondCommit });
  provisionCudaLiteCorpusCheckout({ root: logicalRoot, corpus: pinnedCorpus, gitStdio: "pipe" });
  assertProvisioningError(
    () => withCudaLiteCorpusCheckoutLease({
      root: logicalRoot,
      corpus: pinnedCorpus,
      skipFetch: true,
      gitStdio: "pipe",
    }, () => {
      rmSync(logicalRoot);
      symlinkSync(physicalRootB, logicalRoot, "dir");
    }),
    "BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT",
  );
  rmSync(logicalRoot);
  symlinkSync(physicalRootA, logicalRoot, "dir");

  for (const script of [
    "audit-real-world-cuda-corpora.mjs",
    "provision-real-world-cuda-corpora.mjs",
  ]) {
    const mixedUnknown = spawnSync(process.execPath, [
      path.join(scriptsDir, script),
      "--skip-fetch",
      "--only", "cuda-samples",
      "--only", "unknown-corpus",
    ], { cwd: path.dirname(scriptsDir), encoding: "utf8", shell: false });
    assert.equal(mixedUnknown.status, 2, mixedUnknown.stderr);
    assert.match(mixedUnknown.stderr, /unknown CUDA-lite corpus id/u);
    assert.doesNotMatch(mixedUnknown.stderr, /expected pinned git checkout/u);
  }

  console.log(
    `CUDA-lite corpus provisioning tests ok ` +
      `(local provision ${createdMs.toFixed(3)} ms; ` +
      `snapshot ${transientRestore.admission.auditSnapshotMaterializationMs.toFixed(3)} ms, ` +
      `${transientRestore.admission.auditSnapshotFileCount} files, ` +
      `${transientRestore.admission.auditSnapshotByteCount} bytes)`,
  );
} finally {
  rmSync(testRoot, { recursive: true, force: true });
}

function hostToolchainEnvironment(pathValue, overrides = {}) {
  const environment = { ...process.env, PATH: pathValue };
  delete environment.BROWSERGRAD_CUDA_GIT_EXECUTABLE;
  delete environment.BROWSERGRAD_CUDA_PYTHON_EXECUTABLE;
  return { ...environment, ...overrides };
}

function runHostToolchainChild(environment, provisionOptions) {
  const source = `
import {
  probeCudaLiteCorpusHostToolchain,
  provisionCudaLiteCorpusCheckout,
} from ${JSON.stringify(provisioningModuleUrl)};
try {
  const capability = probeCudaLiteCorpusHostToolchain();
  const admission = ${provisionOptions === undefined
    ? "undefined"
    : `provisionCudaLiteCorpusCheckout(${JSON.stringify(provisionOptions)})`};
  console.log(JSON.stringify({ ok: true, capability, admission }));
} catch (error) {
  console.log(JSON.stringify({
    ok: false,
    code: error?.code ?? null,
    message: error?.message ?? String(error),
  }));
}
`;
  const result = spawnSync(process.execPath, ["--input-type=module", "-e", source], {
    cwd: path.dirname(scriptsDir),
    encoding: "utf8",
    env: environment,
    shell: false,
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout.trim());
}

function reservationResidues(root) {
  return readdirSync(root)
    .filter((entry) => entry.startsWith(".browsergrad-corpus-reservation-"))
    .sort();
}

function writeSyntheticSnapshotResidue(root, markerTemplate, suffix, overrides = {}) {
  const basename = `.browsergrad-corpus-snapshot-` +
    `${markerTemplate.targetPathSha256.slice(0, 16)}-${suffix}`;
  const residuePath = path.join(root, basename);
  mkdirSync(residuePath, { mode: 0o700 });
  writeFileSync(
    path.join(residuePath, ".browsergrad-owned-audit-snapshot"),
    `${JSON.stringify({
      ...markerTemplate,
      token: `${suffix}${suffix}`,
      basename,
      ...overrides,
    })}\n`,
    { mode: 0o600 },
  );
  writeFileSync(
    path.join(residuePath, "kernel.cu"),
    "__global__ void refreshed() {}\n",
    { mode: 0o600 },
  );
  return residuePath;
}

function corpus({ path: checkoutPath, repo, commit }) {
  return {
    id: path.basename(checkoutPath).replaceAll(/[^A-Za-z0-9._-]/gu, "-"),
    name: "local corpus fixture",
    repo,
    commit,
    path: checkoutPath,
    localFixtureRepository: true,
    expectations: {},
  };
}

function initRepository(directory) {
  mkdirSync(directory);
  git(directory, ["init", "--quiet"]);
  git(directory, ["config", "user.email", "browsergrad-tests@example.invalid"]);
  git(directory, ["config", "user.name", "BrowserGrad Tests"]);
  git(directory, ["config", "commit.gpgsign", "false"]);
}

function commitFile(directory, relativePath, contents, message) {
  writeFileSync(path.join(directory, relativePath), contents);
  git(directory, ["add", "--", relativePath]);
  git(directory, ["commit", "--quiet", "-m", message]);
  const commit = git(directory, ["rev-parse", "HEAD"]).trim();
  assert.match(commit, /^[0-9a-f]{40}$/u);
  return commit;
}

function git(cwd, args) {
  const result = spawnSync("git", args, {
    cwd,
    encoding: "utf8",
    shell: false,
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `git ${args.join(" ")} failed: ${result.stderr}`);
  return result.stdout;
}

function assertProvisioningError(callback, code) {
  assert.throws(callback, (error) =>
    error instanceof CudaLiteCorpusProvisioningError && error.code === code);
}

function lstatExists(target) {
  return existsSync(target);
}
