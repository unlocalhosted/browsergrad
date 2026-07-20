import {
  accessSync,
  closeSync,
  constants as fsConstants,
  fstatSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";

const CORPUS_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const FULL_GIT_COMMIT = /^[0-9a-f]{40}$/u;
const GITHUB_HTTPS_REPO = /^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\.git$/u;
const AUDIT_SOURCE = /\.(?:md|markdown|cu|cuh|cpp|cc|cxx|h|hpp)$/iu;
const RESERVATION_MARKER = ".browsergrad-owned-corpus-reservation";
const FAILED_RESERVATION_MARKER = ".browsergrad-failed-corpus-reservation";
const SNAPSHOT_MARKER = ".browsergrad-owned-audit-snapshot";
const LEASE_MARKER = "owner-token";
const GIT_EXECUTABLE_ENV = "BROWSERGRAD_CUDA_GIT_EXECUTABLE";
const PYTHON_EXECUTABLE_ENV = "BROWSERGRAD_CUDA_PYTHON_EXECUTABLE";
const WITHOUT_AUDIT_SNAPSHOT = Symbol("without-audit-snapshot");
const FORBIDDEN_LOCAL_CONFIG = /^(?:url\.|filter\.|protocol\.|credential\.|include(?:if)?\.|https?\.|core\.(?:hookspath|fsmonitor|sshcommand|attributesfile|excludesfile|gitproxy|usereplacerefs)|remote\.[^.]+\.(?:proxy|vcs|uploadpack|receivepack))/iu;
const HIDDEN_GIT_STATES = [
  "MERGE_HEAD",
  "CHERRY_PICK_HEAD",
  "REVERT_HEAD",
  "BISECT_LOG",
  "rebase-merge",
  "rebase-apply",
  "sequencer",
  "index.lock",
  RESERVATION_MARKER,
  FAILED_RESERVATION_MARKER,
];
let cachedHostToolchainCapability;

export class CudaLiteCorpusProvisioningError extends Error {
  constructor(code, message, options) {
    super(message, options);
    this.name = "CudaLiteCorpusProvisioningError";
    this.code = code;
  }
}

/** Resolve, pin, and probe the host tools required by corpus provisioning. */
export function probeCudaLiteCorpusHostToolchain() {
  if (cachedHostToolchainCapability !== undefined) return cachedHostToolchainCapability;
  if (process.platform !== "darwin" && process.platform !== "linux") {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN",
      `corpus provisioning requires a probed macOS or Linux host toolchain; ` +
        `${process.platform} is unsupported`,
    );
  }

  const git = resolveHostExecutable("git", GIT_EXECUTABLE_ENV);
  const python = resolveHostExecutable("python3", PYTHON_EXECUTABLE_ENV);
  const gitProbe = spawnSync(git.executable, ["--no-replace-objects", "--version"], {
    encoding: "utf8",
    env: safeGitEnvironment(),
    maxBuffer: 64 * 1024,
    shell: false,
  });
  if (gitProbe.error !== undefined || gitProbe.status !== 0 ||
      !/^git version \S+/u.test(gitProbe.stdout?.trim() ?? "")) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN",
      `Git capability probe failed for ${git.executable}: ` +
        hostProbeFailureDetail(gitProbe),
      gitProbe.error === undefined ? undefined : { cause: gitProbe.error },
    );
  }

  const pythonProbeSource = `
import ctypes, json, os, sys
primitive = "renamex_np" if sys.platform == "darwin" else "renameat2" if sys.platform.startswith("linux") else None
facts = {
    "platform": sys.platform,
    "pythonVersion": sys.version.split()[0],
    "descriptorRelativeIo": hasattr(os, "fchdir") and all(
        function in os.supports_dir_fd for function in (os.stat, os.open, os.unlink, os.rmdir)
    ) and os.stat in os.supports_follow_symlinks,
    "noFollowOpen": hasattr(os, "O_NOFOLLOW"),
    "atomicNoReplacePrimitive": primitive,
    "atomicNoReplaceSymbolAvailable": False,
}
if primitive is not None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        getattr(libc, primitive)
        facts["atomicNoReplaceSymbolAvailable"] = True
    except (AttributeError, OSError):
        pass
print(json.dumps(facts, sort_keys=True))
`;
  const pythonProbe = spawnSync(
    python.executable,
    ["-I", "-c", pythonProbeSource],
    {
      encoding: "utf8",
      env: safeHelperEnvironment(),
      maxBuffer: 64 * 1024,
      shell: false,
    },
  );
  let pythonFacts;
  try {
    pythonFacts = JSON.parse(pythonProbe.stdout?.trim() ?? "");
  } catch {
    pythonFacts = undefined;
  }
  const expectedPrimitive = process.platform === "darwin" ? "renamex_np" : "renameat2";
  if (pythonProbe.error !== undefined || pythonProbe.status !== 0 ||
      pythonFacts?.platform !== process.platform ||
      pythonFacts?.descriptorRelativeIo !== true || pythonFacts?.noFollowOpen !== true ||
      pythonFacts?.atomicNoReplacePrimitive !== expectedPrimitive ||
      pythonFacts?.atomicNoReplaceSymbolAvailable !== true ||
      typeof pythonFacts?.pythonVersion !== "string" || pythonFacts.pythonVersion.length === 0) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN",
      `Python host-helper capability probe failed for ${python.executable}; required: ` +
        `${process.platform}, descriptor-relative no-follow I/O, and libc.${expectedPrimitive}; ` +
        hostProbeFailureDetail(pythonProbe),
      pythonProbe.error === undefined ? undefined : { cause: pythonProbe.error },
    );
  }

  cachedHostToolchainCapability = Object.freeze({
    kind: "browsergrad-cuda-lite-corpus-host-toolchain-capability",
    version: 1,
    platform: process.platform,
    git: Object.freeze({
      executable: git.executable,
      selection: git.selection,
      version: gitProbe.stdout.trim(),
    }),
    python: Object.freeze({
      executable: python.executable,
      selection: python.selection,
      version: pythonFacts.pythonVersion,
    }),
    descriptorRelativeIo: true,
    noFollowOpen: true,
    atomicNoReplacePrimitive: expectedPrimitive,
  });
  return cachedHostToolchainCapability;
}

/** Verify one exact checkout under a short cooperative lease. */
export function verifyCudaLiteCorpusCheckout(options) {
  return withCudaLiteCorpusCheckoutLease(
    { ...options, skipFetch: true, [WITHOUT_AUDIT_SNAPSHOT]: true },
    () => undefined,
  ).admission;
}

/** Provision or refresh one exact checkout under a short cooperative lease. */
export function provisionCudaLiteCorpusCheckout(options) {
  return withCudaLiteCorpusCheckoutLease(
    { ...options, skipFetch: false, [WITHOUT_AUDIT_SNAPSHOT]: true },
    () => undefined,
  ).admission;
}

/**
 * Hold a stable, sibling-directory lease from admission through a synchronous
 * consumer such as the corpus audit, then revalidate the exact physical tree.
 * Non-cooperating mutation is detected by the mandatory post-consumer check.
 */
export function withCudaLiteCorpusCheckoutLease(options, consume) {
  if (typeof consume !== "function") {
    fail("BG-CUDA-LITE-CORPUS-PROVISION-INVALID", "lease consumer must be a function");
  }
  const hostToolchain = probeCudaLiteCorpusHostToolchain();
  const skipFetch = options?.skipFetch === true;
  const input = normalizeInput(options?.root, options?.corpus, {
    allowLocalFixtureRepo: options?.allowLocalFixtureRepo === true ||
      options?.corpus?.localFixtureRepository === true,
    createRoot: !skipFetch,
    testOnlyBeforeDestinationReservation: options?.testOnlyBeforeDestinationReservation,
    testOnlyAfterDestinationReservation: options?.testOnlyAfterDestinationReservation,
    testOnlyAfterExistingCheckoutPinned: options?.testOnlyAfterExistingCheckoutPinned,
    testOnlyBeforeAuditSnapshotCleanup: options?.testOnlyBeforeAuditSnapshotCleanup,
    hostToolchain,
  });
  const stdio = normalizeGitStdio(options?.gitStdio ?? "inherit");
  const materializeAuditSnapshot = options?.[WITHOUT_AUDIT_SNAPSHOT] !== true;
  const lease = acquireLease(input);
  let snapshot;
  let output;
  let primaryError;
  try {
    const state = skipFetch
      ? verifyCheckoutUnderLease(input)
      : provisionCheckoutUnderLease(input, stdio);
    if (materializeAuditSnapshot) {
      snapshot = createOwnedAuditSnapshot(input, state.inspection);
    }
    const admission = admissionRecord(input, state.action, state.inspection, snapshot);
    const consumerAdmission = snapshot === undefined
      ? admission
      : snapshotAdmissionRecord(admission, snapshot);
    let result;
    try {
      result = consume(consumerAdmission);
      if (result !== null && typeof result === "object" && typeof result.then === "function") {
        fail(
          "BG-CUDA-LITE-CORPUS-PROVISION-INVALID",
          "lease consumer must finish synchronously before checkout revalidation",
        );
      }
    } catch (cause) {
      primaryError = cause;
    }
    try {
      if (snapshot !== undefined) assertExactOwnedAuditSnapshot(input, snapshot);
      assertRootPin(input);
      const postInspection = inspectExistingCheckout(input, {
        absentIsSkipFetchFailure: skipFetch,
      });
      assertExactCommit(input, postInspection.head);
      assertSameBinding(state.inspection.binding, postInspection.binding, input);
    } catch (cause) {
      primaryError = primaryError === undefined
        ? cause
        : new AggregateError(
            [primaryError, cause],
            `${input.id} audit failed and its checkout changed before post-audit admission`,
          );
    }
    output = Object.freeze({ admission, result });
  } catch (cause) {
    primaryError = cause;
  }

  if (snapshot !== undefined) {
    try {
      input.testOnlyBeforeAuditSnapshotCleanup?.(snapshot.path);
      removeOwnedAuditSnapshot(input, snapshot);
    } catch (cause) {
      try { closeSync(snapshot.fd); } catch { /* Already closed by the cleanup helper. */ }
      primaryError = primaryError === undefined
        ? cause
        : new AggregateError(
            [primaryError, cause],
            `${input.id} operation failed and its audit snapshot could not be removed safely`,
          );
    }
  }

  try {
    releaseLease(input, lease);
  } catch (cause) {
    primaryError = primaryError === undefined
      ? cause
      : new AggregateError(
          [primaryError, cause],
          `${input.id} operation failed and its checkout lease could not be released safely`,
        );
  }
  if (primaryError !== undefined) throw primaryError;
  return output;
}

function verifyCheckoutUnderLease(input) {
  const inspection = inspectExistingCheckout(input, { absentIsSkipFetchFailure: true });
  assertExactCommit(input, inspection.head);
  return Object.freeze({ action: "verified", inspection });
}

function provisionCheckoutUnderLease(input, stdio) {
  assertRootPin(input);
  const targetStat = lstatIfPresent(input.path);
  if (targetStat === undefined) {
    input.testOnlyBeforeDestinationReservation?.(input.path);
    const reservation = reserveDestination(input);
    const reservedInput = inputForOwnedReservation(input, reservation);
    try {
      let primaryError;
      try {
        input.testOnlyAfterDestinationReservation?.(input.path, reservation.stagingPath);
        runGitMutation(reservedInput, ["init", "--quiet", "."], stdio);
        moveReservationMarkerIntoGitDirectory(reservedInput, reservation);
        runGitMutation(reservedInput, ["remote", "add", "origin", input.repo], stdio);
        assertSafeLocalGitConfig(reservedInput);
        runGitMutation(reservedInput, [
          "fetch", "--depth=1", "origin", input.commit,
        ], stdio);
        runGitMutation(reservedInput, ["checkout", "--detach", input.commit], stdio);
        if (lstatIfPresent(input.path) !== undefined) {
          fail(
            "BG-CUDA-LITE-CORPUS-PROVISION-TARGET-APPEARED",
            `${input.id} target appeared before atomic reservation installation: ${input.displayPath}`,
          );
        }
        const inspection = inspectExistingCheckout(reservedInput, {
          allowOwnedReservationMarker: true,
        });
        assertExactCommit(reservedInput, inspection.head);
        installOwnedReservation(input, reservation);
        completeReservation(reservedInput, reservation);
        return Object.freeze({ action: "created", inspection });
      } catch (cause) {
        primaryError = cause;
      }
      try {
        abandonOwnedReservation(reservedInput, reservation);
      } catch (cause) {
        primaryError = new AggregateError(
          [primaryError, cause],
          `failed to provision ${input.id} and mark its owned residue safely`,
        );
      }
      throw new CudaLiteCorpusProvisioningError(
        primaryError?.code ?? "BG-CUDA-LITE-CORPUS-PROVISION-FAILED-RESERVATION",
        `${String(primaryError?.message ?? primaryError)}; ` +
          `owned failed reservation was left at ${reservation.installed
            ? input.displayPath
            : reservation.stagingPath} and was not recursively deleted`,
        { cause: primaryError },
      );
    } finally {
      closeSync(reservation.fd);
    }
  }

  const before = inspectExistingCheckout(input, { targetStat });
  if (before.head === input.commit) {
    return Object.freeze({ action: "confirmed", inspection: before });
  }
  const pinned = openPinnedExistingCheckout(input, before.binding);
  try {
    input.testOnlyAfterExistingCheckoutPinned?.(input.path);
    runGitMutation(pinned.input, [
      "fetch", "--depth=1", "origin", input.commit,
    ], stdio);
    runGitMutation(pinned.input, ["checkout", "--detach", input.commit], stdio);
    assertSameFileIdentity(
      input.path,
      pinned.identity,
      input,
      "existing checkout final path changed during descriptor-relative mutation",
    );
    const inspection = inspectExistingCheckout(pinned.input);
    assertExactCommit(pinned.input, inspection.head);
    return Object.freeze({
      action: "refreshed",
      inspection,
    });
  } finally {
    closeSync(pinned.fd);
  }
}

function openPinnedExistingCheckout(input, binding) {
  const fd = openSync(input.path, fsConstants.O_RDONLY | fsConstants.O_DIRECTORY);
  const identity = fileIdentity(fstatSync(fd, { bigint: true }));
  if (identity.device !== binding.checkoutDevice || identity.inode !== binding.checkoutInode) {
    closeSync(fd);
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT",
      `${input.id} existing checkout changed before its mutation handle was pinned`,
    );
  }
  return Object.freeze({
    fd,
    identity,
    input: Object.freeze({ ...input, gitPath: ".", gitExtraFd: fd }),
  });
}

function inspectExistingCheckout(input, options = {}) {
  assertRootPin(input);
  const targetStat = options.targetStat ?? lstatIfPresent(input.path);
  if (targetStat === undefined) {
    if (options.absentIsSkipFetchFailure === true) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-ABSENT",
        `${input.id} expected pinned git checkout at ${input.displayPath}; ` +
          "run without --skip-fetch first",
      );
    }
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-ABSENT",
      `${input.id} expected git checkout at ${input.displayPath}`,
    );
  }
  if (targetStat.isSymbolicLink()) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-SYMLINK",
      `${input.id} checkout path must not be a symbolic link: ${input.displayPath}`,
    );
  }
  if (!targetStat.isDirectory()) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-NON-GIT",
      `${input.displayPath} exists but is not a git checkout`,
    );
  }
  const checkoutIdentity = fileIdentity(targetStat);

  const gitMetadata = lstatIfPresent(path.join(input.path, ".git"));
  if (gitMetadata === undefined || gitMetadata.isSymbolicLink() || !gitMetadata.isDirectory()) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-NON-GIT",
      `${input.displayPath} exists but is not a git checkout`,
    );
  }
  assertNoHiddenGitState(input, options.allowOwnedReservationMarker === true);

  const topLevelResult = gitCapture(input, [
    "-C", gitCheckoutPath(input), "rev-parse", "--show-toplevel",
  ], { returnResult: true });
  if (topLevelResult.status !== 0 ||
      realpathSync(topLevelResult.stdout.trim()) !== realpathSync(input.path)) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-NON-GIT",
      `${input.displayPath} exists but is not the exact usable git worktree root`,
    );
  }
  assertNoReplaceRefs(input);
  assertSafeLocalGitConfig(input);

  const objectFormat = gitCapture(input, [
    "-C", gitCheckoutPath(input), "rev-parse", "--show-object-format",
  ]).trim();
  if (objectFormat !== "sha1") {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSUPPORTED-GIT-STATE",
      `${input.id} checkout must use the pinned SHA-1 object format`,
    );
  }

  const originUrls = gitCapture(input, [
    "-C", gitCheckoutPath(input), "config", "--local", "--get-all", "remote.origin.url",
  ], { allowStatusOne: true });
  const origins = nonemptyLines(originUrls.stdout);
  if (originUrls.status !== 0 || origins.length !== 1 || origins[0] !== input.repo) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-WRONG-ORIGIN",
      `${input.id} expected configured origin ${input.repo} at ${input.displayPath}; ` +
        `got ${origins.length === 0 ? "<missing>" : origins.join(", ")}`,
    );
  }
  const effectiveUrls = gitCapture(input, [
    "-C", gitCheckoutPath(input), "remote", "get-url", "--all", "origin",
  ]);
  const effective = nonemptyLines(effectiveUrls);
  if (effective.length !== 1 || effective[0] !== input.repo) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-WRONG-ORIGIN",
      `${input.id} configured origin is rewritten away from ${input.repo}`,
    );
  }
  const pushUrls = gitCapture(input, [
    "-C", gitCheckoutPath(input), "config", "--local", "--get-all", "remote.origin.pushurl",
  ], { allowStatusOne: true });
  if (pushUrls.status === 0 && nonemptyLines(pushUrls.stdout).length > 0) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-WRONG-ORIGIN",
      `${input.id} checkout at ${input.displayPath} must not override origin push URLs`,
    );
  }

  const headResult = gitCapture(input, [
    "-C", gitCheckoutPath(input), "rev-parse", "--verify", "HEAD^{commit}",
  ], { returnResult: true });
  const head = headResult.stdout.trim();
  if (headResult.status !== 0 || !FULL_GIT_COMMIT.test(head)) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-WRONG-COMMIT",
      `${input.id} checkout at ${input.displayPath} did not resolve HEAD to a full commit`,
    );
  }

  assertNoHiddenIndexFlags(input);
  const binding = exactPhysicalTreeBinding(input, head, checkoutIdentity);
  const status = gitCapture(input, [
    "-C", gitCheckoutPath(input), "status", "--porcelain=v1", "--untracked-files=all",
    "--ignored=matching",
  ]).trim();
  if (status.length > 0) {
    dirty(input, status);
  }
  assertSameFileIdentity(input.path, checkoutIdentity, input, "checkout root changed during admission");
  assertRootPin(input);
  return Object.freeze({ head, binding });
}

function exactPhysicalTreeBinding(input, head, checkoutIdentity) {
  const treeOutput = gitCapture(input, [
    "-C", gitCheckoutPath(input), "ls-tree", "-r", "-z", "--full-tree", head,
  ]);
  const expected = new Map();
  const pinnedGitlinks = new Map();
  for (const raw of treeOutput.split("\0")) {
    if (raw.length === 0) continue;
    const tab = raw.indexOf("\t");
    const metadata = tab < 0 ? [] : raw.slice(0, tab).split(" ");
    const relative = tab < 0 ? "" : raw.slice(tab + 1);
    const [mode, type, objectId] = metadata;
    if (metadata.length !== 3 || !FULL_GIT_COMMIT.test(objectId ?? "") ||
        !safeRelativePath(relative) || expected.has(relative) ||
        pinnedGitlinks.has(relative) || mode === "120000") {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-UNSUPPORTED-GIT-STATE",
        `${input.id} HEAD contains a symlink or unsupported tree entry`,
      );
    }
    if (type === "blob" && mode !== "160000") {
      expected.set(relative, Object.freeze({ mode, objectId }));
      continue;
    }
    if (type === "commit" && mode === "160000") {
      pinnedGitlinks.set(relative, Object.freeze({ mode, objectId }));
      continue;
    }
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSUPPORTED-GIT-STATE",
      `${input.id} HEAD contains an unsupported tree entry at ${relative}`,
    );
  }

  const pinnedGitlinkStates = [];
  for (const [relative, entry] of [...pinnedGitlinks.entries()].sort(([left], [right]) =>
    compareCodeUnits(left, right))) {
    const absolute = path.join(input.path, relative);
    const stat = lstatIfPresent(absolute);
    let physicalState = "absent";
    if (stat !== undefined) {
      if (stat.isSymbolicLink() || !stat.isDirectory()) {
        dirty(input, `pinned gitlink path is not an absent or empty real directory: ${relative}`);
      }
      const children = readdirSync(absolute);
      if (children.length > 0) {
        dirty(input, `pinned gitlink path contains initialized or foreign content: ${relative}`);
      }
      physicalState = "empty-directory";
    }
    pinnedGitlinkStates.push(Object.freeze({ relative, ...entry, physicalState }));
  }

  const actualFiles = [];
  walkPhysicalFiles(input.path, "", input, actualFiles);
  if (actualFiles.length !== expected.size) {
    dirty(
      input,
      `physical file count ${actualFiles.length} differs from HEAD file count ${expected.size}`,
    );
  }

  let auditVisibleFileCount = 0;
  const manifestHash = createHash("sha256");
  const pinnedGitlinkHash = createHash("sha256");
  manifestHash.update("browsergrad.cuda-lite.corpus-physical-tree.v1\0");
  manifestHash.update(`${head}\0`);
  pinnedGitlinkHash.update("browsergrad.cuda-lite.pinned-gitlinks.v1\0");
  for (const relative of actualFiles.sort(compareCodeUnits)) {
    const treeEntry = expected.get(relative);
    if (treeEntry === undefined) dirty(input, `untracked or ignored physical file: ${relative}`);
    const absolute = path.join(input.path, relative);
    const before = lstatSync(absolute, { bigint: true });
    if (!before.isFile() || before.isSymbolicLink()) {
      dirty(input, `audit-visible path is not one regular file: ${relative}`);
    }
    const bytes = readFileSync(absolute);
    const after = lstatSync(absolute, { bigint: true });
    if (!sameBigIntIdentity(before, after) || before.size !== BigInt(bytes.byteLength) ||
        gitBlobObjectId(bytes) !== treeEntry.objectId) {
      dirty(input, `physical bytes differ from HEAD blob: ${relative}`);
    }
    if (AUDIT_SOURCE.test(relative)) auditVisibleFileCount += 1;
    manifestHash.update(`${treeEntry.mode}\0${treeEntry.objectId}\0${relative}\0`);
  }
  for (const gitlink of pinnedGitlinkStates) {
    const record = `${gitlink.mode}\0${gitlink.objectId}\0${gitlink.relative}\0` +
      `${gitlink.physicalState}\0`;
    manifestHash.update(`gitlink\0${record}`);
    pinnedGitlinkHash.update(record);
  }
  return Object.freeze({
    head,
    physicalTreeSha256: manifestHash.digest("hex"),
    exactHeadFileCount: expected.size,
    exactHeadTreeEntryCount: expected.size + pinnedGitlinkStates.length,
    auditVisibleFileCount,
    pinnedSubmoduleCount: pinnedGitlinkStates.length,
    pinnedSubmoduleBindingSha256: pinnedGitlinkHash.digest("hex"),
    checkoutDevice: checkoutIdentity.device,
    checkoutInode: checkoutIdentity.inode,
    entries: Object.freeze([...expected.entries()].map(([relative, entry]) =>
      Object.freeze({ relative, mode: entry.mode, objectId: entry.objectId }))),
  });
}

function createOwnedAuditSnapshot(input, inspection) {
  assertRootPin(input);
  const suffix = randomBytes(16).toString("hex");
  const snapshotPath = path.join(input.root, `.browsergrad-corpus-snapshot-${suffix}`);
  mkdirSync(snapshotPath, { mode: 0o700 });
  const identity = fileIdentity(lstatSync(snapshotPath, { bigint: true }));
  const token = randomBytes(32).toString("hex");
  const markerPath = path.join(snapshotPath, SNAPSHOT_MARKER);
  let markerIdentity;
  let fd;
  try {
    writeFileSync(markerPath, `${token}\n`, { encoding: "utf8", flag: "wx", mode: 0o600 });
    markerIdentity = fileIdentity(lstatSync(markerPath, { bigint: true }));
    fd = openSync(snapshotPath, fsConstants.O_RDONLY | fsConstants.O_DIRECTORY);
    const openedIdentity = fileIdentity(fstatSync(fd, { bigint: true }));
    if (openedIdentity.device !== identity.device || openedIdentity.inode !== identity.inode) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT",
        `${input.id} audit snapshot changed before its handle was pinned`,
      );
    }
  } catch (cause) {
    if (fd !== undefined) closeSync(fd);
    try {
      assertSameFileIdentity(
        snapshotPath,
        identity,
        input,
        "audit snapshot identity changed during marker creation",
      );
      rmdirSync(snapshotPath);
    } catch {
      /* Leave an ambiguous or nonempty snapshot untouched. */
    }
    throw cause;
  }

  const started = process.hrtime.bigint();
  const entries = Object.freeze(inspection.binding.entries.filter(({ relative }) =>
    AUDIT_SOURCE.test(relative)));
  const snapshot = {
    path: snapshotPath,
    identity,
    token,
    markerPath,
    markerIdentity,
    fd,
    entries,
  };
  try {
    if (entries.some(({ relative }) => relative === SNAPSHOT_MARKER)) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-UNSUPPORTED-GIT-STATE",
        `${input.id} HEAD collides with the private audit snapshot marker name`,
      );
    }
    for (const entry of entries) {
      const destination = path.join(snapshotPath, entry.relative);
      mkdirSync(path.dirname(destination), { recursive: true, mode: 0o700 });
      const bytes = readExactHeadBlob(input, entry);
      writeFileSync(destination, bytes, { flag: "wx", mode: 0o600 });
    }
    const binding = exactOwnedAuditSnapshotBinding(input, snapshot);
    snapshot.binding = binding;
    snapshot.materializationMs = Number(process.hrtime.bigint() - started) / 1_000_000;
    return Object.freeze(snapshot);
  } catch (cause) {
    closeSync(fd);
    throw new Error(
      `${String(cause?.message ?? cause)}; incomplete owned audit snapshot was left at ` +
        `${snapshotPath} and was not recursively deleted`,
      { cause },
    );
  }
}

function readExactHeadBlob(input, entry) {
  const absolute = path.join(input.path, entry.relative);
  const before = lstatSync(absolute, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink()) {
    dirty(input, `audit source is not one regular file: ${entry.relative}`);
  }
  const bytes = readFileSync(absolute);
  const after = lstatSync(absolute, { bigint: true });
  if (!sameBigIntIdentity(before, after) || before.size !== BigInt(bytes.byteLength) ||
      gitBlobObjectId(bytes) !== entry.objectId) {
    dirty(input, `audit source bytes differ from HEAD blob: ${entry.relative}`);
  }
  return bytes;
}

function exactOwnedAuditSnapshotBinding(input, snapshot) {
  assertOwnedAuditSnapshotIdentity(input, snapshot);
  const expected = new Map(snapshot.entries.map((entry) => [entry.relative, entry]));
  const actualFiles = [];
  const actualDirectories = [];
  walkSnapshotFiles(snapshot.path, "", input, actualFiles, actualDirectories);
  if (actualFiles.length !== expected.size) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-SNAPSHOT-DRIFT",
      `${input.id} audit snapshot file count changed`,
    );
  }
  const manifestHash = createHash("sha256");
  const identityHash = createHash("sha256");
  manifestHash.update("browsergrad.cuda-lite.audit-snapshot.v1\0");
  identityHash.update("browsergrad.cuda-lite.audit-snapshot-identities.v1\0");
  let byteCount = 0;
  const ownedFiles = [];
  for (const relative of actualFiles.sort(compareCodeUnits)) {
    const entry = expected.get(relative);
    if (entry === undefined) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-SNAPSHOT-DRIFT",
        `${input.id} audit snapshot gained unexpected file ${relative}`,
      );
    }
    const absolute = path.join(snapshot.path, relative);
    const before = lstatSync(absolute, { bigint: true });
    if (!before.isFile() || before.isSymbolicLink()) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-SNAPSHOT-DRIFT",
        `${input.id} audit snapshot path is not one regular file: ${relative}`,
      );
    }
    const bytes = readFileSync(absolute);
    const after = lstatSync(absolute, { bigint: true });
    if (!sameBigIntIdentity(before, after) || before.size !== BigInt(bytes.byteLength) ||
        gitBlobObjectId(bytes) !== entry.objectId) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-SNAPSHOT-DRIFT",
        `${input.id} audit snapshot bytes changed: ${relative}`,
      );
    }
    byteCount += bytes.byteLength;
    manifestHash.update(`${entry.mode}\0${entry.objectId}\0${relative}\0`);
    const identity = fileIdentity(before);
    ownedFiles.push(Object.freeze({ relative, ...identity }));
    identityHash.update(`file\0${relative}\0${identity.device}\0${identity.inode}\0`);
  }
  const ownedDirectories = actualDirectories
    .sort(compareCodeUnits)
    .map((relative) => {
      const identity = fileIdentity(lstatSync(path.join(snapshot.path, relative), { bigint: true }));
      identityHash.update(`directory\0${relative}\0${identity.device}\0${identity.inode}\0`);
      return Object.freeze({ relative, ...identity });
    });
  assertOwnedAuditSnapshotIdentity(input, snapshot);
  return Object.freeze({
    manifestSha256: manifestHash.digest("hex"),
    physicalIdentitySha256: identityHash.digest("hex"),
    fileCount: expected.size,
    byteCount,
    ownedFiles: Object.freeze(ownedFiles),
    ownedDirectories: Object.freeze(ownedDirectories),
  });
}

function walkSnapshotFiles(root, prefix, input, files, directories) {
  const directory = path.join(root, prefix);
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (prefix.length === 0 && entry.name === SNAPSHOT_MARKER) continue;
    const relative = path.join(prefix, entry.name);
    const absolute = path.join(root, relative);
    const stat = lstatSync(absolute);
    if (stat.isSymbolicLink()) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-SNAPSHOT-DRIFT",
        `${input.id} audit snapshot gained a symbolic link: ${relative}`,
      );
    }
    if (stat.isDirectory()) {
      directories.push(relative);
      walkSnapshotFiles(root, relative, input, files, directories);
      continue;
    }
    if (!stat.isFile()) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-SNAPSHOT-DRIFT",
        `${input.id} audit snapshot gained unsupported entry: ${relative}`,
      );
    }
    files.push(relative);
  }
}

function assertExactOwnedAuditSnapshot(input, snapshot) {
  const after = exactOwnedAuditSnapshotBinding(input, snapshot);
  for (const key of [
    "manifestSha256", "physicalIdentitySha256", "fileCount", "byteCount",
  ]) {
    if (after[key] !== snapshot.binding[key]) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-SNAPSHOT-DRIFT",
        `${input.id} audit snapshot binding changed while in use`,
      );
    }
  }
}

function assertOwnedAuditSnapshotIdentity(input, snapshot) {
  assertSameFileIdentity(snapshot.path, snapshot.identity, input, "audit snapshot identity changed");
  assertSameFileIdentity(
    snapshot.markerPath,
    snapshot.markerIdentity,
    input,
    "audit snapshot marker identity changed",
  );
  const markerStat = lstatSync(snapshot.markerPath);
  if (!markerStat.isFile() || markerStat.isSymbolicLink() ||
      readFileSync(snapshot.markerPath, "utf8") !== `${snapshot.token}\n`) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-CLEANUP",
      `${input.id} audit snapshot marker no longer proves ownership`,
    );
  }
}

function removeOwnedAuditSnapshot(input, snapshot) {
  const rootFd = openSync(input.root, fsConstants.O_RDONLY | fsConstants.O_DIRECTORY);
  try {
    const payload = JSON.stringify({
      rootIdentity: snapshot.identity,
      markerIdentity: snapshot.markerIdentity,
      markerName: SNAPSHOT_MARKER,
      token: snapshot.token,
      basename: path.basename(snapshot.path),
      files: snapshot.binding.ownedFiles.map((file) => ({
        ...file,
        objectId: snapshot.entries.find((entry) => entry.relative === file.relative).objectId,
      })),
      directories: snapshot.binding.ownedDirectories,
    });
    const helper = `
import hashlib, json, os, stat, sys
expected = json.load(sys.stdin)
root = os.fstat(3)
if (str(root.st_dev), str(root.st_ino)) != (expected["rootIdentity"]["device"], expected["rootIdentity"]["inode"]):
    sys.exit(41)
os.fchdir(3)
try:
    named = os.stat(expected["basename"], dir_fd=4, follow_symlinks=False)
except FileNotFoundError:
    sys.exit(42)
if (str(named.st_dev), str(named.st_ino)) != (expected["rootIdentity"]["device"], expected["rootIdentity"]["inode"]):
    sys.exit(42)
marker = os.stat(expected["markerName"], dir_fd=3, follow_symlinks=False)
if not stat.S_ISREG(marker.st_mode) or (str(marker.st_dev), str(marker.st_ino)) != (expected["markerIdentity"]["device"], expected["markerIdentity"]["inode"]):
    sys.exit(43)
marker_fd = os.open(expected["markerName"], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=3)
try:
    marker_bytes = os.read(marker_fd, 4096)
finally:
    os.close(marker_fd)
if marker_bytes != (expected["token"] + "\\n").encode():
    sys.exit(43)
expected_files = {item["relative"]: item for item in expected["files"]}
expected_directories = {item["relative"]: item for item in expected["directories"]}
actual_files = set()
actual_directories = set()
for current, directory_names, file_names in os.walk(".", topdown=True, followlinks=False):
    prefix = "" if current == "." else current[2:] + "/"
    for name in directory_names:
        relative = prefix + name
        item = os.stat(relative, dir_fd=3, follow_symlinks=False)
        if not stat.S_ISDIR(item.st_mode):
            sys.exit(44)
        actual_directories.add(relative)
    for name in file_names:
        relative = prefix + name
        if relative == expected["markerName"]:
            continue
        item = os.stat(relative, dir_fd=3, follow_symlinks=False)
        if not stat.S_ISREG(item.st_mode):
            sys.exit(44)
        actual_files.add(relative)
if actual_files != set(expected_files) or actual_directories != set(expected_directories):
    sys.exit(44)
for relative in sorted(actual_files):
    expected_file = expected_files[relative]
    descriptor = os.open(relative, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=3)
    try:
        item = os.fstat(descriptor)
        digest = hashlib.sha1()
        digest.update(b"blob " + str(item.st_size).encode() + b"\\0")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    if (str(item.st_dev), str(item.st_ino)) != (expected_file["device"], expected_file["inode"]):
        sys.exit(45)
    if digest.hexdigest() != expected_file["objectId"]:
        sys.exit(45)
for relative in sorted(actual_files):
    os.unlink(relative, dir_fd=3)
for relative in sorted(actual_directories, key=lambda value: (value.count("/"), value), reverse=True):
    expected_directory = expected_directories[relative]
    item = os.stat(relative, dir_fd=3, follow_symlinks=False)
    if (str(item.st_dev), str(item.st_ino)) != (expected_directory["device"], expected_directory["inode"]):
        sys.exit(46)
    os.rmdir(relative, dir_fd=3)
os.unlink(expected["markerName"], dir_fd=3)
named = os.stat(expected["basename"], dir_fd=4, follow_symlinks=False)
if (str(named.st_dev), str(named.st_ino)) != (expected["rootIdentity"]["device"], expected["rootIdentity"]["inode"]):
    sys.exit(47)
os.rmdir(expected["basename"], dir_fd=4)
`;
    const result = spawnSync(input.hostToolchain.python.executable, ["-I", "-c", helper], {
      cwd: input.root,
      encoding: "utf8",
      env: safeHelperEnvironment(),
      input: payload,
      shell: false,
      stdio: ["pipe", "pipe", "pipe", snapshot.fd, rootFd],
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-CLEANUP",
        `${input.id} exact descriptor-relative audit snapshot cleanup refused ` +
          `(status ${result.status}); residue was left in place`,
      );
    }
  } finally {
    closeSync(rootFd);
    closeSync(snapshot.fd);
  }
}

function walkPhysicalFiles(root, prefix, input, files) {
  const directory = path.join(root, prefix);
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (prefix.length === 0 && entry.name === ".git") continue;
    const relative = path.join(prefix, entry.name);
    const absolute = path.join(root, relative);
    const stat = lstatSync(absolute);
    if (stat.isSymbolicLink()) {
      dirty(input, `symbolic links are not admitted in corpus worktrees: ${relative}`);
    }
    if (stat.isDirectory()) {
      walkPhysicalFiles(root, relative, input, files);
      continue;
    }
    if (!stat.isFile()) dirty(input, `unsupported physical worktree entry: ${relative}`);
    files.push(relative);
  }
}

function assertNoHiddenIndexFlags(input) {
  const tagged = gitCapture(input, ["-C", gitCheckoutPath(input), "ls-files", "-v", "-z"]);
  for (const record of tagged.split("\0")) {
    if (record.length === 0) continue;
    if (!record.startsWith("H ")) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-HIDDEN-INDEX-STATE",
        `${input.id} index hides a tracked path with flag ${record[0] ?? "?"}: ${record.slice(2)}`,
      );
    }
  }
}

function assertNoReplaceRefs(input) {
  const looseReplacePath = path.join(input.path, ".git", "refs", "replace");
  if (lstatIfPresent(looseReplacePath) !== undefined) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-REPLACE-REF",
      `${input.id} checkout contains loose refs/replace state`,
    );
  }
  const packedRefsPath = path.join(input.path, ".git", "packed-refs");
  const packedRefsStat = lstatIfPresent(packedRefsPath);
  if (packedRefsStat !== undefined) {
    if (!packedRefsStat.isFile() || packedRefsStat.isSymbolicLink()) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-REPLACE-REF",
        `${input.id} checkout packed-refs is not one regular file`,
      );
    }
    if (/(?:^|\n)[0-9a-f]+ refs\/replace\//u.test(readFileSync(packedRefsPath, "utf8"))) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-REPLACE-REF",
        `${input.id} checkout contains packed refs/replace state`,
      );
    }
  }
  const effective = gitCapture(input, [
    "-C", gitCheckoutPath(input), "for-each-ref", "--format=%(refname)", "refs/replace",
  ]).trim();
  if (effective.length > 0) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-REPLACE-REF",
      `${input.id} checkout resolves refs/replace state`,
    );
  }
}

function assertNoHiddenGitState(input, allowOwnedReservationMarker) {
  for (const relative of HIDDEN_GIT_STATES) {
    if (allowOwnedReservationMarker && relative === RESERVATION_MARKER) continue;
    if (lstatIfPresent(path.join(input.path, ".git", relative)) !== undefined) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-HIDDEN-GIT-STATE",
        `${input.id} checkout contains active hidden git state: .git/${relative}`,
      );
    }
  }
}

function assertSafeLocalGitConfig(input) {
  const result = gitCapture(input, [
    "-C", gitCheckoutPath(input), "config", "--local", "--name-only", "--get-regexp", ".*",
  ], { allowStatusOne: true });
  for (const key of nonemptyLines(result.stdout)) {
    if (FORBIDDEN_LOCAL_CONFIG.test(key)) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-GIT-CONFIG",
        `${input.id} checkout contains forbidden local Git configuration: ${key}`,
      );
    }
  }
}

function assertExactCommit(input, actual) {
  if (actual !== input.commit) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-WRONG-COMMIT",
      `${input.id} expected ${input.commit}, got ${actual}; ` +
        "run without --skip-fetch to refresh",
    );
  }
}

function assertSameBinding(before, after, input) {
  for (const key of [
    "head", "physicalTreeSha256", "exactHeadFileCount", "exactHeadTreeEntryCount",
    "auditVisibleFileCount", "pinnedSubmoduleCount", "pinnedSubmoduleBindingSha256",
    "checkoutDevice", "checkoutInode",
  ]) {
    if (before[key] !== after[key]) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-LEASE-DRIFT",
        `${input.id} checkout binding changed while its audit lease was held`,
      );
    }
  }
}

function admissionRecord(input, action, inspection, snapshot) {
  return Object.freeze({
    kind: "browsergrad-cuda-lite-corpus-checkout-admission",
    version: 2,
    scope: "configured-origin-head-tree-and-physical-worktree-bytes-only",
    corpus: Object.freeze({
      id: input.id,
      repo: input.repo,
      commit: input.commit,
      path: input.displayPath,
      physicalPath: input.canonicalPath ?? input.path,
      physicalPathKind: "canonical-checkout",
    }),
    action,
    hostToolchainCapability: input.hostToolchain,
    configuredOriginUrlMatched: true,
    exactCommitObserved: true,
    exactPhysicalHeadTreeObserved: true,
    physicalTreeSha256: inspection.binding.physicalTreeSha256,
    exactHeadFileCount: inspection.binding.exactHeadFileCount,
    exactHeadTreeEntryCount: inspection.binding.exactHeadTreeEntryCount,
    auditVisibleFileCount: inspection.binding.auditVisibleFileCount,
    pinnedSubmoduleCount: inspection.binding.pinnedSubmoduleCount,
    pinnedSubmoduleBindingSha256: inspection.binding.pinnedSubmoduleBindingSha256,
    auditConsumerSnapshotMaterialized: snapshot !== undefined,
    auditSnapshotFileCount: snapshot?.binding.fileCount ?? 0,
    auditSnapshotByteCount: snapshot?.binding.byteCount ?? 0,
    auditSnapshotManifestSha256: snapshot?.binding.manifestSha256 ?? null,
    auditSnapshotMaterializationMs: snapshot === undefined
      ? 0
      : Math.round(snapshot.materializationMs * 1000) / 1000,
    cooperativeLeaseHeldDuringConsumer: true,
    postConsumerRevalidationRequired: true,
    corpusAuditExecuted: false,
    browserExecutionObserved: false,
    webgpuExecutionObserved: false,
    productionConformanceAuthorityMinted: false,
    releaseReady: false,
  });
}

function snapshotAdmissionRecord(admission, snapshot) {
  return Object.freeze({
    ...admission,
    scope: "configured-origin-head-tree-and-owned-audit-snapshot-bytes-only",
    corpus: Object.freeze({
      ...admission.corpus,
      physicalPath: snapshot.path,
      physicalPathKind: "private-owned-verified-audit-snapshot",
    }),
    auditSnapshotManifestSha256: snapshot.binding.manifestSha256,
  });
}

function normalizeInput(rootValue, corpusValue, options) {
  if (typeof corpusValue !== "object" || corpusValue === null) {
    fail("BG-CUDA-LITE-CORPUS-PROVISION-INVALID", "corpus input must be an object");
  }
  const id = corpusValue.id;
  const checkoutPath = corpusValue.path;
  if (typeof id !== "string" || !CORPUS_ID.test(id)) {
    fail("BG-CUDA-LITE-CORPUS-PROVISION-INVALID", "corpus id is invalid");
  }
  if (typeof rootValue !== "string" || !path.isAbsolute(rootValue)) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-PATH",
      "provisioning root must be an explicit absolute path",
    );
  }
  const logicalRoot = path.resolve(rootValue);
  if (logicalRoot === path.parse(logicalRoot).root) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-PATH",
      `filesystem root is too broad for corpus provisioning: ${logicalRoot}`,
    );
  }
  if (typeof checkoutPath !== "string" || !path.isAbsolute(checkoutPath)) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-PATH",
      `${id} checkout path must be absolute`,
    );
  }
  const displayPath = path.resolve(checkoutPath);
  if (displayPath === logicalRoot || path.dirname(displayPath) !== logicalRoot) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-PATH",
      `${id} checkout must be exactly one direct child of its explicit provisioning root`,
    );
  }

  let rootStat = lstatIfPresent(logicalRoot);
  if (rootStat === undefined && options.createRoot) {
    mkdirSync(logicalRoot, { recursive: true });
    rootStat = lstatIfPresent(logicalRoot);
  }
  if (rootStat === undefined) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-ABSENT",
      `${id} expected pinned git checkout at ${displayPath}; run without --skip-fetch first`,
    );
  }
  const physicalRoot = realpathSync(logicalRoot);
  const physicalRootStat = statSync(physicalRoot, { bigint: true });
  if (!physicalRootStat.isDirectory()) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-PATH",
      `provisioning root is not a directory: ${logicalRoot}`,
    );
  }

  const repo = normalizeRepository(corpusValue.repo, options.allowLocalFixtureRepo);
  const commit = corpusValue.commit;
  if (typeof commit !== "string" || !FULL_GIT_COMMIT.test(commit)) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-INVALID",
      `${id} commit must be one exact lowercase 40-character git object id`,
    );
  }
  if ([
    options.testOnlyBeforeDestinationReservation,
    options.testOnlyAfterDestinationReservation,
    options.testOnlyAfterExistingCheckoutPinned,
    options.testOnlyBeforeAuditSnapshotCleanup,
  ].some((hook) => hook !== undefined &&
    (!options.allowLocalFixtureRepo || typeof hook !== "function"))) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-INVALID",
      "destination-race test hooks are available only to explicit local fixtures",
    );
  }
  return Object.freeze({
    id,
    repo,
    commit,
    displayPath,
    logicalRoot,
    root: physicalRoot,
    path: path.join(physicalRoot, path.basename(displayPath)),
    rootIdentity: fileIdentity(physicalRootStat),
    hostToolchain: options.hostToolchain,
    allowLocalFixtureRepo: options.allowLocalFixtureRepo,
    testOnlyBeforeDestinationReservation: options.testOnlyBeforeDestinationReservation,
    testOnlyAfterDestinationReservation: options.testOnlyAfterDestinationReservation,
    testOnlyAfterExistingCheckoutPinned: options.testOnlyAfterExistingCheckoutPinned,
    testOnlyBeforeAuditSnapshotCleanup: options.testOnlyBeforeAuditSnapshotCleanup,
  });
}

function normalizeRepository(value, allowLocalFixtureRepo) {
  if (typeof value !== "string" || value.length === 0 || value.startsWith("-") || hasControl(value)) {
    fail("BG-CUDA-LITE-CORPUS-PROVISION-INVALID", "repository URL is invalid");
  }
  if (GITHUB_HTTPS_REPO.test(value)) return value;
  if (allowLocalFixtureRepo && path.isAbsolute(value)) {
    const physical = realpathSync(value);
    if (!statSync(physical).isDirectory()) {
      fail("BG-CUDA-LITE-CORPUS-PROVISION-INVALID", "local fixture repository is not a directory");
    }
    return physical;
  }
  fail(
    "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-REPOSITORY",
    "repository must be one exact GitHub HTTPS .git URL; local fixtures require explicit opt-in",
  );
}

function assertRootPin(input) {
  let currentPhysical;
  try {
    currentPhysical = realpathSync(input.logicalRoot);
  } catch (cause) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT",
      `${input.id} logical provisioning root disappeared while pinned`,
      { cause },
    );
  }
  if (currentPhysical !== input.root) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT",
      `${input.id} logical provisioning root was retargeted while pinned`,
    );
  }
  assertSameFileIdentity(input.root, input.rootIdentity, input, "physical provisioning root changed");
}

function acquireLease(input) {
  assertRootPin(input);
  const suffix = createHash("sha256").update(input.path).digest("hex").slice(0, 32);
  const leasePath = path.join(input.root, `.browsergrad-corpus-lease-${suffix}`);
  try {
    mkdirSync(leasePath, { mode: 0o700 });
  } catch (cause) {
    if (cause?.code === "EEXIST") {
      if (recoverProvablyStaleLease(input, leasePath)) {
        try {
          mkdirSync(leasePath, { mode: 0o700 });
          return finishLeaseAcquisition(input, leasePath);
        } catch (retryCause) {
          if (retryCause?.code !== "EEXIST") throw retryCause;
        }
      }
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-BUSY",
        `${input.id} checkout already has an active provisioning/audit lease`,
      );
    }
    throw cause;
  }
  return finishLeaseAcquisition(input, leasePath);
}

function finishLeaseAcquisition(input, leasePath) {
  const identity = fileIdentity(lstatSync(leasePath, { bigint: true }));
  const token = randomBytes(32).toString("hex");
  const markerPath = path.join(leasePath, LEASE_MARKER);
  const markerText = leaseMarkerText(process.pid, token);
  try {
    writeFileSync(markerPath, markerText, { encoding: "utf8", flag: "wx", mode: 0o600 });
  } catch (cause) {
    try {
      assertSameFileIdentity(
        leasePath,
        identity,
        input,
        "lease directory identity changed during marker creation",
      );
      rmdirSync(leasePath);
    } catch {
      /* Leave an ambiguous or nonempty lease untouched. */
    }
    throw cause;
  }
  const markerIdentity = fileIdentity(lstatSync(markerPath, { bigint: true }));
  return Object.freeze({
    leasePath, markerPath, identity, markerIdentity, token, markerText,
  });
}

function releaseLease(input, lease) {
  assertSameFileIdentity(lease.leasePath, lease.identity, input, "lease directory identity changed");
  assertSameFileIdentity(lease.markerPath, lease.markerIdentity, input, "lease marker identity changed");
  const markerStat = lstatSync(lease.markerPath);
  if (!markerStat.isFile() || markerStat.isSymbolicLink() ||
      readFileSync(lease.markerPath, "utf8") !== lease.markerText) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-CLEANUP",
      `${input.id} lease marker no longer proves ownership`,
    );
  }
  rmSync(lease.markerPath, { force: false });
  rmdirSync(lease.leasePath);
}

function recoverProvablyStaleLease(input, leasePath) {
  let directoryStat;
  let markerStat;
  let markerText;
  let marker;
  const markerPath = path.join(leasePath, LEASE_MARKER);
  try {
    directoryStat = lstatSync(leasePath, { bigint: true });
    markerStat = lstatSync(markerPath, { bigint: true });
    if (!directoryStat.isDirectory() || directoryStat.isSymbolicLink() ||
        !markerStat.isFile() || markerStat.isSymbolicLink() ||
        readdirSync(leasePath).length !== 1 || readdirSync(leasePath)[0] !== LEASE_MARKER) {
      return false;
    }
    markerText = readFileSync(markerPath, "utf8");
    marker = JSON.parse(markerText);
  } catch {
    return false;
  }
  if (marker?.kind !== "browsergrad-corpus-lease-owner" || marker.version !== 1 ||
      !Number.isSafeInteger(marker.pid) || marker.pid <= 0 ||
      typeof marker.token !== "string" || !/^[0-9a-f]{64}$/u.test(marker.token) ||
      markerText !== leaseMarkerText(marker.pid, marker.token)) {
    return false;
  }

  const directoryIdentity = fileIdentity(directoryStat);
  const markerIdentity = fileIdentity(markerStat);
  const leaseFd = openSync(leasePath, fsConstants.O_RDONLY | fsConstants.O_DIRECTORY);
  const rootFd = openSync(input.root, fsConstants.O_RDONLY | fsConstants.O_DIRECTORY);
  try {
    const helper = `
import json, os, sys
expected = json.load(sys.stdin)
opened = os.fstat(3)
if (str(opened.st_dev), str(opened.st_ino)) != (expected["directory"]["device"], expected["directory"]["inode"]):
    sys.exit(20)
named = os.stat(expected["basename"], dir_fd=4, follow_symlinks=False)
if (str(named.st_dev), str(named.st_ino)) != (expected["directory"]["device"], expected["directory"]["inode"]):
    sys.exit(20)
try:
    os.kill(expected["pid"], 0)
    sys.exit(21)
except ProcessLookupError:
    pass
except PermissionError:
    sys.exit(21)
os.fchdir(3)
if os.listdir(".") != [expected["markerName"]]:
    sys.exit(22)
item = os.stat(expected["markerName"], dir_fd=3, follow_symlinks=False)
if (str(item.st_dev), str(item.st_ino)) != (expected["marker"]["device"], expected["marker"]["inode"]):
    sys.exit(22)
descriptor = os.open(expected["markerName"], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=3)
try:
    content = os.read(descriptor, 4096).decode()
finally:
    os.close(descriptor)
if content != expected["markerText"]:
    sys.exit(22)
os.unlink(expected["markerName"], dir_fd=3)
os.rmdir(expected["basename"], dir_fd=4)
`;
    const result = spawnSync(input.hostToolchain.python.executable, ["-I", "-c", helper], {
      cwd: input.root,
      encoding: "utf8",
      env: safeHelperEnvironment(),
      input: JSON.stringify({
        basename: path.basename(leasePath),
        directory: directoryIdentity,
        marker: markerIdentity,
        markerName: LEASE_MARKER,
        markerText,
        pid: marker.pid,
      }),
      shell: false,
      stdio: ["pipe", "pipe", "pipe", leaseFd, rootFd],
    });
    return result.error === undefined && result.status === 0;
  } finally {
    closeSync(leaseFd);
    closeSync(rootFd);
  }
}

function leaseMarkerText(pid, token) {
  return `${JSON.stringify({
    kind: "browsergrad-corpus-lease-owner",
    version: 1,
    pid,
    token,
  })}\n`;
}

function reserveDestination(input) {
  assertRootPin(input);
  if (lstatIfPresent(input.path) !== undefined) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-TARGET-APPEARED",
      `${input.id} target appeared before its owned reservation: ${input.displayPath}`,
    );
  }
  const stagingPath = path.join(
    input.root,
    `.browsergrad-corpus-reservation-${randomBytes(24).toString("hex")}`,
  );
  mkdirSync(stagingPath, { mode: 0o700 });
  const identity = fileIdentity(lstatSync(stagingPath, { bigint: true }));
  const token = randomBytes(32).toString("hex");
  const markerPath = path.join(stagingPath, RESERVATION_MARKER);
  try {
    writeFileSync(markerPath, `${token}\n`, { encoding: "utf8", flag: "wx", mode: 0o600 });
  } catch (cause) {
    try {
      assertSameFileIdentity(
        stagingPath,
        identity,
        input,
        "reserved target identity changed during marker creation",
      );
      rmdirSync(stagingPath);
    } catch {
      /* Leave an ambiguous or nonempty reservation untouched. */
    }
    throw cause;
  }
  const markerIdentity = fileIdentity(lstatSync(markerPath, { bigint: true }));
  const fd = openSync(stagingPath, fsConstants.O_RDONLY | fsConstants.O_DIRECTORY);
  const fdIdentity = fileIdentity(fstatSync(fd, { bigint: true }));
  if (fdIdentity.device !== identity.device || fdIdentity.inode !== identity.inode) {
    closeSync(fd);
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT",
      `${input.id} owned staging reservation changed before its handle was opened`,
    );
  }
  return {
    fd,
    identity,
    token,
    markerIdentity,
    markerRelative: RESERVATION_MARKER,
    stagingPath,
    installed: false,
  };
}

function inputForOwnedReservation(input, reservation) {
  return Object.freeze({
    ...input,
    path: reservation.stagingPath,
    canonicalPath: input.path,
    gitPath: ".",
    gitExtraFd: reservation.fd,
  });
}

function moveReservationMarkerIntoGitDirectory(input, reservation) {
  const next = path.join(".git", RESERVATION_MARKER);
  runOwnedReservationHelper(input, reservation, "rename", next);
  reservation.markerRelative = next;
}

function installOwnedReservation(input, reservation) {
  assertReservationHandleIdentity(input, reservation);
  const result = runAtomicNoReplaceRename(input, reservation, input.path);
  if (result.error) throw result.error;
  if (result.status !== 0) {
    if (result.status === 17 || result.status === 39 ||
        lstatIfPresent(input.path) !== undefined) {
      fail(
        "BG-CUDA-LITE-CORPUS-PROVISION-TARGET-APPEARED",
        `${input.id} target appeared before atomic reservation installation: ${input.displayPath}`,
      );
    }
    throw new Error(
      `atomic no-replace checkout installation failed (${result.status}): ${result.stderr.trim()}`,
    );
  }
  reservation.installed = true;
  assertSameFileIdentity(
    input.path,
    reservation.identity,
    input,
    "installed checkout does not name its owned reservation",
  );
}

function completeReservation(input, reservation) {
  runOwnedReservationHelper(input, reservation, "unlink", "");
}

function abandonOwnedReservation(input, reservation) {
  if (reservation.markerRelative.endsWith(FAILED_RESERVATION_MARKER)) return;
  const parent = path.dirname(reservation.markerRelative);
  const next = parent === "."
    ? FAILED_RESERVATION_MARKER
    : path.join(parent, FAILED_RESERVATION_MARKER);
  runOwnedReservationHelper(input, reservation, "rename", next);
  reservation.markerRelative = next;
}

function assertReservationHandleIdentity(input, reservation) {
  const current = fileIdentity(fstatSync(reservation.fd, { bigint: true }));
  if (current.device !== reservation.identity.device ||
      current.inode !== reservation.identity.inode) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-CLEANUP",
      `${input.id} owned reservation directory handle identity changed`,
    );
  }
}

function runOwnedReservationHelper(input, reservation, operation, nextRelative) {
  assertReservationHandleIdentity(input, reservation);
  const helper = `
import os, stat, sys
operation, marker, next_marker, expected_dev, expected_ino, token = sys.argv[1:]
os.fchdir(3)
root = os.fstat(3)
if str(root.st_dev) != expected_dev or str(root.st_ino) != expected_ino:
    raise RuntimeError("reservation directory identity changed")
item = os.lstat(marker)
if not stat.S_ISREG(item.st_mode) or str(item.st_dev) != ${JSON.stringify(reservation.markerIdentity.device)} or str(item.st_ino) != ${JSON.stringify(reservation.markerIdentity.inode)}:
    raise RuntimeError("reservation marker identity changed")
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
marker_fd = os.open(marker, flags)
try:
    data = os.read(marker_fd, 4096)
finally:
    os.close(marker_fd)
if data != (token + "\\n").encode():
    raise RuntimeError("reservation marker token changed")
if operation == "rename":
    os.rename(marker, next_marker)
elif operation == "unlink":
    os.unlink(marker)
else:
    raise RuntimeError("unknown reservation helper operation")
`;
  const result = spawnSync(input.hostToolchain.python.executable, [
    "-I", "-c", helper,
    operation,
    reservation.markerRelative,
    nextRelative,
    reservation.identity.device,
    reservation.identity.inode,
    reservation.token,
  ], {
    cwd: input.root,
    encoding: "utf8",
    env: safeHelperEnvironment(),
    shell: false,
    stdio: ["ignore", "pipe", "pipe", reservation.fd],
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-UNSAFE-CLEANUP",
      `${input.id} owned reservation helper failed: ${result.stderr.trim()}`,
    );
  }
}

function runAtomicNoReplaceRename(input, reservation, destination) {
  const helper = `
import ctypes, os, sys
source, destination, expected_dev, expected_ino = sys.argv[1:]
opened = os.fstat(3)
named = os.lstat(source)
if (str(opened.st_dev), str(opened.st_ino)) != (expected_dev, expected_ino):
    raise RuntimeError("opened reservation identity changed")
if (str(named.st_dev), str(named.st_ino)) != (expected_dev, expected_ino):
    raise RuntimeError("named reservation no longer matches opened reservation")
libc = ctypes.CDLL(None, use_errno=True)
if sys.platform == "darwin":
    call = libc.renamex_np
    call.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    result = call(os.fsencode(source), os.fsencode(destination), 0x00000004)
elif sys.platform.startswith("linux"):
    call = libc.renameat2
    call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    result = call(-100, os.fsencode(source), -100, os.fsencode(destination), 0x00000001)
else:
    raise RuntimeError("unsupported atomic no-replace platform: " + sys.platform)
if result != 0:
    code = ctypes.get_errno()
    sys.stderr.write(os.strerror(code))
    sys.exit(code if 0 < code < 126 else 125)
`;
  return spawnSync(input.hostToolchain.python.executable, [
    "-I", "-c", helper,
    reservation.stagingPath,
    destination,
    reservation.identity.device,
    reservation.identity.inode,
  ], {
    encoding: "utf8",
    env: safeHelperEnvironment(),
    shell: false,
    stdio: ["ignore", "pipe", "pipe", reservation.fd],
  });
}

function runGitMutation(input, args, stdio) {
  const result = spawnGit(input, args, {
    encoding: stdio === "pipe" ? "utf8" : undefined,
    stdio,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const detail = stdio === "pipe" && result.stderr?.trim().length > 0
      ? `: ${result.stderr.trim()}`
      : "";
    throw new Error(`git ${args.join(" ")} exited ${result.status}${detail}`);
  }
}

function gitCapture(input, args, options = {}) {
  const result = spawnGit(input, args, {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (options.returnResult === true) {
    return Object.freeze({
      status: result.status,
      stdout: result.stdout ?? "",
      stderr: result.stderr ?? "",
    });
  }
  if (result.status !== 0 && !(options.allowStatusOne === true && result.status === 1)) {
    throw new Error(
      `git ${args.join(" ")} exited ${result.status}: ${result.stderr?.trim() ?? ""}`,
    );
  }
  if (options.allowStatusOne === true) {
    return Object.freeze({ status: result.status, stdout: result.stdout ?? "" });
  }
  return result.stdout ?? "";
}

function spawnGit(input, args, options) {
  const protocol = input.allowLocalFixtureRepo ? "file" : "https";
  const safeArgs = [
    "--no-replace-objects",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "-c", "core.autocrlf=false",
    "-c", "protocol.allow=never",
    "-c", `protocol.${protocol}.allow=always`,
    ...args,
  ];
  let command = input.hostToolchain.git.executable;
  let commandArgs = safeArgs;
  let stdio = options.stdio;
  if (input.gitExtraFd !== undefined) {
    command = input.hostToolchain.python.executable;
    commandArgs = [
      "-I", "-c",
      `import os,sys; os.fchdir(3); ` +
        `os.execve(${JSON.stringify(input.hostToolchain.git.executable)}, ` +
        `[${JSON.stringify(input.hostToolchain.git.executable)}, *sys.argv[1:]], os.environ)`,
      ...safeArgs,
    ];
    stdio = stdio === "inherit"
      ? ["inherit", "inherit", "inherit", input.gitExtraFd]
      : ["ignore", "pipe", "pipe", input.gitExtraFd];
  }
  return spawnSync(command, commandArgs, {
    cwd: input.root,
    env: safeGitEnvironment(),
    shell: false,
    ...options,
    ...(stdio === undefined ? {} : { stdio }),
  });
}

function resolveHostExecutable(basename, overrideEnvironmentVariable) {
  const override = process.env[overrideEnvironmentVariable];
  if (override !== undefined) {
    return pinHostExecutable(override, overrideEnvironmentVariable, true);
  }
  for (const directory of (process.env.PATH ?? "").split(path.delimiter)) {
    if (!path.isAbsolute(directory) || hasControl(directory)) continue;
    const pinned = pinHostExecutable(
      path.join(directory, basename),
      "absolute-path-entry",
      false,
    );
    if (pinned !== undefined) return pinned;
  }
  fail(
    "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN",
    `could not resolve executable ${basename} from absolute PATH entries; ` +
      `set ${overrideEnvironmentVariable} to an explicit absolute executable path`,
  );
}

function pinHostExecutable(candidate, selection, required) {
  if (typeof candidate !== "string" || !path.isAbsolute(candidate) || hasControl(candidate)) {
    if (!required) return undefined;
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN",
      `${selection} must name an explicit absolute executable path`,
    );
  }
  try {
    const executable = realpathSync(candidate);
    const executableStat = statSync(executable);
    if (!executableStat.isFile()) throw new Error("resolved path is not a regular file");
    accessSync(executable, fsConstants.X_OK);
    return Object.freeze({ executable, selection });
  } catch (cause) {
    if (!required) return undefined;
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-HOST-TOOLCHAIN",
      `${selection} does not resolve to an executable regular file: ${candidate}`,
      { cause },
    );
  }
}

function hostProbeFailureDetail(result) {
  if (result.error !== undefined) return result.error.message;
  const output = (result.stderr?.trim() || result.stdout?.trim() || "no diagnostic output")
    .slice(0, 512);
  return `status ${String(result.status)} (${output})`;
}

function safeGitEnvironment() {
  const environment = { ...process.env };
  for (const key of Object.keys(environment)) {
    if (key.startsWith("GIT_") || key.startsWith("PYTHON")) delete environment[key];
  }
  for (const key of [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
  ]) delete environment[key];
  delete environment[GIT_EXECUTABLE_ENV];
  delete environment[PYTHON_EXECUTABLE_ENV];
  deleteDynamicLoaderEnvironment(environment);
  environment.GIT_CONFIG_NOSYSTEM = "1";
  environment.GIT_CONFIG_GLOBAL = "/dev/null";
  environment.GIT_ATTR_NOSYSTEM = "1";
  environment.GIT_TERMINAL_PROMPT = "0";
  environment.GIT_OPTIONAL_LOCKS = "0";
  environment.GIT_NO_REPLACE_OBJECTS = "1";
  return environment;
}

function safeHelperEnvironment() {
  const environment = { ...process.env };
  for (const key of Object.keys(environment)) {
    if (key.startsWith("PYTHON")) delete environment[key];
  }
  delete environment[GIT_EXECUTABLE_ENV];
  delete environment[PYTHON_EXECUTABLE_ENV];
  deleteDynamicLoaderEnvironment(environment);
  return environment;
}

function deleteDynamicLoaderEnvironment(environment) {
  for (const key of Object.keys(environment)) {
    if (key.startsWith("DYLD_") || key === "LD_PRELOAD" ||
        key === "LD_LIBRARY_PATH" || key === "LD_AUDIT" ||
        key === "LIBPATH" || key === "SHLIB_PATH") {
      delete environment[key];
    }
  }
}

function gitCheckoutPath(input) {
  return input.gitPath ?? input.path;
}

function gitBlobObjectId(bytes) {
  const hash = createHash("sha1");
  hash.update(`blob ${bytes.byteLength}\0`);
  hash.update(bytes);
  return hash.digest("hex");
}

function dirty(input, detail) {
  fail(
    "BG-CUDA-LITE-CORPUS-PROVISION-DIRTY",
    `${input.id} checkout at ${input.displayPath} differs from exact HEAD bytes:\n${detail}`,
  );
}

function safeRelativePath(value) {
  if (value.length === 0 || path.isAbsolute(value) || value.includes("\0")) return false;
  return value.split("/").every((segment) =>
    segment.length > 0 && segment !== "." && segment !== "..");
}

function compareCodeUnits(left, right) {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function fileIdentity(stat) {
  return Object.freeze({ device: String(stat.dev), inode: String(stat.ino) });
}

function sameBigIntIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino &&
    left.size === right.size && left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

function assertSameFileIdentity(target, expected, input, detail) {
  let current;
  try {
    current = fileIdentity(lstatSync(target, { bigint: true }));
  } catch (cause) {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT",
      `${input.id} ${detail}`,
      { cause },
    );
  }
  if (current.device !== expected.device || current.inode !== expected.inode) {
    fail("BG-CUDA-LITE-CORPUS-PROVISION-ROOT-DRIFT", `${input.id} ${detail}`);
  }
}

function lstatIfPresent(target) {
  try {
    return lstatSync(target, { bigint: true });
  } catch (error) {
    if (error?.code === "ENOENT") return undefined;
    throw error;
  }
}

function nonemptyLines(value) {
  return value.split(/\r?\n/u).filter((line) => line.length > 0);
}

function normalizeGitStdio(value) {
  if (value !== "inherit" && value !== "pipe") {
    fail(
      "BG-CUDA-LITE-CORPUS-PROVISION-INVALID",
      "gitStdio must be inherit or pipe",
    );
  }
  return value;
}

function hasControl(value) {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}

function fail(code, message, options) {
  throw new CudaLiteCorpusProvisioningError(code, message, options);
}
