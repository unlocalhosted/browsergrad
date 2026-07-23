import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { lstat, open, realpath, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  cppCuteBrowserReproducibilityResourceBytes,
  verifyCppCuteBrowserReproducibilityResource,
} from "../../dist/cpp_cute_browser_reproducibility.js";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER";
const EVIDENCE_MARKER = "BROWSERGRAD_CPP_CUTE_REAL_COMPILE_EVIDENCE=";
const ANSI_COLOR_PATTERN = new RegExp(
  `${String.fromCharCode(27)}\\[[0-9;]*m`,
  "gu",
);
const MAX_CAPTURED_OUTPUT_BYTES = 8 * 1024 * 1024;
const PACK_FILES = Object.freeze({
  "clang-resource": "clang-resource.headers.bgvfs",
  cuda: "cuda-12.6.3.headers.bgvfs",
  cutlass: "cutlass-3.7.0.headers.bgvfs",
  "cxx-stdlib": "libcxx-22.1.8.headers.bgvfs",
  "linux-sysroot": "linux-sysroot.headers.bgvfs",
});
const scriptRoot = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(scriptRoot, "../..");

export class CppCuteBrowserRealCompileRunnerError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserRealCompileRunnerError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export async function preflightCppCuteBrowserRealCompileInputs(input) {
  const values = exactInput(input);
  const reproducibility = await verifyCppCuteBrowserReproducibilityResource(
    cppCuteBrowserReproducibilityResourceBytes(),
  );
  const wasmPath = await canonicalRegularFile(values.wasmPath, "$.wasmPath");
  const packRoot = await canonicalDirectory(values.packRoot, "$.packRoot");
  const packAssetRoot = join(packRoot, "assets", "browsergrad-cpp-cute");
  await canonicalDirectory(packAssetRoot, "$.packRoot.assets");
  const entries = [
    ["clang-wasm", wasmPath, "application/wasm"],
    ...Object.entries(PACK_FILES).map(([includeRootId, fileName]) => [
      includeRootId,
      join(packAssetRoot, fileName),
      "application/vnd.browsergrad.vfs-pack.v1",
    ]),
  ];
  const assets = await Promise.all(entries.map(async ([assetId, path, mediaType], index) => {
    const canonicalPath = await canonicalRegularFile(path, `$.assets[${index}]`);
    const observed = await hashRegularFile(canonicalPath, `$.assets[${index}]`);
    return Object.freeze({
      assetId,
      path: canonicalPath,
      mediaType,
      sha256: observed.sha256,
      byteLength: observed.byteLength,
    });
  }));
  const wasm = assets[0];
  if (wasm === undefined) {
    invalid("$.assets[0]", "Clang-Wasm observation is missing");
  }
  const pinnedReproducibleWasmMatched =
    wasm.sha256 === reproducibility.wasmSha256 &&
    wasm.byteLength === reproducibility.wasmByteLength;
  if (!pinnedReproducibleWasmMatched && values.requireCompiled) {
    invalid(
      "$.assets[0]",
      "strict compile requires the exact package-pinned two-clean-build Wasm",
    );
  }
  if (!pinnedReproducibleWasmMatched &&
      !values.allowUntrustedDiagnosticWasm) {
    invalid(
      "$.assets[0]",
      "Clang-Wasm differs from the exact package-pinned two-clean-build output",
    );
  }
  return Object.freeze({
    schema: "browsergrad.compiler.cpp-cute.browser-real-compile-inputs",
    version: 2,
    authority: "local-exact-byte-preflight-only",
    wasmPath,
    packRoot,
    assets: Object.freeze(assets),
    wasmAuthority: pinnedReproducibleWasmMatched
      ? "package-pinned-two-clean-build-output"
      : "untrusted-diagnostic-local-byte-observation-only",
    pinnedReproducibleWasmMatched,
    untrustedDiagnosticWasm: !pinnedReproducibleWasmMatched,
    headerDistributionLicenseApproved: false,
    producerTrusted: false,
    workerExecutionObserved: false,
    releaseReady: false,
  });
}

export async function runCppCuteBrowserRealCompile(argv = process.argv.slice(2)) {
  const options = parseArguments(argv);
  const preflight = await preflightCppCuteBrowserRealCompileInputs(options);
  process.stdout.write(`${JSON.stringify(preflight)}\n`);
  if (options.preflightOnly) return preflight;

  const child = spawn(
    "pnpm",
    [
      "exec",
      "vitest",
      "run",
      "--config",
      "vitest.browser.cpp-cute-real-compile.config.ts",
    ],
    {
      cwd: packageRoot,
      env: {
        ...process.env,
        BG_BROWSER_HEADLESS: process.env.BG_BROWSER_HEADLESS ?? "1",
        BG_CPP_CUTE_REAL_COMPILE_INPUTS: JSON.stringify(preflight),
        NO_COLOR: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  let captured = "";
  const capture = (chunk) => {
    const text = String(chunk);
    process.stdout.write(text);
    if (captured.length < MAX_CAPTURED_OUTPUT_BYTES) {
      captured += text.slice(0, MAX_CAPTURED_OUTPUT_BYTES - captured.length);
    }
  };
  child.stdout.on("data", capture);
  child.stderr.on("data", (chunk) => {
    const text = String(chunk);
    process.stderr.write(text);
    if (captured.length < MAX_CAPTURED_OUTPUT_BYTES) {
      captured += text.slice(0, MAX_CAPTURED_OUTPUT_BYTES - captured.length);
    }
  });
  const exitCode = await new Promise((resolveExit, reject) => {
    child.once("error", reject);
    child.once("close", (code, signal) => {
      if (signal !== null) {
        reject(new CppCuteBrowserRealCompileRunnerError(
          "$.browser",
          `browser verifier terminated by ${signal}`,
        ));
        return;
      }
      resolveExit(code ?? 1);
    });
  });
  if (exitCode !== 0) {
    invalid("$.browser", `real browser compile verifier exited with status ${exitCode}`);
  }
  const evidence = parseEvidence(captured);
  if (options.requireCompiled &&
      (evidence.outcome !== "compiled" ||
       !preflight.pinnedReproducibleWasmMatched)) {
    invalid(
      "$.evidence.outcome",
      "strict compile requires a package-pinned Wasm and one compiled Artifact V3 observation",
    );
  }
  if (options.evidenceOutput !== undefined) {
    const outputPath = await canonicalOutputPath(options.evidenceOutput);
    await writeFile(outputPath, `${JSON.stringify(evidence, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    process.stdout.write(`Browser compile evidence: ${outputPath}\n`);
  }
  return evidence;
}

function parseArguments(argv) {
  const values = {};
  const separatorCount = argv.reduce(
    (count, argument) => count + (argument === "--" ? 1 : 0),
    0,
  );
  if (separatorCount > 1) invalid("$.argv", "duplicate argument separator");
  const forwarded = argv.filter((argument) => argument !== "--");
  for (const argument of forwarded) {
    if (argument === "--preflight-only") {
      if (values.preflightOnly === true) invalid("$.argv", "duplicate --preflight-only");
      values.preflightOnly = true;
      continue;
    }
    if (argument === "--require-compiled") {
      if (values.requireCompiled === true) invalid("$.argv", "duplicate --require-compiled");
      values.requireCompiled = true;
      continue;
    }
    if (argument === "--allow-untrusted-diagnostic-wasm") {
      if (values.allowUntrustedDiagnosticWasm === true) {
        invalid("$.argv", "duplicate --allow-untrusted-diagnostic-wasm");
      }
      values.allowUntrustedDiagnosticWasm = true;
      continue;
    }
    const separator = argument.indexOf("=");
    if (!argument.startsWith("--") || separator < 3) {
      invalid("$.argv", `unsupported argument ${JSON.stringify(argument)}`);
    }
    const name = argument.slice(2, separator);
    const value = argument.slice(separator + 1);
    if (value.length === 0 || Object.hasOwn(values, name)) {
      invalid("$.argv", `missing or duplicate value for --${name}`);
    }
    if (name === "wasm") values.wasmPath = value;
    else if (name === "pack-root") values.packRoot = value;
    else if (name === "evidence-output") values.evidenceOutput = value;
    else invalid("$.argv", `unsupported option --${name}`);
  }
  return exactInput(values);
}

function exactInput(value) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    invalid("$.input", "expected one plain input record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const allowed = new Set([
    "wasmPath",
    "packRoot",
    "evidenceOutput",
    "preflightOnly",
    "requireCompiled",
    "allowUntrustedDiagnosticWasm",
  ]);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== "string" || !allowed.has(key)) {
      invalid("$.input", "input contains an unknown field");
    }
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor) || descriptor.enumerable !== true) {
      invalid(`$.input.${String(key)}`, "input fields must be enumerable data properties");
    }
  }
  const wasmPath = requiredAbsolutePath(descriptors.wasmPath?.value, "$.wasmPath");
  const packRoot = requiredAbsolutePath(descriptors.packRoot?.value, "$.packRoot");
  const evidenceOutput = descriptors.evidenceOutput === undefined ||
      descriptors.evidenceOutput.value === undefined
    ? undefined
    : requiredAbsolutePath(descriptors.evidenceOutput.value, "$.evidenceOutput");
  const preflightOnly = descriptors.preflightOnly?.value ?? false;
  if (typeof preflightOnly !== "boolean") invalid("$.preflightOnly", "expected boolean");
  const requireCompiled = descriptors.requireCompiled?.value ?? false;
  if (typeof requireCompiled !== "boolean") {
    invalid("$.requireCompiled", "expected boolean");
  }
  const allowUntrustedDiagnosticWasm =
    descriptors.allowUntrustedDiagnosticWasm?.value ?? false;
  if (typeof allowUntrustedDiagnosticWasm !== "boolean") {
    invalid("$.allowUntrustedDiagnosticWasm", "expected boolean");
  }
  return {
    wasmPath,
    packRoot,
    evidenceOutput,
    preflightOnly,
    requireCompiled,
    allowUntrustedDiagnosticWasm,
  };
}

function requiredAbsolutePath(value, path) {
  if (typeof value !== "string" || value.length === 0 || !isAbsolute(value)) {
    invalid(path, "expected a non-empty absolute path");
  }
  return value;
}

async function canonicalDirectory(path, diagnosticPath) {
  let metadata;
  let canonical;
  try {
    [metadata, canonical] = await Promise.all([lstat(path), realpath(path)]);
  } catch (cause) {
    invalid(diagnosticPath, "directory is unavailable", { cause });
  }
  if (!metadata.isDirectory() || metadata.isSymbolicLink() || canonical !== path) {
    invalid(diagnosticPath, "expected one canonical non-symlink directory");
  }
  return canonical;
}

async function canonicalRegularFile(path, diagnosticPath) {
  let metadata;
  let canonical;
  try {
    [metadata, canonical] = await Promise.all([lstat(path), realpath(path)]);
  } catch (cause) {
    invalid(diagnosticPath, "file is unavailable", { cause });
  }
  if (!metadata.isFile() || metadata.isSymbolicLink() || canonical !== path) {
    invalid(diagnosticPath, "expected one canonical non-symlink regular file");
  }
  return canonical;
}

async function hashRegularFile(path, diagnosticPath) {
  let handle;
  try {
    handle = await open(path, "r");
    const before = await handle.stat();
    if (!before.isFile() || before.size <= 0) {
      invalid(diagnosticPath, "input must be one non-empty regular file");
    }
    const digest = createHash("sha256");
    const stream = handle.createReadStream({ autoClose: false });
    for await (const chunk of stream) digest.update(chunk);
    const after = await handle.stat();
    if (before.dev !== after.dev || before.ino !== after.ino ||
        before.size !== after.size || before.mtimeMs !== after.mtimeMs) {
      invalid(diagnosticPath, "input identity changed while it was hashed");
    }
    return { sha256: digest.digest("hex"), byteLength: before.size };
  } catch (cause) {
    if (cause instanceof CppCuteBrowserRealCompileRunnerError) throw cause;
    invalid(diagnosticPath, "input could not be read exactly", { cause });
  } finally {
    await handle?.close();
  }
}

async function canonicalOutputPath(path) {
  const parent = dirname(path);
  await canonicalDirectory(parent, "$.evidenceOutput.parent");
  try {
    await stat(path);
    invalid("$.evidenceOutput", "refusing to overwrite an existing evidence file");
  } catch (cause) {
    if (cause instanceof CppCuteBrowserRealCompileRunnerError) throw cause;
    if (cause?.code !== "ENOENT") {
      invalid("$.evidenceOutput", "evidence output could not be admitted", { cause });
    }
  }
  return path;
}

function parseEvidence(output) {
  const clean = output.replaceAll(ANSI_COLOR_PATTERN, "");
  const matches = clean.split(/\r?\n/u).filter((line) => line.includes(EVIDENCE_MARKER));
  if (matches.length !== 1) {
    invalid("$.evidence", `expected exactly one ${EVIDENCE_MARKER} record`);
  }
  const line = matches[0];
  const offset = line.indexOf(EVIDENCE_MARKER);
  try {
    const evidence = JSON.parse(line.slice(offset + EVIDENCE_MARKER.length));
    const compiled = evidence?.outcome === "compiled";
    const rejected = evidence?.outcome === "rejected";
    const blocked = evidence?.outcome === "blocked";
    const workerExecuted = compiled || rejected;
    if (evidence?.schema !== "browsergrad.compiler.cpp-cute.browser-real-compile-observation" ||
        evidence?.version !== 1 ||
        (!workerExecuted && !blocked) ||
        evidence?.workerExecutionObserved !== workerExecuted ||
        evidence?.authority !== (workerExecuted
          ? "local-real-browser-worker-execution-observation-only"
          : "local-real-browser-worker-terminal-observation-only") ||
        evidence?.inputs?.wasmAuthority !==
          (evidence?.inputs?.pinnedReproducibleWasmMatched === true
            ? "package-pinned-two-clean-build-output"
            : "untrusted-diagnostic-local-byte-observation-only") ||
        evidence?.inputs?.untrustedDiagnosticWasm !==
          !evidence?.inputs?.pinnedReproducibleWasmMatched ||
        evidence?.releaseReady !== false) {
      invalid("$.evidence", "browser evidence has an invalid authority boundary");
    }
    return evidence;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserRealCompileRunnerError) throw cause;
    invalid("$.evidence", "browser evidence is not valid JSON", { cause });
  }
}

function invalid(path, message, options) {
  throw new CppCuteBrowserRealCompileRunnerError(path, message, options);
}

if (process.argv[1] !== undefined &&
    import.meta.url === new URL(`file://${resolve(process.argv[1])}`).href) {
  runCppCuteBrowserRealCompile().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
