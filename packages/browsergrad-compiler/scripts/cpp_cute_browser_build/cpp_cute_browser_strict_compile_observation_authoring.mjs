import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  open,
  readFile,
  realpath,
  writeFile,
} from "node:fs/promises";
import {
  dirname,
  isAbsolute,
  resolve,
} from "node:path";
import {
  fileURLToPath,
  pathToFileURL,
} from "node:url";

import {
  canonicalJsonBytes,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserHeaderDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserHeaderDistributionReproducibilityResource,
} from "../../dist/cpp_cute_browser_header_distribution_reproducibility.js";
import {
  cppCuteBrowserReproducibilityResourceBytes,
  verifyCppCuteBrowserReproducibilityResource,
} from "../../dist/cpp_cute_browser_reproducibility.js";
import {
  verifyCppCuteBrowserWorkerBundle,
} from "../../dist/cpp_cute_browser_worker_bundle.js";
import {
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE,
} from "../../dist/resources/cpp_cute_browser_strict_compile_observation_v1.js";
import {
  CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
} from "../../dist/resources/cpp_cute_browser_wasm_verifier_bundle_v1.js";
import {
  prepareCppCuteBrowserRealCompileMatrix,
} from "./cpp_cute_browser_real_compile_matrix.mjs";

const ERROR_CODE =
  "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-OBSERVATION-AUTHORING";
const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIRECTORY = dirname(SCRIPT_PATH);
const PACKAGE_ROOT = resolve(SCRIPT_DIRECTORY, "../..");
const RESOURCE_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_strict_compile_observation_v1.ts",
);
const IDENTITY_PATH = resolve(
  PACKAGE_ROOT,
  "src/resources/cpp_cute_browser_strict_compile_observation_identity_v1.ts",
);
const MAX_INPUT_BYTE_LENGTH = 1_048_576;
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

export class CppCuteBrowserStrictCompileObservationAuthoringError
  extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserStrictCompileObservationAuthoringError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

export function parseCppCuteBrowserStrictCompileObservationAuthoringArguments(
  argv,
) {
  const arguments_ = argv[0] === "--" ? argv.slice(1) : argv;
  if (arguments_.length === 1 && arguments_[0] === "--check") {
    return Object.freeze({ check: true, inputPath: undefined });
  }
  if (arguments_.length === 1 && arguments_[0].startsWith("--input=")) {
    const inputPath = arguments_[0].slice("--input=".length);
    if (!isAbsolute(inputPath) || inputPath.includes("\0")) {
      invalid(
        "$.arguments[0]",
        "--input requires one absolute NUL-free path",
      );
    }
    return Object.freeze({ check: false, inputPath });
  }
  invalid(
    "$.arguments",
    "expected exactly --check or --input=/absolute/matrix.json",
  );
}

export async function projectCppCuteBrowserStrictCompileObservation(
  value,
) {
  const sourceRevision = value?.packageBinding?.matrixSourceRevision;
  const prepared = prepareCppCuteBrowserRealCompileMatrix(
    value?.cases,
    sourceRevision,
  );
  const inputBytes = canonicalJsonBytes(value);
  const preparedBytes = canonicalJsonBytes(prepared);
  if (!equalBytes(inputBytes, preparedBytes)) {
    invalid(
      "$.matrix",
      "matrix differs from the exact closed runner projection",
    );
  }
  if (prepared.claims.pinnedReproducibleWasmMatched !== true ||
      prepared.claims.untrustedDiagnosticWasm !== false) {
    invalid(
      "$.matrix.claims",
      "package authoring requires package-pinned reproducible Wasm",
    );
  }

  const [reproducibility, headers, worker] = await Promise.all([
    verifyCppCuteBrowserReproducibilityResource(
      cppCuteBrowserReproducibilityResourceBytes(),
    ),
    verifyCppCuteBrowserHeaderDistributionReproducibilityResource(
      cppCuteBrowserHeaderDistributionReproducibilityResourceBytes(),
    ),
    verifyCppCuteBrowserWorkerBundle(),
  ]);
  if (prepared.packageBinding.compilerWorkerSha256 !== worker.sha256 ||
      prepared.packageBinding.verifierWorkerSha256 !==
        CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256) {
    invalid(
      "$.matrix.packageBinding",
      "matrix does not bind the current package Worker pair",
    );
  }
  for (const [index, observed] of prepared.cases.entries()) {
    if (observed.inputs.wasmSha256 !== reproducibility.wasmSha256 ||
        observed.inputs.headerDistributionReproducibilityId !==
          headers.reproducibilityId ||
        observed.inputs.headerDistributionOutputVerificationId !==
          headers.outputVerificationId) {
      invalid(
        `$.matrix.cases[${index}].inputs`,
        "strict case does not bind current reproducibility and header authorities",
      );
    }
  }

  const resourceSha256 = digest(preparedBytes);
  return Object.freeze({
    schema:
      "browsergrad.compiler.cpp-cute.strict-compile-observation-authoring-projection",
    version: 1,
    authority: "package-authoring-projection-only",
    matrix: prepared,
    resourceSha256,
    resourceByteLength: preparedBytes.byteLength,
    sourceRevision,
    workerBundleSha256: worker.sha256,
    verifierWorkerBundleSha256:
      CPP_CUTE_BROWSER_WASM_VERIFIER_BUNDLE_V1_SHA256,
    wasmSha256: reproducibility.wasmSha256,
    wasmByteLength: reproducibility.wasmByteLength,
  });
}

export function renderCppCuteBrowserStrictCompileObservationResource(
  projection,
) {
  exactProjection(projection);
  const value = JSON.stringify(projection.matrix, null, 2);
  return `import {\n` +
    `  deepFreezeJson,\n` +
    `  type JsonObject,\n` +
    `} from "@unlocalhosted/browsergrad-semantic-core/schema";\n\n` +
    `/** Generated by cpp_cute_browser_strict_compile_observation_authoring.mjs. */\n` +
    `const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE = ${value} as const satisfies JsonObject;\n\n` +
    `export type CppCuteBrowserStrictCompileObservationV1Resource =\n` +
    `  typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE;\n\n` +
    `export const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE =\n` +
    `  deepFreezeJson(\n` +
    `    CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_VALUE,\n` +
    `  ) as unknown as CppCuteBrowserStrictCompileObservationV1Resource;\n`;
}

export function renderCppCuteBrowserStrictCompileObservationIdentity(
  projection,
) {
  exactProjection(projection);
  return `/** Generated by cpp_cute_browser_strict_compile_observation_authoring.mjs. */\n` +
    `export const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256 = ${JSON.stringify(projection.resourceSha256)};\n` +
    `export const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH = ${projection.resourceByteLength};\n` +
    `export const CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION = ${JSON.stringify(projection.sourceRevision)};\n` +
    `export const CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256 = ${JSON.stringify(projection.workerBundleSha256)};\n` +
    `export const CPP_CUTE_BROWSER_STRICT_COMPILE_VERIFIER_WORKER_BUNDLE_SHA256 = ${JSON.stringify(projection.verifierWorkerBundleSha256)};\n`;
}

async function readCanonicalMatrix(path) {
  let entry;
  try {
    entry = await lstat(path);
  } catch (cause) {
    invalid("$.input", "matrix input is unavailable", { cause });
  }
  if (!entry.isFile() || entry.isSymbolicLink() ||
      entry.size <= 0 || entry.size > MAX_INPUT_BYTE_LENGTH ||
      await realpath(path) !== path) {
    invalid(
      "$.input",
      "matrix input must be one canonical bounded non-symlink regular file",
    );
  }
  let handle;
  try {
    handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const before = await handle.stat();
    if (!before.isFile() || before.dev !== entry.dev ||
        before.ino !== entry.ino || before.size !== entry.size) {
      invalid("$.input", "matrix input identity changed before read");
    }
    const bytes = await handle.readFile();
    const after = await handle.stat();
    if (!after.isFile() || after.dev !== before.dev ||
        after.ino !== before.ino || after.size !== before.size ||
        after.mtimeMs !== before.mtimeMs || after.ctimeMs !== before.ctimeMs) {
      invalid("$.input", "matrix input changed during read");
    }
    let value;
    try {
      value = JSON.parse(UTF8_DECODER.decode(bytes));
    } catch (cause) {
      invalid("$.input", "matrix input is not strict UTF-8 JSON", { cause });
    }
    if (!equalBytes(bytes, canonicalJsonBytes(value))) {
      invalid("$.input", "matrix input is not canonical JSON");
    }
    return value;
  } finally {
    await handle?.close();
  }
}

function exactProjection(value) {
  if (value?.schema !==
        "browsergrad.compiler.cpp-cute.strict-compile-observation-authoring-projection" ||
      value.version !== 1 ||
      value.authority !== "package-authoring-projection-only" ||
      typeof value.resourceSha256 !== "string" ||
      !/^[0-9a-f]{64}$/u.test(value.resourceSha256) ||
      !Number.isSafeInteger(value.resourceByteLength) ||
      value.resourceByteLength <= 0 ||
      typeof value.sourceRevision !== "string" ||
      !/^[0-9a-f]{40}$/u.test(value.sourceRevision) ||
      typeof value.workerBundleSha256 !== "string" ||
      !/^[0-9a-f]{64}$/u.test(value.workerBundleSha256) ||
      typeof value.verifierWorkerBundleSha256 !== "string" ||
      !/^[0-9a-f]{64}$/u.test(value.verifierWorkerBundleSha256)) {
    invalid("$.projection", "expected one authentic bounded authoring projection");
  }
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function equalBytes(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= left[index] ^ right[index];
  }
  return difference === 0;
}

function invalid(path, message, options) {
  throw new CppCuteBrowserStrictCompileObservationAuthoringError(
    path,
    message,
    options,
  );
}

async function main() {
  const options =
    parseCppCuteBrowserStrictCompileObservationAuthoringArguments(
      process.argv.slice(2),
    );
  const matrix = options.check
    ? CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE
    : await readCanonicalMatrix(options.inputPath);
  const projection =
    await projectCppCuteBrowserStrictCompileObservation(matrix);
  const resource =
    renderCppCuteBrowserStrictCompileObservationResource(projection);
  const identity =
    renderCppCuteBrowserStrictCompileObservationIdentity(projection);
  if (options.check) {
    if (await readFile(RESOURCE_PATH, "utf8") !== resource ||
        await readFile(IDENTITY_PATH, "utf8") !== identity) {
      invalid("$.resources", "checked-in strict observation resources are stale");
    }
  } else {
    await writeFile(RESOURCE_PATH, resource, { encoding: "utf8", mode: 0o644 });
    await writeFile(IDENTITY_PATH, identity, { encoding: "utf8", mode: 0o644 });
  }
  process.stdout.write(`${JSON.stringify({
    schema: projection.schema,
    version: projection.version,
    authority: projection.authority,
    checked: options.check,
    resourceSha256: projection.resourceSha256,
    resourceByteLength: projection.resourceByteLength,
    sourceRevision: projection.sourceRevision,
    workerBundleSha256: projection.workerBundleSha256,
    verifierWorkerBundleSha256: projection.verifierWorkerBundleSha256,
    wasmSha256: projection.wasmSha256,
    wasmByteLength: projection.wasmByteLength,
  })}\n`);
}

if (process.argv[1] !== undefined &&
    import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((cause) => {
    const error = cause instanceof Error
      ? cause
      : new Error("unknown strict-observation authoring failure");
    process.stderr.write(`${error.name}: ${error.message}\n`);
    process.exitCode = 1;
  });
}
