/**
 * @unlocalhosted/browsergrad-runtime — public surface.
 *
 * ```ts
 * import { createSession } from "@unlocalhosted/browsergrad-runtime";
 *
 * const session = await createSession({
 *   pyodideIndexURL: "/pyodide/v0.26.4/",
 *   packages: ["numpy"],
 * });
 *
 * const result = await session.exec({
 *   code: `
 *     import browsergrad as bg
 *     bg.assert_pass("basic_math")
 *     bg.log("status", "all good")
 *   `,
 *   onAssertion: (a) => console.log(a),
 *   onArtifact: (a) => console.log(a),
 * });
 *
 * await session.dispose();
 * ```
 *
 * The public API of this package is exactly what this file re-exports.
 * Importing from any other path is unsupported.
 */

export type {
  Session,
  SessionFS,
  SessionOptions,
  PackageProgressEvent,
  ExecOptions,
  ExecResult,
  ExecError,
  ExecErrorKind,
  Assertion,
  AssertionPass,
  AssertionFail,
  AssertionError,
  Artifact,
  ArtifactLog,
  ArtifactJson,
  ArtifactImage,
} from "./types.js";

export { BrowsergradError } from "./types.js";
export { createSession } from "./client.js";

// Lab manifest + semver gate. For platforms that ship versioned content
// units and want to refuse mismatched runtime pins at boot.
export {
  parseManifest,
  isSemverCompatible,
  assertCompatibleRuntime,
  LabRuntimeMismatch,
  type LabManifest,
  type LabDataset,
  type ManifestParseResult,
} from "./lab.js";
