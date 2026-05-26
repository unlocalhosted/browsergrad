/**
 * @unlocalhosted/browsergrad-jit — public surface.
 *
 * The lazy-tracing successor to `@unlocalhosted/browsergrad-grad`. Same
 * PyTorch-flavored Python API on the user side; underneath, every op
 * builds a UOp IR node and defers computation until a realization
 * trigger (`.numpy()`, `.tolist()`, `.item()`, `.backward()`, et al.).
 * Designed as the substrate for downstream fusion (PRD-006), symbolic
 * backward (PRD-007), pipeline caching (PRD-008), WGSL/megakernel codegen
 * (PRD-010/012), function transforms (PRD-014), and ONNX export (PRD-016).
 *
 * ```ts
 * import { createSession } from "@unlocalhosted/browsergrad-runtime";
 * import { installJit } from "@unlocalhosted/browsergrad-jit";
 *
 * const session = await createSession({ pyodideIndexURL: "/pyodide/v0.26.4/", packages: ["numpy"] });
 * await installJit(session);
 *
 * await session.exec({
 *   code: `
 *     import browsergrad_jit as jit
 *     # The full TensorProxy API arrives across Weeks 2-6 of PRD-005.
 *     print(jit.__version__)
 *   `,
 * });
 * ```
 *
 * Works with any Pyodide-shaped target that has `exec({code})` — not just
 * the runtime package's Session.
 */

export { installJit } from "./install.js";
export type { JitTarget, InstallOptions } from "./types.js";
export { JitInstallError } from "./types.js";

// The Python source itself is exported at the `./source` subpath
// for tools that want to install jit through their own Pyodide bootstrap.
