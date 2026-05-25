/**
 * @unlocalhosted/browsergrad-grad — public surface.
 *
 * ```ts
 * import { createSession } from "@unlocalhosted/browsergrad-runtime";
 * import { installGrad } from "@unlocalhosted/browsergrad-grad";
 *
 * const session = await createSession({ pyodideIndexURL: "/pyodide/v0.26.4/", packages: ["numpy"] });
 * await installGrad(session);
 *
 * await session.exec({
 *   code: `
 *     import browsergrad_grad as grad
 *     x = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
 *     y = (x * x).sum()
 *     y.backward()
 *     print(x.grad.tolist())   # [2.0, 4.0, 6.0]
 *   `,
 * });
 * ```
 *
 * Works with any Pyodide-shaped target that has `exec({code})` — not just
 * the runtime package's Session.
 */

export { installGrad } from "./install.js";
export type { GradTarget, InstallOptions } from "./types.js";
export { GradInstallError } from "./types.js";

// The Python source itself is exported at the `./source` subpath
// for tools that want to install grad through their own Pyodide bootstrap.
