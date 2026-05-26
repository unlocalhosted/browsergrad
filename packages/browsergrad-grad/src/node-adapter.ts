/**
 * `createNodePyodideTarget` — wraps a Pyodide instance as a `GradTarget`.
 *
 * Use this when you `loadPyodide()` yourself in Node (CI scripts, server-side
 * inference, docs site renderers) and want to call `installGrad` against it
 * without going through the runtime's Worker-backed Session. The browser-side
 * path is still `@unlocalhosted/browsergrad-runtime`'s Session.
 *
 *     import { loadPyodide } from "pyodide";
 *     import { installGrad } from "@unlocalhosted/browsergrad-grad";
 *     import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-grad/node-adapter";
 *
 *     const py = await loadPyodide();
 *     await py.loadPackage(["numpy"]);
 *     await installGrad(createNodePyodideTarget(py));
 *
 * The Adapter is intentionally narrow — it satisfies `GradTarget` and nothing
 * more. Test glue (PyProxy unwrap helpers, namespace clearing) stays in the
 * caller because that's a test-locality concern, not Adapter surface.
 */

import type { GradTarget } from "./types.js";

/**
 * Minimal structural shape we need from a Pyodide instance. Declared locally
 * so this module has no type-level dependency on the `pyodide` package — the
 * factory accepts anything that quacks this way, mirroring how `GradTarget`
 * itself is duck-typed.
 */
export interface PyodideInterface {
  runPythonAsync(code: string): Promise<unknown>;
  FS: {
    writeFile(path: string, content: string, opts?: { encoding?: string }): void;
    mkdirTree(path: string): void;
  };
}

function dirname(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx <= 0 ? "/" : path.slice(0, idx);
}

export function createNodePyodideTarget(pyodide: PyodideInterface): GradTarget {
  return {
    exec: async ({ code }) => pyodide.runPythonAsync(code),
    fs: {
      write: async (path, content) => {
        const dir = dirname(path);
        if (dir && dir !== "/") {
          pyodide.FS.mkdirTree(dir);
        }
        pyodide.FS.writeFile(path, content, { encoding: "utf8" });
      },
    },
  };
}
