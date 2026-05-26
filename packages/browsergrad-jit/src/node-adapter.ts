/**
 * `createNodePyodideTarget` — wraps a Pyodide instance as a `JitTarget`.
 *
 * Use this in Node contexts (CI scripts, server-side smoke tests, the
 * jit integration test suite) where you've called `loadPyodide()` yourself
 * and want to call `installJit` against the resulting Pyodide handle
 * without going through the runtime's Worker-backed Session.
 *
 *     import { loadPyodide } from "pyodide";
 *     import { installJit } from "@unlocalhosted/browsergrad-jit";
 *     import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-jit/node-adapter";
 *
 *     const py = await loadPyodide();
 *     await py.loadPackage(["numpy"]);
 *     await installJit(createNodePyodideTarget(py));
 *
 * Mirrors browsergrad-grad's adapter exactly — intentional, so consumers
 * can swap install paths without re-learning the API.
 */

import type { JitTarget } from "./types.js";

/**
 * Minimal structural shape we need from a Pyodide instance. Declared
 * locally so this module has no type-level dependency on the `pyodide`
 * package — the factory accepts anything that quacks this way, mirroring
 * how `JitTarget` itself is duck-typed.
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

export function createNodePyodideTarget(pyodide: PyodideInterface): JitTarget {
  if (typeof pyodide?.runPythonAsync !== "function") {
    throw new TypeError(
      "createNodePyodideTarget: expected a Pyodide instance with " +
        "`runPythonAsync` — pass the value returned by `await loadPyodide()`.",
    );
  }
  if (
    typeof pyodide?.FS?.writeFile !== "function" ||
    typeof pyodide?.FS?.mkdirTree !== "function"
  ) {
    throw new TypeError(
      "createNodePyodideTarget: this Pyodide instance is missing FS.writeFile " +
        "or FS.mkdirTree. Make sure you're passing the real Pyodide API object " +
        "(not a wrapper) — these methods are always present on the value from " +
        "`loadPyodide()`.",
    );
  }
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
