/**
 * Pyodide-in-node host shared by every integration test.
 *
 * Boots Pyodide once per process, preloads numpy, installs the grad library
 * via our public `installGrad` API. Subsequent tests reuse the same Pyodide
 * instance — namespace is reset between tests via `clearNamespace`.
 */

import { loadPyodide } from "pyodide";
import { installGrad, type GradTarget } from "../src/index";

interface PyodideAPI {
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (names: readonly string[]) => Promise<void>;
  globals: { get: (name: string) => unknown };
}

let cached: Promise<PyodideAPI> | null = null;

export async function getPyodide(): Promise<PyodideAPI> {
  if (cached) return cached;
  cached = (async () => {
    // Node pyodide finds its assets via the installed package dir automatically.
    // Default packages directory is on a CDN — for offline / deterministic tests
    // we explicitly point at the local install.
    const py = await loadPyodide({
      // Suppress info-level logs that flood test output.
      stdout: () => {},
      stderr: () => {},
    });
    await py.loadPackage(["numpy"]);
    return py as unknown as PyodideAPI;
  })();
  return cached;
}

/**
 * A GradTarget that wraps Pyodide. Used to install the library once
 * (in test setup) and to run Python from individual tests.
 *
 * `exec({code})` returns the value of the last Python expression converted
 * to a JS value via Pyodide's automatic conversions:
 *   - Python lists/tuples → JS arrays
 *   - Python dicts → JS Maps (we convert to plain objects via toJs where needed)
 *   - Numbers → numbers
 *   - None → undefined
 */
export function makeTarget(py: PyodideAPI): GradTarget & {
  run: <T = unknown>(code: string) => Promise<T>;
} {
  const exec = async ({ code }: { code: string }): Promise<unknown> => {
    return py.runPythonAsync(code);
  };
  const run = async <T = unknown>(code: string): Promise<T> => {
    const result = await py.runPythonAsync(code);
    // PyProxy values have a .toJs method; native conversions return as-is.
    if (result && typeof result === "object" && "toJs" in result) {
      const proxy = result as unknown as {
        toJs: (opts?: { dict_converter?: unknown }) => unknown;
        destroy?: () => void;
      };
      const j = proxy.toJs({ dict_converter: Object.fromEntries });
      proxy.destroy?.();
      return j as T;
    }
    return result as T;
  };
  return { exec, run };
}

/**
 * Boot Pyodide + install grad. Returns a ready-to-use target with `run()`.
 * Cached at module level; safe to await from `beforeAll`.
 */
let installed: Promise<ReturnType<typeof makeTarget>> | null = null;
export async function getGradTarget(): Promise<ReturnType<typeof makeTarget>> {
  if (installed) return installed;
  installed = (async () => {
    const py = await getPyodide();
    const target = makeTarget(py);
    await installGrad(target);
    return target;
  })();
  return installed;
}

/**
 * Reset Python's user namespace between tests. Keeps modules / imports alive
 * (browsergrad_grad stays installed) but drops user variables.
 *
 * Helper-local names use dunder prefixes so `_k.startswith("__")` auto-skips
 * them — otherwise the loop would happily delete its own iteration variables
 * partway through.
 */
export async function clearNamespace(target: { run: (code: string) => Promise<unknown> }): Promise<void> {
  await target.run(`
import sys as __bg_sys__
for __bg_k__ in list(globals().keys()):
    if __bg_k__.startswith("__"):
        continue
    __bg_v__ = globals().get(__bg_k__)
    if isinstance(__bg_v__, type(__bg_sys__)):
        continue
    del globals()[__bg_k__]
`);
}
