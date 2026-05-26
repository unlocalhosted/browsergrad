/**
 * Pyodide-in-node host shared by every browsergrad-jit integration test.
 *
 * Boots Pyodide once per process, preloads numpy, installs the jit library
 * via our public `installJit` API. Subsequent tests reuse the same Pyodide
 * instance — namespace is reset between tests via `clearNamespace`.
 *
 * Mirrors browsergrad-grad's host file 1:1 by design — the harness shape
 * stays consistent so a contributor familiar with one package can read
 * the other's tests without re-learning the conventions.
 */

import { loadPyodide } from "pyodide";
import { installJit, type JitTarget } from "../src/index";
import { createNodePyodideTarget, type PyodideInterface } from "../src/node-adapter";

interface PyodideAPI extends PyodideInterface {
  loadPackage: (names: readonly string[]) => Promise<void>;
  globals: { get: (name: string) => unknown };
}

let cached: Promise<PyodideAPI> | null = null;

export async function getPyodide(): Promise<PyodideAPI> {
  if (cached) return cached;
  cached = (async () => {
    const py = await loadPyodide({
      stdout: () => {},
      stderr: () => {},
    });
    await py.loadPackage(["numpy"]);
    return py as unknown as PyodideAPI;
  })();
  return cached;
}

/**
 * A JitTarget that wraps Pyodide, plus a `.run<T>()` helper for tests.
 * The JitTarget shape comes from the published `createNodePyodideTarget`
 * factory — same path real consumers take. The `.run<T>()` wrapper is
 * purely test glue (PyProxy unwrap + dict conversion + cleanup) and
 * stays here.
 */
export function makeTarget(py: PyodideAPI): JitTarget & {
  run: <T = unknown>(code: string) => Promise<T>;
} {
  const adapter = createNodePyodideTarget(py);
  const run = async <T = unknown>(code: string): Promise<T> => {
    const result = await py.runPythonAsync(code);
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
  return { ...adapter, run };
}

/**
 * Boot Pyodide + install jit. Returns a ready-to-use target with `run()`.
 * Cached at module level; safe to await from `beforeAll`.
 */
let installed: Promise<ReturnType<typeof makeTarget>> | null = null;
export async function getJitTarget(): Promise<ReturnType<typeof makeTarget>> {
  if (installed) return installed;
  installed = (async () => {
    const py = await getPyodide();
    const target = makeTarget(py);
    await installJit(target);
    return target;
  })();
  return installed;
}

/**
 * Reset Python's user namespace between tests. Keeps modules / imports
 * alive (browsergrad_jit stays installed) but drops user variables.
 *
 * Helper-local names use dunder prefixes so `_k.startswith("__")` auto-skips
 * them — otherwise the loop would happily delete its own iteration
 * variables partway through.
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
