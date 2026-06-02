/**
 * Node-mode helpers. Pyodide-in-Node bootstrap, cached across tests.
 */

import { loadPyodide } from "pyodide";

export interface PyodideAPI {
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (packages: string[]) => Promise<void>;
  globals: { get: (name: string) => unknown };
  registerJsModule: (name: string, mod: unknown) => void;
}

let cachedPy: PyodideAPI | null = null;

export async function getPyodide(): Promise<PyodideAPI> {
  if (cachedPy) return cachedPy;
  const py = (await loadPyodide({ stdout: () => {}, stderr: () => {} })) as unknown as PyodideAPI;
  await py.loadPackage(["numpy"]);
  cachedPy = py;
  return py;
}

export async function pyJson<T>(code: string): Promise<T> {
  const py = await getPyodide();
  const raw = await py.runPythonAsync(code);
  return JSON.parse(raw as string) as T;
}

export async function pyStr(code: string): Promise<string> {
  const py = await getPyodide();
  return (await py.runPythonAsync(code)) as string;
}

export async function pyBool(code: string): Promise<boolean> {
  const py = await getPyodide();
  return (await py.runPythonAsync(code)) as boolean;
}
