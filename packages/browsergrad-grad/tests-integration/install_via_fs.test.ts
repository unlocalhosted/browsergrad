/**
 * Refactor #3 — install_via_fs.test.ts.
 *
 * Until the NodePyodideTarget Adapter shipped, `installViaFs` was a published
 * code path with no real-Pyodide coverage — every integration test ran through
 * `installViaExec`. This test asserts that the FS-write branch actually wrote
 * the package's __init__.py into the Pyodide virtual filesystem.
 *
 * Proof shape: after `installGrad` succeeds, the file at
 * `/lib/browsergrad_grad_src/browsergrad_grad/__init__.py` exists on the
 * Pyodide FS. That can only happen if `target.fs.write` was called, i.e.
 * `installViaFs` was the active branch.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { getGradTarget, getPyodide } from "./pyodide-host";

describe("createNodePyodideTarget exercises installViaFs against real Pyodide", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  it("writes browsergrad_grad/__init__.py to the Pyodide virtual FS", async () => {
    const py = await getPyodide();
    const result = await (
      py as unknown as {
        runPythonAsync: (code: string) => Promise<boolean>;
      }
    ).runPythonAsync(`
import os
os.path.exists("/lib/browsergrad_grad_src/browsergrad_grad/__init__.py")
`);
    expect(Boolean(result)).toBe(true);
  });

  it("writes nested subpackage utils/data.py via the mkdirTree+writeFile path", async () => {
    const py = await getPyodide();
    const result = await (
      py as unknown as {
        runPythonAsync: (code: string) => Promise<boolean>;
      }
    ).runPythonAsync(`
import os
os.path.exists("/lib/browsergrad_grad_src/browsergrad_grad/utils/data.py")
`);
    expect(Boolean(result)).toBe(true);
  });
});
