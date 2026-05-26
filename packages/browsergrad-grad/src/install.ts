/**
 * `installGrad` — writes the browsergrad_grad Python package into a Pyodide
 * target and verifies it imports cleanly.
 *
 * Two install paths:
 *
 *   1. If `target.fs.write` is available (the runtime Session has this),
 *      we write each .py file to the virtual FS, add the mount root to
 *      sys.path, and `import browsergrad_grad`. Clean — the Python module
 *      is on disk and behaves like any other package.
 *
 *   2. If `target.fs` is absent, we fall back to a single `exec` that
 *      installs the modules in-memory via importlib + types.ModuleType.
 *      Less elegant but works with raw Pyodide setups.
 */

import { SOURCE_FILES, MOUNT_ROOT } from "./python/index.js";
import { GradInstallError, type GradTarget, type InstallOptions } from "./types.js";

export async function installGrad(
  target: GradTarget,
  options: InstallOptions = {},
): Promise<void> {
  const mountRoot = options.mountRoot ?? MOUNT_ROOT;

  if (target.fs?.write) {
    await installViaFs(target, mountRoot);
  } else {
    await installViaExec(target, mountRoot);
  }

  if (!options.skipImportCheck) {
    await assertImports(target);
  }
}

async function installViaFs(target: GradTarget, mountRoot: string): Promise<void> {
  for (const file of SOURCE_FILES) {
    const fullPath = `${mountRoot}/${file.path}`;
    try {
      await target.fs!.write(fullPath, file.content);
    } catch (err) {
      throw new GradInstallError(
        `failed to write ${fullPath}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }
  await target.exec({
    code: `
import sys
_mount = ${JSON.stringify(mountRoot)}
if _mount not in sys.path:
    sys.path.insert(0, _mount)
del _mount
`,
  });
}

async function installViaExec(target: GradTarget, mountRoot: string): Promise<void> {
  // Build a single Python script that writes the package via os.makedirs + open.
  const lines: string[] = [
    "import os, sys",
    `_root = ${JSON.stringify(mountRoot)}`,
    `_pkg_dir = os.path.join(_root, "browsergrad_grad")`,
    `os.makedirs(_pkg_dir, exist_ok=True)`,
  ];
  for (const file of SOURCE_FILES) {
    // file.path looks like "browsergrad_grad/utils/__init__.py" — we want absolute.
    const relPath = file.path.split("/").slice(1).join("/"); // strip "browsergrad_grad/"
    const fullExpr = `os.path.join(_pkg_dir, ${JSON.stringify(relPath)})`;
    lines.push(`_p = ${fullExpr}`);
    lines.push(`os.makedirs(os.path.dirname(_p), exist_ok=True)`);
    lines.push(`with open(_p, "w") as _f:`);
    lines.push(`    _f.write(${pythonStringLiteral(file.content)})`);
  }
  lines.push(`if _root not in sys.path:`);
  lines.push(`    sys.path.insert(0, _root)`);
  lines.push(`del _root, _pkg_dir, _f, _p`);

  await target.exec({ code: lines.join("\n") });
}

async function assertImports(target: GradTarget): Promise<void> {
  await target.exec({
    code: `
import browsergrad_grad as _bg_check
assert _bg_check.__version__ == "0.4.17", f"unexpected version {_bg_check.__version__}"
del _bg_check
`,
  });
}

/**
 * Encode a Python expression that evaluates to `source`, base64-roundtripped
 * to bypass all quote/backslash escaping concerns. `btoa` only handles latin1,
 * so we first encode `source` to UTF-8 bytes and rebuild a latin1 string from
 * the byte values — a standard idiom that works in browsers and Node 16+.
 */
function pythonStringLiteral(source: string): string {
  if (typeof globalThis.btoa !== "function") {
    throw new GradInstallError(
      "btoa is not available in this environment — install requires a modern browser or Node ≥ 16",
    );
  }
  const bytes = new TextEncoder().encode(source);
  const latin1 = String.fromCharCode(...bytes);
  const b64 = globalThis.btoa(latin1);
  return `__import__("base64").b64decode(${JSON.stringify(b64)}).decode("utf-8")`;
}
