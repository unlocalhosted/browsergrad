/**
 * `installJit` — writes the browsergrad_jit Python package into a Pyodide
 * target and verifies it imports cleanly.
 *
 * Same install protocol as browsergrad-grad's `installGrad`: prefers the
 * `target.fs.write` virtual-FS path; falls back to an inlined exec that
 * writes files via Python's open() when fs is absent. Either way, the
 * mount root is added to sys.path before the smoke-test import.
 *
 * The two packages can coexist in the same Pyodide worker — they install
 * to different mount roots (`browsergrad_grad_src` vs `browsergrad_jit_src`)
 * and register their `torch` aliases through a shared owner-token protocol
 * (see Week 6 — `install_torch_alias`).
 */

import { SOURCE_FILES, MOUNT_ROOT } from "./python/index.js";
import { JitInstallError, type JitTarget, type InstallOptions } from "./types.js";
import pkg from "../package.json" with { type: "json" };

export async function installJit(
  target: JitTarget,
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

async function installViaFs(target: JitTarget, mountRoot: string): Promise<void> {
  for (const file of SOURCE_FILES) {
    const fullPath = `${mountRoot}/${file.path}`;
    try {
      await target.fs!.write(fullPath, file.content);
    } catch (err) {
      throw new JitInstallError(
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

async function installViaExec(target: JitTarget, mountRoot: string): Promise<void> {
  const lines: string[] = [
    "import os, sys",
    `_root = ${JSON.stringify(mountRoot)}`,
    `_pkg_dir = os.path.join(_root, "browsergrad_jit")`,
    `os.makedirs(_pkg_dir, exist_ok=True)`,
  ];
  for (const file of SOURCE_FILES) {
    // file.path looks like "browsergrad_jit/_ir.py" — we want absolute.
    const relPath = file.path.split("/").slice(1).join("/"); // strip "browsergrad_jit/"
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

async function assertImports(target: JitTarget): Promise<void> {
  await target.exec({
    code: `
import browsergrad_jit as _bg_jit_check
assert _bg_jit_check.__version__ == "${pkg.version}", f"expected ${pkg.version}, got {_bg_jit_check.__version__}"
del _bg_jit_check
`,
  });
}

/**
 * Encode a Python expression that evaluates to `source`, base64-roundtripped
 * to bypass all quote/backslash escaping concerns. `btoa` only handles
 * latin1, so we first encode `source` to UTF-8 bytes and rebuild a latin1
 * string from the byte values — standard idiom that works in browsers and
 * Node 16+.
 */
function pythonStringLiteral(source: string): string {
  if (typeof globalThis.btoa !== "function") {
    throw new JitInstallError(
      "btoa is not available in this environment — install requires a modern browser or Node ≥ 16",
    );
  }
  const bytes = new TextEncoder().encode(source);
  const latin1 = bytesToLatin1(bytes);
  const b64 = globalThis.btoa(latin1);
  return `__import__("base64").b64decode(${JSON.stringify(b64)}).decode("utf-8")`;
}

function bytesToLatin1(bytes: Uint8Array): string {
  const chunkSize = 0x8000;
  let out = "";
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    out += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return out;
}
