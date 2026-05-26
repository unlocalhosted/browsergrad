/**
 * Public types for @unlocalhosted/browsergrad-jit.
 *
 * Stability contract (0.x line):
 *   - Adding a new optional field is non-breaking.
 *   - Removing fields or making them required is breaking → major bump.
 *   - Anything not exported from `./index.ts` is private and may change
 *     in any minor or patch release (this includes the Python `_ir` /
 *     `_buffer_table` / `_tensor_proxy` modules).
 *
 * The Python-side semver story:
 *   - PUBLIC: TensorProxy attributes/methods, nn.* shapes, optim.*, the
 *             `browsergrad_jit` top-level namespace, error classes.
 *   - INTERNAL: `_ir` module, opcode strings, UOp dataclass, realize()
 *               internals, BufferTable.
 *   - NOT PROMISED: IR serialization, trace cache format, per-opcode
 *                   numerical equivalence with browsergrad-grad beyond
 *                   the documented tolerance (1e-4 for fp32).
 */

/**
 * Duck-typed target the library installs itself into.
 *
 * Anything with an async `exec({code})` works — a Session from
 * `@unlocalhosted/browsergrad-runtime` and a hand-rolled wrapper around
 * `pyodide.runPythonAsync(code)` both satisfy this shape. Same contract
 * as browsergrad-grad's `GradTarget` — intentional so consumers can swap
 * the two packages without touching the install wiring.
 */
export interface JitTarget {
  exec(options: { code: string }): Promise<unknown>;

  /**
   * Optional virtual-FS write — preferred when available because Python
   * package files are written to disk and imported as a proper package.
   * Falls back to a single inlined `exec` that creates the modules via
   * runtime metaprogramming if absent.
   */
  fs?: {
    write(path: string, content: string): Promise<void>;
  };
}

export interface InstallOptions {
  /**
   * Mount path inside the Pyodide virtual FS where the package source is
   * written. Default `/lib/browsergrad_jit_src`. Override only if you
   * have a collision with another mount.
   */
  mountRoot?: string;

  /**
   * Skip the `import browsergrad_jit` smoke test at the end of install.
   * Default `false` — the smoke test catches "files written but import
   * fails" early.
   */
  skipImportCheck?: boolean;
}

export class JitInstallError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "JitInstallError";
  }
}
