/**
 * Public types for @unlocalhosted/browsergrad-grad.
 *
 * Stability contract:
 * - Adding a new optional field is non-breaking.
 * - Removing fields or making them required is breaking.
 * - Anything not exported from `./index.ts` is private.
 */

/**
 * Duck-typed target the library installs itself into.
 *
 * Anything with an async `exec({code})` works — including a Session from
 * `@unlocalhosted/browsergrad-runtime` and a hand-rolled wrapper around
 * `pyodide.runPythonAsync(code)`.
 *
 * The return value is intentionally typed loosely: we only care that the
 * call completed without throwing. If the Python ran with an error, the
 * caller is responsible for surfacing that — we don't define the error shape
 * because different consumers (a runtime Session vs. a raw Pyodide handle)
 * have different ones.
 */
export interface GradTarget {
  exec(options: { code: string }): Promise<unknown>;

  /**
   * Optional virtual-FS write — preferred when available because Python
   * package files are written to disk and imported as a proper package.
   * If absent, we fall back to a single inlined `exec` that creates the
   * modules via runtime metaprogramming.
   */
  fs?: {
    write(path: string, content: string): Promise<void>;
  };
}

export interface InstallOptions {
  /**
   * Mount path inside the Pyodide virtual FS where the package source is
   * written. Default `/lib/browsergrad_grad_src`. Override only if you
   * have a collision with another mount.
   */
  mountRoot?: string;

  /**
   * Skip the `import browsergrad_grad` smoke test at the end of install.
   * Default `false` — the smoke test catches "files written but import
   * fails" early.
   */
  skipImportCheck?: boolean;
}

export class GradInstallError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GradInstallError";
  }
}
