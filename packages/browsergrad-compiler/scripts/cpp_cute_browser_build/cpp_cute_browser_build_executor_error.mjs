export class CppCuteBrowserBuildExecutorError extends Error {
  /**
   * @param {import("./cpp_cute_browser_build_executor.mjs").CppCuteBrowserBuildExecutorErrorCode} code
   * @param {string} path
   * @param {string} message
   * @param {ErrorOptions} [options]
   */
  constructor(code, path, message, options) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserBuildExecutorError";
    this.code = code;
    this.path = path;
  }
}
