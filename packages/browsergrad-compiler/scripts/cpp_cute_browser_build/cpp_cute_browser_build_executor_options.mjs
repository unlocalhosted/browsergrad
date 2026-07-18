const ABORTED_GETTER = Object.getOwnPropertyDescriptor(
  AbortSignal.prototype,
  "aborted",
)?.get;

/**
 * @param {import("./cpp_cute_browser_build_executor.mjs").CppCuteBrowserBuildExecutorOptions | import("./cpp_cute_browser_build_executor.mjs").CppCuteBrowserBuildExecutionOptions} options
 * @param {boolean} execution
 * @param {(path: string, message: string, options?: ErrorOptions) => never} invalid
 */
export function normalizeCppCuteBrowserBuildExecutorOptions(
  options,
  execution,
  invalid,
) {
  const descriptors = exactDataObjectOptional(
    options,
    execution ? ["buildDirectoryPolicy", "mirrorOutput", "signal"] : ["signal"],
    invalid,
  );
  const signal = descriptors.signal?.value;
  if (signal !== undefined && !isAbortSignal(signal)) {
    invalid("$options.signal", "signal must be an AbortSignal");
  }
  const mirrorOutput = descriptors.mirrorOutput?.value ?? false;
  if (typeof mirrorOutput !== "boolean") {
    invalid("$options.mirrorOutput", "mirrorOutput must be a boolean");
  }
  const buildDirectoryPolicy = descriptors.buildDirectoryPolicy?.value ?? "clean";
  if (buildDirectoryPolicy !== "clean" &&
      buildDirectoryPolicy !== "reuse-untrusted-diagnostic") {
    invalid(
      "$options.buildDirectoryPolicy",
      "build directory policy must be clean or reuse-untrusted-diagnostic",
    );
  }
  return Object.freeze({ signal, mirrorOutput, buildDirectoryPolicy });
}

/**
 * @param {unknown} value
 * @param {readonly string[]} allowedKeys
 * @param {(path: string, message: string, options?: ErrorOptions) => never} invalid
 */
function exactDataObjectOptional(value, allowedKeys, invalid) {
  let prototype;
  let descriptors;
  try {
    prototype = Object.getPrototypeOf(value);
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch (cause) {
    invalid("$options", "expected an inspectable plain object", { cause });
  }
  if (typeof value !== "object" || value === null || prototype !== Object.prototype) {
    invalid("$options", "expected a plain object");
  }
  const keys = Reflect.ownKeys(descriptors);
  if (keys.some((key) => typeof key !== "string" || !allowedKeys.includes(key))) {
    invalid("$options", "object contains unknown fields");
  }
  for (const key of keys) {
    const descriptor = descriptors[/** @type {string} */ (key)];
    if (descriptor === undefined || !("value" in descriptor) ||
        descriptor.enumerable !== true) {
      invalid(`$options.${String(key)}`, "field must be an enumerable data property");
    }
  }
  return descriptors;
}

/** @param {unknown} value */
function isAbortSignal(value) {
  if (ABORTED_GETTER === undefined) return false;
  try {
    return typeof ABORTED_GETTER.call(value) === "boolean";
  } catch {
    return false;
  }
}
