export function createKernelCompilationUnit({
  kernel,
  siblingKernels = [],
  definesByName = new Map(),
  functionDeclarations = [],
  deviceFunctions = [],
  constantDeclarations = [],
  textureDeclarations = [],
}) {
  const params = new Set(kernelParamNames(kernel));
  const defines = [...definesByName]
    .filter(([name]) => !params.has(name))
    .map(([name, value]) => `#define ${name} ${value}`);
  const referencedDeviceFunctions = referencedDeviceFunctionClosure(kernel, deviceFunctions);
  const referencedSiblingKernels = siblingKernels.filter((sibling) => {
    const name = kernelDefinitionName(sibling);
    return name !== undefined && sourceLaunchesKernel(kernel, name);
  });
  const macroScope = [
    kernel,
    ...referencedDeviceFunctions.map((fn) => fn.source),
    ...referencedSiblingKernels,
  ].join("\n");
  const functionMacros = functionDeclarations.filter((declaration) => {
    const name = functionDefineName(declaration);
    return name !== undefined && sourceMentionsIdentifier(macroScope, name);
  });
  return [
    defines.join("\n"),
    functionMacros.join("\n"),
    referencedDeviceFunctions.map((fn) => fn.source).join("\n"),
    constantDeclarations.join("\n"),
    textureDeclarations.join("\n"),
    referencedSiblingKernels.join("\n"),
    kernel,
  ].filter((part) => part.trim().length > 0).join("\n");
}

export function kernelDefinitionName(kernel) {
  return /__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/u.exec(kernel)?.[1];
}

function referencedDeviceFunctionClosure(kernel, deviceFunctions) {
  const byName = new Map(deviceFunctions.map((fn) => [fn.name, fn]));
  const included = new Map();
  const pending = [kernel];
  while (pending.length > 0) {
    const source = pending.pop();
    for (const [name, fn] of byName) {
      if (included.has(name) || !sourceMentionsIdentifier(source, name)) continue;
      included.set(name, fn);
      pending.push(fn.source);
    }
  }
  return [...included.values()];
}

function sourceLaunchesKernel(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\s*<<\\s*<`, "u").test(source);
}

function sourceMentionsIdentifier(source, name) {
  return new RegExp(`\\b${escapeRegExp(name)}\\b`, "u").test(source);
}

function functionDefineName(declaration) {
  return /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/u.exec(declaration)?.[1];
}

function kernelParamNames(kernel) {
  const signature = /__global__\s+(?:void\s+[A-Za-z_][A-Za-z0-9_]*\s*)?\(([\s\S]*?)\)\s*\{/u.exec(kernel);
  if (signature === null) return [];
  return signature[1]
    .split(",")
    .map((param) => /([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$/u.exec(param.trim())?.[1])
    .filter(Boolean);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
