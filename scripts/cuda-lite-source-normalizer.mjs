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
  return kernelSignature(kernel)?.name;
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
  return (kernelSignature(kernel)?.params ?? "")
    .split(",")
    .map((param) => /([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$/u.exec(param.trim())?.[1])
    .filter(Boolean);
}

function kernelSignature(kernel) {
  const header = kernel.slice(0, kernel.indexOf("{") < 0 ? kernel.length : kernel.indexOf("{"));
  const match = /__global__[\s\S]*?\bvoid\b\s*(?:(?:__launch_bounds__)\s*\([^)]*\)\s*)*(?:static\s+|extern\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(/u.exec(header);
  if (match?.[1] === undefined || match.index === undefined) return undefined;
  const open = header.indexOf("(", match.index + match[0].length - 1);
  if (open < 0) return undefined;
  let depth = 0;
  for (let index = open; index < header.length; index++) {
    const char = header[index];
    if (char === "(") depth++;
    else if (char === ")") {
      depth--;
      if (depth === 0) {
        return { name: match[1], params: header.slice(open + 1, index) };
      }
    }
  }
  return undefined;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
