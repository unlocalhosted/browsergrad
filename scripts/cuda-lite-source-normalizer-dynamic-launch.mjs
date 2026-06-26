import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";
import { parseCudaGlobalFunction } from "./cuda-lite-source-normalizer-cute-la.mjs";

const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const skipWhitespace = (...args) => requireNormalizerHelpers().skipWhitespace(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);

export function normalizeBarrierFreeDynamicKernelInlining(source) {
  if (!/<<<[\s\S]*>>>/u.test(source)) return source;
  const kernels = collectGlobalKernels(source);
  if (kernels.size === 0) return source;

  let out = "";
  let cursor = 0;
  const cloneSources = new Map();
  const globalRe = /\b__global__\b/gu;
  let match;
  while ((match = globalRe.exec(source)) !== null) {
    const fn = parseCudaGlobalFunction(source, match.index);
    if (!fn || fn.headerStart < cursor) continue;
    const replacement = rewriteKernelLaunchesInBody(fn, kernels, cloneSources);
    if (replacement === undefined) {
      globalRe.lastIndex = fn.bodyEnd + 1;
      continue;
    }
    out += source.slice(cursor, fn.headerStart);
    out += `${source.slice(fn.headerStart, source.indexOf("{", match.index) + 1)}${replacement}\n}`;
    cursor = fn.bodyEnd + 1;
    globalRe.lastIndex = fn.bodyEnd + 1;
  }
  if (cursor === 0) return source;
  const clones = [...cloneSources.values()].join("\n");
  return `${clones}${clones.length > 0 ? "\n" : ""}${out}${source.slice(cursor)}`;
}

function collectGlobalKernels(source) {
  const kernels = new Map();
  const globalRe = /\b__global__\b/gu;
  let match;
  while ((match = globalRe.exec(source)) !== null) {
    const fn = parseCudaGlobalFunction(source, match.index);
    if (!fn) continue;
    kernels.set(fn.name, fn);
    globalRe.lastIndex = fn.bodyEnd + 1;
  }
  return kernels;
}

function rewriteKernelLaunchesInBody(fn, kernels, cloneSources) {
  let body = fn.body;
  let changed = false;
  for (const [callee, child] of kernels) {
    if (callee === fn.name || !isBarrierFreeInlineCandidate(child)) continue;
    const rewritten = replaceLaunches(body, callee, (launch) => {
      const loop = inlineLaunchLoop(callee, child, launch);
      if (loop === undefined) return undefined;
      cloneSources.set(callee, emitInlineClone(callee, child));
      return loop;
    });
    if (rewritten !== body) {
      body = rewritten;
      changed = true;
    }
  }
  return changed ? body : undefined;
}

function isBarrierFreeInlineCandidate(fn) {
  return !/\b__shared__\b|\bextern\s+__shared__\b|\b__syncthreads\s*\(|\b__syncwarp\s*\(|\b(?:cg|cooperative_groups)\s*::\s*sync\s*\(|\.\s*sync\s*\(|\b(?:deviceAllocate|streamOrderedAllocate|cudaMalloc)\s*\(|<<<[\s\S]*>>>/u.test(fn.body);
}

function replaceLaunches(source, callee, replacer) {
  const re = new RegExp(`\\b${escapeRegExp(callee)}\\s*<<<`, "gu");
  let out = "";
  let cursor = 0;
  let match;
  while ((match = re.exec(source)) !== null) {
    const launch = parseLaunchStatement(source, match.index, callee);
    if (!launch) continue;
    const replacement = replacer(launch);
    if (replacement === undefined) {
      re.lastIndex = launch.end;
      continue;
    }
    out += source.slice(cursor, launch.start);
    out += replacement;
    cursor = launch.end;
    re.lastIndex = launch.end;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function parseLaunchStatement(source, start, callee) {
  const configOpen = source.indexOf("<<<", start + callee.length);
  if (configOpen < 0) return undefined;
  const configClose = source.indexOf(">>>", configOpen + 3);
  if (configClose < 0) return undefined;
  const argsOpen = skipWhitespace(source, configClose + 3);
  if (source[argsOpen] !== "(") return undefined;
  const argsClose = findBalanced(source, argsOpen, "(", ")");
  if (argsClose === undefined) return undefined;
  const semi = skipWhitespace(source, argsClose + 1);
  if (source[semi] !== ";") return undefined;
  return {
    start,
    end: semi + 1,
    config: splitTopLevel(source.slice(configOpen + 3, configClose)).map((item) => item.trim()),
    args: splitTopLevel(source.slice(argsOpen + 1, argsClose)).map((item) => item.trim()),
  };
}

function inlineLaunchLoop(callee, child, launch) {
  const grid = launch.config[0];
  const block = launch.config[1];
  if (!grid || !block || !isOneDimLaunchExpression(grid) || !isOneDimLaunchExpression(block)) return undefined;
  if (isBareIdentifier(grid) || isBareIdentifier(block)) return undefined;
  if (isIntegerLiteral(grid) && isIntegerLiteral(block)) return undefined;
  if (launch.args.length !== child.params.length) return undefined;
  const argText = launch.args.join(", ");
  const maybeComma = argText.length > 0 ? ", " : "";
  const suffix = stableSuffix(callee, grid, block, argText);
  return `for (int bg_inline_bx_${suffix} = 0; bg_inline_bx_${suffix} < (${grid}); ++bg_inline_bx_${suffix}) {
  for (int bg_inline_tx_${suffix} = 0; bg_inline_tx_${suffix} < (${block}); ++bg_inline_tx_${suffix}) {
    ${inlineCloneName(callee)}(${argText}${maybeComma}bg_inline_bx_${suffix}, bg_inline_tx_${suffix}, (${grid}), (${block}));
  }
}`;
}

function isOneDimLaunchExpression(expression) {
  const trimmed = expression.trim();
  if (/^dim3\b/u.test(trimmed)) return false;
  return !/[,{}]/u.test(trimmed);
}

function isIntegerLiteral(expression) {
  return /^\(?\s*\d+[uUlL]*\s*\)?$/u.test(expression.trim());
}

function isBareIdentifier(expression) {
  return /^[A-Za-z_][A-Za-z0-9_]*$/u.test(expression.trim());
}

function emitInlineClone(callee, fn) {
  const params = fn.params.map((param) => normalizeParamForClone(param));
  if (params.some((param) => param === undefined)) return "";
  const body = fn.body
    .replace(/\bthreadIdx\s*\.\s*x\b/gu, "bg_threadIdx_x")
    .replace(/\bthreadIdx\s*\.\s*y\b/gu, "0")
    .replace(/\bthreadIdx\s*\.\s*z\b/gu, "0")
    .replace(/\bblockIdx\s*\.\s*x\b/gu, "bg_blockIdx_x")
    .replace(/\bblockIdx\s*\.\s*y\b/gu, "0")
    .replace(/\bblockIdx\s*\.\s*z\b/gu, "0")
    .replace(/\bblockDim\s*\.\s*x\b/gu, "bg_blockDim_x")
    .replace(/\bblockDim\s*\.\s*y\b/gu, "1")
    .replace(/\bblockDim\s*\.\s*z\b/gu, "1")
    .replace(/\bgridDim\s*\.\s*x\b/gu, "bg_gridDim_x")
    .replace(/\bgridDim\s*\.\s*y\b/gu, "1")
    .replace(/\bgridDim\s*\.\s*z\b/gu, "1");
  return `__device__ void ${inlineCloneName(callee)}(${[
    ...params,
    "int bg_blockIdx_x",
    "int bg_threadIdx_x",
    "int bg_gridDim_x",
    "int bg_blockDim_x",
  ].join(", ")}) {${body}
}`;
}

function normalizeParamForClone(param) {
  const name = paramName(param);
  return name === undefined ? undefined : param.replace(/\s*=\s*[^,]+$/u, "").trim();
}

function paramName(param) {
  return /([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$/u.exec(param.replace(/\s*=\s*[^,]+$/u, "").trim())?.[1];
}

function inlineCloneName(callee) {
  return `bg_inline_${callee}`;
}

function stableSuffix(...parts) {
  let hash = 0x811c9dc5;
  const text = parts.join("\0");
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
