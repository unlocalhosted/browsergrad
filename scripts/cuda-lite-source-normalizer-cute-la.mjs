import { requireNormalizerHelpers } from "./cuda-lite-source-normalizer-context.mjs";

const escapeRegExp = (...args) => requireNormalizerHelpers().escapeRegExp(...args);
const evaluateTemplateIntegerExpression = (...args) => requireNormalizerHelpers().evaluateTemplateIntegerExpression(...args);
const findBalanced = (...args) => requireNormalizerHelpers().findBalanced(...args);
const normalizeTemplateTypeArgument = (...args) => requireNormalizerHelpers().normalizeTemplateTypeArgument(...args);
const skipWhitespace = (...args) => requireNormalizerHelpers().skipWhitespace(...args);
const splitTopLevel = (...args) => requireNormalizerHelpers().splitTopLevel(...args);

export function normalizeCuteRank2TransposeKernels(source) {
  if (!/\bmake_tensor\s*\(\s*make_gmem_ptr\s*\(/u.test(source) || !/\blocal_tile\s*\(/u.test(source)) return source;
  if (!/\b(?:copy_if|copy)\s*\(/u.test(source)) return source;
  let out = "";
  let cursor = 0;
  const globalRe = /\b__global__\b/gu;
  let match;
  while ((match = globalRe.exec(source)) !== null) {
    const globalStart = match.index;
    if (globalStart < cursor) continue;
    const fn = parseCudaGlobalFunction(source, globalStart);
    if (fn === undefined) {
      globalRe.lastIndex = globalStart + "__global__".length;
      continue;
    }
    const replacement = lowerCuteRank2TransposeFunction(fn);
    if (replacement === undefined) {
      globalRe.lastIndex = fn.bodyEnd + 1;
      continue;
    }
    out += source.slice(cursor, fn.headerStart);
    out += replacement;
    cursor = fn.bodyEnd + 1;
    globalRe.lastIndex = fn.bodyEnd + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

export function parseCudaGlobalFunction(source, globalStart) {
  let open = source.indexOf("(", globalStart + "__global__".length);
  while (open >= 0) {
    const close = findBalanced(source, open, "(", ")");
    if (close === undefined) return undefined;
    const bodyOpen = skipWhitespace(source, close + 1);
    if (source[bodyOpen] === "{") {
      const bodyEnd = findBalanced(source, bodyOpen, "{", "}");
      if (bodyEnd === undefined) return undefined;
      const signaturePrefix = source.slice(globalStart, open);
      const name = /([A-Za-z_][A-Za-z0-9_]*)\s*$/u.exec(signaturePrefix)?.[1];
      if (name === undefined) return undefined;
      return {
        headerStart: cudaGlobalHeaderStart(source, globalStart),
        globalStart,
        name,
        signaturePrefix: signaturePrefix.trimEnd(),
        params: splitTopLevel(source.slice(open + 1, close)).map((param) => param.trim()).filter(Boolean),
        body: source.slice(bodyOpen + 1, bodyEnd),
        bodyEnd,
      };
    }
    open = source.indexOf("(", close + 1);
  }
  return undefined;
}

function cudaGlobalHeaderStart(source, globalStart) {
  const before = source.slice(0, globalStart);
  const template = /\btemplate\s*<[^<>]*>\s*$/u.exec(before);
  return template?.index ?? globalStart;
}

function lowerCuteRank2TransposeFunction(fn) {
  const motif = collectCuteRank2TransposeMotif(fn.body);
  if (motif === undefined) return undefined;
  const paramsByName = new Map();
  for (const param of fn.params) {
    const name = cudaParamName(param);
    if (name !== undefined && !paramsByName.has(name)) paramsByName.set(name, param);
  }
  const keepNames = [motif.inputPointer, motif.outputPointer, motif.rows, motif.cols];
  const keptParams = [];
  const seen = new Set();
  for (const name of keepNames) {
    if (seen.has(name)) continue;
    const param = paramsByName.get(name);
    if (param === undefined) return undefined;
    keptParams.push(param);
    seen.add(name);
  }
  return [
    `${fn.signaturePrefix}(${keptParams.join(", ")}) {`,
    ...cuteRank2TransposeBody(motif).map((line) => `  ${line}`),
    "}",
  ].join("\n");
}

function collectCuteRank2TransposeMotif(body) {
  const bases = collectCuteRank2GmemTensors(body);
  if (bases.length < 2) return undefined;
  const pair = findCuteRank2TransposeTensorPair(bases);
  if (pair === undefined) return undefined;
  const tiles = collectCuteRank2Tiles(body);
  const inputTile = tiles.find((tile) => tile.base === pair.input.name);
  const outputTile = tiles.find((tile) => tile.base === pair.output.name);
  if (inputTile === undefined || outputTile === undefined) return undefined;
  const tileRows = evaluateTemplateIntegerExpression(inputTile.extents[0], new Map());
  const tileCols = evaluateTemplateIntegerExpression(inputTile.extents[1], new Map());
  const outputRows = evaluateTemplateIntegerExpression(outputTile.extents[0], new Map());
  const outputCols = evaluateTemplateIntegerExpression(outputTile.extents[1], new Map());
  if (tileRows === undefined || tileCols === undefined || outputRows === undefined || outputCols === undefined) return undefined;
  if (Number(tileRows) !== Number(outputCols) || Number(tileCols) !== Number(outputRows)) return undefined;
  if (!cuteRank2BodyCopiesBetweenTiles(body, inputTile.name, outputTile.name)) return undefined;
  const aliases = collectCuteIndexAliases(body);
  const coords = inputTile.coords.map((coord) => resolveCuteIndexAlias(coord, aliases));
  if (coords.length !== 2) return undefined;
  return {
    inputPointer: pair.input.pointer,
    outputPointer: pair.output.pointer,
    rows: pair.input.shape[0],
    cols: pair.input.shape[1],
    tileRows: String(tileRows),
    tileCols: String(tileCols),
    blockRow: coords[0],
    blockCol: coords[1],
  };
}

function collectCuteRank2GmemTensors(body) {
  const out = [];
  const re = /\b(?:auto|Tensor)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor\s*\(\s*make_gmem_ptr\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*,\s*make_layout\s*\(\s*make_shape\s*\(([^)]*)\)\s*,\s*GenRowMajor\s*\{\s*\}\s*\)\s*\)\s*;/gu;
  for (const match of body.matchAll(re)) {
    if (match[1] === undefined || match[2] === undefined || match[3] === undefined) continue;
    const shape = splitTopLevel(match[3]).map((item) => item.trim());
    if (shape.length !== 2 || shape.some((item) => item.length === 0)) continue;
    out.push({ name: match[1], pointer: match[2], shape });
  }
  return out;
}

function findCuteRank2TransposeTensorPair(bases) {
  for (const input of bases) {
    for (const output of bases) {
      if (input === output) continue;
      if (input.pointer === output.pointer) continue;
      if (input.shape[0] === output.shape[1] && input.shape[1] === output.shape[0]) return { input, output };
    }
  }
  return undefined;
}

function collectCuteRank2Tiles(body) {
  const out = [];
  const re = /\b(?:auto|Tensor)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*local_tile\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*make_shape\s*\(\s*Int<([^>]+)>\s*\{\s*\}\s*,\s*Int<([^>]+)>\s*\{\s*\}\s*\)\s*,\s*make_coord\s*\(([^)]*)\)\s*\)\s*;/gu;
  for (const match of body.matchAll(re)) {
    if (match[1] === undefined || match[2] === undefined || match[3] === undefined || match[4] === undefined || match[5] === undefined) continue;
    const coords = splitTopLevel(match[5]).map((item) => item.trim());
    if (coords.length !== 2 || coords.some((item) => item.length === 0)) continue;
    out.push({
      name: match[1],
      base: match[2],
      extents: [match[3].trim(), match[4].trim()],
      coords,
    });
  }
  return out;
}

function cuteRank2BodyCopiesBetweenTiles(body, inputTileName, outputTileName) {
  const partitionSources = new Map();
  const partitionRe = /\b(?:Tensor|auto)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*\b(?:local_partition|partition_[SD])\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\b[^;]*;/gu;
  for (const match of body.matchAll(partitionRe)) {
    if (match[1] !== undefined && match[2] !== undefined) partitionSources.set(match[1], match[2]);
  }
  for (const match of body.matchAll(/\b(?:Tensor|auto)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor_like\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu)) {
    if (match[1] !== undefined && match[2] !== undefined) partitionSources.set(match[1], match[2]);
  }
  for (const match of body.matchAll(/\b(?:Tensor|auto)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*\bretile_[SD]\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu)) {
    if (match[1] !== undefined && match[2] !== undefined) partitionSources.set(match[1], match[2]);
  }
  const sharedTensors = new Set();
  for (const match of body.matchAll(/\b(?:auto|Tensor)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor\s*\(\s*make_smem_ptr\s*\(/gu)) {
    if (match[1] !== undefined) sharedTensors.add(match[1]);
  }
  const resolvesTo = (name, target) => {
    let current = name;
    for (let depth = 0; depth < 8; depth++) {
      if (current === target) return true;
      const next = partitionSources.get(current);
      if (next === undefined || next === current) break;
      current = next;
    }
    return false;
  };
  const resolvesToShared = (name) => [...sharedTensors].some((shared) => resolvesTo(name, shared));
  let inputToShared = false;
  let sharedToOutput = false;
  for (const match of body.matchAll(/\bcopy(?:_if)?\s*\(([^;]+)\)\s*;/gu)) {
    const args = splitTopLevel(match[1] ?? "").map((arg) => arg.trim()).filter(Boolean);
    if (args.length < 2) continue;
    const source = args.at(-2);
    const dest = args.at(-1);
    if (source === undefined || dest === undefined) continue;
    if (resolvesTo(source, inputTileName) && resolvesTo(dest, outputTileName)) return true;
    if (resolvesTo(source, inputTileName) && resolvesToShared(dest)) inputToShared = true;
    if (resolvesToShared(source) && resolvesTo(dest, outputTileName)) sharedToOutput = true;
  }
  return inputToShared && sharedToOutput;
}

function collectCuteIndexAliases(body) {
  const aliases = new Map();
  for (const match of body.matchAll(/\b(?:const\s+)?(?:int|auto)\s+([^;]*\b(?:blockIdx|threadIdx|blockDim)\.[xyz][^;]*)\s*;/gu)) {
    const declarators = splitTopLevel(match[1] ?? "").map((item) => item.trim()).filter(Boolean);
    for (const declarator of declarators) {
      const parsed = /^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*((?:blockIdx|threadIdx|blockDim)\.[xyz])$/u.exec(declarator);
      if (parsed?.[1] !== undefined && parsed[2] !== undefined) aliases.set(parsed[1], parsed[2]);
    }
  }
  return aliases;
}

function resolveCuteIndexAlias(expr, aliases) {
  const trimmed = expr.trim();
  return aliases.get(trimmed) ?? trimmed;
}

export function cudaParamName(param) {
  return /([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*$/u.exec(param.trim())?.[1];
}

export function cudaPointerParamValueType(param) {
  if (param === undefined) return undefined;
  const cleaned = param
    .replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  const type = /^([A-Za-z_][A-Za-z0-9_:]*(?:\s+[A-Za-z_][A-Za-z0-9_:]*)*)\s*\*/u.exec(cleaned)?.[1];
  return type === undefined ? undefined : normalizeTemplateTypeArgument(type) ?? type;
}

function cuteRank2TransposeBody(motif) {
  const total = `(${motif.tileRows} * ${motif.tileCols})`;
  const rowInTile = "bg_cute_row";
  const colInTile = "bg_cute_col";
  const row = "bg_cute_m";
  const col = "bg_cute_n";
  const linear = "bg_cute_linear";
  return [
    `for (int ${linear} = threadIdx.x; ${linear} < ${total}; ${linear} = ${linear} + blockDim.x) {`,
    `  int ${rowInTile} = ${linear} / ${motif.tileCols};`,
    `  int ${colInTile} = ${linear} % ${motif.tileCols};`,
    `  int ${row} = ((${motif.blockRow}) * ${motif.tileRows}) + ${rowInTile};`,
    `  int ${col} = ((${motif.blockCol}) * ${motif.tileCols}) + ${colInTile};`,
    `  if (${row} < ${motif.rows} && ${col} < ${motif.cols}) {`,
    `    ${motif.outputPointer}[(${col} * ${motif.rows}) + ${row}] = ${motif.inputPointer}[(${row} * ${motif.cols}) + ${col}];`,
    "  }",
    "}",
  ];
}

export function normalizeCuteRowBroadcastGemvKernels(source) {
  if (!/\bmake_tensor\s*\(\s*make_gmem_ptr\s*\(/u.test(source) || !/\bmake_stride\s*\(/u.test(source)) return source;
  if (!/\b(?:warp_reduce_sum_f16|gemm|copy_if)\b/u.test(source)) return source;
  let out = "";
  let cursor = 0;
  const globalRe = /\b__global__\b/gu;
  let match;
  while ((match = globalRe.exec(source)) !== null) {
    const globalStart = match.index;
    if (globalStart < cursor) continue;
    const fn = parseCudaGlobalFunction(source, globalStart);
    if (fn === undefined) {
      globalRe.lastIndex = globalStart + "__global__".length;
      continue;
    }
    const replacement = lowerCuteRowBroadcastGemvFunction(fn);
    if (replacement === undefined) {
      globalRe.lastIndex = fn.bodyEnd + 1;
      continue;
    }
    out += source.slice(cursor, fn.headerStart);
    out += replacement;
    cursor = fn.bodyEnd + 1;
    globalRe.lastIndex = fn.bodyEnd + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function lowerCuteRowBroadcastGemvFunction(fn) {
  const motif = collectCuteRowBroadcastGemvMotif(fn.body);
  if (motif === undefined) return undefined;
  const paramsByName = new Map();
  for (const param of fn.params) {
    const name = cudaParamName(param);
    if (name !== undefined && !paramsByName.has(name)) paramsByName.set(name, param);
  }
  const keepNames = [motif.matrixPointer, motif.vectorPointer, motif.outputPointer, motif.rows, motif.cols];
  const keptParams = [];
  const seen = new Set();
  for (const name of keepNames) {
    if (seen.has(name)) continue;
    const param = paramsByName.get(name);
    if (param === undefined) return undefined;
    keptParams.push(param);
    seen.add(name);
  }
  const valueType = cudaPointerParamValueType(paramsByName.get(motif.outputPointer)) ??
    cudaPointerParamValueType(paramsByName.get(motif.matrixPointer)) ??
    "float";
  return [
    `${fn.signaturePrefix}(${keptParams.join(", ")}) {`,
    ...cuteRowBroadcastGemvBody({ ...motif, valueType }).map((line) => `  ${line}`),
    "}",
  ].join("\n");
}

function collectCuteRowBroadcastGemvMotif(body) {
  const tensors = collectCuteRank2StridedGmemTensors(body);
  if (tensors.length < 3) return undefined;
  const matrix = tensors.find((tensor) => tensor.shape[0] !== undefined && tensor.shape[1] !== undefined &&
    cuteExprEquals(tensor.stride[0], tensor.shape[1]) && cuteExprIsOne(tensor.stride[1]));
  if (matrix === undefined) return undefined;
  const vector = tensors.find((tensor) => tensor !== matrix &&
    tensor.shape[0] === matrix.shape[0] &&
    tensor.shape[1] === matrix.shape[1] &&
    cuteExprIsZero(tensor.stride[0]) &&
    cuteExprIsOne(tensor.stride[1]));
  const output = tensors.find((tensor) => tensor !== matrix && tensor !== vector &&
    tensor.shape[0] === matrix.shape[0] &&
    cuteExprIsOne(tensor.shape[1]) &&
    cuteExprIsOne(tensor.stride[0]) &&
    cuteExprIsZero(tensor.stride[1]));
  if (vector === undefined || output === undefined) return undefined;
  const env = collectCuteIntegerEnv(body);
  const blockM = collectCuteTileExtentForBase(body, matrix.name, env);
  if (blockM === undefined) return undefined;
  if (!/(?:\bwarp_reduce_sum_f16\s*(?:<[^<>]*>)?|\bgemm)\s*\(/u.test(body)) return undefined;
  return {
    matrixPointer: matrix.pointer,
    vectorPointer: vector.pointer,
    outputPointer: output.pointer,
    rows: matrix.shape[0],
    cols: matrix.shape[1],
    blockRows: blockM,
  };
}

function collectCuteRank2StridedGmemTensors(body) {
  const out = [];
  const add = (name, pointer, rawShape, rawStride) => {
    if (name === undefined || pointer === undefined || rawShape === undefined || rawStride === undefined) return;
    const shape = splitTopLevel(rawShape).map(normalizeCuteScalarExpression);
    const stride = splitTopLevel(rawStride).map(normalizeCuteScalarExpression);
    if (shape.length !== 2 || stride.length !== 2) return;
    out.push({ name, pointer, shape, stride });
  };
  const gmemPointer = String.raw`make_gmem_ptr(?:\s*<[^<>]+>)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)`;
  const layoutRe = new RegExp(String.raw`\b(?:auto|Tensor)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor\s*\(\s*${gmemPointer}\s*,\s*make_layout\s*\(\s*make_shape\s*\(([^)]*)\)\s*,\s*make_stride\s*\(([^)]*)\)\s*\)\s*\)\s*;`, "gu");
  for (const match of body.matchAll(layoutRe)) {
    add(match[1], match[2], match[3], match[4]);
  }
  const directRe = new RegExp(String.raw`\b(?:auto|Tensor)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor\s*\(\s*${gmemPointer}\s*,\s*make_shape\s*\(([^)]*)\)\s*,\s*make_stride\s*\(([^)]*)\)\s*\)\s*;`, "gu");
  for (const match of body.matchAll(directRe)) {
    add(match[1], match[2], match[3], match[4]);
  }
  const rowMajorRe = new RegExp(String.raw`\b(?:auto|Tensor)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor\s*\(\s*${gmemPointer}\s*,\s*make_shape\s*\(([^)]*)\)\s*,\s*GenRowMajor\s*\{\s*\}\s*\)\s*;`, "gu");
  for (const match of body.matchAll(rowMajorRe)) {
    const rawShape = match[3];
    const shape = rawShape === undefined ? [] : splitTopLevel(rawShape).map(normalizeCuteScalarExpression);
    if (shape.length !== 2) continue;
    add(match[1], match[2], rawShape, `${shape[1]}, Int<1>{}`);
  }
  return out;
}

function collectCuteTileExtentForBase(body, baseName, env) {
  const re = /\b(?:auto|Tensor)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*local_tile\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*make_shape\s*\(\s*Int<([^>]+)>\s*\{\s*\}\s*,\s*Int<([^>]+)>\s*\{\s*\}\s*\)/gu;
  for (const match of body.matchAll(re)) {
    if (match[1] !== baseName || match[2] === undefined) continue;
    const value = evaluateTemplateIntegerExpression(match[2], env);
    if (value !== undefined) return String(value);
  }
  return undefined;
}

export function collectCuteIntegerEnv(body) {
  const env = new Map();
  for (const match of body.matchAll(/\b(?:constexpr\s+)?int\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);/gu)) {
    if (match[1] === undefined || match[2] === undefined) continue;
    const value = evaluateTemplateIntegerExpression(normalizeCuteScalarExpression(match[2]), env);
    if (value !== undefined) env.set(match[1], value);
  }
  return env;
}

export function normalizeCuteScalarExpression(raw) {
  return raw
    .replace(/\bInt\s*<([^<>]+)>\s*\{\s*\}/gu, (_match, expr) => `(${String(expr).trim()})`)
    .trim();
}

export function cuteExprEquals(left, right) {
  return normalizeCuteScalarExpression(left) === normalizeCuteScalarExpression(right);
}

function cuteExprIsZero(expr) {
  return normalizeCuteScalarExpression(expr) === "0";
}

function cuteExprIsOne(expr) {
  const normalized = normalizeCuteScalarExpression(expr).replace(/^\(([\s\S]*)\)$/u, "$1").trim();
  return normalized === "1";
}

function cuteRowBroadcastGemvBody(motif) {
  const tid = "bg_cute_tid";
  const rowLocal = "bg_cute_row_local";
  const row = "bg_cute_row";
  const k = "bg_cute_k";
  const sum = "bg_cute_sum";
  const stride = "((blockDim.x * blockDim.y) * blockDim.z)";
  return [
    `int ${tid} = threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y);`,
    `for (int ${rowLocal} = ${tid}; ${rowLocal} < ${motif.blockRows}; ${rowLocal} = ${rowLocal} + ${stride}) {`,
    `  int ${row} = (blockIdx.x * ${motif.blockRows}) + ${rowLocal};`,
    `  if (${row} < ${motif.rows}) {`,
    `    ${motif.valueType} ${sum} = 0.0f;`,
    `    for (int ${k} = 0; ${k} < ${motif.cols}; ${k} = ${k} + 1) {`,
    `      ${sum} = ${sum} + (${motif.matrixPointer}[(${row} * ${motif.cols}) + ${k}] * ${motif.vectorPointer}[${k}]);`,
    "    }",
    `    ${motif.outputPointer}[${row}] = ${sum};`,
    "  }",
    "}",
  ];
}

export function normalizeCuteTnGemmKernels(source) {
  if (!/\bmake_tensor\s*\(\s*make_gmem_ptr(?:\s*<[^<>]+>)?\s*\(/u.test(source) || !/\blocal_tile\s*\(/u.test(source)) return source;
  if (!/\bcute::gemm\s*\(/u.test(source) && !/\bgemm\s*\(/u.test(source) && !/::\s*consumer\s*\(/u.test(source)) return source;
  let out = "";
  let cursor = 0;
  const globalRe = /\b__global__\b/gu;
  let match;
  while ((match = globalRe.exec(source)) !== null) {
    const globalStart = match.index;
    if (globalStart < cursor) continue;
    const fn = parseCudaGlobalFunction(source, globalStart);
    if (fn === undefined) {
      globalRe.lastIndex = globalStart + "__global__".length;
      continue;
    }
    const replacement = lowerCuteTnGemmFunction(fn);
    if (replacement === undefined) {
      globalRe.lastIndex = fn.bodyEnd + 1;
      continue;
    }
    out += source.slice(cursor, fn.headerStart);
    out += replacement;
    cursor = fn.bodyEnd + 1;
    globalRe.lastIndex = fn.bodyEnd + 1;
  }
  return cursor === 0 ? source : out + source.slice(cursor);
}

function lowerCuteTnGemmFunction(fn) {
  const motif = collectCuteTnGemmMotif(fn.body);
  if (motif === undefined) return undefined;
  const paramsByName = new Map();
  for (const param of fn.params) {
    const name = cudaParamName(param);
    if (name !== undefined && !paramsByName.has(name)) paramsByName.set(name, param);
  }
  const keepNames = [motif.aPointer, motif.bPointer, motif.dPointer, motif.rows, motif.cols, motif.depth];
  const keptParams = [];
  const seen = new Set();
  for (const name of keepNames) {
    if (seen.has(name)) continue;
    const param = paramsByName.get(name);
    if (param === undefined) return undefined;
    keptParams.push(param);
    seen.add(name);
  }
  const valueType = cudaPointerParamValueType(paramsByName.get(motif.dPointer)) ??
    cudaPointerParamValueType(paramsByName.get(motif.aPointer)) ??
    "float";
  return [
    `${fn.signaturePrefix}(${keptParams.join(", ")}) {`,
    ...cuteTnGemmBody({ ...motif, valueType }).map((line) => `  ${line}`),
    "}",
  ].join("\n");
}

function collectCuteTnGemmMotif(body) {
  const tensors = collectCuteRank2StridedGmemTensors(body);
  if (tensors.length < 3) return undefined;
  const tiles = collectCuteRank2LocalTiles(body);
  const aliases = collectCuteExpressionAliases(body);
  const env = collectCuteIntegerEnv(body);
  for (const d of tensors) {
    if (!cuteExprIsOne(d.stride[1])) continue;
    for (const a of tensors) {
      if (a === d || !cuteExprEquals(a.shape[0], d.shape[0]) || !cuteExprIsOne(a.stride[1])) continue;
      if (!cuteExprEquals(a.stride[0], a.shape[1])) continue;
      for (const b of tensors) {
        if (b === d || b === a) continue;
        if (!cuteExprEquals(b.shape[0], d.shape[1]) || !cuteExprEquals(b.shape[1], a.shape[1])) continue;
        if (!cuteExprEquals(b.stride[0], b.shape[1]) || !cuteExprIsOne(b.stride[1])) continue;
        if (!cuteExprEquals(d.stride[0], d.shape[1])) continue;
        const aTile = tiles.find((tile) => tile.base === a.name);
        const bTile = tiles.find((tile) => tile.base === b.name);
        const dTile = tiles.find((tile) => tile.base === d.name);
        if (aTile === undefined || bTile === undefined || dTile === undefined) continue;
        if (!cuteExprEquals(aTile.extents[0], dTile.extents[0])) continue;
        if (!cuteExprEquals(aTile.extents[1], bTile.extents[1])) continue;
        if (!cuteExprEquals(bTile.extents[0], dTile.extents[1])) continue;
        const aCoords = aTile.coords.map((coord) => resolveCuteExpressionAlias(coord, aliases));
        const bCoords = bTile.coords.map((coord) => resolveCuteExpressionAlias(coord, aliases));
        const dCoords = dTile.coords.map((coord) => resolveCuteExpressionAlias(coord, aliases));
        if (aCoords.length !== 2 || bCoords.length !== 2 || dCoords.length !== 2) continue;
        if (!cuteCoordIsWildcard(aCoords[1]) || !cuteCoordIsWildcard(bCoords[1])) continue;
        if (!cuteExprEquals(aCoords[0], dCoords[0]) || !cuteExprEquals(bCoords[0], dCoords[1])) continue;
        const blockRows = evaluateTemplateIntegerExpression(aTile.extents[0], env) ?? aTile.extents[0];
        const blockCols = evaluateTemplateIntegerExpression(bTile.extents[0], env) ?? bTile.extents[0];
        return {
          aPointer: a.pointer,
          bPointer: b.pointer,
          dPointer: d.pointer,
          rows: d.shape[0],
          cols: d.shape[1],
          depth: a.shape[1],
          blockRows,
          blockCols,
          blockRow: dCoords[0],
          blockCol: dCoords[1],
        };
      }
    }
  }
  return undefined;
}

function collectCuteRank2LocalTiles(body) {
  const out = [];
  const re = /\b(?:auto|Tensor)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*local_tile\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*make_(?:tile|shape)\s*\(\s*Int<([^>]+)>\s*\{\s*\}\s*,\s*Int<([^>]+)>\s*\{\s*\}\s*\)\s*,\s*make_coord\s*\(([^)]*)\)\s*\)\s*;/gu;
  for (const match of body.matchAll(re)) {
    if (match[1] === undefined || match[2] === undefined || match[3] === undefined || match[4] === undefined || match[5] === undefined) continue;
    const coords = splitTopLevel(match[5]).map((item) => item.trim());
    if (coords.length !== 2 || coords.some((item) => item.length === 0)) continue;
    out.push({
      name: match[1],
      base: match[2],
      extents: [match[3].trim(), match[4].trim()],
      coords,
    });
  }
  return out;
}

export function collectCuteExpressionAliases(body) {
  const aliases = new Map(collectCuteIndexAliases(body));
  for (const match of body.matchAll(/\b(?:const\s+)?(?:int|auto)\s+([^;]*\b(?:blockIdx|threadIdx|blockDim|gridDim)\.[xyz][^;]*)\s*;/gu)) {
    const declarators = splitTopLevel(match[1] ?? "").map((item) => item.trim()).filter(Boolean);
    for (const declarator of declarators) {
      const parsed = /^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\s\S]+)$/u.exec(declarator);
      if (parsed?.[1] === undefined || parsed[2] === undefined) continue;
      aliases.set(parsed[1], normalizeCuteIndexExpression(parsed[2]));
    }
  }
  return aliases;
}

function normalizeCuteIndexExpression(expr) {
  return expr
    .replace(/\(\s*\(\s*int\s*\)\s*([01])\s*\)/gu, "$1")
    .replace(/\btrue\b/gu, "1")
    .replace(/\bfalse\b/gu, "0")
    .trim();
}

export function resolveCuteExpressionAlias(expr, aliases) {
  const resolved = resolveCuteIndexAlias(expr, aliases);
  return normalizeCuteIndexExpression(resolved);
}

function cuteCoordIsWildcard(expr) {
  return expr.trim() === "_";
}

function cuteTnGemmBody(motif) {
  const tid = "bg_cute_tid";
  const linear = "bg_cute_linear";
  const localRow = "bg_cute_row_local";
  const localCol = "bg_cute_col_local";
  const row = "bg_cute_row";
  const col = "bg_cute_col";
  const kk = "bg_cute_k";
  const acc = "bg_cute_acc";
  const stride = "((blockDim.x * blockDim.y) * blockDim.z)";
  return [
    `int ${tid} = threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y);`,
    `for (int ${linear} = ${tid}; ${linear} < (${motif.blockRows} * ${motif.blockCols}); ${linear} = ${linear} + ${stride}) {`,
    `  int ${localRow} = ${linear} / ${motif.blockCols};`,
    `  int ${localCol} = ${linear} % ${motif.blockCols};`,
    `  int ${row} = ((${motif.blockRow}) * ${motif.blockRows}) + ${localRow};`,
    `  int ${col} = ((${motif.blockCol}) * ${motif.blockCols}) + ${localCol};`,
    `  if (${row} < ${motif.rows} && ${col} < ${motif.cols}) {`,
    `    float ${acc} = 0.0f;`,
    `    for (int ${kk} = 0; ${kk} < ${motif.depth}; ${kk} = ${kk} + 1) {`,
    `      ${acc} = ${acc} + ((float)${motif.aPointer}[(${row} * ${motif.depth}) + ${kk}] * (float)${motif.bPointer}[(${col} * ${motif.depth}) + ${kk}]);`,
    "    }",
    `    ${motif.dPointer}[(${row} * ${motif.cols}) + ${col}] = (${motif.valueType})${acc};`,
    "  }",
    "}",
  ];
}

export function normalizeCute1dAffineTileCopies(source) {
  if (!/\b(?:Tensor|auto)\b[\s\S]*\blocal_tile\s*\(/u.test(source) || !/\brecast\s*</u.test(source)) return source;
  let out = source;
  let cursor = 0;
  while (cursor < out.length) {
    const startMatch = /\bTensor\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*make_tensor\s*\(/gu.exec(out.slice(cursor));
    if (startMatch === null) break;
    const start = cursor + startMatch.index;
    const finalCopy = findCuteFinalTileCopy(out, start);
    if (finalCopy === undefined) {
      cursor = start + startMatch[0].length;
      continue;
    }
    const segment = out.slice(start, finalCopy.end);
    const replacement = lowerCute1dAffineTileSegment(segment);
    if (replacement === undefined) {
      cursor = start + startMatch[0].length;
      continue;
    }
    out = `${out.slice(0, start)}${replacement}${out.slice(finalCopy.end)}`;
    cursor = start + replacement.length;
  }
  return out;
}

function findCuteFinalTileCopy(source, start) {
  const copyRe = /\bcopy\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu;
  copyRe.lastIndex = start;
  let match;
  while ((match = copyRe.exec(source)) !== null) {
    const prefix = source.slice(start, match.index);
    if (!/\brecast\s*<\s*(?:half|float)\s*>\s*\(/u.test(prefix)) continue;
    return { end: copyRe.lastIndex };
  }
  return undefined;
}

function lowerCute1dAffineTileSegment(segment) {
  const baseTensors = new Map();
  const baseRe = /\bTensor\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor\s*\(\s*make_gmem_ptr\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*,\s*make_shape\s*\(([^)]*)\)\s*\)\s*;/gu;
  for (const match of segment.matchAll(baseRe)) {
    if (match[1] && match[2]) baseTensors.set(match[1], { pointer: match[2], shape: (match[3] ?? "").trim() });
  }
  if (baseTensors.size === 0) return undefined;

  const tiles = new Map();
  const tileRe = /\bTensor\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*local_tile\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*make_shape\s*\(\s*Int<([^>]+)>\s*\{\s*\}\s*\)\s*,\s*make_coord\s*\(([^)]*)\)\s*\)\s*;/gu;
  for (const match of segment.matchAll(tileRe)) {
    const name = match[1];
    const base = match[2];
    const rawLength = match[3];
    const coord = match[4];
    const baseInfo = base === undefined ? undefined : baseTensors.get(base);
    const length = rawLength === undefined ? undefined : evaluateTemplateIntegerExpression(rawLength, new Map());
    if (!name || !baseInfo || length === undefined || coord === undefined) continue;
    tiles.set(name, { pointer: baseInfo.pointer, length, coord: coord.trim() });
  }
  if (tiles.size === 0) return undefined;

  const registers = new Map();
  for (const match of segment.matchAll(/\bTensor\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor_like\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu)) {
    if (match[1] && match[2]) registers.set(match[1], match[2]);
  }

  const recasts = new Map();
  for (const match of segment.matchAll(/\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*recast\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu)) {
    if (match[1] && match[2] && match[3]) recasts.set(match[1], { type: match[2], source: match[3] });
  }

  const scalarVectors = new Map();
  for (const match of segment.matchAll(/\b(?:half2|float2)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*\2\s*\}\s*;/gu)) {
    if (match[1] && match[2]) scalarVectors.set(match[1], match[2]);
  }

  const finalCopy = [...segment.matchAll(/\bcopy\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu)].at(-1);
  const finalHalfAlias = finalCopy?.[1];
  const outTileName = finalCopy?.[2];
  const outTile = outTileName === undefined ? undefined : tiles.get(outTileName);
  const outVectorAlias = finalHalfAlias === undefined ? undefined : recasts.get(finalHalfAlias)?.source;
  if (!outTile || outVectorAlias === undefined) return undefined;

  const forMatch = /for\s*\(\s*int\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\s*;\s*\1\s*<\s*size\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;[^)]*\)\s*\{([\s\S]*?)\}/u.exec(segment);
  if (forMatch?.[1] === undefined || forMatch[2] === undefined || forMatch[3] === undefined) return undefined;
  const loopIndex = forMatch[1];
  const loopAlias = forMatch[2];
  const loopBody = forMatch[3];
  if (loopAlias !== outVectorAlias) return undefined;
  const assignment = new RegExp(`\\b${escapeRegExp(outVectorAlias)}\\s*\\(\\s*${escapeRegExp(loopIndex)}\\s*\\)\\s*=\\s*([^;]+);`, "u").exec(loopBody);
  const rhs = assignment?.[1];
  if (rhs === undefined) return undefined;

  const aliasPointers = new Map();
  for (const [alias, recast] of recasts) {
    if (recast.type !== "half2" && recast.type !== "float2") continue;
    const regSource = registers.get(recast.source);
    const tile = regSource === undefined ? undefined : tiles.get(regSource);
    if (tile !== undefined) aliasPointers.set(alias, tile.pointer);
  }
  if (!aliasPointers.has(outVectorAlias)) aliasPointers.set(outVectorAlias, outTile.pointer);

  const item = "bg_cute_i";
  const pos = "bg_cute_pos";
  let scalarExpression = rhs;
  for (const [alias, pointer] of aliasPointers) {
    scalarExpression = scalarExpression.replace(new RegExp(`\\b${escapeRegExp(alias)}\\s*\\(\\s*${escapeRegExp(loopIndex)}\\s*\\)`, "gu"), `${pointer}[${pos}]`);
  }
  for (const [vectorName, scalarName] of scalarVectors) {
    scalarExpression = scalarExpression.replace(new RegExp(`\\b${escapeRegExp(vectorName)}\\b`, "gu"), scalarName);
  }
  if (/\b[A-Za-z_][A-Za-z0-9_]*\s*\(/u.test(scalarExpression)) return undefined;

  const offset = `((${outTile.coord}) * ${outTile.length})`;
  return [
    `for (int ${item} = 0; ${item} < ${outTile.length}; ++${item}) {`,
    `  int ${pos} = ${offset} + ${item};`,
    `  ${outTile.pointer}[${pos}] = ${scalarExpression};`,
    "}",
  ].join("\n");
}
