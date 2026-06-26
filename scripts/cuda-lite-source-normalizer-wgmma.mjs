export function normalizeWgmmaTmaGemmKernels(source, helpers) {
  if (!/\bCUtensorMap\b/u.test(source)) return source;
  if (!/\bcp_async_bulk_tensor_2d_global_to_shared\b/u.test(source) && !/\bwgmma\.mma_async\b/u.test(source) && !/\bWGMMA_M64N128K16_/u.test(source)) return source;
  let out = "";
  let cursor = 0;
  const globalRe = /\b__global__\b/gu;
  let match;
  while ((match = globalRe.exec(source)) !== null) {
    const globalStart = match.index;
    if (globalStart < cursor) continue;
    const fn = helpers.parseCudaGlobalFunction(source, globalStart);
    if (fn === undefined) {
      globalRe.lastIndex = globalStart + "__global__".length;
      continue;
    }
    const replacement = lowerWgmmaTmaGemmFunction(fn, helpers);
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

function lowerWgmmaTmaGemmFunction(fn, helpers) {
  const motif = collectWgmmaTmaGemmMotif(fn, helpers);
  if (motif === undefined) return undefined;
  const paramsByName = new Map();
  for (const param of fn.params) {
    const name = helpers.cudaParamName(param);
    if (name !== undefined && !paramsByName.has(name)) paramsByName.set(name, param);
  }
  const keptParams = [
    paramsByName.get(motif.rows),
    paramsByName.get(motif.cols),
    paramsByName.get(motif.inner),
    paramsByName.get(motif.outputPointer),
    `half *${motif.aPointer}`,
    `half *${motif.bPointer}`,
  ];
  if (keptParams.some((param) => param === undefined)) return undefined;
  return [
    `${fn.signaturePrefix}(${keptParams.join(", ")}) {`,
    ...wgmmaTmaGemmBody(motif).map((line) => `  ${line}`),
    "}",
  ].join("\n");
}

function collectWgmmaTmaGemmMotif(fn, helpers) {
  const body = fn.body;
  if (!/\bcp_async_bulk_tensor_2d_global_to_shared\b/u.test(body) || !/\bWGMMA_M64N128K16_/u.test(body)) return undefined;
  const paramsByName = new Map();
  for (const param of fn.params) {
    const name = helpers.cudaParamName(param);
    if (name !== undefined && !paramsByName.has(name)) paramsByName.set(name, param);
  }
  const rows = paramsByName.has("M") ? "M" : undefined;
  const cols = paramsByName.has("N") ? "N" : undefined;
  const inner = paramsByName.has("K") ? "K" : undefined;
  const outputPointer = [...paramsByName].find(([name, param]) =>
    name !== rows && name !== cols && name !== inner &&
    /\*/u.test(param) &&
    !/\bCUtensorMap\b/u.test(param) &&
    /\b(?:half|__half|float)\b/u.test(param)
  )?.[0];
  const tensorMaps = [...paramsByName].filter(([, param]) => /\bCUtensorMap\b/u.test(param)).map(([name]) => name);
  if (rows === undefined || cols === undefined || inner === undefined || outputPointer === undefined || tensorMaps.length < 2) return undefined;
  const tileRows = helpers.sourceMentionsIdentifier(body, "BM") ? "BM" : "128";
  const tileCols = helpers.sourceMentionsIdentifier(body, "BN") ? "BN" : "128";
  const blockCol = helpers.sourceMentionsIdentifier(body, "BLOCK_SWIZZLE")
    ? "(((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x)"
    : "blockIdx.x";
  return {
    rows,
    cols,
    inner,
    outputPointer,
    aPointer: `${tensorMaps[0]}__base`,
    bPointer: `${tensorMaps[1]}__base`,
    valueType: helpers.cudaPointerParamValueType(paramsByName.get(outputPointer)) ?? "half",
    tileRows,
    tileCols,
    blockRow: "blockIdx.y",
    blockCol,
  };
}

function wgmmaTmaGemmBody(motif) {
  const tid = "bg_wgmma_tid";
  const linear = "bg_wgmma_linear";
  const localRow = "bg_wgmma_local_row";
  const localCol = "bg_wgmma_local_col";
  const row = "bg_wgmma_row";
  const col = "bg_wgmma_col";
  const k = "bg_wgmma_k";
  const acc = "bg_wgmma_acc";
  const stride = "((blockDim.x * blockDim.y) * blockDim.z)";
  return [
    `int ${tid} = threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y);`,
    `for (int ${linear} = ${tid}; ${linear} < (${motif.tileRows} * ${motif.tileCols}); ${linear} = ${linear} + ${stride}) {`,
    `  int ${localRow} = ${linear} / ${motif.tileCols};`,
    `  int ${localCol} = ${linear} % ${motif.tileCols};`,
    `  int ${row} = (${motif.blockRow} * ${motif.tileRows}) + ${localRow};`,
    `  int ${col} = (${motif.blockCol} * ${motif.tileCols}) + ${localCol};`,
    `  if (${row} < ${motif.rows} && ${col} < ${motif.cols}) {`,
    `    float ${acc} = 0.0f;`,
    `    for (int ${k} = 0; ${k} < ${motif.inner}; ${k} = ${k} + 1) {`,
    `      ${acc} = ${acc} + ((float)${motif.aPointer}[(${row} * ${motif.inner}) + ${k}] * (float)${motif.bPointer}[(${col} * ${motif.inner}) + ${k}]);`,
    "    }",
    `    ${motif.outputPointer}[(${row} * ${motif.cols}) + ${col}] = (${motif.valueType})${acc};`,
    "  }",
    "}",
  ];
}
