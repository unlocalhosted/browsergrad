export function normalizeCuteFlashAttentionKernels(source, helpers) {
  if (!/\bmake_tensor\s*\(\s*make_gmem_ptr(?:\s*<[^<>]+>)?\s*\(/u.test(source) || !/\bglobal_row_denominator\b/u.test(source)) return source;
  if (!/\bgemm\s*\(/u.test(source) || !/\bexp\s*\(/u.test(source)) return source;
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
    const replacement = lowerCuteFlashAttentionFunction(fn, helpers);
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

function lowerCuteFlashAttentionFunction(fn, helpers) {
  const motif = collectCuteFlashAttentionMotif(fn.body, fn.params, helpers);
  if (motif === undefined) return undefined;
  const paramsByName = new Map();
  for (const param of fn.params) {
    const name = helpers.cudaParamName(param);
    if (name !== undefined && !paramsByName.has(name)) paramsByName.set(name, param);
  }
  const keepNames = [
    motif.qPointer,
    motif.kPointer,
    motif.vPointer,
    motif.oPointer,
    motif.batch,
    motif.heads,
    motif.queryLength,
    motif.keyLength,
    motif.headDim,
    motif.scale,
  ];
  const keptParams = [];
  const seen = new Set();
  for (const name of keepNames) {
    if (seen.has(name)) continue;
    const param = paramsByName.get(name);
    if (param === undefined) return undefined;
    keptParams.push(param);
    seen.add(name);
  }
  const valueType = helpers.cudaPointerParamValueType(paramsByName.get(motif.oPointer)) ??
    helpers.cudaPointerParamValueType(paramsByName.get(motif.qPointer)) ??
    "float";
  return [
    `${fn.signaturePrefix}(${keptParams.join(", ")}) {`,
    ...cuteFlashAttentionBody({ ...motif, valueType }).map((line) => `  ${line}`),
    "}",
  ].join("\n");
}

function collectCuteFlashAttentionMotif(body, params, helpers) {
  const tensors = collectCuteRank4RowMajorGmemTensors(body, helpers);
  if (tensors.length < 4) return undefined;
  const env = helpers.collectCuteIntegerEnv(body);
  const aliases = helpers.collectCuteExpressionAliases(body);
  const q = tensors.find((tensor) => tensor.name === "Q");
  const k = tensors.find((tensor) => tensor.name === "K");
  const v = tensors.find((tensor) => tensor.name === "V");
  const o = tensors.find((tensor) => tensor.name === "O");
  if (q === undefined || k === undefined || v === undefined || o === undefined) return undefined;
  if (!helpers.cuteExprEquals(q.shape[0], o.shape[0]) || !helpers.cuteExprEquals(q.shape[1], o.shape[1]) ||
    !helpers.cuteExprEquals(q.shape[2], o.shape[2]) || !helpers.cuteExprEquals(q.shape[3], o.shape[3])) return undefined;
  if (!helpers.cuteExprEquals(k.shape[0], q.shape[0]) || !helpers.cuteExprEquals(k.shape[1], q.shape[1]) ||
    !helpers.cuteExprEquals(v.shape[0], q.shape[0]) || !helpers.cuteExprEquals(v.shape[1], q.shape[1])) return undefined;
  if (!helpers.cuteExprEquals(k.shape[2], v.shape[2]) || !helpers.cuteExprEquals(k.shape[3], q.shape[3]) || !helpers.cuteExprEquals(v.shape[3], q.shape[3])) return undefined;
  const qTile = collectCuteFlashQueryTile(body, q.name, env, aliases, helpers);
  if (qTile === undefined) return undefined;
  const scale = inferCuteFlashScaleParam(params, body, helpers);
  if (scale === undefined) return undefined;
  if (!helpers.sourceMentionsIdentifier(body, scale)) return undefined;
  return {
    qPointer: q.pointer,
    kPointer: k.pointer,
    vPointer: v.pointer,
    oPointer: o.pointer,
    batch: q.shape[0],
    heads: q.shape[1],
    queryLength: q.shape[2],
    keyLength: k.shape[2],
    headDim: normalizeCuteHeadDim(q.shape[3], body, helpers),
    scale,
    blockQueries: qTile.blockQueries,
    blockBatch: qTile.batch,
    blockHead: qTile.head,
    blockQuery: qTile.queryBlock,
  };
}

function inferCuteFlashScaleParam(params, body, helpers) {
  for (const param of params) {
    if (!/\bfloat\b/u.test(param) || /\*/u.test(param)) continue;
    const name = helpers.cudaParamName(param);
    if (name !== undefined && helpers.sourceMentionsIdentifier(body, name)) return name;
  }
  return undefined;
}

function normalizeCuteHeadDim(expr, body, helpers) {
  if (helpers.sourceMentionsIdentifier(body, "D")) return "D";
  return expr;
}

function collectCuteRank4RowMajorGmemTensors(body, helpers) {
  const out = [];
  const gmemPointer = String.raw`make_gmem_ptr(?:\s*<[^<>]+>)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)`;
  const re = new RegExp(String.raw`\b(?:auto|Tensor)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*make_tensor\s*\(\s*${gmemPointer}\s*,\s*make_layout\s*\(\s*make_shape\s*\(([^)]*)\)\s*,\s*GenRowMajor\s*\{\s*\}\s*\)\s*\)\s*;`, "gu");
  for (const match of body.matchAll(re)) {
    if (match[1] === undefined || match[2] === undefined || match[3] === undefined) continue;
    const shape = helpers.splitTopLevel(match[3]).map(helpers.normalizeCuteScalarExpression);
    if (shape.length !== 4) continue;
    out.push({ name: match[1], pointer: match[2], shape });
  }
  return out;
}

function collectCuteFlashQueryTile(body, qName, env, aliases, helpers) {
  const re = new RegExp(String.raw`\blocal_tile\s*\(\s*${helpers.escapeRegExp(qName)}\s*,\s*make_shape\s*\(\s*_1\s*\{\s*\}\s*,\s*_1\s*\{\s*\}\s*,\s*Int<([^>]+)>\s*\{\s*\}\s*,\s*Int<([^>]+)>\s*\{\s*\}\s*\)\s*,\s*make_coord\s*\(([^)]*)\)\s*\)`, "u");
  const match = re.exec(body);
  if (match?.[1] === undefined || match[3] === undefined) return undefined;
  const coords = helpers.splitTopLevel(match[3]).map((coord) => helpers.resolveCuteExpressionAlias(coord, aliases));
  if (coords.length !== 4) return undefined;
  const blockQueries = helpers.evaluateTemplateIntegerExpression(match[1], env) ?? match[1].trim();
  return {
    blockQueries,
    batch: coords[0],
    head: coords[1],
    queryBlock: coords[2],
  };
}

function cuteFlashAttentionBody(motif) {
  const tid = "bg_attn_tid";
  const linear = "bg_attn_linear";
  const queryLocal = "bg_attn_query_local";
  const dim = "bg_attn_dim";
  const batch = "bg_attn_b";
  const head = "bg_attn_h";
  const query = "bg_attn_q";
  const key = "bg_attn_kv";
  const hidden = "bg_attn_hd";
  const score = "bg_attn_score";
  const maxScore = "bg_attn_max";
  const denom = "bg_attn_denom";
  const acc = "bg_attn_acc";
  const weight = "bg_attn_weight";
  const stride = "((blockDim.x * blockDim.y) * blockDim.z)";
  const qBase = `(((((${batch} * ${motif.heads}) + ${head}) * ${motif.queryLength}) + ${query}) * ${motif.headDim})`;
  const kBase = `(((((${batch} * ${motif.heads}) + ${head}) * ${motif.keyLength}) + ${key}) * ${motif.headDim})`;
  const oBase = qBase;
  return [
    `int ${tid} = threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y);`,
    `for (int ${linear} = ${tid}; ${linear} < (${motif.blockQueries} * ${motif.headDim}); ${linear} = ${linear} + ${stride}) {`,
    `  int ${queryLocal} = ${linear} / ${motif.headDim};`,
    `  int ${dim} = ${linear} % ${motif.headDim};`,
    `  int ${batch} = ${motif.blockBatch};`,
    `  int ${head} = ${motif.blockHead};`,
    `  int ${query} = (${motif.blockQuery} * ${motif.blockQueries}) + ${queryLocal};`,
    `  if (${batch} < ${motif.batch} && ${head} < ${motif.heads} && ${query} < ${motif.queryLength} && ${dim} < ${motif.headDim}) {`,
    `    float ${maxScore} = -3.402823e38f;`,
    `    for (int ${key} = 0; ${key} < ${motif.keyLength}; ${key} = ${key} + 1) {`,
    `      float ${score} = 0.0f;`,
    `      for (int ${hidden} = 0; ${hidden} < ${motif.headDim}; ${hidden} = ${hidden} + 1) {`,
    `        ${score} = ${score} + ((float)${motif.qPointer}[${qBase} + ${hidden}] * (float)${motif.kPointer}[${kBase} + ${hidden}]);`,
    "      }",
    `      ${score} = ${score} * ${motif.scale};`,
    `      ${maxScore} = max(${maxScore}, ${score});`,
    "    }",
    `    float ${denom} = 0.0f;`,
    `    float ${acc} = 0.0f;`,
    `    for (int ${key} = 0; ${key} < ${motif.keyLength}; ${key} = ${key} + 1) {`,
    `      float ${score} = 0.0f;`,
    `      for (int ${hidden} = 0; ${hidden} < ${motif.headDim}; ${hidden} = ${hidden} + 1) {`,
    `        ${score} = ${score} + ((float)${motif.qPointer}[${qBase} + ${hidden}] * (float)${motif.kPointer}[${kBase} + ${hidden}]);`,
    "      }",
    `      float ${weight} = expf((${score} * ${motif.scale}) - ${maxScore});`,
    `      ${denom} = ${denom} + ${weight};`,
    `      ${acc} = ${acc} + (${weight} * (float)${motif.vPointer}[${kBase} + ${dim}]);`,
    "    }",
    `    ${motif.oPointer}[${oBase} + ${dim}] = (${motif.valueType})(${acc} / ${denom});`,
    "  }",
    "}",
  ];
}
