export function normalizeDeviceRuntimeNoopCalls(source) {
  const out = source
    .replace(/\bcudaMalloc\s*\(\s*\(\s*void\s*\*\s*\*\s*\)\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*[^;{}]+?\)\s*;/gu, ";")
    .replace(/\bcudaStreamCreateWithFlags\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^;{}]+?)\s*\)\s*;/gu, "$1 = $2;")
    .replace(/\bcudaStreamCreate\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu, "$1 = cudaStreamDefault;")
    .replace(/\bcudaStreamDestroy\s*\([^;{}]*\)\s*;/gu, ";")
    .replace(/\bcudaStreamSynchronize\s*\([^;{}]*\)\s*;/gu, ";")
    .replace(/\bcudaFree\s*\([^;{}]*\)\s*;/gu, ";")
    .replace(/\bcudaPeekAtLastError\s*\(\s*\)/gu, "0")
    .replace(/\bcudaGetLastError\s*\(\s*\)/gu, "0")
    .replace(/\bcudaGetErrorString\s*\([^)]*\)/gu, "\"\"")
    .replace(/\bcudaSuccess\b/gu, "0");
  return inlineRuntimeAllocationMetadata(out);
}

function inlineRuntimeAllocationMetadata(source) {
  let out = source.replace(
    /\bif\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*==\s*NULL\s*\)\s*\{\s*([A-Za-z_][A-Za-z0-9_]*\s*\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\]\s*=\s*[A-Za-z_][A-Za-z0-9_]*\s*;)\s*;?\s*\}/gu,
    "$1",
  );
  const replacements = [];
  const assignRe = /\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*;/gu;
  for (const match of out.matchAll(assignRe)) {
    if (match.index === undefined || match[1] === undefined || match[2] === undefined || match[3] === undefined) continue;
    if (!out.slice(match.index + match[0].length).includes("<<<")) continue;
    replacements.push({ after: match.index + match[0].length, array: match[1], index: match[2], value: match[3] });
  }
  for (const item of replacements.reverse()) {
    const before = out.slice(0, item.after);
    const after = out.slice(item.after);
    const readRe = new RegExp(String.raw`\b${escapeRegExp(item.array)}\s*\[\s*${escapeRegExp(item.index)}\s*\]`, "gu");
    out = before + after.replace(readRe, item.value);
  }
  return out;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}
