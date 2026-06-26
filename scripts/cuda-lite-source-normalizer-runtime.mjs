export function normalizeDeviceRuntimeNoopCalls(source) {
  return source
    .replace(/\bcudaStreamCreateWithFlags\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^;{}]+?)\s*\)\s*;/gu, "$1 = $2;")
    .replace(/\bcudaStreamCreate\s*\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu, "$1 = cudaStreamDefault;")
    .replace(/\bcudaStreamDestroy\s*\([^;{}]*\)\s*;/gu, ";")
    .replace(/\bcudaStreamSynchronize\s*\([^;{}]*\)\s*;/gu, ";")
    .replace(/\bcudaFree\s*\([^;{}]*\)\s*;/gu, ";");
}
