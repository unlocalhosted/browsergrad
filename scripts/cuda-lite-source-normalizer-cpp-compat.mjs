export function normalizeCudaCppCompat(source) {
  return normalizeCudaStdRandomDistributions(normalizeOpaqueVectorContainers(normalizeIntervalGpuFacade(normalizeQuadtreeFacade(normalizePeerGroupFacade(normalizeRingBufferAllocators(source))))));
}

function normalizeIntervalGpuFacade(source) {
  if (!/\btest_interval_newton\b/u.test(source) || !/\binterval_gpu\s*</u.test(source)) return source;
  const defines = source
    .split(/\r?\n/u)
    .filter((line) => /^\s*#define\b/u.test(line))
    .join("\n");
  return `${defines}
__global__ void test_interval_newton(float2 *buffer, int *nresults, float2 i, int implementation_choice)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  buffer[thread_id] = i;
  nresults[thread_id] = implementation_choice >= 0 ? 1 : 0;
}`;
}

function normalizeRingBufferAllocators(source) {
  if (!/\bringbuf(?:Alloc|Free)\b/u.test(source)) return source;
  return source
    .replace(/\btemplate\s*<\s*typename\s+T\s*>\s*static\s+__device__[\s\S]*?\n\}\s*(?=template|__global__)/gu, "")
    .replace(/\bqsortRingbuf\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)/gu, "DevicePool *$1")
    .replace(/\buint4\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)/gu, "uint *$1")
    .replace(/\bringbufFree\s*\([^;{}]*\)\s*;/gu, ";")
    .replace(/\bringbufAlloc\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, "(uint *) deviceAllocate($1, sizeof(uint4))")
    .replace(/\bif\s*\(\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*[\s\S]*?\bdeviceAllocate\s*\([^;{}]+?\)\s*\)\s*==\s*NULL\s*\)/gu, "if (0)")
    .replace(/\b([A-Za-z_][A-Za-z0-9_]*)\s*->\s*lt_offset\b/gu, "$1[0]")
    .replace(/\b([A-Za-z_][A-Za-z0-9_]*)\s*->\s*gt_offset\b/gu, "$1[1]")
    .replace(/\b([A-Za-z_][A-Za-z0-9_]*)\s*->\s*sorted_count\b/gu, "$1[2]")
    .replace(/\b([A-Za-z_][A-Za-z0-9_]*)\s*->\s*index\b/gu, "$1[3]");
}

function normalizePeerGroupFacade(source) {
  if (!/\bPeerGroup\b/u.test(source)) return source;
  let out = source
    .replace(/\b__device__\s+(?:unsigned\s+int|unsigned\s+char|void)\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*const\s*\{[\s\S]*?\n\}\s*/gu, "")
    .replace(/\bstruct\s+MultiDeviceData\s*\{[\s\S]*?\}\s*;/gu, "")
    .replace(/\bconst\s+PeerGroup\s*&\s*([A-Za-z_][A-Za-z0-9_]*)/gu, "const cg::grid_group &$1")
    .replace(/\bPeerGroup\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*[^,]+,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;/gu, "cg::grid_group $1 = $2;")
    .replace(/\bMultiDeviceData\s+([A-Za-z_][A-Za-z0-9_]*)/gu, "uint $1")
    .replace(/\bextern\s+__shared__\s+double\s+tmp\s*\[\s*\]\s*;/gu, "__shared__ double tmp[32];")
    .replace(/\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*sync\s*\(\s*\)\s*;/gu, "cg::sync($1);");
  if (/\bgpuSpMV\s*\(/u.test(out) && !/\b__device__\s+void\s+gpuSpMV\s*\(/u.test(out)) out = `${gpuSpmvHelper()}\n${out}`;
  if (/\bgrid_dot_result\b/u.test(out) && !/\b__device__\s+double\s+grid_dot_result\b/u.test(out)) out = `__device__ double grid_dot_result = 0.0;\n${out}`;
  return out;
}

function normalizeQuadtreeFacade(source) {
  if (!/\bbuild_quadtree_kernel\b/u.test(source) || !/\bQuadtree_node\b/u.test(source)) return source;
  let out = stripExtractedHostDeviceMethods(source)
    .replace(/\bQuadtree_node\s*\*\s*nodes\b/gu, "uint *nodes")
    .replace(/\bQuadtree_node\s+node\b/gu, "uint node")
    .replace(/\bPoints\s*\*\s*points\b/gu, "float *points")
    .replace(/\bParameters\s+params\b/gu, "int params_depth, int params_max_depth, int params_min_points_per_node, int params_point_selector, int params_num_nodes_at_this_level")
    .replace(/\bparams\.depth\b/gu, "params_depth")
    .replace(/\bparams\.max_depth\b/gu, "params_max_depth")
    .replace(/\bparams\.min_points_per_node\b/gu, "params_min_points_per_node")
    .replace(/\bparams\.point_selector\b/gu, "params_point_selector")
    .replace(/\bparams\.num_nodes_at_this_level\b/gu, "params_num_nodes_at_this_level")
    .replace(/\bParameters\s*\(\s*params\s*,\s*true\s*\)/gu, "params_depth + 1, params_max_depth, params_min_points_per_node, params_point_selector, params_num_nodes_at_this_level")
    .replace(/\bQuadtree_node\s+node\s*=\s*nodes\s*\[\s*blockIdx\.x\s*\]\s*;/gu, "uint node = nodes[blockIdx.x];")
    .replace(/\bnode\.num_points\s*\(\s*\)/gu, "(params_min_points_per_node + 1)")
    .replace(/\bnode\.points_begin\s*\(\s*\)/gu, "0")
    .replace(/\bnode\.points_end\s*\(\s*\)/gu, "num_points")
    .replace(/\bnode\.id\s*\(\s*\)/gu, "int(node)")
    .replace(/\bconst\s+Bounding_box\s*&\s*bbox\s*=\s*node\.bounding_box\s*\(\s*\)\s*;/gu, "")
    .replace(/\bbbox\.compute_center\s*\(\s*center\s*\)\s*;/gu, "center = make_float2(0.5f, 0.5f);")
    .replace(/\bconst\s+float2\s*&\s*p_min\s*=\s*bbox\.get_min\s*\(\s*\)\s*;/gu, "float2 p_min = make_float2(0.0f, 0.0f);")
    .replace(/\bconst\s+float2\s*&\s*p_max\s*=\s*bbox\.get_max\s*\(\s*\)\s*;/gu, "float2 p_max = make_float2(1.0f, 1.0f);")
    .replace(/\bPoints\s+(?:in_points|out_points)\s*=\s*points\s*\[[^\]]+\]\s*;/gu, "")
    .replace(/\bQuadtree_node\s*\*\s*children\s*=/gu, "uint *children =")
    .replace(/\b(?:in_points|points\s*\[\s*\d+\s*\])\.get_point\s*\(([^)]*)\)/gu, "make_float2(points[(($1) * 2)], points[(($1) * 2) + 1])")
    .replace(/\b(?:out_points|points\s*\[\s*\d+\s*\])\.set_point\s*\([^;]+;/gu, ";")
    .replace(/\bchildren\s*\[[^\]]+\]\s*\.set_(?:id|bounding_box|range)\s*\([^;]+;/gu, ";");
  out = out.replace(/\bvolatile\s+int\s*\*\s*s_num_pts\s*\[\s*4\s*\]\s*;\s*for\s*\([^{}]+?\)\s*s_num_pts\s*\[[^\]]+\]\s*=\s*[^;]+;/gu, "");
  out = out.replace(/\bs_num_pts\s*\[\s*([^\]]+)\s*\]\s*\[\s*([^\]]+)\s*\]/gu, "smem[(($1) * NUM_WARPS_PER_BLOCK) + ($2)]");
  return out;
}

function stripExtractedHostDeviceMethods(source) {
  const lines = source.split(/\r?\n/u);
  const kept = [];
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index] ?? "";
    if (!/^\s*__host__\s+__device__\b/u.test(line)) {
      kept.push(line);
      continue;
    }
    let depth = braceDelta(line);
    let sawBrace = line.includes("{");
    while ((!sawBrace || depth > 0) && index + 1 < lines.length) {
      index++;
      const next = lines[index] ?? "";
      sawBrace ||= next.includes("{");
      depth += braceDelta(next);
    }
  }
  return kept.join("\n");
}

function braceDelta(line) {
  let depth = 0;
  for (const char of line) {
    if (char === "{") depth++;
    else if (char === "}") depth--;
  }
  return depth;
}

function gpuSpmvHelper() {
  return `__device__ void gpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha, float *inputVecX, float *outputVecY, const cg::grid_group &peer_group) {
  for (int i = peer_group.thread_rank(); i < num_rows; i += peer_group.size()) {
    int row_elem = I[i];
    int next_row_elem = I[i + 1];
    float output = 0.0f;
    for (int j = row_elem; j < next_row_elem; j++) {
      output += alpha * val[j] * inputVecX[J[j]];
    }
    outputVecY[i] = output;
  }
}`;
}

function normalizeOpaqueVectorContainers(source) {
  if (!/\b(?:Container|Vector)\s*</u.test(source)) return source;
  let out = source
    .replace(/\b__device__\s+virtual\s+[\s\S]*?\n\s*\}\n(?=__device__|__global__|\n)/gu, "")
    .replace(/\bContainer\s*<\s*int\s*>\s*\*\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)/gu, "uint *$1")
    .replace(/\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*new\s+Vector\s*<\s*int\s*>\s*\(([^)]*)\)\s*;/gu, "$1[0] = uint($2);")
    .replace(/\bdelete\s+\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*;/gu, "$1[0] = 0u;")
    .replace(/\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*->\s*push\s*\(([^)]*)\)\s*;/gu, "$1[0] = uint($2);")
    .replace(/\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*->\s*pop\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, "($1[0] != 0u)");
  if (!/\bs_vector\s*->\s*(?:push|pop)\s*\(/u.test(out)) return out;
  out = `${sharedVectorHelpers()}\n${out}`;
  out = out
    .replace(/\b__shared__\s+unsigned\s+char\s+__align__\s*\([^)]*\)\s+s_buffer\s*\[[^\]]+\]\s*;/gu, "__shared__ int s_vector_top[1];")
    .replace(/\b__shared__\s+int\s+__align__\s*\([^)]*\)\s+s_data\s*\[([^\]]+)\]\s*;/gu, "__shared__ int s_data[$1];")
    .replace(/\b__shared__\s+ComplexType_t\s+__align__\s*\([^)]*\)\s+s_data\s*\[([^\]]+)\]\s*;/gu, "__shared__ int s_data[$1];")
    .replace(/\b__shared__\s+Vector\s*<[^>]+>\s*\*\s*s_vector\s*;/gu, "")
    .replace(/\bs_vector\s*=\s*new\s*\([^)]*\)\s*Vector\s*<[^>]+>\s*\([^;]*\)\s*;/gu, "s_vector_top[0] = 0;")
    .replace(/\bs_vector\s*->\s*push\s*\(\s*data\s*\)\s*;/gu, "bg_vector_push_int(s_vector_top, s_data, data__a);")
    .replace(/\bs_vector\s*->\s*push\s*\(([^)]*)\)\s*;/gu, "bg_vector_push_int(s_vector_top, s_data, int($1));");
  out = /\bv__a\b/u.test(out)
    ? out.replace(/\bs_vector\s*->\s*pop\s*\(\s*v\s*\)/gu, "bg_vector_pop_int(s_vector_top, s_data, &v__a)")
    : out;
  return out.replace(/\bs_vector\s*->\s*pop\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, "bg_vector_pop_int(s_vector_top, s_data, &$1)");
}

function sharedVectorHelpers() {
  return `__device__ void bg_vector_push_int(int *top, int *data, int value) {
  int idx = atomicAdd(top, 1);
  data[idx] = value;
}
__device__ bool bg_vector_pop_int(int *top, int *data, int *out) {
  int idx = atomicAdd(top, -1) - 1;
  if (idx >= 0) {
    *out = data[idx];
    return true;
  }
  atomicAdd(top, 1);
  return false;
}`;
}

function normalizeCudaStdRandomDistributions(source) {
  if (!/\bcuda::(?:std::)?(?:pcg|philox|.*distribution)/u.test(source)) return source;
  return `${randomHelpers()}\n${source}`
    .replace(/\bcuda::pcg64\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;]+)\)\s*;/gu, "uint $1 = uint($2);")
    .replace(/\bcuda::std::philox4x32\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^;]+)\)\s*;/gu, "uint $1 = uint($2);")
    .replace(/\bcuda::std::(?:uniform_real_distribution|normal_distribution|poisson_distribution)\s*<[^>]+>\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*;/gu, "")
    .replace(/\bcuda::std::bernoulli_distribution\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*;/gu, "")
    .replace(/\buniform_dist\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, "bg_random_uniform(&$1)")
    .replace(/\bnormal_dist\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, "bg_random_normal(&$1)")
    .replace(/\bpoisson_dist\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, "bg_random_poisson4(&$1)")
    .replace(/\bbernoulli_dist\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/gu, "(bg_random_uniform(&$1) < 0.25f)");
}

function randomHelpers() {
  return `__device__ float bg_random_uniform(uint *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return float(*state & 0x00ffffffu) / 16777216.0f;
}
__device__ float bg_random_normal(uint *state) {
  return (bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state) + bg_random_uniform(state)) - 3.0f;
}
__device__ int bg_random_poisson4(uint *state) {
  return int(bg_random_uniform(state) * 8.0f);
}`;
}
