/**
 * Flash Attention v2 forward (Dao 2023) — single fused kernel.
 *
 * Inputs:
 *   Q: [B, H, Sq, D]
 *   K: [B, H, Sk, D]
 *   V: [B, H, Sk, D]
 *   mask (optional): [B', H', Sq, Sk] additive logits mask, broadcastable
 *                    over batch and head dims (B' in {1, B}, H' in {1, H}).
 *   scale: scalar multiplier (typically 1 / sqrt(D)).
 *
 * Output: [B, H, Sq, D]
 *
 * v0 scope per PRD-011.5 review:
 *   - Forward-only. No backward (the recompute-from-Q,K,V backward kernel
 *     is its own follow-on PRD).
 *   - Fixed tile sizes (BR=32, BC=32). The kernel handles any D ≤ 128
 *     by carrying d in the per-thread accumulator; larger D fall back
 *     by being processed in chunks by the caller.
 *   - One workgroup per (B, H, q_block). Each workgroup walks the K
 *     blocks once, maintaining running (m_i, l_i, O_i) — the online
 *     softmax of Dao 2023 §3.1.
 *   - f32 throughout. f16/AMP variant is PRD-012a's job.
 *
 * Numerical contract: matches composed attention (Q @ K^T scaled →
 * softmax → @ V) within 1e-4 abs on f32. Workgroup-internal reductions
 * may reorder additions vs the composed reference; 1e-4 is the honest
 * tolerance (see PRD-012 review §Q5 — bitwise determinism is a non-goal).
 */

import {
  runDirect,
  type KernelDescriptor,
  type DirectDispatchResult,
} from "../runner.js";
import { KernelError, type KernelDevice } from "../types.js";

const BR = 32;
const BC = 32;
// MAX_D bounds the workgroup-shared K_tile + V_tile memory:
//   2 × BR × MAX_D × 4 bytes = workgroup memory.
// WebGPU min spec is 16384 bytes → MAX_D ≤ 64 with BR=BC=32.
// Larger D should chunk D-dimension across multiple FA passes (PRD-014b).
const MAX_D = 64;

const WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> Mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;

struct Params {
  B: u32,
  H: u32,
  Sq: u32,
  Sk: u32,
  D: u32,
  has_mask: u32,
  mask_B: u32,    // 1 or B
  mask_H: u32,    // 1 or H
  scale: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};
@group(0) @binding(5) var<uniform> params: Params;

// One workgroup per (batch * head, q-block). Inside the workgroup, each
// thread handles a single Q row of its q-block and walks all K-blocks
// once, maintaining the running (m_i, l_i, O_i).
const BR_LOCAL: u32 = ${BR}u;
const BC_LOCAL: u32 = ${BC}u;
const MAX_D_LOCAL: u32 = ${MAX_D}u;

var<workgroup> K_tile: array<f32, BC_LOCAL * MAX_D_LOCAL>;
var<workgroup> V_tile: array<f32, BC_LOCAL * MAX_D_LOCAL>;

@compute @workgroup_size(${BR}, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let bh = wg_id.x;                         // 0..B*H
  let q_block = wg_id.y;                    // 0..ceil(Sq/BR)
  let i_local = lid.x;                      // 0..BR-1
  let q_idx = q_block * BR_LOCAL + i_local; // global Q row

  let b = bh / params.H;
  let h = bh % params.H;

  // WGSL requires uniform control flow at workgroupBarrier(). Threads
  // whose q_idx is out of range must STILL execute the loop body and
  // hit every barrier — they just refrain from writing the final output.
  let active = q_idx < params.Sq;
  let q_idx_safe = select(0u, q_idx, active);

  let qkv_row_base = ((b * params.H + h) * params.Sq + q_idx_safe) * params.D;
  let k_row_base_bh = (b * params.H + h) * params.Sk * params.D;
  let mask_b = select(0u, b, params.mask_B > 1u);
  let mask_h = select(0u, h, params.mask_H > 1u);
  let mask_row_base = ((mask_b * params.mask_H + mask_h) * params.Sq + q_idx) * params.Sk;

  // Load Q row for this thread into registers (private array indexed by d).
  var q_reg: array<f32, MAX_D_LOCAL>;
  for (var d: u32 = 0u; d < params.D; d = d + 1u) {
    q_reg[d] = Q[qkv_row_base + d];
  }

  // Running softmax state for this row.
  var m_i: f32 = -3.40282347e38;  // -INF
  var l_i: f32 = 0.0;
  var O_i: array<f32, MAX_D_LOCAL>;
  for (var d: u32 = 0u; d < params.D; d = d + 1u) { O_i[d] = 0.0; }

  let num_k_blocks = (params.Sk + BC_LOCAL - 1u) / BC_LOCAL;
  for (var kb: u32 = 0u; kb < num_k_blocks; kb = kb + 1u) {
    let k_start = kb * BC_LOCAL;

    // Cooperatively load this K and V block into workgroup memory.
    // Each thread loads BC/BR rows × D elements.
    let rows_per_thread = (BC_LOCAL + BR_LOCAL - 1u) / BR_LOCAL;
    for (var t: u32 = 0u; t < rows_per_thread; t = t + 1u) {
      let local_row = i_local + t * BR_LOCAL;
      if (local_row < BC_LOCAL) {
        let k_idx = k_start + local_row;
        let in_bounds = k_idx < params.Sk;
        for (var d: u32 = 0u; d < params.D; d = d + 1u) {
          let kv_idx = k_row_base_bh + k_idx * params.D + d;
          let kv_safe = select(0.0, K[kv_idx], in_bounds);
          K_tile[local_row * MAX_D_LOCAL + d] = kv_safe;
          V_tile[local_row * MAX_D_LOCAL + d] = select(0.0, V[kv_idx], in_bounds);
        }
      }
    }
    workgroupBarrier();

    // Compute scores S_ij = Q_i @ K_j^T (BC entries, one per K row in block).
    var S: array<f32, BC_LOCAL>;
    var m_ij: f32 = -3.40282347e38;
    for (var j: u32 = 0u; j < BC_LOCAL; j = j + 1u) {
      let k_idx = k_start + j;
      if (k_idx < params.Sk) {
        var s: f32 = 0.0;
        for (var d: u32 = 0u; d < params.D; d = d + 1u) {
          s = s + q_reg[d] * K_tile[j * MAX_D_LOCAL + d];
        }
        s = s * params.scale;
        if (params.has_mask == 1u) {
          s = s + Mask[mask_row_base + k_idx];
        }
        S[j] = s;
        if (s > m_ij) { m_ij = s; }
      } else {
        S[j] = -3.40282347e38;
      }
    }

    // Online softmax update: rescale O_i by exp(m_i_prev - m_i_new).
    let m_new = max(m_i, m_ij);
    let alpha = exp(m_i - m_new);
    var l_ij: f32 = 0.0;
    for (var j: u32 = 0u; j < BC_LOCAL; j = j + 1u) {
      let k_idx = k_start + j;
      if (k_idx < params.Sk) {
        let p = exp(S[j] - m_new);
        S[j] = p;
        l_ij = l_ij + p;
      } else {
        S[j] = 0.0;
      }
    }
    // O_i ← alpha * O_i + P @ V_j
    for (var d: u32 = 0u; d < params.D; d = d + 1u) {
      var acc: f32 = alpha * O_i[d];
      for (var j: u32 = 0u; j < BC_LOCAL; j = j + 1u) {
        acc = acc + S[j] * V_tile[j * MAX_D_LOCAL + d];
      }
      O_i[d] = acc;
    }
    l_i = alpha * l_i + l_ij;
    m_i = m_new;

    workgroupBarrier();
  }

  // Final normalize: O_i / l_i, write back to global — only for active threads.
  let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
  if (active) {
    for (var d: u32 = 0u; d < params.D; d = d + 1u) {
      Out[qkv_row_base + d] = O_i[d] * inv_l;
    }
  }
}
`;

const DESCRIPTOR_WITH_MASK: KernelDescriptor = {
  name: "flash_attention_v2_masked",
  wgsl: WGSL,
  workgroupSize: [BR, 1, 1],
};

const DESCRIPTOR_NO_MASK: KernelDescriptor = {
  // Same WGSL; different name keeps the pipeline cache from confusing
  // the masked-vs-unmasked binding-count cases.
  name: "flash_attention_v2",
  wgsl: WGSL,
  workgroupSize: [BR, 1, 1],
};

/**
 * Direct-dispatch FA-v2 forward. Inputs and output are GPUBuffers.
 * `mask` may be null; pass a zero-length buffer in its place (the
 * shader's has_mask flag gates the read). The mask shape must be
 * broadcastable over batch/head dims (B' in {1, B}, H' in {1, H}).
 */
export function flashAttentionDirect(
  device: KernelDevice,
  Q: GPUBuffer,
  K: GPUBuffer,
  V: GPUBuffer,
  mask: GPUBuffer | null,
  shapes: {
    B: number;
    H: number;
    Sq: number;
    Sk: number;
    D: number;
    maskB?: number;
    maskH?: number;
  },
  scale: number,
): DirectDispatchResult {
  if (shapes.D > MAX_D) {
    throw new KernelError(
      `flash_attention: D=${shapes.D} exceeds MAX_D=${MAX_D}. ` +
        `Larger head dims need PRD-012a's tiled-D variant.`,
    );
  }
  const params = new Uint32Array(12);
  params[0] = shapes.B;
  params[1] = shapes.H;
  params[2] = shapes.Sq;
  params[3] = shapes.Sk;
  params[4] = shapes.D;
  params[5] = mask ? 1 : 0;
  params[6] = shapes.maskB ?? (mask ? shapes.B : 1);
  params[7] = shapes.maskH ?? (mask ? shapes.H : 1);
  // params[8] is scale as f32 — pack via DataView.
  const view = new DataView(params.buffer);
  view.setFloat32(8 * 4, scale, true);

  const outputLength = shapes.B * shapes.H * shapes.Sq * shapes.D;
  const numQBlocks = Math.ceil(shapes.Sq / BR);

  // The kernel binds Mask whether or not it's present; pass a 1-element
  // dummy GPUBuffer if mask is null to satisfy the bind group layout.
  const effectiveMask = mask ?? makeDummyBuffer(device);
  try {
    return runDirect(device, mask ? DESCRIPTOR_WITH_MASK : DESCRIPTOR_NO_MASK, {
      inputBuffers: [Q, K, V, effectiveMask],
      outputLength,
      params,
      dispatchCount: [shapes.B * shapes.H * BR, numQBlocks, 1],
      cacheKeySuffix: `f32-${mask ? "masked" : "unmasked"}-D${shapes.D}-BR${BR}-BC${BC}`,
    });
  } finally {
    if (!mask) {
      effectiveMask.destroy();
    }
  }
}

function makeDummyBuffer(device: KernelDevice): GPUBuffer {
  // 16-byte zero buffer satisfying STORAGE | COPY_DST + read-only-storage binding.
  // Cheap to allocate per call; could be cached on the device adapter in PRD-012a.
  const impl = device.gpu;
  const buf = impl.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  impl.queue.writeBuffer(buf, 0, new Float32Array(4));
  return buf;
}
