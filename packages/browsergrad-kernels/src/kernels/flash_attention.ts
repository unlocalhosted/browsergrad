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
  type DirectDispatchProfileOptions,
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

// One workgroup per (batch * head, q-block). Each thread handles a single Q
// row and walks all K rows once, maintaining running online-softmax state.
const BR_LOCAL: u32 = ${BR}u;
const MAX_D_LOCAL: u32 = ${MAX_D}u;

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

  let is_active = q_idx < params.Sq;
  let q_idx_safe = select(0u, q_idx, is_active);

  let qkv_row_base = ((b * params.H + h) * params.Sq + q_idx_safe) * params.D;
  let k_row_base_bh = (b * params.H + h) * params.Sk * params.D;
  let mask_b = select(0u, b, params.mask_B > 1u);
  let mask_h = select(0u, h, params.mask_H > 1u);
  let mask_row_base = ((mask_b * params.mask_H + mask_h) * params.Sq + q_idx) * params.Sk;
  var q_reg: array<f32, MAX_D_LOCAL>;
  for (var d: u32 = 0u; d < params.D; d = d + 1u) {
    q_reg[d] = select(0.0, Q[qkv_row_base + d], is_active);
  }

  var m_i: f32 = -1.0e20;  // finite -INF sentinel accepted by WGSL parsers
  var l_i: f32 = 0.0;
  var O_i: array<f32, MAX_D_LOCAL>;
  for (var d: u32 = 0u; d < params.D; d = d + 1u) { O_i[d] = 0.0; }

  for (var j: u32 = 0u; j < params.Sk; j = j + 1u) {
    var s: f32 = 0.0;
    for (var d: u32 = 0u; d < params.D; d = d + 1u) {
      s = s + q_reg[d] * K[k_row_base_bh + j * params.D + d];
    }
    s = s * params.scale;
    if (params.has_mask == 1u) {
      s = s + Mask[mask_row_base + j];
    }

    let m_new = max(m_i, s);
    let alpha = exp(m_i - m_new);
    let p = exp(s - m_new);
    for (var d: u32 = 0u; d < params.D; d = d + 1u) {
      O_i[d] = alpha * O_i[d] + p * V[k_row_base_bh + j * params.D + d];
    }
    l_i = alpha * l_i + p;
    m_i = m_new;
  }

  if (is_active) {
    let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
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
  profile?: DirectDispatchProfileOptions,
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
      ...(profile ? { profile } : {}),
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
