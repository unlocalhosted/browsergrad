/**
 * Pure-JS reference implementations of every WGSL kernel.
 *
 * Three roles, three audiences:
 *   1. Conformance oracle — tests compare WGSL output to these.
 *   2. CPU fallback — consumers without WebGPU can `import { reference }`
 *      and get the same surface that runs on the CPU. Slower; still correct.
 *   3. Lesson material — these are deliberately readable. A student should
 *      be able to read `referenceAttention` and understand attention.
 *
 * No micro-optimizations. Clarity over speed. The WGSL versions are where
 * speed lives.
 */

import {
  referenceGelu,
  referenceLayerNorm,
  referenceMatmul,
  referenceRelu,
  referenceSoftmax,
} from "./reference-basic.js";
import {
  estimateAttentionMemory,
  referenceAttention,
  referenceBlockedAttention,
  referenceFlashAttention,
  referenceFlashAttentionBackward,
  referenceFusedAttention,
} from "./reference-attention.js";

export {
  referenceGelu,
  referenceLayerNorm,
  referenceMatmul,
  referenceRelu,
  referenceSoftmax,
  type LayerNormOptions,
} from "./reference-basic.js";
export {
  estimateAttentionMemory,
  referenceAttention,
  referenceBlockedAttention,
  referenceFlashAttention,
  referenceFlashAttentionBackward,
  referenceFusedAttention,
  type AttentionMemoryEstimate,
  type AttentionMemoryEstimateInput,
  type AttentionOptimizationOptions,
  type AttentionOptimizationReferenceResult,
  type FlashAttentionBackwardResult,
  type FlashAttentionForwardResult,
  type FlashAttentionOptions,
} from "./reference-attention.js";

export const reference = {
  matmul: referenceMatmul,
  softmax: referenceSoftmax,
  relu: referenceRelu,
  gelu: referenceGelu,
  layernorm: referenceLayerNorm,
  attention: referenceAttention,
  attentionBlocked: referenceBlockedAttention,
  attentionFused: referenceFusedAttention,
  attentionMemory: estimateAttentionMemory,
  flashAttention: referenceFlashAttention,
  flashAttentionBackward: referenceFlashAttentionBackward,
};
