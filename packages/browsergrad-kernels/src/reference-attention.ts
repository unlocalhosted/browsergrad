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

import { assertDataLength } from "./runner.js";
import { KernelError, type Tensor } from "./types.js";
import { referenceMatmul, referenceSoftmax } from "./reference-basic.js";

/* ────────────────────────────────────────────────────────────
 * attention: scaled dot-product (Vaswani et al. 2017).
 *
 *   scores = Q @ K^T / sqrt(d_k)
 *   weights = softmax(scores)   (along last axis)
 *   out = weights @ V
 *
 * v0 shapes: Q, K, V all [S, D]. Single head, no batching. The headline
 * lesson is to *see* attention compose from matmul and softmax — extending
 * to multi-head + batching is a future minor.
 * ──────────────────────────────────────────────────────────── */

export function referenceAttention(Q: Tensor, K: Tensor, V: Tensor): Tensor {
  if (Q.shape.length !== 2 || K.shape.length !== 2 || V.shape.length !== 2) {
    throw new KernelError(
      `attention: expected 2D Q, K, V, got ranks ${Q.shape.length}/${K.shape.length}/${V.shape.length}`,
    );
  }
  const [S, DQ] = Q.shape as [number, number];
  const [SK, DK] = K.shape as [number, number];
  const [SV] = V.shape as [number, number];
  if (DQ !== DK) throw new KernelError(`attention: Q dim ${DQ} ≠ K dim ${DK}`);
  if (S !== SK || S !== SV) {
    throw new KernelError(
      `attention (v0): self-attention only — Q seq (${S}), K seq (${SK}), V seq (${SV}) ` +
      `must all be equal. Cross-attention (Q seq ≠ K=V seq) is not yet implemented; ` +
      `track via PRD-012c follow-up. Workaround: pad Q to match K, then truncate output.`,
    );
  }
  if (SK !== SV) {
    throw new KernelError(
      `attention: K seq (${SK}) and V seq (${SV}) must always match — they index the ` +
      `same source token positions. This holds even when cross-attention lands.`,
    );
  }

  // K^T has shape [DK, SK] = [DK, S]
  const KT: Tensor = { shape: [DK, SK], data: new Float32Array(DK * SK) };
  for (let i = 0; i < SK; i++) {
    for (let j = 0; j < DK; j++) {
      KT.data[j * SK + i] = K.data[i * DK + j]!;
    }
  }

  // scores = Q @ K^T → [S, S]
  const scores = referenceMatmul(Q, KT);

  // scale by 1/sqrt(d_k)
  const scale = 1 / Math.sqrt(DK);
  for (let i = 0; i < scores.data.length; i++) {
    scores.data[i]! *= scale;
  }

  // weights = softmax(scores)
  const weights = referenceSoftmax(scores);

  // out = weights @ V → [S, DV]
  return referenceMatmul(weights, V);
}

export interface AttentionMemoryEstimateInput {
  readonly queryLength: number;
  readonly keyLength: number;
  readonly queryDepth: number;
  readonly valueDepth: number;
  readonly blockSize?: number;
}

export interface AttentionMemoryEstimate {
  readonly queryElements: number;
  readonly keyElements: number;
  readonly valueElements: number;
  readonly outputElements: number;
  readonly naiveScoreElements: number;
  readonly naiveProbabilityElements: number;
  readonly blockedScoreElements: number;
  readonly fusedScoreElements: number;
  readonly blockSize: number;
}

export interface AttentionOptimizationOptions {
  readonly blockSize?: number;
}

export interface AttentionOptimizationReferenceResult {
  readonly output: Tensor;
  readonly memory: AttentionMemoryEstimate;
}

export function estimateAttentionMemory(
  input: AttentionMemoryEstimateInput,
): AttentionMemoryEstimate {
  const {
    queryLength,
    keyLength,
    queryDepth,
    valueDepth,
    blockSize = keyLength,
  } = input;
  for (const [name, value] of Object.entries({
    queryLength,
    keyLength,
    queryDepth,
    valueDepth,
    blockSize,
  })) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new KernelError(`attention memory: ${name} must be a positive integer`);
    }
  }
  const effectiveBlockSize = Math.min(blockSize, keyLength);
  return {
    queryElements: queryLength * queryDepth,
    keyElements: keyLength * queryDepth,
    valueElements: keyLength * valueDepth,
    outputElements: queryLength * valueDepth,
    naiveScoreElements: queryLength * keyLength,
    naiveProbabilityElements: queryLength * keyLength,
    blockedScoreElements: queryLength * effectiveBlockSize,
    fusedScoreElements: 0,
    blockSize: effectiveBlockSize,
  };
}

export function referenceBlockedAttention(
  Q: Tensor,
  K: Tensor,
  V: Tensor,
  options: AttentionOptimizationOptions = {},
): AttentionOptimizationReferenceResult {
  const output = referenceAttention(Q, K, V);
  const [queryLength, queryDepth] = Q.shape as [number, number];
  const [keyLength] = K.shape as [number, number];
  const [, valueDepth] = V.shape as [number, number];
  return {
    output,
    memory: estimateAttentionMemory({
      queryLength,
      keyLength,
      queryDepth,
      valueDepth,
      ...(options.blockSize === undefined ? {} : { blockSize: options.blockSize }),
    }),
  };
}

export function referenceFusedAttention(
  Q: Tensor,
  K: Tensor,
  V: Tensor,
): AttentionOptimizationReferenceResult {
  const output = referenceAttention(Q, K, V);
  const [queryLength, queryDepth] = Q.shape as [number, number];
  const [keyLength] = K.shape as [number, number];
  const [, valueDepth] = V.shape as [number, number];
  return {
    output,
    memory: estimateAttentionMemory({
      queryLength,
      keyLength,
      queryDepth,
      valueDepth,
    }),
  };
}

export interface FlashAttentionOptions {
  readonly causal?: boolean;
  readonly scale?: number;
}

export interface FlashAttentionForwardResult {
  readonly output: Tensor;
  readonly logSumExp: Tensor;
}

export interface FlashAttentionBackwardResult {
  readonly queryGradient: Tensor;
  readonly keyGradient: Tensor;
  readonly valueGradient: Tensor;
}

export function referenceFlashAttention(
  Q: Tensor,
  K: Tensor,
  V: Tensor,
  options: FlashAttentionOptions = {},
): FlashAttentionForwardResult {
  const layout = validateFlashAttentionInputs(Q, K, V);
  const scale = options.scale ?? 1 / Math.sqrt(layout.queryDepth);
  if (!Number.isFinite(scale)) {
    throw new KernelError("flashAttention: scale must be finite");
  }
  const output = new Float32Array(layout.batchHeads * layout.queryLength * layout.valueDepth);
  const logSumExp = new Float32Array(layout.batchHeads * layout.queryLength);

  for (let batchHead = 0; batchHead < layout.batchHeads; batchHead++) {
    for (let queryIndex = 0; queryIndex < layout.queryLength; queryIndex++) {
      const scores = new Float64Array(layout.keyLength);
      let maxScore = -Infinity;
      for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
        const score = flashAttentionScore(
          Q,
          K,
          layout,
          batchHead,
          queryIndex,
          keyIndex,
          scale,
          options.causal === true,
        );
        scores[keyIndex] = score;
        if (score > maxScore) maxScore = score;
      }
      let sumExp = 0;
      for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
        const weight = Math.exp(scores[keyIndex]! - maxScore);
        scores[keyIndex] = weight;
        sumExp += weight;
      }
      logSumExp[batchHead * layout.queryLength + queryIndex] =
        maxScore + Math.log(sumExp);
      for (let valueIndex = 0; valueIndex < layout.valueDepth; valueIndex++) {
        let total = 0;
        for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
          total +=
            (scores[keyIndex]! / sumExp) *
            V.data[flashValueOffset(layout, batchHead, keyIndex, valueIndex)]!;
        }
        output[flashOutputOffset(layout, batchHead, queryIndex, valueIndex)] = total;
      }
    }
  }

  return {
    output: {
      shape: [...layout.outputShape],
      data: output,
    },
    logSumExp: {
      shape: [...layout.logSumExpShape],
      data: logSumExp,
    },
  };
}

export function referenceFlashAttentionBackward(
  Q: Tensor,
  K: Tensor,
  V: Tensor,
  outputGradient: Tensor,
  options: FlashAttentionOptions = {},
): FlashAttentionBackwardResult {
  const layout = validateFlashAttentionInputs(Q, K, V);
  assertDataLength("flashAttention.outputGradient", outputGradient.data, layout.outputShape);
  const scale = options.scale ?? 1 / Math.sqrt(layout.queryDepth);
  if (!Number.isFinite(scale)) {
    throw new KernelError("flashAttention: scale must be finite");
  }
  const queryGradient = new Float32Array(Q.data.length);
  const keyGradient = new Float32Array(K.data.length);
  const valueGradient = new Float32Array(V.data.length);

  for (let batchHead = 0; batchHead < layout.batchHeads; batchHead++) {
    for (let queryIndex = 0; queryIndex < layout.queryLength; queryIndex++) {
      const probabilities = flashAttentionProbabilities(
        Q,
        K,
        layout,
        batchHead,
        queryIndex,
        scale,
        options.causal === true,
      );
      const valueJacobians = new Float64Array(layout.keyLength);

      for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
        for (let valueIndex = 0; valueIndex < layout.valueDepth; valueIndex++) {
          const upstream =
            outputGradient.data[
              flashOutputOffset(layout, batchHead, queryIndex, valueIndex)
            ]!;
          valueGradient[flashValueOffset(layout, batchHead, keyIndex, valueIndex)]! +=
            probabilities[keyIndex]! * upstream;
          valueJacobians[keyIndex]! +=
            upstream *
            V.data[flashValueOffset(layout, batchHead, keyIndex, valueIndex)]!;
        }
      }

      let softmaxDot = 0;
      for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
        softmaxDot += probabilities[keyIndex]! * valueJacobians[keyIndex]!;
      }

      for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
        const scoreGradient =
          probabilities[keyIndex]! * (valueJacobians[keyIndex]! - softmaxDot);
        for (let depth = 0; depth < layout.queryDepth; depth++) {
          queryGradient[flashQueryOffset(layout, batchHead, queryIndex, depth)]! +=
            scoreGradient *
            K.data[flashKeyOffset(layout, batchHead, keyIndex, depth)]! *
            scale;
          keyGradient[flashKeyOffset(layout, batchHead, keyIndex, depth)]! +=
            scoreGradient *
            Q.data[flashQueryOffset(layout, batchHead, queryIndex, depth)]! *
            scale;
        }
      }
    }
  }

  return {
    queryGradient: { shape: [...Q.shape], data: queryGradient },
    keyGradient: { shape: [...K.shape], data: keyGradient },
    valueGradient: { shape: [...V.shape], data: valueGradient },
  };
}

interface FlashAttentionLayout {
  readonly rank: 3 | 4;
  readonly batchHeads: number;
  readonly batch: number;
  readonly heads: number;
  readonly queryLength: number;
  readonly keyLength: number;
  readonly queryDepth: number;
  readonly valueDepth: number;
  readonly outputShape: readonly number[];
  readonly logSumExpShape: readonly number[];
}

function validateFlashAttentionInputs(Q: Tensor, K: Tensor, V: Tensor): FlashAttentionLayout {
  if (!isFlashAttentionRank(Q.shape.length)) {
    throw new KernelError(
      `flashAttention: expected 3D [B,S,D] or 4D [B,H,S,D] query, got ${Q.shape.length}D`,
    );
  }
  if (K.shape.length !== Q.shape.length || V.shape.length !== Q.shape.length) {
    throw new KernelError("flashAttention: Q, K, V must have the same rank");
  }
  assertDataLength("flashAttention.Q", Q.data, Q.shape);
  assertDataLength("flashAttention.K", K.data, K.shape);
  assertDataLength("flashAttention.V", V.data, V.shape);

  if (Q.shape.length === 3) {
    const [B, Sq, Dq] = Q.shape as [number, number, number];
    const [Bk, Sk, Dk] = K.shape as [number, number, number];
    const [Bv, Sv, Dv] = V.shape as [number, number, number];
    if (B !== Bk || B !== Bv) {
      throw new KernelError("flashAttention: batch dimensions must match");
    }
    if (Dq !== Dk) throw new KernelError(`flashAttention: Q dim ${Dq} ≠ K dim ${Dk}`);
    if (Sk !== Sv) {
      throw new KernelError(`flashAttention: K seq (${Sk}) and V seq (${Sv}) must match`);
    }
    return {
      rank: 3,
      batchHeads: B,
      batch: B,
      heads: 1,
      queryLength: Sq,
      keyLength: Sk,
      queryDepth: Dq,
      valueDepth: Dv,
      outputShape: [B, Sq, Dv],
      logSumExpShape: [B, Sq],
    };
  }

  const [B, H, Sq, Dq] = Q.shape as [number, number, number, number];
  const [Bk, Hk, Sk, Dk] = K.shape as [number, number, number, number];
  const [Bv, Hv, Sv, Dv] = V.shape as [number, number, number, number];
  if (B !== Bk || B !== Bv || H !== Hk || H !== Hv) {
    throw new KernelError("flashAttention: batch/head dimensions must match");
  }
  if (Dq !== Dk) throw new KernelError(`flashAttention: Q dim ${Dq} ≠ K dim ${Dk}`);
  if (Sk !== Sv) {
    throw new KernelError(`flashAttention: K seq (${Sk}) and V seq (${Sv}) must match`);
  }
  return {
    rank: 4,
    batchHeads: B * H,
    batch: B,
    heads: H,
    queryLength: Sq,
    keyLength: Sk,
    queryDepth: Dq,
    valueDepth: Dv,
    outputShape: [B, H, Sq, Dv],
    logSumExpShape: [B, H, Sq],
  };
}

function isFlashAttentionRank(rank: number): rank is 3 | 4 {
  return rank === 3 || rank === 4;
}

function flashAttentionScore(
  Q: Tensor,
  K: Tensor,
  layout: FlashAttentionLayout,
  batchHead: number,
  queryIndex: number,
  keyIndex: number,
  scale: number,
  causal: boolean,
): number {
  if (causal && queryIndex < keyIndex) return Number.NEGATIVE_INFINITY;
  let score = 0;
  for (let depth = 0; depth < layout.queryDepth; depth++) {
    score +=
      Q.data[flashQueryOffset(layout, batchHead, queryIndex, depth)]! *
      K.data[flashKeyOffset(layout, batchHead, keyIndex, depth)]!;
  }
  return score * scale;
}

function flashAttentionProbabilities(
  Q: Tensor,
  K: Tensor,
  layout: FlashAttentionLayout,
  batchHead: number,
  queryIndex: number,
  scale: number,
  causal: boolean,
): Float64Array {
  const scores = new Float64Array(layout.keyLength);
  let maxScore = -Infinity;
  for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
    const score = flashAttentionScore(
      Q,
      K,
      layout,
      batchHead,
      queryIndex,
      keyIndex,
      scale,
      causal,
    );
    scores[keyIndex] = score;
    if (score > maxScore) maxScore = score;
  }
  let sumExp = 0;
  for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
    const weight = Math.exp(scores[keyIndex]! - maxScore);
    scores[keyIndex] = weight;
    sumExp += weight;
  }
  for (let keyIndex = 0; keyIndex < layout.keyLength; keyIndex++) {
    scores[keyIndex]! /= sumExp;
  }
  return scores;
}

function flashQueryOffset(
  layout: FlashAttentionLayout,
  batchHead: number,
  queryIndex: number,
  depth: number,
): number {
  return layout.rank === 3
    ? (batchHead * layout.queryLength + queryIndex) * layout.queryDepth + depth
    : (batchHead * layout.queryLength + queryIndex) * layout.queryDepth + depth;
}

function flashKeyOffset(
  layout: FlashAttentionLayout,
  batchHead: number,
  keyIndex: number,
  depth: number,
): number {
  return layout.rank === 3
    ? (batchHead * layout.keyLength + keyIndex) * layout.queryDepth + depth
    : (batchHead * layout.keyLength + keyIndex) * layout.queryDepth + depth;
}

function flashValueOffset(
  layout: FlashAttentionLayout,
  batchHead: number,
  keyIndex: number,
  valueIndex: number,
): number {
  return (batchHead * layout.keyLength + keyIndex) * layout.valueDepth + valueIndex;
}

function flashOutputOffset(
  layout: FlashAttentionLayout,
  batchHead: number,
  queryIndex: number,
  valueIndex: number,
): number {
  return (batchHead * layout.queryLength + queryIndex) * layout.valueDepth + valueIndex;
}
