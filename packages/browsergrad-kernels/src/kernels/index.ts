/**
 * Kernel registry — bundles every WGSL kernel into a single namespace.
 *
 * Usage:
 *   import { kernels } from "@unlocalhosted/browsergrad-kernels";
 *   const C = await kernels.matmul(device, A, B);
 */

import { matmul } from "./matmul.js";
import { softmax } from "./softmax.js";
import { relu, gelu } from "./activations.js";
import { layernorm } from "./layernorm.js";
import { attention } from "./attention.js";

export const kernels = {
  matmul,
  softmax,
  relu,
  gelu,
  layernorm,
  attention,
};

export type Kernels = typeof kernels;

export type { LayerNormOptions } from "./layernorm.js";
export { matmul, softmax, relu, gelu, layernorm, attention };
