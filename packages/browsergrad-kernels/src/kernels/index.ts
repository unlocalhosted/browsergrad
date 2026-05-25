/**
 * Kernel registry — bundles every WGSL kernel into a single namespace.
 *
 * Usage:
 *   import { kernels } from "@unlocalhosted/browsergrad-kernels";
 *   const C = await kernels.matmul(device, A, B);
 */

import { matmul } from "./matmul";
import { softmax } from "./softmax";
import { relu, gelu } from "./activations";
import { layernorm } from "./layernorm";
import { attention } from "./attention";

export const kernels = {
  matmul,
  softmax,
  relu,
  gelu,
  layernorm,
  attention,
};

export type Kernels = typeof kernels;

export type { LayerNormOptions } from "./layernorm";
export { matmul, softmax, relu, gelu, layernorm, attention };
