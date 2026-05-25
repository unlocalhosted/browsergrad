/**
 * @unlocalhosted/browsergrad-kernels — public surface.
 *
 * ```ts
 * import { createDevice, kernels, tensor } from "@unlocalhosted/browsergrad-kernels";
 *
 * const device = await createDevice();
 * const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
 * const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));
 *
 * const C = await kernels.matmul(device, A, B);
 * console.log(C.shape, C.data);  // [2, 2], Float32Array(4)
 * ```
 *
 * Pure-JS reference implementations are available at the `./reference` subpath:
 *
 * ```ts
 * import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
 * const C = reference.matmul(A, B);  // runs on CPU; same surface as kernels.matmul
 * ```
 */

export type {
  Tensor,
  KernelDevice,
  KernelDeviceOptions,
  KernelDeviceStats,
} from "./types";

export { tensor, KernelError } from "./types";

export { createDevice } from "./device";

export {
  kernels,
  matmul,
  softmax,
  relu,
  gelu,
  layernorm,
  attention,
  type Kernels,
  type LayerNormOptions,
} from "./kernels/index";

// Re-export reference as a top-level convenience too (alongside the subpath).
export { reference } from "./reference";
