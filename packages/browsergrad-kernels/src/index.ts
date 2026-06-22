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
} from "./types.js";

export { tensor, KernelError } from "./types.js";

export { createDevice } from "./device.js";

export {
  createBrowsergradKernelRubric,
  createKernelRubric,
  kernelRubricFailureToAssertionDetails,
  type BrowsergradKernelAssertionTarget,
  type KernelRubric,
  type KernelRubricOptions,
  type KernelCloseOptions,
  type KernelRubricAssertion,
  type KernelRubricFailureDetails,
  type KernelAssertionDetails,
} from "./rubric.js";

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
} from "./kernels/index.js";

// PRD-011.5: WebGPU realizer bridge for browsergrad-jit.
// Construct via `createWebGpuRealizerBridge(device)` and hand to
// `bg.register_webgpu_bridge(...)` on the Python side.
export {
  createWebGpuRealizerBridge,
  type WebGpuRealizerBridge,
} from "./realizer.js";

// Direct-dispatch helpers — public so PRD-012's Python codegen can call
// individual kernels with GPUBuffer residency. The bridge above is the
// recommended path; these are escape hatches.
export {
  runDirect,
  materializeFloat32,
  uploadFloat32,
  type DirectDispatchOptions,
  type DirectDispatchResult,
} from "./runner.js";

export { matmulDirect } from "./kernels/matmul.js";
export { matmulTiledDirect, matmulTiled } from "./kernels/matmul_tiled.js";
export {
  fusedElementwiseDirect,
  generateFusedWgsl,
  type FusedOp,
} from "./kernels/fused_elementwise.js";
export { flashAttentionDirect } from "./kernels/flash_attention.js";

export {
  defineCuda1DProgram,
  emitCuda1DProgramWgsl,
  simulateCuda1DProgram,
  type Cuda1DCondition,
  type Cuda1DExpression,
  type Cuda1DLaunch,
  type Cuda1DProgram,
  type Cuda1DProgramInput,
  type Cuda1DProgramRunInput,
  type Cuda1DStatement,
  type Cuda1DWriteStatement,
  type Cuda1DIfStatement,
} from "./cuda_program.js";

export {
  referenceExclusiveScan,
  referenceFindRepeats,
  referenceSaxpy,
  simulateCuda1DGrid,
  type Cuda1DGridInput,
  type Cuda1DGridResult,
  type Cuda1DGridStats,
  type Cuda1DKernel,
  type Cuda1DMemoryAccess,
  type Cuda1DThreadContext,
  type Cuda1DThreadTrace,
  type Cuda1DViolation,
  type SaxpyInput,
} from "./cuda_concepts.js";

// Re-export reference as a top-level convenience too (alongside the subpath).
export { reference } from "./reference.js";
export {
  referenceFlashAttention,
  referenceFlashAttentionBackward,
  type FlashAttentionBackwardResult,
  type FlashAttentionForwardResult,
  type FlashAttentionOptions,
} from "./reference.js";
