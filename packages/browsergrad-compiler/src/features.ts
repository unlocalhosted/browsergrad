import type { KernelFeatureSet } from "@unlocalhosted/browsergrad-kernels";
import type { CompileCudaLiteOptions, CudaLiteFeatureOptions } from "./types.js";

export type CudaLiteKernelFeatureSource = Partial<
  Pick<KernelFeatureSet, "shaderF16" | "subgroups" | "compatibilityMode" | "features">
>;

export function cudaLiteFeatureOptionsFromKernelFeatures(
  source: CudaLiteKernelFeatureSource,
): CudaLiteFeatureOptions {
  const rawFeatures = new Set(source.features ?? []);
  return {
    ...(source.shaderF16 === true || rawFeatures.has("shader-f16") ? { "shader-f16": true } : {}),
    ...(source.subgroups === true || rawFeatures.has("subgroups") ? { subgroups: true } : {}),
    ...(source.compatibilityMode === true || rawFeatures.has("compatibility") ? { compatibility: true } : {}),
  };
}

export function compileCudaLiteOptionsFromKernelFeatures(
  source: CudaLiteKernelFeatureSource,
  options: CompileCudaLiteOptions = {},
): CompileCudaLiteOptions {
  return {
    ...options,
    features: {
      ...cudaLiteFeatureOptionsFromKernelFeatures(source),
      ...options.features,
    },
  };
}
