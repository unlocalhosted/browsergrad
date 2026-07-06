import {
  kernels,
  tensor,
  type KernelDevice,
  type Tensor as KernelTensor,
} from "@unlocalhosted/browsergrad-kernels";

type Shape = readonly number[];

interface KernelDispatch {
  matmul(device: KernelDevice, a: KernelTensor, b: KernelTensor): Promise<KernelTensor>;
  softmax(device: KernelDevice, x: KernelTensor): Promise<KernelTensor>;
  layernorm(
    device: KernelDevice,
    x: KernelTensor,
    opts?: {
      gamma?: KernelTensor;
      beta?: KernelTensor;
      eps?: number;
    },
  ): Promise<KernelTensor>;
  attention(
    device: KernelDevice,
    q: KernelTensor,
    k: KernelTensor,
    v: KernelTensor,
  ): Promise<KernelTensor>;
}

export interface GradKernelDeviceBridge {
  matmul(
    a: readonly number[],
    aShape: Shape,
    b: readonly number[],
    bShape: Shape,
  ): Promise<readonly number[]>;
  softmax(x: readonly number[], shape: Shape): Promise<readonly number[]>;
  layernorm(
    x: readonly number[],
    shape: Shape,
    gamma: readonly number[] | null,
    beta: readonly number[] | null,
    eps: number,
  ): Promise<readonly number[]>;
  attention(
    q: readonly number[],
    qShape: Shape,
    k: readonly number[],
    kShape: Shape,
    v: readonly number[],
    vShape: Shape,
  ): Promise<readonly number[]>;
}

export function createGradKernelDeviceBridge(
  device: KernelDevice,
  dispatch: KernelDispatch = kernels,
): GradKernelDeviceBridge {
  return {
    async matmul(a, aShape, b, bShape) {
      const out = await dispatch.matmul(
        device,
        toKernelTensor(aShape, a),
        toKernelTensor(bShape, b),
      );
      return Array.from(out.data);
    },

    async softmax(x, shape) {
      const out = await dispatch.softmax(device, toKernelTensor(shape, x));
      return Array.from(out.data);
    },

    async layernorm(x, shape, gamma, beta, eps) {
      const lastDim = shape[shape.length - 1];
      if (lastDim === undefined) {
        throw new Error("layernorm: scalar tensor not supported");
      }
      const opts = {
        eps,
      } as {
        eps: number;
        gamma?: KernelTensor;
        beta?: KernelTensor;
      };
      if (gamma !== null) opts.gamma = toKernelTensor([lastDim], gamma);
      if (beta !== null) opts.beta = toKernelTensor([lastDim], beta);
      const out = await dispatch.layernorm(device, toKernelTensor(shape, x), opts);
      return Array.from(out.data);
    },

    async attention(q, qShape, k, kShape, v, vShape) {
      const out = await dispatch.attention(
        device,
        toKernelTensor(qShape, q),
        toKernelTensor(kShape, k),
        toKernelTensor(vShape, v),
      );
      return Array.from(out.data);
    },
  };
}

function toKernelTensor(shape: Shape, values: readonly number[]): KernelTensor {
  return tensor(shape, Float32Array.from(values));
}
