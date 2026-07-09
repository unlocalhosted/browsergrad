import type {
  KernelMemoryAccess,
  KernelThreadTrace,
} from "./types.js";

export interface MutableReferenceTrace {
  readonly blockIdx: readonly [number, number, number];
  readonly threadIdx: readonly [number, number, number];
  readonly reads: KernelMemoryAccess[];
  readonly writes: KernelMemoryAccess[];
  readonly sharedReads: KernelMemoryAccess[];
  readonly sharedWrites: KernelMemoryAccess[];
}

export function freezeReferenceTrace(trace: MutableReferenceTrace): KernelThreadTrace {
  return {
    blockIdx: trace.blockIdx,
    threadIdx: trace.threadIdx,
    reads: trace.reads.map((item) => ({ ...item })),
    writes: trace.writes.map((item) => ({ ...item })),
    sharedReads: trace.sharedReads.map((item) => ({ ...item })),
    sharedWrites: trace.sharedWrites.map((item) => ({ ...item })),
  };
}
