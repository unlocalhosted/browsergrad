import {
  isCudaRuntimeMemcpyCall,
  isCudaRuntimeMemset2DCall,
  isCudaRuntimePeerCopyCall,
  isCudaRuntimeSymbolCopyCall,
  isCudaRuntimeSymbolMemsetCall,
} from "./cuda_runtime_copies.js";
import { isHostManagedRuntimeNoopCall } from "./cuda_runtime_noops.js";
import { isCudaRuntimeQueryWriteCall } from "./cuda_runtime_queries.js";

export function isCudaHostDynamicNoopCall(name: string | undefined): boolean {
  return name !== undefined &&
    !isCudaRuntimeQueryWriteCall(name) &&
    (
      isHostManagedRuntimeNoopCall(name) ||
      isCudaRuntimeMemcpyCall(name) ||
      isCudaRuntimePeerCopyCall(name) ||
      isCudaRuntimeSymbolCopyCall(name) ||
      isCudaRuntimeMemset2DCall(name) ||
      isCudaRuntimeSymbolMemsetCall(name) ||
      name === "printf"
    );
}

export function isCudaPeerCopyHostNoopCall(name: string | undefined): boolean {
  return name !== undefined &&
    !isCudaRuntimeQueryWriteCall(name) &&
    (
      isHostManagedRuntimeNoopCall(name) ||
      name === "printf"
    );
}

export function isCudaHostSideEffectFreeRuntimeCall(name: string | undefined): boolean {
  return name !== undefined &&
    (
      isHostManagedRuntimeNoopCall(name) ||
      isCudaRuntimeSymbolMemsetCall(name)
    );
}
