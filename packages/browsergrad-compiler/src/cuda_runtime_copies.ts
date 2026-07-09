export function isCudaRuntimeCopyCall(name: string): boolean {
  return isCudaRuntimeMemcpyCall(name) ||
    isCudaRuntimePeerCopyCall(name) ||
    isCudaRuntimeSymbolCopyCall(name) ||
    isCudaRuntimeMemsetCall(name) ||
    isCudaRuntimeSymbolMemsetCall(name);
}

export function isCudaRuntimeMemcpyCall(name: string): boolean {
  return name === "cudaMemcpy" ||
    name === "cudaMemcpyAsync" ||
    isCudaRuntimeMemcpy2DCall(name);
}

export function isCudaRuntimeMemcpy2DCall(name: string | undefined): boolean {
  return name === "cudaMemcpy2D" || name === "cudaMemcpy2DAsync";
}

export function isCudaRuntimePeerCopyCall(name: string): boolean {
  return name === "cudaMemcpyPeer" || name === "cudaMemcpyPeerAsync";
}

export function isCudaRuntimeSymbolCopyCall(name: string | undefined): boolean {
  return name === "cudaMemcpyToSymbol" ||
    name === "cudaMemcpyToSymbolAsync" ||
    name === "cudaMemcpyFromSymbol" ||
    name === "cudaMemcpyFromSymbolAsync";
}

export function isCudaRuntimeMemsetCall(name: string | undefined): boolean {
  return name === "cudaMemset" || name === "cudaMemsetAsync" || isCudaRuntimeMemset2DCall(name);
}

export function isCudaRuntimeMemset2DCall(name: string | undefined): boolean {
  return name === "cudaMemset2D" || name === "cudaMemset2DAsync";
}

export function isCudaRuntimeSymbolMemsetCall(name: string | undefined): boolean {
  return name === "cudaMemsetToSymbol" || name === "cudaMemsetToSymbolAsync";
}
