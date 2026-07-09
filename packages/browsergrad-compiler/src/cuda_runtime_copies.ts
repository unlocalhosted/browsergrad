export type CudaRuntimeCopyShape =
  | { readonly kind: "copy1d"; readonly srcIndex: number; readonly countIndex: number }
  | { readonly kind: "copy2d"; readonly srcIndex: number }
  | { readonly kind: "symbol"; readonly direction: "to-symbol" | "from-symbol"; readonly symbolIndex: number; readonly pointerIndex: number; readonly srcIndex: number; readonly countIndex: number; readonly offsetIndex?: number };

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

export function cudaRuntimeCopyShapeForCall(
  name: string | undefined,
  hasOffsetArg = false,
): CudaRuntimeCopyShape | undefined {
  if (name === "cudaMemcpy" || name === "cudaMemcpyAsync") return { kind: "copy1d", srcIndex: 1, countIndex: 2 };
  if (name === "cudaMemcpy2D" || name === "cudaMemcpy2DAsync") return { kind: "copy2d", srcIndex: 2 };
  if (name === "cudaMemcpyPeer" || name === "cudaMemcpyPeerAsync") return { kind: "copy1d", srcIndex: 2, countIndex: 4 };
  if (name === "cudaMemcpyToSymbol" || name === "cudaMemcpyToSymbolAsync") {
    return { kind: "symbol", direction: "to-symbol", symbolIndex: 0, pointerIndex: 1, srcIndex: 1, countIndex: 2, ...(hasOffsetArg ? { offsetIndex: 3 } : {}) };
  }
  if (name === "cudaMemcpyFromSymbol" || name === "cudaMemcpyFromSymbolAsync") {
    return { kind: "symbol", direction: "from-symbol", symbolIndex: 1, pointerIndex: 0, srcIndex: 1, countIndex: 2, ...(hasOffsetArg ? { offsetIndex: 3 } : {}) };
  }
  return undefined;
}
