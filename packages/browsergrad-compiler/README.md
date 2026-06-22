# @unlocalhosted/browsergrad-compiler

CUDA-lite compiler for browser-native GPU labs.

The default path is small and inspectable:

```text
CUDA-lite source -> BrowserGrad Kernel IR -> WGSL -> WebGPU
                     \-> lockstep CPU reference
```

This package is independent of Pyodide. It uses
`@unlocalhosted/browsergrad-kernels` for WGSL dispatch.
