# browsergrad

Open-source library family for running Python and ML workloads in the browser.

```
@unlocalhosted/browsergrad-runtime   Pyodide-in-Worker host: exec, fs mount, AbortSignal cancel
@unlocalhosted/browsergrad-kernels   WGSL kernel catalog for ML ops (matmul, softmax, attention, ...)
@unlocalhosted/browsergrad-grad      Readable tensor + autograd library, runs in Pyodide
```

## Status

Pre-alpha. The runtime package is the first focus; kernels and grad will follow once the runtime surface stabilizes.

## Design

Each package exposes **primitives, not workflows.** No baked-in concept of a "notebook," "lab," or "lesson." Consumers compose their own product on top.

- `browsergrad-runtime` doesn't know what tests are.
- `browsergrad-kernels` doesn't know what a tensor is — it dispatches WGSL with typed buffers.
- `browsergrad-grad` is the tensor library that *could* use the kernels, but doesn't require them.

This is deliberate. Packages can be used in isolation or together.

## Repository layout

```
browsergrad/
├── packages/
│   ├── browsergrad-runtime/   ← start here
│   ├── browsergrad-kernels/   (planned)
│   └── browsergrad-grad/      (planned)
├── package.json               npm workspaces root
└── LICENSE                    MIT
```

## License

MIT. See [LICENSE](./LICENSE).
