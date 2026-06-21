# Primitive Package Architecture

BrowserGrad packages are reusable browser ML primitives. Curriculum profiles are
adapters that use those primitives for guided labs.

`@unlocalhosted/browsergrad-primitives` is the canonical facade for small
helpers. Leaf packages may remain as implementation shards for compatibility or
release mechanics, but new public guidance should teach the facade first.

## Keep In Packages

- Tensor/runtime/kernels primitives with enough weight to justify a package.
- Small text/data/scaling/alignment/snapshot/simulation helpers under the
  primitive facade unless they prove a separate release seam.
- Deterministic references and simulators with generic names.
- Browser execution substrates and behavior gates.
- Small adapters that connect primitive results to BrowserGrad assertion shapes.

## Keep Out Of Packages

- Course acronyms in public export names.
- Assignment-specific oracle module names such as `_bg_gpu_puzzle_oracles`.
- Handoff-only fixture layout, grader wording, or upstream test names.
- Platform product identity.

## Naming Rule

Use primitive names in public interfaces and profile oracle adapters:

- `_bg_byte_bpe`, not `_bg_cs336_tokenizer`.
- `_bg_cuda_concepts`, not `_bg_gpu_puzzle_oracles`.
- `_bg_cpu_parallelism`, not `_bg_cs149_cpu_oracles`.
- `createByteBpeReference()` in public primitive code.
- `createByteBpeRuntimeAdapter()` when crossing into Pyodide/profile glue.

Benchmark assignments remain valuable. They prove coverage and expose missing
capabilities. They do not get to name core modules.
