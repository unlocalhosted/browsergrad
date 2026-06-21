# Primitive Package Architecture

BrowserGrad packages are reusable browser ML primitives. Curriculum profiles are
adapters that use those primitives for guided labs.

## Keep In Packages

- Tensor/runtime/kernels/tokenizers/data/scaling/alignment/snapshot primitives.
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
- `createByteBpeOracleModule()`, not `createCs336TokenizerOracleModule()`.

Benchmark assignments remain valuable. They prove coverage and expose missing
capabilities. They do not get to name core modules.
