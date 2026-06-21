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

## Keep Out Of Packages

- Course acronyms in public export names.
- Assignment-specific oracle module names such as `_bg_gpu_puzzle_oracles`.
- Python/Pyodide bridge adapters whose method names only exist for one profile.
- Handoff-only fixture layout, grader wording, or upstream test names.
- Platform product identity.

## Package Split Test

Before adding a new package, run the deletion test:

- If deleting the package only moves helpers into `browsergrad-primitives`, the
  package was a shallow implementation shard.
- If deleting the package would force tensor/runtime/GPU/compiler concerns into
  unrelated callers, the package is a real seam.

Use `browsergrad-primitives` for small references, comparators, fixtures,
simulators, parsers, and data-cleaning helpers. Split only when bundle weight,
backend constraints, release cadence, or ownership proves a durable seam.

## Naming Rule

Use primitive names in public interfaces and profile oracle adapters:

- `_bg_byte_bpe`, not `_bg_cs336_tokenizer`.
- `_bg_cuda_concepts`, not `_bg_gpu_puzzle_oracles`.
- `_bg_cpu_parallelism`, not `_bg_cs149_cpu_oracles`.
- `_bg_data_cleaning`, not `_bg_cs336_data_oracles`.
- `createByteBpeReference()` in public primitive code.
- Profile-local wrappers may translate camelCase primitive references into
  Python-safe snake_case or JSON-string methods, but those wrappers belong in
  runtime/profile glue, not the primitive package surface.

Benchmark assignments remain valuable. They prove coverage and expose missing
capabilities. They do not get to name core modules.
