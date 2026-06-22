# Primitive Package Architecture

BrowserGrad packages are reusable browser ML primitives. Curriculum profiles are
adapters that use those primitives for guided labs.

`@unlocalhosted/browsergrad-primitives` is the canonical facade for small
helpers. Text, data, scaling, RL math, snapshot comparison, and deterministic
simulation code lives behind this facade until a future split passes the package
split test.
See `docs/platform/package-consolidation-audit.md` for the current package
classification.

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
backend constraints, release cadence, or ownership proves a durable seam. Do
not create a sibling package for a helper whose complexity disappears when it
moves into the facade.

## Naming Rule

Use primitive names in public interfaces and profile oracle adapters:

- `_bg_byte_bpe`, not `_bg_cs336_tokenizer`.
- `_bg_cuda_concepts`, not `_bg_gpu_puzzle_oracles`.
- `_bg_cpu_parallelism`, not `_bg_cs149_cpu_oracles`.
- `_bg_data_cleaning`, not `_bg_cs336_data_oracles`.
- `_bg_rl_math`, not `_bg_alignment_oracles`.
- `_bg_attention_math`, not `_bg_attention_oracles`.
- `_bg_distributed_training`, not `_bg_distributed_oracles`.
- `_bg_task_graph`, not `_bg_task_runtime_oracles`.
- `createByteBpeReference()` in public primitive code.
- `createByteBpeReferenceModule()` when exposing a JSON-friendly reference
  object to a profile.
- Profile-local wrappers may translate camelCase primitive references into
  Python-safe snake_case or JSON-string methods, but those wrappers belong in
  runtime/profile glue, not the primitive package surface.

Benchmark assignments remain valuable. They prove coverage and expose missing
capabilities. They do not get to name core modules.
