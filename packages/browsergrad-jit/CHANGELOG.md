# Changelog

All notable changes to `@unlocalhosted/browsergrad-jit` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/) per the [compatibility
contract in the README](README.md#compatibility-contract).

## [Unreleased]

## [0.1.0-pre.1] — 2026-05-26

PRD-005 Week 1: scaffolding + IR foundation.

### Added

- Initial package scaffold mirroring `@unlocalhosted/browsergrad-grad`.
- `_ir.py`: 23-opcode UOp IR. `BUFFER`, `LOAD`, `STORE`, `CONST`, `RANDOM`,
  `CAST`, `ADD`, `MUL`, `DIV`, `NEG`, `EXP`, `LOG`, `CMP`, `MATMUL`,
  `REDUCE`, `RESHAPE`, `PERMUTE`, `SLICE`, `PAD`, `WHERE`, `INDEX`,
  `MASK`, `CUSTOM`. Frozen-dataclass nodes with cached structural hash;
  shape-validation at construction; `toposort()` + `all_buffers()` walkers.
- `_errors.py`: typed exception hierarchy rooted at `JitError`. Subtypes:
  `ShapeError`, `TorchAliasConflict`, `NoBackwardError`,
  `JitNotImplementedError`, `RealizationError`, `BufferTableError`.
- `_buffer_table.py`: per-session `BufferTable` with session-token-prefixed
  ids. Refuses cross-session buffer reuse; refuses shape/dtype-changing
  updates.
- `_tensor_proxy.py`: `TensorProxy` stub. Metadata access works
  (`.shape`, `.dtype`, `.ndim`, `len(t)`, `__repr__`, `size()`, `numel()`).
  Realization triggers (`.numpy()`, `.tolist()`, `.item()`, `.backward()`,
  `__bool__`, `__float__`, `__int__`, `__iter__`) raise
  `JitNotImplementedError` — they land in PRD-005 Weeks 3–5.
- `.data` and `__array__` deliberately raise to enforce explicit
  realization. See PRD-005 critique for rationale.
- `Session` class + `new_session()` / `get_default_session()` /
  `set_default_session()` for per-loop isolation.
- TS install pipeline: `installJit`, `createNodePyodideTarget`, codegen
  script — all mirroring the `browsergrad-grad` shapes exactly so consumers
  can swap installs without re-learning the API.

### Internal

- `_ir`, `_buffer_table`, `_tensor_proxy` are explicitly internal modules
  (leading underscore). The `Session` class and error types are part of
  the public surface.
