# Changelog

All notable changes to `@unlocalhosted/browsergrad-semantic-core`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Added

- Explicit `/graph` export with the closed `browsergrad.host-graph@1`
  multi-dispatch DAG profile. Per-rank resources bind exact dtype, resolved
  allocation bytes, alignment, and lifetime role; dispatch effects and geometry
  derive from supplied opaque verified kernel/layout artifacts plus exact
  dimension bindings. Verification bounds nodes, edges, ranks, semantic
  artifacts, and aggregate bytes; rejects cycles, dangling references, input
  mutation, read-before-write, unordered hazards, incompatible bindings, and
  invalid all-reduce rank/numerical contracts; and fixes a fail-stop,
  no-partial-output-commit failure model without claiming execution, transport,
  topology, retries, or backend pipelines.
- Authority-bound `browsergrad.host-graph.cpu-reference@1` preparation and
  execution for the initial dispatch/all-reduce profile. The reference
  snapshots rank-local inputs into private storage, bounds memory, element
  operations, preparation/execution time, and cancellation, applies explicit
  finite rank-ordered f32 and wrapping/exact 32-bit integer collective policy,
  and returns outputs only after the complete graph succeeds.
- Explicit `/capability` export with immutable versioned semantic capability
  and backend definitions plus program/artifact-scoped lowering decisions.
  Positive decisions require a registered preservation level; conditional
  decisions retain exact feature, limit, and runtime-guard requirements;
  refusals cannot claim preservation. Static definitions contain no support or
  evidence outcome.
- Explicit `/requirement` export with immutable versioned assignment
  requirement definitions and environment-scoped available/unavailable
  resolutions. Availability requires one named provider, one closed provider
  mode, and deterministic evidence IDs; it does not imply semantic lowering or
  backend execution.
- Closed `browsergrad.kernel.attention-forward@1` artifacts for backend-neutral
  rank-4 f32 scaled-dot-product attention. The canonical dense constructor
  derives disjoint Q/K/V/output layouts, exact inverse-square-root scale bits,
  stable-softmax phases, finite-domain requirements, non-causal or upper-left
  causal meaning, the initial abs-or-relative comparison policy, and an
  explicit forward-only VJP refusal without admitting schedule or backend
  fields.
- Closed `browsergrad.schedule.attention-online-kv-tile@1` artifacts bind one
  exact attention-forward semantic hash to physical query/key tiles,
  cooperative single-buffered K/V staging, increasing tile traversal,
  tile-wise online-softmax rescaling, uniform barriers, scalar vectors, and
  masks that exclude invalid or logical-mask keys before online-state updates.
  The schedule owns no logical dtype, scale, view, comparison, backend, or
  performance claim.
- Authority-bound attention schedule specialization composes one prepared
  logical proof with one exact schedule and derives resource-bounded workgroup,
  K/V staging, private-state, key-tile, and 3D dispatch geometry without
  reconstructing logical addresses or granting device legality/preservation.
- Schedule-independent attention-forward specialization and CPU execution with
  bounded dense-address proof, scalar-work limits, fixed unshared input
  snapshots, cooperative yielding/cancellation, finite-domain enforcement,
  atomic destination commit, canonical stepwise f32 stable-softmax evaluation,
  causal/non-causal traces, and the declared abs-or-relative comparator.
- Shared hardened CPU binding admission for direct fixed unshared `Uint8Array`
  values, exact property capture, length/alignment proof, overlap checks, and
  native byte snapshots. Logical GEMM CPU execution now uses the same helper
  instead of retaining a second binding parser.
- Closed `browsergrad.kernel.gemm-tile@1` artifacts for dense static rank-2 f32
  operands with exact logical tiles, boundary masks, pairwise-disjoint effects,
  increasing-K round-to-nearest-even accumulation, and explicit prohibitions on
  contraction and reassociation. The package includes one canonical constructor,
  resource-bounded specialization, and exact CPU reference executor.
- Closed `browsergrad.kernel.gemm-exact-f32-input@1` certificates bind one
  logical GEMM specialization to private snapshots and SHA-256 commitments of
  exact host allocation bytes. The bounded proof admits only finite
  nonnegative integer f32 inputs whose every product and output sum is at most
  2^24, making contraction/reassociation value preserving for that input
  domain without weakening the strict logical policy. Existing resident GPU
  buffers are deliberately not authorized by a host-byte certificate.
- Closed `browsergrad.schedule.gemm-tile@1` artifacts that bind one exact
  verified logical GEMM semantic hash to backend-neutral physical tiling,
  cooperative workgroup staging, full-workgroup boundary participation,
  uniform acquire-release barriers, masks, and scalar memory vectors.
- Explicit `/schedule` package export. Schedule verification rejects logical
  numerical policy and backend-specific fields rather than weakening or
  reconstructing the referenced kernel meaning.
- Authority-bound schedule specialization composes one already-prepared logical
  GEMM proof with multiple physical schedules, deriving bounded workgroup,
  staging, and dispatch geometry without repeating logical address proof or
  introducing device/numerical claims.
- Allocation-free, authority-bound standalone layout preparation and coordinate
  tracing. The API verifies shared index semantics without inventing dtype,
  allocation, alias, byte-address, effect, or backend claims.
- Exact monorepo repository metadata required for npm trusted publishing and
  provenance identity verification.
- Versioned rank-2/rank-3 dense-permutation fixtures with exact source/output
  f32 bit patterns and pinned layout/kernel hashes. The exact JSON subpath is
  release-packed, architecture-hash guarded, and excludes routing identity.

## [0.2.0] - 2026-07-15

### Added

- Closed `browsergrad.kernel@1` view-copy envelope tied to the exact semantic
  hash and canonical view IDs of a verified layout artifact.
- Explicit read/write effects, fail-closed overlap, reject-or-exact-f32-fill
  behavior, and structured kernel diagnostics.
- Independently versioned `view-copy@1.0` semantics, one-operation kernel-v1
  envelopes, and a shared positive-affine portable-profile verifier.
- Compiled canonical view accessors plus a bounded CPU reference materializer
  covering transpose, permutation, strided slice, broadcast, padding, dynamic
  shapes, and zero extents without per-element artifact verification or hashing.
- Binding-sensitive specialization hashes; independent element, evaluation-step,
  scratch-memory, and wall-time budgets; cooperative browser yielding and abort;
  destination-injectivity proofs; native buffer-slot, alignment, overlap, and
  shared-memory validation.
- Backend-neutral prepared view-copy specializations shared by the CPU
  reference and device backends, with source-offset caching enabled only for
  interpreters that consume it.
- Canonical view-copy artifact construction plus a dense-permutation wrapper;
  frontends provide no entity IDs, output shape, strides, effects, or alias
  policy, and provenance metadata cannot affect semantic hashes.
- Canonical-JSON-safe, resource-bounded layout substitution that expands
  repeated coordinates into independent tree nodes instead of hidden aliases.
- Explicit `/kernel` package export; the package still has no root barrel or
  runtime dependencies.

## [0.1.0] - 2026-07-15

### Added

- Closed `browsergrad.layout@1` envelope, verification, normalization, and
  deterministic semantic hashing.
- Bounded dimension, constraint, dtype, numerical-policy, allocation, view,
  index-map, coordinate-trace, and alias-trace contracts.
- Explicit `/schema` and `/layout` package exports with no root barrel.
- Independent Python parity oracle and versioned rank-2/rank-3 golden fixtures.
