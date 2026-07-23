# ADR-0032: Migrate JIT BatchNorm1d to typed ordered state

- Status: Accepted
- Date: 2026-07-23
- Scope: `@unlocalhosted/browsergrad-jit`,
  `@unlocalhosted/browsergrad-grad`

## Context

`nn.BatchNorm1d` executed through an opaque NumPy `CUSTOM` callback. Training
mutated raw module arrays inside that callback and saved normalized values in a
mutable closure dictionary. Every realization recomputed the callback, so
reading an output and then running backward updated the running statistics
twice. The implementation normalized with biased batch variance and also wrote
that biased value into `running_var`, did not expose
`num_batches_tracked` or cumulative averaging, and could let later state
mutation change an earlier eval graph.

The callback had no symbolic VJP, functional-gradient, checkpoint, transform,
export, resource, runtime-array, or portable-backend contract. Grad duplicated
the same biased running variance and replaced public running-buffer objects on
every training call.

## Decision

Add public `BATCH_NORM_1D` and internal
`BATCH_NORM_1D_STATS_UPDATE`/`BATCH_NORM_1D_VJP` opcodes governed by
`browsergrad.jit.framework.module.batch-norm-1d.v1`.

The initial closed profile accepts exact float32 `(N,C)` and `(N,C,L)` input,
float32 affine parameters and running statistics, positive bounded channel
count, finite non-negative epsilon, `None` or finite `[0,1]` momentum, and
exact boolean flags. Batch-stat modes require more than one sample per channel.
Rank, extents, output bytes, projected element visits, and conservative
workspace are bounded before execution.

Training normalization uses the biased population variance. Persistent
`running_var` uses the unbiased estimator, and `num_batches_tracked` is a
scalar int64 buffer. Fixed momentum updates use
`(1-momentum)*old + momentum*observed`; `momentum=None` uses
`1/num_batches_tracked`.

Each tracked training call reserves the next sequence on a session-owned effect
stream bound to the exact three target buffers. The BufferTable retains only
reserved/applied watermarks per stream, keeping replay metadata constant-memory
per stateful module instead of per forward call. It validates the stream token,
effect kind, sequence, all target buffer identities, shapes, and dtypes before
atomically mutating the three registered arrays in place. An effect sequence
may commit once. Later training calls depend on the preceding uncommitted
update, preserving construction order even when outputs realize out of order;
applied tails are removed from subsequent graphs.

Running-state buffer names use a minted module token rather than `id(module)`.
Long-lived Pyodide sessions retain registered buffers after short-lived Python
modules are collected, so recyclable process-local object IDs are not valid
storage identities.

State inspection and loading synchronize pending updates. Tracked evaluation
synchronizes its predecessor and captures an immutable running-stat snapshot,
so later module mutation cannot alter that forward or its derivative. Untracked
training and evaluation use pure batch statistics.

Forward, closure VJP, symbolic VJP, and functional gradients consume the same
typed contract and runtime ndarray validation. Checkpoint rewriting remaps
`vjp_of` to the cloned forward node. Clones retain the original effect identity,
so recomputation cannot replay the running-state mutation.

The v1 transform and backend decisions are explicit refusals. `vmap` does not
guess whether a leading axis is a new independent batch or part of the
normalization sample set, and it cannot duplicate running-state effects. ONNX
export waits for an admitted immutable-running-stat profile. Tensor-plan and
WebGPU wait for canonical normalization and ordered device-state lowerings.
Graphs containing the state-update opcode are excluded from trace caching.

Grad consumes the same closed public contract. It registers all three state
buffers as non-parameters, updates them in place, saves immutable backward
statistics and affine weight values, supports cumulative averaging and reset
methods, and rejects dtype or shape drift before NumPy normalization.

The old `jit.custom.batch-norm-1d.v0` identity and
`nn.batch-norm-1d-forward` constructor site are retired from the current opaque
inventory while remaining in the historical original-ID partition.

## Consequences

One training call now advances running state exactly once across repeated
realization, backward, functional gradients, checkpoint recomputation, and
state serialization. JIT and Grad agree on batch versus persistent variance,
counting, cumulative averaging, affine derivatives, validation, and public
buffer identity.

The opaque baseline narrows from six constructor calls and six operations to
five of each. The executable typed registry contains thirty-four retirements
and preserves the exact partition of all thirty-nine original opaque operation
identities.

This decision does not claim float16/bfloat16 normalization, mapped
normalization, ONNX BatchNormalization export, portable WebGPU execution, or a
device-resident state effect.
