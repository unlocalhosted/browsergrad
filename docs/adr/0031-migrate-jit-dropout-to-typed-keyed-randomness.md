# ADR-0031: Migrate JIT dropout to typed keyed randomness

- Status: Accepted
- Date: 2026-07-23
- Scope: `@unlocalhosted/browsergrad-jit`,
  `@unlocalhosted/browsergrad-grad`

## Context

`torch.nn.functional.dropout` was a stateful opaque `CUSTOM` callback. Every
realization sampled a new mask and replaced a mutable mask captured by the
closure backward. Realizing the same graph before backward could therefore
make the observed forward and its gradient use different masks. The callback
also bypassed symbolic VJP, checkpoint replay, transform and export decisions,
runtime result validation, and resource bounds. Trace caching could retain one
random graph and silently reuse its prior stochastic state.

Grad separately rejected `p=1`, cast every active output to float32, validated
probability only after identity branches, and duplicated module and functional
implementations.

## Decision

Add public `DROPOUT` and internal `DROPOUT_VJP` opcodes governed by
`browsergrad.jit.framework.functional.dropout.v1`.

The functional signature is
`dropout(input, p=0.5, training=True, inplace=False)`. Probability must be an
exact finite real scalar in `[0,1]`; `training` and `inplace` must be exact
booleans. Validation precedes all branches. `inplace=True` fails explicitly
until BrowserGrad has typed mutation and alias semantics.

The contract has three modes:

- evaluation, `p=0`, and empty input return the exact input and consume no RNG;
- `p=1` emits a deterministic typed drop-all operation with an owning,
  dtype-preserving zero result;
- stochastic training accepts float16, float32, or float64 and captures one
  unsigned 63-bit seed from the global seeded sequence at graph construction.

Forward and both closure and symbolic backward instantiate a local NumPy
generator from that immutable seed. They derive the same elementwise keep mask
and inverted scale without shared mutable state. Re-realization and checkpoint
recomputation therefore replay the exact mask. `manual_seed` reproduces the
operation-key sequence while separately constructed operations receive
separate keys.

Rank, extents, output bytes, projected visits, and output-plus-random-mask
workspace are bounded before seed consumption or allocation. CPU forward and
VJP validate exact ndarray, shape, and dtype at execution and return owning
arrays.

Deterministic drop-all supports leading-axis `vmap`. Stochastic `vmap` refuses
until the public transform accepts an explicit `same` or `different`
randomness policy; BrowserGrad does not silently select one. Training dropout
also refuses ONNX inference export and tensor-plan/WebGPU execution because
neither boundary currently preserves the keyed stochastic contract.

The trace cache refuses graphs containing `CUSTOM`, `RANDOM`, or `DROPOUT`.
Identity dropout remains cacheable because it emits no operation. Checkpoint
defaults to `preserve_rng_state=True`; keyed IR replay is deterministic for
either boolean setting, while opaque stochastic callbacks remain excluded by
the symbolic-VJP requirement.

Grad consumes the same branch and validation profile, preserves float16/32/64
output and gradient dtype, admits dtype-preserving `p=1`, snapshots its mask,
and delegates `nn.Dropout` to the functional implementation.

The old `jit.custom.dropout.v0` identity and `functional.dropout` constructor
site are retired from the current opaque inventory while remaining in the
historical original-ID partition.

## Consequences

Repeated realization, closure backward, functional gradient, and checkpoint
recomputation now agree on one exact dropout mask. Support reporting,
construction, CPU execution, VJP, transform refusal, export refusal, plan
refusal, residency, and materialization all consume one typed contract.

The opaque baseline narrows from seven constructor calls and seven operations
to six of each. The executable typed registry contains thirty-three
retirements and preserves the exact partition of all thirty-nine original
opaque operation identities.

This decision does not claim Philox compatibility, PyTorch bitwise RNG parity,
stochastic `vmap`, ONNX training-graph export, or portable WebGPU dropout.
