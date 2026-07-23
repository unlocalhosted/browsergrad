# ADR-0030: Migrate JIT cross entropy to a typed stable class-axis loss

- Status: Accepted
- Date: 2026-07-23
- Scope: `@unlocalhosted/browsergrad-jit`,
  `@unlocalhosted/browsergrad-grad`

## Context

`torch.nn.functional.cross_entropy` was one of the remaining public JIT
surfaces implemented as an opaque `CUSTOM` NumPy callback. The callback only
accepted two-dimensional logits and one-dimensional class indices. It omitted
class weights, ignored targets, spatial and unbatched inputs, probability
targets, label smoothing, legacy reduction aliases, symbolic VJP, `vmap`, and
portable export decisions. Its callback result and captured backward state
also bypassed the typed framework-operation validator.

Cross entropy is the stable composition of log-softmax and a class-axis loss.
It must not materialize a naive softmax followed by `log`, and it must preserve
the distinction between discrete class-index targets and differentiable
floating probability targets.

## Decision

Add public `CROSS_ENTROPY` and internal `CROSS_ENTROPY_VJP` opcodes governed by
`browsergrad.jit.framework.functional.cross-entropy.v1`.

The contract:

- accepts floating `float16`, `float32`, or `float64` logits shaped `(C)`,
  `(N,C)`, or `(N,C,...)`;
- derives the class axis after any transform-owned leading batch axes;
- accepts either exact `int64` index targets with the class axis removed or
  same-shape floating probability targets with the logits dtype;
- supports optional class weights, ignored index targets, `none`, `sum`, and
  `mean` reductions, legacy reduction aliases, and finite label smoothing in
  `[0,1]`;
- computes half inputs in float32 and uses a max-subtracted log-sum-exp;
- uses selected-class weight totals for index-target means and position count
  for probability-target means, matching the two PyTorch contracts;
- propagates logits gradients for both target modes and target gradients only
  for floating probability targets; class weights remain non-differentiable;
- snapshots eager Grad backward state and validates exact runtime ndarray
  shape and dtype before JIT execution;
- bounds rank, extents, output bytes, projected visits, and complete
  stable-softmax/target/gradient workspace before allocation.

Leading-axis `vmap` is admitted for inputs, either target mode, and captured or
mapped class weights. Opset-17 ONNX export uses
`SoftmaxCrossEntropyLoss` only for the unmapped class-index profile with zero
label smoothing. Probability targets, smoothing, and mapped class axes fail
closed at export.

Tensor-plan and WebGPU execution remain refused until a canonical stable
class-axis loss reduction and gradient lowering exists. CPU execution and ONNX
export do not establish a portable device claim.

The old `jit.custom.cross-entropy.v0` identity and
`functional.cross-entropy` constructor site are retired from the current
opaque inventory while remaining in the historical original-ID partition.

## Consequences

The public CrossEntropyLoss modules now retain registered class-weight buffers,
work across sessions in JIT, and expose the complete admitted functional
signature. The executable registry, closure autograd, symbolic VJP, transform,
export, plan refusal, and support-reporting paths all consume the same typed
contract.

Making cross entropy symbolically differentiable exposed a pre-existing ReLU
graph mismatch: its `WHERE` node had three IR operands but a one-proxy backward
context. ReLU now uses the canonical `where` constructor so symbolic and
closure autograd traverse the same operand topology.
