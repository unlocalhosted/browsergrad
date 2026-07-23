# ADR-0033: Migrate JIT interpolate to typed spatial resampling

- Status: Accepted
- Date: 2026-07-23
- Scope: `@unlocalhosted/browsergrad-jit`,
  `@unlocalhosted/browsergrad-grad`

## Context

`torch.nn.functional.interpolate` was the final framework operation executed by
the JIT through an opaque NumPy `CUSTOM` callback. Its graph exposed neither
shape nor coordinate semantics to transforms and backends. Nearest mode had an
ad hoc closure derivative with nested output-pixel loops; bilinear mode built a
forward graph whose backward failed only during execution. Symbolic gradients,
functional gradients, vmap, ONNX export, resource limits, typed runtime
validation, and portable-backend decisions were absent.

The old callback rounded scale-derived output extents instead of using the
floor rule, recomputed coordinate scales unconditionally, forced float32
results, and accepted coercive or ambiguous argument shapes. Grad separately
duplicated these behaviors and used nested Python loops for both derivatives.

## Decision

Add public `INTERPOLATE_2D` and internal `INTERPOLATE_2D_VJP` opcodes governed
by `browsergrad.jit.framework.functional.interpolate-2d.v1`.

The initial closed profile accepts exact rank-four `(N,C,H,W)` float16,
float32, or float64 input. It supports nearest and bilinear modes with exactly
one of positive scalar/pair `size` or `scale_factor`. Scale-derived output
extents use `floor(input_extent * scale)`. Nearest requires
`align_corners=None`; bilinear maps `None` to false and accepts an exact
boolean. `recompute_scale_factor` is available only with an explicit scale, and
`antialias=True` remains outside v1.

Coordinate geometry is immutable graph data. Explicit scale without
recomputation uses its reciprocal. Size requests and recomputed scales use the
input/output extent ratio. Aligned bilinear mode preserves endpoint centers.
The CPU reference uses vectorized NumPy gather for forward execution and
bounded flattened `add.at` scatter for the transpose, accumulating float16 in
float32 and returning a fresh array in the declared dtype. Input and output
extents, bytes, projected element visits, and conservative workspace are
validated before allocation or execution.

Closure autograd and symbolic VJP consume the same contract and transpose.
Leading-axis vmap preserves the four public spatial-tensor axes and broadcasts
captured cotangents when batching a symbolic derivative. Checkpoint replay
remaps the derivative to the cloned typed forward.

ONNX opset 17 export uses `Resize`. Nearest emits asymmetric coordinates with
floor selection. Bilinear emits half-pixel or aligned-corner coordinates.
Explicit non-recomputed scale requests use ONNX scales; size and recomputed
requests use sizes. Gradient-only VJP export is refused.

The tensor GPU plan explicitly refuses both opcodes until BrowserGrad has a
canonical spatial-resampling lowering and kernel. Parser acceptance, CPU
execution, and ONNX representation therefore do not imply portable WebGPU
execution.

Grad implements the same public profile, coordinate rules, dtype preservation,
resource ceilings, vectorized forward, and transpose derivative. A shared
conformance fixture pins nearest, bilinear, aligned-corner, non-integral scale,
recomputed-scale, dtype, ownership, derivative, and refusal behavior.

The old `jit.custom.interpolate.v0` identity,
`functional.interpolate` CUSTOM constructor site, and final legacy NumPy
callback policy are retired from the current opaque inventory while the
historical operation ID remains in the original-ID partition.

## Consequences

The JIT framework registry contains thirty-five typed retirements. The opaque
baseline narrows from five constructor calls and operations to four: two
explicit legacy WebGPU routes and two constructor-only compatibility surfaces.
No framework-owned NumPy callback remains in the opaque inventory.

Nearest and bilinear now have replay-safe closure and symbolic derivatives,
functional gradients, leading-axis batching, checkpoint behavior, explicit
ONNX representation, and bounded failure semantics across JIT and Grad.

This decision does not claim rank-three or rank-five interpolation, linear,
bicubic, trilinear, area, nearest-exact, antialiasing, size/scale combinations,
arbitrary-axis vmap, gradient ONNX export, tensor-plan lowering, WebGPU
execution, or device residency.
