# BrowserGrad Semantic Systems Architecture and Low-Level Requirements

- **Status:** normative platform architecture; implementation status is not implied
- **Last reviewed:** 2026-07-22
- **Implementation ledger:**
  [`docs/internal/package-requirements-implementation-ledger.md`](../internal/package-requirements-implementation-ledger.md)
- **Scope:** compiler frontends, tensor/layout semantics, kernel semantics,
  scheduling, CPU reference execution, WebGPU execution, optional native
  execution, JIT/eager integration, runtime capability contracts, wire formats,
  and release evidence.

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY**
describe requirement strength. They are used deliberately rather than as
general emphasis.

## Implementation Checkpoint — Active 2026-07-22

Gates 0 through 2 and Gates 4 through 5 are verified; Gate 3 remains active;
Gate 6 is in progress; Gate 7 has not started. Gate 4 is verified for its initial
closed portable profile only: dense row-major, 4-byte-aligned, certified exact-input `f32`
GEMM. The strict native producer, pinned no-shell Clang-Wasm executor,
isolated clean/reproducibility authorities, and independent raw-Wasm ABI review
are implemented on `main`. The current source additionally owns exact
frontend-work instrumentation, the local Wasm C-ABI runner, and a canonical
Worker result-control encoder. It now pins the current clean-built 27,125-byte
generated factory inside one exact 571,098-byte package-owned compiler Worker
module and a separate exact 151,555-byte package-owned raw-Wasm verifier Worker;
both have zero static or dynamic imports. The production controller captures
browser effects at module evaluation, runs the exact verifier before preparing
the compiler invocation, transfers only its bounded canonical derivative
evidence, composes only package-owned Worker bytes, and terminalizes every
Worker/Blob/timer/listener effect before returning. Caller-supplied verifier or
conformance records cannot substitute for the retained host authority.
Immediately before minting execution evidence the controller re-unwraps the
exact protocol-issued validated frame and cross-binds its invocation, request,
profile, request-binding, artifact, and verifier identities; a structural copy
cannot substitute for that authority. Validated terminal frames retain the
exact observed-execution, invocation, request, profile, asset-manifest, VFS,
runtime-ABI, package-Worker, and verifier authority chain. An accepted layout
from that exact chain can be prepared through the shared Gate 2 layout seam as
an opaque observed semantic candidate. A separate host-only policy authority
can now independently admit the exact predicate, trust-store digest, builder,
key, and policy version without changing the signature binding's
`producerTrusted=false` claim. Only that exact opaque producer authority plus
the exact observed candidate may mint canonical local semantic-lowering
authority. The transition re-authenticates the retained profile, manifest,
asset-set, package-Worker, invocation, request, artifact, and prepared layout;
backend execution, distribution, and release remain false. The same exact
observed Worker lineage can now prepare an opaque `view-copy` candidate. Only
that candidate plus the independently admitted exact producer may converge
into the existing canonical authorized frontend artifact. Candidate and
authorization identities exclude allocation sizes and byte offsets; explicit
storage remains a later lowering fact and is never inferred from CuTe
`cosize`. Current browser tests use a synthetic authority fixture, and the
producer tests use a synthetic key and policy, so no real production producer
or valid production browser-Worker compile has been observed. Hostile
view-copy graphs are checked with non-recursive, target-intrinsic cycle
detection before any semantic authority can be minted. The resulting exact
producer-neutral authorized `view-copy` entry can lower genuine f32 ABI,
tensor-engine, affine-layout, copy-effect, and explicit runtime-allocation
facts into the canonical Gate 2 layout/kernel artifacts. The initial profile
is positive flat static rank 2 or rank 3, two distinct non-null global pointers,
synchronous portable copy, reject on an invalid source, and forbid overlap.
Its CPU reference proves the pinned rank-2 and `(2,3,4)` rank-3 transposes
bit-for-bit with nonzero allocation offsets. The two exact canonical payloads
reproduced by that authorization path now also execute bit-for-bit through the
canonical WebGPU backend, including complete destination comparison and
nonzero-offset canaries. The required actual-WebGPU lane executes both cases.
This browser evidence is fixture-payload convergence across realms: it
explicitly records
`productionBrowserCompileObserved=false` and does not mint backend,
distribution, or release authority. Capability commits `a8a861e2`,
`c4e2d799`, `08f4b102`, `6355289c`, `e546a124`, and `dca33be5`, plus
their tests, prove the candidate/authorization, rank-2/rank-3 lowering, and
required-WebGPU boundaries. Exact-source CI run `29811673981` completed with
all eight jobs green. The complete compiler suite passes 95 files and 1,591
tests at that checkpoint. Gate 4 capability commits `2cb7cea3` through
`fe5581d3` now define one frontend-neutral logical GEMM tile independently from
physical schedule, staging, and backend mapping; derive and verify separate
8x8x8 and 16x16x16 scalar cooperative schedules; and bind exact retained input
bytes to the logical and schedule-specialization hashes. A typed compiler
artifact can lower into the same canonical constructor, but it does not claim
source-body equivalence or a production browser compile. The kernels package
implements zero-filled cooperative workgroup staging, boundary-masked stores,
uniform barriers, and increasing-K scalar accumulation. Required real-WebGPU
evidence executes the irregular 17x23 by 23x19 case under both schedules and
compares every destination byte against one CPU reference and against each
other. The terminal evidence reports `portable-webgpu-core`,
`portable-relegalized`, and `bit-exact-on-certified-inputs`; it makes no native
MMA, preserved CUDA/CuTe schedule, general-f32, resident-buffer, distribution,
or source-compatibility claim. CI run `29818182317` contains the successful
dedicated required semantic-GEMM WebGPU job; its whole run is not a success
record because the required-native lane exposed the stale package Worker pin.
Commit `343523fe` regenerates and repins the deterministic zero-import compiler
Worker at 571,098 bytes and SHA-256
`01a4c1d10d606773bfa241284160f3af787dec856e1a17e22edd7c34dae043a3`.
Gate 5's first software slice now defines the closed frontend-neutral
`browsergrad.kernel.attention-forward@1` artifact. It binds verified rank-4
Q/K/V/output views, exact f32 inverse-square-root scale bits, non-causal or
upper-left causal masking, stable-softmax phases, finite scaled-score and
online-state preconditions, a named `1e-4` absolute-or-relative comparison
policy, pairwise-disjoint effects, and explicit VJP refusal. The artifact
contains no physical tiles, staging, barriers, WGSL, backend, or performance
meaning and grants no CPU or WebGPU execution claim. A separate closed
`browsergrad.schedule.attention-online-kv-tile@1` artifact now binds the exact
logical hash to physical query/key tiles, increasing key traversal,
cooperative single-buffered K/V staging, tile-wise online-softmax rescaling,
uniform workgroup barriers, scalar vectors, and masks that exclude invalid or
logical-mask keys before online-state updates. It contains no logical dtype,
scale, view, comparison, backend, execution, performance, or named fused-kernel
claim. A schedule-independent CPU oracle now proves the dense row-major address
profile under explicit work/time/cancellation limits, snapshots fixed unshared
inputs before yielding, enforces every finite-domain requirement, evaluates
the stable-softmax and weighted-value phases with stepwise f32 rounding, and
commits destination bytes only after complete success. Its comparator
implements the named absolute-or-relative policy and rejects non-finite output.
This grants CPU reference evidence only. Schedule specialization now composes the
exact prepared logical and schedule authorities and derives bounded workgroup,
K/V staging, private-state, key-tile, and dispatch geometry without granting
device legality or preservation. Kernels now consumes that exact composition
to prepare a bounded scalar WGSL module with cooperative K/V workgroup staging,
two uniform barriers per key tile, private Q/output state, tile-wise online
softmax, pre-update causal/tail masks, and suppressed boundary stores. The
backend identity reports `block-tiled-kv-online-softmax-forward` and
`portable-relegalized`; preparation alone grants no device execution,
numerical-preservation, performance, FlashAttention-v2, or frontend claim. The
host execution path now snapshots exact fixed unshared finite Q/K/V allocations
before yielding or touching WebGPU, verifies device limits, scopes asynchronous
pipeline/dispatch errors, bounds waits, and publishes only a complete finite
destination. Required headed Chromium evidence on Apple Metal 3 executes
causal and non-causal irregular rank-4 inputs under 8x8 and 8x16 schedules; all
four complete outputs pass the semantic-core `1e-4` absolute-or-relative CPU
comparator and same-mask cross-schedule comparison. The trace reports that
comparison is required rather than converting one successful run into a global
numerical claim. A separate required performance record compares the production
host APIs for `(B=1,H=2,Sq=256,Sk=256,D=Dv=32)` after 16 warmups with 20
alternating paired samples, complete readback, and queue drain. It names the
existing row-wise online-softmax implementation only as a baseline, retains raw
samples and lifecycle differences, and explicitly makes no superiority or
regression-threshold claim. Correctness remains in the separate conformance
record. The initial closed f32 Gate 5 profile therefore meets its exit without
claiming FlashAttention-v2, general dtype/layout coverage, or frontend schedule
preservation. Two
distinct cache-free builds now prove exact
extractor Wasm/factory reproducibility; the package binds their canonical v3
evidence without claiming reproducibility of the still-incomplete distributed
asset set.

Gate 6 has started by retiring public `Tensor.expand`, `Tensor.abs`,
`Tensor.sign`, `Tensor.sin`, `Tensor.cos`, `Tensor.clamp`, `Tensor.flip`,
`Tensor.gather`, `Tensor.masked_fill`, `Tensor.prod`, `Tensor.repeat`,
`Tensor.repeat_interleave`, `Tensor.tril`, `Tensor.triu`, `Tensor.cumsum`,
`Tensor.var`, top-level `torch.cat`, and top-level `torch.stack` from the frozen opaque callback
inventory. Expand emits the existing typed `BROADCAST_TO`
primitive. One shared contract validates exact arity, closed shape arguments,
output-shape identity, dtype preservation, rank direction, and broadcast
compatibility at construction and again at CPU, VJP, vmap, ONNX, and
tensor-plan boundaries so mutation of the legacy argument dictionary fails
locally. Valid calls retain owning CPU materialization and closure autograd;
symbolic VJP sums expanded axes, vmap keeps its batch axis outside the declared
broadcast, ONNX emits `Expand`, and materializing/resident tensor-plan routes
contain no `CUSTOM`. Grad eager `Tensor.expand` consumes the same cross-package
shape/dtype conformance fixture, rejects the same invalid dimension classes
before NumPy execution, preserves float16 and integer dtypes, and explicitly
retains owning contiguous materialization rather than claiming PyTorch view
aliasing. `abs` and `sign` now emit typed `ABS` and `SIGN`; their shared unary
profile preserves real-numeric shape and dtype, rejects bool, returns owning
CPU arrays, defines closure and symbolic gradients, supports leading-axis
vmap, and exports direct ONNX unary nodes. Tensor-plan and WebGPU execution
remain explicit refusals because this slice adds no portable kernel. `sin` and
`cos` now emit mutually differentiable typed `SIN` and `COS`. They preserve
float16/32/64 shape and dtype, reject bool/integer inputs before execution,
provide owning CPU results and typed symbolic derivatives, vmap across a
leading batch axis, and emit direct ONNX unary nodes. Their plan/WebGPU profile
also remains an explicit refusal. Clamp now emits typed `CLAMP` with closed
finite optional bounds, floating dtype preservation, inclusive-bound closure
and symbolic gradients, leading-axis vmap, and ONNX `Clip` optional-input
lowering. Hostile scalar coercion and integer dtype drift fail before UOp
construction; tensor-plan/WebGPU remain explicit refusals. The opaque baseline
is therefore narrowed to 18 constructor calls and 21 operations under ADR-0002
and ADR-0004 through ADR-0018. Flip now emits typed `FLIP` with one strictly
normalized axis, owning CPU reversal, involutive closure and symbolic VJP,
leading-batch vmap axis shifting, and ONNX `Slice` export for the exact
float32/int32/int64/bool exporter profile. It rejects bool,
floating, hostile-conversion, scalar-rank, and out-of-range axes before
execution. Tensor-plan and WebGPU explicitly refuse its negative-stride
profile, so this migration does not widen the Gate 2 positive-stride contract.
Repeat now emits typed `REPEAT` with bounded exact tile multipliers, owning
dtype-preserving CPU realization, tile-block closure and symbolic reduction,
batch-axis-preserving vmap, and ONNX `Tile` for the exact
float32/int32/int64/bool exporter profile. Its tensor-plan/WebGPU profile
explicitly refuses execution until canonical tile/index layout semantics
exist. Grad consumes the same repeat conformance fixture, preserves input
dtype, and rejects malformed multipliers before NumPy. Repeat-interleave now
emits typed `REPEAT_INTERLEAVE` with one strictly normalized selected axis and
bounded exact repeat count. CPU returns an owning dtype-preserving array;
closure and symbolic VJP split the expanded axis into source/repeat blocks and
sum the repeat block; vmap shifts the axis past its leading batch dimension;
ONNX emits an exact `Unsqueeze`/`Tile`/`Reshape` decomposition for
float32/int32/int64/bool. Its tensor-plan/WebGPU profile explicitly refuses
execution until canonical selected-axis replication semantics exist. Grad
consumes the shared repeat-interleave conformance fixture and preserves output
and gradient dtype. Product now emits typed `PROD` with canonical static axes,
exact keepdims, owning dtype-preserving scalar/tensor CPU results, and a
zero-aware closure/symbolic derivative. Vmap shifts every reduction axis past
its batch dimension, and ONNX emits `ReduceProd` for float32/int32/int64 while
other exporter dtypes fail explicitly. Tensor-plan/WebGPU remain explicit
refusals until portable product reduction exists. Grad consumes the shared
product fixture, preserves output and gradient dtype, and corrects the
single-zero derivative. Through product this is ten migrated operations, not Gate 6
completion: remaining Grad view/dtype debt,
runtime/profile consumption, and the remaining advertised opaque operations
remain open.

Gather now emits typed `INDEX` with one exact normalized axis and a same-rank
int64 index. The closed contract preserves source dtype, derives the output
shape from the index, allows smaller non-gather extents, rejects NumPy-style
negative index wrapping through runtime bounds checks, and returns an owning
CPU result. Closure and symbolic VJP use deterministic duplicate-accumulating
`SCATTER_ADD`; paired vmap shifts the axis past the shared leading batch;
ONNX emits `GatherElements` for float32/int32/int64/bool. Tensor-plan and
WebGPU explicitly refuse until a deterministic bounds-checked index/scatter
lowering exists. Grad consumes the same gather fixture and preserves output
and gradient dtype. Grad indexing, reshape, transpose, and permute also preserve
float16, bool, and integer storage dtype plus compatible NumPy aliasing, so an
int64 index cannot silently become float32 while passing through an ordinary
slice/unsqueeze path. This is eleven migrated operations, not Gate 6
completion.

Variance now emits typed `VAR` with canonical sorted static axes, exact signed
32-bit correction, exact keepdims, floating dtype preservation, and owning CPU
scalar/tensor results. The default correction is one; the legacy `unbiased`
alias is accepted only when correction is absent. Closure and symbolic VJP use
the centered correction-aware derivative; vmap shifts every reduction axis
past the leading batch; ONNX decomposes the exact float32 profile into
`ReduceMean`, `Sub`, `Mul`, `ReduceSum`, and `Div`. Tensor-plan and WebGPU
explicitly refuse until a portable variance reduction exists. Grad consumes
the same value/dtype/refusal fixture and no longer casts variance or its
gradient to float32. This is twelve migrated operations, not Gate 6
completion.

Masked fill now reuses typed `WHERE(mask, fill, source)`. The closed contract
requires an actual bool tensor mask that broadcasts into but cannot enlarge
the source, an exact scalar `CONST` normalized to the source dtype, and a
source-shaped dtype-preserving output. CPU returns an owning array; closure and
symbolic VJP route source cotangents through the mask complement; vmap supports
a leading mapped source with captured or mapped mask broadcasting; ONNX emits
`Where` for float32/int32/int64/bool. The planner validates and refuses `WHERE`
until the kernels tensor-plan owns portable masked selection. Grad consumes the
same mask/value/dtype/refusal fixture and no longer coerces masks or casts
results and gradients to float32. This is thirteen migrated operations, not
Gate 6 completion.

Lower triangular selection now emits typed `TRIL` over the final two axes of
a matrix or batch of matrices. Its closed contract requires rank at least two,
preserves source shape and dtype, accepts only an exact built-in or NumPy
integer diagonal, and saturates that diagonal into the matrix-derived
all-zero/all-input semantic range before IR construction. CPU returns an
owning array; closure and symbolic VJP apply the same idempotent triangular
selection; vmap preserves the final matrix axes while inserting a leading
batch; ONNX emits `Trilu` with `upper=0` for float32/int32/int64/bool.
Tensor-plan and WebGPU explicitly refuse until portable triangular selection
exists. Grad consumes the same values, dtype, saturation, gradient, and
refusal fixture. This is fourteen migrated operations, not Gate 6 completion.

Upper triangular selection now emits typed `TRIU` through the same canonical
triangular-selection seam. Its exact nonempty diagonal range is
`[1 - rows, columns]`, from the all-input to the all-zero representative; empty
matrices canonicalize to zero. Construction and every execution/transform
boundary share the strict rank, dtype, exact-integer, shape, and mutation
checks. CPU returns an owning dtype-preserving array; closure and symbolic VJP
apply the same idempotent upper selection; vmap preserves the final matrix
axes; ONNX emits `Trilu` with `upper=1` for float32/int32/int64/bool; and
tensor-plan/WebGPU refuse until portable triangular selection exists. Grad now
exposes equivalent instance and top-level spellings and consumes the shared
two-variant triangular conformance harness. This is fifteen migrated
operations, not Gate 6 completion.

Cumulative sum now emits typed `CUMSUM` with one exact normalized axis and a
closed scan dtype contract. Floating defaults preserve dtype; integral and
boolean defaults promote to int64; an explicit supported dtype casts before
accumulation. CPU returns an owning exact-dtype array, closure and symbolic VJP
scan the cotangent in the opposite direction, vmap shifts the scan axis past a
leading batch, and ONNX emits exact `Cast`/`CumSum` wiring for the supported
float32/int32/int64 output profile. Tensor-plan and WebGPU explicitly refuse
until portable scan lowering exists. Grad consumes the same value, axis,
dtype, empty, hostile-input, mutation-refusal, and gradient fixture. This is
sixteen migrated operations, not Gate 6 completion.

Concatenation now emits typed variadic `CONCAT` over one exact normalized
existing axis. A closed contract requires a nonempty plain tuple/list of exact
same-session tensors, caps arity at 1,024 and output storage at 256 MiB,
requires matching substantive ranks and non-concatenated extents, and preserves
the PyTorch-compatible rank-1 `(0,)` empty exception. Dtype promotion is closed
over bool, uint8, signed int8/16/32/64, and float16/32/64 using dimensioned-
tensor category precedence. CPU realization casts inputs and returns one owning
array in the exact declared dtype. Closure and symbolic VJP split the cotangent
through typed internal `NARROW`, reshape a legacy empty segment, and cast each
floating gradient to source dtype. Vmap shifts the axis past a leading batch
and broadcasts captured inputs. ONNX emits exact `Cast`/`Concat` wiring for
float32/int32/int64/bool and explicitly refuses the remaining output dtypes.
Tensor-plan and WebGPU refuse until canonical variadic copy lowering exists;
`out=` remains refused until a shared effect contract exists. Grad consumes the
same value, shape, axis, dtype-promotion, legacy-empty, ownership, gradient,
resource, hostile-input, mutation-refusal, and boundary fixture. This is
seventeen migrated operations, not Gate 6 completion.

Stacking now emits typed variadic `STACK` over one exact normalized inserted
axis. A closed contract requires a nonempty plain tuple/list of no more than
1,024 exact same-session tensors, identical input shapes, and an output no
larger than 256 MiB. Scalars and identically shaped empty tensors remain valid.
It reuses concatenation's closed dimensioned-tensor promotion lattice. CPU
casts every input and returns an owning `np.stack` copy; closure and symbolic
VJP select each static source index through typed internal `NARROW`, reshape
away the inserted axis, and cast floating gradients to source dtype. Vmap
shifts the stack axis and broadcasts captured inputs. ONNX emits exact
per-input `Cast`/`Unsqueeze` plus `Concat` for float32/int32/int64/bool and
explicitly refuses other output dtypes. Tensor-plan/WebGPU and `out=` remain
explicit refusals until canonical variadic copy lowering and a typed effect
contract exist. Grad consumes the same eager/lazy conformance fixture. This is
eighteen migrated operations, not Gate 6 completion.

The first executable framework-operation registry now removes the hand-written
support-reporting seam for typed migrations. Its bounded package-owned v1 JSON
records bind `Tensor.abs`, `torch.cat`, `Tensor.clamp`, `Tensor.cos`, `Tensor.expand`,
`Tensor.flip`, `Tensor.gather`, `Tensor.masked_fill`, `Tensor.prod`,
`Tensor.repeat_interleave`, `Tensor.repeat`, `Tensor.sign`, `Tensor.sin`,
`torch.stack`, `Tensor.tril`, `Tensor.triu`, `Tensor.cumsum`, and `Tensor.var` to the same
validators invoked by construction and every admitted execution, transform,
export, or plan boundary. Import rejects duplicate keys,
open fields, unknown decisions, invalid versions, duplicate identities, and
records without an exact executable validator. Public
`framework_operation_support()` returns a detached deterministic table with
explicit shape, dtype, CPU, autograd, transform, export, plan, WebGPU-profile,
residency, and materialization decisions. A WebGPU profile is eligibility, not
device availability or execution evidence. The architecture gate independently
checks the registry and preserves the exact partition of the original 39
opaque IDs into 21 still-opaque and eighteen typed retirements. ADR-0003 records
this public contract. The table currently covers typed migrations only;
completing the remaining operation families and making runtime/profile UI
consume these records remain Gate 6 work.

Browser asset manifest v1.4 now binds one build-signature predicate, exact
trust-store digest, and canonical builder allowlist into the profile-pinned
asset-set identity. One strict DSSE/in-toto verifier rebinds a valid P-256
signature to the exact profile/compilation contract, manifest, build-input
lock and recipe, package Worker/factory, and cycle-free build subject. This
remains a manifest-policy signature binding: the opaque signature result
explicitly keeps `producerTrusted`, exact-asset verification, complete
reproducibility, legal approval, distribution, Worker execution, and release
readiness false. The separate trust-policy transition is implemented and
cannot mutate that narrower authority. The tests still use ephemeral synthetic
keys and policy bytes. A package-controlled production policy, externally
controlled key, and externally issued statement over the exact current build
subject remain required before the current distributed producer can be claimed
as trusted.

The header-pack harness now binds seven exact source archives to the build lock:
LLVM 22.1.8, CUTLASS 3.7.0, CUDA 12.6.3 CCCL/cudart/nvcc, and Ubuntu Noble
glibc/Linux UAPI cross-development packages. It streams and verifies all
252,406,685 archive bytes and package-pins the current Darwin arm64
normalization closure: exact `/usr/bin/bsdtar` bytes/version, exact Node 25.9.0
executable bytes, Zstandard 1.5.7, empty runtime flags, and absent
`NODE_OPTIONS`. It uses that bounded Node decoder for Debian `data.tar.zst` and
normalizes only the eight selected header subtrees plus eight exact upstream
license/copyright review files through a strict streamed tar parser. The parser
rejects traversal, links, special files, duplicate virtual paths, malformed
PAX metadata, invalid checksums/padding, and budget overflow. Extracted files
use a content-addressed flat host store, so a case-insensitive macOS filesystem
cannot collapse distinct Linux virtual paths such as `xt_CONNMARK.h` and
`xt_connmark.h`.

One same-process opaque pipeline rereads all 5,769 selected files and their
67,092,008 content bytes, detects conflicting overlays, and materializes the
five canonical BrowserGrad VFS packs. The locked WebAssembly-only Clang stage
activates none of LLVM 22.1.8's ARM/AArch64/RISC-V TableGen resource-header
rules. The pipeline verifies the exact 25,049-byte upstream CMake manifest,
records an empty generated-file set, and omits that build-only manifest from
the distributed pack. The resulting 5,768 files occupy 67,470,214 pack bytes
and cover the complete configured Clang resource output, libc++, CUDA, CUTLASS,
and the Linux sysroot. The same pipeline admits the exact 49,142-byte CUDA
12.6.3 redistribution index and writes a canonical 1,203,103-byte
`license-inventory.json`. Its 5,768 unique file-map entries bind every
distributed virtual path and content identity to the materialized pack,
license component, and relevant package notice while also binding the complete
ten-resource notice set, eight extracted upstream evidence files totaling
250,207 bytes, and the selected CUDA index records. The pipeline now copies the
ten verifier-retained package notice snapshots into their exact declared
component-license paths and constructs a deterministic 115,316-byte aggregate
notice containing ten components, nine of them third party. The eleven new
notice files total 226,326 bytes, are created without clobber at mode `0400`,
are independently reread, and complete an exact 17-file private output tree
with the five packs and review input. Three final direct runs produced stable
review-input, notice-materialization, aggregate, and path-independent pipeline
identities in 23.1 to 25.4 seconds. These are real
source-derived, independently inspected non-release pack observations. They do
establish the complete configured header universe under one package-reviewed
builder identity and the complete engineering input for external header-file
license review. They do not establish independent third-party implementation
attestation, external review or approval, full distribution
reproducibility, independently trusted signed provenance, Worker execution, or
release authority.

The two-root reproducibility command now runs that exact pipeline twice from
one common archive/index/tool closure under four distinct non-overlapping source
and pack roots. It then concurrently rehashes every immutable output, checks
the complete file/directory trees before and after hashing, and confirms all 17
paths, 68,899,643 bytes, and identities match. Two final direct commands
completed in 44.45 and 46.84 seconds with stable reproducibility ID
`bg.cpp.browser-header-distribution-reproducibility.sha256.986bcd7b462b1bac0653f33e61dba69403295fd69df4fdb1780bb27092de9337`
and output-verification ID
`bg.cpp.distribution-output-file-verification.sha256.6eb7779eee72c9f65589656ac28b02f10caa9d9362a98d1d981f198de13a3937`.
This proves exact reproducibility only for the five header packs, license
inventory, ten component notices, and aggregate notice. The Wasm/factory,
Worker, remaining deterministic outputs, detached provenance, external legal
review, and approved package asset set are not part of this subset, so full
distribution reproducibility and release remain false.

The package now pins the path-independent 4,042-byte header reproducibility
record at SHA-256
`eb85b8fb54d1bad1b932bd4bfb3ffc7aff8d66d6185c9926ddd3c9f6084d918a`.
Admission accepts only those exact bytes, verifies the current build-input lock
and its resource identity, validates the sorted 17-output projection and byte
total, and independently rederives both the output-verification and
reproducibility identities before minting an opaque subset authority. This
closes the gap between local live-run evidence and package-consumable technical
evidence without turning reproducible header bytes into legal approval, signed
provenance, full-distribution reproducibility, Worker execution, or release
authority.

The exact source plan is
`bg.cpp.browser-header-source-plan.sha256.613b74fcf41ee5a9d4d8878af219c8e9da8ba6c91f067eb074140325feac457d`;
the seven-archive admission is
`bg.cpp.browser-header-source-archive-admission.sha256.9ef05aeebc15a47b926b289250d26c414a1eec21314bb37697ae901a768ec8f3`;
the pinned normalization environment is
`bg.cpp.pinned-archive-normalization-environment.sha256.d9461759522fbe616b0244ab63267854eb249f546a1b8560c7b7b0cd6b6df818`;
and the composed observation is
`bg.cpp.browser-header-pack-pipeline.sha256.35e0d574519cf799844dfb3f72c2b6e6ae00532224ee34ddbcab2ba3a6e03556`.
The header review input is
`bg.cpp.header-distribution-review-input.sha256.28346afa0239011e988de9cd40818b9eff6bbeeecc6e7e4bf508137697f4dc82`
with exact file SHA-256
`0c809a69554731e3b78acc8c0717c1c0939883b8df811637120a067e28be97b7`.
The notice materialization is
`bg.cpp.browser-header-notice-materialization.sha256.a560867d2614feeb1c460c2192435b5472780c5538ab8d0c11c2cf6207df430a`;
its exact aggregate SHA-256 is
`9933f791012ac5662cc87a63cdf56d794893e59d49ac8a013c05f29709b4e30b`.
The five output SHA-256 identities are `fd7fb977...` (configured Clang resource),
`f795494a...` (CUDA), `4f1c39b7...` (CUTLASS), `f66f1284...` (libc++), and
`d04a460d...` (Linux sysroot); the implementation ledger retains the complete
hashes and byte counts.

The original cold diagnostic run `29658164083` spent 97 minutes 5 seconds in
isolated execution before failing closed at link. That result exposed two
different costs that the harness had conflated: reusable LLVM/Clang
provisioning and ordinary extractor edit validation. The cached diagnostic lane
now reuses only the content-addressed toolchain directories, keeps the cache
untrusted inside the networkless/capability-free container, and never grants
clean-build, reproducibility, provenance, or release authority. Run
`29668611133` completed in 5 minutes 3 seconds. Final no-migration proof
`29668822793` at `056aaf02` then completed in 4 minutes 39 seconds: the
JavaScript boundary took 27 seconds, cache restoration took 2 seconds, the
isolated compile/link executor took 3 minutes 14 seconds, and ABI review took 3
seconds. Subsequent source-only production validations remained stable from 4
minutes 27 seconds through 5 minutes 4 seconds. Link policy and extractor
source identities no longer invalidate the expensive toolchain cache, and all
temporary cache-migration shims have been removed. The canonical local command
`pnpm --filter @unlocalhosted/browsergrad-compiler run
verify:browser-clang-wasm:fast` builds once, checks the lock without a second
clean, then runs 700 tests across 80 files covering the build plan, runtime ABI,
browser profile, browser asset identity chain, exact Worker-bundle authoring,
package invocation, Worker entry, production controller, verifier evidence,
observed layout- and view-copy-candidate preparation, independently admitted
producer trust, producer-authorized local layout and view-copy lowering,
exact header-tree
inventory/materialization, seven-archive admission, strict archive
normalization/extraction, the reviewed builder identity, CUDA-index admission,
the complete header distribution review input, distribution-notice
verification, exact notice-output materialization, two-root distribution
reproducibility, exact package admission of that evidence, browser build-subject
syntax, and signature/build-subject binding. The fast configuration now uses
family globs for all browser and C++/CuTe lowering tests instead of a drifting
per-file browser allowlist. The current Node 25 focused Vitest phase passes 700
tests across the same 80 files in 11.54 seconds. The complete compiler suite
after rank-3 convergence passes 95 files and 1,591 tests. The local
feedback loop remains measured in tens of seconds.
Clean validation and two-build
reproducibility still restore no cache and remain intentionally more expensive.

The remote build workflow no longer runs the complete JavaScript verification
suite serially at the front of every expensive matrix build. One independent
verification job now runs that suite once in parallel with all clean/cached
compiler builds; each compiler job still materializes its own exact JavaScript
runtime closure, and the final reproducibility verifier depends on both the
verification job and both clean build jobs. This removes the previously
observed 86-to-134-second test phase from the compiler critical path without
changing build inputs or granting evidence when either branch fails. The two
reproducibility builds already run concurrently, and each build remains pinned
to the four cores available on the current standard runner.
Workflow concurrency is now mode-scoped: fast feedback no longer queues behind
clean or reproducibility evidence, and a newer fast request cancels only an
older fast request. Clean validation and reproducibility never cancel one
another and retain separate isolated runners and roots.

The comprehensive compiler command now keeps its three true prerequisites
serial, then feeds the exact remaining 14 commands into four bounded,
fail-fast lanes with process-group cancellation and per-lane serialization.
The complete local command passed in 2 minutes 6.39 seconds. CI additionally
separates source/dist real-world corpus shards, the required-native harness,
Node 20/24/25 surface checks, and required WebGPU. Exact-source run `29697264202`
completed all of those lanes successfully in about 4 minutes 15 seconds,
compared with about 6 minutes 14 seconds for pre-sharding run `29695555899`.
Commands that clean or rewrite the same package build output remain mutually
exclusive; parallelism is applied only where ownership and dependency edges
are independent.

The real-world verifier now also runs its four read-only compile/codegen corpus
audits concurrently and divides the browser corpus into two bounded child
processes inside each already-built source or distribution job. Exact-source CI
`29769844668` passed both bundles with complete 159-case browser coverage split
80/79 and no failed or skipped case. The source shards took 96.57 and 96.69
seconds concurrently and the distribution shards took 96.48 and 97.10 seconds;
complete verifier time was 131.15 and 132.17 seconds rather than the sum of the
parallel stages. Shard evidence rejects missing, duplicate, or unexpected case
outcomes before the gate can pass. Runner setup, dependency installation,
workspace build, corpus ownership, and final aggregation remain intentionally
single-owner operations.

Corpus provisioning now uses target-scoped canonical ownership records for
interrupted snapshots and reservations. While holding the same target lease, a
later operation may inspect at most 32 candidates and reclaim at most four
exact self-owned residues, each bounded to 4,096 entries. Descriptor-relative
no-follow traversal, UID/root/target/process binding, exact snapshot Git-blob
identity, and conservative retention protect foreign or ambiguous state. The
remaining final leaf-unlink interval assumes cooperating same-UID writers honor
the lease; no protection from hostile same-UID leaf replacement is claimed.

Exact-source signature-binding run `29698350889` then passed the same complete
graph in about 4 minutes 12 seconds, including Node 20/24/25, required-native,
source/dist corpus, real Chromium/WebGPU, and Pyodide integration lanes.

Exact-source remote run `29695343749` proves the revised graph on Linux/Node
24. The JavaScript branch passed in 56 seconds while the cached compiler branch
continued independently; its locked executor took 3 minutes 15 seconds and the
complete build job took 4 minutes 30 seconds. The run also exposed and closed a
real harness race: a small fast-exiting normalizer could close stdout while its
strict consumer was still creating private parser output. The process harness
now attaches a bounded backpressured stream bridge immediately, awaits process,
transport, parser, and stderr settlement, and preserves strict rejection and
owned-output cleanup. Tests cover output beyond pipe capacity and malformed
tar rejection without weakening the two-zero-block requirement.

The remaining cache-free build cost is inside the LLVM/Clang dependency graph,
not JavaScript verification or workflow queueing. Successful clean run
`29681845216` spent 45 minutes 37 seconds in the locked build: about 12 seconds
in native configure, 3 minutes 26 seconds in native TableGen, 1 minute 49
seconds in Wasm configure, and 40 minutes 6 seconds in the four-way Wasm
compile/link. LLVM acquisition/extraction plus builder-image acquisition took
about 37 seconds. The two reproducibility builds already occupy separate
runners and each CMake build uses all four cores of the current standard
runner. Moving to a larger pinned runner and repinning build parallelism is the
next low-risk wall-clock lever; a pinned Ninja graph is the next build-system
experiment. Removing the one LibTooling use would prune only the small
Tooling/DependencyScanning closure; a deeper direct-`cc1` refactor reaches more
files but changes CUDA invocation semantics and is not an iteration-speed
shortcut.

The current CMake-stable primary cache is now proved at exact source.
Migration run `29680686426` completed in 4 minutes 14 seconds and populated the
new primary key. Exact-primary run `29680831101` completed in 4 minutes 17
seconds: JavaScript verification took 32 seconds, cache restore took 3 seconds,
the isolated compile/link took 2 minutes 25 seconds, and raw-Wasm review took 2
seconds. Exact cache hits skip cache staging and saving. This is diagnostic
iteration evidence only and cannot satisfy clean-build or reproducibility
requirements.

The first successful uncached clean validation, run `29674887505`, completed in
39 minutes 29 seconds rather than the earlier 97-minute failed cold diagnostic.
Its locked build step took 36 minutes 25 seconds and produced the same
31,307,826-byte exact-interface-conforming module as the reviewed diagnostic
lane. This historical clean evidence predates the current ABI 1.2 frontend-work
record. Current-source clean run `29678087663` failed after 46 minutes 25
seconds because the generated patched `ExprConstant.cpp` retained its upstream
relative include of `ByteCode/Context.h`, while the external extractor target
did not include Clang's private `clang/lib/AST` source directory. It produced
only failure observation at `$.steps[3]` and grants no output, ABI, clean,
reproducibility, or release authority. The target now attaches the canonical
private AST directory, and configured-target review requires that exact include
before compilation. Reproducibility run `29676333678` completed both clean
builds but its final comparator rejected the newly added per-build ABI-review
sidecar as undeclared. Commit `eda1ad9d` closes that harness defect by admitting,
binding, and comparing the canonical sidecar; no reproducibility authority is
claimed from the failed run.

Current-lock clean run `29681845216` at `aca7ee4e` completed successfully in 49
minutes 2 seconds. Its isolated build took 45 minutes 37 seconds and produced
the same 31,641,377-byte Wasm and 27,125-byte generated factory as the exact
warm lane. The current admission boundary accepts its exact closed artifact
tree, independently reruns the raw-Wasm inspector over the admitted bytes, and
requires its canonical 1,678,025-byte report to match byte for byte before
exposing a no-clobber factory candidate. The resulting factory SHA-256 is
`796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12`.
Two-clean-build run `29683677087` at `96ad7b16` completed both cache-free
builds. Build 1 took 48 minutes 33 seconds including a 44-minute-57-second
locked build; build 2 took 27 minutes 37 seconds including a
24-minute-51-second locked build. The original v2 comparator failed only
because the non-distributed linker maps embedded their intentionally distinct
absolute build roots; their Wasm, factory, native TableGen tools, runtime
closure, ABI-review sidecar, commands, and environments matched. Reproducibility
v3 preserves each raw map identity and compares a strict boundary-aware
projection that substitutes only the six recorded roots, rejects foreign-root
references and reserved placeholders, and still requires all semantic map
content to match. Verifier-only run `29685632925` at `c1a79c0d` independently
admitted the existing immutable artifacts in 1 minute 7 seconds. Its exact
3,470-byte evidence has SHA-256
`6c7aebe1376edf0f9a526b55bacd930e7e9e0fd454a95213d97931117548f31a`
and is package-pinned. This proves extractor-output reproducibility only, not
headers, provenance, the complete distribution, Worker execution, or release.

Every admitted build runs the independent production-scale raw-Wasm inspector
and uploads its exact report. The original 98-function import surface contained
92 generated Emscripten imports, including forbidden clock, random, process,
environment, and ambient-filesystem services. Those capabilities were closed
inside the module before review. The current observed 72-function surface is exactly six
`browsergrad_vfs_v1` functions plus 66 hash-pinned generated functions: 62
JavaScript-exception/control-flow shims, two bounded memory-growth helpers, one
stack-overflow trap, and output-only `fd_write`. The 29 worker-internal support
functions and one fixed 15,166-entry `funcref` dispatch table are separately pinned; absent
`target_features` metadata is advisory because static opcode/section inspection
remains authoritative. ABI 1.8 keeps browser-visible required features
separate from the inspector-only `bulk-memory-opt` opcode-subset marker.
Production run `29674599138` at `348d7373` completed in 4 minutes 44 seconds and
reviewed a 31,307,826-byte module with SHA-256
`b7a5daf6d121c306a2d07b5d3c14c00a664aaa2ff4ae3357a8b389326eeeb06f`, zero
ABI mismatches, raw-Wasm verification, and exact interface conformance. This
does not grant Worker or release authority: reproducibility, header-license,
valid Worker-instantiation, and production execution evidence remain separate
gates.

Exact-primary run `29680831101` produced a 31,641,378-byte module with SHA-256
`20cf9ee448af03cd91395eb098e29a9c04be741097f4fb8c7d41fc602b68fc0a`.
Its first detached comparison exposed 14 additional generated `invoke_*`
signatures and table growth from 14,549 to 15,166 entries. Review of the exact
generated factory confirmed the additions remain bounded exception-control-flow
bridges without ambient capabilities. Runtime ABI 1.8 now hash-pins the 66
generated imports and table projection; local detached review of those exact
Wasm bytes passes with zero mismatches. Exact-source run `29681607575` then
rebuilt the repinned source in 5 minutes, including 3 minutes 19 seconds for the
isolated build. Its 31,641,377-byte module has SHA-256
`5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758` and
passes raw review with zero mismatches. The ABI 1.8 local fast gate passes 30
files/338 tests in 24.7 seconds on Node 25; the later controller-complete gate
passed 34 files/381 tests in 25.26 seconds, and the current reproducibility-,
archive-, extraction-, and header-capable gate passes 50 files/437 tests in
35.32 seconds.

The harness audit found strong isolation, exact-input closure, bounded logs,
independent Wasm parsing, and separate authority tiers. It also records real
maintenance debt: the JavaScript executor sits close to its line-count ratchet,
native producer files remain large, and several semantic TypeScript modules are
5,000 to 8,000 lines. Observed uncached locked builds now range from 24 minutes
51 seconds to 45 minutes 37 seconds, while the cached link loop still spends
about 29 seconds regenerating Emscripten system
libraries. Current diagnostic cache selection excludes extractor CMake and
final-link changes are reapplied by mandatory exact configuration, every
BrowserGrad-owned object is invalidated, and generated flags are independently
reviewed before compilation. They therefore do not invalidate reusable
LLVM/Clang objects; upstream, builder, recipe compiler inputs, platform, and
selected Clang libraries still do. Local
entrypoints that clean `dist` are not safe to run concurrently in one worktree;
the canonical sequential fast command avoids that race. The expanded fast gate
also prevents runtime-ABI/profile/asset fixture drift, while the native gate
retains platform-libc signature coverage. Generated upstream source compiled
from a different directory MUST have every private relative-include root
represented in the CMake target and independently required by configured-target
review; the `clang/lib/AST` regression is the first enforced instance. These are explicit follow-on
optimization and decomposition tasks; they do not weaken the current
fail-closed boundary or turn cached diagnostics into release evidence.

Cross-runtime CI run `29674595640` confirms the source and harness on Node 20,
24, and 25. In particular, the Node 24 native gate passes after the BrowserHost
`rename` and `renameat` shims were made to inherit the selected libc exception
specification instead of assuming one platform declaration.

An active or failed run is not build, ABI, reproducibility, Worker, browser, or
release evidence. The Worker-local runtime now executes the pinned generated
factory and C ABI, verifies exact frontend-work/VFS/runtime observations, and
emits canonical control plus artifact bytes. The package-owned Worker graph is
pinned at SHA-256 `d9bd0eea4b9084eb7dd0768b35fadd0f14667b3a09dd4662a3bc052fb331c4e9`.
The production host path now consumes those bytes through the captured platform
and package invocation. It remains evidence-blocked from lowering until a valid
real Worker compile authenticates the exact terminal frame, Artifact V3,
frontend-work record, and VFS observations.
Use the linked implementation ledger
for exact chronology, failures, and evidence. This checkpoint is informational: the
remainder of this document continues to define the normative target and does
not become a mutable status dashboard.

## Purpose

BrowserGrad is a browser-native, inspectable compiler and ML execution
substrate. Multiple real frontends lower into shared, versioned semantic
contracts. A deterministic CPU reference proves meaning; portable WebGPU is
the primary accelerated execution product; optional native backends may add
hardware-specific lowerings without changing the program's declared meaning.

### Portable-product critical path

The required C++/CUDA/CuTe product path is:

```text
source -> Clang/extractor compiled to Wasm -> dedicated browser Worker
  -> verified BrowserGrad semantic artifact -> CPU reference or WGSL/WebGPU
```

The compiler/extractor runs as Wasm. BrowserGrad does not link the user's C++
into a Wasm application and does not execute a user-produced native or Wasm
binary. It extracts verified portable meaning, which the BrowserGrad semantic
backends execute.

Docker is outside this runtime graph. A pinned container MAY be used by
maintainers to build reproducible browser assets, and an optional native/AOT
CI lane MAY compare producer parity. Neither lane satisfies the browser-local
Gate 3 exit, and neither Docker nor a network compiler service may be required
to use the portable product. Implementation sequencing MUST prioritize the
browser Worker, closed VFS, real Clang-Wasm asset, semantic convergence, and
actual WebGPU evidence before optional native parity work.

Guided labs, notebooks, courses, demos, and framework-shaped APIs are demanding
consumers of this substrate. They are not permission to weaken language,
tensor, numerical, memory, or execution semantics.

A capability is not complete because an API name is accepted, a source string
parses, a shader compiles, or a demo returns a plausible value. It is complete
only at the exact tier for which representation, verification, execution,
diagnostics, and evidence all agree.

## Document Contract

This document owns:

- Stable architectural boundaries and dependency direction.
- The semantic concepts that must remain consistent across packages.
- Versioning, diagnostics, security, and operational requirements.
- The migration gates required to reach the target architecture.
- The evidence required before making a public capability claim.

This document does not act as a hand-maintained feature-status dashboard.
Current capability status MUST come from package tests and a generated
capability manifest. Assignment readiness MUST additionally come from a
versioned requirement registry plus environment-specific resolutions; a profile
label is not capability proof. Package READMEs describe the shipping public
API. Mutable project sequencing MAY live in a roadmap, but it MUST reference
the gates in this document rather than inventing a second completion
definition.

Changing a normative boundary in this document requires an architecture
decision record that states the new invariant, migration path, compatibility
impact, and deletion plan for the superseded design.

## Product Goal and Non-Goals

### Goal

BrowserGrad aims to make serious systems and ML programs inspectable and
executable in browser environments without confusing portable behavior with
CUDA- or vendor-specific behavior. It should support several source and graph
frontends through one coherent semantic stack:

- The shipping CUDA-lite frontend for CUDA-shaped browser labs and user kernels.
- A versioned CUDA-capable C++/CuTe frontend for real upstream source.
- The lazy JIT tensor graph for framework operations and autograd.
- Explicit eager-package bridges where their transfer/materialization behavior
  is part of the API.
- Direct WGSL or structured-kernel APIs for browser-native kernel authors.

CuTe/CUTLASS compatibility is a flagship proof that the architecture preserves
real layout, tensor, tiling, and hardware-boundary concepts. It is not
BrowserGrad's sole product identity and MUST NOT shape the shared semantic model
around CuTe source spellings.

### Explicit non-goals

The following are not required for the portable browser product:

- Making WebGPU expose CUDA streams, PTX, Tensor Cores, WGMMA, TMA, peer memory,
  or NCCL.
- Executing arbitrary native C++ binaries in a browser.
- Claiming schedule- or instruction-level fidelity when only observable tensor
  results are preserved.
- Supporting every CUTLASS header, architecture, example, or release under one
  undifferentiated "CUTLASS support" label.
- Turning `browsergrad-grad` into a GPU-resident eager framework through
  incremental `device=` branches. That requires a separately approved storage
  and autograd architecture.
- Making native execution or distributed multi-device execution a release gate
  for the portable WebGPU product.

## Engineering Invariants

1. **Build semantic machines, not syntax collections.** Frontends preserve real
   user programs; shared IR represents their meaning. Source-spelling handlers
   do not substitute for missing semantics.
2. **Keep the architecture frontend-neutral.** CUDA-lite, C++/CuTe, JIT, eager,
   and direct-kernel frontends converge on shared concepts without importing one
   another's internal types.
3. **Use multiple IR levels.** Source facts, value/layout meaning, effectful
   kernel behavior, schedules, host orchestration, and backend plans change for
   different reasons and MUST NOT be collapsed into one god IR.
4. **No silent semantic substitution.** `bf16` is not `f32`; a CPU reference is
   not WebGPU execution; a row-wise attention loop is not block-tiled
   FlashAttention; an opaque host callback is not a portable kernel operation.
5. **Correctness precedes acceleration, but CPU correctness is not acceleration
   evidence.** CPU reference and actual-device tests prove different things.
6. **Backend limits are data.** Feature requirements, limits, legalizations,
   numerical differences, and unavailable states are structured results rather
   than source-family guesses or free-form labels.
7. **Materialization is explicit.** Host/device transfers, readbacks, view
   materialization, dtype conversion, and host-lifted operations are observable
   operations with ownership and cost.
8. **Generated and serialized state is versioned.** Python/JS, worker/host, and
   frontend/backend boundaries use validated, deterministic schemas.
9. **Unsupported behavior fails at the true boundary.** A frontend may accept a
   program whose backend cannot execute it, but the failure MUST name the
   unsupported semantic or execution capability.
10. **Education is a quality bar.** Coordinate traces, layouts, memory effects,
    synchronization, numerical policy, and capability boundaries should be
    explainable to users, not hidden behind compatibility shims.

## Execution Tiers and Preservation Claims

Execution tier and preservation level are independent axes. A backend claim
MUST state both.

### Execution tiers

| Tier | Contract |
| --- | --- |
| **Semantic reference** | Deterministic or explicitly policy-controlled CPU execution of the verified semantic program. Used for conformance, traces, and diagnostics. |
| **Portable WebGPU core** | Actual device execution using required WebGPU core behavior only. Device limits are still explicit. |
| **Portable WebGPU enhanced** | Actual WebGPU execution that requires named optional features such as `shader-f16` or subgroups. |
| **Native companion** | Optional CUDA/native lowering for named architectures and facilities. It consumes the same semantic artifact but may use target-specific scheduling and instructions. |
| **Simulation** | An explicitly named teaching or analysis model. It is never execution evidence for another tier. |

### Preservation levels

| Level | Meaning |
| --- | --- |
| **observable-equivalent** | Values, allowed numerical error, memory effects, aliasing behavior, and synchronization-visible results satisfy the declared program contract. |
| **portable-relegalized** | Observable behavior is preserved, but invocation mapping, memory placement, tiling, or algorithm schedule is changed to meet a portable backend. |
| **schedule-preserving** | The backend preserves the declared logical tile/thread mapping and ordering constraints after a documented scope mapping. |
| **native-facility** | The backend proves use of the named architecture-specific instruction, memory path, or collective facility. |

A portable WebGPU implementation of a CUDA/CuTe program will often be
`portable-relegalized`, not `schedule-preserving`. That is a valid product
capability when named accurately.

## Multi-Level Semantic Architecture

```text
frontends
  CUDA-lite | CUDA-capable C++/CuTe | JIT graph | eager bridge | direct kernel
      |
      v
L0: frontend semantic artifact
  source spans, instantiated types/templates, source-language control flow,
  source ABI facts, target intrinsics
      |
      v
L1: value and layout semantics
  DType + DimExpr + ConstraintSet + LayoutExpr + TensorView + IndexMap
      |
      v
L2: kernel semantic IR
  structured compute, reads/writes/atomics, reductions, collectives,
  barriers, active masks, logical tiles
      |
      v
L3: schedule IR
  invocation mapping, workgroup/subgroup decomposition, staging,
  vectorization, physical tiles, backend requirements
      |
      v
L4: host execution graph
  allocations, bindings, dispatches, copies, dependencies, materialization
      |
      v
L5: backend execution plan
  CPU interpreter plan | WGSL/resources/pipelines | native launch plan
```

No layer may reach upward. Lowering is explicit and produces a new verified
artifact or a typed failure. Backends MUST NOT reconstruct missing layout,
aliasing, bounds, or dtype meaning from source names or incidental plan fields.

### Layer ownership

| Layer | Owns | Must not own |
| --- | --- | --- |
| Frontend semantic artifact | Source-language types, declarations, template arguments, spans, source ABI, and unresolved target intrinsics. | WGSL strings, `GPUBuffer`, framework tensors, or shared backend policy. |
| Value/layout semantics | Pure shape, dtype, layout, view, binding-slot, alias, and coordinate meaning. | Source AST nodes, schedules, device resources, or host workflow. |
| Kernel semantic IR | Observable computation, structured control flow, memory effects, collectives, and synchronization preconditions. | Source spellings, pipeline caches, or target instruction strings. |
| Schedule IR | Mapping logical work to physical execution, staging, vectorization, and declared backend requirements. | User-facing tensor meaning or actual device objects. |
| Host execution graph | Observable multi-dispatch order, allocations, copies, bindings, events, and materialization. | Hidden backend heuristics or course policy. |
| Backend plan | Target-legal code, resources, pipelines, dispatch sizes, caches, and recovery metadata. | New semantic meaning not present in verified upper layers. |

The architecture MAY use several concrete TypeScript data structures rather
than a general compiler framework. The required property is the layer boundary,
not adoption of a particular compiler toolkit.

## Core Value and Layout Contract

The following sketches define required information and illegal-state
boundaries. Exact TypeScript names MAY change before the first schema version
is frozen, but equivalent information MUST remain representable.

### Symbolic dimensions and constraints

```ts
type WireI64 = string; // canonical signed 64-bit base-10 integer
type WireU64 = string; // canonical unsigned 64-bit base-10 integer

interface DimSymbol {
  id: string;
  domain: { min: WireI64; max?: WireI64 };
}

type DimExpr =
  | { kind: "const"; value: WireI64 }
  | { kind: "symbol"; id: string }
  | { kind: "add"; terms: DimExpr[] }
  | { kind: "mul"; lhs: DimExpr; rhs: DimExpr }
  | { kind: "floorDiv"; value: DimExpr; divisor: DimExpr }
  | { kind: "ceilDiv"; value: DimExpr; divisor: DimExpr }
  | { kind: "mod"; value: DimExpr; divisor: DimExpr }
  | { kind: "min" | "max"; values: DimExpr[] };

type ShapeConstraint =
  | { kind: "equal"; lhs: DimExpr; rhs: DimExpr }
  | { kind: "lessEqual"; lhs: DimExpr; rhs: DimExpr }
  | { kind: "nonNegative"; value: DimExpr }
  | { kind: "positive"; value: DimExpr }
  | { kind: "divisible"; value: DimExpr; divisor: DimExpr };
```

The expression language is data, never executable source. The verifier MUST
bound expression depth and node count. Static analysis MAY recognize affine or
other decidable subsets, but an optimization subset MUST NOT become the
definition of general layout meaning.

Wire integers reject leading `+`, leading zeroes, `-0`, whitespace, exponent
notation, non-ASCII digits, and values outside their declared 64-bit range.
Evaluation uses arbitrary-precision integers in both TypeScript and Python;
field-specific verification applies signed or unsigned bounds to the result.
The evaluator checks a configurable maximum integer bit width after every
arithmetic operation and consumes an explicit arithmetic-operation budget;
node/depth limits alone are insufficient against multiplicative BigInt growth.
`add`, `min`, and `max` require at least one operand. Division, modulo, and the
`divisible` constraint require a strictly positive evaluated divisor.
`floorDiv` is mathematical floor,
`ceilDiv(x, d) = -floorDiv(-x, d)`, and modulo is Euclidean in `[0, d)`. These
rules apply equally for negative intermediate values and are wire-versioned
semantics, not host-language defaults.

Dynamic symbols have declared domains and bindings. Unresolved constraints
produce runtime guards or a lowering refusal; they MUST NOT become implicit
assumptions. Shape specialization and pipeline caches MUST include the resolved
symbol values and guard set in their cache keys.

### Dtype and numerical policy

`DType` identifies storage representation. Operations separately declare
input, compute, accumulator, and output types. The numerical policy includes:

- Rounding and conversion behavior.
- Overflow and integer wrap/saturation behavior.
- Denormal and flush-to-zero behavior where relevant.
- Whether contraction, fusion, or reassociation is allowed.
- Reduction/atomic order guarantees and determinism class.
- NaN, infinity, and signed-zero expectations.
- The comparison policy used by conformance tests.

Each storage dtype also defines bit width, alignment, byte order, and canonical
pack/unpack behavior. Serialized tensor bytes declare their byte order; canonical
BrowserGrad fixtures use little-endian encoding. A backend legalization may
change physical packing only through an explicit storage conversion.

`bf16` means IEEE bfloat16 storage and conversion semantics. Because core WGSL
does not currently expose a native `bf16` scalar type, the portable backend MAY
use packed 16-bit storage with explicit conversion/rounding, MAY widen
arithmetic under a declared policy, or MAY report unavailable. It MUST NOT
rename f32 storage as bf16. `f16` requires the WebGPU `shader-f16` feature when
native WGSL f16 operations are emitted.

The CPU reference MUST implement the declared numerical policy rather than
assuming host JavaScript or NumPy defaults automatically match WebGPU or native
reduction behavior.

Schema v1 has a closed builtin scalar storage-dtype registry: `bool`, `i8`,
`u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f16`, `bf16`, `f32`, and
`f64`. Each ID resolves to one versioned definition; unknown builtin-looking IDs
fail closed. Extension dtypes use a namespaced required extension and carry a
complete storage definition. Vector values are scalar dtype plus lane count,
not aliases such as `half2` or `float4`. V1 storage dtypes are byte-addressable
(`storageBits % 8 === 0`); sub-byte storage requires a future schema extension
and a `bit` location unit.

### Allocation, binding, and view model

```ts
type MemorySpace =
  | { kind: "host" }
  | { kind: "global" }
  | { kind: "shared"; scope: "subgroup" | "workgroup" | "cluster" }
  | { kind: "local"; scope: "invocation" }
  | { kind: "constant" }
  | { kind: "target"; targetId: string; spaceId: string };

interface AllocationSpec {
  allocationId: string;
  byteLength: DimExpr;
  memorySpace: MemorySpace;
  alignmentBytes: number;
  aliasSetId: string;
}

interface TensorView {
  viewId: string;
  allocationId: string;
  dtype: string;
  byteOffset: DimExpr;
  shape: DimExpr[];
  indexMapId: string;
  requiredAlignmentBytes: number;
}
```

Actual `GPUBuffer`, NumPy array, native pointer, or PyProxy objects do not cross
the semantic wire format. L1 `AllocationSpec` owns geometry, semantic memory
space, alignment, and alias identity only. L4 owns lifecycle and binding:

```ts
type BindingState =
  | { kind: "slot"; slotId: string }
  | { kind: "unbound" }
  | { kind: "null"; permittedByAbi: true };

interface AllocationBinding {
  allocationId: string;
  ownership: "owned" | "borrowed" | "external";
  binding: BindingState;
}
```

Views reference a canonical L1 allocation table by ID so byte length,
alignment, memory space, and alias metadata cannot diverge across duplicated
view records. Host graphs reference a separate canonical binding table by the
same allocation ID. Verified executable programs MUST NOT dereference an
allocation with an `unbound` or `null` binding. Nullability is a tagged ABI
state rather than a collection of booleans.

Allocation byte length and view byte offset are symbolic because dynamic shapes
can determine both. Before binding or execution they MUST resolve respectively
to an unsigned 64-bit value and a non-negative in-allocation offset. Access mode
does not belong to immutable view geometry; L2 memory effects declare whether a
particular operation reads, writes, or atomically accesses the view.

Backend lowering maps semantic spaces to target spaces—for example, semantic
global/shared/local storage may become WGSL storage/workgroup/private or CUDA
global/shared/register storage. The mapping and any unsupported scope are part
of the lowering decision, not `TensorView` interpretation.

Rank is data. The canonical model has no product-level rank-4 cap. A backend
may publish a narrower rank capability and legalize or reject accordingly.
Artifact decoders may enforce configurable resource budgets, including rank
limits, for denial-of-service protection; a budget result is distinct from
saying the semantic model cannot represent that rank.

### Layout and index semantics

A `LayoutExpr` is construction algebra, not a second executable truth.
Schema v1 supports affine/strided layouts, composition, permutation, slicing,
broadcasting, and padding. Normalization produces the canonical `IndexMap`
referenced by `TensorView`; executable wire artifacts MUST NOT carry an
independent layout and index map that can disagree. Static swizzles, bit shifts,
XOR/masks, and sub-byte addressing are deferred behind a required extension
that defines exact bit width and signedness.

`IndexExpr` has distinct tagged references for logical coordinate axes and for
resolved dimension symbols. Its v1 operators are integer constant, coordinate,
dimension, non-empty add, binary multiply, floor/ceil divide, Euclidean modulo,
and non-empty min/max under the same signed BigInt/budget rules as `DimExpr`.
`PredicateExpr` owns boolean literals, signed integer comparisons (`equal`,
`lessEqual`), and non-empty `and`/`or` plus unary `not`. Neither expression
domain reuses the generic `DimExpr.symbol` tag. An `IndexMap` declares coordinate
rank, an explicit location unit (`element` or `byte` in v1), a location
expression, and an in-bounds predicate. The canonical evaluator returns the
logical coordinate, element/byte location, root-allocation location, and
predicate result as a trace; it never silently clamps coordinates.

The model MUST distinguish:

- Logical coordinates from element offsets and byte offsets.
- Shape from storage extent (`size` from `cosize`-like allocation reach).
- A view transformation from a materializing copy.
- A broadcasted read-only alias from writable unique storage.
- Root-allocation bounds from view bounds.
- Layout equivalence from layout structural identity.

One canonical evaluator and verifier own coordinate meaning. CPU reference,
schedule construction, and backend lowering consume its normalized result or a
proved specialization. They MUST NOT independently reimplement offset/stride
logic.

### Memory effects, aliasing, and undefined behavior

Every effectful operation declares read, write, or atomic regions over
allocations plus ordering and synchronization requirements.

- Overlapping non-atomic writes MUST be proved disjoint, ordered, or rejected.
- Out-of-bounds access MUST be rejected or guarded according to declared source
  semantics; it MUST NOT be silently clamped for convenience.
- Alignment assumptions MUST be verified at binding or expressed as runtime
  guards.
- View aliases MUST retain a common root allocation and alias set.
- Materialization MUST create a new allocation and an explicit copy operation.
- Source-language undefined behavior MAY remain undefined only when the
  compatibility profile says so. BrowserGrad MUST NOT accidentally turn it
  into a stronger portable guarantee.

## Kernel Semantics, Scheduling, and Host Graphs

### Kernel semantic IR

Kernel IR is effectful and structured. Its core operations include typed
elementwise computation, loads/stores, reductions, gather/scatter,
convolution/im2col, copies, MMA semantics, barriers, atomics, and explicit
collectives. User-authored opaque kernels remain an escape hatch outside the
portable core.

Each operation has:

- Stable operation identifier and version.
- Operand/result types and shape constraints.
- Memory effects and alias rules.
- Numerical policy or inherited policy.
- CPU reference semantics or a precise reference-unavailable reason.
- Verification rules.
- Differentiability/VJP status where it is exposed through JIT.

Unknown required operations cause a version/capability failure. Readers MAY
round-trip unknown optional metadata, but MUST NOT execute an unknown operation.

#### Initial materializing view-copy operation

The first concrete L2 operation is a verified, materializing view copy. Its
wire contract references one verified layout artifact and names a source view,
a destination view, an exact dtype, explicit source-read and destination-write
effects, an invalid-source policy, and an overlap policy. The operation carries
its own `1.0` version inside the `browsergrad.kernel@1` envelope. Kernel v1
contains exactly one standalone operation; sequencing and dependencies belong
to L4 host graphs rather than an ambiguous operation-array order. The initial
overlap policy is `forbid`; later in-place or overlapping copies require a
separate operation or a proved traversal rule. The invalid-source policy is either
`reject` or `fill` with an exact scalar bit pattern matching the operation
dtype. Padding fill is therefore operation meaning, not L1 view geometry.

Execution iterates the destination logical shape. The source and destination
`IndexMap` records are the only coordinate-to-address truth. Broadcast means
repeated reads from the source allocation; it never grants writable broadcast
aliasing. A false source predicate must reject before memory access or take a
structured guarded fill path. A backend MUST NOT express the guard as an eager
conditional whose invalid load arm is still evaluated, and MUST NOT rely on
robust-buffer zeroing, ignored writes, or other target behavior as padding
semantics.

Generic L2 verification checks operation version, references, matching logical
shapes and dtypes, writable destination space, exact fill dtype, and declared
cross-allocation non-aliasing. Backend/profile legalization is separate. The
initial portable profile is same-dtype `f32`, rank-2/rank-3, global memory,
positive-affine indexing, and a dense injective destination. Signed strides and
integer division/modulo remain unavailable until the target integer profile is
proved. These are lowering decisions, not restrictions of the L1 or L2 model.

CPU reference and WGSL lowering consume the same verified normalized
expressions or a specialization accompanied by a differential proof against
the canonical evaluator. Preparation compiles coordinate evaluators once,
resolves bindings and guards, proves destination injectivity and every guarded
source access, and emits a specialization hash over the layout hash, kernel
hash, profile, operation, resolved bindings, shapes, offsets, and allocation
sizes. Element count, aggregate evaluation steps, prepared scratch bytes, and
preparation wall time have independent configurable hard limits. Long
preparations yield to the browser scheduler with a timer fallback and recheck
an abort signal after every yield. Runtime bindings validate native
buffer slots, exact lengths, declared alignment, and overlap; shared memory is
rejected until synchronization and ownership are explicit. Artifact hashing,
expression compilation, and coordinate proofs stay outside per-dispatch copy
loops.

The initial WGSL backend profile uses interval-proved signed `i32` arithmetic
for canonical index and predicate expressions. Positive affine maps may still
have negative intercepts outside a padding predicate, so lowering the whole
expression as `u32` would be incorrect. Conversion to a word index occurs only
inside the proved true source guard. Whole root allocations bind at offset zero
as `array<u32>` so view offsets remain semantic expressions and f32 values,
including NaN payloads, copy bit-for-bit. A backend specialization hash extends
the shared semantic-specialization hash with lowerer/profile version, generated
WGSL, workgroup schedule, the selected feature profile, and every device limit
used for legalization. Adapter identity, the full available feature inventory,
and input hashes belong to execution evidence,
not correctness cache keys.

### Logical tiles versus physical schedules

`LogicalTile` describes a collective region and its observable operand,
accumulation, reduction, mask, and synchronization semantics. `ScheduleTile`
describes a chosen physical mapping to invocations, subgroups, workgroups,
staging memory, or native atoms.

This split prevents a portable backend from pretending that a changed workgroup
mapping preserved a CUDA warp schedule. Architecture hints are namespaced,
optional schedule inputs. They are never required to interpret logical tensor
or kernel meaning.

### Uniformity and active masks

Barriers and collectives declare scope, participating invocations, memory
semantics, and active-mask preconditions. Uniformity analysis proves their
legality for the selected schedule.

A conservative proof failure is a typed lowering failure. Emitters MUST NOT
insert guessed barriers or rely on control-flow patterns that merely appear
uniform in common fixtures.

### Host execution graph

The host graph represents observable work that cannot live inside one dispatch:
allocations, binding, copies, dispatches, readback/materialization, events, and
dependencies.

Version 1 is a validated DAG. Cycles are illegal. Bounded repetition, dynamic
launch, or conditional host control require explicit versioned node kinds and
cancellation points rather than hidden emitter loops. The verifier performs
resource lifetime and read/write hazard checks before execution.

## Frontend Contracts

### Frontend-neutral lowering contract

Every frontend emits a versioned `FrontendArtifact` containing:

- Frontend identity and version.
- Source hashes, include/header hashes, flags, and target profile.
- Source spans and stable declaration/node identifiers.
- Resolved types, constants, overloads, and source ABI facts.
- Structured control flow and memory-space meaning.
- Instantiated template facts where applicable.
- Typed target intrinsics that are represented but not yet portable.
- Diagnostics produced before semantic-core lowering.

Frontend artifacts are compiler-owned and MAY be frontend-specific. Shared
semantic-core types begin only after explicit lowering. This prevents C++ AST
concerns or JIT UOp implementation details from becoming cross-package APIs.

### CUDA-capable C++/CuTe compatibility

The compatibility target is actual, version-pinned source—not a CuTe-shaped
mini-language. A standards-only C++ parser is insufficient for CUDA/CUTLASS
translation units. The frontend profile MUST pin and report:

- C++ language mode.
- CUDA language/header/toolkit compatibility level.
- Frontend/compiler build and flags.
- CuTe/CUTLASS commit or release.
- Include roots and header content hashes.
- Requested source target architecture, when present.
- Supported source features and typed unsupported-intrinsic families.

The CUDA-capable frontend owns preprocessing, lookup, overload resolution,
template instantiation, CUDA language extensions, and source diagnostics.
BrowserGrad owns lowering of resolved facts into shared semantics.

Compilation may use one of three explicit deployment modes:

| Mode | Contract |
| --- | --- |
| **Browser-local frontend** | A pinned WASM/browser compiler processes source locally within declared memory/time limits. |
| **Trusted compiler service** | A sandboxed service processes source and returns a signed or hash-addressed frontend/semantic artifact. Source-upload and retention policy are explicit. |
| **Ahead-of-time artifact** | A build tool emits a versioned artifact consumed later by the browser. The artifact retains source maps and provenance. |

Portable product flow:

```text
C++/CUDA/CuTe source -> Clang-WASM browser worker -> verified frontend artifact
  -> shared semantics -> CPU reference or WGSL/WebGPU execution
```

Optional parity flow:

```text
native/Docker CI producer -> same verified frontend artifact -> parity policy
```

Docker appears only in the optional native/AOT producer flow and in the
reproducible build that produces pinned browser assets. It is never part of the
portable browser execution graph. Shipping or using the browser-local profile
MUST NOT require a Docker daemon, container runtime, native compiler
installation, or compiler service. The browser executes the verified Worker
and Clang-WASM assets; Docker may only produce or independently compare those
content-addressed assets and artifacts.

For the portable BrowserGrad product, the browser-local frontend is the
primary Gate 3 path. A normal user MUST be able to compile a supported,
profile-pinned C++/CuTe source unit in the browser without Docker, a native
toolchain installation, or a network compiler service. The browser-local
profile MUST run the pinned CUDA-capable Clang frontend as WASM in a dedicated
worker, use the same closed virtual-filesystem and artifact contracts as every
other producer, and enforce explicit memory, work, output, and cancellation
limits. Clang-WASM performs source analysis and emits the frontend artifact; it
does not execute a user-produced native binary. In this compilation path, the
compiler/extractor module—not the user C++ program—runs as WASM. BrowserGrad
does not link or execute user C++ as a WASM program; verified
portable semantics execute only through the CPU reference or WGSL/WebGPU
backend.

Trusted-service and AOT producers remain valid optional protocol modes. A
native or Docker-hosted producer MAY serve CI, corpus qualification,
cross-implementation comparison, release-time precompilation, or deployments
that explicitly select that profile. It MUST NOT be a browser runtime
dependency, a prerequisite for the portable product, or an alternate semantic
authority. When both browser-local and native producers support a profile,
their canonical frontend facts, diagnostics, selected entries, and lowered
semantic identities MUST be compared under one versioned parity policy;
producer-specific provenance and resource observations remain distinct.

"Browser-native" always promises browser execution for the portable backend.
It promises browser-local compilation only for profiles that explicitly name
the browser-local frontend mode.

A content hash proves artifact identity and integrity relative to expected
content; it does not establish producer trust. Profiles that require trusted
provenance MUST verify a signature or an equivalent allowlisted attestation.
The trust root and signer policy MUST be admitted independently of the
manifest or statement being authenticated; a manifest-selected key can prove
only cryptographic consistency, never producer trust by itself.

The compiler service or WASM frontend MUST treat input as untrusted. It MUST
use an allowlisted virtual filesystem, bounded preprocessing/template work,
bounded output, cancellation, and no execution or linking of user-produced
native binaries during semantic extraction.

#### Browser-local producer architecture

The browser producer is a BrowserGrad-owned semantic extractor built from a
version-pinned LLVM/Clang source revision and cross-compiled with a
version-pinned Emscripten toolchain. It SHOULD link only the Clang/LLVM
libraries and Emscripten runtime support needed for preprocessing, parsing,
Sema, instantiated AST/constant evaluation,
diagnostics, and BrowserGrad artifact emission. It MUST NOT require the Clang
driver as a subprocess, LLD, native code generation, PTX assembly, a CUDA
driver, or execution of user-produced code. A reproducible build MAY run in a
pinned container; that is a maintainer/build-time input, never a browser or
end-user runtime dependency.

The implementation uses a custom `FrontendAction`/AST consumer and a custom
`llvm::vfs::FileSystem` with no physical/ambient fallback. The browser-local
profile serves verified file ranges lazily from host-owned packs into
extractor-owned regions of the Worker's WASM memory; it does not eagerly
duplicate the complete mounted file universe in linear memory. Opening a file
creates a logical full-file reservation, not evidence that those bytes are
resident in WASM. Actual WASM residency comes from memory/allocator
instrumentation and MUST remain a separate observation. One Worker-execution
session owns aggregate live-open metering, transfer/disposal, and cancellation.
It does not depend on an
unofficial prebuilt browser Clang distribution. Release evidence therefore
owns the LLVM revision, Emscripten
revision, native TableGen tools, builder image, patches, build flags, source
epoch, license inventory, and reproducibility proof. The selected profile owns
an exact file-level license/notice allowlist for every distributed asset. It
MUST NOT package an entire CUDA toolkit include tree merely because it is
installed in the builder; redistribution rights and required notices are
reviewed for the exact pinned header files. Two clean builds in distinct paths
MUST produce identical asset hashes before a compiler profile is released.

Two sysroots remain separate:

- The Emscripten build sysroot compiles the extractor itself to WASM.
- The parsed-program virtual sysroot defines the source program's Clang
  resource headers, C++ library headers, CUDA compatibility headers,
  CuTe/CUTLASS/CCCL headers, include order, macros, target triple, CUDA pass,
  and target architecture.

Parsed programs MUST NOT inherit `wasm32-emscripten`, `__EMSCRIPTEN__`, the
worker's ambient filesystem, locale, clock, time zone, current directory, or
network. CUDA profiles MUST model host and device semantic passes explicitly;
one accidental host-only parse is not CUDA compatibility. Compatibility claims
name the pinned Clang-CUDA dialect and header profile rather than claiming
undifferentiated NVCC compatibility.

Virtual paths and file modification times are canonical. The compilation
contract either rejects `__DATE__`, `__TIME__`, and `__TIMESTAMP__` or pins
their exact expansions. `__FILE__` expands only from the canonical virtual
path; no builder, cache, Blob, or browser path may enter source semantics.

The source ABI and compiler runtime ABI are distinct contracts. The source ABI
describes the program being analyzed. The compiler runtime ABI separately pins
WASM width/features, the complete import/export interface, memory-sharing mode
and ownership, initial and maximum pages, stack ceiling, VFS storage model,
input/result framing, lifecycle, and worker/module compatibility. Version 1
requires one unshared memory. Source target pointer width MUST NOT be used as
evidence about the Clang-WASM runtime.

The canonical runtime-ABI manifest is design authority, not evidence that a
particular `.wasm` file conforms. Release requires a separate bounded raw-WASM
inspection authority that compares the complete observed module against the
independently reviewed manifest. The observed module MUST NOT extend its own
allowlist. In particular, every Emscripten-generated import, support export,
table/global projection, memory limit, tag/start policy, and allowed custom
section MUST be closed and reviewed independently before release. The
`target_features` section is metadata to cross-check against decoded structure
and opcodes; it is not browser capability evidence. Browser reflection APIs
that expose only import/export names and kinds are insufficient for this gate
because function signatures, memory limits, and structural feature use remain
unproved. The manifest MUST name one instruction-set baseline, the exact
allowed extensions, and forbid every unlisted extension.

Raw inspection MUST distinguish WebAssembly validity, exact byte identity, and
BrowserGrad ABI conformance. It MUST accept every encoding permitted by the
selected WebAssembly binary-format version, including padded LEB128 integers
within the format's width limit, while rejecting overflow, invalid unused
terminal bits, malformed section boundaries, and invalid indices or function
bodies. It MUST NOT silently normalize or re-encode the module; the asset hash
continues to bind the exact distributed bytes. After bounded structural and
opcode inspection, standards-compliant module validation MUST succeed before
exact ABI conformance authority can be minted. Generic WebAssembly names are
validated UTF-8; the independent ABI allowlists, not a parser-wide ASCII
restriction, decide which import, export, and custom-section names are allowed.
Because engine validation and hashing are not synchronously preemptible, an
untrusted or large compiler module MUST be inspected in a disposable verifier
Worker. Host cancellation terminates and replaces that Worker; an
`AbortSignal` checked inside a synchronous parser is not claimed as hard
preemption or main-thread responsiveness.

The `target_features` parser follows the WebAssembly tool-conventions wire
vocabulary. Feature names MUST be unique but are not assumed to be sorted.
Their observed order is retained in the raw projection and compared with the
independently reviewed manifest. Recognized-but-unlisted proposals are rejected
by the closed extension policy; they are not misclassified as malformed
metadata or as an allowed parent feature.

Every synchronous VFS import MUST use checked non-wrapping wasm32 range
arithmetic, validate all input and output ranges before the first access,
snapshot path bytes before any overlapping output write, reject overlapping
output ranges, enforce the declared record alignment, forbid memory growth
during the call, and leave memory unchanged on invalid ranges. The opened-file
linear-memory ceiling is aggregate across all live session materializations,
not a per-file allowance. Linear-memory coexistence arithmetic includes stack,
compiler working bytes, aggregate opened VFS bytes, the live input frame, and
the maximum result bytes.

The reusable semantic compilation contract has its own schema and version,
independent from the full frontend-profile wire version. Deployment-only
profile changes such as Worker packaging, container policy, or browser asset
delivery MUST NOT invalidate semantic compilation identities or caches when
language, target, compiler resources, dependencies, VFS semantics, adapter,
and semantic limits are unchanged.

The package emits one self-contained module-worker resource without static or
dynamic imports. Because module workers have no subresource-integrity option,
the host MUST verify the exact package-owned worker bytes and length before
creating a Blob-backed module worker. A worker handshake cannot attest its own
code identity. Deployments MUST declare the required CSP/browser capability;
if verified Blob workers are unavailable, compilation fails with a typed
capability result and MUST NOT fall back to an unverified URL worker.
The host also verifies the separate Clang/extractor WASM bytes, then transfers
those bytes or a compiled `WebAssembly.Module` into the worker. The bundled
Emscripten factory MUST accept that verified object and MUST NOT resolve or
fetch a `.wasm` file relative to the Blob module URL. The worker has no network
authority: same-origin, redirect-free asset acquisition and content-addressed
cache admission belong to the host before verified authorities are transferred
into the worker.

Decoding a caller-supplied result frame proves only that the frame and artifact
are internally consistent. It MUST NOT mint Worker-execution or lowering
authority. Browser execution evidence requires a host-owned controller that
creates the exact verified Worker, binds a fresh package-generated invocation
nonce, accepts one terminal message only from that Worker instance, measures
host time, and owns terminate-and-replace cleanup. The pure result verifier is
wrapped by that controller; it is not a public self-attestation path.
Immediately before minting execution evidence, the controller MUST re-unwrap
the exact protocol-issued validated frame and cross-bind its invocation,
request, profile, request-binding, and artifact identities. Re-decoded or
structurally copied frames MUST NOT substitute for the opaque validator-issued
authority.

#### Closed browser VFS assets

Compiler-resource and dependency-header packs use the closed media type
`application/vnd.browsergrad.vfs-pack.v1`, not tar, zip, or a general archive.
Version 1 is identity-encoded and contains:

```text
96-byte header:
  magic "BGVFSPK1"
  u16le major, u16le minor, u32le fileCount
  u64le indexByteLength, u64le fileDataByteLength
  32-byte indexSha256, 32-byte contentSetSha256

repeated canonical index entries:
  u16le pathByteLength, UTF-8 path bytes
  u64le fileByteLength, 32-byte fileSha256

file bytes concatenated in index order
```

In the outer asset manifest, identity encoding means `byteLength` equals
`unpackedByteLength` and covers the complete header + index + data bytes.
Pack-specific `fileContentByteLength` separately covers only the concatenated
file-data region. Per-asset and aggregate ceilings exist for both categories;
one field never changes meaning by asset kind.

Entries are regular files only; directories are implicit. Version-1 paths are
strict UTF-8/NFC relative POSIX paths whose segments match the portable ASCII
allowlist `[A-Za-z0-9._+@=-]+`; entries are sorted uniquely by encoded bytes.
Absolute paths, empty/dot/parent segments, backslashes, control characters,
normalization aliases, duplicate paths, and file/directory collisions are
invalid. The format has no links, sparse files, devices, permissions, owners,
timestamps, xattrs, PAX/GNU extensions, or caller-selected offsets. Index
length, file count, per-path bytes, per-file bytes, total file bytes, and total
pack bytes are independently bounded. Entry lengths consume the data region
exactly; truncation and trailing bytes are invalid. The content-set hash covers
only canonical path, content hash, and file length, while the outer asset hash
covers the complete pack bytes.

Source-archive extraction MUST preserve this virtual-path identity without
depending on host filesystem case folding or Unicode normalization. A portable
extractor MUST use collision-free storage keyed independently from the display
path, or prove that its storage medium preserves every admitted virtual path
byte-for-byte. Extracting a Linux header universe directly into a default
case-insensitive host directory is non-conforming even when the resulting tree
appears complete.

All header integers are parsed with checked unsigned-64-bit arithmetic and
compared with profile and actual-buffer limits before conversion to JavaScript
`Number`, slicing, copying, or allocation. `contentSetSha256` is SHA-256 over
canonical JSON with the exact domain
`browsergrad.compiler.cpp-cute.browser-vfs-content-set.v1` and this projection:

```json
{
  "domain": "browsergrad.compiler.cpp-cute.browser-vfs-content-set.v1",
  "files": [
    {
      "virtualPath": "sorted/path.h",
      "contentSha256": "lowercase hex",
      "byteLength": "canonical WireU64 decimal"
    }
  ]
}
```

The `files` array follows index order. Offsets, pack framing, and index bytes
are absent from this semantic projection and remain covered by the index and
outer pack hashes.

Compression is intentionally absent from v1. A future compressed transport
requires a separately versioned, pinned streaming decoder with an enforced
output ceiling before the closed pack verifier runs. It MUST NOT widen the VFS
entry model or import general archive semantics.

Asset acquisition, cache admission, VFS verification, mount construction, and
worker execution mint separate opaque authorities. Cache hits are rehashed
before use. Mount construction rejects cross-pack path collisions after adding
the profile-owned virtual root. The final mounted absolute UTF-8 path MUST fit
the runtime ABI path ceiling; validating only the relative pack path is
insufficient. Asset fetch/pack-verification time is excluded
from compiler execution time, but resident bytes are never excluded from
memory ceilings. The runtime profile names one storage model:

- An eager in-WASM VFS requires mounted VFS + source + AST/template state +
  output + allocator/stack headroom to fit the maximum WASM memory.
- A host-backed lazy VFS separately meters retained JavaScript pack bytes,
  logical full-file reservations for live handles, actual WASM-owned memory,
  and result bytes. A range copied into an extractor-owned buffer is accounted
  by actual WASM memory instrumentation; it is not reclassified as a
  permanently resident full-file copy.

The browser profile and runtime ABI also pin `maxIndexedNodes` and
`maxIndexLogicalByteLength` for the Worker-side file/directory index. Logical
index bytes are the exact sum, for every indexed file or directory, of the
32-byte ABI metadata record, canonical absolute-path UTF-8 bytes, and immediate
basename UTF-8 bytes; the root basename contributes zero. Session preparation
MUST enforce both ceilings incrementally before admitting another node and
MUST fail before Worker execution when either is exceeded. These counters are
deterministic logical budgets, not measured JavaScript heap usage. A release
profile MUST choose narrower values from the measured final pack inventory and
browser peak-memory qualification; the runtime-ABI maxima are not deployment
defaults.

For the browser-local profile, `maxMemoryBytes` is the enforced WASM
linear-memory ceiling. It retains one meaning across the worker lifecycle and
is not reused for host pack/cache bytes. Stack, compiler working allocation,
actual VFS read destinations, and maximum output reservations MUST fit inside
that ceiling, which in turn MUST fit the runtime ABI's maximum page count. The
separate logical live-open ceiling limits handle amplification but MUST NOT be
reported as WASM residency. The stack reservation MUST fit initial pages.
Host-retained verified pack bytes have a
separate ceiling; evidence reports host-retained and WASM-owned categories
separately instead of inventing a JavaScript heap peak.

#### Producer-neutral requests and browser evidence

The common frontend request binds the producer-neutral compilation-contract
hash, exact source descriptors and snapshots, main virtual path, source anchor,
and expected artifact schema/version. It does not contain a deployment-profile
hash, worker/container/assets, compiler arguments, environment, host paths,
URLs, repository/revision, or pre-known output/header/input-closure hashes.
Repository/revision, when supplied, is a detached caller-declared source
reference. It is signed metadata, not source-acquisition provenance. Only a
separate authority that binds repository, revision, paths, and digests to the
exact request bytes may claim that provenance. Expected hashes are detached
conformance assertions, not ordinary compile-request fields.

Request JSON contains content-addressed source descriptors, not executable
host paths or embedded mutable byte views. Exact source snapshots cross the API
as unshared bytes, are checked against declared length before allocation,
copied before asynchronous work, and retained behind opaque request authority.
Request-owned ceilings may narrow only semantic/input/output ceilings from the
compilation contract. Wall time, CPU time, producer memory, process count, and
other deployment limits belong to the later invocation/evidence authority and
do not fragment the common request identity. Version 1 admits both entry
families required by this gate: a layout variable or a view-copy function,
selected by a bounded source-token anchor that the real frontend must resolve.

The profile pins the complete available virtual header universe. A produced
artifact and worker evidence separately record a canonical unique set of files
whose content was successfully read and contributed to preprocessing or Sema
for one request. Failed lookup/stat probes and duplicate opens are excluded.
The actual opened-header hash varies legitimately by source
and MUST NOT be compared to one profile-wide expected-opened-header constant.

Browser authority follows this fail-closed chain:

```text
prepared browser profile -> prepared asset manifest -> verified asset bytes
  -> verified VFS installation -> prepared common request/source snapshots
  -> prepared browser invocation -> strict one-shot worker result
  -> host-verified browser evidence -> common lowering authorization
```

The worker result cannot authorize itself. The host verifies exact canonical
control and artifact bytes, reconstructs trusted profile/asset/source facts
from opaque inputs, and treats malformed, duplicate, late, crashed, timed-out,
or cancelled results as terminal failures without lowering authority. Hard
timeout or cancellation terminates and replaces the dedicated worker. Browser
evidence records host-observable wall time, WASM memory pages, instrumented
frontend counters, opened inputs, diagnostics, and output bytes; it MUST NOT
synthesize OS process metrics, browser CPU time, or JavaScript heap peaks.

### CuTe/CUTLASS requirements

- **CUTE-001 — Versioned compatibility profile.** Unmodified fixtures are pinned
  by source and dependency hashes. "CuTe support" without a profile is invalid.
- **CUTE-002 — Real source semantics.** No source-spelling handler substitutes
  for C++ lookup, templates, overloads, or CuTe layout evaluation.
- **CUTE-003 — Layout algebra.** Shape/stride hierarchy, composition, slicing,
  products/divides, coordinate mapping, size/cosize, and supported swizzles
  lower to the general layout/index model.
- **CUTE-004 — Dynamic tensor views.** Rank-2 and rank-3 transpose, strided
  slice, broadcast, packed-head, padded, NCHW, and NHWC layouts use the runtime
  view ABI rather than source-generated special cases.
- **CUTE-005 — Tensor objects.** `Tensor<Engine, Layout>`, pointer arrays,
  rebinding, indirect tensors, nullable boundary views, and device helper
  arguments lower through explicit binding and alias rules.
- **CUTE-006 — Target intrinsics.** Architecture-specific copy/MMA/pipeline
  facilities remain typed operations or capability requirements; they do not
  degrade into opaque strings or parser rejection when their surrounding
  program is otherwise representable.
- **CUTE-007 — Tiered outcome.** Frontend acceptance, semantic lowering,
  reference execution, portable lowering, schedule preservation, and native
  facility evidence are reported separately.

### Conformance workload ladder

The first C++/CuTe proof is deliberately smaller than tiled attention:

1. Unmodified pinned layout-only fixtures produce expected coordinate traces.
2. `Tensor<Engine, Layout>` views bind dynamic rank-2/3 storage and execute copy,
   transpose, strided slice, and broadcast through CPU reference.
3. The same view operations execute on real WebGPU.
4. Tiled GEMM proves logical tile, schedule, mask, staging, and boundary behavior.
5. Tiled attention proves online softmax across K/V tiles and is compared with
   the existing fused row-wise baseline.

Tiled attention is the flagship milestone, not the first tracer bullet.

### Cross-cutting platform workload suite

The architecture is not accepted on CuTe fixtures alone. Shared capability is
also driven by:

1. General dynamic view/layout transformations from CUDA-lite and JIT frontends.
2. Tiled GEMM and convolution/im2col with irregular dimensions and layouts.
3. Reductions, normalization, optimizer updates, and explicit materialization.
4. A teaching-scale transformer forward/backward/update step that audits
   residency and hidden readbacks.
5. Multi-kernel host graphs with allocation lifetime and dependency hazards.
6. Browser-worker collectives whose transport/failure contract remains distinct
   from native distributed execution.

Each workload adds reusable semantic and backend capability. None may add an
assignment-local execution path to make the workload pass.

## Backend Contracts

### CPU semantic reference

The reference backend consumes verified L1/L2/L4 semantics. It MUST share the
same coordinate evaluator, dtype definitions, memory-effect rules, and host
graph ordering contract as other backends.

It SHOULD optimize enough to keep conformance practical, but reference
optimizations MUST preserve traceability. A coordinate/effect trace can identify
which semantic node produced a value or failure.

### Portable WebGPU

The WebGPU backend lowers verified kernel semantics plus a selected schedule to
WGSL, resource bindings, pipelines, and dispatches. It MUST:

- Negotiate features and limits on the created `GPUDevice`, not infer them from
  browser names or adapter presence.
- Validate workgroup size, workgroup memory, binding count, binding alignment,
  buffer range, dispatch dimensions, and storage limits before submission.
- Distinguish WebGPU core from optional-feature profiles.
- Treat shader creation, pipeline creation, validation, out-of-memory, and
  device-loss failures as distinct diagnostic codes.
- Keep readback explicit and asynchronous.
- Invalidate all device-owned resources and caches on device loss; recovery
  requires a new device and resource recreation.
- Include semantic-program hash, schedule hash, backend version, feature set,
  relevant limits, and numerical policy in pipeline cache keys.
- Treat prepared backend plans as authority-bound immutable values. Runtime
  execution MUST reject caller-forged or mutated plans whose WGSL, bindings,
  schedule, or hashes could diverge.
- Bound owned host/GPU working bytes and in-flight work independently from
  hardware maximum limits. The initial view-copy profile permits at most one
  in-flight operation per `GPUDevice`; a timed-out or aborted submission keeps
  that slot until its device work and owned-resource cleanup settle.
- Keep WebGPU error scopes around the synchronous creation/submission window,
  pop every scope in LIFO order before awaiting readback, and distinguish
  shader, pipeline, validation, out-of-memory, internal, device-loss, timeout,
  cancellation, and execution diagnostics. Device-loss watchers invalidate
  every participating owner cache even when another rejection wins a race.
- Provide a required-device conformance mode in which missing adapter/device,
  validation errors, out-of-memory, device loss, or skipped cases are failures.
  Advisory browser tests MAY record a real not-run result, but MUST NOT report
  adapter absence as a passing execution test.

The current direct attention implementation remains a fused row-wise
online-softmax baseline. It may be named block-tiled FlashAttention only after
the implementation stages K/V tiles or proves an equivalent declared memory
strategy, maps query tiles, satisfies uniformity/barrier rules, maintains online
softmax across tiles, and passes boundary and real-device conformance.

### Native companion

The native companion is optional and separately deployable. It consumes a
versioned semantic artifact and MAY lower typed intrinsics to CUDA facilities.
It MUST report compiler/toolkit versions, target architecture, facility use,
and preservation level. It MUST NOT redefine layout/index/numerical meaning to
match a convenient native implementation.

Browser worker meshes and native multi-device execution are separate products.
They may share collective semantics, but transport, failure, topology, and
performance claims remain distinct.

## Capability, Lowering, and Evidence Model

Capability, a lowering decision, and execution evidence are different facts.
They MUST NOT be flattened into five booleans or one `supported` label.

```ts
type PreservationLevel =
  | "observable-equivalent"
  | "portable-relegalized"
  | "schedule-preserving"
  | "native-facility";

type ExecutionTier =
  | "semantic-reference"
  | "webgpu-core"
  | "webgpu-enhanced"
  | "native-companion"
  | "simulation";

interface SemanticCapabilityDefinition {
  capabilityId: string;
  semanticVersion: string;
  operationVersion?: string;
  preservationLevels: PreservationLevel[];
}

type SupportState =
  | "supported"
  | "conditional"
  | "unsupported"
  | "unknown"
  | "not-applicable";

interface LoweringDecision {
  capabilityId: string;
  backendId: string;
  executionTier: ExecutionTier;
  state: SupportState;
  preservationLevel?: PreservationLevel;
  requiredFeatures: string[];
  requiredLimits: Record<string, number>;
  runtimeGuardIds: string[];
  legalizationIds: string[];
  numericalPolicyId?: string;
  reasonCode?: string;
}

type EvidenceOutcome = "not-run" | "passed" | "failed";

interface ExecutionEvidence {
  capabilityId: string;
  artifactHash: string;
  backendId: string;
  environmentId: string;
  producerVersions: Record<string, string>;
  deviceProfileHash?: string;
  recordedAt: string;
  outcome: EvidenceOutcome;
  comparisonPolicyId?: string;
  diagnosticCodes: string[];
}
```

A conformance run emits exactly one authoritative terminal
`ExecutionEvidence`. Per-case observations are non-terminal and MUST NOT say
`passed` while later cases, queue drainage, or uncaptured-error checks remain.
The terminal artifact hash binds the prepared case set; reproducibility also
records ordered case IDs, input hashes, module/specialization hashes, logical
invocation counts, submitted workgroup counts, pipeline count, and the current
stage/case on failure. Adapter-supported features and negotiated device
features are distinct facts. Producer package versions come from package
metadata rather than duplicated source constants.

Required release lanes retain the complete terminal-evidence log under the
tested commit. `passed` is legal only after the full ordered matrix completes,
the queue and late-error window drain, no uncaptured error remains, and a
runtime validator accepts the record. Missing device evidence is `failed` in a
required lane and `not-run` in an advisory lane.

Static capability does not contain `passed` or `not-run`. Test evidence does not
decide whether the current user's device supports a feature. Runtime scheduling
uses a fresh lowering decision derived from the actual program, device profile,
and declared policy.

`environmentId` identifies an immutable environment record containing the
observable producer, runtime, browser, feature, and limit facts used by the
test. Unknown support fails closed for execution. Conditional support proceeds
only after every referenced runtime guard is evaluated successfully.

Capability records MUST be monotonic in meaning: adding evidence may strengthen
confidence, but it cannot silently redefine what a feature identifier means.

Assignment prerequisites are a separate model. A device feature, local oracle,
simulator, fixture, external service, or policy gate is not automatically a
semantic capability and does not receive a `LoweringDecision` merely because a
profile requires it.

```ts
type RequirementKind =
  | "semantic-feature"
  | "runtime-facility"
  | "device-feature"
  | "oracle"
  | "simulator"
  | "fixture"
  | "external-service"
  | "policy";

interface RequirementDefinition {
  requirementId: string;
  semanticVersion: string;
  kind: RequirementKind;
  owner: string;
  capabilityId?: string; // semantic-feature only; never inferred from spelling
}

type RequirementResolution =
  | {
      requirementId: string;
      environmentId: string;
      availability: "available";
      route: "browser" | "simulated" | "external";
      providerId: string;
    }
  | {
      requirementId: string;
      environmentId: string;
      availability: "unavailable";
      reasonCode: string;
    }
  | {
      requirementId: string;
      environmentId: string;
      availability: "unknown";
      reasonCode?: string;
    };
```

Assignment readiness consumes `RequirementResolution` records. Only a
`semantic-feature` requirement with an explicit `capabilityId` may point to a
semantic capability, and that link does not replace the program-specific
lowering decision. Existing assignment strings remain versioned legacy routing
requirement IDs until migrated; they are not canonical backend IDs, execution
tiers, support states, or evidence outcomes.

Environment-scoped resolutions describe reusable providers and device facts.
A source subset, legalization result, runtime guard, or executable-plan result
is artifact/program-scoped and MUST be keyed by artifact hash or program ID; it
must not be promoted into a reusable environment merely because one program
lowered successfully.

Canonical semantic capability, backend, diagnostic, and requirement IDs use
lowercase dot-separated namespaces and identify one immutable predicate.
Versions, support outcomes, execution tiers, device names, and words such as
`passed` or `supported` stay in separate fields. Legacy hyphenated assignment
IDs are registered exactly as compatibility identifiers and are not a naming
precedent for new protocols.

## Wire Format and Schema Evolution

Create an initially private workspace package
`@unlocalhosted/browsergrad-semantic-core`. Its target subpaths are:

- `/layout` — value, dtype, shape, constraints, layouts, views, and index maps.
- `/kernel` — kernel semantic operations, effects, collectives, and verifier
  interfaces.
- `/schedule` — schedule representation and verification, not schedule-selection
  policy.
- `/host` — host-graph representation, resource hazards, and verification.
- `/schema` — wire envelopes, canonical serialization, and validators.
- `/diagnostic` — universal diagnostic stages and structured diagnostic schema.
- `/capability` — capability, lowering-decision, and evidence schemas only.
- `/requirement` — assignment requirement definitions and resolution records;
  no semantic lowering policy.

Gate 1 exports only `/schema`, `/layout`, and `/package.json`. `/kernel`,
`/schedule`, `/host`, `/diagnostic`, `/capability`, and `/requirement` are added
only when concrete contracts and real consumers exist; empty placeholder
barrels are prohibited.

The package is justified by live compiler, kernels, JIT bridge, and runtime
protocol consumers. `private` describes only its initial standalone Gate 1
incubation state; it is not a permitted unpublished runtime dependency of a
published, unbundled package. Before compiler, kernels, JIT, or runtime ships a
runtime import from semantic-core, semantic-core MUST either become a packed,
release-tested `0.x` package or the consumer MUST have an explicitly verified
bundling strategy that preserves one schema implementation. BrowserGrad's
current unbundled `tsc` packages therefore require the packed-package path.
Internal API status is communicated by `0.x` versioning, narrow subpaths, and
release policy—not by leaving downstream installations with an unavailable
dependency. Promotion to `1.0` still requires one major schema version to
survive two frontend and two backend integrations. Runtime may import only
`/diagnostic`, `/capability`, and `/requirement`; architecture checks MUST
prevent it from importing tensor evaluation or kernel lowering code.

### Envelope and version rules

Every serialized artifact begins with an envelope equivalent to:

```ts
interface WireEnvelope<T> {
  schema: string;
  version: { major: number; minor: number };
  producer: { id: string; version: string };
  artifactId: string;
  payload: T;
  requiredExtensions: string[];
  optionalMetadata?: JsonObject;
}
```

`JsonObject` and `JsonValue` are closed JSON-safe recursive types. `unknown`,
`undefined`, functions, symbols, `bigint`, sparse arrays, cycles, accessors,
class instances, and non-finite JSON numbers are not canonicalizable values.

- Unknown major versions are rejected.
- Minor versions may add optional fields without changing existing meaning,
  but only inside declared open metadata/extension bags whose unknown JSON
  fields are preserved losslessly for canonical re-encoding and hashing.
  Closed semantic records reject unknown fields; adding a semantic field to a
  closed record requires a new major version or required extension.
- Unknown required extensions are rejected; unknown optional metadata may be
  preserved or ignored but never executed.
- Deprecated fields have a producer window, reader window, migration fixture,
  and removal major version.
- Stable node IDs are deterministic within canonical serialization; random IDs
  do not participate in hashes or cache keys.
- Signed/unsigned integers that may exceed JavaScript's safe integer range are
  canonical decimal strings on the wire.
- Floating constants that require bit identity are encoded as validated bit
  patterns, not JSON numbers. NaN payloads, infinities, and negative zero are
  therefore not lost through JSON normalization.
- Hashing uses SHA-256 over UTF-8 canonical JSON bytes and lowercase hexadecimal
  digests. Canonical JSON follows RFC 8785 object-key ordering and JSON string
  escaping. BrowserGrad semantic wire payloads permit JSON numbers only for
  safe integers; lexical `-0`, decimal, and exponent JSON-number forms are
  rejected even when their parsed numerical value is an integer. Lone UTF-16
  surrogates are rejected. Python canonicalization implements RFC 8785 UTF-16
  property ordering explicitly. Semantic floating-point values use bit-pattern
  records. This deliberately removes cross-language float-to-decimal drift.
- Hash inputs include schema version and semantic content. Compiler flags,
  frontend/header revisions, target profile, schedule, and backend policy are
  separate named hash components rather than undocumented cache salt.

The semantic artifact hash projects exactly `domain`, `schema`, `version`,
sorted unique `requiredExtensions`, and the normalized semantic `payload`, with
`domain` fixed to `browsergrad.semantic-artifact.v1`. It excludes
`artifactId`, producer, optional metadata, evidence, timestamps, and random or
transport IDs. `artifactId` MAY be the domain-separated semantic digest but
never participates in its own digest. Composite cache keys hash canonical JSON
of `{ domain: "browsergrad.cache-key.v1", components: { ... } }`; component
names are mandatory and sorted canonically. Pure immutable value nodes MAY use
the full domain-separated SHA-256 digest of normalized ID-free content.
Entity IDs—allocations, views, operations, and other records where two
structurally identical instances remain distinct—MUST NOT be content-addressed
by content alone. They derive deterministically from artifact scope plus entity
kind and canonical declaration/path/ordinal, or from a frontend-stable scoped
ID. Full digests are used when hashing IDs; truncated digests are not canonical
IDs.

Decoders operate in the order byte-budget check, duplicate-key-aware JSON parse,
envelope/schema/extension/version validation, raw structural and reference
validation, normalization plus deterministic ID remapping, normalized semantic
verification, deep freeze, and opaque verified-wrapper construction. The
wrapper holds the already-frozen artifact through a private constructor and is
itself immutable; code never mutates a frozen object to add a brand. Only opaque,
deeply immutable verified artifacts may enter evaluators or hashes used for
execution. Unknown fields in declared open bags are preserved as JSON or
discarded only where the schema explicitly excludes them from semantic hashes;
they are never interpreted by an old reader.

Every decoder validates schema, nesting depth, node count, string/array sizes,
identifier uniqueness, reference integrity, integer bounds, and extension
requirements before allocating backend resources.

Cross-language golden fixtures MUST prove TypeScript and Python encode, decode,
validate, canonicalize, and hash the same artifacts.

## Package Ownership and Dependency Direction

| Package | Owns | Must stop owning or inferring |
| --- | --- | --- |
| `browsergrad-semantic-core` *(private during standalone incubation; packed `0.x` before public-package adoption)* | Pure shared value/layout/kernel/schedule/host schemas, verifiers, canonical wire format, coordinate evaluation, and diagnostic/capability/requirement protocol subpaths. | Source parsing, framework APIs, actual device resources, schedule-selection policy, course policy. |
| `browsergrad-compiler` | CUDA-lite and C++/CuTe frontends, frontend artifacts, semantic lowering, compiler-owned schedule/host-graph construction, compiler reference orchestration, and source diagnostics. | A private second view/layout model; source-spelling CuTe compatibility; direct device resource ownership; framework/JIT scheduling. |
| `browsergrad-kernels` | WebGPU schedule legalization/selection, backend plans, WGSL, device resources, resident storage, pipeline/buffer caches, device profiling, and actual-device conformance. | Primary tensor/layout semantics; source parsing; Python/framework rules. |
| `browsergrad-jit` | PyTorch-shaped lazy graph, symbolic VJP, transforms, fusion intent, and framework-owned lowering into shared kernel/schedule/host representations. | Compiler internals; WGSL layout rules; permanent `CUSTOM` implementations for advertised portable core operations. |
| `browsergrad-grad` | Readable NumPy eager reference/autograd and explicit bridge behavior. | Naming-only dtype/device/view compatibility; accidental GPU-resident architecture. |
| `browsergrad-runtime` | Worker lifecycle, profile/rubric orchestration, cancellation, structured events, requirement resolution, and presentation/scheduling from canonical decisions. | Tensor evaluation, index-map interpretation, semantic lowering, backend heuristics, or compiler policy. |
| `browsergrad-primitives` | Browser-safe text, data, evaluation, simulation, hosted-training, and RL/math helpers. | Tensor/layout semantics or actual distributed/device execution. |
| `browsergrad-dogfood` | Packed/published-package and cross-package compatibility proof. | Reimplementation of package internals as test-only glue. |

Permitted dependency direction:

```text
compiler frontends -> compiler-owned lowering --+
                                                 |
JIT graph -> JIT-owned lowering -----------------+-> shared schedule/host artifacts
                                                 |       -> kernels WebGPU legalization/execution
explicit eager adapters -> validated bridge -----+

shared artifact representation and verification:
  semantic-core/layout + semantic-core/kernel + semantic-core/schema
  semantic-core/schedule + semantic-core/host where applicable

runtime -> semantic-core/diagnostic + semantic-core/capability + semantic-core/requirement only
primitives remains outside the tensor/compiler dependency chain
dogfood consumes public or explicitly private test contracts only
```

`browsergrad-kernels` MUST NOT import compiler internals.
`browsergrad-compiler` MUST NOT import JIT or Grad internals.
`browsergrad-jit` MUST NOT import compiler internals.
JIT Python MUST NOT encode WGSL layout rules.
Runtime MUST remain tensor-agnostic.

### Existing product contracts retained

The semantic migration does not weaken established package responsibilities:

- Runtime retains same-origin Pyodide loading, Worker isolation, stdout/stderr,
  timeouts, `AbortSignal`, cooperative interrupt where available, termination
  fallback, filesystem lifecycle, assertions/artifacts, and versioned
  manifest/profile validation.
- Grad remains readable closure-based eager autograd, NumPy-backed by default,
  with explicit forward-only bridge behavior and clear unsupported APIs.
- JIT remains lazy until explicit realization/materialization boundaries and
  keeps NumPy/CPU as reference while moving advertised core ops away from opaque
  callbacks. Its framework hot path does not depend on the source compiler
  package.
- Kernels remains Python-agnostic and owns actual WebGPU resources, resident
  buffers, prepared execution, feature/limit detection, and explicit readback.
- Compiler retains the shipping CUDA-lite frontend, stable source diagnostics,
  CPU reference, WGSL/WebGPU path, corpus gates, and prepared execution.
- Primitives remains course-agnostic and dependency-light.
- Dogfood continues to prove packed and published artifacts rather than
  workspace-link behavior.

## Diagnostics and Failure Semantics

All public compiler/execution failures use stable diagnostic codes and a
structured record:

```ts
interface Diagnostic {
  code: string;
  stage:
    | "preprocess"
    | "frontend"
    | "semantic-lowering"
    | "verification"
    | "scheduling"
    | "backend-lowering"
    | "device-validation"
    | "execution"
    | "evidence";
  severity: "error" | "warning" | "note";
  message: string;
  sourceSpan?: { fileId: string; start: number; end: number };
  semanticNodeId?: string;
  backendId?: string;
  capabilityId?: string;
  requirementId?: string;
  remediation?: string;
}
```

Tests assert codes and structured fields, not complete prose. Messages remain
human-readable and may improve without becoming a breaking API. A backend
limitation must identify the semantic operation, target profile, failed guard,
or missing device feature/limit.

The nine stage IDs above are a closed registry for this schema version. A
diagnostic records the boundary that rejected the artifact, not the package
that happened to emit it. Codes use a stable lowercase dot-separated namespace
owned by that boundary (for example `layout.rank.mismatch`); changing the
meaning of an existing code is a breaking change. Runtime error `kind` strings,
compiler feature-reason strings, readiness states, and prose are not diagnostic
codes unless explicitly registered and emitted through this record.

## Security, Reliability, and Operational Requirements

### Untrusted input and resource bounds

Source, semantic artifacts, schemas, shapes, and runtime bindings are untrusted
inputs. The platform MUST bound:

- Source/include bytes and include depth.
- Browser compiler worker/WASM transfer bytes, verified VFS pack/index/file
  bytes, cache bytes, source snapshots, WASM stack/linear memory, and allocator
  headroom as separate budgets. Mounted VFS bytes MUST fit the declared storage
  model without consuming memory reserved for Clang AST/template work.
- Preprocessor expansion and template-instantiation work.
- Diagnostic count and size.
- IR nodes, nesting, symbols, rank, and expression depth per artifact.
- Integer bit width and arithmetic-operation count during symbolic evaluation.
- CPU reference steps, allocations, and wall time.
- WGSL size, pipeline count, buffer bytes, dispatch dimensions, and queued work.
- Host-graph nodes, edges, materializations, and retained evidence.

Budgets are explicit configuration with safe defaults. Budget exhaustion is a
stable diagnostic, not an internal crash or browser hang.

### Cancellation and lifecycle

- Compiler passes and CPU reference execution poll cooperative cancellation at
  bounded intervals.
- Worker termination is the fallback when cooperation is unavailable.
- WebGPU command submission cannot be retroactively cancelled; cancellation
  stops future submissions, suppresses stale results, and releases owned
  resources when safe.
- Every session, device, resident buffer, prepared plan, compiler service job,
  and artifact cache has a documented owner and disposal path.
- Device loss invalidates device-scoped handles and caches. Reuse after loss is
  impossible by type/state or rejected by validation.

### Determinism, caching, and reproducibility

- Semantic verification and canonical serialization are deterministic.
- CPU reference randomness requires an explicit algorithm and seed.
- Cache keys name every semantic, schedule, backend, feature, limit, and
  numerical-policy input that can change output or legality.
- Performance measurements are never included in correctness hashes.
- Reproducing a failure requires artifact hash, producer versions, target
  profile, input hashes or seeds, device/browser facts, and diagnostic codes.

### Observability

Compile and execution stages emit structured timing and resource events with
artifact and stage IDs. Runtime wall time, compiler phase time, host estimates,
and WebGPU-owned measurements remain separate. The metrics contract in
`docs/platform/resource-metrics.md` applies; unavailable measurements are not
synthesized.

### Operational budgets

Each production profile publishes safety limits and regression budgets rather
than relying on one machine-specific global number. At minimum it records:

- Source, header, semantic-artifact, and generated-WGSL size limits.
- Cold and warm frontend/lowering/pipeline-creation latency.
- Peak worker, host, and device-owned bytes.
- First-result latency and steady-state dispatch/readback latency where claimed.
- Package and optional compiler-asset transfer size.

Safety-limit failure is a typed capability/budget result. Performance-budget
regression requires a recorded benchmark comparison and explicit acceptance;
it is never hidden by widening the budget in the same change without evidence.

## Extensibility and Debt-Prevention Rules

### Admission test for a new core semantic operation

A new core operation is admitted only when all are true:

1. Its observable semantics can be defined without source spelling or backend
   code fragments.
2. Operand/result, shape, dtype, numerical, memory-effect, and alias rules are
   specified.
3. The verifier and CPU reference behavior exist, or reference-unavailable has
   an approved architectural reason.
4. At least one real frontend lowers to it and one real backend consumes it.
5. JIT transformation/VJP/export decisions are explicit when the operation is
   framework-visible.
6. Negative tests prove illegal states and unsupported targets fail at the
   intended boundary.

If those conditions are absent, keep the behavior frontend-specific or
backend-specific until the common semantics are understood. Duplication during
discovery is preferable to freezing the wrong shared abstraction.

### Extension rules

- Core unions are closed and versioned. Extension operation IDs are namespaced.
- Required extensions fail closed when unknown.
- Backend hints are optional, namespaced schedule inputs. They do not affect
  semantic interpretation.
- Public compatibility surfaces have deprecation windows and migration
  fixtures; private adapters still require an exit gate.
- New package proposals must pass the repository package-consolidation deletion
  test. `semantic-core` passes because deleting it after adoption would force
  duplicated wire schemas and semantic validators across compiler, kernels,
  JIT, and runtime.

### Architecture checks

Automated architecture checks MUST enforce:

- No dependency cycles across semantic layers or packages.
- Representation/schema modules do not import passes, emitters, device APIs, or
  source frontends.
- Runtime imports only the diagnostic, capability, and requirement subpaths.
- Kernels do not import compiler internals.
- No new uses of frozen adapters from newly added features.
- No source-spelling CuTe handlers outside the frozen compatibility adapter.
- No advertised portable JIT operation lowers through opaque `CUSTOM`.
- Compatibility inventories and their executable fixtures remain complete,
  mutually referenced, and pinned to the reviewed source definitions.
- Generated Python bundles match editable Python sources.
- Public imports use package exports rather than `src/` or `dist/` paths.

### Compatibility behavior inventories

A compatibility inventory records current observable behavior that must not be
mistaken for target architecture. It is not a semantic-capability registry and
does not make compatibility debt normative for new code. The freeze manifest
owns permission to retain or change the adapter; the inventory explains the
behavior; executable fixtures prove it. None of those three substitutes for
the others.

Every dtype, view, materialization, interop, conversion, or opaque-operation
record MUST state the dimensions that vary by behavior. A closed
inventory-level execution context MAY state residency, backend, transform, and
export decisions once only when they are invariant for every record; an
individual record cannot override those values implicitly:

- Public surface and behavior class.
- Observation/evidence status separately from target conformance and the named
  reference contract. “Verified current behavior” does not mean “conformant.”
- Logical dtype/token identity versus physical storage dtype, compute dtype,
  and byte width where dtype is relevant, including the exact condition under
  which dtype preservation changes.
- Storage relationship (`must-alias`, `must-not-alias`, `same-object`, or a
  named condition), contiguity, and materialization behavior.
- Autograd/VJP, transform, export, backend, and residency decisions where the
  surface participates in those systems. `not-applicable` is explicit; a
  missing field is not a decision.
- Failure policy, source definitions including shared graph/context builders,
  fixture case IDs, and evidence tier.
- Environment versions whenever behavior is delegated to Pyodide, NumPy, a
  browser API, a native toolchain, or another independently versioned system.

An opaque-operation inventory has additional mandatory structure because an
opcode count or label allowlist cannot describe executable semantics. It MUST
record every constructor site and every reachable label separately, including:

- The exact label field (`name` for a CPU callback versus `op` for backend
  dispatch), enclosing definition, one record per constructor call,
  input-arity rule, shape rule, and constructor reachability condition.
  Multiple calls in one definition are separate records even when they form
  one public return value. Dynamic label helpers require a closed set of
  reviewed callers.
- Declared IR dtype rules separately from realized-result dtype behavior and
  validation. If a callback result is only checked to be an array, the
  inventory MUST say that shape and dtype are delegated and unvalidated; a
  declared dtype is not evidence of realized dtype. Version-pinned cross-dtype
  fixtures MUST cover the delegated behavior.
- Callback presence, effect class, determinism, and replay contract. RNG use,
  captured backward state, module-state mutation, and construction-time host
  shape probes are distinct effects; none may be called “pure.”
- CPU realization, closure backward, symbolic VJP, functional-grad, vmap,
  export, legacy backend, residency, and materialization decisions. Portable
  tensor-plan decisions MUST distinguish default admission, any explicit
  inspection/serialization opt-in, and executable backend admission. A plan
  builder accepting an opaque node behind `allowCustom` does not prove that an
  executor can run it. Forward-only results that set `requires_grad=false` are
  recorded as graph disconnection, not as a loud unsupported-gradient failure.
- Current failure boundary and target conformance. A constructor accepted by a
  frontend but rejected by every realizer is `constructor-only`, not supported.

The guard compares the executable label set exactly in both directions and
keeps `name` and `op` namespaces distinct. It MUST reject same-count site moves,
relabels to an already registered label, label removal, callback/arity/shape/
dtype/context changes, a new dynamic-helper caller, and stale generated Python.
Stateful callbacks also require replay evidence: a realization or backward walk
may not be assumed single-shot merely because the public API was invoked once.
Conditional construction and effects require fixtures on both sides: identity
or primitive fast paths do not inherit the callback record, and module state
mutation is recorded only for the exact mode and configuration that performs
it.

Conditional behavior MUST name and test both sides of the condition. For
example, “reshape is a view” is invalid if aliasing depends on input strides;
“bf16 is supported” is invalid when the public token resolves to four-byte f32
storage and f32 computation. Normalized source-definition hashes detect
implementation drift, but exclude comments and leading docstrings and are
never sufficient evidence without the observable fixture. Decorator lines
directly attached to definitions, signatures, executable statements, and
behaviorally relevant string literals remain fingerprinted.

Inventory schemas and enum vocabularies are closed and versioned. Every
fixture case is referenced by exactly one inventory behavior, every recorded
source definition is frozen, and every frozen definition that owns the
behavior is referenced by the inventory. Changing an environment version,
source behavior, physical dtype, alias relation, autograd decision, or fixture
result is an adapter-baseline change under ADR-0001, even when the public
method signature does not change.

The current Grad baseline is
`architecture/grad-compatibility-inventory.json`, with executable Pyodide
evidence in `packages/browsergrad-grad/tests-integration/fixtures/grad-view-bf16.v0.json`.
It explicitly records NumPy-dependent dtype fallback, the full alias registry,
Tensor versus torch constructor defaults, conditional reshape/index aliasing,
copying expand/detach/numpy behavior, live NumPy array-protocol exposure,
`contiguous()` as a non-materializing compatibility no-op, and cross-dtype
`to()` graph detachment.

### Adapter ledger

Every adapter has an owner, permitted callers, new-use prohibition, retirement
gate, and compatibility-removal version. An adapter is not "temporary" without
all five.

| Existing adapter/debt | Owner | Permitted callers | New-use prohibition | Retirement gate | Compatibility-removal version |
| --- | --- | --- | --- | --- | --- |
| Compiler pointer/scalar memory fields | Compiler | Existing CUDA-lite semantic lowering and the Gate 2 adapter only. | No new view, layout, dtype, or alias feature may add fields. | CPU reference and WGSL consume shared offsets, bounds, and alias facts. | Compiler `1.0.0`. |
| `cute_static_layout` parser path | Compiler | Existing parser integration and pinned regression fixtures only. | No new spellings, ranks, queries, or call sites. | Pinned real frontend handles its fixtures through layout semantics. | Compiler `1.0.0`. |
| `TensorGpuPlan` shape-only/f32 assumptions | Kernels | Existing JIT plan serializer and kernels executor/bridge only. | No new operation, dtype, view, offset, or alias semantics may enter this schema. | New view/dtype features enter through shared schemas; plan has no unique semantics. | Kernels `1.0.0`. |
| JIT `OP_CUSTOM` for core framework ops | JIT | Existing `_tensor_proxy.py`, `_functional.py`, `_nn.py`, `_webnn.py`, explicit user-kernel code, and embedded legacy wrappers only. | No new core labels or constructor sites; only explicitly user-authored kernel IDs may extend. | Advertised core operations have typed IR, CPU/VJP decisions, and backend decisions. | JIT `1.0.0`. |
| Eager view materialization and bf16 aliasing | Grad | Existing Tensor/torch compatibility adapters only where accurately documented. | No new API may claim view aliasing or bf16 storage through these paths. | View/alias fixtures agree or reject; bf16 is real storage/conversion or rejected. | Grad `1.0.0`. |
| `flashAttentionDirect` name | Kernels | Existing public export and realizer compatibility call only. | New APIs and docs use the accurate row-wise online-softmax name. | Real block-tiled implementation is proven and a normal major-removal window passes. | Kernels `1.0.0`. |
| Generic runtime backend labels | Runtime | Existing assignment requirement compatibility mapping only. | New readiness features use canonical requirement definitions/resolutions; semantic lowering uses capability decisions only where an explicit link exists. | All readiness UI consumes requirement resolutions and program-specific lowering records. | Runtime `1.0.0`. |

Each migration stage MUST delete an adapter, narrow its allowed callers, or make
its retirement gate measurably closer. Adding only new handlers, special cases,
or capability errors is not migration progress.

## Historical Migration-Baseline Corrections

This section preserves the migration baseline reviewed on 2026-07-15. It is
not current implementation status or the generated capability manifest. The
implementation ledger records current gate truth.

These were baseline gaps, not achievements to paper over with broader
acceptance or better wording:

| Area | Baseline state | Required before stronger claim |
| --- | --- | --- |
| Shared semantic core | No `browsergrad-semantic-core` package or canonical cross-package wire schema exists. | L1/L2 schemas, validators, canonical serialization, and two-consumer adoption. |
| C++/CuTe | Shipping compiler is CUDA-lite with limited scalar/static-layout handling, not a CUDA-capable C++/CuTe frontend. | CUTE-001 through CUTE-007. |
| Tensor layouts | Pointer/scalar memory semantics exist, but no canonical general `TensorView`/`IndexMap` model exists. | Value/layout contract plus differential CPU/WebGPU tests. |
| Scheduling | Existing Kernel IR and tensor plans mix useful semantics with backend-shaped constraints. | Explicit L2 semantic to L3 schedule boundary and frozen adapter rules. |
| Attention | Direct attention is fused row-wise online softmax, without proved workgroup K/V tiling. | Tiled attention workload gate and accurate naming. |
| JIT | Core primitive paths exist, but many public operations still use opaque `CUSTOM` NumPy closures. | Operation-by-operation typed IR and capability ledger. |
| Eager bf16/views | bf16 aliases normalize to f32 and some view behavior materializes. | Numerical/view conformance or early rejection. |
| Capability and readiness reporting | Status is spread across summaries, assignment labels, backend labels, and tests. | Generated semantic capability definitions/lowering decisions/evidence plus a separate assignment requirement registry and resolution records. |
| Real-device gate | Browser GPU suites may skip unavailable adapters and are not a required lane for every WebGPU claim. | Stable required actual-device lane for each released WebGPU capability. |

Existing compiler semantic IR, CPU references, tensor-plan execution, tiled
GEMM, WebGPU orchestration, resident buffers, JIT primitives, runtime rubrics,
and release dogfood are valuable substrate. The migration extends these seams;
it does not replace them with a second source-shaped stack.

## Authoritative Migration Sequence

This is the only normative delivery order in this document. Work MAY proceed in
parallel after its dependencies are satisfied, but a later gate cannot be used
to claim an earlier gate complete.

### Gate 0 — Truth, vocabulary, and freeze lines

- Freeze new `cute_static_layout`, shape-only `TensorGpuPlan`, and core
  `OP_CUSTOM` expansion.
- Define stable diagnostic-stage, semantic-capability, backend, and assignment-
  requirement identifiers without deriving one namespace from another.
- Inventory public dtype/view/materialization behavior, opaque operations, and
  legacy assignment requirement labels with explicit classifications.
- For JIT opaque operations, close the exact constructor-site/label matrix and
  record CPU, autograd, transform, export, backend, residency, effect, and
  replay decisions operation by operation; remove token-scan false positives.
- Add architecture checks for forbidden new dependencies and frozen adapters.

**Exit:** every frozen adapter has an owner, caller boundary, retirement gate,
machine-readable behavior inventory where applicable, source guard, and
executable fixture; stable vocabularies are registered; no new feature can
extend or silently reinterpret a frozen adapter without an accepted
architecture exception.

### Gate 1 — Value/layout core and wire foundation

- Create the private semantic-core package and narrow subpaths.
- Implement `DimExpr`, constraints, dtype/numerical policy, allocation/view,
  layout/index, verifier, canonical serialization, hashing, and resource limits.
- Add cross-language TypeScript/Python fixtures and property generators.
- Add coordinate/address/alias traces and stable schema diagnostics. Effect
  traces begin with the concrete L2 `/kernel` contract rather than being
  invented by L1.

**Exit:** serialized fixtures have one verified meaning and deterministic hash
across TypeScript and Python.

### Gate 2 — Multi-frontend, multi-backend view slice

- Implement the verified L2 materializing view-copy operation with explicit
  effects, exact padding-fill bits, fail-closed overlap policy, and a shared CPU
  evaluator.
- Start with rank-2 transpose as the tracer bullet. Lower a read-only compiler
  storage-pointer binding and the typed JIT `PERMUTE` family into the same
  verified layout/view-copy artifact and require identical semantic hashes.
- Extend the unchanged operation shape to rank-3 permutation, positive strided
  slice, read-only broadcast, nonzero byte offset, and padded rank-2/3 views.
  Signed/negative-stride profiles remain rejected until backend integer
  division/modulo semantics are proved equivalent to the canonical BigInt
  rules.
- Execute the exact same artifacts and buffers through CPU reference and real
  WebGPU. Padding uses structured guarded loads; eager `select`, CPU zero-fill,
  or robust-buffer behavior is not conformance evidence.
- Keep verified layout/kernel artifacts in a side table keyed by stable
  operation/value identity. Derive the legacy tensor plan only for compatible
  scheduling/liveness facts; do not extend it or recover offsets, predicates,
  aliasing, dtype, or view meaning from its shape/axes fields.
- Make the conformance lane require a WebGPU adapter/device and fail on skips.
  Evidence records the artifact hash, input hash, comparison policy, adapter,
  features, limits, and environment failure when execution is unavailable.

**Exit:** two frontend paths and two execution tiers consume the same view/index
fixtures; reference and WebGPU do not reconstruct offsets independently;
transpose, strided slice, broadcast, and padded rank-2/3 cases pass without
widening the frozen tensor-plan schema or treating device absence as success.

### Gate 3 — Real C++/CuTe frontend slice

- Pin one browser-local CUDA-capable Clang-WASM profile and CuTe/CUTLASS
  revision, including exact compiler/WASM assets, headers, options, target ABI,
  compiler runtime ABI, virtual include roots, worker bytes, build provenance,
  license inventory, and reproducible build recipe.
- Build one BrowserGrad LibTooling/AST extractor with pinned LLVM/Clang and
  Emscripten revisions. Keep its build sysroot separate from the parsed-program
  virtual sysroot, and prove explicit CUDA host/device semantic passes.
- Verify compiler-resource and dependency headers through the closed v1 VFS
  pack and collision-free installation authorities. General archives and
  corpus-minimal header closures do not satisfy the profile.
- Compile supported unmodified source inside a dedicated browser worker with
  bounded input, preprocessing/template work, memory, output, time, and
  cancellation. Docker, native Clang, and a compiler service are absent from
  this required path.
- Use one producer-neutral source request. Keep any declared source reference
  and conformance assertions detached from request identity. Verify exact
  package-owned worker bytes before Blob-worker construction; no
  self-attestation or unverified-URL fallback is allowed.
- Prove layout-only fixtures, then dynamic `Tensor<Engine, Layout>` binding and
  copy/view operations.
- Preserve source spans and typed unsupported target intrinsics.
- Verify the browser-produced artifact through the producer-neutral artifact
  contract and lower it through the same semantic-core seams used by Gate 2.
- If a service or native/AOT producer is retained, prove it as an optional
  profile with the same protocol and explicit cross-producer parity evidence;
  its availability is not a Gate 3 exit condition for the portable product.

**Exit:** without Docker, a native toolchain, or a compiler service, unmodified
pinned source compiles in a supported browser into a verified artifact,
produces the same verified view/index semantics as Gate 2, and runs its
portable subset on real WebGPU. Retained optional producers pass their declared
parity profiles, but their absence cannot block this browser-local exit.

### Gate 4 — Tiled GEMM and schedule separation

- Add logical tile and schedule IR separation.
- Implement workgroup staging, active masks, uniformity analysis, and portable
  tiled MMA accumulation policy.
- Compare CPU reference and real WebGPU across boundary tiles, irregular sizes,
  alignments, and supported dtypes.

**Exit:** tiled GEMM is lowered without source handlers or backend strings in
semantic IR, and its preservation level is accurately reported.

### Gate 5 — Tiled attention flagship

- Keep the row-wise online-softmax implementation as a named baseline.
- Implement tiled Q/K/V views, K/V staging, online softmax across tiles,
  boundary masks, and synchronization legality.
- Validate causal/non-causal and declared dtype/numerical policies.
- Record correctness separately from performance on named devices/browsers.

**Exit:** the block-tiled/FlashAttention claim is allowed only for the exact
proved algorithm and execution profiles.

### Gate 6 — Framework convergence

- Migrate advertised JIT core operations off opaque callbacks and direct
  backend layout ownership.
- Give each migrated operation typed semantics, CPU handler, VJP or refusal,
  transform/export decision, WebGPU decision, and residency/materialization
  behavior.
- Make Grad view/dtype behavior agree with conformance fixtures or reject before
  execution.
- Make runtime/profile UI consume assignment requirement resolutions plus
  program-specific semantic capability/lowering records where applicable.

**Exit:** framework-facing support tables are generated from the same contracts
used to compile and execute.

### Gate 7 — Host graphs and optional systems expansion

- Generalize multi-dispatch host graphs, bounded dynamic control, pipelines,
  and explicit collectives.
- Add native companion lowerings only for named facilities and profiles.
- Keep browser worker meshes and native distributed systems as separate
  execution products with shared high-level collective meaning where valid.

**Exit:** each systems claim names failure model, transport/topology, semantic
preservation, and actual execution evidence.

## Proof Matrix and Release Gates

| Claim | Required minimum evidence |
| --- | --- |
| Wire/schema compatibility | Cross-language golden fixtures, canonical-hash fixtures, old-reader/new-writer and new-reader/old-writer tests, malformed/adversarial decode tests. |
| Layout/view behavior | Property tests and coordinate traces over hierarchy, offsets, strides, slices, broadcasts, aliases, bounds, and rank-2/3 dynamic cases. |
| Source compatibility | Pinned unmodified source through the real compatibility profile, with source spans and semantic artifact inspection. |
| CPU reference | Numerical, memory-effect, alias, bounds, and host-order tests against declared policy. |
| Portable WebGPU | Actual device execution matching reference under an explicit comparison policy; feature/limit and device-loss paths tested. |
| Native facility | Native test proving the named facility and matching the same semantic contract. |
| JIT framework op | Typed IR, CPU handler, VJP/refusal, transform/export decisions, backend decision, residency/materialization tests. |
| Performance | Named implementation and baseline, recorded hardware/browser/configuration, warmup/statistical method, and separate correctness proof. |
| Package release | Build, typecheck, lint, package tests, packed-tarball verification, public export checks, and dogfood against installed artifacts. |

Evidence outcomes are `not-run`, `passed`, or `failed`. A skipped adapter is an
environment result, not a passing WebGPU test. Release notes and docs MUST state
the exact capability profile proven by required actual-device lanes.

Use narrow gates during development:

```sh
pnpm --filter @unlocalhosted/browsergrad-runtime test
pnpm --filter @unlocalhosted/browsergrad-grad test
pnpm --filter @unlocalhosted/browsergrad-grad test:integration
pnpm --filter @unlocalhosted/browsergrad-jit test
pnpm --filter @unlocalhosted/browsergrad-jit test:integration
pnpm --filter @unlocalhosted/browsergrad-kernels test
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser
pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler
pnpm --filter @unlocalhosted/browsergrad-compiler verify:real-world-cuda -- --skip-fetch --require-webgpu
pnpm --filter @unlocalhosted/browsergrad-primitives test
pnpm --filter @unlocalhosted/browsergrad-dogfood test:node
```

Before release-level confidence:

```sh
pnpm -r run build
pnpm -r run typecheck
pnpm -r run test
pnpm test:release-packages
```

Production-ready means package gates, Pyodide integration, claimed actual-device
WebGPU lanes, compiler corpus gates, packed-artifact tests, and documentation
all match the same capability manifest. No aggregate green check can override a
missing tier-specific gate.

## Terminology Rules

| Avoid as a product claim | Use instead |
| --- | --- |
| "toy compiler", "just a lab compiler", "small supported subset" | The exact frontend, semantic family, and executable backend tier. |
| "CuTe support" | Compatibility profile: compiler/CUDA/CuTe revisions, frontend outcome, semantic coverage, backend tiers, and preservation level. |
| "same semantics" | Name observable, numerical, memory/effect, schedule, and native-facility preservation separately. |
| "FlashAttention v2" | "fused row-wise online-softmax baseline" until the tiled-attention gate is proven. |
| "bf16 support" for f32 values | "bf16 storage/conversion with widened arithmetic" when true, or reject. |
| "GPU supported" | Operation + execution tier + backend profile + residency/materialization behavior. |
| "fallback" | The named behavior: CPU reference, explicit conversion, materializing transfer, host-lifted operation, or unavailable capability. |
| "passed" for a skipped GPU environment | `not-run` with the environment reason. |

Historical documents may describe the shipping CUDA-lite API or an older
release. They MUST label that scope and MUST NOT present it as BrowserGrad's
architectural destination.

## External Technical Anchors

This architecture is informed by, but does not copy, the following upstream
contracts:

- [WebGPU specification](https://www.w3.org/TR/webgpu/) for device features,
  limits, validation, asynchronous operation, and device loss.
- [WGSL specification](https://www.w3.org/TR/WGSL/) for types, memory/address
  spaces, uniformity, barriers, and floating-point accuracy.
- [NVIDIA CUTLASS/CuTe documentation](https://docs.nvidia.com/cutlass/latest/overview.html)
  for hierarchical layouts, `Tensor<Engine, Layout>`, tiled copy/MMA concepts,
  and architecture-specific facilities.
- [LLVM CUDA compilation documentation](https://llvm.org/docs/CompileCudaWithLLVM.html)
  for the distinction between ordinary C++ compilation and CUDA-capable source
  compilation.
- [MLIR rationale](https://mlir.llvm.org/docs/Rationale/Rationale/) and
  [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) as design
  evidence for progressive lowering and separation of payload semantics from
  transformation/scheduling control.
- [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html) as a reference for
  deterministic JSON canonicalization; BrowserGrad additionally uses explicit
  encodings for large integers and floating bit patterns.
