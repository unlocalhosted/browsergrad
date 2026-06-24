# PRD-020 — Real-World CUDA Compatibility Ladder

## Problem Statement

BrowserGrad can now run a large curated CUDA-learning corpus, but fresh
real-world CUDA repositories show a different failure mode: not missing GPU
dispatch, but CUDA/C++ intake. `CUDA-120-DAYS--CHALLENGE` is fully runnable
through strict WGSL or host-orchestrated WebGPU, while NVIDIA `cuda-samples`,
Karpathy `llm.c`, and LeetCUDA mostly fail before Kernel IR because realistic
files carry macros, templates, vector pack idioms, half intrinsics, host binding
glue, and library-shaped helper code around otherwise teachable kernels.

The platform vision needs more than one course corpus. We need a repeatable
compatibility ladder that takes arbitrary educational CUDA repos, audits them
without repo-specific patches, classifies gaps by semantic family, and advances
generic BrowserGrad compiler primitives only when they help multiple kernels or
teach a first-principles GPU concept.

## Solution

Add a corpus-driven compatibility program for `@unlocalhosted/browsergrad-compiler`.
The program keeps LeetCUDA, NVIDIA `cuda-samples`, Karpathy `llm.c`, and
`CUDA-120-DAYS--CHALLENGE` as pressure corpora; turns their failures into
stable semantic gap reports; and implements generic primitives in ladder order:
source normalization, CUDA math/half intrinsics, vector pack memory views,
constexpr/template specialization for simple kernels, warp/subgroup lowering,
and selected runtime/library islands.

This PRD is intentionally planning-only. It records the research, coverage
truth, first vertical slice, and test gates before more compiler code lands.
No support should be added because a file path contains `LeetCUDA`; support is
valid only when it is a reusable CUDA primitive with parser, analyzer,
reference, WGSL, browser, and corpus evidence.

## User Stories

1. As a learner, I want LeetCUDA-style elementwise, activation, reduction, and matmul kernels to run in the browser, so that I can practice real CUDA lessons without installing CUDA.
2. As a lab author, I want a corpus coverage report with semantic gap families, so that I can choose assignments based on what BrowserGrad honestly supports today.
3. As a compiler contributor, I want corpus failures minimized into generic CUDA primitives, so that I can improve the compiler without adding repo-specific hacks.
4. As a platform maintainer, I want coverage thresholds per corpus, so that new compiler work cannot silently regress CUDA-120 while chasing LeetCUDA or `llm.c`.
5. As a learner debugging a kernel, I want unsupported macros, half intrinsics, vector packs, templates, and warp calls to produce source-spanned diagnostics, so that I know what CUDA concept is outside the browser subset.
6. As a performance-minded user, I want vectorized load/store idioms such as `float4` and `half2` to lower to efficient WebGPU memory operations when safe, so that browser kernels do not become toy scalar-only exercises.
7. As a curriculum builder, I want one compatibility vocabulary across CS149, CS336, LeetCUDA, CUDA samples, and custom labs, so that the platform does not become a pile of course adapters.
8. As a future researcher, I want deep modules around source normalization and intrinsic lowering, so that later Triton-like or LLVM-backed frontends can reuse the same Kernel IR and test oracles.

## Research Dossier

Repo exploration:

- Current compiler architecture already separates lexer/parser, analyzer,
  Kernel IR, CPU reference, WGSL emitter, runtime planning, WebGPU orchestration,
  and compatibility families. This PRD should extend those seams, not bypass
  them.
- `docs/platform/cuda-compatibility-spine.md` says every CUDA gap maps to a
  family: `frontend`, `memory`, `atomic`, `texture`, `subgroup`, `library`,
  `runtime`, `feature`, or `safety`.
- Current `CUDA-120-DAYS--CHALLENGE` gate reports `240/240` WebGPU-runnable
  kernel definitions: `225/240` strict direct WGSL and `15/240` host-orchestrated
  real WebGPU.

Local corpus audits on 2026-06-24:

- `NVIDIA/cuda-samples` at `b7c5481`: `357` kernel definitions, `282` direct
  WebGPU-runnable after source/context normalization plus intrinsic-ledger
  expansion, scalarized CUDA vector storage views, and simple C++ alias /
  constexpr intake plus cooperative-groups namespace call forms and typed
  `reinterpret_cast<T*>` storage views, bounded CUDA template constants, and
  fast math/bit intrinsics plus 2D float texture-object lowering and
  launch-context template specialization plus `half2` f16 vector storage, with
  mutable pointer-parameter rebasing, initialized `__constant__` memory,
  CUDA integer bitwise atomics, and C-style multi-declaration `for` loops with
  comma update clauses, plus CUDA byte-vector aliases, implicit helper-call
  scalar casts, deterministic synthetic `clock()`, concrete template argument
  preference over unresolved wrapper parameters, and `SharedMemory<T>` dynamic
  shared-memory helper lowering, plus CUDA integer helper intrinsics, subgroup
  vote intrinsics, and `__sincosf` source lowering, with multiline templated
  launch detection, call-shaped pointer-parameter device-helper intake,
  define-backed device-helper template defaults, and dynamic-launch target
  context for host orchestration, with CUDA integer/vector intrinsic additions,
  scalar vector-constructor splats, and semantic `cp.async` source
  normalization, safe CUDA opaque/index alias intake, `volatile` qualifier
  handling, `atomicAdd_system`, and UMUL/UMAD helper intrinsics, plus CUDA
  declarator qualifier/attribute intake, constructor-style vector locals,
  shadowable `lerp`, and `atomicExch_system`, with cooperative-group helper
  parameter handles, conservative `__syncwarp` lowering, and CUDA half2
  conversion aliases, real CUDA `while` statement lowering, alias-backed
  value template argument resolution with shared `sizeof`/`alignof` layout
  folding, output-only inline PTX `laneid` lowering, and PTX `bfind.u32`
  lowering, plus C-style assignment-chain statement lowering and deterministic
  WGSL alpha-renaming for source symbols that collide with output identifiers,
  and modeled device-pointer `atomicAdd` helper dispatch through storage/shared
  buffer ids, same-name template value fallback propagation, and dynamic extern
  shared-memory context from translation units and device helpers,
  and translation-unit `__shared__` scratch-array injection plus conservative
  in-kernel `#if`/`#ifdef` branch pruning and `static __global__` intake, plus
  pointer/null identity lowering, broader CUDA system-scope atomic aliases,
  `clock_t*` helper intake, and `half2` f16 FMA/lane extraction plus native
  vector arithmetic, and typed CUDA texture-vector reads plus vector-scalar
  arithmetic over CUDA vector values and real multi-channel WebGPU textures,
  plus generic unary pointer dereference lvalues, alias-preserving vector
  member writes, `tex2DLod` / `tex1Dfetch` texture aliases, and guarded
  `surf2Dread` surface loads, plus vector reinterpret memory-view helpers
  through the device pointer ABI, plus subgroup ballot/coalesced-group vote
  primitives and CUDA `__popc`, plus reverse translation-unit alias/helper
  context for header-only kernels, POD-record vector alias lowering, numeric
  object-macro folding, local const/template integer dimension folding, and
  scalar bitwise compound assignments, default kernel parameter initializer
  intake, CUDA cache-hint load/store family lowering, conservative unresolved
  template type fallback for pointer parameters, and signed `ptrdiff_t` index
  alias intake, u32-backed `bool*` storage ABI support, braced CUDA vector
  helper arguments, empty statement bodies, explicit `double` diagnostics,
  side-effect canonicalization, conservative `for`-scope renaming, nullable
  conditional storage-pointer helper args, and storage-pointer assignment
  rebasing, plus CUDA pipeline `cuda::memcpy_async` lowering into existing
  `cp.async` copy/fence primitives, plus scalarized CUDA POD record storage,
  translation-unit constant-record reachability, self-alias-safe field typing,
  macro-sized record arrays, DirectX-style float vector field aliases, and
  C-style array typedef vector aliases, plus CUDA inverse trig aliases and
  vector `length(v)` helper lowering, explicit opt-in f64-to-f32 compatibility
  lowering for educational WebGPU runs, and `__usad4` plus
  `vabsdiff4.u32.u32.u32.add` byte-lane SAD lowering, plus generic
  `thread_group` helper-parameter lowering with block/tile metadata,
  adjacent C string literal intake, scalar `std::size_t`/`auto` declarations,
  scalar brace constructors such as `__half{expr}`, and driver API
  `CUtexObject` / `CUsurfObject` texture-surface aliases, texture-handle
  device-helper params, and generic tile-reduce helper params, plus
  atlas-backed `tex1D` / `tex2DLayered` / `tex3D` / `texCubemap`
  point-sampling, templated `surf2Dread<T>` return-form loads,
  scalarized `surf1Dwrite` / `surf2DLayeredwrite` writes, vector min/max
  overloads, CUDA vector assignment chains, and POD-style vector field aliases
  such as `.S` / `.MuByT`, plus local scalar out-params in device helpers,
  CUDA opaque RNG handle aliases, function-pointer typedef intake,
  numeric scalar-template fallback for count/index params, C `frexp` exponent
  out-params, and `typename vecN<T>::Type` vector-carrier alias lowering,
  plus conservative mutable integer reference-param lowering for pointer-form
  atomics, device-pointer `atomicExch` / `atomicCAS` helper dispatch, and
  reference-runtime pointer-rebinding hardening for shared scalars,
  plus `short`/`uint16_t` pointer-base alias intake and scalar device helpers
  containing `static_cast<T>` expressions, plus top-level mutable `__device__`
  scalar/array globals carried through corpus context as read-write WebGPU
  storage, and inferred dynamic extern shared-memory launch metadata for CUDA
  vector, aligned byte, and f64 educational fallback scratch buffers,
  with `75` hard
  gaps.
  Main failures:
  parser/frontend gaps, texture/vector
  operators, remaining `half2` intrinsics, templates, and runtime library
  shape.
- `karpathy/llm.c` at `f1e2ace`: `148` kernel definitions, `148` direct
  WebGPU-runnable after source/context normalization, intrinsic-ledger
  expansion, CUDA/C named constants, CUDA cache-hint memory builtins, local
  header context, simple C++ alias / constexpr intake, and typed storage
  pointer aliases plus `warpSize` / `NULL` named constants, in-kernel
  namespace aliases, cooperative tile meta-group queries, and dynamic
  `float4` lane reads plus local array initializer intake, local pointer-alias
  cache-hint loads, shared-array pointer decay, explicit 32-bit pointer
  reinterprets, alias-aware atomics, CUDA `atomicInc` / `atomicDec`, and
  C-style pointer truthiness for conditional helper pointer args, plus scoped
  `Packed128<half|bf16>` pointer-view scalarization over shared byte buffers
  and function-pointer-like template symbol propagation through launch wrappers.
  semantics plus stricter constant hygiene and CUDA helper intrinsics such as
  `div_ceil`, fixed register fills, shared-address conversion, and mutable
  pointer-parameter rebasing, plus generic `Packed128<float>` alias lowering,
  128-bit load/store helper normalization, vector `.size`, and local vector
  dynamic lane read/write semantics, plus call-shaped helper intake and
  define-backed device-helper template defaults, plus conservative
  `__syncwarp` lowering plus header-carried `Packed128<float>::size` folding,
  const pointer rebasing plus pointer/null comparison semantics, generic
  dereferenced local pointer/address value typing, vector reinterpret
  memory-view helpers through the device pointer ABI, multi-dimensional shared
  helper pointer params, and semantic `blockReduce<warpReduce*>` lowering, plus
  C line-continuation folding, simple statement-lambda inlining, scalar
  template helper inference, shadow-safe context defines, CUDA `break`,
  POD-record vector alias lowering, numeric object-macro folding, local
  const/template integer dimension folding, scalar bitwise compound
  assignments, CUDA bf16 logical type/intrinsic intake, `__trap`, unary
  bitwise-not, mutable local storage-pointer handles, CUDA cache-hint
  load/store family lowering, conservative unresolved template type fallback
  for pointer parameters, signed `ptrdiff_t` index alias intake, template-aware
  portable device-helper closure, POD-return cooperative-group reference helper
  intake, explicit template-id device specialization discovery, and CUDA
  `__reduce_add_sync` subgroup lowering, plus scalarized
  `Packed128<half|bf16>` register packs and bf16 cache-hint pointer helpers,
  plus C++ block-scope shadowing and bool template-carrier parameter
  substitution plus alias-backed helper closure that skips semantic builtin
  cache/load helper shadowing, plus explicit opt-in f64-to-f32 compatibility
  lowering for educational WebGPU runs, plus generic block-size template
  fallback, atomic-wrapper forwarding, and pointer-store wrapper forwarding,
  plus bf16 dynamic extern shared-memory inference and scalar 128-bit
  cache-load assignment expansion into lane-wise stores, plus custom
  CUDA-vector `cg::reduce` lowering through scalar subgroup shuffle-XOR loops,
  with `0` hard gaps.
- `xlite-dev/LeetCUDA` at `c5dde9a`: `293` kernel definitions, `264` direct
  WebGPU-runnable after source/context normalization plus intrinsic-ledger
  expansion, scalarized CUDA vector storage views, local header context, and
  simple C++ alias / constexpr intake plus `FLOAT4(x)`-style typed storage
  views, bounded integer template defaults, and concrete launch-context
  template specialization plus `half2` f16 vector storage and braced vector
  initializers, static shared declarations, flexible device helper attributes,
  `half2` arithmetic intrinsics, builtin infinity lowering, and standalone C
  block scopes plus CUDA shuffle/fence/conversion intrinsics, and generic warp
  reduction aliases plus CUDA half conversion aliases, object-macro device
  helper discovery, POSIX/C math constants, call-shaped helper intake, and
  define-backed device-helper template defaults, with semantic `cp.async`
  source normalization, synchronous pointer-form lowering, bounded dependent
  carrier alias/constexpr folding, qualifier-macro helper discovery, and
  unsupported inline-PTX section parsing, fewer raw parser gaps, and
  multi-dimensional shared-memory address lowering for
  `__cvta_generic_to_shared`, plus homogeneous POD-record lowering into CUDA
  vectors, safe numeric object-macro folding, local const/template integer
  dimension folding, scalar bitwise compound assignments, CUDA bf16
  logical type/intrinsic intake plus integer warp-reduction aliases, and C++
  std math aliases such as `std::isinf` /
  `std::numeric_limits<float>::infinity()`, plus a shared inline-PTX classifier
  and multi-output register-carrier ABI for `ldmatrix` and
  `mma.sync.m16n8k16` in CPU reference and WGSL. This is intentionally a v0
  register-carrier model, not full lane/layout-accurate tensor-core simulation
  yet. CUDA reciprocal intrinsic `__frcp_rn` now lowers through the shared
  intrinsic table, closing the post-PTX flash-attention math gap. Shared-memory
  pipeline template params such as stage count, padding, and warp-swizzle flags
  get conservative defaults only inside shared/pipeline contexts, leaving `29`
  hard gaps.
  The pre-normalizer baseline was `3/293`, which proved context isolation was
  the first ladder rung.

External sources:

- LeetCUDA is a broad educational CUDA kernel repository with many deep-learning
  kernels and PyTorch extension bindings:
  https://github.com/xlite-dev/LeetCUDA
- NVIDIA CUDA C++ Programming Guide documents CUDA execution, memory, warp
  functions, cooperative groups, inline assembly, and the C++ programming model
  that real corpora use:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- NVIDIA CUDA Math API documents half and half2 intrinsics such as arithmetic
  and comparison operations that appear in LeetCUDA:
  https://docs.nvidia.com/cuda/cuda-math-api/
- NVIDIA PTX ISA documents inline PTX instruction semantics including
  `bfind.u32`, which BrowserGrad lowers into the reference interpreter and
  WGSL:
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- WGSL is the browser-native target and includes workgroup memory, barriers,
  atomics, and feature-gated shader capabilities:
  https://www.w3.org/TR/WGSL/
- Chrome WebGPU subgroups require explicit WGSL enablement and are now a real
  browser feature surface, not merely a future idea:
  https://developer.chrome.com/blog/new-in-webgpu-134
- Chrome WebGPU compatibility mode broadens hardware reach while constraining
  available features, so compiler diagnostics must stay feature-gated:
  https://developer.chrome.com/blog/new-in-webgpu-146
- CUTLASS/CuTe shows the modern CUDA frontier for template-heavy tensor-core
  code. BrowserGrad should learn from its motifs, but full C++ template
  compatibility is not the first ladder rung:
  https://docs.nvidia.com/cutlass/

What this changes:

- We should stop measuring only curated lab corpora. Every compiler milestone
  needs one curated gate and at least one hostile real-world corpus gate.
- The next work is not "make LeetCUDA pass." It is a generic compatibility
  ladder whose first proof happens to improve LeetCUDA, `llm.c`, and samples.
- The most valuable first code slice is frontend/context normalization plus
  reusable intrinsic tables, not another runtime orchestration feature.
- The current live aggregate gate is `934/1038` WebGPU-runnable across the four
  pinned corpora: CUDA-120 `240/240`, `cuda-samples` `282/357`, `llm.c`
  `148/148`, and LeetCUDA `264/293`.

## Grill Decisions

Question: What is the smallest vertical slice that proves this idea end-to-end
without closing the door on broader CUDA compatibility?

Recommended answer: Add LeetCUDA as a named audit corpus, implement generic
single-kernel context isolation in the audit/compiler intake path, then add a
small CUDA intrinsic table for `fma`/`fmaf` and scalar half arithmetic/comparison
calls. This should improve coverage without touching any LeetCUDA file names.

Decision: Accept this as the first implementation slice for PRD-020.

Question: Should the compiler parse full CUDA C++ templates now?

Recommended answer: No. Start with constexpr-like template parameter
specialization for simple kernels after context normalization works. Full C++
template parsing belongs to a later compatibility backend or a dedicated
front-end module.

Decision: Template support starts as bounded specialization, not general C++.

Question: Should vector pack macros such as `FLOAT4(x)` and `HALF2(x)` be
treated as parser hacks?

Recommended answer: No. Treat them as memory-view primitives: reinterpret-load,
reinterpret-store, lane access, alignment diagnostics, and WebGPU lowering. The
macro spelling is only one source form.

Decision: Add vector pack support through a semantic memory-view abstraction.

Question: Should unsupported CUTLASS/CuTe/MMA code be rejected permanently?

Recommended answer: No. Keep it as an expansion track. First support explicit
educational kernels; later explore motif lowering for `wmma`, `mma`, and WGMMA
patterns when the frontend and vector memory model are stronger.

Decision: MMA/CuTe is future compatibility pressure, not v1 of this ladder.

Question: Should corpus audit scripts hide unsupported kernels to make numbers
look better?

Recommended answer: No. The audit may classify non-kernel host binding glue,
docs, and pseudocode separately, but every extracted CUDA kernel should either
compile, become reference/WebGPU liftable, or produce a stable blocker.

Decision: Coverage reports stay honest and machine-readable.

## Novelty Reach

- Browser-native CUDA corpus normalizer: a source-intake stage that builds a
  minimal kernel compilation unit from macros, constants, called device helpers,
  launched sibling kernels, and relevant type aliases. This is lighter than a C++
  compiler but stronger than regex extraction.
- Semantic vector pack ABI: represent `float4`, `int4`, `half2`, and 128-bit
  load/store idioms as IR memory views with alignment and lane semantics, then
  lower safe cases to WGSL vectors or scalarized storage operations.
- Intrinsic ledger: model CUDA math, half, warp, atomic, and runtime intrinsics
  as a data-driven table with analyzer arity/type rules, CPU reference semantics,
  WGSL lowering, required WebGPU features, and fallback classification.
- Corpus delta dashboard: each PRD slice records before/after counts per corpus
  and per semantic family. This keeps "Carmack-style" progress empirical rather
  than vibe-based.
- Motif lowering research lane: for future CUTLASS/CuTe-like code, detect
  recognizable tensor-core/tiling motifs and lower the educational intent to
  BrowserGrad IR, instead of trying to embed an entire native C++ compiler in
  the default browser path.

## Implementation Decisions

- Add LeetCUDA to the dogfood corpus matrix as an external pressure corpus,
  alongside CUDA-120, NVIDIA `cuda-samples`, and `llm.c`.
- Extend the corpus audit output with stable corpus identity: repo URL, commit,
  kernel count, direct WebGPU count, host-orchestrated count, reference-only
  count, hard gaps, error codes, and semantic families.
- Add thresholds per corpus. CUDA-120 should stay at `240/240`
  WebGPU-runnable. LeetCUDA, samples, and `llm.c` are hard no-regression gates
  from their measured baselines, then graduate to higher minimums after generic
  compatibility primitives land.
- Build a source normalizer module as a deep module. It should create a minimal
  compilation unit for one kernel and include only relevant context: safe object
  macros, expression macros, constants, type aliases, called device helpers,
  and actually launched sibling kernels.
- Keep unsafe or not-yet-modeled macros loud. `reinterpret_cast`-style macros
  should become a memory-view diagnostic or memory-view IR primitive, not raw
  token expansion that poisons unrelated kernels.
- Build an intrinsic ledger as a deep module. Initial rows should include
  `fma`, `fmaf`, scalar half conversion/arithmetic/comparison functions, and
  then expand to half2 and warp calls.
- Add vector pack memory views after scalar half intrinsics. `float4` and
  `half2` support should include reference semantics, WGSL lowering, and
  diagnostics for unsupported alignment or aliasing.
- Add bounded template/constexpr specialization only after context
  normalization and vector pack semantics are tested.
- Preserve package boundaries. Compiler owns CUDA semantics and IR. Kernels
  package owns WebGPU device dispatch. Platform/craftingattention consumes
  compiler capability reports but does not duplicate compatibility heuristics.
- Treat all corpus changes as generic primitives. No file-name, repo-name, or
  assignment-name branches in compiler behavior.

Suggested implementation chunks:

1. PRD-020 and corpus registry docs.
2. Corpus audit registry plus LeetCUDA report gate.
3. Minimal-kernel context isolation and source-normalizer tests.
4. Intrinsic ledger for `fma`/`fmaf` and scalar half intrinsics.
5. Vector pack memory-view IR for `float4` and `half2`.
6. Bounded template/constexpr specialization for common educational kernels.
7. Warp/subgroup lowering expansion for reduction-heavy kernels.
8. Corpus dashboard and threshold graduation.

## Testing Decisions

- Test external behavior: a kernel either compiles to source-spanned IR, runs in
  CPU reference, runs on real WebGPU, or produces a stable diagnostic. Avoid
  tests that pin parser implementation details unless they protect a public
  diagnostic.
- Add source-normalizer fixtures for mixed files containing multiple kernels,
  sibling kernels, safe macros, unsafe macros, constants, device helpers, and
  host binding glue.
- Add intrinsic-ledger tests that assert arity, feature gates, reference output,
  WGSL output, and unsupported fallback classification for each intrinsic row.
- Add reference tests for half scalar math with the existing float16 array
  helper so CI does not depend on native `Float16Array` availability.
- Add browser tests for at least one f32 intrinsic kernel and one `shader-f16`
  kernel when the browser exposes the feature. Feature absence should skip with
  a deterministic reason.
- Add vector pack tests with scalar fallback equivalence first, then WebGPU
  emission checks once memory-view lowering lands.
- Add corpus regression gates:
  `audit:cuda-120` remains blocking;
  `audit:real-world-cuda` runs CUDA-120 plus LeetCUDA, NVIDIA `cuda-samples`,
  and `llm.c` as hard no-regression gates from pinned commits.
- Add before/after numbers to docs for every coherent chunk. A compatibility
  claim without corpus delta is not accepted.

Acceptance criteria for the first slice:

- Working tree has a PRD and corpus registry update before implementation.
- Real-world corpus audit can be run by
  `pnpm --filter @unlocalhosted/browsergrad-compiler audit:real-world-cuda`.
- The audit wrapper fetches or verifies pinned corpus commits before running.
- Gate output records stable corpus metadata: repo, commit, path, kernel count,
  WebGPU-runnable count, hard-gap count, error codes, and semantic families.
- `NVIDIA/cuda-samples` at `b7c5481` remains `357` total kernel definitions,
  `>=282` WebGPU-runnable, and `<=75` hard gaps.
- `karpathy/llm.c` at `f1e2ace` remains `148` total kernel definitions,
  `>=148` WebGPU-runnable, and `0` hard gaps.
- `xlite-dev/LeetCUDA` at `c5dde9a` remains `293` total kernel definitions,
  `>=264` WebGPU-runnable, and `<=29` hard gaps.
- Context isolation improves coverage without repo-specific branching and has
  unit tests.
- Intrinsic-ledger expansion improves coverage through generic CUDA math and
  scalar-half primitives without repo-specific branching.
- CUDA/C named constants such as `INFINITY`, `FLT_MAX`, `M_PI`, and runtime
  enum-style values lower through a shared analyzer/reference/WGSL registry.
- Initialized CUDA constant memory such as `__constant__ T x = ...` and
  unsized arrays such as `__constant__ short Q[] = {...}` lower as embedded
  read-only constants in parser, analyzer, CPU reference, WGSL, WebGPU input
  packing, and corpus gates.
- CUDA cache-hint builtins `__ldca`, `__ldcg`, `__ldcs`, `__ldcv`, `__ldg`,
  `__ldlu`, `__stcg`, `__stcs`, `__stwb`, and `__stwt` lower as plain storage
  pointer loads/stores in analyzer, CPU reference, and WGSL.
- CUDA integer atomics include `atomicAnd`, `atomicOr`, and `atomicXor` across
  analyzer validation, CPU reference semantics, WGSL emission, browser tests,
  and corpus gates.
- CUDA `double` is accepted only behind `f64Mode: "f32"` compatibility lowering,
  emits `f64-lowered-to-f32`, uses f32 storage/WGSL/reference ABI, and remains
  unsupported by default so labs cannot accidentally claim true f64 behavior.
- CUDA u8x4 sum-of-absolute-differences lowers through both `__usad4(a, b, c)`
  and `vabsdiff4.u32.u32.u32.add` inline PTX, with analyzer validation, CPU
  reference semantics, WGSL emission, and corpus gate evidence.
- CUDA vector storage types `float2/3/4`, `int2/3/4`, and `uint2/3/4` lower
  through one semantic vector ABI with scalar storage buffers, lane
  member access, `make_*` constructors, CPU reference, and WebGPU coverage.
- CUDA `bool*` pointer params use a u32-backed storage ABI with bool
  decode/encode at loads and stores, preserving browser host-shareable layout
  instead of emitting invalid WGSL storage `bool` arrays.
- Simple C++ intake supports scalar/vector `typedef` / `using` aliases,
  local quoted header context in audits, `constexpr` integer expressions in
  array dimensions and template arguments, CUDA `static` kernel qualifiers,
  late `__launch_bounds__` placement, and ignored `static_assert` checks.
- Cooperative-groups namespace calls parse and lower for `cg::sync(block)`,
  `cg::sync(grid)` runtime planning, and tile-scoped `cg::reduce(tile, value,
  cg::plus<T>{})` / `cg::greater<T>{}` forms.
- C++ pointer casts `reinterpret_cast<T*>`, `static_cast<T*>`, and
  `const_cast<T*>` parse into the existing CUDA-lite pointer-cast AST.
  Scalarized storage memory views such as `FLOAT4(x)` lower through parser,
  analyzer, CPU reference, and WGSL without treating repo macro names as
  special.
- Local typed pointer aliases such as `float4* p = reinterpret_cast<float4*>(&x[i])`
  lower as storage views without emitting fake pointer storage variables.
- Mutable local storage-pointer aliases such as
  `const float* p = input + i; p = other + j; out[i] = p[0];` lower through one
  pointer-handle model. Static aliases stay optimized away; mutable aliases
  track modeled storage buffer plus element base in parser/analyzer/reference
  and WGSL.
- Mutable CUDA pointer parameters such as `p += offset`, `p -= offset`, and
  `p++` lower through one pointer-base model in analyzer, CPU reference, WGSL
  emission, atomics/device-pointer argument paths, and corpus audit gates.
- Bounded integer template defaults on kernels and device helpers are preserved
  by corpus extraction and compile into integer constant expressions. This
  includes `template <const int N = ...>` defaults, functional scalar casts such
  as `float(i)`, and parser-safe named integer constants such as `warpSize`.
  Missing stage/padding/swizzle template values get conservative defaults only
  when the kernel body is already in a shared-memory pipeline context; general
  layout/object template params remain unresolved.
- CUDA default kernel parameter initializers such as `int *partial = NULL` are
  accepted by parser intake without changing required runtime bindings.
- Unresolved template type params used as pointer or reference parameters get a
  conservative audit instantiation to a scalar storage type such as `float`.
  Non-pointer layout/object params stay unresolved so CUTE-style layout
  compatibility is not faked.
- Corpus audit helper discovery follows template-aware device-helper closure:
  POD-return helpers, cooperative-group reference params, ordinary templated
  pointer helpers, and explicit template-id specializations such as
  `cast_value<float, float>` are included only through reusable source
  structure, never repo/file-name branching.
- CUDA signed index aliases such as `ptrdiff_t` lower through parser, layout,
  source normalization, reference execution, and WGSL as signed `i32` browser
  index scalars.
- CUDA fast math/bit intrinsic ledger includes `__saturatef`, `__fdividef`,
  `__frcp_rn`, `__expf`, `__logf`, `rsqrtf`, `__clz`, `__mul24`,
  `__umul24`, and `assert` with parser/analyzer, CPU reference, WGSL, and test
  coverage.
- CUDA bf16 intake includes `__nv_bfloat16`, `nv_bfloat16`, `__nv_bfloat162`,
  `__float2bfloat16*`, `__bfloat162float`, `__ushort_as_bfloat16`, and
  `__halves2bfloat162` with rounded reference semantics and WGSL f32 lowering.
  Packed `u32` bf16 storage ABI remains the next hardening gate before calling
  bf16 native-like.
- CUDA warp-reduction aliases include integer variants such as
  `warp_reduce_sum_i8_i32` and `warp_reduce_sum_i32_i32`, lowering through the
  same subgroup/reference path as existing float/half aliases.
- CUDA masked warp reduction intrinsic `__reduce_add_sync(mask, value)` lowers
  through analyzer feature gating, CPU reference, and WGSL `subgroupAdd(value)`.
- CUDA 2D float texture-object lowering maps `cudaTextureObject_t` params,
  scalar `tex2D<float>` calls, typed texture-vector reads such as
  `tex2D<float4>` / `tex2D<uchar4>`, and multi-channel WebGPU texture uploads
  to named WebGPU texture bindings with CPU-reference, WGSL, and browser test
  coverage.
- Source/context normalization stays generic: no repo-name, file-name, or
  assignment-name branching.
- At least one broad intrinsic gap from `llm.c` or LeetCUDA lands with parser,
  analyzer, reference, WGSL, and test coverage.
- CUDA-120 remains `240/240` WebGPU-runnable.

## Out of Scope

- Full CUDA C++ compatibility.
- Full CUTLASS/CuTe, `wmma`, WGMMA, tensor-core, or cooperative matrix support.
- Running PyTorch C++ extension host binding code in the browser.
- Native CUDA performance parity.
- Repo-specific transformations keyed on LeetCUDA, `llm.c`, or NVIDIA samples.
- Silent CPU fallback presented as GPU execution.
- Replacing BrowserGrad IR with LLVM/chipStar/clspv as the default path.

## Further Notes

The LeetCUDA result is useful precisely because it is bad today. It proves the
next frontier is CUDA dialect compatibility, not another hand-picked example.
The first slice should be small enough to ship with confidence, but the ladder
should stay ambitious: every unsupported CUDA idiom becomes a named primitive,
a clear diagnostic, or a deliberate future research lane.

References used in this PRD:

- https://github.com/xlite-dev/LeetCUDA
- https://github.com/NVIDIA/cuda-samples
- https://github.com/karpathy/llm.c
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/cuda-math-api/
- https://docs.nvidia.com/cutlass/
- https://www.w3.org/TR/WGSL/
- https://developer.chrome.com/blog/new-in-webgpu-134
- https://developer.chrome.com/blog/new-in-webgpu-146
