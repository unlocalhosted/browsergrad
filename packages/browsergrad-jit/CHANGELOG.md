# Changelog

All notable changes to `@unlocalhosted/browsergrad-jit` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/) per the [compatibility
contract in the README](README.md#compatibility-contract).

## [Unreleased]

## [0.5.0] — 2026-05-26

PRD-009 v0 — gradient checkpointing via IR rewrite. Ships the
mechanism, not the perf win — on NumPy realizer the memory savings
are modest. The IR-rewrite pattern is load-bearing for PRD-012's
WGSL megakernels (where activation memory genuinely binds) and for
PRD-014's vmap (which needs a graph-rewrite backward).

### Added

- `bg.utils.checkpoint.checkpoint(fn, *args)` — PyTorch-shaped
  gradient checkpointing. The forward call's intermediate UOps are
  recorded as "interior to a region"; the backward pass rewrites any
  reference to those interior UOps into fresh clones rooted at the
  region's anchor inputs. The realizer re-computes the cloned
  subgraph independently of the original forward's value_table.
- `torch.utils.checkpoint.checkpoint` resolves to the same function
  via the `install_torch_alias()` shim.
- `_checkpoint.py` registry: `CheckpointRegion`, `_open_region`,
  `_close_region`, `apply_checkpoint_rewrite`. The rewrite reuses
  the same `_substitute`-style tree walk PRD-006 fusion uses.

### Refusal modes (deliberate v0 cuts per the review)

- `use_reentrant=True` — PyTorch's deprecated reentrant autograd.
  Refused with explanation.
- `preserve_rng_state=True` — Philox per-op seeds aren't captured
  yet. Deferred to PRD-007's dropout-decomposition follow-up.
- Nested `checkpoint(checkpoint(...))` — region-scoping isn't
  designed for nesting; refused with clear error.
- Any op in the region without a registered VJP rule — symbolic
  backward is the only safe checkpoint host; closure path can't be
  cleanly checkpointed.

### Deferred (PRD-009.2)

- Selective recompute policy (Korthikanti 2022).
- `checkpoint_sequential`, `checkpoint_wrap`.
- `bg.memory_stats()` instrumentation.
- `viz_regions` Graphviz emitter.
- Wall-time benchmarks (meaningful only on WGSL — PRD-012).

## [0.4.0] — 2026-05-26

PRD-008 v0 — safetensors I/O + trace cache. The WGSL pipeline cache
the original PRD scoped is explicitly deferred to PRD-012 (no WGSL
backend exists in jit yet); the OPFS HTTP cache layer is deferred to
PRD-008.2 once the runtime bridge for zero-copy ArrayBuffer transfer
is built. What ships here is the two highest-leverage pieces that
work against the substrate as-is.

### Added — safetensors

- `bg.load_safetensors(source, *, session=None, progress=None, dtype=None)`
  parses the HuggingFace safetensors format (8-byte header-length +
  JSON metadata + raw tensor data) and returns `dict[str, TensorProxy]`.
  Supports bytes, memoryview, and `file://` / local path sources.
- `bg.save_safetensors(tensors, path, *, metadata=None)` writes the
  reverse — round-trip-safe with the loader.
- All NumPy-supported dtypes (F32/F16, I64/I32/I16/I8, U64/U32/U16/U8,
  BOOL). BF16 raises `NotImplementedError` pointing at PRD-010.
- HTTP URLs raise `NotImplementedError` pointing at PRD-008.2 (the
  runtime bridge follow-up).
- Zero-copy where the source allows — `np.frombuffer` over memoryview
  slices, copy only on registration into the BufferTable.

### Added — trace cache

- In-memory cache keyed on `(id(module), training_flag, shape+dtype
  signature)`. Wired into `nn.Module.__call__` — first call traces
  and records, subsequent calls with matching signature rebuild zero
  UOps.
- Refuses to cache outputs with `_ctx` (closure backward would point
  at stale input proxies). For training graphs with learnable
  parameters this means the cache is effectively inference-only;
  PRD-014 (`torch.func` transforms) makes training-graph caching safe.
- `bg.jit.use_trace_cache(bool)`, `bg.jit.trace_cache_enabled()`,
  `bg.jit.trace_cache_stats()`, `bg.jit.clear_trace_cache()` control
  surface.
- `bg.cache_stats()` and `bg.clear_cache(scope='all'|'trace')` package-
  level aggregator API. The scope set is forward-compatible — future
  PRDs add `'opfs'` / `'pipelines'`.

### Deferred (explicit non-goals for this release)

- WGSL pipeline cache — moves to PRD-012 with the WGSL backend itself.
- OPFS HTTP blob cache — moves to PRD-008.2 with the JS↔Python
  zero-copy ArrayBuffer bridge.
- Trace cache for training graphs — requires PRD-014's graph-rewrite
  backward to make safe.
- BF16 dtype support — moves to PRD-010 (mixed precision).

## [0.3.0] — 2026-05-26

PRD-007 symbolic backward — Weeks 1+2 ship: VJP rule registry +
dispatcher that routes `.backward()` through the registry when all ops
on the chain have rules, with a closure fallback for unregistered ops.
The symbolic path runs with fusion ON (the workaround from PRD-006 only
applies to the closure fallback).

### Added

- IR opcodes `OP_SCATTER_ADD` and `OP_BROADCAST_TO` (total: 27).
  `SCATTER_ADD`'s NumPy realizer uses `np.add.at` — deterministic by
  construction. PRD-012's future WGSL lowering must preserve
  determinism (sort-and-segment-reduce default). `BROADCAST_TO` makes
  implicit shape extension explicit in the IR — needed because the
  metadata-shape contract doesn't coerce NumPy op outputs.
- `TensorProxy.backward()` dispatcher: when every UOp on the path has
  a registered VJP rule, runs the symbolic path (build backward IR,
  realize once with fusion ON). Otherwise falls back to the legacy
  closure path (with fusion forced off, as PRD-006 requires).
- `_vjp.py`: VJP rule registry (`register_vjp` decorator, `get_rule`,
  `list_registered`) with 11 rules for: `ADD`, `MUL`, `DIV`, `NEG`,
  `EXP`, `LOG`, `MATMUL`, `REDUCE` (sum/mean), `RESHAPE`, `PERMUTE`,
  `CAST`. `REDUCE(max/min/argmax/argmin)` and the rest defer to PRD-007
  W2/W3.
- Every VJP-emitted UOp carries `arg["vjp_of"] = <forward_uop>` so
  PRD-009 can identify recompute candidates without inspecting closures.
- EXP VJP references the forward output directly (`dy * y`, not
  `dy * exp(x)`) — the Flash-Attention-v2 reuse pattern that lets
  PRD-006/012 fuse the forward EXP and backward MUL into one kernel
  with the EXP value resident in shared memory.
- `TensorProxy.backward(loss_scale=1.0)` argument — PRD-010 mixed
  precision's GradScaler will pass `2**16` here to keep fp16
  backward gradients in the representable range.

### Internal

- `_unbroadcast_uop` helper produces the IR sequence that sums dy back
  to a target shape over broadcast-extended axes.
- `_broadcast_batch_shape` for MATMUL VJP's batched broadcasting logic.

## [0.2.0] — 2026-05-26

PRD-006 kernel fusion (NumPy realizer scope). The fusion pass is on by
default; toggle with `bg.jit.use_fusion(False)` or `BG_DISABLE_FUSION=1`.
WGSL fusion stays in PRD-012; PRD-006 v0 establishes the graph-rewrite
mechanism on the NumPy backend so PRD-012 is a backend swap, not a new
compiler. 77 tests green (8 unit + 69 integration); zero regressions.

### Added

- IR opcodes `OP_FUSED_ELEMENTWISE` and `OP_FUSED_SOFTMAX` (total: 25).
- `_fusion.py` graph-rewrite pass with two matchers:
  - Elementwise-chain matcher absorbing linear sequences of
    `{ADD, MUL, DIV, NEG, EXP, LOG}` of length ≥ 2 with matching
    shape/dtype and single-consumer intermediates.
  - Softmax DAG matcher absorbing the canonical 6-node template
    (REDUCE(max,keepdims) → NEG → ADD → EXP → REDUCE(sum,keepdims) → DIV)
    with the EXP's two-consumer (diamond) structure.
- `_realize.py` handlers for both fused opcodes — the fused-elementwise
  handler eliminates intermediate ndarray allocations between chain
  steps; the fused-softmax handler runs three NumPy calls instead of six.
- `bg.jit` namespace with `use_fusion(bool)`, `fusion_enabled()`,
  `debug_fused_kernels()`, `debug_unfused_reasons()`.
- `_fusion_config.py` with the `BG_DISABLE_FUSION` env-var override.
- Autograd-safety mechanism: backward's intermediate-value pre-pass runs
  with fusion disabled, so closures that capture original UOps by
  identity continue to find them by id in the value cache. PRD-007's
  symbolic backward will lift this restriction.

### Fixed

- `nn.Linear` now uses `np.random.uniform` (legacy global API) instead
  of `np.random.default_rng().uniform` so `bg.manual_seed()` is
  respected for parameter init. The previous behavior silently broke
  deterministic-init expectations; surfaced by the PRD-006
  fusion-on-vs-off cross-test.

### Internal

- Two-pass design: softmax first (DAG), elementwise second (chain).
  Order matters — softmax owns nodes the chain matcher would otherwise
  greedily absorb, producing a worse fusion overall.
- Per-trace introspection (`FusionReport`) tracks fused groups + matcher
  rejections with reasons, enabling "why didn't my softmax fuse?"
  debugging without external profiling.

## [0.1.0] — 2026-05-26

First public release of the JIT epoch. PRD-005 minimum-viable scope:
elementwise + Linear MLP tier. Conv/RNN/Attention land in 0.1.x patches.

### Added — IR foundation

- 23-opcode UOp IR (`_ir.py`). Opcodes: `BUFFER`, `LOAD`, `STORE`, `CONST`,
  `RANDOM`, `CAST`, `ADD`, `MUL`, `DIV`, `NEG`, `EXP`, `LOG`, `CMP`,
  `MATMUL`, `REDUCE`, `RESHAPE`, `PERMUTE`, `SLICE`, `PAD`, `WHERE`,
  `INDEX`, `MASK`, `CUSTOM`.
- Frozen-dataclass UOp nodes with cached structural hash; shape-validated
  construction; iterative `toposort()` that handles 5000-deep graphs
  without Python recursion limits.
- 2^30-element ceiling on any single tensor; refuses pathological shapes
  at IR construction time.

### Added — Runtime + tensor surface

- `TensorProxy` (alias: `Tensor`) — the lazy tensor. Metadata access
  (`.shape`, `.dtype`, `.ndim`, `len`, `__repr__`, `size`, `numel`) never
  realizes. `.data` and `__array__` raise to enforce explicit realization.
- Realization triggers: `.numpy()`, `.tolist()`, `.item()`, `.peek()`,
  `.backward()`, `__bool__`, `__float__`, `__int__`, `__iter__`.
- Arithmetic surface: `+`, `-`, `*`, `/`, unary `-`, `@`, comparison
  dunders (`==`, `!=`, `<`, `<=`, `>`, `>=`). All decompose to IR ops
  with broadcasting + dtype promotion via NumPy's rules.
- Reductions: `.sum()`, `.mean()`, `.max()`, `.min()`, `.argmax()` with
  axis + keepdims.
- Shape ops: `.reshape()`, `.view()`, `.transpose()`, `.permute()`, `.T`.
- dtype casts: `.cast()`, `.float()`, `.long()`, `.bool()`.
- Factories: `tensor`, `from_numpy`, `zeros`, `ones`, `randn`, `arange`.
- `Session` + `new_session()` / `get_default_session()` /
  `set_default_session()` for per-loop isolation.
- `manual_seed` for deterministic factories.

### Added — Realizer

- `_realize.py`: NumPy dispatch table covering all 23 opcodes. One
  topological walk + one dispatch-table call per node.

### Added — Autograd

- Closure-based backward via `_BackwardCtx`. VJP rules for every op in
  the public surface: add, mul, div, neg, exp, log, matmul, sum, mean,
  reshape, permute, cast, where, plus cross_entropy and mse_loss in
  `_functional`.
- `.backward()` walks the proxy DAG, accumulates gradients per leaf
  parameter. Per-PyTorch semantics: gradients accumulate across calls.
- Defensive design: the collect-proxies walk and the accumulation map
  are separated to avoid the "root-skip" bug class.

### Added — `nn` module

- `nn.Module` with auto-registered Parameters and submodules,
  `parameters()` / `named_parameters()`, `train()` / `eval()`,
  `state_dict()` / `load_state_dict()`, `zero_grad()`.
- `nn.Linear`, `nn.Sequential`, `nn.Dropout`.
- Activation modules: `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.GELU`,
  `nn.Softmax`.
- Loss modules: `nn.MSELoss`, `nn.CrossEntropyLoss`, `nn.NLLLoss`.

### Added — `nn.functional`

- `F.relu`, `F.sigmoid`, `F.tanh`, `F.gelu` (tanh approx).
- `F.softmax`, `F.log_softmax`.
- `F.cross_entropy`, `F.mse_loss`, `F.nll_loss`.
- `F.linear`, `F.dropout`.

### Added — `optim`

- `optim.SGD` with momentum + weight_decay.
- `optim.Adam` with standard PyTorch defaults.
- `optim.AdamW` with decoupled weight decay.

### Added — Torch alias

- `install_torch_alias()` / `uninstall_torch_alias()` with the owner-token
  protocol per PRD-005 critique P1-2. Refuses to shadow another package's
  torch namespace without explicit `force=True`. Registers `torch.nn`,
  `torch.nn.functional`, and `torch.optim` so existing PyTorch code runs
  against the supported surface.

### Conformance

- 9 unit tests, 64 integration tests against real Pyodide-in-Node — all
  green. Conformance covers IR construction, BufferTable lifecycle,
  arithmetic, Python protocol realization, autograd correctness, full
  forward+backward+optimizer training loops (MSE regression converges,
  cross-entropy 2-class MLP converges to >85% accuracy), and the torch
  alias protocol including the conflict path.

### Internal

- `_ir`, `_buffer_table`, `_tensor_proxy`, `_realize`, `_nn`,
  `_functional`, `_optim`, `_torch_compat` are explicitly internal
  modules (leading underscore). Their opcode strings, attribute names,
  and helper functions can change in any minor release.

## [0.1.0-pre.1] — 2026-05-26

PRD-005 Week 1 scaffolding: IR + BufferTable + TensorProxy stub. See
git history for the pre-release commit.
