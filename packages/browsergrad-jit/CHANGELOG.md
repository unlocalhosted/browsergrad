# Changelog

All notable changes to `@unlocalhosted/browsergrad-jit` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/) per the [compatibility
contract in the README](README.md#compatibility-contract).

## [Unreleased]

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
