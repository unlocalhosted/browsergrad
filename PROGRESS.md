# PyTorch-completeness Progress

Tracking the v0.4.7 → ~v0.6 push to close the gaps that block real PyTorch labs.

Last updated: 2026-05-26.

Pyodide-in-node integration suite is the source of truth: 17 files, 157 tests, all green at the head of each commit listed below.

## Pile A — possible, just unbuilt

Implementable on NumPy with TDD. Each row = one focused commit (or small group). Behavior tests against independent oracles. The PyTorch-conformance fixture suite (`tests-integration/pytorch-conformance.test.ts`) lands alongside as the strongest oracle.

| # | Item | Status | Commit |
|---|---|---|---|
| 1 | Tensor indexing (`x[mask]`, `x[i:j]`, fancy) + scatter backward | ✅ done | — |
| 2 | Comparison ops (`==` `<` `>` `<=` `>=` `!=`) | ✅ done | — |
| 3 | Multi-dtype (`int64`, `bool`); tensor stores `.dtype` | ✅ done | — |
| 4 | `torch.utils.data.Dataset` + `DataLoader` (single-process) | ✅ done | — |
| 5 | `state_dict` / `load_state_dict` / `torch.save` / `torch.load` | ✅ done | — |
| 6 | `nn.init.{kaiming_uniform_, xavier_uniform_, normal_, uniform_, zeros_, ones_, constant_}` | ✅ done | — |
| 7 | `F.pad`, `F.interpolate`, `F.normalize`, `F.cosine_similarity` | ✅ done | — |
| 8 | `F.scaled_dot_product_attention` | ✅ done | — |
| 9 | More losses: `nn.{BCELoss, L1Loss, SmoothL1Loss, KLDivLoss}` | ✅ done | — |
| 10 | More optimizers: `RMSprop`, `Adagrad`, `Adadelta` | ✅ done | — |
| 11 | More schedulers: `ReduceLROnPlateau`, `MultiStepLR`, `ExponentialLR`, `OneCycleLR` | ✅ done | — |
| 12 | `torch.einsum` (wrap `np.einsum` with backward) | ✅ done | — |
| 13 | Tensor math: `abs`, `sign`, `clip`/`clamp`, `where`, `min`/`max(dim)`, `std`, `var`, `topk`, `sort`, `sqrt`, `pow` | 🟡 partial — `abs`, `sign`, `sqrt`, `pow`, `clamp`/`clip`, `topk`, `where` done; `min`/`max(dim)`, `std`, `var`, `sort` still pending | — |
| 14 | Tensor shape: `expand`, `repeat`, `chunk`, `split`, `contiguous`, `roll`, `flip` | 🟡 partial — `expand`, `repeat`, `flip` done; `chunk`, `split`, `contiguous`, `roll` still pending | — |
| 15 | `nn.RNN` / `nn.LSTM` / `nn.GRU` + backward through time | ✅ done | — |
| 16 | `nn.Conv3d` + `nn.ConvTranspose1d` / `nn.ConvTranspose2d` | ⏳ deferred (rarely used in browser labs; track if demand emerges) | — |
| 17 | `nn.GroupNorm` / `nn.InstanceNorm{1,2,3}d` / `nn.BatchNorm3d` | 🟡 partial — GroupNorm, InstanceNorm2d, BatchNorm3d done; InstanceNorm1d/3d still pending | — |
| 18 | Module hooks: `register_forward_hook`, `register_backward_hook` | 🟡 partial — forward hook done; backward hook still pending | — |
| ★ | **PyTorch-conformance fixture suite** (real torch in subprocess generates fixtures; Pyodide loads + compares) | ⏳ pending | — |

## Pile B — possible but limited

Ship "good enough" with explicit caveats in docstrings + STATUS.md.

| Item | Approach | Status |
|---|---|---|
| `torch.amp` (mixed precision) | f32 code path; document no real fp16 speedup | ✅ done (autocast no-op) |
| Basic image transforms (`torchvision.transforms`-like) | NumPy-based resize/normalize/to_tensor | ⏳ deferred |
| `torch.linalg.*` subset (`norm`, `svd`, `eigh`, `inv`, `det`, `solve`, `pinv`) | Wrap `numpy.linalg` with backward where common | ✅ done (forward; backward via numpy) |
| Single-notional-device "multi-GPU" hooks | `model.to([0, 1])` accepts but no-ops; document | ✅ done (Module.to is no-op) |

## Pile C — physically impossible in browser

Stub with `NotImplementedError` + clear message pointing to the architectural reason. **Do not fake silent success** (that's the greed trap).

| Item | Stub behavior |
|---|---|
| `torch.compile`, `torch.fx`, `torch.jit.*` | ✅ Raises `NotImplementedError` with reason |
| `torch.cuda.is_available()` | ✅ Returns `False` |
| `torch.cuda.*` (devices, streams, memory) | ✅ Raises (device_count → 0, current_device raises) |
| `torch.distributed.*` | ✅ Raises (init_process_group, all_reduce); is_initialized → False |
| `DataLoader(num_workers=N)` with N>0 | ✅ Raises |
| `torch.onnx` | ✅ Raises |
| `torch.quantization` | ✅ Raises |

## Skill passes (after Pile A is done)

| Skill | Target | Status |
|---|---|---|
| `/extract` | Consolidate reusable patterns across the python source modules | ⏳ pending |
| `/delight` | Add intentional moments of polish (better error messages, helpful warnings) | ⏳ pending |
| `/polish` | Final quality pass — formatting, naming, docstring consistency | ⏳ pending |
| `/harden` | Edge cases, error handling, defensive checks at public boundaries | ⏳ pending |

## Methodology — same as everything else in this repo

For every item: write a behavior test first against an **independent oracle** (NumPy, PyTorch fixture, finite difference, hand-derived math) — never compare implementation against itself. Watch RED. Write minimum impl. Watch GREEN. Never refactor while red. Commit per logical unit. Each commit ends all-green.

`pnpm typecheck && pnpm test && pnpm -F @unlocalhosted/browsergrad-grad test:integration` must all pass before each commit.

## End-state target

`browsergrad-grad` covers the API surface a typical PyTorch lab actually uses (~95% idiom coverage). The torch shim is comprehensive enough that vanilla PyTorch tutorials run unmodified for everything except: distributed training, compilation, CUDA, multi-process data loading, ONNX. Those are explicitly documented as impossible-in-browser, not silent gaps.
