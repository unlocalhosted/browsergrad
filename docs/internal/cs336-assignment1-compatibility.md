# CS336 Assignment 1 Compatibility

Assessment date: 2026-06-21.

Upstream: <https://github.com/stanford-cs336/assignment1-basics>

## Verdict

Yes, CS336 Assignment 1 can be supported on this library family and platform, but not by running the upstream assignment repo unmodified inside the browser.

Recommended path:

1. Package the assignment as platform labs using `browsergrad-runtime` manifests and structured assertions.
2. Use `browsergrad-grad` as the immediate execution backend for A1 model and optimizer exercises.
3. Keep `browsergrad-jit` as the future accelerated backend after filling torch-alias gaps.
4. Port tokenizer/BPE tests as pure-Python platform rubrics instead of relying on native `tiktoken`, `psutil`, or Linux `resource` behavior in-browser.

## What The Upstream Assignment Expects

The upstream repository is a Python `uv` project. Its README instructs students to run:

```sh
uv run pytest
```

The `pyproject.toml` requires Python `>=3.12,<3.14` and dependencies including `torch~=2.11.0`, `tiktoken`, `psutil`, `pytest`, `regex`, `einops`, `einx`, `numpy`, and `wandb`.

The assignment exposes all student work through `tests/adapters.py`. The major adapter groups are:

- Transformer/model ops: linear, embedding, SwiGLU, RMSNorm, RoPE, scaled dot-product attention, multi-head attention, transformer block, transformer LM.
- Training utilities: batch sampling, softmax, cross entropy, gradient clipping, AdamW, cosine schedule.
- Serialization: save/load checkpoint.
- Tokenization: BPE tokenizer and BPE training.

## Fit By Area

| Area | Current Fit | Notes |
| --- | --- | --- |
| Linear, matmul, softmax, cross entropy, AdamW | Good | Present in `grad`; mostly present in `jit`. |
| Embedding | Good in `grad`, missing in `jit` | A1 needs token embeddings. |
| Transformer forward pass | Feasible in `grad` | `grad` has tensor indexing, reshape, permute, `where`, `triu`, `sin`, `cos`, `cat`, `stack`, and `Embedding`. |
| RoPE | Feasible in `grad` | Needs `sin`, `cos`, reshape/permute, and integer position indexing. |
| Gradient clipping | Mostly present in `grad` | Source has `clip_grad_norm_`; current dist probe did not expose `torch.nn.utils` as an attribute. |
| Serialization | Good in `grad`, missing in `jit` torch alias | `grad` exposes `torch.save` and `torch.load`. |
| Data batching | Feasible | Use NumPy plus tensor factory. Need explicit failure for CUDA devices in browser. |
| Tokenizer encode/decode | Feasible as pure Python | Avoid hard dependency on `tiktoken` in browser runner; use reference fixtures/rubric comparisons. |
| BPE training | Feasible but performance-sensitive | Browser timing may differ from native pytest. Use platform budgets calibrated for Pyodide. |
| Raw upstream pytest | Not browser-ready | Depends on native PyTorch and native/package APIs outside the current platform contract. |

## Probe Results

I ran a Pyodide symbol probe against built local dist packages.

`browsergrad-grad` torch alias exposed:

- Present: `torch.randn`, `torch.cat`, `torch.nn.Embedding`, `torch.nn.Linear`, `torch.nn.functional.softmax`, `torch.nn.functional.cross_entropy`, `torch.save`, `torch.load`.
- Missing from probe: `torch.rand`, `torch.randint`, `torch.clone`, `torch.allclose`, `torch.is_tensor`, `torch.nn.functional.silu`.
- `torch.nn.utils.clip_grad_norm_` exists in source intent, but the current dist probe did not expose it through `torch.nn.utils`.

`browsergrad-jit` torch alias exposed:

- Present: `torch.randn`, `torch.nn.Linear`, `torch.nn.functional.softmax`, `torch.nn.functional.cross_entropy`.
- Missing from probe: `torch.rand`, `torch.randint`, `torch.cat`, `torch.clone`, `torch.allclose`, `torch.is_tensor`, `torch.nn.Embedding`, `torch.nn.functional.silu`, `torch.save`, `torch.load`.

## Platform Packaging Plan

Use one lab manifest per assignment slice. A1 should not be one large all-or-nothing lab.

Suggested slices:

1. `cs336-a1-bpe-tokenizer`
2. `cs336-a1-bpe-training`
3. `cs336-a1-tensor-utils`
4. `cs336-a1-attention-rope`
5. `cs336-a1-transformer-block`
6. `cs336-a1-transformer-lm`
7. `cs336-a1-optimizer-checkpoint`

Each slice should include:

- `manifest.json` with `requires_browsergrad`, `required_ops`, `starter_path`, `reference_path`, and `rubric_path`.
- A starter adapter file matching the upstream adapter function names for that slice.
- A rubric that emits structured assertions through the runtime, not raw pytest output.
- Small bundled fixtures. Large datasets should be optional downloads or platform-managed assets.

## Missing Work Before A1 Feels Native

For `browsergrad-grad`:

- Add or expose `torch.rand`.
- Add or expose `torch.randint`.
- Add `torch.clone`.
- Add `torch.allclose`.
- Add `torch.is_tensor`.
- Expose `torch.nn.utils.clip_grad_norm_` reliably as both importable module and `torch.nn.utils` attribute.
- Ensure `torch.nn.functional.silu` is exported in the built package.

For `browsergrad-jit`:

- Add `nn.Embedding`.
- Add `torch.cat` and likely `stack`.
- Add `torch.rand`, `torch.randint`, `clone`, `allclose`, and `is_tensor`.
- Add or expose `F.silu`.
- Add checkpoint serialization parity or document that checkpoint labs use `grad`.
- Add transformer-friendly helpers only if they preserve the lazy IR contract.

For the platform:

- Build a small assignment ingestion layer that maps upstream adapter functions to platform lab files.
- Replace native pytest snapshots with structured assertion events.
- Replace native-only memory tests with browser/Pyodide calibrated checks.
- Keep `tiktoken` as an offline/reference dependency, not a browser runtime dependency.

## Practical Conclusion

This repo is already pointed in the right direction. PRD-013 explicitly names Stanford CS336 labs as a target, and the runtime already has the manifest and structured assertion pieces needed to host them. The shortest useful implementation is not a full PyTorch replacement. It is a CS336 A1 adapter pack over `browsergrad-grad`, plus a handful of torch-alias compatibility shims.
