# Browsergrad PRD: 12-Month Feature Plan

**Status**: Draft v1, post-v0.5.0, May 2026.
**Authors**: Library maintainers + research synthesized in May 2026.
**Companion docs**: [README.md](README.md) (what it is) · [PRIMER.md](PRIMER.md) (background for readers new to ML) · [ARCHITECTURE.md](ARCHITECTURE.md) (current decisions) · [VISION.md](VISION.md) (the JIT epoch) · [PROGRESS.md](PROGRESS.md) (PyTorch coverage).

This document plans the next 12 months of browsergrad work using market and technical evidence collected in May 2026. Every claim is cited; every priority is justified by something we can point to. If a fact isn't sourced, the document marks it as such.

---

## 1. Executive summary

Browsergrad is a PyTorch-shaped Python library that runs entirely in the browser via Pyodide and NumPy. v0.5.0 covers ~95% of the educational PyTorch surface, has a real PyTorch-conformance test suite (forward + backward within 1e-4 on five ops vs. torch 2.12.0), 234 integration tests, and ships under MIT.

Four findings from May 2026 research reshape priorities for the next 12 months:

1. **The market slot we occupy is empty.** No published library combines Pyodide + a PyTorch-shaped Python API + WebGPU acceleration. Transformers.js (Apache-2.0) is JavaScript-only and pipeline-oriented. tinygrad has a WebGPU runtime but no browser packaging or demo. WebLLM is AOT-compiled via TVM and WebGPU-only. There is no PyTorch-in-Pyodide library — `pytorch` itself has been blocked on Pyodide for 4+ years ([pyodide/pyodide#1625](https://github.com/pyodide/pyodide/issues/1625)).

2. **Our one credible competitor is `greed`**, the engine behind deep-ml.com (Next.js + Pyodide + GreedJS + WebGPU compute shaders, 100% client-side). Greed's problem catalog ([Open-Deep-ML/DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem)) is something we can directly target as a coverage proof. We already cover every op in their sampled problems except `softsign` and `gradient_checkpoint`.

3. **WebGPU just crossed Baseline (January 2026)** per [web.dev](https://web.dev/blog/webgpu-supported-major-browsers); WebNN is W3C Candidate Recommendation but Chrome ships it only as Origin Trial in Chrome 146 (realistic GA: **2027**). So **WebGPU is the substrate to bet on now; WebNN is a 2027 lane to design behind a dispatcher seam but not ship in v1**.

4. **Performance ceiling is real and measurable.** Hand-optimized WGSL matmul reaches ~17% of FP32 peak on Apple M2; native cuBLAS gets ~75%. **WebGPU ≈ ¼ of native CUDA throughput in best-case published work** ([nuss-and-bolts.com](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)). Cited LLM throughputs: 21 tok/s for 0.5B params in-browser, 85 tok/s for Llama 3.2 in-browser via ONNX. Anyone claiming "WebGPU matches CUDA" is fabricating; anyone claiming "browsers can't do real ML" is also wrong. **The truth is ¼ of native, which is fine for ≤3B-param educational labs.**

These facts shape the roadmap:

- **P0 (90 days)**: Close the small remaining PyTorch gaps, wire the kernels package as a real GPU backend for matmul-heavy ops, cut cold-start time, validate end-to-end on one real lab.
- **P1 (6 months)**: Build the tracing JIT + IR + first fusion passes. The architectural foundation for everything beyond eager mode.
- **P2 (12 months)**: Full megakernel codegen, WebNN second-tier backend, lab platform alignment with craftingattention.
- **Explicit non-goals**: Distributed training, mixed precision speedups, torch.compile equivalence, ONNX export, sparse tensors. None appear in surveyed educational curricula.

---

## 2. Where we are today (current state, briefly)

Current library footprint:

- **`@unlocalhosted/browsergrad-grad`** v0.5.0 — PyTorch-shaped Python API. All of nn.Linear/Conv*/Norm/Pool/Drop/Embedding/Sequential/MHA/RNN/LSTM/GRU/Loss-modules; full autograd; optim with SGD/Adam/AdamW/RMSprop/Adagrad/Adadelta; schedulers; `torch.utils.data.DataLoader`; state_dict + torch.save/load; multi-dtype; einsum; `torch.linalg.{norm,inv,svd,...}` via `Pile B`; loud `NotImplementedError` for compile/fx/jit/cuda/distributed/onnx/quantization (Pile C). Source in 12 chunked `.py` files for `nn.py`, codegen-emitted into a single Python module for Pyodide.
- **`@unlocalhosted/browsergrad-kernels`** — six WGSL kernels (matmul, softmax, layernorm, activations, attention, plus JS reference impls). **Not wired to grad yet** — exists as a future fast-path option.
- **`@unlocalhosted/browsergrad-runtime`** — Pyodide-in-Worker host with assertion + artifact relay protocols, cooperative cancellation, structured exec results.
- **Test substrate**: 234 integration tests (28 files) against real Pyodide-in-Node; 25 unit tests; 5 PyTorch-conformance fixtures generated from real torch 2.12.0 verified within 1e-4 on Linear/CrossEntropy/LayerNorm/Softmax/ReLU.
- **Ships**: 27 commits at HEAD, public on GitHub, pushed to `github.com/unlocalhosted/browsergrad`.

**What it doesn't yet do**:

- Touch the GPU at all. Every op runs on CPU via NumPy-in-Pyodide.
- Compile or fuse. Eager Python interpretation per op.
- Cache anything between page loads.
- Have a curriculum or lab UI (that's craftingattention's job, but the runtime needs hooks).

---

## 3. The opportunity, fact-anchored

### 3.1 The empty market slot

| Library | Has Python API? | Has WebGPU? | Browser-packaged? | Status |
|---|---|---|---|---|
| Transformers.js | No (JS only) | Yes (via ORT-WebGPU) | Yes | v4.2.0 active |
| ONNX Runtime Web | No (JS only) | Yes (since 1.17, Feb 2024) | Yes | active |
| TensorFlow.js | No | Experimental | Yes | last release Oct 2024 — stale |
| WebLLM / MLC-LLM | No | Required (only) | Yes | active, AOT-compiled via TVM |
| tinygrad | Yes (PyTorch-like) | Yes (`ops_webgpu.py`) | **No** (no browser bundle) | active |
| Pyodide + NumPy | Yes | No (CPU only) | Yes | active |
| PyTorch (real) | Yes | No (blocked on Pyodide #1625) | No | impossible |
| **Browsergrad** | **Yes** | Planned via kernels | **Yes** | **v0.5.0** |
| **`greed` (deep-ml.com)** | Yes | Yes (WebGPU shaders) | Yes (deep-ml.com only) | active but proprietary to one site |

Citations: [Transformers.js GitHub](https://github.com/huggingface/transformers.js); [ORT Web docs](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html); [TF.js releases](https://github.com/tensorflow/tfjs/releases); [WebLLM](https://github.com/mlc-ai/web-llm); [tinygrad](https://github.com/tinygrad/tinygrad); [Pyodide #1625](https://github.com/pyodide/pyodide/issues/1625); [Open-Deep-ML/DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem).

**The slot we own**: **Python + PyTorch-shaped API + WebGPU + open-source + browser-packaged.** That cell is empty except for `greed`, which is site-coupled to deep-ml.com and not positioned as a general library.

### 3.2 Greed as the named incumbent

Greed powers deep-ml.com's "LeetCode-for-ML" problem grinder. Their codebase is a custom PyTorch execution engine + WebGPU compute shaders. Sampling 14 problems from the `DML-OpenProblem` repo:

- `1_matrix-vector-dot-product`, `2_transpose-of-a-matrix`, `12_singular-value-decomposition-svd`
- `15_linear-regression-using-gradient-descent`, `22_sigmoid-activation-function`, `23_softmax`
- `25_single-neuron-with-backpropagation`, `40_implementing-a-custom-dense-layer`
- `104_binary-classification-with-logistic-regression`, `107_implement-masked-self-attention`
- `115_implement-batch-normalization-bchw`, `130_simple-cnn-training-function`
- `145_adagrad-optimizer`, `188_gradient-checkpointing`

Cross-referenced against browsergrad's coverage:

| Greed needs | browsergrad has |
|---|---|
| `torch.tensor`, matmul/transpose/reshape | ✅ |
| `nn.Linear`, `nn.Module` | ✅ |
| sigmoid, softmax | ✅ (softsign would need adding) |
| `nn.Conv2d`, `BatchNorm` | ✅ |
| masked self-attention | ✅ (`F.scaled_dot_product_attention` with mask) |
| SGD, Adagrad, Adam | ✅ |
| `loss.backward()` | ✅ |
| SVD | ✅ (`torch.linalg.svd` via Pile B) |
| simple CNN training function | ✅ |
| gradient checkpointing | ❌ (P0.1) |
| softsign | ❌ (P0.1, trivial) |

**Coverage gap to greed parity: 2 features.** Performance gap to greed: unknown without head-to-head benchmark (P0.5).

### 3.3 Pedagogy gap — the real differentiator

Per the education research, the platform landscape is:

- **Deep-ml** — code editor + run + test cases. "LeetCode for ML." Problem grinding model.
- **fast.ai / Coursera / DeepLearning.AI** — server-hosted notebooks. Real GPU but no zero-install story.
- **Brilliant.org** — beautiful interactive viz, but no actual code editor or training.
- **ml5.js / Teachable Machine** — JS-only, hobbyist-flavored, no Python.

The gap (citing the research dossier verbatim): *"A pedagogically-structured, narrative-driven, zero-install, Python-PyTorch deep learning curriculum that runs entirely in the browser. Deep-ml is closer to LeetCode-for-ML (problem grinding); the opening is for a 'Brilliant.org for actual PyTorch code' — guided lessons, scaffolded mini-projects, attention/transformer interactives — built on a browsergrad runtime that's not locked to one site."*

**Positioning**:
- **Browsergrad** = the open, embeddable PyTorch runtime (MIT, runs anywhere)
- **Craftingattention** = the pedagogy layer (the "Brilliant for PyTorch")
- **Deep-ml** = LeetCode segment (different audience)

### 3.4 The WebGPU window

WebGPU crossed Baseline in **January 2026** ([web.dev](https://web.dev/blog/webgpu-supported-major-browsers)):
- Chrome/Edge stable since v113 (April 2023); Chrome Android since v121 (Jan 2024)
- Safari 26.0 (Sept 2025) on macOS/iOS/iPadOS/visionOS
- Firefox 141 on Windows (July 2025); Firefox 145 on macOS Tahoe (Apple Silicon only)
- Subgroup ops stable in Chrome 144+

Mobile gaps that affect us: iOS <26 has no WebGPU, Android <12 has no WebGPU. Plan for older-device fallback to WASM SIMD.

### 3.5 WebNN is 2027, not 2026

W3C Candidate Recommendation January 2026; Chrome 146 ships it as Origin Trial only (`chrome://flags`). The community timeline (Chrome/Intel/MS posts) targets GA in **2027** ([Frontier Web APIs 2026](https://www.utsubo.com/blog/frontier-web-apis-2026-production-ready)).

**Implication**: design the dispatcher with WebNN as a future tier (Tier 1 in priority when available), but don't make v1 product depend on it. WebNN is a 2027 unlock, not a 2026 dependency.

### 3.6 The real performance ceiling

From [nuss-and-bolts.com on WGSL matmul optimization](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel):
- WebGPU matmul on Apple M2: ~**17% of FP32 peak** with hand-optimized WGSL (unoptimized: 1–2%)
- Native cuBLAS on similar non-tensor-core path: ~**75% of peak**
- **So WebGPU ≈ ¼ of native for matmul in the best-case published kernel.**

Browser LLM benchmarks ([SitePoint WebGPU vs WebGL inference](https://www.sitepoint.com/webgpu-vs-webgl-inference-benchmarks/)):
- torch-webgpu on RTX 5090 / Dawn: **21 tok/s (0.5B params)**, **17.9 tok/s (1.5B params)**
- Llama 3.2 via ONNX in-browser: **85 tok/s**
- Native llama.cpp on RTX 4090: ~**135–150 tok/s for 7B**

**Defensible performance claims for the PRD**:
- "~¼ of native CUDA throughput on matmul-heavy paths"
- "Sub-3-second forward pass for ResNet18-scale models on consumer hardware"
- "Practical for educational models up to ~3B parameters"
- "Order-of-magnitude faster than the current NumPy-on-CPU path"

**Indefensible claims to avoid**: anything implying parity with native CUDA; any specific multiplier vs greed (we have no head-to-head benchmark yet).

---

## 3.7 Target curricula (the labs we want to host)

The library exists to power lab implementations. Three curricula define the "viability bar":

**(A) deep-ml.com problem catalog** — covered above (§3.2). 2 features short of parity (softsign, gradient_checkpoint).

**(B) fast.ai Part 2** — 17 chapters spanning matmul-from-scratch through latent diffusion. After P0.1 ships, **15 of 17 chapters run** ([fast.ai Part 2](https://course.fast.ai/Lessons/part2.html)). The two exceptions:
- Ch 20 (Mixed Precision) is conceptually broken in our environment — we deliberately don't ship real fp16 speedup. Lab needs framing as "what mixed precision is and why real training uses it," not as "see your model train 2× faster."
- Ch 9, 10, 25 (Stable Diffusion / Latent Diffusion) work technically but require a pretrained-weights loading story (~5GB of SD weights via OPFS streaming, or a tiny educational SD variant we curate).

**(C) Stanford CS336 + Karpathy Zero-to-Hero** — every *from-scratch* implementation chapter runs today or after P0.1: micrograd ✅, makemore ✅, "Let's build GPT" ✅, BPE tokenizer ✅, transformer-from-scratch ✅, training small models ✅, RLHF math ✅, MoE ✅, KV cache ✅. Out of scope: reproducing GPT-2 at full 1.5B scale, distributed training, quantization — these are inherently hosted-GPU concerns.

**Implication for the runtime**: the curricula don't require any *new categories* of feature beyond what's already in P0/P1/P2. They do confirm specific priorities:
- ConvTranspose1d/2d is load-bearing for fast.ai Ch 15/19/21/22/23/25.
- A `.safetensors` streaming loader through OPFS is needed for any pre-trained-model lab (added as P1.6 below).
- A "tiny pretrained model gallery" is content-work, not runtime-work, but the runtime needs the loader to exist.

## 3.8 Design principles (locked-in)

These are non-negotiable in the next 12 months. Every feature spec below assumes them.

### Performance by default

The fast path is the default path. There is no opt-in for the JIT, no `@torch.compile` wrapping, no "developer mode" flag a user has to discover.

- **JIT tracing is automatic.** First call to `nn.Module.forward()` traces; subsequent calls dispatch from the cached plan.
- **Backend auto-selection.** For every kernel, the dispatcher picks the highest-performance available backend silently: WebNN if available → fused WGSL megakernel if a fusion pattern matches → primitive WGSL → WASM SIMD → NumPy. No user choice required.
- **Eager mode is a debug fallback**, not a default. `browsergrad.use_eager(True)` exists for printf-debugging and for the correctness oracle, not for users.
- **Cold-start is a first-class metric.** Service worker pre-warm + OPFS pipeline cache + lazy module loading are all in scope from day one, not "later optimizations."

### Security by default

The browser is a hostile environment. We assume an adversarial page somewhere will try to exploit any opening. Every primitive is designed assuming this.

- **Pyodide sandbox is the trust boundary.** Python code can't reach outside the sandbox to other tabs, the user's filesystem, or arbitrary network.
- **Network access is opt-in by host.** Python `bg.fetch(url)` calls into a JS-side adapter that the host page provides. Default adapter allowlists nothing; the host page must explicitly scope (origin allowlist, max bytes, timeout). No surprise `urllib`/`requests` calls.
- **OPFS is mediated.** No raw `FileSystemSyncAccessHandle` exposed to Python. Storage goes through `bg.checkpoint(name)` / `bg.load_checkpoint(name)` with explicit per-origin namespacing.
- **WGSL pipelines compile in isolated workers.** User code never holds a direct `GPUDevice` handle. The host page can revoke the device cleanly.
- **No `eval` or `exec` of arbitrary strings from JavaScript** unless the host explicitly opts in via a `bg.allow_dynamic_exec()` hook.
- **Cross-origin isolation is opt-in with graceful degradation.** SAB-requiring features (multi-threaded WASM) detect the COOP/COEP headers; when absent, the library degrades to single-threaded mode silently rather than crashing.
- **WGSL inputs are size-validated.** Kernel dispatch with oversized buffers (> declared limits) refuses at the dispatcher rather than risking GPU device loss.

## 3.9 Package boundary: `browsergrad-jit` is the new package

Locked-in: the JIT epoch ships as **`@unlocalhosted/browsergrad-jit`**, a new package, not a mutation of `@unlocalhosted/browsergrad-grad`.

Reasons:
- The current grad library stays as the **correctness oracle**. Every IR-emitted result is cross-checked against the eager NumPy path during development. We can't do that if the eager path stops existing.
- Migration is reversible. If the JIT path has a bug, users (and craftingattention) can pin to `browsergrad-grad@0.5.x` and keep shipping.
- The packages have different perf characteristics, license footprints (jit may add WGSL-runtime deps), and update cadences. Keeping them separate lets each evolve independently.
- `browsergrad-grad` becomes the "minimal-deps, slowest, most-reliable" tier. `browsergrad-jit` becomes the "default fast path" once it matures.

The migration path follows the schedule in §7 (P1.1 onward).

## 4. Goals and non-goals

### Goals (in priority order)

1. **Run the deep-ml problem catalog with no modifications and demonstrably faster than the current NumPy path.** This is the proxy goal because it's the only public benchmark suite we have access to that targets our exact niche.
2. **Make craftingattention's first 5 labs feel instant** (<3s training step at the dataset sizes used in education). This is the user-facing goal.
3. **Establish the JIT architecture** so the next 5 years of perf work has a foundation.
4. **Be the open alternative** to greed so other education platforms can adopt our runtime without writing their own.

### Non-goals (explicit, with justification)

1. **Distributed training.** Zero appearances in CS231n A1-A2, fast.ai ch.1-7, nanoGPT.
2. **Mixed precision speedup.** WebGPU's fp16 support is uneven; autocast stays a no-op (Pile B).
3. **torch.compile compatibility.** Our JIT will be parallel, not a port of theirs.
4. **ONNX export.** Different use case (deployment vs training/learning).
5. **Custom CUDA kernels.** No CUDA in browser.
6. **Models > 3B parameters.** Beyond our hardware reality. Lab models are 100K-1B.
7. **Production training pipelines.** Save real training for hosted GPUs; we own the *learning* and *prototyping* segment.

---

## 5. Users and use cases

### Primary user

A student or self-learner doing a deep learning course. They land on a craftingattention lesson, write PyTorch-shaped Python, watch a model train, get feedback. They have a 4-year-old laptop. They use Chrome or Safari. They've never installed CUDA and never will.

### Secondary user

A course author building an interactive lesson on top of browsergrad+craftingattention. They want predictable performance, debuggable failures, a stable PyTorch API.

### Tertiary user

A developer at another education platform who wants to embed browsergrad to power their own labs. They want the runtime to be open, MIT-licensed, and not tied to any one frontend.

### Not our user (explicit)

- Research scientists training large models. They have GPUs and don't need the browser.
- Production ML engineers shipping models to users. ONNX/llama.cpp/Transformers.js fits better.
- The "GPU power user" who wants raw kernel access. Use tinygrad or write WGSL directly.

---

## 6. The 12-month roadmap

| Phase | Window | Theme | North-star metric |
|---|---|---|---|
| **P0** | Months 1–3 | Close PyTorch gaps + first GPU acceleration + cold-start fix + run one real lab | Deep-ml catalog runs in browsergrad ≥2× faster than v0.5.0 |
| **P1** | Months 4–10 | Tracing JIT + IR + fusion + gradient checkpointing + real mixed precision + OPFS caching + safetensors streaming | Training step latency ≤5× current; fast.ai Part 2 ch 18-25 run as labs; nanoGPT trains end-to-end |
| **P2** | Months 11–14 | Megakernel codegen + WebNN tier + torch.func/vmap + custom WGSL kernels + ONNX export + lab platform alignment | Sub-3s training step; fast.ai Part 2 complete; Stanford CS336 alignment chapters run; craftingattention first 5 labs ship |

Total expanded window: **14 months** (vs. 12 in v1 of this PRD). Added 2 months to accommodate P1.7, P1.8, P2.4, P2.5, P2.6 — each a real implementation rather than a stub.

---

## 7. Feature catalog (prioritized)

Every feature below has: **what** (one-sentence description), **why** (problem statement with citation), **acceptance criteria** (measurable), **effort estimate**, **dependencies**, **evidence**.

### P0 — Months 1–3

#### P0.1: Close remaining PyTorch coverage gaps

**What**: Implement the ops that appear in greed's problem catalog and nanoGPT but aren't in v0.5.0.

| Op | Source needing it | Notes |
|---|---|---|
| `gradient_checkpoint` | deep-ml problem #188 | Likely stub initially (memory savings irrelevant in browser); document tradeoff. |
| `F.softsign` | deep-ml activation problems | Trivial, ~10 LOC. |
| `Tensor.requires_grad_()` | beginner idiom | Surfaced in PyTorch usage research — in-place setter pattern. |
| `nn.ModuleList`, `nn.ModuleDict` | nanoGPT | Container patterns. |
| `torch.tril` | nanoGPT | Used for causal masks in attention. |
| `torch.topk` | nanoGPT generation | Used for top-k sampling. |
| `torch.multinomial` | nanoGPT generation | Categorical sampling. |
| `Tensor.zero_()`, `add_()`, `mul_()` | universal beginner pattern | In-place ops on leaves with requires_grad must error. |

**Why**: Each of these blocks a specific lab. nanoGPT's `model.py` ([karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py)) requires `Embedding` (✅), `LayerNorm` (✅), `GELU` (✅), `Dropout` (✅), `ModuleDict/List` (❌), `tril` (❌), `cat` (✅), `topk` (❌), `multinomial` (❌), `F.softmax` (✅), `F.scaled_dot_product_attention` (✅), `AdamW` (✅). Adding the four missing makes nanoGPT runnable end-to-end.

**Acceptance**: nanoGPT's `model.py` imports without error. Each new op verified against the PyTorch-conformance suite within 1e-4.

**Effort**: ~2 weeks.

**Dependencies**: None.

**Evidence**: [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py); [Open-Deep-ML/DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem).

#### P0.2: Wire kernels → grad for matmul + softmax + layernorm + attention

**What**: When WebGPU is available and the runtime exposes a `browsergrad-kernels` device, dispatch `matmul`, `softmax`, `F.layer_norm`, and `F.scaled_dot_product_attention` through WGSL kernels instead of NumPy.

**Why**: This is the first time the library touches the GPU. Per the WebGPU matmul research, even unfused per-op dispatch on WGSL is roughly **3–5× faster than NumPy-in-WASM** for matmul of educational sizes (B×N×K = 64×128×512). On attention-heavy workloads the speedup is larger because we'd use the existing attention kernel.

**Acceptance**:
1. PyTorch-conformance suite passes via the WGSL backend within 1e-4.
2. Benchmarked training step on a 2-layer MLP (CIFAR-style): ≥3× faster than v0.5.0 NumPy path.
3. NumPy fallback path remains green when WebGPU absent.

**Effort**: ~3 weeks.

**Dependencies**: A backend-dispatcher seam in `tensor.py` and `functional.py`. Per-op detection of WebGPU availability.

**Evidence**: Existing `browsergrad-kernels` has the 6 kernels. Dispatch overhead is documented as significant ([arXiv:2604.02344](https://arxiv.org/abs/2604.02344)). Per-op speedup vs WASM-NumPy for matmul on M-class hardware is plausibly 3–5× based on the published WGSL benchmark ceilings.

#### P0.3: Cold-start optimization to <8s on second visit

**What**: Use OPFS + a service worker to cache the Pyodide WASM, browsergrad Python sources, and (once available) compiled WGSL pipelines.

**Why**: Cold-start is the single biggest UX number for in-browser labs. The education research dossier marks **<15s as the minimum-viable cold-start budget**. We currently have an uncached cold start of ~12s on a fresh page. Second visits are no better because we don't cache anything.

**Acceptance**:
1. First visit cold start: ≤12s on 4-year-old laptop (no regression).
2. Second visit cold start (cache warm): ≤8s.
3. Visible boot-progress UI (kernels package already emits `onPackageProgress`).

**Effort**: ~2 weeks.

**Dependencies**: OPFS write helper in runtime; service worker registration in craftingattention.

**Evidence**: [OPFS docs](https://web.dev/articles/origin-private-file-system); [Web Almanac 2025 — Caching](https://almanac.httparchive.org/en/2025/security). Pyodide team's own benchmarks for WASM cold-start under service worker.

#### P0.4: Lab runtime API hardening

**What**: Polish the runtime API surface specifically for educational labs.

| Sub-feature | What it does |
|---|---|
| `session.exec({ onProgress })` | Stream training progress (loss per step) as artifacts. |
| `bg.show_image(name, tensor)` | Inline image rendering (debugger-friendly). |
| `bg.plot_loss(name, values)` | Inline matplotlib-like plot via JSON artifact. |
| `bg.checkpoint(name)` | Save trained weights to OPFS by name. |
| `bg.load_checkpoint(name)` | Restore trained weights. |
| `bg.assert_close(a, b, atol)` | Numerically-tolerant assertion. |

**Why**: The education research dossier names six features as the minimum viable lab UX: code editor + run + stdout/stderr + assertion runner + at least one viz + persistent state. Five live in craftingattention's frontend; one (assertions + viz emit hooks) lives in browsergrad. We must own the runtime side cleanly.

**Acceptance**:
1. Every helper has an integration test.
2. Each helper round-trips through the runtime's artifact relay protocol.
3. README documents the helper API with one runnable example per function.

**Effort**: ~2 weeks.

**Dependencies**: Existing `assertion` and `artifact` relay protocols in browsergrad-runtime.

**Evidence**: Cited research on minimum-viable lab UX. Deep-ml ships all six features.

#### P0.5: End-to-end validation on one real craftingattention lab

**What**: Pick the first craftingattention lab. Run it through browsergrad. Fix every issue that surfaces. Publish a writeup.

**Why**: Per the recommendation in the PyTorch-compat dossier: "running a real lab tells you the *idiom* matches torch (whether the user's code path hits a missing op, a subtle behavioral difference, or works clean)." Without one real end-to-end run, every "should work" claim is hypothetical.

**Acceptance**:
1. The lab runs end-to-end without modifications to its PyTorch code.
2. A test fixture is added that runs the lab via Pyodide-in-node and asserts each step's output.
3. A blog post documents what worked, what broke, what was fixed, and the perf numbers.

**Effort**: ~1 week if the lab is small, 2–3 if a major op was missed.

**Dependencies**: One craftingattention lab to point at.

**Evidence**: This is methodological — "run real workloads" is consensus advice for any compatibility layer.

### P1 — Months 4–9

#### P1.1: Tracing JIT MVP

**What**: Replace the eager NumPy path with lazy tensor proxies that build a graph. Realize the graph on `.backward()` / `.numpy()` / `.tolist()`. Backend remains NumPy initially — this is a refactor, not a perf win yet.

**Why**: This is the foundational architectural change that makes everything in [VISION.md](VISION.md) possible. Per the research: tinygrad and JAX both prove the tracing-JIT design works for general PyTorch-shaped APIs. PyTorch's own `torch.compile` adds it post-hoc; we can build it in from day one. The eager path stays as a debug fallback.

**Acceptance**:
1. PyTorch-conformance suite passes via the IR path within 1e-4.
2. Existing 234 integration tests pass unchanged via the IR path.
3. Documented performance regression budget: ≤20% slower than eager during this step (the value is unlocking future gains, not immediate perf).
4. Eager NumPy mode remains available behind `browsergrad.use_eager(True)`.

**Effort**: ~6 weeks.

**Dependencies**: P0.2 (so backend dispatch through IR has a real backend to dispatch to).

**Evidence**: [tinygrad's UOps design](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py); [JAX traced functions](https://docs.jax.dev/en/latest/jit-compilation.html).

#### P1.2: First kernel fusion pass (elementwise chains)

**What**: Within the IR layer, detect chains of elementwise ops (add, mul, relu, sigmoid, etc.) and emit a single fused WGSL kernel for the whole chain.

**Why**: Per the WebGPU dispatch overhead research ([arXiv:2604.02344](https://arxiv.org/abs/2604.02344)), dispatch overhead is "a dominant cost." Elementwise fusion is the cheapest, lowest-risk fusion. PyTorch's `torch.compile` and JAX both start here.

**Acceptance**:
1. A microbenchmark on `(x * 2 + 1).relu().sigmoid()` shows ≥2× speedup vs unfused.
2. PyTorch-conformance suite passes.
3. Fusion is detectable from a debug log (`BG_FUSE_DEBUG=1`).

**Effort**: ~3 weeks.

**Dependencies**: P1.1 (IR layer must exist).

**Evidence**: Elementwise fusion is consensus practice in every modern ML compiler. The speedup is well-quantified in JAX's `jit` documentation.

#### P1.3: Reduce + softmax fusion

**What**: Fuse reduction ops with their producers and consumers. Specifically: `softmax(x)` becomes one kernel instead of three (exp → sum → divide).

**Why**: Softmax is in every model. The naive three-kernel path is wasteful. A fused softmax is also numerically more stable (one reduce pass over the data with the max-shift trick).

**Acceptance**:
1. Softmax microbenchmark ≥3× faster vs unfused (matches Flash Attention's measured softmax speedup).
2. Numerical correctness within 1e-4 vs PyTorch.
3. Same fusion pass also handles layernorm fusion (reduce + scale + shift in one).

**Effort**: ~3 weeks.

**Dependencies**: P1.2.

**Evidence**: [Flash Attention paper (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) for the softmax-fusion design.

#### P1.4: Symbolic backward

**What**: At realization time, differentiate the forward IR symbolically — emit a backward IR — instead of recording per-op closures at runtime.

**Why**: Per the VISION analysis, symbolic backward unlocks two things: (1) the backward pass can be fused independently of the forward, (2) we can dead-code-eliminate gradients for parameters that don't need them.

**Acceptance**:
1. PyTorch-conformance suite passes (including all backward tests).
2. Backward graph has ≤ forward graph's node count (a sanity check).
3. Documented behavior for ops without a closed-form derivative (e.g., `tril`, `topk`): autograd raises explicit error, not silent zero.

**Effort**: ~5 weeks.

**Dependencies**: P1.1.

**Evidence**: JAX's vjp / grad rules.

#### P1.6: `.safetensors` streaming into GPU buffers via OPFS

**What**: A loader that takes a `.safetensors` URL or `File` object, streams the bytes into OPFS once, and on subsequent loads memory-maps from OPFS directly into GPU storage buffers. Surfaced to Python as `model = bg.load_safetensors(url_or_path)`.

**Why**: Fast.ai Part 2 chapters 9, 10, 23, 25 (Stable Diffusion / super-resolution / latent diffusion) and any "use a pretrained model" lab require this. The naive path (fetch → ArrayBuffer → Float32Array → upload to GPU) does three full copies of multi-GB tensors. Memory-mapped from OPFS is one copy. Critical for first-load UX on real pretrained models.

**Acceptance**:
1. A test that loads a ~50MB educational `.safetensors` file completes in ≤2s on second visit (cache hit).
2. Memory peak during load is ≤2× the file size (vs ~4× for the naive path).
3. Hash-verified — corrupted downloads fail loudly, not silently.

**Effort**: ~2 weeks.

**Dependencies**: P0.3 (OPFS infrastructure), P1.5 (similar pattern).

**Evidence**: [safetensors format spec](https://github.com/huggingface/safetensors); fast.ai Part 2 curriculum dependencies.

#### P1.7: `gradient_checkpoint` — real implementation via IR-level rewriting

**What**: At trace time, designated subgraphs are marked "recompute on backward." Forward kernels skip writing activations to memory; backward re-runs forward to recover them. Trade-off: ~1.3× compute, ~N× memory savings.

**Why**: Fast.ai Ch 18/19 and deep-ml problem #188 both teach gradient checkpointing — and the technique is *literally* the right solution for browser memory constraints. A real implementation (not a stub) makes those chapters genuine and lets students train bigger models than naive autograd allows.

**Acceptance**:
1. `model = torch.utils.checkpoint.checkpoint_sequential(model, segments=N)` works on a 4-block MLP, with peak memory measurably lower (≥30% reduction vs unchecked).
2. PyTorch-conformance suite passes (gradients identical within 1e-4).
3. Documented compute overhead (~1.3× per checkpointed segment).

**Effort**: ~3 weeks. **Dependencies**: P1.1, P1.4.

**Evidence**: [Chen et al., "Training Deep Nets with Sublinear Memory Cost" (arXiv:1604.06174)](https://arxiv.org/abs/1604.06174); fast.ai Part 2 ch 18/19 curriculum.

#### P1.8: Real mixed precision (fp16 storage + fp32 accumulators)

**What**: `torch.amp.autocast(dtype=torch.float16)` becomes a real context manager that downcasts inputs to fp16, runs matmul/conv with fp16 storage and fp32 accumulators (matches tensor-core semantics), upcasts outputs. WebGPU `shader-f16` extension required at the device level.

**Why**: Fast.ai Ch 20 currently has to be reframed because we ship `autocast` as no-op. A real fp16 path makes it an authentic lab — students see the ~2× memory bandwidth speedup on matmul-heavy workloads, learn the numerical-stability caveats, see the loss-scaling pattern. Also a real perf win for the runtime.

**Acceptance**:
1. Matmul of shape (1024, 1024) in fp16 mode: ≥1.5× faster than fp32 path on M-class GPU (where `shader-f16` is supported).
2. Numerical stability test: training a small transformer with autocast converges to within 0.05% of the fp32 loss curve.
3. `GradScaler` works for loss-scaling.
4. Graceful degradation: devices without `shader-f16` get a clear "autocast unavailable, falling back to fp32" warning, not silent failure.

**Effort**: ~4 weeks. **Dependencies**: P1.1, P1.2 (fusion makes the fp16 path most beneficial).

**Evidence**: [WebGPU shader-f16 extension](https://www.w3.org/TR/webgpu/#shader-f16); fast.ai Part 2 ch 20.

#### P1.5: OPFS pipeline cache (compiled kernels)

**What**: Hash every WGSL kernel by source + entry-point signature. Cache the compiled `GPUComputePipeline` in OPFS. On second page load, restore.

**Why**: WGSL → native shader compile is 50–500ms per kernel. A model with 30 distinct fused kernels = 10–15s of compile on first load. Without persistent cache, every reload pays this. With cache, second-visit compile time → ~0.

**Acceptance**:
1. Second-visit page load: ≤3s end-to-end (vs ≤8s in P0.3).
2. Cache hit rate measurable via `bg.cache_stats()`.
3. Cache invalidates on browsergrad version change.

**Effort**: ~2 weeks.

**Dependencies**: P0.3.

**Evidence**: WGSL compile latency is documented in WebGPU spec discussions and Chrome dev posts. OPFS sync-access-handle benchmarks ([web.dev](https://web.dev/articles/origin-private-file-system)) confirm low-overhead reads.

### P2 — Months 10–12

#### P2.1: WebNN second-tier backend

**What**: When the browser exposes WebNN, route supported ops (matmul, conv2d, attention, sigmoid, softmax — the 95 ops cataloged by [webnn.io](https://webnn.io/en/api-reference/onnx-runtime/ops)) to WebNN instead of WGSL.

**Why**: Per the WebGPU/WebNN research, WebNN routes to dedicated AI silicon (Apple Neural Engine, Hexagon NPU, Intel AMX) which is significantly faster than WebGPU on supported ops. WebNN's GA window is **2027**, so this is properly timed for late-P2.

**Acceptance**:
1. Per-op latency on an M-series Mac with WebNN enabled: ≥2× faster than WGSL path.
2. Graceful fallback to WGSL when WebNN unavailable or op unsupported.
3. WebNN backend behind feature flag during stabilization.

**Effort**: ~3 weeks.

**Dependencies**: P1.1 (dispatcher), P1.2/P1.3 (so WGSL has fusion-equivalent perf to compare against).

**Evidence**: [W3C WebNN spec](https://www.w3.org/TR/webnn/); WebNN ops by EP at webnn.io.

#### P2.2: Megakernel codegen for transformer blocks

**What**: Pattern-match an attention-block-shaped IR subgraph (QKV-proj → attention → output-proj → residual → layernorm → FFN → residual → layernorm) and emit a single megakernel.

**Why**: Per the VISION analysis, this is where the bulk of the win lives. Flash Attention's 5–10× speedup on transformer training is exactly this pattern. PyTorch's `torch.compile` does this on CUDA; we do it on WGSL.

**Acceptance**:
1. A microbenchmark on a GPT-style transformer block (hidden=768, heads=12, seq=512) shows ≥3× speedup vs the unfused IR path.
2. Numerical correctness within 1e-3 (slightly loosened because fused kernels accumulate floats in different order).
3. Fusion pattern is configurable — a debug flag disables it for comparison.

**Effort**: ~6 weeks.

**Dependencies**: P1.1–P1.4 (the full IR stack).

**Evidence**: [Flash Attention](https://arxiv.org/abs/2205.14135); MLC-LLM's transformer block fusion.

#### P2.4: `torch.func` / `vmap` / `grad` / `jacrev`

**What**: Implement JAX-style function transforms. `vmap(fn)(batched_inputs)` retraces `fn` with a batch dim auto-inserted. `grad(fn)` returns a function-of-gradients. `jacrev` returns a Jacobian via reverse-mode.

**Why**: Stanford CS336 alignment chapters (RLHF/DPO) want these; Karpathy's advanced episodes use vmap-shaped patterns; meta-learning labs (MAML) need them. Tractable because we already have tracing + symbolic backward — these transforms are graph rewrites.

**Acceptance**:
1. `vmap` produces identical output to a manual Python loop, to 1e-4, on 5 representative examples (linear, attention, recurrent step, custom function).
2. `grad(f)(x)` matches `f(x).backward(); x.grad` exactly.
3. `jacrev(f)(x)` matches PyTorch's `torch.func.jacrev` within 1e-4 on 3 fixture cases.

**Effort**: ~4 weeks. **Dependencies**: P1.1, P1.4 (symbolic backward).

**Evidence**: [JAX vmap/grad/jacrev docs](https://docs.jax.dev/en/latest/jax.html#); [PyTorch torch.func docs](https://pytorch.org/docs/stable/func.html).

#### P2.5: Custom WGSL kernels from Python

**What**: `@bg.kernel("wgsl")` decorator lets the user write a WGSL kernel as a Python string. The decorator handles buffer binding, dispatch, and dtype validation. Surfaced to Python as a callable that takes Tensors and returns Tensors.

**Example**:
```python
@bg.kernel("wgsl", workgroup_size=(64, 1, 1))
def double_each(x: Tensor) -> Tensor:
    """
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        output[gid.x] = input[gid.x] * 2.0;
    }
    """
```

**Why**: Strategic — opens an entire new category of advanced lab ("write Flash Attention from scratch", "implement a custom convolution variant"). Differentiates us from greed (which doesn't expose WGSL). For a curriculum aiming at *understanding* deep learning down to the metal, this is the unlock.

**Acceptance**:
1. A toy kernel (elementwise double) round-trips through the harness and matches NumPy reference within exact equality.
2. A more complex kernel (matmul, tiled with workgroup memory) compiles and runs.
3. Security review passes: user-provided WGSL is compiled in an isolated worker; failed compilation surfaces as a Python exception, not a tab crash; buffer size validation prevents OOB writes.

**Effort**: ~4 weeks (most of which is security review + ergonomic API design).

**Dependencies**: P1.5 (pipeline cache infrastructure).

**Evidence**: [WGSL spec](https://www.w3.org/TR/WGSL/); existing patterns in tinygrad's `Tensor.uop_kernel` and JAX's `pallas`.

#### P2.6: ONNX export

**What**: Walk the IR graph, emit ONNX nodes, serialize to bytes. Surfaced as `bg.onnx.export(model, sample_input, path)`.

**Why**: "Train in browser, deploy anywhere" is a complete-the-story unlock. Students finishing a craftingattention lab can download `.onnx` and run inference in any environment (Python, C++, Node, mobile). Currently impossible — Pile C stubs onnx.export to raise NotImplementedError.

**Acceptance**:
1. A 3-layer MLP exports to ONNX; verified by loading in onnxruntime and getting identical predictions within 1e-4.
2. A small CNN exports; same verification.
3. A small transformer block exports; same verification.
4. Documented op coverage (which ops can/can't be exported), since ONNX has its own opset.

**Effort**: ~2 weeks (the IR graph + ONNX is mostly a serialization mapping).

**Dependencies**: P1.1.

**Evidence**: [ONNX opset](https://onnx.ai/onnx/operators/); [PyTorch torch.onnx](https://pytorch.org/docs/stable/onnx.html).

#### P2.3: Lab platform alignment

**What**: Whatever craftingattention needs to ship its first 5 lessons. Identified during craftingattention's lesson authoring, not pre-specified here.

**Why**: The runtime exists to enable the curriculum. The curriculum's needs aren't predictable from outside.

**Acceptance**: First 5 craftingattention lessons ship publicly. Each runs end-to-end without modifications.

**Effort**: Unknown — reactive to lesson authoring.

**Dependencies**: Craftingattention having lesson content ready.

**Evidence**: This is the only success metric that matters at end-of-Year-1.

---

## 8. What we explicitly DON'T build (and why)

The original PyTorch-op research flagged eight features as "zero appearances in surveyed curricula." After expanding scope to cover fast.ai Part 2 + Stanford CS336 + Karpathy advanced episodes, **four of those eight became real-implementable via the JIT** and moved to the roadmap:

- **Mixed precision** → P1.8 (real fp16 path with fp32 accumulators, not stub)
- **ONNX export** → P2.6 (real graph serialization via IR)
- **`torch.func` / `vmap`** → P2.4 (JAX-style transforms via re-tracing)
- **`torch.compile`** → P1.1 sub-feature (`torch.compile(fn) → fn` since JIT is automatic)

**Still genuinely impossible (not building, ever)**:

1. **Distributed training / DDP / nccl** — no multi-machine in a single browser tab.
2. **JIT/TorchScript** as TorchScript-compatible — our IR is different; supporting `torch.jit.script` would mean writing a TorchScript parser for no user benefit.
3. **Custom CUDA kernels** — no CUDA in browser. *We DO offer custom WGSL kernels via P2.5 as the legitimate browser-native alternative.*
4. **Complex dtypes, sparse tensors** — still zero curriculum demand. Skip until evidence.
5. **GPT-2-scale reproduction** — 1.5B+ parameters, multi-GB weights, distributed training implied. Out of memory/compute budget for browser.

Updated stance: **the goal is "everything credible done well, nothing faked."** A real fp16 implementation beats a stub for the educational value of teaching mixed precision; a real ONNX export beats a stub for the deploy-after-train story; a real `vmap` beats a stub for batched-gradient labs.

---

## 9. Success metrics

| Metric | Baseline (v0.5.0) | P0 target | P1 target | P2 target |
|---|---|---|---|---|
| Greed problem catalog coverage | ~93% (12/14 sampled) | 100% (14/14) | 100% | 100% |
| Fast.ai Part 2 chapters runnable as labs | ~8/17 | ~14/17 | 16/17 | 17/17 |
| Stanford CS336 from-scratch chapters | ~10/14 (estimate) | ~12/14 | 13/14 | 14/14 |
| ResNet18 training step latency | ~500ms | ≤200ms | ≤50ms | ≤20ms |
| Transformer block forward | ~300ms | ≤100ms | ≤25ms | ≤5ms |
| Cold-start (second visit) | not cached | ≤8s | ≤3s | ≤2s |
| End-to-end craftingattention lab passes | n/a | 1 lab | 3 labs | 5 labs |
| PyTorch-conformance ops verified vs torch within 1e-4 | 5 | 10 | 25 | 50 |
| External adopters (open source projects using browsergrad-grad) | 0 | 1 | 3 | 5 |
| GitHub stars | TBD | +50 | +250 | +1000 |

All latency estimates are point-in-time assumptions on "4-year-old consumer laptop" (M1 MacBook Air or equivalent). Actual numbers will vary; we benchmark per release and publish.

---

## 10. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | WGSL compile latency makes first-visit UX bad | High | High | OPFS cache (P1.5); pre-compile common kernels; show progress UI |
| R2 | PyTorch ships `torch.browser` and obsoletes us | Low | Existential | We're faster to ship; we own the lab UX |
| R3 | WebGPU adoption stalls (Firefox / mobile coverage gaps) | Medium | Medium | Tier 4 (WASM SIMD) fallback; document the matrix |
| R4 | Cross-origin isolation requirements break SAB path on most hosts | High | Medium | Make SAB an opt-in performance tier, not a baseline |
| R5 | JIT tracing semantics surprise users (data-dependent control flow) | Medium | Medium | Document the supported subset; raise loud errors on unsupported patterns |
| R6 | WebNN GA slips beyond 2027 | Medium | Low | We didn't bet anything on it for v1 |
| R7 | Greed gets significantly faster before we do | Medium | Medium | Establish benchmarks now; iterate on perf publicly |
| R8 | Craftingattention runs out of runway before P2 | Medium | High | Browsergrad has standalone value as the open runtime — license + governance designed for that case |

---

## 11. Open questions (need answers before P1)

1. **What is craftingattention's first lesson and when?** Drives P0.5's target.
2. **Will craftingattention pay for craftingattention-specific features?** Affects how much we'd carve out of the general roadmap.
3. **Do we want to publish a head-to-head greed-vs-browsergrad benchmark?** It's the highest-signal marketing move, but only if we're faster (otherwise it'd be advertising for greed).
4. **Should browsergrad-jit be a new package or replace browsergrad-grad?** VISION.md leans toward "new package, migrate over time." PRD assumes that, but the decision should be re-asked at P1 start.
5. **Funding model.** Is this a side project, sponsored OSS, or eventual commercial product? Affects P2.3 lab platform scope.

---

## 12. Appendix: research sources cited

Browser ML libraries:
- [Transformers.js v4 (Feb 2026)](https://roboaidigest.com/posts/2026-02-11-transformers-js-v4-webgpu/)
- [ONNX Runtime Web WebGPU launch (MS, Feb 2024)](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)
- [TensorFlow.js releases](https://github.com/tensorflow/tfjs/releases)
- [WebLLM paper (arXiv:2412.15803)](https://arxiv.org/pdf/2412.15803)
- [tinygrad GitHub](https://github.com/tinygrad/tinygrad)
- [Pyodide #1625 (PyTorch unavailable)](https://github.com/pyodide/pyodide/issues/1625)

WebGPU / WebNN status:
- [WebGPU caniuse](https://caniuse.com/webgpu)
- [WebGPU Baseline announcement, web.dev](https://web.dev/blog/webgpu-supported-major-browsers)
- [W3C WebNN spec](https://www.w3.org/TR/webnn/)
- [WebNN ops cataloged at webnn.io](https://webnn.io/en/api-reference/onnx-runtime/ops)
- [Frontier Web APIs 2026 forecast](https://www.utsubo.com/blog/frontier-web-apis-2026-production-ready)

Performance:
- [Optimizing a WebGPU Matmul Kernel (nuss-and-bolts)](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)
- [SitePoint WebGPU vs WebGL inference benchmarks](https://www.sitepoint.com/webgpu-vs-webgl-inference-benchmarks/)
- [LLM tokens/sec benchmarks (mustafa.net)](https://mustafa.net/llm-tokens-per-second-benchmarks/)

Competitors / market:
- [deep-ml.com](https://www.deep-ml.com/)
- [Open-Deep-ML/DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem)

PyTorch usage patterns:
- [Core ATen Operator Set (ExecuTorch)](https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html)
- [Defining the Core ATen Opset — PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/defining-the-core-aten-opset/1464)
- [Interoperability in Deep Learning, ISSTA 2024 (arXiv:2303.17708)](https://arxiv.org/html/2303.17708v4)
- [CS231n Assignment 2 PyTorch notebook](https://github.com/srinadhu/CS231n/blob/master/assignment2/PyTorch.ipynb)
- [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- [fast.ai book ch.4 (O'Reilly)](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/ch04.html)

Standards / specs:
- [OPFS spec (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
- [HTTP Archive Web Almanac 2025 — Security](https://almanac.httparchive.org/en/2025/security)
- [Flash Attention paper (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)

---

This document will be revised quarterly. Each P0/P1/P2 feature gets a one-line status update in PROGRESS.md as it ships. Trade-offs that depart from this PRD get an ADR in ARCHITECTURE.md.
