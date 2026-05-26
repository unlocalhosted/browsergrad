# Browsergrad: Vision and Architecture

This document is for someone who hasn't built a deep learning library before. It explains what we have, where it hits a ceiling, and what we'd build next if we were unafraid of the work. Every technical term is explained the first time it appears.

For the day-to-day "what shipped this week" view, see [PROGRESS.md](PROGRESS.md). For decisions we've already made on the *current* library, see [ARCHITECTURE.md](ARCHITECTURE.md). This file is forward-looking.

---

## Part 1: The world this library lives in

Most deep learning today happens in PyTorch. A researcher writes a few hundred lines of Python, calls `model.cuda()` to move the work to a graphics card, and trains a neural network. The card is a Nvidia GPU costing somewhere between $1,000 and $40,000, and the work is being sent to it by a sophisticated piece of software stack called CUDA, which is proprietary to Nvidia.

This setup is brilliant if you're a researcher at a lab. It's miserable if you're a student trying to learn. The friction of getting CUDA working, finding a machine with a compatible GPU, installing PyTorch with the right CUDA version, dealing with driver mismatches — that friction is the reason most online deep learning courses are taught with hosted notebooks (Colab, Kaggle) rather than running on your own machine. The student doesn't own the environment; someone else does.

**Browsergrad is the bet that the browser is enough.** Every modern browser has access to the GPU through a standardized API (WebGPU). Every modern browser can run Python (via Pyodide, which compiles Python to WebAssembly). Every modern browser can run multi-threaded code (via Web Workers + SharedArrayBuffer). If you compose those primitives carefully, you can build a deep learning environment that runs entirely in a webpage. No install. No driver mismatches. No proprietary runtime. The student lands on a URL and they're training a model in fifteen seconds.

What we have today is the proof-of-concept of that bet. What this document describes is what we'd build next to turn the proof-of-concept into something that competes with PyTorch on its own terms.

---

## Part 2: A quick mental model of how a deep learning library works

If you already know this material, skip to Part 3. If you don't, the rest of this document won't make sense without it.

### Tensors

A **tensor** is a multi-dimensional array of numbers. A grayscale image is a 2D tensor (height × width). A color image is 3D (height × width × channels). A batch of 32 color images is 4D (32 × height × width × channels). A weight matrix in a neural network might be 2D (1024 × 4096).

In code, a tensor is just a block of memory holding the numbers, plus some metadata: the shape (the dimensions), the dtype (whether the numbers are 32-bit floats, 16-bit floats, 8-bit ints, etc.), and the stride (how to walk through the memory to get to the next element along each dimension). Everything else a deep learning library does is operations on tensors.

### Ops

An **op** (operation) is a function that takes one or more tensors and produces one or more tensors. Examples: addition (elementwise sum of two equal-shaped tensors), matmul (matrix multiplication), softmax (normalize each row into a probability distribution), ReLU (replace negative values with zero), convolution (sliding a small filter across an image).

Real-world neural networks are stacks of ops. A small CNN classifier (a convolutional neural network that recognizes images) might apply 50 ops to its input. A transformer language model might apply 1000.

### The graph

When you write `y = (x @ W) + b` in PyTorch, you're declaring a tiny graph: x and W feed a matmul, the result feeds an add with b, that produces y. PyTorch builds this graph implicitly as you execute the code. Every variable holds not just its value but a record of how it was computed.

**Forward pass** = walk the graph in the direction of computation: read inputs, apply ops, produce outputs. This is what `y = model(x)` does.

**Backward pass** = walk the graph in *reverse*, computing how much each input contributed to the loss. The backward pass is what makes training work. It uses a technique called **automatic differentiation** (autograd, for short) — you write only the forward pass; the library figures out the backward pass by applying the chain rule from calculus to each op in reverse order.

Every op has a forward function (how to compute the output) and a backward function (how to compute the gradient given the upstream gradient). Compose forward functions, get a forward pass. Compose backward functions in reverse, get a backward pass.

### Why this matters for performance

Here's the thing the casual user never sees. When PyTorch runs `y = x @ W + b`, it does *not* run one fused operation. It runs three:

1. Allocate a buffer for the matmul result. Launch a matmul kernel on the GPU. Wait for it to finish.
2. Allocate a buffer for the add result. Launch an add kernel on the GPU. Wait for it to finish.
3. Return the final buffer.

Each of those "launch a kernel" steps has overhead — maybe 10 microseconds of CPU work to set up the dispatch, plus the kernel itself runs for maybe another 10 microseconds for small tensors. For a transformer block with 20 ops, that's 200 microseconds of overhead before any actual math happens. On large tensors this overhead doesn't matter; the math dominates. On small or medium tensors it kills you. **Dispatch overhead is the silent ceiling on PyTorch's performance**, and the entire reason `torch.compile` exists is to crash through it.

### Kernel, briefly

A **kernel** in this context is a small program that runs on the GPU, doing the actual math for one op. It's written in a shading language — for WebGPU, the language is WGSL (WebGPU Shading Language). The kernel describes what one thread does; the GPU runs millions of threads in parallel to compute the whole tensor.

Now we have enough vocabulary.

---

## Part 3: The bet

Here's the endgame, stated bluntly:

> **A JIT-tracing autograd compiler that emits megakernels at runtime, dispatches across WebNN / WebGPU / WASM / NumPy depending on what hardware the user has, persists everything to OPFS so the second tab load is instant, and exposes a PyTorch-shaped API in Python so the user notices none of it.**

Three sentences. Five terms that need explanation. Let's unpack them in order.

**JIT** = Just-In-Time. As opposed to AOT (Ahead-Of-Time, like C++ compilation) or interpretation (like vanilla Python). JIT means we compile code into a faster form *the first time it runs*, then reuse the compiled form on subsequent runs. The browser's JavaScript engine is a JIT — that's why your second visit to a website is faster than the first.

**Megakernel** = a single GPU kernel that does the work of many ops fused together. Instead of 20 dispatch calls for a transformer block, you compile one kernel that does all 20 ops without touching slow memory between them. This is what Flash Attention does (it fuses softmax + matmul into one kernel) and it's the reason Flash Attention is 5–10× faster than the unfused version. Megakernels are where the magic is.

**Tracing** = the act of running the user's code symbolically, recording every op into a graph, instead of actually computing the results. You then look at the recorded graph and decide how to compile it. JAX does this. So does `torch.compile`. Tinygrad does this.

**WebNN** = a browser API (shipping in Chrome, in spec at W3C) that lets a webpage tell the operating system "I have a matmul of these dimensions; please run it on the best available hardware." If the user has an Apple M-series Mac, that's the Apple Neural Engine. If they have a recent Snapdragon laptop, that's the Hexagon NPU (neural processing unit). If they have an Intel CPU with AVX-512, that's vectorized CPU code. WebNN is *faster than WebGPU* on devices with dedicated AI silicon, because the dedicated silicon is purpose-built for the ops we run.

**OPFS** = Origin-Private File System. A real filesystem inside the browser, persistent across page loads, with synchronous read/write. Per-origin (so amazon.com can't see github.com's files). You can store gigabytes there. We'd use it to cache compiled kernels and weight tensors — first page load is slow, every subsequent load is instant.

Now read the thesis again. It says: trace the user's PyTorch code, fuse the ops into megakernels, dispatch each kernel to the best hardware path available, cache everything to disk, give the user a PyTorch API on top. It's saying we'd compete with PyTorch on its strongest axis (performance via compilation) using the unfair advantage that we own the entire stack from Python down to the GPU.

---

## Part 4: The architecture, in five layers

Imagine the system as a stack of five layers. The user lives at the top. The hardware lives at the bottom. Each layer talks only to the one directly below it.

```
┌───────────────────────────────────────────────────┐
│ Layer 1: Python frontend (looks like PyTorch)     │
├───────────────────────────────────────────────────┤
│ Layer 2: Tensor proxies + lazy graph capture       │
├───────────────────────────────────────────────────┤
│ Layer 3: IR + kernel fusion                        │
├───────────────────────────────────────────────────┤
│ Layer 4: Multi-tier backend dispatch               │
├───────────────────────────────────────────────────┤
│ Layer 5: Persistence (OPFS, pipeline cache)        │
└───────────────────────────────────────────────────┘
```

Let's walk through each.

### Layer 1: The Python frontend

The user writes Python that looks like PyTorch. Same `nn.Module`, same `Tensor`, same `Linear` and `Conv2d` and `LayerNorm`. Same training loop with `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`. If their code already runs in real PyTorch, it should run in browsergrad with no changes (except where the underlying ops genuinely don't exist in the browser, like `torch.cuda.is_available()` returning False).

This layer is mostly already built. It's what `browsergrad-grad` ships today. It runs eagerly — every op produces a real tensor immediately by calling NumPy.

**The change in the new architecture**: every op no longer returns a real tensor. It returns a "tensor proxy" — an object that *looks* like a tensor (same `.shape`, same `.dtype`, supports the same operators) but doesn't hold computed values. It holds a recipe for how to compute its values, but defers the actual computation until someone asks for the numbers.

This is called **lazy evaluation**. The user's code still reads like eager code. Internally, nothing has actually been computed yet.

### Layer 2: Lazy graph capture

When the user writes `y = (x @ W) + b`, Layer 1 calls the matmul op, which returns a tensor proxy. The proxy's recipe is: "I am the matmul of x and W, and my shape is (rows of x) × (cols of W)." Then the user adds `b`. Layer 1 calls the add op, which returns another tensor proxy whose recipe is "I am the add of the matmul-proxy and b."

By the time the user has written 20 lines of model code, Layer 2 has built a graph — a tree where each node is an op, and the leaves are the model's parameters and inputs. Nothing has computed yet.

The graph stays purely symbolic until one of three things happens:
1. The user reads a concrete value (calling `.numpy()`, `.tolist()`, `.item()`, or printing the tensor).
2. The user calls `.backward()`.
3. The user calls `optimizer.step()` (which internally reads the gradients and updates weights).

At that moment, we **realize** the graph. Realization means: figure out what kernels to launch, in what order, and actually launch them.

**The reason this matters**: if we wait until the user wants the numbers, we get to see the *whole* computation before we run any of it. We can rearrange, fuse, dead-code-eliminate, choose better algorithms — all things you can't do if you eagerly execute every op the moment the user writes it.

In the deep learning world, this design is called a **lazy graph** or **trace-based execution**. Tinygrad (a research library by George Hotz) is the cleanest example. JAX does it with a different API. PyTorch added it later via `torch.compile`. We'd build it in from day one.

### Layer 3: IR and kernel fusion

**IR** stands for Intermediate Representation. It's a translation of the user's high-level graph into a lower-level form that's easier for a compiler to optimize. Real compilers (LLVM, GCC) use IRs because they let you write optimizations once that work across many source languages and many target hardwares.

Tinygrad has a beautifully small IR — about 15 different node types: LOAD, STORE, MUL, ADD, REDUCE, CAST, RESHAPE, BUFFER, etc. Every PyTorch op decomposes into a tiny graph of these IR nodes. A matmul becomes a LOAD + LOAD + MUL + REDUCE. A softmax becomes a sequence of EXP + REDUCE + DIV.

Once the user's whole forward pass is in IR form, we can do the thing that makes everything fast: **kernel fusion**. Look at the graph. Find sequences of ops where the output of one feeds directly into the next, and where intermediate results don't need to leave the GPU. Fuse them into a single kernel that does all the work without touching slow memory between steps.

Concrete example. A standard self-attention block reads roughly:

```
Q = x @ W_q + b_q              # matmul, add
K = x @ W_k + b_k              # matmul, add
V = x @ W_v + b_v              # matmul, add
scores = Q @ K.T / sqrt(d)     # matmul, divide
attn   = softmax(scores)       # exp, reduce, divide
output = attn @ V              # matmul
```

That's 11 separate ops, which would mean 11 separate kernel launches. But notice: the intermediate `scores` matrix is huge and we never look at it again after computing `attn`. There's no reason to write it to GPU memory; we can keep it in fast on-chip memory (called **workgroup memory** in WebGPU, **shared memory** in CUDA). That's exactly what Flash Attention does, and it makes the whole block roughly 5× faster on large sequences.

A good fusion pass detects this pattern in the IR and emits one megakernel that does qkv-projection + attention + output-projection without ever materializing the intermediate tensors. Same numerical result. Five times less data moved. One kernel launch instead of eleven.

**This is where the unfair advantage lives.** PyTorch can't do fusion in eager mode because it doesn't know what's coming next; by the time it's run the first matmul, you might or might not be about to use it. We always know what's coming next, because we traced the whole graph before we ran anything.

### Layer 4: Multi-tier backend dispatch

Once the IR is fused into a kernel plan (a sequence of "run kernel K with these input buffers and these output buffers"), we have to actually run it. Different hardware paths have different sweet spots; we want to pick the best one per kernel.

There are five tiers of backend, in priority order. Each kernel gets the highest-priority tier that supports it on the user's actual machine.

**Tier 1: WebNN.** If the user's browser exposes WebNN, and the kernel is one of WebNN's supported ops (matmul, conv2d, sigmoid, softmax, attention, layernorm), and the shapes match WebNN's constraints — dispatch there. WebNN routes to the user's most specialized silicon: Apple Neural Engine on Mac, Hexagon NPU on Snapdragon, NPU on recent Intel chips, fallback to GPU otherwise. Always at least as fast as WebGPU, often dramatically faster. *Cost to use: zero, except the engineering to detect support per op.*

**Tier 2: Fused megakernel via WebGPU + WGSL.** For ops that don't fit WebNN's API but match a fusion pattern we've codegen'd, run our hand-written or compiler-emitted megakernel. Cost: kernel compile time on first use (cached afterward), then near-peak GPU throughput.

**Tier 3: Primitive WGSL kernels.** Fallback for ops that haven't been fused yet. Matmul, softmax, layernorm, elementwise — written once, dispatched per op. This is what `browsergrad-kernels` ships today, conceptually unchanged.

**Tier 4: WASM SIMD with workers.** If the user is on a device without WebGPU (older browser, locked-down corporate machine), fall back to multi-core CPU. WASM SIMD gives us 128-bit-wide vector operations. Web Workers give us as many cores as `navigator.hardwareConcurrency` reports. SharedArrayBuffer lets the workers communicate without copying. Performance is maybe 5–20% of WebGPU on the same machine — not great, but the code still runs.

**Tier 5: NumPy in Pyodide.** The slow-but-correct floor. Whatever happens, the op runs on NumPy in WASM. This is what every op runs on today. It's slow but it's our correctness oracle — every higher-tier backend's result is cross-checked against this one in tests.

The dispatcher itself is a Python function — no, actually a JavaScript function reachable from Python — that takes the IR's `KernelOp` and picks the tier. It's per-kernel, not per-graph. A single fused graph might dispatch its first kernel to WebNN, its second to a fused WebGPU megakernel, and a tiny third one (a reduction over a vector of size 5) to NumPy because the dispatch overhead would dominate at that size.

The grad layer above doesn't know which backend ran. It only sees a Tensor with numbers in it. The backend is genuinely an interchangeable adapter — a real seam in the LANGUAGE.md sense, with five real implementations.

### Layer 5: Persistence

Compiling kernels is expensive. WGSL → SPIR-V → native shader binary is a chain that takes 50–500ms per kernel, depending on complexity. We cannot afford to do that on every page load. We cache.

OPFS is the durable store. We hash every IR fragment (so the same kernel pattern across users gets the same cache key) and write the compiled pipeline binary to OPFS keyed by that hash. The cache is per-origin, persistent across sessions, and survives browser restarts.

The persistence layer also caches:
- **Traced IRs** keyed by a hash of the model's Python source. First call traces; second call to the same model skips Python entirely.
- **Weight tensors**, mmap-style. The user uploads a `.safetensors` file once; we hash it, write to OPFS, and on every subsequent page load we read the bytes directly into GPU buffers without going through JavaScript's ArrayBuffer.
- **Training checkpoints**. Every N steps, optimizer state goes to OPFS. Close the tab, come back tomorrow, training resumes.

The persistence layer is what makes the system *feel* native. First visit: slow (cold caches). Every subsequent visit: instant. Same model + same dataset + same seed = bit-for-bit reproducible across machines.

---

## Part 5: What happens when the user trains a model

To make all the above concrete, let's walk through what the system actually does when a user runs:

```python
model = ResNet18()                  # define a CNN
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(10):
    for x, y in train_loader:
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Page load.** The Python file gets parsed. `import torch` runs, which calls `install_torch_alias()` and points the torch namespace at our shim. No GPU work yet.

**First iteration of the training loop, first time `model(x)` is called.** Layer 1 receives the call. Each layer of ResNet18 is an `nn.Module`; each module's `forward()` runs through Python. But every op call returns a tensor proxy from Layer 2, not a real tensor. By the end of `model(x)`, Layer 2 holds a graph representing the entire forward pass — maybe 200 nodes for ResNet18.

`F.cross_entropy(model(x), y)` does the same thing: it returns a tensor proxy whose recipe is "I am the cross-entropy of [200-node graph] and y."

`loss.backward()` is the moment of realization. Layer 2 hands the graph to Layer 3.

Layer 3 walks the graph and emits IR. The IR is the same kind of LOAD/MUL/ADD/REDUCE nodes you'd see in tinygrad. The forward IR is one graph. Layer 3 then runs a **symbolic differentiation** pass — it walks the forward IR and emits a backward IR. Symbolic differentiation here means: for each forward IR node, mechanically generate the IR for its derivative. Add's derivative is identity. Multiply's derivative is the chain rule. Matmul's derivative is two matmuls. Etc. This is exactly what JAX does, and it produces a *smaller* graph than runtime autograd because the optimizer can dead-code-eliminate paths that don't affect the output.

Now Layer 3 has a forward IR and a backward IR. It runs the fusion pass: chains of elementwise ops merge into one kernel; reductions fuse with their producers; pattern-matching detects "this is a Flash-Attention-shaped subgraph" and emits the fused megakernel. The output is a **kernel plan** — a list of "execute this kernel with these buffer bindings."

Layer 4 walks the kernel plan and dispatches each kernel. ResNet18 forward maybe becomes 12 fused kernels. Backward maybe 14. Each is dispatched to the best backend that supports it. The dispatcher is itself fast (microseconds).

The kernels run. `loss.backward()` returns when all of them have finished and the gradients are in the parameters' `.grad` buffers.

`optimizer.step()` is another graph (the Adam update rule), traced, fused into one elementwise kernel per parameter (or fused across all parameters into a single megakernel), dispatched.

Total wall-clock time for the first iteration: maybe 2 seconds, dominated by kernel compilation.

**Second iteration of the training loop.** Same model, same shapes — the trace cache hits. We skip Python entirely. The kernel plan is already in memory. We just bind the new input buffer and re-dispatch. Total wall-clock time: maybe 8 milliseconds.

**Page reload, day later.** OPFS has the cached pipelines. The first call still traces (or also caches the trace — implementation choice). Within a few seconds, training is back at second-iteration speed.

Compare this to the current `browsergrad-grad`. Current implementation: every op is an eager Python call, every op runs NumPy in WASM, no fusion, no caching, no GPU. Every iteration of the same training loop does Python interpretation work, allocates intermediate NumPy arrays, never touches the GPU at all. ResNet18 training step today is maybe 500ms; with the vision architecture it would be 8ms. **That's the gap we're trying to close.**

---

## Part 6: What we build, what we don't, what we kill

### Build

- **A UOps-style IR.** ~500 lines of Python. The single most important file in the codebase. Borrows tinygrad's design.
- **A tracer.** Replaces NumPy execution with proxy execution. ~300 lines.
- **A WGSL codegen pass.** Takes IR nodes, emits WGSL. ~800 lines (matmul, reduce, elementwise, softmax, layernorm).
- **A fusion pass.** Pattern-matches IR subgraphs against megakernel templates. ~500 lines.
- **A backend dispatcher.** Looks at each kernel, picks the tier. ~200 lines.
- **A WebNN backend.** ~400 lines, mostly building MLGraph objects from IR.
- **OPFS-backed pipeline cache.** ~200 lines.
- **A symbolic backward pass.** ~600 lines, biggest cognitive load piece in the system.

Total: maybe 4,000 lines of new code. Less than `browsergrad-grad` is today. Most of the lines come *out*, not in, because the NumPy-per-op code shrinks dramatically when the IR layer handles dispatch.

### Don't build

- **A new tensor library.** The user-facing API stays PyTorch-shaped. We don't invent new ergonomics; PyTorch's are fine.
- **A custom Python interpreter.** Pyodide stays as our Python runtime. We trace through it; we don't replace it.
- **A graph debugger UI in the first cut.** Useful, but it's the lab's job, not the library's.
- **Multi-GPU.** The browser doesn't really have multi-GPU. WebGPU lets you pick a device; we pick one. Real distributed training needs a separate product.
- **Mixed precision.** WebGPU's fp16 support is uneven. We do fp32 everywhere and let WebNN do fp16 where it does it natively. Mixed-precision Python APIs become no-ops.

### Kill

- **The current `_ctx` autograd machinery in `tensor.py`.** Replaced by symbolic backward at the IR level. We delete maybe 300 lines.
- **Per-op backward closures in every functional file.** All replaced by IR rewrites.
- **The eager NumPy path as the default.** It stays as the floor (Tier 5) but it's no longer the hot path. NumPy code in `tensor.py` goes from being the main implementation to being the reference implementation.
- **Hand-written WGSL kernels for ops that the codegen handles.** They become unit-test reference impls.

---

## Part 7: Migration path

You don't throw away a working library to build a new one. You build the new layer underneath the existing API, route the existing tests through it, watch them stay green, then flip the switch.

The actual sequencing:

**Step 1 — IR layer as a new package.** New repo subdirectory, `packages/browsergrad-jit/`. Build the IR + tracer + NumPy backend. The NumPy backend is the same NumPy operations the current grad library uses, just dispatched from IR instead of called directly from Python. The PyTorch-conformance test suite has to pass through this path.

**Step 2 — re-implement `browsergrad-grad` on top of the IR.** The Tensor class, the ops in `functional`, the layers in `nn` — none of their public API changes. Internally, every op produces an IR node instead of calling NumPy directly. NumPy still runs at realization time. The user observes no difference. The conformance suite passes.

**Step 3 — WGSL backend.** New tier in the dispatcher. Initially behind a feature flag. Cross-validate every kernel against the NumPy reference. Once the conformance suite is green on the WGSL path, raise its priority in the dispatcher.

**Step 4 — fusion pass.** Start with the easiest fusions (chains of elementwise ops). Verify they produce the same output as the unfused version. Add reduce fusion, then softmax fusion, then attention fusion. Each fusion ships behind a separate flag that can be disabled if it goes wrong.

**Step 5 — WebNN backend.** Detect at runtime whether the user has it. Implement the supported ops behind it. Raise its dispatch priority for those ops.

**Step 6 — OPFS persistence.** Cache pipelines, then traces, then weights.

**Step 7 — the lab.** A separate product, layered on the now-fast library.

Each step lands as a separate epoch in the codebase, each behind a feature flag, each with the conformance suite as the gate. At any step, we can disable everything below it and fall back to the current `browsergrad-grad` behavior. The current library is the bottom rung of the ladder.

**Timeline (honest estimate, single engineer, focused):**
- Step 1: 3 weeks.
- Step 2: 4 weeks.
- Step 3: 6 weeks (WGSL is fiddly).
- Step 4: 4 weeks per fusion family; total 12 weeks across the meaningful ones.
- Step 5: 3 weeks.
- Step 6: 2 weeks.
- Step 7: separate product, not estimated here.

Total: ~6 months of focused full-time work to get to "fast lab demos." Another 6 months to be production-grade for non-trivial models.

---

## Part 8: Risks and unknowns

I'm not going to pretend this is a sure thing. Here's what could go wrong, in rough order of how worried I am.

### Compilation latency on first use is bad UX

Compiling WGSL at runtime takes hundreds of milliseconds per kernel. A model with 30 distinct fused kernels could take 10–15 seconds to compile fully on first run. If the user's first impression of the lab is "the page is frozen for 12 seconds before anything happens," they leave.

**Mitigations:** ship a pre-compiled bundle of the most common kernels with the library. Aggressive pipeline reuse — once we compile matmul-1024×1024, we use that pipeline for every matmul of that shape across all models. Optimistic background compilation — start compiling the next likely kernel before the user asks for it. None of these eliminate the first-load problem; they shrink it.

### WebGPU is still inconsistent across browsers

Chrome ships WebGPU stable. Firefox ships it behind a flag. Safari shipped it in 17.4 but with quirks. Most enterprise browsers (locked-down corporate IT setups) don't have it. The Tier 4 (WASM SIMD) fallback exists for this reason but it's slow.

**Mitigation:** the lab needs to gracefully degrade. A student on a locked-down browser still sees the model train, just slower. Show them a notice. Don't let them be confused about why.

### Tracing is hard

JIT tracing assumes the model's structure doesn't change between calls. If the user writes Python that branches on tensor values (e.g., `if loss > threshold: do_thing()`), tracing breaks. JAX has whole machinery to handle this (`jax.lax.cond`, `jax.lax.scan`). We'd need similar.

**Mitigation:** in v1, just don't support data-dependent control flow inside `nn.Module.forward()`. Document it. 95% of educational labs don't need it.

### WebNN's API is volatile

WebNN is still in W3C draft. The API has changed in incompatible ways between Chrome versions. Building on it now means being prepared to refactor.

**Mitigation:** keep the WebNN backend isolated behind the dispatcher interface. If WebNN changes, only one file changes.

### Symbolic backward is harder than it looks

Most ops have textbook gradients. Some don't. Scatter operations, gather operations, attention masks with arbitrary patterns, custom losses. Getting symbolic backward right for every op the library ships is the largest cognitive load piece of the new architecture.

**Mitigation:** for ops where symbolic backward is brittle, fall back to manual backward (define forward and backward IR by hand). The IR doesn't care where the backward came from.

### We could build all this and PyTorch ships `torch.browser` next year

This is the asymmetric risk. PyTorch the project is actively looking at browser-side execution (there's a `torch.compile` backend for ONNX and there's research-stage work on WebAssembly). If PyTorch ships first-party browser support, the value of browsergrad collapses to "easier on-boarding."

**Mitigation:** we ship faster than PyTorch's release cadence and we focus on the use case PyTorch won't (education, zero-install, full-stack lab experience). Most likely outcome even in the worst case: we contribute back what we learn.

---

## Part 9: Glossary

For reference, every technical term used in this document, in one place.

- **AOT (Ahead-Of-Time)** — compilation that happens before execution, as opposed to JIT.
- **Autograd (automatic differentiation)** — the technique that computes the backward pass automatically from the forward pass, by applying calculus's chain rule to each op.
- **Backward pass** — walking the computation graph in reverse to compute gradients.
- **CUDA** — Nvidia's proprietary GPU computing platform. Not available in browsers.
- **Dispatch overhead** — the CPU work needed to launch each GPU kernel. Hundreds of microseconds per launch.
- **Eager execution** — running each op the moment the user writes it. PyTorch's default mode.
- **Flash Attention** — a fused-kernel implementation of attention that's 5–10× faster than the naive version.
- **Forward pass** — applying ops to inputs to produce outputs.
- **Fusion (kernel fusion)** — combining multiple ops into a single kernel to avoid memory round-trips.
- **GPU buffer** — a region of GPU memory holding tensor data.
- **IR (Intermediate Representation)** — a lower-level form of the user's computation, used for compiler-style optimization.
- **JAX** — a research library from Google that uses tracing + JIT compilation from day one. Architectural inspiration.
- **JIT (Just-In-Time)** — compilation that happens at runtime, the first time the code runs.
- **Kernel** — a small program that runs on the GPU, doing the math for one op (or one fused group of ops).
- **Kernel launch** — the act of telling the GPU "execute this kernel on these buffers."
- **Lazy evaluation** — deferring computation until the result is actually needed.
- **Megakernel** — a single GPU kernel that does the work of many fused ops.
- **MLIR** — a multi-level compiler IR framework. Used by Triton and IREE.
- **NumPy** — the standard Python library for CPU array math. Our current realization backend.
- **NPU (Neural Processing Unit)** — purpose-built silicon for ML inference on devices like Apple M-series, Snapdragon X, recent Intel CPUs.
- **Op** — short for operation. A function on tensors (matmul, softmax, ReLU, etc.).
- **OPFS (Origin-Private File System)** — a persistent filesystem in the browser, per-origin, synchronous access.
- **Pyodide** — CPython compiled to WebAssembly. Lets us run real Python in the browser.
- **PyTorch** — the dominant deep learning library. Eager-by-default; `torch.compile` adds JIT.
- **Realization** — actually executing a lazy graph and producing concrete tensor values.
- **SharedArrayBuffer (SAB)** — JavaScript API for memory shared between Web Workers. Requires cross-origin isolation.
- **Shape signature** — the tuple of (shape, dtype) for each input. Used as a cache key for traced graphs.
- **SIMD (Single Instruction Multiple Data)** — vector operations that process multiple values per CPU instruction. WASM has 128-bit SIMD.
- **Symbolic differentiation** — generating the backward IR by mechanically transforming the forward IR, rather than recording closures at runtime.
- **Tensor** — a multi-dimensional array of numbers, the basic data structure of deep learning.
- **Tensor proxy** — an object that looks like a tensor but holds a recipe for computing values, not the values themselves.
- **Tinygrad** — a small research library by George Hotz. Architectural inspiration for the IR design.
- **Tracing** — running the user's code symbolically to build a computation graph, without executing the math.
- **Triton** — an OpenAI library for writing custom GPU kernels in Python-DSL. Conceptual inspiration.
- **UOp (micro-op)** — a node in tinygrad's IR. LOAD, STORE, MUL, ADD, REDUCE, etc.
- **WASM (WebAssembly)** — a binary instruction format for the web. Near-native CPU performance.
- **Web Worker** — a JavaScript thread separate from the main thread. Multiple workers + SAB = real multi-core.
- **WebGPU** — the modern browser GPU API. Replaces WebGL for compute. Cross-platform.
- **WebNN** — a browser API for ML inference, routes to the best available hardware (NPU/GPU/CPU).
- **WGSL (WebGPU Shading Language)** — the shader language WebGPU uses. Required for any GPU compute.
- **Workgroup memory** — fast on-chip memory in WebGPU, shared by threads in one workgroup. Equivalent to CUDA's "shared memory." Critical for fusion.

---

## Closing thought

The brutal version of why this is worth doing: **PyTorch is a product of the era when GPUs were a remote service.** It assumes you ssh into a machine with a GPU and submit work to it. The dispatch overhead, the eager-by-default semantics, the python-as-orchestrator-of-cuda model — all of that is a product of a specific historical setup.

In the browser, we don't have that setup. The "GPU" is right next to the JavaScript engine; latency between Python and WGSL is microseconds, not milliseconds. The browser is more like an embedded system — heterogeneous silicon, tight memory, but everything close together. The right software architecture for that setup is *not* "miniature PyTorch." It's more like a compiler with a Python frontend.

Building that compiler is hard. But the primitives to build it on already exist, individually. We'd be wiring them together.

If we don't do this, we'll be a slightly-friendlier-than-PyTorch PyTorch. If we do it, we'll be a fundamentally faster system that happens to look like PyTorch on the surface — and that's the only architecture in which "deep learning in the browser" actually competes with "deep learning on a $10k workstation."

We have the primitives. We have a working starting point. The next person to write 4,000 lines of carefully-architected code gets to find out if the bet works.
