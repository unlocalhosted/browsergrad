# Browsergrad: A Web Dev's Crash Course

This document is for a web developer who hasn't written deep learning code before. By the end, you'll understand how the library works, why it's slow, what would make it fast, and roughly what the next year of engineering looks like.

I'll start from web dev concepts you already know and build outward. Every analogy is meant to be load-bearing — when I say "tensor is like a Float32Array," I mean it literally, not as a metaphor.

This is a companion to [VISION.md](VISION.md). VISION goes deep on the new architecture; this primer fills in the *background* you need for VISION to make sense.

---

## Part 0: The shortest possible orientation

You already know how to build web apps. You know JavaScript, the DOM, `fetch()`, `async/await`, Web Workers, maybe a framework like React or Svelte.

You're about to learn how *machine learning* works at the systems level — not the math, just the mechanics. Then you'll see how our library wires up to do machine learning in the browser, why it's currently slow, and what we'd build to make it fast.

The whole field reduces, at the systems level, to one phrase: **matrix multiplication, at scale, repeated millions of times**. Once you internalize that, everything else falls into place.

---

## Part 1: What deep learning is, at the systems level

Forget the AI hype. Forget transformers and LLMs for now. Here's the actual mechanics.

### A "model" is a function with knobs

A deep learning model is a function that takes some input (an image, a sentence, a sensor reading) and produces some output (a category label, a translation, a prediction). The function has *parameters* — numbers, lots of them — that determine its behavior.

If you've used React state, think of parameters as state. The model has a few million pieces of state, and the function reads them every time it runs.

```
function model(input, params) {
  // do some math involving input and params
  return output;
}
```

A small model has thousands of parameters. A medium one has millions. GPT-4 has roughly 1.7 trillion. The parameters are just numbers — usually 32-bit or 16-bit floats — stored in a flat array.

### "Training" is hill-climbing on those knobs

Training is the process of tuning the parameters so the function does what you want. You feed it lots of (input, correct-output) pairs and nudge the parameters in the direction that makes its predictions less wrong.

```
loss = how_wrong_was(model(input, params), correct_output)
gradient = derivative_of_loss_with_respect_to(params)
params = params - learning_rate * gradient
```

That last line — "subtract a tiny step in the direction of the gradient" — is repeated millions of times. Each repetition is called a **training step**. The whole process is called **gradient descent**, and the technique that figures out the gradient automatically is called **autograd** (automatic differentiation).

A web dev analogy: imagine you're shipping a feature and there's a config knob with a million dials. Every dial slightly affects whether users like the feature. You can't tune them by hand. So you write code that:
1. Tries the current settings.
2. Sees how unhappy users are.
3. Figures out which way to nudge each dial to make users less unhappy.
4. Nudges, repeats.

That's training. The "configuration" is the parameters; the "unhappiness" is the loss; the "which way to nudge" is the gradient; the "automatic" part is autograd.

### "Inference" is just running the function once

Once you've trained the parameters, inference is the cheap part — you just call `model(input, params)` and get the output. No nudging, no gradients. This is what runs when a user clicks "translate" or "auto-complete" or "is this image a cat."

### Why this is hard

Three things make this hard at the engineering level:

1. **The numbers**. A model has millions of parameters. Every training step touches all of them. You need fast math.
2. **The math is repetitive but huge**. Every step is the same shape of computation (matmul, add, activation, repeat) but with massive arrays. CPUs are bad at this. GPUs are great at this.
3. **Autograd**. To compute the gradient, you have to track every math operation in a graph, then walk that graph backward applying chain-rule rules. This is bookkeeping the library has to do for you.

---

## Part 2: Why this is all about matmul, and why matmul is about GPUs

Here's the key shape-of-the-workload insight.

A neural network is almost entirely matrix multiplications. Linear layers, attention, convolutions — all of them are matmul at their core, sometimes with a small wrapper around it. A typical "transformer block" might be 70% matmul by FLOP count (FLOPs = floating-point operations = "how many multiply-adds happened").

So if you make matmul fast, you've made neural networks fast.

### Why CPUs are bad at matmul

A CPU is optimized for *sequential* code with lots of branching. It has 8 or 16 cores, deep pipelines, complex branch prediction, large caches per core. It's brilliant at running a browser, an OS, a database.

It's *bad* at matmul because matmul is the opposite kind of workload: do the same multiply-add ten billion times, with no branches. The CPU's complexity is wasted; we're just multiplying numbers.

A modern CPU might get 100 gigaFLOPs (10^11 multiply-adds per second) on dense matmul if you carefully use SIMD instructions. That sounds like a lot until you compare it to a GPU.

### Why GPUs are good at matmul

A GPU is the opposite design. It has thousands of tiny cores (instead of 8 big ones), each one slow individually, with shallow pipelines and almost no branch prediction. But they all run the same code at the same time on different data. This is called SIMD — Single Instruction Multiple Data.

For graphics — the original use case — this is perfect: every pixel runs the same shader code. For matmul, it's also perfect: every element of the output runs the same multiply-add code.

A modern consumer GPU (RTX 4090, M3 Max) does 30-80 *teraFLOPs* (10^13 multiply-adds per second). That's 300-800× faster than the CPU on the same task. *That's* why ML happens on GPUs.

### How code gets to the GPU

The CPU and GPU are separate processors with separate memory. To run code on the GPU, you have to:

1. Write a **kernel** — a small program that describes "what one thread does." A kernel is written in a shading language (GLSL, HLSL, CUDA C, WGSL, MSL — these are all variants).
2. Copy the input data from CPU memory to GPU memory.
3. Tell the GPU "launch this kernel on N threads."
4. Wait for the GPU to finish.
5. Copy the output data back to CPU memory.

Each of those steps has overhead. The kernel launch itself takes maybe 10 microseconds on a modern setup. The data copy is slow if you do it a lot. If you launch a thousand kernels per second with small data, you spend most of your time on the orchestration and almost none on the math. **This is the dispatch overhead problem**, and it's the central enemy of every ML library's performance.

### The CUDA monopoly

Nvidia spent two decades building **CUDA** — a programming model + runtime + libraries + drivers that lets you write GPU code in C-like syntax and run it on Nvidia GPUs. PyTorch and TensorFlow are built on top of CUDA. So is most of the deep learning ecosystem.

CUDA is proprietary. It only works on Nvidia hardware. If you have an Apple Silicon Mac, an Intel Arc GPU, an AMD Radeon, or any phone, CUDA doesn't help you. PyTorch has Metal (Apple) and ROCm (AMD) backends, but they're second-class citizens.

For the browser — where the user might have *any* of these — we need a vendor-neutral GPU API. That API is **WebGPU**, and it's the entire reason browser-side deep learning is possible at all.

---

## Part 3: How the current library works, end to end

Let's trace through what actually happens when a user writes this code in the browser:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)
x = torch.randn(3, 10)
y = model(x)
print(y.tolist())
```

This is the simplest possible "run a neural network" example. Three lines of meaningful work: build a linear layer, generate random input, run the forward pass. Behind the scenes, here's what happens.

### The stack, top to bottom

The browser is running JavaScript. To run Python, we use Pyodide. **Pyodide** is the real CPython interpreter compiled to WebAssembly (WASM). When the browser loads Pyodide, it loads about 10MB of WASM that contains a working Python 3.12 interpreter. The Python code the user writes runs *inside* that interpreter, which runs *inside* WASM, which runs *inside* the browser.

```
┌─────────────────────────────────────────────┐
│ Browser tab (JavaScript)                     │
│  ┌────────────────────────────────────────┐  │
│  │ Pyodide WASM runtime                    │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │ CPython 3.12                      │  │  │
│  │  │  ┌──────────────────────────┐    │  │  │
│  │  │  │ User's Python code        │    │  │  │
│  │  │  │ User's import of torch    │    │  │  │
│  │  │  │ (aliased to browsergrad)  │    │  │  │
│  │  │  └──────────────────────────┘    │  │  │
│  │  └──────────────────────────────────┘  │  │
│  │ NumPy WASM (also CPython-imported)      │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

When the user calls `nn.Linear(10, 5)`, they're calling code that we put inside Pyodide's filesystem (this is what `installGrad()` does — it writes our Python source into Pyodide's virtual file system and adds it to `sys.path`). Python imports it like any other library.

### What `model(x)` actually does

In the user's mental model, `model(x)` runs the forward pass. In reality, here's the chain:

1. `model(x)` calls `Linear.__call__(x)`, which calls `Linear.forward(x)`, which is Python code we wrote.
2. `forward()` says `return x @ self.weight.T + self.bias`. Python's `@` operator triggers `Tensor.__matmul__`.
3. `__matmul__` is our code. It calls NumPy's `np.matmul()` on the underlying arrays. NumPy is itself a C library compiled to WASM, called by Python via Python's normal foreign-function interface.
4. NumPy does the matmul on the CPU (because that's all NumPy does in WASM — there's no GPU integration).
5. The result is wrapped back into a `Tensor` object. If `requires_grad` was True on any input, we also record an autograd entry: a tuple of (parent tensors, backward function) attached to the result.
6. `__add__` for the bias does the same dance.
7. The final result is a `Tensor` whose data lives in NumPy land but whose Python wrapper knows how to compute gradients.

At no point did anything touch the GPU. Every multiplication happened on the CPU, in NumPy's compiled WASM code.

### Where each piece of code lives

We organize the library this way:

- **Frontend (Python source)**: lives at `packages/browsergrad-grad/src/python/`. After the recent refactor, this is a bunch of `.py` files (split into chunks for `nn.py`). A codegen step turns them into TypeScript constants at build time.
- **Installer (TypeScript)**: lives at `packages/browsergrad-grad/src/install.ts`. Job: write all those Python files into Pyodide's filesystem when the page loads.
- **Adapter (TypeScript)**: `packages/browsergrad-grad/src/node-adapter.ts` and the runtime package. These wrap Pyodide into a `GradTarget` interface so the installer doesn't care whether you're running in Node or the browser.
- **Kernels (WGSL)**: lives at `packages/browsergrad-kernels/`. Hand-written GPU kernels in WGSL. **Currently not wired up to the grad library.** They exist as a future fast-path option.

That's the whole library, structurally. The Python files define a PyTorch-shaped API. The autograd is built using Python operator overloading and runtime closures. The math is NumPy. The "GPU kernels" package exists but isn't used.

### A web dev analogy for the autograd

If you've used MobX or Vue's reactivity system, the autograd has a similar feel. In MobX, every read of an observable is tracked; the framework builds a graph of "this computed value depends on these observables." When an observable changes, MobX walks the graph backward and re-runs anything that depends on it.

Our autograd is the same idea but for math operations. Every Tensor op is tracked: "this result was produced by matmul of parent1 and parent2, with this specific backward function." When the user calls `.backward()`, we walk the graph in reverse and apply each backward function to get the gradient.

The implementation is dead simple — about 100 lines for the core. Tensor has an attribute `_ctx = (parents, backward_fn)`. Backward walks the graph, calls each `backward_fn` in topologically sorted order, accumulates gradients into each parent's `.grad` field.

---

## Part 4: Where the time actually goes

Now the unpleasant part. Let's measure where the time is going.

Imagine we run a tiny training loop: a model with two linear layers, 32 inputs, 16 hidden, 8 outputs, batch of 64. One forward + backward + optimizer step. Real PyTorch on a laptop CPU does this in maybe 200 microseconds. Our library does it in roughly 50 milliseconds — 250× slower.

Where does the time go?

### The four costs

1. **Python interpreter overhead**. Every line of Python in our source — `forward()`, `__matmul__`, `_build_ctx`, the autograd traversal — runs through the CPython interpreter. CPython is slow. Every attribute access is a hashtable lookup; every function call is a stack frame allocation. For tight loops over millions of elements this would be catastrophic, but for our case (a few thousand Python lines per training step) it costs maybe 5-10ms per step.

2. **NumPy round-trip**. Every Tensor op extracts the underlying NumPy array, calls a NumPy function, wraps the result. NumPy itself is fast (vectorized C), but the Python→NumPy boundary involves copying argument arrays into NumPy's internal format. For each op, this is maybe 20-50 microseconds. Across 50 ops per training step, that's another 1-2.5ms.

3. **Pyodide ↔ JavaScript boundary**. Pyodide runs *inside* a WASM module. Every time Python code reads from JavaScript (or vice versa), there's an overhead crossing that boundary. For our library this is rare during a training step (we only cross when loading code or returning a result), so this is minor — maybe 0.1ms.

4. **The math itself**. NumPy doing the actual matmul, on CPU, in WASM. WASM is slower than native (it can't use SIMD as efficiently as native code, and Pyodide's NumPy build doesn't enable all optimizations). A 64×32 matmul that would take 5 microseconds on native NumPy takes maybe 50 microseconds in Pyodide-NumPy. Across all the ops, this is maybe 30-40ms.

Add it up: ~5 + ~2 + ~0.1 + ~35 ≈ 42ms. The numbers are rough but the breakdown is right. **The math itself is most of the time, but we're doing it on the CPU when the user has a GPU sitting idle in their machine.**

### The cliff

Here's what makes this worse than it looks. As the model gets bigger, the constant overheads (Python, NumPy boundary, Pyodide) stay roughly the same, but the math grows. For small models, overheads dominate and we're 250× slower than PyTorch. For large models, math dominates and we're roughly 50× slower than PyTorch-on-CPU and roughly 1000× slower than PyTorch-on-GPU.

That's the cliff. Today, browsergrad is "fast enough" for educational toy problems — tiny MNIST classifiers, small transformer demos, MLPs on tabular data. The moment a lab tries to train ResNet18 on real CIFAR, training a single epoch takes hours instead of seconds. The lab is unusable.

### Why this isn't just "make NumPy faster"

You might think the fix is "make NumPy in Pyodide faster," and there's a little of that available (better SIMD support, threading). But the gap between *best possible CPU* and *modern GPU* is roughly 500×. No amount of CPU optimization closes that gap. The only way is to actually use the GPU.

---

## Part 5: The browser primitives we have to work with

The browser is no longer just an HTML renderer. In 2026 it's a full computing platform with primitives for high-performance compute. Most web devs haven't met some of these yet. Let's walk through them.

### Pyodide

You already met it. CPython compiled to WASM, plus NumPy, SciPy, pandas, and a long tail of pure-Python and C-extension libraries. About 10MB to load. Runs at maybe 1/5 to 1/3 of native Python speed. Critical to our pitch — without Pyodide there's no Python in the browser, and without Python we can't be PyTorch-shaped.

Mental model: like having a server-side Python service that happens to live in the same tab as your JavaScript, with synchronous calls between the two.

### WebGPU

The modern browser API for GPU compute. Released stable in Chrome in 2023; Safari shipped it in 17.4; Firefox is mostly there. Replaces WebGL for compute purposes.

Conceptually it's the JavaScript equivalent of Vulkan or Metal. You:
1. Request a `GPUAdapter` (the browser picks the best GPU on the device).
2. Create a `GPUDevice` (your handle to that GPU).
3. Allocate buffers (regions of GPU memory you can read/write).
4. Compile a kernel from WGSL source.
5. Bind your buffers and dispatch the kernel.

A minimal compute example:

```javascript
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const buffer = device.createBuffer({
  size: 1024 * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

const shader = device.createShaderModule({
  code: `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      data[id.x] = data[id.x] * 2.0;
    }
  `,
});

const pipeline = device.createComputePipeline({
  layout: 'auto',
  compute: { module: shader, entryPoint: 'main' },
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, /* bindings */);
pass.dispatchWorkgroups(16);  // 16 workgroups × 64 threads = 1024 threads
pass.end();
device.queue.submit([encoder.finish()]);
```

That dispatched 1024 threads in parallel on the GPU, each one doubling one element of an array. The whole thing — compile shader, dispatch, finish — takes a few hundred microseconds. For the GPU to do meaningful work, you want each dispatch to handle a lot of math, not a tiny array.

Web dev analogy: WebGPU is like queueing a task in Web Workers, except the "worker" is the GPU and you have thousands of them all running the same code simultaneously.

### WGSL

The shader language WebGPU expects. WGSL stands for *WebGPU Shading Language*. It's syntactically similar to Rust, semantically similar to GLSL or HLSL. You write kernels in WGSL as strings inside JavaScript and pass them to `device.createShaderModule()`.

A WGSL kernel describes "what one thread does." The GPU runs N copies of that kernel in parallel, each one with a different thread index (which it uses to figure out which element of the array to work on).

Important caveat: WGSL has no `printf`, no real debugger, no exceptions. If your kernel has a bug, you get wrong numbers or a tab crash. Most of the engineering pain of GPU code is debugging shaders without these tools.

### WebNN

Even newer. WebNN stands for *Web Neural Network API*. It's a spec for "describe a neural network computation graph, ask the browser to run it on the best available hardware."

The "best available hardware" is the killer feature. On an M-series Mac, WebNN can dispatch to the Apple Neural Engine — purpose-built silicon that does matmul faster *and more efficiently* than the GPU. On a Snapdragon X laptop, it uses the Hexagon NPU. On a recent Intel CPU, it uses AVX-512 or AMX. On older machines, it falls back to GPU or CPU.

The API itself looks like:

```javascript
const builder = await navigator.ml.createModelBuilder();
const input = builder.input('x', { type: 'float32', dimensions: [batch, features] });
const w = builder.constant({ type: 'float32', dimensions: [features, hidden] }, weights);
const output = builder.matmul(input, w);
const model = await builder.build({ output });
const result = await model.compute({ x: input_buffer });
```

You describe the graph in TypeScript, hand it to WebNN, and the browser decides how to execute it. We don't pick the hardware; we just say "this is a matmul of these shapes."

Status: shipping in Chrome behind a flag, stable in some configurations, spec still moving. Worth designing for but not betting everything on.

### WASM SIMD

WebAssembly added 128-bit SIMD instructions in 2021. SIMD means "do four float adds in one instruction instead of four." It's a 2-4× speedup for code that can use it. NumPy in Pyodide doesn't fully exploit SIMD because of how it's compiled, but a custom WASM module can.

For us, this is the fallback path: if the user has no WebGPU, we want WASM SIMD doing the math, not naive scalar JavaScript. There are libraries (xtensor, Eigen-compiled-to-WASM) that handle this.

### Web Workers + SharedArrayBuffer + Atomics

A Web Worker is a separate JavaScript thread. SharedArrayBuffer (SAB) is a memory region you can share between workers. Atomics are operations like "atomic compare-and-swap" that let workers coordinate without locks.

Together they give us *real multi-threading*. We can spawn one worker per CPU core, give them shared access to the same tensor data, and parallelize the work across cores. This is how Tier 4 in the VISION architecture works.

Requirement: the page must be *cross-origin isolated* (specific HTTP headers set). Without that, SharedArrayBuffer doesn't work. This is a real deployment constraint.

### OPFS (Origin-Private File System)

A persistent filesystem inside the browser, per-origin (so amazon.com can't see github.com's files), with synchronous read/write inside a worker. You can store gigabytes. It survives page reloads, browser restarts, even (sometimes) tab closures.

This is the cache layer. Compiled WGSL pipelines, traced ML graphs, weight tensors all go here. First page load is slow; second load is instant because everything is already on disk.

Web dev analogy: like Cache API or IndexedDB, but actually fast and actually a filesystem. The fact that this exists is one of the under-appreciated browser stories of the last few years.

### A unified picture

```
JS top-level (the page)
├─ WebGPU (for fast math)
├─ WebNN (for even faster math on supported hardware)
├─ Web Workers (for CPU parallelism)
│   └─ SharedArrayBuffer (shared memory between workers)
│   └─ WASM SIMD (vectorized CPU code)
│       └─ Pyodide (the Python interpreter)
│           └─ User Python code calling browsergrad
├─ OPFS (the persistent cache)
└─ Service Worker (for network caching, optional)
```

These are all stable enough to bet on in 2026. None are research-stage. The hard part isn't "do these primitives work?" It's "how do you compose them into a system that's easy to use?"

---

## Part 6: The two problems, named precisely

Now that you understand the landscape, here are the two specific problems we need to solve.

### Problem A: We never touch the GPU

The user's machine has a GPU. The browser exposes it via WebGPU. We don't use it. Every op runs on the CPU, in NumPy, inside Pyodide, inside WASM.

Solving this means writing GPU kernels (or using WebNN where available) for the heavy ops — matmul, attention, conv. Once a single op lives on the GPU, we get massive speedups on that op alone.

This is the *easy* problem. The kernels package already has a head start. The wiring from Python down to WebGPU is fiddly but well-understood.

### Problem B: Even with GPU, eager mode is slow

Suppose we wire up every op to dispatch to a WGSL kernel instead of NumPy. We'd be faster, but not as fast as PyTorch — because PyTorch with `torch.compile` fuses many ops into one kernel and gets 2-4× over its own eager mode.

Here's the issue. A modern transformer block has 20+ ops. If each one is one kernel launch, you do 20 launches. Each launch costs ~50 microseconds of CPU work plus a memory round-trip. The kernels themselves might only take 100 microseconds total. You spend half the time on overhead.

A fused kernel does all 20 ops in one launch, keeping intermediate data in on-chip GPU memory. That's the Flash Attention move. PyTorch's `torch.compile` does this automatically. We need to do it too.

This is the *hard* problem. It requires building a compiler — a thing that takes a graph of ops and emits a fused kernel.

---

## Part 7: The compiler architecture, framed for a web dev

If you've worked with bundlers (Webpack, esbuild, Vite), you already have the right mental model. A bundler:

1. Reads many JavaScript modules.
2. Builds a dependency graph.
3. Resolves imports, applies optimizations (tree-shaking, minification, code-splitting).
4. Emits a few bundled files instead of hundreds of individual modules.

Our compiler does the same thing but for *math*. It:

1. Watches the user's Python code run.
2. Builds a graph of every op.
3. Resolves the graph, applies optimizations (fusion, dead code, algebraic simplification).
4. Emits a few GPU kernels instead of hundreds of individual op dispatches.

Same mental shape. Different domain.

### The five layers, recapped for context

1. **Frontend** — what the user types. PyTorch-shaped Python. We don't change this; the user shouldn't notice.

2. **Tensor proxies (lazy capture)** — instead of computing right away, every op returns a *proxy* that records what would have been computed. Think of it as Promises that haven't resolved: you can chain them, but no work has happened yet.

3. **IR + fusion** — the recorded ops are translated to a lower-level form (an *intermediate representation*, IR) and a fusion pass combines ops that should run together. This is where the optimization happens.

4. **Dispatch** — for each fused kernel, pick the best backend. WebNN if available; WebGPU otherwise; WASM SIMD if WebGPU isn't there; NumPy as the floor.

5. **Persistence** — cache the compiled kernels and traced graphs to OPFS so the second page load is instant.

### Where the speedup comes from

Let me concretize this. Suppose the user trains for 1000 steps. With the current library, each step does:

- 5ms Python overhead
- 35ms NumPy math on CPU
- 0.5ms autograd graph traversal
- = 40.5ms per step
- = 40 seconds total

With the JIT architecture:

- **First step**: ~2000ms. We trace, build IR, fuse, compile WGSL pipelines, run. Slow because of compilation.
- **Steps 2-1000**: ~3ms each. The trace is cached, the kernels are compiled, every step just binds buffers and dispatches. The actual math runs on the GPU.
- = 2000 + 999 * 3 = ~5 seconds total

Ratio: roughly 8× speedup for this size. On bigger models, the ratio goes to 50× because the math grows but the overhead doesn't.

And — critically — second page load:

- All kernels already in OPFS. First step is ~10ms. Steps 2-1000 are 3ms each. Total ~3 seconds.

That's the experience we're chasing. **First visit: a few seconds of compile. Every visit after: instant.**

### Why JIT, not AOT

You might ask: why compile at runtime? Why not ship a pre-compiled binary?

Two reasons.

First, the user's model isn't known ahead of time. They're writing it in Python *in the browser*. Their architecture, their input shapes, their hyperparameters — all decided live. We can't pre-compile what we don't know.

Second, GPUs are heterogeneous. The same WGSL code compiles differently for Apple Silicon vs Nvidia vs Intel. We could pre-compile for known targets, but we'd ship a different binary for every device. Easier to ship the source and let the browser compile on demand.

This is the same reason JavaScript engines JIT-compile JS instead of shipping native binaries. Same trade-off, same conclusion.

---

## Part 8: Walking through what changes

Here's what the current code does for our simple example, and what the new code would do.

### Current path

```python
y = model(x)
loss = ((y - target) ** 2).mean()
loss.backward()
```

Execution:
1. `model(x)` calls `Linear.forward`, which does `x @ W.T + b`.
2. `__matmul__` calls NumPy. **CPU dispatch.** Result is a Tensor with autograd ctx.
3. `__add__` calls NumPy. **CPU dispatch.** Result is a Tensor.
4. `(y - target)` calls NumPy. **CPU dispatch.**
5. `** 2` calls NumPy. **CPU dispatch.**
6. `.mean()` calls NumPy. **CPU dispatch.**
7. `.backward()` walks the autograd graph, calling 5 backward functions, each calling NumPy.

Total: about 10 NumPy CPU dispatches, ~40ms.

### JIT path

```python
y = model(x)
loss = ((y - target) ** 2).mean()
loss.backward()
```

Same user code. Different internals:

1. `model(x)` runs Python, but every op returns a proxy. **No math happens.** We just build a graph node.
2. `__matmul__`, `__add__`, etc., all return proxies.
3. `.mean()` returns a proxy.
4. `.backward()` is the realization point. We hand the graph to Layer 3.
5. Layer 3 traces the graph, generates IR, fuses (matmul+add can fuse; (y-target)**2 can fuse with mean; backward kernels are derived symbolically).
6. Result: maybe 3 fused kernels for forward, 3 for backward.
7. Layer 4 dispatches each kernel. **First time**: compile the WGSL, ~600ms total. Cache to OPFS.
8. Run the kernels on the GPU. ~2ms.

First call: ~600ms (dominated by compilation). Second call (next training step with the same model shape): ~2ms (cache hit, just run).

### What the user sees

Nothing different in their code. They might see a one-second pause when they first call `model(x)` (we could put a tiny "warming up" indicator). After that, training is fast.

This is the same UX pattern as a JIT-compiled JS engine. First run is slow as V8 figures out the hot path. Every subsequent run is fast.

---

## Part 9: Why hasn't someone already done this?

It's worth being suspicious of any "great idea no one has done." Let me walk through the actual reasons.

### PyTorch was designed for a different era

PyTorch shipped in 2016, designed around a CUDA-only world where you ssh into a remote machine to run your code. "GPU code" meant CUDA. There was no WebGPU, no WebNN. The whole notion of "deep learning in the browser" was absurd.

PyTorch's eager-by-default design was a deliberate response to TensorFlow 1.x's graph-only design — researchers wanted to write Python that felt like normal Python, not declarative graph specs. Eager-by-default was the right call for productivity. It made debugging easier. It also made compilation hard, which is why `torch.compile` is a 2023 addition rather than a day-one feature.

For our setup — small models, browser-bound, fresh design space — the trade-offs are completely different. We can afford a tracing JIT because our users aren't writing million-line research codebases; they're writing 200-line educational labs that will be JIT-compiled in one shot.

### WebGPU is recent enough that most ML projects haven't reckoned with it

WebGPU stable shipped in Chrome in 2023. Safari got it in 17.4 (March 2024). Firefox is still partial. Most ML libraries that tried browser support were built on WebGL2 + GLSL fragment shaders, which was the only option for years. WebGL2 is terrible for compute (you have to pretend you're doing graphics). WebGPU is the first time browser compute has been first-class.

Tinygrad, TensorFlow.js, ONNX Runtime Web — all have WebGPU backends now, but they're recent. The space is actively being built. We're not late; we're roughly on time.

### The economics aren't there for FAANG

Browser-side ML is a small market compared to "training $1T-parameter models on H100 clusters." Google's TensorFlow.js team is maybe 10 people. Meta is not investing here. OpenAI is not investing here. The market opportunity is "tools and education," not "platform for serious training."

This actually plays in our favor. We're not racing PyTorch or TF.js — we're filling a niche they don't care about, with a focus (educational labs, zero-install) they're not optimizing for.

### The labor is finicky

Building a tracing compiler with kernel fusion takes a focused engineer 6+ months. Most ML projects can't justify that without a clear ROI. We can, because we have a specific use case (educational labs need to feel snappy) and a specific user (the craftingattention platform) that benefits.

This is the right kind of project for a small, focused team with a 1-year horizon.

---

## Closing

You came in not knowing what a tensor was. You should now understand:

- A deep learning model is a function with parameters; training is gradient-descent on those parameters.
- The workload is matmul-heavy, which is why GPUs (not CPUs) win.
- Our current library does it all on CPU via NumPy-in-Pyodide. Slow.
- The browser has primitives (WebGPU, WebNN, WASM SIMD, OPFS) that let us do real GPU compute.
- The real performance ceiling is dispatch overhead — each op being one launch. The fix is kernel fusion, which requires a compiler.
- A tracing JIT with IR + fusion + multi-tier dispatch is the architecture that solves this.
- This is buildable. It's roughly 6 months of focused work. The primitives exist. The labor exists. The user need exists.

The library today is the prototype. The library tomorrow is the compiler. The architecture in [VISION.md](VISION.md) is what that compiler looks like.

Read VISION.md next. Now that you have the background, it'll make sense.
