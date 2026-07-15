# Browser-Native Kernel and Compiler Foundation

BrowserGrad’s kernel platform is a serious execution and compiler effort that
happens to be browser-native. Courses and labs are important consumers because
they demand inspectable, deterministic behavior, but they do not define a
weaker semantic target.

The platform must be honest about WebGPU: a browser does not expose CUDA
streams, Tensor Cores, WGMMA, TMA, peer memory, or NCCL. The answer is not to
pretend those facilities exist, nor to reduce the source language to a
collection of worksheet patterns. The answer is a shared semantic machine with
portable and native execution tiers.

The normative requirements are in
[BrowserGrad Semantic Systems Architecture and Low-Level Requirements](./package-requirements-lld.md).
This document describes how kernel/lab work follows that contract.

## Direction

```text
actual source language and tensor/layout semantics
  -> frontend facts -> value/layout semantics -> effectful kernel semantics
  -> schedule IR + host execution graph
  -> CPU conformance | portable WGSL/WebGPU | optional native companion
  -> labs, notebooks, profiling, and rubric adapters
```

The teaching surface must expose the real concepts of views, layouts, memory
spaces, tiles, reductions, synchronization, masks, and numerical behavior.
It may present these concepts progressively, but it cannot swap them for a
different semantics behind an API that claims CUDA, CuTe, or tensor behavior.

## Current Substrate

The following components are useful implementation substrate and remain
supported with their current names and boundaries:

- `@unlocalhosted/browsergrad-kernels` owns WGSL device resources, dispatch,
  prepared sequences, resident buffers, and JavaScript reference execution.
- `createKernelRubric()` provides structured CPU-oracle assertions for browser
  rubrics. It is a rubric/reference facility, not a GPU execution substitute.
- `runThreadGrid()` gives deterministic thread/block traces for concept and
  correctness analysis. It is a simulator, not a CUDA performance runtime.
- `defineKernel1DProgram()` has a browser-owned 1-D program representation
  with CPU reference and WGSL lowering. It remains useful for direct WGSL and
  introductory execution work, but is not the universal tensor/layout IR.
- `@unlocalhosted/browsergrad-compiler` owns the current CUDA-lite frontend,
  canonical Kernel IR, semantic CPU reference, WGSL emission, and WebGPU
  execution planning.

These components must converge on shared tensor/layout/tile semantics for
advanced workloads. They must not become parallel, source-shaped compatibility
stacks.

## Required Compiler Shape

The current CUDA-lite parser is a shipping frontend with finite coverage. It
is not the promised model for arbitrary C++ or CuTe source. For real CuTe and
CUTLASS support, BrowserGrad requires a versioned, standards-aware C++ frontend
that consumes the actual upstream headers and performs preprocessing, lookup,
templates, overload resolution, and diagnostics before semantic lowering.

The lowering target must distinguish:

| Semantic concept | Why it is necessary |
| --- | --- |
| `Layout` algebra | CuTe composition, hierarchy, slicing, coordinate mapping, and static swizzles cannot be represented safely as source patterns. |
| `Tensor<Engine, Layout>` | A tensor combines addressability/engine information with a layout; the compiler must preserve both. |
| `TensorView` | Dynamic pointers, offsets, rank/shape, stride/layout, element type, address space, alignment, and nullable/indirect bindings require an explicit runtime ABI. |
| `IndexMap` | CPU reference and backends need the same typed coordinate program; affine maps are an important optimization subset, not the whole model. |
| `Tile` and collective operations | Tiled copies, reductions, MMA, shared-memory staging, masks, and barriers need structured semantics and uniformity preconditions. |
| `HostExecutionGraph` | Multi-dispatch launches, copies, events, and browser-hosted sequences are real program behavior and must not be hidden in emitter heuristics. |

Every frontend reports separate status for source acceptance, semantic
representation, CPU reference execution, portable WebGPU execution, and native
execution. “Supported” without a tier is not a valid platform claim.

## Portable and Native Backends

### Portable WebGPU

The portable backend lowers the canonical program to WGSL and performs actual
browser device execution. It may choose a correct workgroup-tiled algorithm
when the source requests tiled MMA or attention. It must publish feature
requirements (`shader-f16`, subgroup features, limits) and actual measurement
availability.

The direct attention kernel currently serves as a fused row-wise online-softmax
baseline. It is useful correctness and performance substrate. It is not
block-tiled FlashAttention until it stages K/V tiles (or proves an equivalent
memory strategy), maps a query tile, synchronizes correctly, maintains online
softmax across tiles, and passes tile-boundary conformance tests.

### Native companion

The native companion consumes the same verified value/layout, kernel, and host
artifacts on CUDA/native environments. It is the appropriate product for
hardware-specific atoms, Tensor-Core behavior, TMA/WGMMA pipelines, peer-memory
experiments, and NCCL-like multi-device collectives. It may add specialized
schedules and lowering; it may not change source/layout/index semantics and
must state its preservation level.

### Distributed work

Browser worker meshes and native multi-device execution are separate products:

- Browser labs use explicit message transport and deterministic collectives.
- Native/remote environments use real devices, peer-memory topology, and
  communication libraries.

Do not describe one as the other.

## Workload-Driven Development

The platform’s capability tests are serious systems workloads, not an API
checklist:

1. Dynamic rank-2/3 views: transpose, strided slices, broadcasts, packed
   attention heads, padded tensors, and NCHW/NHWC-style transformations.
2. Actual upstream CuTe layout and `Tensor<Engine, Layout>` fixtures at pinned
   versions.
3. Tiled GEMM with explicit view/layout maps, shared-memory staging, masks,
   vectorized paths where legal, and differential CPU/WebGPU checks.
4. Attention progression: row-wise online baseline, tiled attention, online
   softmax across tiles, then correctly named fused/block-tiled attention.
5. A teaching-scale GPT stack: embedding, normalization, GEMM, attention,
   residual, classifier, and optimizer steps, progressively fused only when
   the canonical semantics and evidence exist.
6. Multi-kernel host graphs, optimizer pipelines, and distributed collectives
   with explicit dependency/order semantics.

The first cross-cutting flagship is an unmodified pinned CuTe-style tiled
attention source program. It forces C++ source compatibility, layout/tensor
lowering, dynamic views, tile semantics, CPU conformance, WGSL execution, and
honest performance comparison against the row-wise baseline.

## Rules for Lab and Curriculum Authors

- Use real WGSL, the current CUDA-lite frontend, or the eventual C++/CuTe
  frontend according to the capability required by the task. Do not invent a
  BrowserGrad-only tensor/layout syntax to stand in for CuTe.
- A simulation or CPU oracle is valuable for explanation and debugging; label
  it as such and do not count it as real WebGPU or native execution evidence.
- Capability gates must describe the missing tier and provide a browser-safe
  alternative only when the alternative’s semantics are named.
- The platform remains Pyodide-optional for JavaScript/WGSL labs. Python is a
  consumer of the kernel platform, not its ownership boundary.
- Put broadly useful compiler, view, layout, tile, and backend work in
  packages. Keep assignment-specific wording, fixture policy, and rubrics in
  profiles or platform documentation.

## Evidence Rules

A kernel/compiler claim needs the evidence appropriate to its tier:

| Claim | Minimum evidence |
| --- | --- |
| CuTe/CUTLASS source compatibility | Pinned upstream source fixture through the real C++ frontend and semantic inspection. |
| Tensor/layout behavior | Coordinate-map/property tests plus CPU reference tests over offsets, strides, masks, broadcasts, and tile boundaries. |
| Portable WebGPU execution | Actual device execution matching the CPU reference, with skipped environments reported as not run. |
| Hardware-specific acceleration | Native test proving the named facility and matching the same canonical reference. |
| Performance | Recorded hardware/browser configuration, named baseline, and separate correctness proof. |

When a feature is absent, report the semantic or backend capability that is
missing. Do not turn the limitation into a vague “lab subset” label.
