# BrowserGrad Context

## Domain Terms

- **Primitive package**: reusable BrowserGrad package whose public interface is
  useful outside one course, assignment, or platform. Public names should speak
  in domain primitives: byte BPE, kernel, tensor, snapshot, scaling law,
  alignment loss, data filter, task graph, worker mesh.
- **Curriculum profile**: assignment- or course-specific JSON/docs that map an
  upstream lab onto BrowserGrad primitives. Profiles may mention CS336, CS149,
  GPU Puzzles, or Crafting Attention; primitive packages should not depend on
  those names.
- **Capability**: named runtime affordance used for launch decisions, such as
  `webgpu`, `wgsl-kernel`, `byte-bpe`, `worker-mesh`, or `external-runner`.
- **Substrate**: execution host behind a profile route: Pyodide, JavaScript,
  WebGPU/WGSL, Worker mesh, or external native runner.
- **Oracle adapter**: profile-level object that exposes primitive package
  helpers to rubrics. Adapter names should name the primitive family, not the
  assignment that first needed it.
- **Benchmark assignment**: upstream assignment used to pressure-test
  BrowserGrad. Benchmarks shape requirements but must not become package
  identity.

## Architecture Rules

- Core packages stay primitive-first and assignment-agnostic.
- Course and assignment facts live in curriculum profiles, handoff docs,
  fixtures, and platform adapters.
- A public package export must pass this test: someone building a non-course
  browser ML tool can plausibly use it without learning a course acronym.
- HIPScript/CUDA-style work belongs under the generic kernel authoring core:
  simulator trace first, WGSL/WebGPU lowering next, native runners optional.
