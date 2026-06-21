# Feature PRDs

Each PRD here is a detailed engineering brief for one feature in the 14-month roadmap defined by [../../PRD.md](../../PRD.md). One feature = one PRD = one (or a small set of) commits when implementation lands.

Every PRD includes: background with cited evidence, user stories, architecture (with diagrams), concrete API surface, week-by-week implementation plan, acceptance criteria, test strategy, risks, open questions, references.

## Index

### P0 — months 1–3 (close gaps, first GPU acceleration, prove on one real lab)

- [PRD-001 — PyTorch coverage closeout](PRD-001-pytorch-coverage-closeout.md)
- [PRD-002 — First GPU acceleration via kernels package](PRD-002-first-gpu-acceleration.md)
- [PRD-003 — Cold-start UX optimization](PRD-003-cold-start-ux.md)
- [PRD-004 — Lab runtime API + end-to-end validation](PRD-004-lab-runtime-api.md)

### P1 — months 4–10 (the JIT epoch foundation)

- [PRD-005 — `browsergrad-jit` MVP (IR + tracer + NumPy backend)](PRD-005-jit-foundation.md)
- [PRD-006 — Kernel fusion (elementwise + reduce/softmax)](PRD-006-kernel-fusion.md)
- [PRD-007 — Symbolic backward](PRD-007-symbolic-backward.md)
- [PRD-008 — OPFS pipeline cache + safetensors streaming](PRD-008-persistent-caching.md)
- [PRD-009 — Gradient checkpointing (real, not stub)](PRD-009-gradient-checkpointing.md)
- [PRD-010 — Real mixed precision (fp16 storage + fp32 accumulators)](PRD-010-mixed-precision.md)

### P2 — months 11–14 (megakernels, hardware tiers, transforms, lab content)

- [PRD-011 — WebNN backend tier](PRD-011-webnn-backend.md)
- [PRD-012 — Megakernel codegen for transformer blocks](PRD-012-megakernel-codegen.md)
- [PRD-013 — craftingattention lab platform alignment](PRD-013-lab-platform-alignment.md)
- [PRD-014 — `torch.func` / `vmap` / `grad` / `jacrev`](PRD-014-torch-func-transforms.md)
- [PRD-015 — Custom WGSL kernels from Python](PRD-015-custom-wgsl-kernels.md)
- [PRD-016 — ONNX export](PRD-016-onnx-export.md)

### Platform process

- [PRD-017 — Research-gated PRD workflow](PRD-017-research-gated-prd-workflow.md)

## How to use these PRDs

- Before starting implementation on a feature, **read the PRD end-to-end**.
- For new PRDs, follow the research-gated workflow in
  [`../platform/research-gated-prd-workflow.md`](../platform/research-gated-prd-workflow.md).
- If the PRD has open questions, resolve them with the user before coding.
- If implementation diverges from the PRD (a decision changes), update the PRD AND log the decision in [../../ARCHITECTURE.md](../../ARCHITECTURE.md).
- Each PRD has measurable acceptance criteria. Don't ship the feature until they pass.
- The PyTorch-conformance test suite is the floor — every new feature gets a fixture verified against real `torch`.

## Quality bar (every PRD must satisfy)

- **Evidence-backed**: every load-bearing claim cites a source.
- **Tractable**: the implementation plan is week-by-week, not "eventually."
- **Concrete**: actual code samples, actual file paths, actual line-count estimates.
- **Self-contained**: a new engineer should be able to start implementation from the PRD alone.
- **Honest**: if a number or behavior is uncertain, say so explicitly rather than guessing.
