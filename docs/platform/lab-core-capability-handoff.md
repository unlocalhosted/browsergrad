# Lab Core Capability Handoff

This handoff is for platform consumers such as `craftingattention`. BrowserGrad
owns the runtime/library primitives; the platform owns UI, authoring workflows,
learner navigation, and issue-level lab rollout.

## Source Of Truth

- Meta PRD: `docs/prd/PRD-018-lab-core-capability-spine.md`
- Runtime API: `@unlocalhosted/browsergrad-runtime`
- Assignment profile parser: `packages/browsergrad-runtime/src/assignment.ts`
- Authoring guide: `docs/platform/assignment-authoring.md`
- Architecture guide: `docs/platform/curriculum-platform-architecture.md`
- Kernel guide: `docs/platform/kernel-lab-foundation.md`

## Platform Contract

For every lab profile, the platform should:

1. Parse the assignment profile with `parseAssignmentProfile`.
2. Convert oracle specs with `profileOracleJsModules`.
3. Build a substrate-neutral run plan with `createAssignmentRunPlan`.
4. Evaluate capability readiness before launching the lab.
5. Show missing capability gates as preflight status, not as runtime crashes.
6. Route runnable labs to the right substrate: Pyodide, TS/JS oracle, WebGPU,
   Worker mesh, external/native runner, or future custom compiler.
7. Log one `unlocalhosted/craftingattention` issue for each platform handoff or
   implementation slice.

## Capability Vocabulary

Capability names are strings. Keep them descriptive and reusable:

| Capability | Meaning |
| --- | --- |
| `pyodide` | Python execution in browser Worker. |
| `torch-compat` | BrowserGrad PyTorch-shaped teaching surface is sufficient. |
| `webgpu` | Browser WebGPU adapter is available. |
| `wgsl-kernel` | Lab can run WGSL kernels directly. |
| `cuda-compatible-subset` | Lab targets a BrowserGrad CUDA-like educational subset. |
| `worker-mesh` | Multiple Workers can simulate distributed participants. |
| `distributed-simulator` | Deterministic simulator for DDP/FSDP/task-system behavior. |
| `dataset-fixture` | Small checked-in fixture replaces a large external dataset. |
| `large-file-streaming` | Lab can stream large files instead of loading whole corpora. |
| `snapshot-oracle` | Expected outputs are stored as JSON/NPZ/safetensors snapshots. |
| `tokenizer-oracle` | JS/TS tokenizer oracle is available to rubrics. |
| `rl-loss-oracle` | Alignment/RL losses have independent reference checks. |
| `hosted-api-mock` | Hosted API behavior is reproduced by a deterministic local mock. |
| `native-cpp-external` | Lab requires external native C++ build/run support. |
| `ispc-external` | Lab requires external ISPC support or a simulator. |
| `openmp-external` | Lab requires external OpenMP support or a simulator. |
| `vllm-external` | Lab requires an external vLLM service. |
| `flash-attn-external` | Lab requires external flash-attn/CUDA support. |

This table is a living convention, not a runtime enum. Add names when a new
assignment family needs them, but prefer reusing names across courses.

## Profile Gate Shape

Use a capability gate when availability can be checked before execution:

```json
{
  "name": "flash_attention_kernel_path",
  "kind": "capability",
  "options": {
    "requires": ["torch-compat"],
    "any_of": [["webgpu", "wgsl-kernel"], ["triton-compatible"], ["native-cuda-external"]],
    "message": "FlashAttention labs need a browser kernel path or an external native path."
  }
}
```

- `requires` means every listed capability must be present.
- `any_of` means at least one group must be fully present.
- `message` is platform-facing explanatory text.
- Non-capability gates such as streaming and timeout should remain separate
  because they are checked by rubrics or watchdogs during execution.

## Benchmark Pressure Matrix

Machine-readable profile drafts live in `docs/internal/*.profile.json`. They are
parsed by `packages/browsergrad-runtime/tests/benchmark-profiles.test.ts` so the
handoff matrix cannot drift silently from runtime profile validation.

| Benchmark | First platform slice | Core capabilities |
| --- | --- | --- |
| CS336 A2 Systems | FlashAttention fixture + DDP/FSDP simulator preflight | `torch-compat`, `webgpu`, `worker-mesh`, `distributed-simulator` |
| CS336 A3 Scaling | Hosted API mock + scheduler tests | `http-client`, `hosted-api-mock`, `server-fixture` |
| CS336 A4 Data | Small Common Crawl fixtures + data-quality rubrics | `dataset-fixture`, `large-file-streaming`, `classifier-oracle`, `pii-oracle` |
| CS336 A5 Alignment | GRPO/DPO math snapshot labs | `torch-compat`, `transformers-compatible`, `snapshot-oracle`, `rl-loss-oracle` |
| GPU Puzzles | WGSL puzzle runner | `webgpu`, `wgsl-kernel`, `kernel-visualizer` |
| CS149 A1/A2 | Thread/SIMD/task-system simulator | `pthreads-simulator`, `simd-simulator`, `distributed-simulator` |
| CS149 A3 | CUDA scan/SAXPY/render concepts | `webgpu`, `cuda-compatible-subset`, `performance-rubric` |
| CS149GPT | CPU attention optimization oracle | `native-cpp-external`, `attention-oracle`, `simd-simulator` |

## Platform Issue Convention

For each handoff or implementation slice, create a craftingattention issue with:

- BrowserGrad source doc or PRD link.
- Lab benchmark family.
- Required capabilities.
- Platform UI states: runnable, simulated, external-only, blocked.
- Fixture/mount expectations.
- Rubric/oracle expectations.
- Acceptance checks the platform can run.

Use the issue title pattern:

```text
BrowserGrad handoff: <lab or capability slice>
```

## Next Platform Slice

After PRD-018 lands, craftingattention should add a preflight panel that:

1. Reads an assignment profile.
2. Builds a BrowserGrad run plan.
3. Calls BrowserGrad capability evaluation from the run plan.
4. Shows packages, oracle modules, file mounts, and satisfied/missing capability
   groups.
5. Offers the learner a runnable browser path, simulated path, or external-runner
   note depending on the profile result.
