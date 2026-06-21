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
4. Classify the rubric with `assignmentRubricKind`.
5. Provide `capabilityModes` for detected capabilities when the platform knows
   a capability is `browser`, `simulated`, or `external`.
6. Call `assignmentRunReadiness(plan)` before launching the lab, or call
   `createAssignmentPreflightReport(profile, environment)` when the platform
   wants the run plan, readiness, rubric kind, required capabilities, and mount
   plan together.
7. Show `runnable`, `simulated`, `external-only`, or `blocked` as preflight
   status, not as runtime crashes.
8. Build a file/dataset mount plan with `createAssignmentMountPlan`.
9. Dry-run platform-provided contents with `evaluateAssignmentMountContents`.
10. Materialize provided file and dataset contents with
   `materializeAssignmentMountPlan`, use `runAssignmentRubric` for the common
   Pyodide mount-and-execute path, or use `runAssignmentJavascriptRubric` for
   browser-native JS rubrics.
   Contents may be strings or `Uint8Array` bytes for compact binary fixtures
   such as `.pt`, `.npz`, or `.safetensors`.
   Use `createAssignmentMountPreflightReport` when rendering one platform
   preflight result for missing content plus hash verification.
   Verify dataset `sha256:<64 hex>` declarations with
   `verifyAssignmentMountContentHashes` before writing to runtime FS.
   Use `Session.fs.readBytes(path)` when the platform needs to verify mounted
   worker bytes against cache, hash, or snapshot metadata.
11. Route runnable labs to the right substrate: Pyodide, TS/JS oracle, WebGPU,
   Worker mesh, external/native runner, or future custom compiler.
12. For Pyodide-backed labs, create the rubric execution request with
   `createAssignmentRubricExecRequest`.
   The request uses the shorter runtime watchdog from `test_ms` and `worker_ms`;
   keep `setup_ms` for package preload/cache UI.
13. For JavaScript-backed labs, pass the imported rubric function, declared
    oracle objects, and browser substrates such as WebGPU devices to
    `runAssignmentJavascriptRubric`.
14. In Python rubrics, call profile-registered JS oracles with
    `browsergrad.oracle("<module-name>")`.
15. In Python rubrics, read root, fixture, allowed-test, and behavioral-gate
    context with `browsergrad.assignment_context()`.
16. In Python rubrics, enforce streaming gates with
    `browsergrad.streaming_gate(name, iterable)` plus
    `gate.wrap_output(student_output)` so eager consumers fail before launchers
    need Linux RSS behavior.
17. In Python rubrics, enforce forbidden-read gates with
    `browsergrad.forbidden_read_gate(name, text)` so eager `read()` or
    `readlines()` calls fail while incremental line reads still work.
18. Log one `unlocalhosted/craftingattention` issue for each platform handoff or
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

## Readiness Modes

Profiles declare capability names; platforms declare what those capabilities
mean in the current environment:

- `browser`: direct in-browser execution such as Pyodide, WebGPU, or JS oracle
  execution.
- `simulated`: deterministic Worker/oracle/fixture substitutes that preserve the
  learning objective without native infrastructure.
- `external`: native or hosted runner paths such as CUDA, ISPC, vLLM, Modal, or
  external servers.

Pass these labels through `capabilityModes` when calling
`createAssignmentRunPlan`, then use `assignmentRunReadiness(plan)`.
`external` wins over `simulated`, and failed capability preflight becomes
`blocked`.
When several `any_of` groups are available, BrowserGrad selects the strongest
group by mode: direct `browser` path first, then `simulated`, then `external`.
This prevents a teaching simulator from hiding a real browser-native path such
as WGSL.
Render each gate from `plan.capabilityEvaluation.gates`: `status` is the
gate-level route state, `selectedAnyOf` is the chosen alternative group, and
`selectedCapabilities` is the complete selected path including required caps.

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
That test also builds `createAssignmentPreflightReport` for every benchmark
profile under a browser-teaching environment and checks expected readiness
states. It dry-runs empty mount contents for every profile with
`evaluateAssignmentMountContents` so missing rubric files and datasets stay
visible before filesystem writes. Runtime integration tests also mount binary
fixture bytes through the Pyodide path.

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
3. Classifies rubric kind with `assignmentRubricKind`.
4. Calls BrowserGrad capability evaluation from the run plan.
5. Calls `assignmentRunReadiness(plan)` and renders its status, selected
   capabilities, and missing capabilities.
6. Or uses `createAssignmentPreflightReport(profile, environment)` to get all
   preflight fields in one object.
7. Renders `plan.capabilityEvaluation.gates` as preflight rows using each gate's
   `status`, `selectedAnyOf`, `selectedCapabilities`, and missing fields.
8. Builds the BrowserGrad mount plan for runnable or inspectable labs.
9. Fetches or provides assignment file/dataset contents, then calls
   `evaluateAssignmentMountContents` to show missing files/datasets.
10. Materializes validated contents
   into `Session.fs`.
11. Shows packages, oracle modules, rubric kind, file mounts, and
   satisfied/missing capability groups.
12. For runnable Pyodide labs, uses `runAssignmentRubric` to mount contents and
   launch the rubric through `Session.exec`, or uses
   `createAssignmentRubricExecRequest` when the platform needs manual staging.
   Binary fixtures can be verified after staging with `Session.fs.readBytes`.
   Dataset hashes should be verified before staging with
   `verifyAssignmentMountContentHashes`.
13. For runnable JavaScript labs, imports the rubric module and calls
   `runAssignmentJavascriptRubric`; JS rubrics read binary fixtures with
   `ctx.readBytes(path)`.
14. Offers the learner a runnable browser path, simulated path, or external-runner
   note depending on the profile result.
