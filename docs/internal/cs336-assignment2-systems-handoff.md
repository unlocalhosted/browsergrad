# CS336 Assignment 2 Systems Handoff

This adopts `stanford-cs336/assignment2-systems` as the next BrowserGrad
assignment-profile candidate. Treat it as a systems/distributed-training lab,
not as a reason to make BrowserGrad course-specific.

Upstream source: <https://github.com/stanford-cs336/assignment2-systems>

## Assignment Shape

The upstream repo contains:

- `cs336-basics`: staff assignment 1 implementation used as a dependency.
- `cs336_systems`: empty student module for systems work.
- `tests/test_attention.py`: PyTorch FlashAttention autograd tests plus
  CUDA/Triton tests.
- `tests/test_ddp.py`: distributed data parallel correctness using
  `torch.multiprocessing`, `torch.distributed`, and the `gloo` backend.
- `tests/test_sharded_optimizer.py`: optimizer-state sharding correctness using
  spawned processes.
- `tests/test_fsdp.py`: fully sharded data parallel correctness and mixed
  precision using spawned processes and collectives.
- `tests/fixtures/*.pt`: small PyTorch fixture files for DDP data/labels.

The handout describes the learning goals as benchmarking/profiling, activation
checkpointing, FlashAttention-2/Triton kernels, distributed data parallel
training, optimizer state sharding, and fully sharded data parallel training.

## BrowserGrad Adoption Strategy

Start from `docs/internal/cs336-assignment2-systems.profile.json`.

Portable first slice:

- Port `test_flash_forward_pass_pytorch` and `test_flash_backward_pytorch` to a
  browser-safe attention oracle.
- Use BrowserGrad `jit`/`kernels` attention references where possible.
- Keep the CUDA/Triton tests declared as capability-gated, not failed browser
  tests.

Second slice:

- Build a single-worker distributed simulator for DDP, sharded optimizer, and
  FSDP semantics.
- Verify model-state equivalence, gradient averaging, reduce-scatter/all-gather
  behavior, tied weights, and non-trainable parameters without real processes.
- Preserve upstream test intent, but replace `torch.multiprocessing.spawn` and
  `torch.distributed` process groups with deterministic simulator calls.

Future slice:

- Add Worker-mesh collectives when the platform needs realistic multi-worker
  scheduling.
- Add WebGPU FlashAttention comparisons when custom kernels are stable enough
  to be part of the learner-facing rubric.
- Build the kernel-programming pieces on the tiny core described in
  `docs/platform/kernel-lab-foundation.md`, not on a full CUDA/Triton clone.

## Portable As-Is

- Small `.pt` fixtures are tiny and can be mounted as assignment fixtures once
  BrowserGrad can read their format or they are converted to JSON/NumPy arrays.
- FlashAttention PyTorch-path tests express a clear numerical contract:
  output, saved log-sum-exp tensor, and backward gradients must match reference
  attention within tolerance.

## Needs Replacement

- `torch.cuda.is_available()` and CUDA device placement.
- Triton kernels and CUDA-only FlashAttention tests.
- `torch.multiprocessing.spawn`.
- `torch.distributed.init_process_group`, `barrier`, `all_gather`,
  `broadcast`, `all_reduce`, and `reduce_scatter`.
- `gloo` backend and process environment variables such as `MASTER_ADDR` and
  `MASTER_PORT`.
- Native PyTorch optimizer/process behavior around sharded state.

## Platform Gaps To Track

- Attention oracle package or runtime helper for FlashAttention-style
  forward/backward checks.
- Distributed simulator API for rank-local model copies and collectives.
- Fixture conversion path for small `.pt` files.
- Assignment-profile capability gates in the UI, so native-only tests are shown
  as intentionally replaced/skipped.
- Clear failures for simulator mismatches: wrong gradient average, missing
  parameter broadcast, stale all-gather, bad tied-weight handling, or incorrect
  sharded optimizer update.

## Non-Goals For First Adoption

- Running Triton in-browser.
- Emulating `torch.distributed` at the CPython process level.
- Matching Nsight Systems or CUDA profiler deliverables.
- Claiming multi-GPU performance parity. First adoption is correctness and
  guided systems intuition.
