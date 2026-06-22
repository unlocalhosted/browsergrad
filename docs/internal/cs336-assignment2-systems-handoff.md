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
- Use `@unlocalhosted/browsergrad-kernels` for:
  - `referenceFlashAttention()` to return output and log-sum-exp tensors.
  - `referenceFlashAttentionBackward()` to return Q/K/V gradients.
- Keep the CUDA/Triton tests declared as capability-gated, not failed browser
  tests.

Second slice:

- Build a single-worker distributed simulator for DDP, sharded optimizer, and
  FSDP semantics.
- Verify model-state equivalence, gradient averaging, reduce-scatter/all-gather
  behavior, tied weights, and non-trainable parameters without real processes.
- Preserve upstream test intent, but replace `torch.multiprocessing.spawn` and
  `torch.distributed` process groups with deterministic simulator calls.
- Use `@unlocalhosted/browsergrad-primitives` for:
  - `simulation.simulateDdpGradientSynchronization()` to average per-parameter gradients
    across rank-local minibatches.
  - `simulation.simulateFsdpParameterSharding()` to create deterministic parameter shard
    ownership plus all-gather expectations.
  - `simulation.simulateFsdpGradientReduceScatter()` to reduce averaged gradients back to
    owned shards or replicated parameters.
  - `simulation.simulateShardedAdamWStep()` to prove rank-owned optimizer state produces
    the same full-parameter update as non-sharded AdamW.

Future slice:

- Add Worker-mesh collectives when the platform needs realistic multi-worker
  scheduling.
- Add WebGPU FlashAttention comparisons when custom kernels are stable enough
  to be part of the learner-facing rubric.
- Build the kernel-programming pieces on the tiny core described in
  `docs/platform/kernel-lab-foundation.md`; let CUDA/Triton-style
  compatibility grow as an expansion path instead of blocking the first slice.

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

- Platform JS oracle wiring for `referenceFlashAttention()` and
  `referenceFlashAttentionBackward()` through the `_bg_attention_math`
  profile module in A2 rubrics.
- Distributed simulator API for rank-local model copies, collectives, FSDP
  sharding, and sharded optimizer state through the `_bg_distributed_training`
  profile module.
- Fixture conversion path for small `.pt` files.
- Assignment-profile capability gates in the UI, so native-only tests are shown
  as intentionally replaced/skipped.
- Clear failures for simulator mismatches: wrong gradient average, missing
  parameter broadcast, stale all-gather, bad reduce-scatter range, bad
  tied-weight handling, or incorrect sharded optimizer update.

## First Adoption Boundaries

The first adoption does not need to guarantee Triton-in-browser,
CPython-process-level `torch.distributed`, Nsight parity, or multi-GPU
performance parity.

Those are compatibility ambitions for later layers. The first adoption should
ship correctness and guided systems intuition while preserving a path for
tinkering with broader GPU/distributed semantics.
