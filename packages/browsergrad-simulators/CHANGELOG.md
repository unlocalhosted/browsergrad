# Changelog

All notable changes to `@unlocalhosted/browsergrad-simulators`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial deterministic mesh simulator for browser-safe systems and distributed
  learning rubrics.
- `createTaskGraphSimulator()` records deterministic ready/start/finish traces
  for dependency-constrained task-system labs.
- Deterministic DDP gradient averaging, FSDP parameter sharding/all-gather,
  FSDP reduce-scatter, and sharded AdamW optimizer state/update simulators for
  CS336 A2-style distributed-training rubrics.
- CS149 A1-style CPU/SIMD teaching helpers:
  `simulateCs149ClampedExpVector()`, `simulateCs149ArraySumVector()`, and
  `partitionStaticWork()` for lane-mask utilization, vector reductions, and
  static thread decomposition rubrics.
