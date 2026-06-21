# CS336 Assignment 5 Alignment Handoff

This note captures the browser-safe path for
`stanford-cs336/assignment5-alignment`. Keep it as an assignment profile record,
not root platform identity.

## Upstream Shape

- CS336 Spring 2026 describes Assignment 5 as alignment and reasoning RL:
  supervised finetuning, expert iteration, GRPO, and optional safety/RLHF/DPO.
- The upstream repository uses `uv`, PyTorch, Transformers, Qwen-style model
  workflows, and optional vLLM/flash-attn/native inference paths.
- The upstream tests expose pure math/text hooks for DPO, MMLU/GSM8K parsing,
  rollout rewards, group-normalized rewards, policy-gradient loss, and masked
  aggregation. Those are browser-safe fixture targets.

## Browser-Safe First Slice

- Use `docs/internal/cs336-assignment5-alignment.profile.json` as source
  profile.
- Use `@unlocalhosted/browsergrad-alignment` for:
  - `computePerInstanceDpoLoss()` for optional safety/RLHF DPO fixtures.
  - `parseMmluResponse()` and `parseGsm8kResponse()` for metrics fixtures.
  - `computeRolloutRewards()` for reward-function wiring checks.
  - `computeGroupNormalizedRewards()` for GRPO/Dr.GRPO/MaxRL advantage checks.
  - `computePolicyGradientLoss()` for on-policy and off-policy token losses.
  - `aggregateLossAcrossMicrobatch()` for masked sequence/constant loss
    normalization.
- Use `@unlocalhosted/browsergrad-snapshots` for expected numeric fixture
  outputs.
- Use `@unlocalhosted/browsergrad-tokenizers` for prompt/output tokenization
  fixtures where pure token IDs are enough.

## Non-Portable Upstream Assumptions

- Full Qwen model loading and training inside the browser.
- vLLM serving, flash-attn kernels, and CUDA-native inference.
- Hugging Face datasets/model downloads at lab-run time.
- Long-running GRPO training loops as the first browser fixture.

## Platform Work

- Register a JS oracle module from `@unlocalhosted/browsergrad-alignment` as
  `_bg_alignment_oracles`.
- Route `rl-loss-oracle` to the alignment package.
- Route `response-parser-oracle` to `parseMmluResponse()` and
  `parseGsm8kResponse()`.
- Keep full inference/training behind `vllm-external`, `flash-attn-external`,
  or future browser-native model-runtime gates.
- Render failures as alignment-specific rubric messages, for example:
  - `DPO loss mismatch`
  - `group-normalized advantage mismatch`
  - `policy gradient clipping mismatch`
  - `GSM8K response parser selected the wrong number`
  - `MMLU parser emitted an invalid answer option`

## Later Slices

- Add prompt/output tokenizer alignment fixtures.
- Add response log-prob snapshot helpers once a tiny causal-LM fixture surface is
  stable.
- Add external runner handoff for vLLM/flash-attn/Qwen execution.
