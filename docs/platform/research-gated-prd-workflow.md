# Research-Gated PRD Workflow

Use this workflow for new implementation PRDs. The point is to turn broad,
ambitious ideas into implementable work without losing the ambition or lying
about constraints.

## Readiness Rule

A PRD issue can receive `ready-for-agent` only after it has:

- Repo exploration notes.
- External research notes with links.
- Upstream/course research when applicable.
- Grill decisions with one question and one recommended answer per decision.
- Novelty reach: at least one browser-native or custom-tooling idea considered.
- Deep module sketch.
- Testing decisions.
- Explicit out-of-scope boundaries.

## Role Passes

Treat these as subagent roles even when one agent performs them.

### Research Scout

Find the external reality:

- Official docs and primary sources.
- Upstream repositories, tests, dependencies, and handouts.
- Browser/runtime constraints.
- Existing libraries we can use or learn from.
- Unstable assumptions that need current verification.

Output: research dossier with links and “what this changes” notes.

### Codebase Cartographer

Ground the plan in BrowserGrad:

- Current packages and boundaries.
- Existing docs/PRDs/ADRs in the area.
- Nearby tests and integration patterns.
- Reusable seams such as assignment profiles, capability gates, JS oracles,
  Pyodide sessions, Worker boundaries, and kernel bridges.

Output: current-state map plus likely touched modules.

### Grill Interviewer

Stress-test decisions one at a time:

- Ask only questions that materially change implementation.
- Do not ask facts discoverable from the repo or research.
- Provide a recommended answer for every question.
- Resolve dependency order: substrate before API, API before tests, tests before
  rollout.

Output: decision log with accepted answers and remaining assumptions.

### Novelty Scout

Push beyond copy/paste compatibility:

- Browser-native alternatives.
- Custom TS/JS libraries.
- WebGPU/WGSL kernels.
- Worker mesh or simulator designs.
- Restricted CUDA/Triton-like educational subsets.
- Ways to preserve learning intent better than the upstream harness.

Output: novelty reach and selected/non-selected ideas.

### Architecture Editor

Convert decisions into deep modules:

- Prefer small stable interfaces with large internal leverage.
- Keep assignment-specific glue out of core packages.
- Separate first stable guarantee from compatibility ambition.
- Name capability gates for native-only requirements.

Output: implementation decisions and module boundaries.

### Test Strategist

Define confidence before code:

- Test external behavior, not implementation details.
- Use independent oracles.
- Prefer focused unit tests for deep modules.
- Add integration tests where boundaries matter.
- For assignments, classify tests as portable, replaced, skipped with
  capability gate, or future compatibility target.

Output: testing decisions and acceptance criteria.

## PRD Shape

Use the standard PRD sections:

1. Problem Statement
2. Solution
3. User Stories
4. Research Dossier
5. Grill Decisions
6. Novelty Reach
7. Implementation Decisions
8. Testing Decisions
9. Out of Scope
10. Further Notes

The `Research Dossier`, `Grill Decisions`, and `Novelty Reach` sections are
mandatory for new PRDs. They can be short for small features, but they cannot be
empty.

## Publishing

1. Add the PRD under `docs/prd/`.
2. Add it to `docs/prd/README.md`.
3. Run `pnpm validate:prd docs/prd/<your-prd>.md`.
4. Open a GitHub issue using the research-gated PRD template.
5. Apply `ready-for-agent` only when the readiness rule is satisfied.

## Default First Question

When using a grill pass, start with:

> What is the smallest vertical slice that proves this idea end-to-end without
> closing the door on the larger compatibility ambition?

Recommended default answer for BrowserGrad platform work:

> Ship a thin profile-to-runner-to-rubric path first, then add deeper native
> browser capability behind stable seams.
