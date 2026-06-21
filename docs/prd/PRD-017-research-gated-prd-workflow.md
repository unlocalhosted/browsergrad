# PRD-017 — Research-Gated PRD Workflow

## Problem Statement

BrowserGrad is becoming a multi-course guided-lab platform, not a single
library or one assignment port. The project now needs a repeatable PRD workflow
that can turn broad ideas into implementation-ready work without flattening
them into premature tickets. Every PRD should be grounded in codebase reality,
external research, and a deliberate search for novel browser-native approaches.

Today, PRDs can be written from local context, but the process does not enforce
research depth, adversarial questioning, subagent-style role separation, or a
clear readiness gate. That creates risk: assignments may inherit native Linux,
CUDA, PyTorch, or course-harness assumptions without asking whether BrowserGrad
should instead build a better browser-native substrate.

## Solution

Create a research-gated PRD workflow for BrowserGrad. The workflow should
synthesize existing context, run mandatory research, stress-test assumptions
through a grill-style interrogation phase, and only then publish a PRD as a
`ready-for-agent` issue.

The workflow should treat `to-prd` as the publishing/synthesis step and bake in
`grill-me` as a required design gate before readiness. The user should not have
to manually remember this order. The PRD process itself should require the
right subagent roles, artifacts, and decision checkpoints.

## User Stories

1. As the project owner, I want every PRD to start from repo exploration, so that implementation plans match the current architecture.
2. As the project owner, I want every PRD to include external research, so that plans do not rely on stale assumptions.
3. As the project owner, I want every PRD to include upstream/course research when porting assignments, so that BrowserGrad preserves learning intent.
4. As the project owner, I want every PRD to challenge native-only assumptions, so that the platform builds browser-native alternatives where appropriate.
5. As the project owner, I want a grill-style questioning phase, so that hidden decisions become explicit before implementation.
6. As the project owner, I want the grill phase to ask one decision at a time, so that dependencies between decisions are resolved cleanly.
7. As the project owner, I want each grill question to include a recommended answer, so that the process moves forward instead of becoming open-ended.
8. As the project owner, I want questions skipped when repo exploration can answer them, so that I am not asked about discoverable facts.
9. As the project owner, I want subagent roles baked into the PRD process, so that research, architecture, novelty, and testing are not all compressed into one vague pass.
10. As an implementer, I want a research dossier attached to each PRD, so that I can verify the reasoning behind the plan.
11. As an implementer, I want the PRD to name deep modules, so that implementation can produce stable, testable interfaces.
12. As an implementer, I want the PRD to distinguish reusable platform capability from assignment-specific glue, so that package boundaries stay clean.
13. As an implementer, I want the PRD to include capability gates, so that native-only dependencies are handled honestly.
14. As an implementer, I want the PRD to include novel-idea candidates, so that BrowserGrad is not limited to copying upstream tools.
15. As an implementer, I want the PRD to separate first stable guarantees from compatibility ambitions, so that small slices can ship without giving up on larger goals.
16. As a reviewer, I want the PRD to show what evidence was consulted, so that I can challenge weak assumptions.
17. As a reviewer, I want the PRD to record rejected approaches, so that the same dead ends are not rediscovered.
18. As a future agent, I want a consistent PRD template, so that I can pick up work without reconstructing the conversation.
19. As a future agent, I want readiness labels to mean the same thing every time, so that `ready-for-agent` issues are actually implementable.
20. As a course-porting agent, I want PRDs to inspect upstream tests and deps, so that browser replacements target the right learning contract.
21. As a course-porting agent, I want PRDs to decide which tests are portable, replaced, skipped, or future-gated, so that adoption is honest.
22. As a systems-lab author, I want PRDs to consider WebGPU, Worker mesh, native Dawn runners, and JS oracles, so that Pyodide is not treated as the only substrate.
23. As a platform maintainer, I want each PRD to identify test strategy before code starts, so that correctness is designed into the work.
24. As a platform maintainer, I want each PRD to avoid premature UI details, so that the core platform can mature independently.
25. As a learner, I want assignments built from these PRDs to produce clear feedback, so that failures teach concepts rather than browser internals.

## Implementation Decisions

- Build the workflow as a project convention first, then automate it once it is stable.
- Treat PRD creation as a gated pipeline with these role passes:
  - Research scout: external docs, upstream repos, package availability, browser/runtime constraints.
  - Codebase cartographer: current modules, package boundaries, nearby tests, prior PRDs.
  - Grill interviewer: decision-tree questioning, one question at a time, with recommended answers.
  - Novelty scout: browser-native alternatives, possible custom libraries, compatibility expansion paths.
  - Architecture editor: deep modules, interfaces, sequencing, and scope boundaries.
  - Test strategist: external behavior tests, fixtures, integration tests, acceptance criteria.
- Add a PRD readiness rule: no issue receives `ready-for-agent` until the research dossier, grill decisions, module sketch, and testing decisions exist.
- Use the existing package vocabulary: assignment profiles, capability gates, rubrics, oracles, runtime substrate, kernel lab foundation, Worker mesh, browser-safe gates.
- Prefer deep modules over shallow glue. Candidate deep modules for the current roadmap include assignment runner, rubric kit, fixture registry, distributed simulator, attention oracle, and kernel lab core.
- Require every assignment-port PRD to classify upstream tests into portable, replaced, skipped with capability gate, or future compatibility target.
- Require every systems/GPU PRD to consider whether Pyodide, native TS/JS, WebGPU/WGSL, Worker mesh, or a future custom compiler is the right substrate.
- Require every PRD to include a “novel idea reach” section inside Implementation Decisions, even if the final decision is conservative.
- Keep compatibility ambition explicit. PRDs may start with small stable slices, but should not frame broader CUDA/Triton/PyTorch-style experimentation as abandoned.
- Publish final PRDs to GitHub issues with the `ready-for-agent` label after the gates pass.

## Testing Decisions

- Test the workflow by running it on one concrete near-term PRD: the assignment runner plus CS336 A1 tokenizer slice.
- Good tests should verify external behavior of the produced PRD workflow: required sections exist, research artifacts are present, decisions are explicit, and readiness labels are only applied after gates pass.
- If automated later, test a PRD linter module that validates required sections and gate artifacts without depending on prose wording.
- Test issue publication with a dry-run or fake GitHub client before mutating real issues in unit tests.
- Reuse current repo patterns: pure TypeScript unit tests for validators, runtime integration tests for platform behavior, and docs/PRD review for process adoption.

## Out of Scope

- Fully automating subagents in the first implementation.
- Building a UI for PRD authoring.
- Creating a new project management system beyond GitHub issues.
- Solving all course-porting architecture in this PRD.
- Replacing the existing PRD archive.
- Enforcing this process on historical PRDs retroactively.

## Further Notes

This PRD intentionally changes process, not just documentation. The intended
behavior is: future PRDs cannot be “vibes only.” They need enough research and
adversarial questioning to be executable by another agent without re-opening
major decisions.

Research anchors already relevant to the direction:

- Pyodide package and WebAssembly constraints show why Pyodide should be one
  backend, not the whole platform reality.
- `gpu.cpp` demonstrates a small low-boilerplate WebGPU/Dawn compute API worth
  learning from.
- HipScript demonstrates that CUDA/HIP-to-WebGPU experimentation is possible,
  even if heavy and partial today.
- CS336 assignment 1 and 2 inspections show why BrowserGrad needs capability
  gates, browser-safe oracles, and native browser simulators rather than direct
  Linux/PyTorch emulation.
