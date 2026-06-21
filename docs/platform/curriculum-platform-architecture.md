# Curriculum Platform Architecture

BrowserGrad should be the execution substrate for many guided labs, ported from
many courses, lectures, papers, and class assignments. The product direction is
not one course clone. It is a browser-native lab layer that can sit beside a
YouTube lecture, course page, reading group, or workshop and give learners a
structured place to do the work.

## North Star

- Port high-value assignments from many classes into browser-runnable labs.
- Preserve the learning intent of each assignment while replacing native-only
  harness assumptions with browser-safe rubrics.
- Let a lecture link directly to a runnable lab with starter files, fixtures,
  feedback, and deterministic grading.
- Keep reusable execution capability in packages; keep course-specific policy
  in profiles and docs.

## Architecture Layers

1. **Content registry**: course, lecture, source URL, tags, assignment version,
   and fixture/dataset declarations.
2. **Assignment profile**: runtime packages, file mounts, timeouts, allowed
   tests, oracle modules, and behavior gates.
3. **Rubric adapter**: Python or JS glue that translates upstream tests into
   BrowserGrad structured assertions.
4. **Reusable oracle libraries**: small TS/Python packages for exact reference
   behavior unavailable or impractical in Pyodide.
5. **Runtime substrate**: Pyodide worker execution, filesystem mounts,
   cancellation, structured assertions, and JS oracle registration.
6. **ML libraries/backends**: `grad`, `jit`, `kernels`, and future libraries
   that provide reusable capability across many assignments.
7. **Kernel lab foundation**: a small WebGPU/WGSL core for systems assignments
   that can mature into native runners, CUDA-lite syntax, and Worker-mesh
   collectives.

## Design Rules

- Profiles are the unit of assignment portability.
- A profile may point at lectures and upstream course material, but execution
  does not depend on a particular website or video host.
- Upstream tests are evidence, not law. Keep the educational contract; replace
  native OS assumptions with browser-safe gates.
- Any helper useful across more than one assignment should become a package API.
- Any one-off course workaround should stay in the assignment profile, fixtures,
  or internal handoff docs.
- Failure output should teach: tell the learner which concept broke, not which
  browser API was used.

## Implementation Shape

- Use `parseAssignmentProfile()` as the shared validation point.
- Use optional profile `metadata` to connect labs to course/source/lecture
  context.
- Use `profileOracleJsModules()` and `createSession({ jsModules })` for
  browser-safe reference helpers, then call them from Python rubrics with
  `browsergrad.oracle(name)`.
- Use profile capability gates for upstream requirements such as CUDA, Triton,
  process groups, browser APIs, or Worker meshes that may be skipped or replaced
  in the browser edition.
- Use worker `timeoutMs`, streaming gates, fixture checks, and deterministic
  oracles for grading.
- Build GPU/systems labs on the tiny kernel core described in
  `docs/platform/kernel-lab-foundation.md`.
- Keep future course families independent: CS336, fast.ai, MIT, CMU, Berkeley,
  paper replications, and local workshops should all use the same platform
  primitives.

## What Not To Do

- Do not make root package code depend on a specific course.
- Do not encode lecture-specific UI assumptions in runtime packages.
- Do not emulate Linux process/resource behavior when a browser behavior gate
  can test the learning contract directly.
- Do not fork a new platform path per course unless the difference is truly
  architectural.
