# Compiler Bugbash Progress

Last updated: 2026-07-02T20:10:15Z

Purpose: make compiler bugbash visible. Update this file whenever a new bug, fixture, gate, or remaining risk changes.

## Dashboard

| Field | Current |
| --- | --- |
| Overall status | Active bugbash, not complete |
| Fixed failure movement | Started from 87 failing real-world/audit cases; current verifier gate is green at src `453/0/0`, dist `453/0/0`; cuda-samples compile/codegen audit now has `0` hard fails; real corpus WebGPU fixture outputs are pinned `98/98` |
| Current focus | Pointer/vector storage correctness, texture/vector conversion, active-lane/control semantics, and hot-loop test speed |
| Active work item | Validate surface boundary semantics plus keep perf/source-dist verifier gates green |
| Skip policy | No added skips. WebGPU commands must use `--forbid-skips` |
| Worktree | Compiler-owned files should be clean after each batch; unrelated JIT dirty files may remain outside compiler bugbash |
| Next proof command | `pnpm --filter @unlocalhosted/browsergrad-compiler run verify:changed:plan` |

## How To Track This

Use this file as the source of truth for the current bugbash.

- Quick status, latest proof, active failures, and remaining probes: `pnpm --filter @unlocalhosted/browsergrad-compiler run bugbash:status`
- Replay current blocker: `pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:last-failures`
- See scoped test plan: `pnpm --filter @unlocalhosted/browsergrad-compiler run verify:changed:plan`
- Read `Dashboard` first for the state in one screen.
- Read `Latest Proven Green Gates` before trusting any "fixed" claim.
- Read `Bugs Found During Current Run` to see root fixes, not shallow patches.
- Read `Remaining Probe Map` to see what is still not proven.
- Read `Gate Ladder` to see why I am running a focused command instead of a repo-wide command.

Every work chunk should update:

- `Last updated`
- `Dashboard.Active work item`
- `Latest Proven Green Gates`
- `Bugs Found During Current Run`
- `Remaining Probe Map`

Done means all of these are true:

- No current failing focused cases.
- No new skips in touched compiler tests or WebGPU fixtures.
- Full compiler unit suite passes, not only filtered `-t` runs.
- WebGPU smoke passes with `0 skipped`.
- Source and dist real-world verifier pass with `0 skipped`.
- Remaining unsupported cases have explicit diagnostics, not emit-time crashes or silent wrong output.

## Latest Proven Green Gates

Current verified gates:

- changed gate after corpus fixture input materializer hardening: fixture tests passed; WebGPU smoke `290/0/0`; skips `0`
- unreachable helper atomic lowering fix: fail-first unit reproduced `atomicDeviceGlobals=["gCounter"]` from an unused helper; focused atomic unit slice `441/0`; `verify:changed` typecheck + compiler unit `425/0`; lint passed
- `surface:helper-vector-read-multiple-surfaces`: `1 passed / 0 failed / 0 skipped`
- focused surf2Dread unit run: `386 passed`
- `surface:layered-write`: `1 passed / 0 failed / 0 skipped`
- focused surf2DLayeredwrite unit run: `387 passed`
- compiler typecheck: passed
- compiler lint: passed
- WebGPU smoke: `100 passed / 0 failed / 0 skipped`
- full source WebGPU e2e: `221 passed / 0 failed / 0 skipped`
- real-world CUDA verifier: src `253/0/0`, dist `253/0/0`
- real-world CUDA verifier after loop-local continue/class-member fixes: src `419/0/0`, dist `419/0/0`
- new surface/texture probe group: `4 passed / 0 failed / 0 skipped`
- WebGPU smoke after new probes: `104 passed / 0 failed / 0 skipped`
- pointer/control probe group: `5 passed / 0 failed / 0 skipped`
- WebGPU smoke after pointer/control probes: `109 passed / 0 failed / 0 skipped`
- cross-space vector alias probe: `1 passed / 0 failed / 0 skipped`
- compiler unit suite after texture vector conversion fix: `374 passed / 0 failed`
- WGSL module unit suite: `16 passed / 0 failed`
- WebGPU smoke after shared-vector fix: `110 passed / 0 failed / 0 skipped`
- test-scope tooling: passed
- bugbash status progress summary: passed
- mixed local/storage helper pointer diagnostic unit: `1 passed`
- pointer-array + shared-vector targeted WebGPU pair: `2 passed / 0 failed / 0 skipped`
- hot cross-space vector alias loop: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `4.2ms`, speedup `1.26`
- fast auto-corpus WebGPU gate: `32 passed / 0 failed / 0 skipped`
- texture helper vector cast/coercion: `1 passed / 0 failed / 0 skipped`
- WebGPU smoke after texture coercion fixture: `111 passed / 0 failed / 0 skipped`
- loop-internal return/barrier replay: `control:active-lane-loop-internal-return-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after loop-internal return/barrier regression: `375 passed / 0 failed`
- WebGPU smoke after loop-internal return/barrier fixture: `112 passed / 0 failed / 0 skipped`
- alternate-branch return/barrier fixture: `control:active-lane-alternate-return-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after alternate-return active-lane regression: `376 passed / 0 failed`
- WGSL module suite after alternate-return split: `16 passed / 0 failed`
- WebGPU smoke after alternate-return fixture: `113 passed / 0 failed / 0 skipped`
- nested return/barrier fixture: `control:active-lane-nested-return-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after nested active-lane regression: `377 passed / 0 failed`
- WebGPU smoke after nested return/barrier fixture: `114 passed / 0 failed / 0 skipped`
- loop alternate-return/barrier fixture: `control:active-lane-loop-alternate-return-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after loop alternate-return probe: `378 passed / 0 failed`
- WebGPU smoke after loop alternate-return fixture: `115 passed / 0 failed / 0 skipped`
- loop return-side-effect/barrier fixture: `control:active-lane-loop-return-side-effect-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after loop return-side-effect probe: `379 passed / 0 failed`
- WebGPU smoke after loop return-side-effect fixture: `116 passed / 0 failed / 0 skipped`
- hot loop return-side-effect fixture: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.6ms`, speedup `1.14`
- vector lane return-side-effect fixture: `control:active-lane-vector-return-side-effect-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after vector lane return-side-effect probe: `380 passed / 0 failed`
- WebGPU smoke after vector lane return-side-effect fixture: `117 passed / 0 failed / 0 skipped`
- hot vector lane return-side-effect fixture: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.9ms`, speedup `1.10`
- pointer alias return-side-effect fixture: `control:active-lane-pointer-alias-return-side-effect-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after pointer alias return-side-effect probe: `381 passed / 0 failed`
- WebGPU smoke after pointer alias return-side-effect fixture: `118 passed / 0 failed / 0 skipped`
- hot pointer alias return-side-effect fixture: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.7ms`, speedup `1.19`
- atomic return-side-effect fixture: `control:active-lane-atomic-return-side-effect-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after atomic return-side-effect probe: `382 passed / 0 failed`
- WebGPU smoke after atomic return-side-effect fixture: `119 passed / 0 failed / 0 skipped`
- hot atomic return-side-effect fixture: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `1.7ms`, speedup `1.06`
- shared return-side-effect fixture: `control:active-lane-shared-return-side-effect-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after shared return-side-effect probe: `383 passed / 0 failed`
- WebGPU smoke after shared return-side-effect fixture: `120 passed / 0 failed / 0 skipped`
- hot shared return-side-effect fixture: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.4ms`, speedup `1.21`
- surface return-side-effect fixture: `surface:active-lane-return-side-effect` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after surface return-side-effect probe: `384 passed / 0 failed`
- WebGPU smoke after surface return-side-effect fixture: `121 passed / 0 failed / 0 skipped`
- hot surface return-side-effect fixture: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `1.2ms`, speedup `3.17`
- compiler typecheck after active-lane return/barrier fixes: passed
- texture active-lane return-read side-effect fixture: `texture:active-lane-return-read-side-effect` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after texture active-lane read probe: `385 passed / 0 failed`
- WebGPU smoke after texture active-lane read probe: `122 passed / 0 failed / 0 skipped`
- hot texture active-lane read probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.7ms`, speedup `1.27`
- texture/surface active-lane return side-effect fixture: `texture-surface:active-lane-return-side-effect` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after texture/surface active-lane probe: `386 passed / 0 failed`
- WebGPU smoke after texture/surface active-lane probe: `123 passed / 0 failed / 0 skipped`
- hot texture/surface active-lane probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `4.0ms`, speedup `0.50`
- texture atlas/layered active-lane return-read fixture: `texture:atlas-active-lane-return-read-side-effect` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after atlas/layered active-lane probe: `387 passed / 0 failed`
- WebGPU smoke after atlas/layered active-lane probe: `124 passed / 0 failed / 0 skipped`
- hot atlas/layered active-lane probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.3ms`, speedup `1.36`
- deep helper texture active-lane vector-store fixture: `texture:deep-helper-active-lane-vector-store` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after deep helper vector-store probe: `388 passed / 0 failed`
- WebGPU smoke after deep helper vector-store probe: `125 passed / 0 failed / 0 skipped`
- hot deep helper vector-store probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `4.0ms`, speedup `1.05`
- mixed scalar/vector texture active-lane store fixture: `texture:mixed-scalar-vector-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after mixed scalar/vector texture probe: `389 passed / 0 failed`
- WebGPU smoke after mixed scalar/vector texture probe: `126 passed / 0 failed / 0 skipped`
- hot mixed scalar/vector texture probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `4.0ms`, speedup `1.13`
- texture pointer-alias active-lane store fixture: `texture:pointer-alias-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after texture pointer-alias probe: `390 passed / 0 failed`
- WebGPU smoke after texture pointer-alias probe: `127 passed / 0 failed / 0 skipped`
- hot texture pointer-alias probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `4.6ms`, speedup `1.11`
- texture pointer-alias atomic active-lane fixture: `texture:pointer-alias-atomic-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after texture pointer-alias atomic fix: `391 passed / 0 failed`
- WebGPU smoke after texture pointer-alias atomic fix: `128 passed / 0 failed / 0 skipped`
- hot texture pointer-alias atomic probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `1.8ms`, speedup `3.00`
- texture pointer-alias atomic vector readback fixture: `texture:pointer-alias-atomic-vector-readback` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after atomic vector readback fix: `392 passed / 0 failed`
- WebGPU smoke after atomic vector readback fix: `129 passed / 0 failed / 0 skipped`
- hot atomic vector readback probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `1.5ms`, speedup `3.80`
- texture pointer-alias atomic vector compound fixture: `texture:pointer-alias-atomic-vector-compound` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after atomic vector helper index fix: `393 passed / 0 failed`
- WebGPU smoke after atomic vector helper index fix: `130 passed / 0 failed / 0 skipped`
- hot atomic vector compound probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `5.2ms`, speedup `1.10`
- surface layered read fixture: `surface:layered-read` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after surface layered read fix: `394 passed / 0 failed`
- WebGPU smoke after surface layered read fix: `131 passed / 0 failed / 0 skipped`
- hot surface layered read probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `1.8ms`, speedup `2.56`
- surface 3D read fixture: `surface:surf3d-read` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after surface 3D read fix: `395 passed / 0 failed`
- WebGPU smoke after surface 3D read fix: `132 passed / 0 failed / 0 skipped`
- hot surface 3D read probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.7ms`, speedup `1.19`
- surface layered/3D vector read fixture pair: `2 passed / 0 failed / 0 skipped`
- compiler unit suite after surface vector read type fix: `396 passed / 0 failed`
- WebGPU smoke after surface vector read type fix: `134 passed / 0 failed / 0 skipped`
- hot surface layered/3D vector read probe: `6 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.6ms` / `3.7ms`, speedups `1.19` / `1.08`
- surface vector read before active-lane return fixture: `surface:vector-read-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- WebGPU smoke after active-lane surface vector read probe: `135 passed / 0 failed / 0 skipped`
- hot active-lane surface vector read probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `2.8ms`, speedup `1.79`
- surface vector write before active-lane return fixture: `surface:vector-write-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- WebGPU smoke after active-lane surface vector write probe: `136 passed / 0 failed / 0 skipped`
- hot active-lane surface vector write probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `4.0ms`, speedup `1.43`
- texture-fed layered surface vector write before active-lane return fixture: `texture-surface:vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- WebGPU smoke after texture-fed surface vector active-lane probe: `137 passed / 0 failed / 0 skipped`
- hot texture-fed surface vector active-lane probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `3.7ms`, speedup `1.32`
- layered/3D texture vector read into 3D surface vector write before active-lane return fixture: `texture-surface:volume-vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after volume texture/surface probe: passed
- compiler typecheck after volume texture/surface probe: passed
- WebGPU smoke after volume texture/surface active-lane probe: `138 passed / 0 failed / 0 skipped`
- hot volume texture/surface active-lane probe: `3 passed / 0 failed / 0 skipped`, warmup `1`, best warm `4.3ms`, speedup `1.19`
- float3 layered surface vector write/read before active-lane return fixture: `surface:float3-vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after float3 surface active-lane probe: passed
- compiler typecheck after float3 surface active-lane probe: passed
- WebGPU smoke after float3 surface active-lane probe: `139 passed / 0 failed / 0 skipped`
- hot float3 surface active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.4ms`, speedup `1.44`; earlier repeat `3` run was correctness-green but speedup-noisy at `0.88`
- float3 texture vector read/store before active-lane return fixture: `texture:float3-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after float3 texture active-lane probe: passed
- compiler typecheck after float3 texture active-lane probe: passed
- WebGPU smoke after float3 texture active-lane probe: `140 passed / 0 failed / 0 skipped`
- hot float3 texture active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.2ms`, speedup `1.22`
- uint3 texture vector read/store before active-lane return fixture: `texture:uint3-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after uint3 texture active-lane probe: passed
- compiler typecheck after uint3 texture active-lane probe: passed
- WebGPU smoke after uint3 texture active-lane probe: `141 passed / 0 failed / 0 skipped`
- hot uint3 texture active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.6ms`, speedup `1.31`
- int3 texture vector read/store before active-lane return fixture: `texture:int3-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after int3 texture active-lane probe: passed
- compiler typecheck after int3 texture active-lane probe: passed
- WebGPU smoke after int3 texture active-lane probe: `142 passed / 0 failed / 0 skipped`
- hot int3 texture active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.5ms`, speedup `1.34`
- uint3 layered surface vector write/read before active-lane return fixture: `surface:uint3-vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after uint3 surface active-lane probe: passed
- compiler typecheck after uint3 surface active-lane probe: passed
- WebGPU smoke after uint3 surface active-lane probe: `143 passed / 0 failed / 0 skipped`
- hot uint3 surface active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `1.7ms`, speedup `2.88`
- int3 layered surface vector write/read before active-lane return fixture: `surface:int3-vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after int3 surface active-lane probe: passed
- compiler typecheck after int3 surface active-lane probe: passed
- WebGPU smoke after int3 surface active-lane probe: `144 passed / 0 failed / 0 skipped`
- hot int3 surface active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.3ms`, speedup `1.36`
- uint4/int4 layered surface vector write/read before active-lane return fixture pair: `2 passed / 0 failed / 0 skipped`
- compiler fixture test after uint4/int4 surface active-lane probes: passed
- compiler typecheck after uint4/int4 surface active-lane probes: passed
- WebGPU smoke after uint4/int4 surface active-lane probes: `146 passed / 0 failed / 0 skipped`
- hot uint4 surface active-lane probe: repeat `7`, warmup `1`, `7 passed / 0 failed / 0 skipped`, best warm `3.5ms`, speedup `1.49`; earlier paired hot run was correctness-green but speedup-noisy at `0.46`
- hot int4 surface active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `1.5ms`, speedup `1.47`
- float2/uint2/int2 layered surface vector write/read before active-lane return fixture group: `3 passed / 0 failed / 0 skipped`
- compiler fixture test after 2-lane surface active-lane probes: passed
- compiler typecheck after 2-lane surface active-lane probes: passed
- WebGPU smoke after 2-lane surface active-lane probes: `149 passed / 0 failed / 0 skipped`
- hot 2-lane surface active-lane probe group: repeat `5`, warmup `1`, `15 passed / 0 failed / 0 skipped`, best warm `1.4ms` / `1.3ms` / `1.2ms`, speedups `1.14` / `2.92` / `1.25`
- float2/uint2/int2 texture vector read/store before active-lane return fixture group: `3 passed / 0 failed / 0 skipped`
- compiler fixture test after 2-lane texture active-lane probes: passed
- compiler typecheck after 2-lane texture active-lane probes: passed
- WebGPU smoke after 2-lane texture active-lane probes: `152 passed / 0 failed / 0 skipped`
- hot 2-lane texture active-lane probe group: repeat `5`, warmup `1`, `15 passed / 0 failed / 0 skipped`; `int2` isolated repeat `7` after paired perf noise was `7 passed / 0 failed / 0 skipped`, best warm `3.3ms`, speedup `1.33`
- float4/uint4/int4 texture vector read/store before active-lane return fixture group: `3 passed / 0 failed / 0 skipped`
- compiler fixture test after 4-lane texture active-lane probes: passed
- compiler typecheck after 4-lane texture active-lane probes: passed
- WebGPU smoke after 4-lane texture active-lane probes: `155 passed / 0 failed / 0 skipped`
- hot 4-lane texture active-lane probe group: repeat `5`, warmup `1`, `15 passed / 0 failed / 0 skipped`, best warm `0.7ms` / `2.9ms` / `2.8ms`, speedups `5.29` / `1.31` / `1.14`
- float2/uint2/int2 texture-to-layered-surface vector active-lane fixture group: `3 passed / 0 failed / 0 skipped`
- compiler fixture test after 2-lane texture-to-surface active-lane probes: passed
- compiler typecheck after 2-lane texture-to-surface active-lane probes: passed
- WebGPU smoke after 2-lane texture-to-surface active-lane probes: `158 passed / 0 failed / 0 skipped`
- hot 2-lane texture-to-surface active-lane probe group: repeat `5`, warmup `1`, `15 passed / 0 failed / 0 skipped`, best warm `3.1ms` / `2.9ms` / `3.1ms`, speedups `1.29` / `1.38` / `1.13`
- float3/uint3/int3 texture-to-layered-surface vector active-lane fixture group: `3 passed / 0 failed / 0 skipped`
- compiler fixture test after 3-lane texture-to-surface active-lane probes: passed
- compiler typecheck after 3-lane texture-to-surface active-lane probes: passed
- WebGPU smoke after 3-lane texture-to-surface active-lane probes: `161 passed / 0 failed / 0 skipped`
- hot 3-lane texture-to-surface active-lane probe group: repeat `5`, warmup `1`, `15 passed / 0 failed / 0 skipped`, best warm `1.0ms` / `1.4ms` / `1.0ms`, speedups `3.90` / `2.71` / `3.40`
- float4/uint4/int4 texture-to-layered-surface vector active-lane fixture group: `3 passed / 0 failed / 0 skipped`
- compiler fixture test after 4-lane texture-to-surface active-lane probes: passed
- compiler typecheck after 4-lane texture-to-surface active-lane probes: passed
- WebGPU smoke after 4-lane texture-to-surface active-lane probes: `164 passed / 0 failed / 0 skipped`
- hot 4-lane texture-to-surface active-lane probe group: repeat `5`, warmup `1`, `15 passed / 0 failed / 0 skipped`, best warm `3.5ms` / `1.7ms` / `1.2ms`, speedups `1.14` / `2.59` / `3.25`
- mixed scalar/vector texture-to-layered-surface active-lane fixture: `texture-surface:mixed-vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after mixed texture-to-surface active-lane probe: passed
- compiler typecheck after mixed texture-to-surface active-lane probe: passed
- WebGPU smoke after mixed texture-to-surface active-lane probe: `165 passed / 0 failed / 0 skipped`
- hot mixed texture-to-surface active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.7ms`, speedup `1.43`
- float4 layered surface vector write/read before active-lane return fixture: `surface:layered-float4-vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after layered float4 surface active-lane probe: passed
- compiler typecheck after layered float4 surface active-lane probe: passed
- WebGPU smoke after layered float4 surface active-lane probe: `166 passed / 0 failed / 0 skipped`
- hot layered float4 surface active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `1.6ms`, speedup `3.13`
- mixed scalar/vector layered surface active-lane fixture: `surface:layered-mixed-scalar-vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after mixed layered surface active-lane probe: passed
- compiler typecheck after mixed layered surface active-lane probe: passed
- WebGPU smoke after mixed layered surface active-lane probe: `167 passed / 0 failed / 0 skipped`
- hot mixed layered surface active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.9ms`, speedup `1.36`
- surface-read pointer-alias active-lane fixture: `surface:pointer-alias-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surface pointer-alias active-lane probe: passed
- compiler typecheck after surface pointer-alias active-lane probe: passed
- WebGPU smoke after surface pointer-alias active-lane probe: `168 passed / 0 failed / 0 skipped`
- hot surface pointer-alias active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.5ms`, speedup `1.31`
- surface-read atomic pointer-alias active-lane fixture: `surface:pointer-alias-atomic-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surface atomic pointer-alias active-lane probe: passed
- compiler typecheck after surface atomic pointer-alias active-lane probe: passed
- WebGPU smoke after surface atomic pointer-alias active-lane probe: `169 passed / 0 failed / 0 skipped`
- hot surface atomic pointer-alias active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.4ms`, speedup `1.41`
- surface-read atomic vector readback fixture: `surface:pointer-alias-atomic-vector-readback` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surface atomic vector readback probe: passed
- compiler typecheck after surface atomic vector readback probe: passed
- WebGPU smoke after surface atomic vector readback probe: `170 passed / 0 failed / 0 skipped`
- hot surface atomic vector readback probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.5ms`, speedup `1.40`
- surface-read atomic vector compound fixture: `surface:pointer-alias-atomic-vector-compound` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surface atomic vector compound probe: passed
- compiler typecheck after surface atomic vector compound probe: passed
- WebGPU smoke after surface atomic vector compound probe: `171 passed / 0 failed / 0 skipped`
- hot surface atomic vector compound probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.3ms`, speedup `1.23`
- 1D surface read fixture: `surface:surf1d-read` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after 1D surface read probe: passed
- compiler typecheck after 1D surface read probe: passed
- compiler unit suite after 1D surface read probe: `412 passed / 0 failed`
- WebGPU smoke after 1D surface read probe: `172 passed / 0 failed / 0 skipped`
- hot 1D surface read probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.5ms`, speedup `1.11`
- 1D surface vector read fixture: `surface:surf1d-vector-read` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after 1D surface vector read probe: passed
- compiler typecheck after 1D surface vector read probe: passed
- compiler unit suite after 1D surface vector read probe: `412 passed / 0 failed`
- WebGPU smoke after 1D surface vector read probe: `173 passed / 0 failed / 0 skipped`
- hot 1D surface vector read probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.4ms`, speedup `1.24`
- 1D surface vector write fixture: `surface:surf1d-vector-write` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after 1D surface vector write probe: passed
- compiler typecheck after 1D surface vector write probe: passed
- compiler unit suite after 1D surface vector write probe: `412 passed / 0 failed`
- WebGPU smoke after 1D surface vector write probe: `174 passed / 0 failed / 0 skipped`
- hot 1D surface vector write probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.0ms`, speedup `1.33`
- 1D surface vector active-lane fixture: `surface:surf1d-vector-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after 1D surface vector active-lane probe: passed
- compiler typecheck after 1D surface vector active-lane probe: passed
- compiler unit suite after 1D surface vector active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after 1D surface vector active-lane probe: `175 passed / 0 failed / 0 skipped`
- hot 1D surface vector active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.2ms`, speedup `1.31`
- surf1Dread pointer-alias active-lane fixture: `surface:surf1d-pointer-alias-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surf1Dread pointer-alias active-lane probe: passed
- compiler typecheck after surf1Dread pointer-alias active-lane probe: passed
- compiler unit suite after surf1Dread pointer-alias active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after surf1Dread pointer-alias active-lane probe: `176 passed / 0 failed / 0 skipped`
- hot surf1Dread pointer-alias active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.5ms`, speedup `1.29`
- surf1Dread atomic pointer-alias active-lane fixture: `surface:surf1d-pointer-alias-atomic-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surf1Dread atomic pointer-alias active-lane probe: passed
- compiler typecheck after surf1Dread atomic pointer-alias active-lane probe: passed
- compiler unit suite after surf1Dread atomic pointer-alias active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after surf1Dread atomic pointer-alias active-lane probe: `177 passed / 0 failed / 0 skipped`
- hot surf1Dread atomic pointer-alias active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.3ms`, speedup `1.45`
- surf1Dread atomic vector readback fixture: `surface:surf1d-pointer-alias-atomic-vector-readback` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surf1Dread atomic vector readback probe: passed
- compiler typecheck after surf1Dread atomic vector readback probe: passed
- compiler unit suite after surf1Dread atomic vector readback probe: `412 passed / 0 failed`
- WebGPU smoke after surf1Dread atomic vector readback probe: `178 passed / 0 failed / 0 skipped`
- hot surf1Dread atomic vector readback probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.7ms`, speedup `1.38`
- surf1Dread atomic vector compound fixture: `surface:surf1d-pointer-alias-atomic-vector-compound` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surf1Dread atomic vector compound probe: passed
- compiler typecheck after surf1Dread atomic vector compound probe: passed
- compiler unit suite after surf1Dread atomic vector compound probe: `412 passed / 0 failed`
- WebGPU smoke after surf1Dread atomic vector compound probe: `179 passed / 0 failed / 0 skipped`
- hot surf1Dread atomic vector compound probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.1ms`, speedup `1.39`
- surf3D vector write active-lane fixture: `surface:surf3d-vector-write-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surf3D vector write active-lane probe: passed
- compiler typecheck after surf3D vector write active-lane probe: passed
- compiler unit suite after surf3D vector write active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after surf3D vector write active-lane probe: `180 passed / 0 failed / 0 skipped`
- hot surf3D vector write active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `1.1ms`, speedup `4.27`
- multi-surface helper vector active-lane fixture: `surface:helper-vector-multi-surface-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after multi-surface helper vector active-lane probe: passed
- compiler typecheck after multi-surface helper vector active-lane probe: passed
- compiler unit suite after multi-surface helper vector active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after multi-surface helper vector active-lane probe: `181 passed / 0 failed / 0 skipped`
- hot multi-surface helper vector active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.3ms`, speedup `1.39`
- atlas/3D texture vector active-lane fixture: `texture:atlas-vector-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after atlas/3D texture vector active-lane probe: passed
- compiler typecheck after atlas/3D texture vector active-lane probe: passed
- compiler unit suite after atlas/3D texture vector active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after atlas/3D texture vector active-lane probe: `182 passed / 0 failed / 0 skipped`
- hot atlas/3D texture vector active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.2ms`, speedup `1.38`
- atlas/3D texture vector pointer-alias active-lane fixture: `texture:atlas-vector-pointer-alias-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after atlas/3D texture vector pointer-alias active-lane probe: passed
- compiler typecheck after atlas/3D texture vector pointer-alias active-lane probe: passed
- compiler unit suite after atlas/3D texture vector pointer-alias active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after atlas/3D texture vector pointer-alias active-lane probe: `183 passed / 0 failed / 0 skipped`
- hot atlas/3D texture vector pointer-alias active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.0ms`, speedup `1.25`
- atlas/3D texture vector atomic pointer-alias active-lane fixture: `texture:atlas-vector-atomic-pointer-alias-active-lane-store` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after atlas/3D texture vector atomic pointer-alias active-lane probe: passed
- compiler typecheck after atlas/3D texture vector atomic pointer-alias active-lane probe: passed
- compiler unit suite after atlas/3D texture vector atomic pointer-alias active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after atlas/3D texture vector atomic pointer-alias active-lane probe: `184 passed / 0 failed / 0 skipped`
- hot atlas/3D texture vector atomic pointer-alias active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.7ms`, speedup `1.30`
- atlas/3D texture vector atomic pointer-alias readback fixture: `texture:atlas-vector-atomic-pointer-alias-readback` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after atlas/3D texture vector atomic pointer-alias readback probe: passed
- compiler typecheck after atlas/3D texture vector atomic pointer-alias readback probe: passed
- compiler unit suite after atlas/3D texture vector atomic pointer-alias readback probe: `412 passed / 0 failed`
- WebGPU smoke after atlas/3D texture vector atomic pointer-alias readback probe: `185 passed / 0 failed / 0 skipped`
- hot atlas/3D texture vector atomic pointer-alias readback probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `3.9ms`, speedup `1.38`
- atlas/3D texture vector atomic pointer-alias compound fixture: `texture:atlas-vector-atomic-pointer-alias-compound` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after atlas/3D texture vector atomic pointer-alias compound probe: passed
- compiler typecheck after atlas/3D texture vector atomic pointer-alias compound probe: passed
- compiler unit suite after atlas/3D texture vector atomic pointer-alias compound probe: `412 passed / 0 failed`
- WebGPU smoke after atlas/3D texture vector atomic pointer-alias compound probe: `186 passed / 0 failed / 0 skipped`
- hot atlas/3D texture vector atomic pointer-alias compound probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.2ms`, speedup `1.38`
- atlas/3D texture vector atomic pointer-array selection fixture: `texture:atlas-vector-atomic-pointer-array-select` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after atlas/3D texture vector atomic pointer-array selection probe: passed
- compiler typecheck after atlas/3D texture vector atomic pointer-array selection probe: passed
- compiler unit suite after atlas/3D texture vector atomic pointer-array selection probe: `412 passed / 0 failed`
- WebGPU smoke after atlas/3D texture vector atomic pointer-array selection probe: `187 passed / 0 failed / 0 skipped`
- hot atlas/3D texture vector atomic pointer-array selection probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.3ms`, speedup `1.35`
- surface vector atomic pointer-array selection fixture: `surface:pointer-alias-atomic-pointer-array-select` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surface vector atomic pointer-array selection probe: passed
- compiler typecheck after surface vector atomic pointer-array selection probe: passed
- compiler unit suite after surface vector atomic pointer-array selection probe: `412 passed / 0 failed`
- WebGPU smoke after surface vector atomic pointer-array selection probe: `188 passed / 0 failed / 0 skipped`
- hot surface vector atomic pointer-array selection probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.5ms`, speedup `1.27`
- surf1D vector atomic pointer-array selection fixture: `surface:surf1d-pointer-alias-atomic-pointer-array-select` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after surf1D vector atomic pointer-array selection probe: passed
- compiler typecheck after surf1D vector atomic pointer-array selection probe: passed
- compiler unit suite after surf1D vector atomic pointer-array selection probe: `412 passed / 0 failed`
- WebGPU smoke after surf1D vector atomic pointer-array selection probe: `189 passed / 0 failed / 0 skipped`
- hot surf1D vector atomic pointer-array selection probe: repeat `9`, warmup `1`, `9 passed / 0 failed / 0 skipped`, best warm `1.7ms`, speedup `3.06`; earlier repeat `5` run was correctness-green but speedup-noisy at `0.67`
- texture-to-surface uint4 atomic pointer-array selection fixture: `texture-surface:uint4-atomic-pointer-array-select` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after texture-to-surface uint4 atomic pointer-array selection probe: passed
- compiler typecheck after texture-to-surface uint4 atomic pointer-array selection probe: passed
- compiler unit suite after texture-to-surface uint4 atomic pointer-array selection probe: `412 passed / 0 failed`
- WebGPU smoke after texture-to-surface uint4 atomic pointer-array selection probe: `190 passed / 0 failed / 0 skipped`
- hot texture-to-surface uint4 atomic pointer-array selection probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.3ms`, speedup `1.30`
- volume texture-to-surface atomic pointer-array selection fixture: `texture-surface:volume-atomic-pointer-array-select` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after volume texture-to-surface atomic pointer-array selection probe: passed
- compiler typecheck after volume texture-to-surface atomic pointer-array selection probe: passed
- compiler unit suite after volume texture-to-surface atomic pointer-array selection probe: `412 passed / 0 failed`
- WebGPU smoke after volume texture-to-surface atomic pointer-array selection probe: `191 passed / 0 failed / 0 skipped`
- hot volume texture-to-surface atomic pointer-array selection probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.5ms`, speedup `1.31`
- volume texture-to-surface atomic pointer-array active-lane return fixture: `texture-surface:volume-atomic-pointer-array-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after volume texture-to-surface atomic pointer-array active-lane probe: passed
- compiler typecheck after volume texture-to-surface atomic pointer-array active-lane probe: passed
- compiler unit suite after volume texture-to-surface atomic pointer-array active-lane probe: `412 passed / 0 failed`
- WebGPU smoke after volume texture-to-surface atomic pointer-array active-lane probe: `192 passed / 0 failed / 0 skipped`
- hot volume texture-to-surface atomic pointer-array active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.7ms`, speedup `1.28`
- vector-pointer array atomic pointer arithmetic repro: `texture-surface:volume-vector-pointer-array-atomic-active-lane-return` initially failed WGSL pipeline creation with `no matching call to 'atomicAdd(u32, u32)'`
- vector-pointer array atomic pointer arithmetic fixture after root fix: `texture-surface:volume-vector-pointer-array-atomic-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after vector-pointer array atomic pointer arithmetic fix: passed
- compiler typecheck after vector-pointer array atomic pointer arithmetic fix: passed
- compiler unit suite after vector-pointer array atomic pointer arithmetic fix: `413 passed / 0 failed`
- WGSL module suite after vector-pointer array atomic pointer arithmetic fix: `16 passed / 0 failed`
- WebGPU smoke after vector-pointer array atomic pointer arithmetic fix: `193 passed / 0 failed / 0 skipped`
- scoped pointer/storage WebGPU cases after vector-pointer array atomic pointer arithmetic fix: `35 passed / 0 failed / 0 skipped`
- hot vector-pointer array atomic pointer arithmetic probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `5.2ms`, speedup `1.31`
- bugbash status test after vector-pointer array atomic pointer arithmetic fix: passed
- changed-test-scope test after vector-pointer array atomic pointer arithmetic fix: passed
- volume vector-pointer array CAS/exchange active-lane fixture: `texture-surface:volume-vector-pointer-array-cas-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- compiler fixture test after volume vector-pointer array CAS/exchange probe: passed
- compiler typecheck after volume vector-pointer array CAS/exchange probe: passed
- compiler unit suite after volume vector-pointer array CAS/exchange probe: `414 passed / 0 failed`
- WGSL module suite after volume vector-pointer array CAS/exchange probe: `16 passed / 0 failed`
- WebGPU smoke after volume vector-pointer array CAS/exchange probe: `194 passed / 0 failed / 0 skipped`
- hot volume vector-pointer array CAS/exchange probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `2.8ms`, speedup `2.29`
- compiler build after vector pointer flat-lane storage fix: passed
- compiler typecheck after vector pointer flat-lane storage fix: passed
- compiler unit suite after vector pointer flat-lane storage fix: `418 passed / 0 failed`
- WGSL module suite after vector pointer flat-lane storage fix: `16 passed / 0 failed`
- WebGPU fixture tests after vector pointer flat-lane storage fix: passed
- WebGPU smoke after vector pointer flat-lane storage fix: `195 passed / 0 failed / 0 skipped`
- hot volume vector-pointer array compound active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `5.7ms`, speedup `1.33`
- bugbash status tooling after vector pointer flat-lane storage fix: passed
- changed-test-scope tooling after vector pointer flat-lane storage fix: passed
- volume vector-pointer array min/max active-lane fixture: `texture-surface:volume-vector-pointer-array-minmax-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- WebGPU fixture test after min/max active-lane probe: passed
- WebGPU smoke after min/max active-lane probe: `196 passed / 0 failed / 0 skipped`
- hot volume vector-pointer array min/max active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `5.3ms`, speedup `1.36`
- changed-test-scope tooling after min/max active-lane probe: passed
- post-loop divergent break/barrier fixture: `control:active-lane-break-post-loop-barrier` is `1 passed / 0 failed / 0 skipped`
- compiler unit suite after post-loop break/barrier fix: `419 passed / 0 failed`
- compiler typecheck after post-loop break/barrier fix: passed
- WGSL module suite after post-loop break/barrier fix: `16 passed / 0 failed`
- WebGPU fixture test after post-loop break/barrier fix: passed
- WebGPU smoke after post-loop break/barrier fix: `197 passed / 0 failed / 0 skipped`
- hot post-loop break/barrier probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `1.2ms`, speedup `3.83`
- while/do-while post-loop divergent break/barrier fixtures: `2 passed / 0 failed / 0 skipped`
- hot while/do-while post-loop break/barrier probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `3.0ms` / `2.7ms`, speedups `1.17` / `1.07`
- compiler unit suite after while/do-while break/barrier fix: `420 passed / 0 failed`
- compiler typecheck after while/do-while break/barrier fix: passed
- WGSL module suite after while/do-while break/barrier fix: `16 passed / 0 failed`
- WebGPU fixture test after while/do-while break/barrier fix: passed
- WebGPU smoke after while/do-while break/barrier fix: `199 passed / 0 failed / 0 skipped`
- texture pointer-alias atomic pointer-array select fixture: `texture:pointer-alias-atomic-pointer-array-select` is `1 passed / 0 failed / 0 skipped`
- hot texture pointer-alias atomic pointer-array select probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.5ms`, speedup `1.40`
- WebGPU fixture test after texture pointer-array select probe: passed
- WebGPU smoke after texture pointer-array select probe: `200 passed / 0 failed / 0 skipped`
- texture pointer-alias atomic pointer-array active-lane fixture: `texture:pointer-alias-atomic-pointer-array-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- hot texture pointer-alias atomic pointer-array active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.6ms`, speedup `1.26`
- WebGPU fixture test after texture pointer-array active-lane probe: passed
- WebGPU smoke after texture pointer-array active-lane probe: `201 passed / 0 failed / 0 skipped`
- surface pointer-alias atomic pointer-array active-lane fixture: `surface:pointer-alias-atomic-pointer-array-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- hot surface pointer-alias atomic pointer-array active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.6ms`, speedup `1.24`
- WebGPU fixture test after surface pointer-array active-lane probe: passed
- WebGPU smoke after surface pointer-array active-lane probe: `202 passed / 0 failed / 0 skipped`
- surf1D pointer-alias atomic pointer-array active-lane fixture: `surface:surf1d-pointer-alias-atomic-pointer-array-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- hot surf1D pointer-alias atomic pointer-array active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `2.7ms`, speedup `2.07`
- WebGPU fixture test after surf1D pointer-array active-lane probe: passed
- WebGPU smoke after surf1D pointer-array active-lane probe: `203 passed / 0 failed / 0 skipped`
- surf1D pointer-alias atomic pointer-array CAS/exchange active-lane fixture: `surface:surf1d-pointer-alias-atomic-pointer-array-cas-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- hot surf1D pointer-alias atomic pointer-array CAS/exchange active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.9ms`, speedup `1.18`
- WebGPU fixture test after surf1D pointer-array CAS/exchange probe: passed
- WebGPU smoke after surf1D pointer-array CAS/exchange probe: `204 passed / 0 failed / 0 skipped`
- surf1D pointer-alias atomic pointer-array min/max active-lane fixture: `surface:surf1d-pointer-alias-atomic-pointer-array-minmax-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- hot surf1D pointer-alias atomic pointer-array min/max active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.7ms`, speedup `1.30`
- WebGPU fixture test after surf1D pointer-array min/max probe: passed
- WebGPU smoke after surf1D pointer-array min/max probe: `205 passed / 0 failed / 0 skipped`
- layered surface pointer-alias atomic pointer-array CAS/minmax active-lane fixtures: `surface:pointer-alias-atomic-pointer-array-cas-active-lane-return,surface:pointer-alias-atomic-pointer-array-minmax-active-lane-return` are `2 passed / 0 failed / 0 skipped`
- hot layered surface pointer-alias atomic pointer-array CAS/minmax active-lane probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.3ms` / `4.4ms`, speedups `1.28` / `1.25`
- WebGPU fixture test after layered surface pointer-array CAS/minmax probe: passed
- WebGPU smoke after layered surface pointer-array CAS/minmax probe: `207 passed / 0 failed / 0 skipped`
- texture pointer-alias atomic pointer-array CAS/minmax active-lane fixtures: `texture:pointer-alias-atomic-pointer-array-cas-active-lane-return,texture:pointer-alias-atomic-pointer-array-minmax-active-lane-return` are `2 passed / 0 failed / 0 skipped`
- hot texture pointer-alias atomic pointer-array CAS/minmax active-lane probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.3ms` / `4.2ms`, speedups `1.37` / `1.26`
- WebGPU fixture test after texture pointer-array CAS/minmax probe: passed
- WebGPU smoke after texture pointer-array CAS/minmax probe: `209 passed / 0 failed / 0 skipped`
- atlas texture vector atomic pointer-array CAS/minmax active-lane fixtures: `texture:atlas-vector-atomic-pointer-array-cas-active-lane-return,texture:atlas-vector-atomic-pointer-array-minmax-active-lane-return` are `2 passed / 0 failed / 0 skipped`
- hot atlas texture vector atomic pointer-array CAS/minmax active-lane probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.5ms` / `4.5ms`, speedups `1.38` / `1.11`
- WebGPU fixture test after atlas texture pointer-array CAS/minmax probe: passed
- WebGPU smoke after atlas texture pointer-array CAS/minmax probe: `211 passed / 0 failed / 0 skipped`
- texture/atlas vector atomic pointer-array compound active-lane fixtures: `texture:pointer-alias-atomic-pointer-array-compound-active-lane-return,texture:atlas-vector-atomic-pointer-array-compound-active-lane-return` are `2 passed / 0 failed / 0 skipped`
- hot texture/atlas vector atomic pointer-array compound active-lane probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.6ms` / `4.6ms`, speedups `1.15` / `1.33`
- WebGPU fixture test after texture/atlas pointer-array compound probe: passed
- WebGPU smoke after texture/atlas pointer-array compound probe: `213 passed / 0 failed / 0 skipped`
- surf1D/layered surface pointer-array compound active-lane fixtures: `surface:surf1d-pointer-alias-atomic-pointer-array-compound-active-lane-return,surface:pointer-alias-atomic-pointer-array-compound-active-lane-return` are `2 passed / 0 failed / 0 skipped`
- hot surf1D/layered surface pointer-array compound active-lane probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.7ms` / `4.7ms`, speedups `1.19` / `1.19`
- WebGPU fixture test after surf1D/layered surface pointer-array compound probe: passed
- WebGPU smoke after surf1D/layered surface pointer-array compound probe: `215 passed / 0 failed / 0 skipped`
- surf3D surface pointer-array compound active-lane fixture: `surface:surf3d-pointer-alias-atomic-pointer-array-compound-active-lane-return` is `1 passed / 0 failed / 0 skipped`
- hot surf3D surface pointer-array compound active-lane probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `4.6ms`, speedup `1.35`
- WebGPU fixture test after surf3D surface pointer-array compound probe: passed
- WebGPU smoke after surf3D surface pointer-array compound probe: `216 passed / 0 failed / 0 skipped`
- surf3D surface pointer-array CAS/minmax active-lane fixtures: `surface:surf3d-pointer-alias-atomic-pointer-array-cas-active-lane-return,surface:surf3d-pointer-alias-atomic-pointer-array-minmax-active-lane-return` are `2 passed / 0 failed / 0 skipped`
- hot surf3D surface pointer-array CAS/minmax active-lane probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.5ms` / `4.4ms`, speedups `1.16` / `1.09`
- WebGPU smoke after surf3D surface pointer-array CAS/minmax probe: `218 passed / 0 failed / 0 skipped`
- surf3D surface pointer-array select/basic active-lane fixtures: `surface:surf3d-pointer-alias-atomic-pointer-array-select,surface:surf3d-pointer-alias-atomic-pointer-array-active-lane-return` are `2 passed / 0 failed / 0 skipped`
- hot surf3D surface pointer-array select/basic active-lane probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.0ms` / `3.5ms`, speedups `1.18` / `1.31`
- WebGPU smoke after surf3D surface pointer-array select/basic active-lane probe: `220 passed / 0 failed / 0 skipped`
- surf3D surface pointer-alias active/atomic vector fixtures: `surface:surf3d-pointer-alias-active-lane-store,surface:surf3d-pointer-alias-atomic-active-lane-store,surface:surf3d-pointer-alias-atomic-vector-readback,surface:surf3d-pointer-alias-atomic-vector-compound` are `4 passed / 0 failed / 0 skipped`
- hot surf3D surface pointer-alias active/atomic vector probe: repeat `5`, warmup `1`, `20 passed / 0 failed / 0 skipped`, best warm `3.2ms` / `2.9ms` / `3.0ms` / `3.9ms`, speedups `1.16` / `1.24` / `1.23` / `1.21`
- WebGPU smoke after surf3D surface pointer-alias active/atomic vector probe: `224 passed / 0 failed / 0 skipped`
- surf3D helper multi-surface + guarded RHS fixtures: `surface:surf3d-helper-vector-multi-surface-active-lane-return,surface:surf3d-active-lane-guarded-rhs` are `2 passed / 0 failed / 0 skipped`
- hot surf3D helper multi-surface + guarded RHS probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `3.5ms` / `2.9ms`, speedups `1.14` / `1.45`
- WebGPU smoke after surf3D helper multi-surface + guarded RHS probe: `226 passed / 0 failed / 0 skipped`
- atlas/volume texture guarded RHS fixtures: `texture:atlas-active-lane-guarded-rhs,texture-surface:volume-active-lane-guarded-rhs` are `2 passed / 0 failed / 0 skipped`
- hot atlas/volume texture guarded RHS probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.1ms` / `3.8ms`, speedups `1.17` / `1.13`
- WebGPU smoke after atlas/volume texture guarded RHS probe: `228 passed / 0 failed / 0 skipped`
- texture/surface pointer-array false-branch fixtures: `texture:pointer-alias-atomic-pointer-array-select-false-branch,texture:pointer-alias-atomic-pointer-array-active-lane-return-false-branch,texture-surface:uint4-atomic-pointer-array-select-false-branch,texture-surface:volume-atomic-pointer-array-select-false-branch` are `4 passed / 0 failed / 0 skipped`
- hot texture/surface pointer-array false-branch probe: repeat `5`, warmup `1`, `20 passed / 0 failed / 0 skipped`, best warm `2.2ms` / `4.1ms` / `4.2ms` / `4.4ms`, speedups `2.18` / `1.17` / `1.26` / `1.09`
- WebGPU smoke after texture/surface pointer-array false-branch probe: `232 passed / 0 failed / 0 skipped`
- surface/atlas pointer-array false-branch fixtures: `surface:pointer-alias-atomic-pointer-array-select-false-branch,surface:surf1d-pointer-alias-atomic-pointer-array-select-false-branch,surface:surf3d-pointer-alias-atomic-pointer-array-select-false-branch,texture:atlas-vector-atomic-pointer-array-select-false-branch` are `4 passed / 0 failed / 0 skipped`
- hot surface/atlas pointer-array false-branch probe: repeat `5`, warmup `1`, `20 passed / 0 failed / 0 skipped`, best warm `1.7ms` / `1.9ms` / `4.2ms` / `4.2ms`, speedups `2.76` / `1.16` / `1.17` / `1.12`
- WebGPU smoke after surface/atlas pointer-array false-branch probe: `236 passed / 0 failed / 0 skipped`
- volume vector-pointer false-branch fixtures: `texture-surface:volume-vector-pointer-array-atomic-active-lane-return-false-branch,texture-surface:volume-vector-pointer-array-cas-active-lane-return-false-branch,texture-surface:volume-vector-pointer-array-compound-active-lane-return-false-branch,texture-surface:volume-vector-pointer-array-minmax-active-lane-return-false-branch` are `4 passed / 0 failed / 0 skipped`
- hot volume vector-pointer false-branch probe: repeat `5`, warmup `1`, `20 passed / 0 failed / 0 skipped`, best warm `3.9ms` / `2.4ms` / `4.8ms` / `4.7ms`, speedups `1.33` / `2.25` / `1.23` / `1.26`
- WebGPU smoke after volume vector-pointer false-branch probe: `240 passed / 0 failed / 0 skipped`
- surface CAS/minmax false-branch fixtures: `surface:surf3d-pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch,surface:surf3d-pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch,surface:surf1d-pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch,surface:surf1d-pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch,surface:pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch,surface:pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch` are `6 passed / 0 failed / 0 skipped`
- hot surface CAS/minmax false-branch probe: repeat `5`, warmup `1`, `30 passed / 0 failed / 0 skipped`, best warm `3.9ms` / `1.5ms` / `1.7ms` / `1.8ms` / `3.6ms` / `3.7ms`, speedups `1.23` / `2.87` / `2.59` / `2.39` / `1.03` / `1.14`
- WebGPU smoke after surface CAS/minmax false-branch probe: `246 passed / 0 failed / 0 skipped`
- texture CAS/minmax false-branch fixtures: `texture:atlas-vector-atomic-pointer-array-cas-active-lane-return-false-branch,texture:atlas-vector-atomic-pointer-array-minmax-active-lane-return-false-branch,texture:pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch,texture:pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch` are `4 passed / 0 failed / 0 skipped`
- hot texture CAS/minmax false-branch probe: repeat `5`, warmup `1`, `20 passed / 0 failed / 0 skipped`, best warm `4.1ms` / `4.3ms` / `4.0ms` / `4.0ms`, speedups `1.37` / `1.16` / `1.38` / `1.00`
- WebGPU smoke after texture CAS/minmax false-branch probe: `250 passed / 0 failed / 0 skipped`
- compound false-branch fixtures: `surface:surf3d-pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch,surface:surf1d-pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch,surface:pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch,texture:atlas-vector-atomic-pointer-array-compound-active-lane-return-false-branch,texture:pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch` are `5 passed / 0 failed / 0 skipped`
- hot compound false-branch probe: repeat `5`, warmup `1`, `25 passed / 0 failed / 0 skipped`, best warm `4.1ms` / `4.0ms` / `3.9ms` / `4.3ms` / `4.1ms`, speedups `1.32` / `1.33` / `1.21` / `1.16` / `1.22`
- WebGPU smoke after compound false-branch probe: `255 passed / 0 failed / 0 skipped`
- all-inactive guarded-RHS fixtures: `control:active-lane-guarded-rhs-all-inactive,control:active-lane-assignment-guarded-rhs-all-inactive,control:active-lane-vector-atomic-guarded-rhs-all-inactive,control:active-lane-compound-assignment-guarded-rhs-all-inactive,surface:surf3d-active-lane-guarded-rhs-all-inactive,texture:atlas-active-lane-guarded-rhs-all-inactive,texture-surface:volume-active-lane-guarded-rhs-all-inactive` are `7 passed / 0 failed / 0 skipped`
- hot all-inactive guarded-RHS probe: repeat `5`, warmup `1`, `35 passed / 0 failed / 0 skipped`, best warm `2.7ms` / `2.8ms` / `2.8ms` / `2.8ms` / `2.8ms` / `2.7ms` / `3.3ms`, speedups `1.22` / `1.11` / `1.21` / `1.11` / `1.07` / `1.26` / `1.15`
- WebGPU smoke after all-inactive guarded-RHS probe: `262 passed / 0 failed / 0 skipped`
- texture-surface active-lane scalar pointer-array false-branch fixture: `texture-surface:volume-atomic-pointer-array-active-lane-return-false-branch` is `1 passed / 0 failed / 0 skipped`
- hot texture-surface active-lane scalar pointer-array false-branch probe: repeat `5`, warmup `1`, `5 passed / 0 failed / 0 skipped`, best warm `5.0ms`, speedup `1.24`
- WebGPU smoke after texture-surface active-lane scalar pointer-array false-branch probe: `263 passed / 0 failed / 0 skipped`
- stale last-failure status fix: green focused rerun clears `.tmp/cuda-lite-last-failures.json`; bugbash status reports `Active failure cases: 0`
- 2D texture-to-layered-surface active-lane scalar pointer-array fixtures: `texture-surface:uint4-atomic-pointer-array-active-lane-return,texture-surface:uint4-atomic-pointer-array-active-lane-return-false-branch` are `2 passed / 0 failed / 0 skipped`
- hot 2D texture-to-layered-surface active-lane scalar pointer-array probe: repeat `5`, warmup `1`, `10 passed / 0 failed / 0 skipped`, best warm `4.3ms` / `4.7ms`, speedups `1.35` / `1.13`
- WebGPU smoke after 2D texture-to-layered-surface active-lane scalar pointer-array probe: `265 passed / 0 failed / 0 skipped`
- report-output parent-dir fix: WebGPU smoke writes nested JSON, markdown, and progress files under `.tmp/reports`; smoke remains `265 passed / 0 failed / 0 skipped`
- benchmark report-output parent-dir fix: compiler benchmark writes nested markdown; WebGPU benchmark writes nested JSON and markdown
- nested divergent break before post-loop barrier fix: focused compiler unit `421/0`; focused real WebGPU break group `5/0/0`; new fixture `control:active-lane-nested-break-post-loop-barrier` passes with `0` skips
- divergent continue before later barrier safety fix: fail-first compiler unit now `422/0`; safe canonical continue regression remains `422/0`; diagnostic is `divergent-continue-before-barrier`
- changed gate after divergent continue safety fix: typecheck passed; compiler unit file `406/0`; bugbash-status tests passed; corpus audit tests passed; WebGPU smoke `266/0/0`
- exact WebGPU case-filter fix: fail-first harness unit covered exact-vs-fuzzy filters; focused real WebGPU rerun for `texture-surface:volume-vector-pointer-array-minmax-active-lane-return` now runs `1/0/0` instead of also running the `-false-branch` sibling; fuzzy filter `minmax-active-lane-return` still runs `12/0/0`
- changed gate after exact case-filter fix: `test:webgpu-fixtures` passed; WebGPU smoke `266/0/0`; bugbash-status tests passed; active failures `0`
- compact WebGPU summary-output fix: `summarizeReport` now keeps small filter lists verbatim but summarizes large filter lists with `count`, `first`, and `last`; harness unit passed; 21-case real WebGPU sample `21/0/0` emitted compact filters
- WebGPU smoke after compact summary-output fix: `266/0/0`; summary output now shows compact `caseFilters` count/sample instead of the full 266-case list
- WebGPU smoke after wrapper output fix: `266/0/0`; `e2e:webgpu:smoke` now invokes `run-cuda-lite-tool.mjs` directly, avoiding the inner `pnpm run e2e:webgpu:case -- --cases ...` echo
- corpus audit class-member helper fix: cuda-samples compile/codegen improved from `355/357` to `356/357`; hard fails improved from `2` to `1`; `missing-kernel` is gone for `multiGpuConjugateGradient`
- loop-local divergent continue fix: compiler unit file `407/0`; focused unit run `423/0`; llm.c audit restored `148/0/0`; full real-world corpus audit green; verifier src/dist `419/0/0`
- WebGPU smoke after loop-local continue/class-member fixes: `266/0/0`
- shared-byte reinterpret local pointer fix: focused byte-reinterpret unit `424/0`; `storage:shared-byte-reinterpret` `1/0/0`; WebGPU smoke `267/0/0`; cuda-samples audit compile/codegen `357/357`, hard fails `0`
- real-world CUDA verifier after shared-byte reinterpret fix: src `420/0/0`, dist `420/0/0`
- concurrent shared-byte packed write fix: fail-first `storage:shared-byte-concurrent-writes` reproduced `0x04000000` instead of `0x04030201`; after atomic packed-byte lowering, focused unit `424/0`, real WebGPU case `1/0/0`, storage group `37/0/0`, WebGPU smoke `268/0/0`
- real-world CUDA verifier after concurrent shared-byte write fix: src `421/0/0`, dist `421/0/0`
- packed shared-byte float view fix: fail-first `storage:shared-byte-float-reinterpret` failed WGSL pipeline creation with returned `u32`, expected `f32`; after bitcast carrier policy, focused unit `425/0`, compiler unit `409/0`, storage group `38/0/0`, real WebGPU case `1/0/0`, WebGPU smoke `269/0/0`
- real-world CUDA verifier after packed shared-byte float view fix: src `422/0/0`, dist `422/0/0`
- scoped local pointer handle fix: fail-first `storage:shared-byte-half-reinterpret` first failed WGSL pipeline creation with redeclared `value_buffer`/`value_base`; after span-scoped local pointer handle names, compiler unit `411/0`
- packed shared-byte half view fix: after scoped pointer fix, fail-first `storage:shared-byte-half-reinterpret` failed compare with actual `0`; after 16-bit lane read/write and helper compatibility fix, focused real WebGPU case `1/0/0`; compiler unit `411/0`
- changed gate after scoped pointer/half shared-byte fixes: typecheck passed; lint passed; compiler unit `411/0`; WGSL modules `16/0`; storage focused group `4/0/0`; selected pointer/storage WebGPU group `39/0/0`; WebGPU smoke `270/0/0`; fixture/status/scope tests passed
- real-world CUDA verifier after scoped pointer/half shared-byte fixes: src `423/0/0`, dist `423/0/0`
- packed shared-byte bf16 view fix: fail-first `storage:shared-byte-bf16-reinterpret` failed with `unsupported-local-pointer` for `__nv_bfloat16*` over `uchar` shared storage; after analyzer/helper/storage/reference fixes, compiler unit `412/0`, focused real WebGPU case `1/0/0`
- changed gate after packed shared-byte bf16 fix: typecheck passed; lint passed; compiler unit `412/0`; WGSL modules `16/0`; storage focused group `5/0/0`; selected pointer/storage WebGPU group `40/0/0`; WebGPU smoke `271/0/0`; fixture/status/scope tests passed
- real-world CUDA verifier after packed shared-byte bf16 fix: src `424/0/0`, dist `424/0/0`
- WMMA shared-memory alias shadowing fix: fail-first minimal reproduced same-name `tile_ptr` scopes where an earlier alias was skipped because a later declaration needed a local pointer handle; focused unit `1 passed`; cuda-samples compile/codegen audit improved to `357/357`, hard fails `0`
- real-world CUDA verifier after helper byte pointer overlay fix: src `429/0/0`, dist `429/0/0`
- unaligned typed byte-storage overlay fix: fail-first probe showed `(float *)&scratch[1]` over `uchar*` accepted but placed values at word-scaled offsets; focused unit `1 passed`; `storage:param-byte-unaligned-float-helper-reinterpret` `1/0/0`
- compiler unit suite after unaligned typed byte-storage overlay fix: `419 passed / 0 failed`
- changed gate after unaligned typed byte-storage overlay fix: typecheck passed; compiler unit `419/0`; WGSL modules `16/0`; selected storage/pointer WebGPU group `45/0/0`; WebGPU smoke `276/0/0`; fixture/status/scope tests passed
- compiler lint after unaligned typed byte-storage overlay fix: passed
- real-world CUDA verifier after unaligned typed byte-storage overlay fix: src `430/0/0`, dist `430/0/0`
- helper byte-storage atomic probe: fail-first device helper `atomicAdd((uint *)&scratch[0], ...)` over `uchar*` generated `bg_ptr_atomicAdd_u32` with only the default case; after helper atomic byte-root address lowering, focused unit `1 passed`
- atomic packed-byte storage read fix: first WebGPU run for `storage:param-byte-uint-helper-atomic` failed pipeline creation because packed-byte reads shifted `atomic<u32>` storage directly; after atomic-aware packed-byte storage loads/writes, focused fixture `1/0/0`; related storage helper group `3/0/0`; related atomic helper units `7/0`
- compiler unit suite after helper byte-storage atomic fix: `420 passed / 0 failed`
- changed gate after helper byte-storage atomic fix: typecheck passed; compiler unit `420/0`; WGSL modules `16/0`; selected storage/pointer WebGPU group `45/0/0`; WebGPU smoke `276/0/0`; fixture/status/scope tests passed
- signed byte-storage atomic probe: fail-first direct/helper `atomicAdd((int *)&scratch[0], -3)` over `uchar*` either emitted a default-only helper or invalid `atomicAdd` against `atomic<u32>`; after signed-u32 carrier helpers and byte-root address lowering, focused unit group `2/0`
- focused WebGPU after signed byte-storage atomic fix: `storage:param-byte-int-helper-atomic,storage:param-byte-uint-helper-atomic` `2/0/0`
- compiler unit suite after signed byte-storage atomic fix: `421 passed / 0 failed`
- changed gate after signed byte-storage atomic fix: typecheck passed; compiler unit `421/0`; WGSL modules `16/0`; WebGPU smoke `276/0/0`; selected storage/atomic WebGPU group `80/0/0`; fixture/status/scope tests passed
- shared signed byte-storage atomic fixture: `storage:shared-byte-int-helper-atomic` covers signed helper add/min/max over `__shared__ uchar[]` and direct exchange at byte offset 4; focused WebGPU group `3/0/0` with param signed-byte guard and int-vector atomic guard
- int-view atomic helper emission trim: signed-u32 carrier helpers now emit only atomic kinds used by the module instead of the full add/sub/min/max/bitwise/exchange/CAS set
- changed gate after shared signed byte-storage fixture/helper trim: typecheck passed; WGSL modules `16/0`; WebGPU smoke `276/0/0`; selected atomic WebGPU group `35/0/0`; fixture/status/scope tests passed
- expanded shared signed byte-storage atomic fixture: `storage:shared-byte-int-helper-atomic` now covers helper add/sub/min/max/and/or/xor/CAS and direct CAS/and/or/xor/exchange over byte offset 4; focused WebGPU `1/0/0`
- shared byte vector helper pointer fixtures: `storage:shared-byte-half2-reinterpret,storage:shared-byte-bf162-reinterpret` now cover device helper read/write through vector pointers over `__shared__ uchar[]`; focused WebGPU `2/0/0`
- active-lane shared byte vector helper fixtures: `control:active-lane-shared-byte-half2-return-barrier,control:active-lane-shared-byte-bf162-return-barrier` cover pre-return helper writes through packed shared-byte vector pointers before a later barrier; focused WebGPU `2/0/0`
- unreachable half-helper feature-gate fix: focused unit run `438/0`; unused `half2` helpers no longer add `shader-f16` to a selected kernel that only reaches non-half helpers
- unreachable shared-helper declaration fix: focused unit run `439/0`; unused helper-local `__shared__` declarations no longer appear in IR/WGSL workgroup declarations
- unreachable helper atomic lowering fix: fail-first unit reproduced `atomicDeviceGlobals=["gCounter"]` from an unused helper; focused atomic unit slice `441/0`; `verify:changed` typecheck + compiler unit `425/0`; lint passed
- unreachable helper atomic WebGPU fixture: `atomic:unreachable-helper-plain-storage` `1/0/0`; asserts plain `gCounter` storage, plain `scratch` workgroup storage, no unreachable helper atomic WGSL declarations
- changed gate after unreachable-helper WebGPU fixture: fixture tests passed; WebGPU smoke `277/0/0`; bugbash status tests passed; bugbash status active failures `0`; test-scope tests passed
- WebGPU smoke wrapper profiling trim: default smoke no longer passes `--profile-case all`, has one `--case-timeout-ms`, and keeps profiling opt-in via `CUDA_LITE_WEBGPU_SMOKE_PROFILE`; focused smoke stayed `277/0/0`, test-scope passed
- unreachable helper dynamic-launch diagnostic fix: fail-first unit reproduced `unsupported-dynamic-parallelism` from an unused helper; focused dynamic-launch unit slice `442/0`; `runtime:unreachable-dynamic-launch` real WebGPU fixture `1/0/0`
- changed gate after unreachable dynamic-launch fixture: typecheck passed; compiler unit `426/0`; fixture tests passed; WebGPU smoke `278/0/0`; test-scope passed; lint passed
- unreachable helper compatibility diagnostic fix: fail-first unit reproduced unreachable `extern __shared__` diagnostics (`dynamic-shared-memory`, `unsupported-index-target`, `invalid-assignment-target`) and real WebGPU `non-uniform-return-before-barrier` from an unused helper; focused unit slice `444/0`; `runtime:unreachable-helper-compat-diagnostics` `1/0/0`
- changed gate after unreachable helper compatibility diagnostics: typecheck passed; compiler unit `428/0`; fixture tests passed; WebGPU smoke `279/0/0`; test-scope passed; lint passed
- unreachable helper runtime compatibility diagnostic fix: fail-first unit reproduced `unsupported-cuda-runtime` from unused `cudaMemcpy`/`cudaDeviceSynchronize` helper calls; focused unit slice `445/0`; `runtime:unreachable-runtime-compat-diagnostics` `1/0/0`
- changed gate after unreachable runtime compatibility diagnostics: typecheck passed; compiler unit `429/0`; fixture tests passed; WebGPU smoke `280/0/0`; bugbash status active failures `0`; test-scope passed; lint passed
- unreachable helper grid-sync compatibility diagnostic fix: fail-first unit reproduced `unsupported-cooperative-groups` from unused `grid.sync()` / `cg::sync(grid)` helper calls; focused unit slice `446/0`; `runtime:unreachable-grid-sync-compat-diagnostics` `1/0/0`
- changed gate after unreachable grid-sync compatibility diagnostics: typecheck passed; compiler unit `430/0`; fixture tests passed; WebGPU smoke `281/0/0`; bugbash status active failures `0`; test-scope passed; lint passed
- unreachable helper f64/inline-asm compatibility diagnostic fix: fail-first unit reproduced `unsupported-f64` and `unsupported-inline-asm` from unused helpers; focused unit slice `447/0`; `runtime:unreachable-feature-asm-compat-diagnostics` `1/0/0`; typecheck passed
- changed gate after unreachable f64/inline-asm diagnostics: typecheck passed; compiler unit `431/0`; fixture tests passed; WebGPU smoke `282/0/0`; bugbash status active failures `0`; test-scope passed; lint passed
- unreachable helper texture/surface diagnostic and binding fix: fail-first unit reproduced `unsupported-texture` / `unsupported-surface` from unused helpers, then exposed unused texture declarations still requiring CPU-reference texture input; focused unit slice `448/0`; `runtime:unreachable-texture-surface-compat-diagnostics` `1/0/0`; typecheck passed
- changed gate after unreachable texture/surface diagnostics: typecheck passed; compiler unit `432/0`; fixture tests passed; WebGPU smoke `283/0/0`; bugbash status active failures `0`; test-scope passed; lint passed
- unreachable helper constant binding fix: fail-first unit reproduced missing CPU-reference input for `__constant__` memory used only by unreachable helpers; focused unit slice `449/0`; `runtime:unreachable-constant-binding` `1/0/0`; typecheck passed
- changed gate after unreachable constant binding: typecheck passed; compiler unit `433/0`; fixture tests passed; WebGPU smoke `284/0/0`; bugbash status active failures `0`; test-scope passed; lint passed
- unreachable helper device-global binding fix: fail-first unit reproduced `__device__` globals used only by unreachable helpers still present in lowered IR/WGSL; focused unit slice `450/0`; `runtime:unreachable-device-global-binding` `1/0/0`; typecheck passed
- changed gate after unreachable device-global binding: typecheck passed; compiler unit `434/0`; fixture tests passed; WebGPU smoke `285/0/0`; test-scope passed; lint passed
- unreachable helper lowered-IR pruning fix: fail-first unit reproduced unused helper bodies still present in lowered IR; reachability now keeps selected-kernel reachable helpers and function-valued `cg::reduce` operators; focused unit slice `451/0`
- changed gate after lowered-IR helper pruning: typecheck passed; compiler unit `435/0`; bugbash status test passed; bugbash status active failures `0`; test-scope passed; lint passed
- unreferenced half global feature gating fix: fail-first unit reproduced `missing-feature-shader-f16` from unused `__constant__ half` and `__device__ half` globals in a selected-kernel compile with no `shader-f16`; focused unit slice passed; `runtime:unreachable-half-global-feature-binding` `1/0/0`
- changed gate after half global feature gating: typecheck passed; compiler unit `436/0`; fixture tests passed; WebGPU smoke `286/0/0`; test-scope passed
- device-global vector active-lane probe pair: `control:device-global-vector-scalar-atomic-active-lane-return,control:device-global-vector-pointer-array-active-lane-return` is `2/0/0`
- hot device-global vector active-lane probe pair: repeat `5`, warmup `1`, `10/0/0`; best warm `1.4ms` / `1.4ms`, speedups `3.21` / `1.50`
- changed gate after device-global vector active-lane probes: fixture tests passed; WebGPU smoke `288/0/0`; bugbash status active failures `0`; test-scope passed
- dynamic shared vector alias-chain active-lane probe: `control:dynamic-shared-vector-alias-chain-active-lane-return` is `1/0/0`
- hot dynamic shared vector alias-chain active-lane probe: repeat `5`, warmup `1`, `5/0/0`; best warm `4.7ms`, speedup `1.36`
- changed gate after dynamic shared vector alias-chain active-lane probe: fixture tests passed; WebGPU smoke `289/0/0`; bugbash status active failures `0`; test-scope passed
- bugbash status latest-unit proof fix: status now recognizes changed-gate unit-count lines, not only older suite-only wording; regression passed and live status no longer reports stale `421`
- all-inactive device-global side-effect probe: `control:device-global-all-inactive-side-effect-return` is `1/0/0`
- hot all-inactive device-global side-effect probe: repeat `5`, warmup `1`, `5/0/0`; best warm `1.9ms`, speedup `2.00`
- changed gate after all-inactive device-global side-effect probe: fixture tests passed; WebGPU smoke `290/0/0`; bugbash status active failures `0`; test-scope passed
- changed gate after corpus fixture input materializer hardening: fixture tests passed; WebGPU smoke `290/0/0`; skips `0`
- CUDA-120 constant-memory corpus fixture: `corpus:cuda-120:vectorScaleKernel_constant` `1/0/0`; hot repeat `5/0/0`, best warm `1.1ms`, speedup `4.64`
- changed gate after CUDA-120 constant-memory corpus fixture: fixture tests passed; corpus pair `2/0/0`, skips `0`
- full corpus fixture WebGPU gate after CUDA-120 constant-memory fixture: `95/0/0`; expected-output cases `44`; by corpus `cuda-120=3`, `cuda-samples=18`, `llm.c=28`, `leetcuda=46`
- CUDA-120 memory-pool style corpus fixture: `corpus:cuda-120:useMemoryPoolKernel` `1/0/0`; hot repeat `5/0/0`, best warm `3.3ms`, speedup `2.58`
- full corpus fixture WebGPU gate after CUDA-120 memory-pool fixture: `96/0/0`; expected-output cases `45`; by corpus `cuda-120=4`, `cuda-samples=18`, `llm.c=28`, `leetcuda=46`
- CUDA-120 texture-to-surface corpus fixture: `corpus:cuda-120:imageConvolutionKernel_surface` `1/0/0`; hot repeat `5/0/0`, best warm `4.1ms`, speedup `1.15`
- full corpus fixture WebGPU gate after CUDA-120 texture-to-surface fixture: `97/0/0`; expected-output cases `46`; by corpus `cuda-120=5`, `cuda-samples=18`, `llm.c=28`, `leetcuda=46`
- cuda-samples constant-array texture corpus fixture: `corpus:cuda-samples:convolutionRowsKernel_texture` `1/0/0`; hot repeat `5/0/0`, best warm `2.0ms`, speedup `2.80`
- full corpus fixture WebGPU gate after cuda-samples constant-array texture fixture: `98/0/0`; expected-output cases `47`; by corpus `cuda-120=5`, `cuda-samples=19`, `llm.c=28`, `leetcuda=46`
- scoped cuda-samples real-world verifier after corpus-filter tooling fix: audit `357/357` compile/codegen, hard fails `0`; source WebGPU corpus fixture slice `19/0/0`, expected-output `14`, skips `0`; `--auto-corpus-smoke-limit 0` kept auto-corpus at `0`
- scoped LeetCUDA real-world verifier after exact fixture-name corpus filters: audit `293/293` compile/codegen, hard fails `0`; source WebGPU corpus fixture slice `46/0/0`, expected-output `19`, skips `0`
- scoped LeetCUDA real-world verifier after corpus-restricted auto smoke fix: audit `293/293` compile/codegen, hard fails `0`; source WebGPU scoped slice `54/0/0`, with fixture `46/0/0`, auto-corpus `8/0/0`, skips `0`
- exact llm.c auto-corpus trimul profile after reference-size tuning: repeat `3`, warmup `1`, `3/0/0`, skips `0`; expected auto coverage now reports `1`; best warm `113.1ms`; profile `webgpuMs 2.6ms`, `referenceMs 108.2ms`
- scoped llm.c real-world verifier after auto-corpus reference-size tuning: audit `148/148` compile/codegen, hard fails `0`; source WebGPU scoped slice `44/0/0`, with fixture `28/0/0`, auto-corpus `16/0/0`, skips `0`; trimul slow case reduced from about `1.86s` earlier to `170.1ms` in the full scoped verifier
- pinned-output corpus fixture hardening: LeetCUDA fixture slice `46/0/0`, expected-output `35`, skips `0`; llm.c fixture slice `28/0/0`, expected-output `17`, skips `0`; full corpus fixture gate `98/0/0`, expected-output `71`, skips `0`
- second pinned-output corpus fixture hardening: focused new-pinned slice `11/0/0`, expected-output `11`, skips `0`; full corpus fixture gate `98/0/0`, expected-output `82`, skips `0`
- third pinned-output corpus fixture hardening: focused new-pinned slice `11/0/0`, expected-output `11`, skips `0`; full corpus fixture gate `98/0/0`, expected-output `93`, skips `0`
- final pinned-output corpus fixture hardening: focused remaining slice `5/0/0`, expected-output `5`, skips `0`; full corpus fixture gate `98/0/0`, expected-output `98`, skips `0`; fixture tooling test passed
- source/dist real-world verifier after final oracle pinning: compile/codegen audit hard fails `0`; source browser gate `453/0/0`, corpus fixtures `98/0/0`, expected-output `98`, auto-corpus `32/0/0`, skips `0`; dist browser gate `453/0/0`, same coverage, skips `0`
- corpus hot perf gate after final oracle pinning: histogram/scalar repeat `2`, `4/0/0`, expected-output `4`, skips `0`; warm best `5.8ms` / `4.3ms`, speedup `261.97` / `296.28`
- constant texture-to-surface active-lane probe: `texture-surface:constant-active-lane-return` is `1 passed / 0 failed / 0 skipped`; WebGPU smoke after constant texture-to-surface probe `291/0/0`, skips `0`; hot repeat `5/0/0`, best warm `4.4ms`, speedup `19.39`; test-scope passed
- shared/constant texture-to-surface active-lane probe: `texture-surface:shared-constant-active-lane-return` is `1 passed / 0 failed / 0 skipped`; WebGPU smoke after shared/constant texture-to-surface probe `292/0/0`, skips `0`; hot repeat `5/0/0`, best warm `4.8ms`, speedup `31.81`
- surface negative byte-offset boundary fix: `surface:negative-byte-offset` is `1 passed / 0 failed / 0 skipped`; hot repeat `5/0/0`, best warm `1.2ms`, speedup `20.33`; changed gate typecheck passed, compiler unit `436/0`, surface/texture focused group `158/0/0`, WebGPU smoke `293/0/0`, skips `0`

## Bugs Found During Current Run

| Status | Area | Symptom | Root Fix | Proof |
| --- | --- | --- | --- | --- |
| Fixed | surface negative byte offset | `surf2Dread(..., -1, ...)` and `surf2Dwrite(..., -1, ...)` divided byte offset before bounds check, so `-1 / 4` became lane `0` and could read/write the first surface cell | CPU reference and WGSL surface helpers now reject negative byte offsets before converting bytes to element indexes; added pinned `surface:negative-byte-offset` regression to smoke | focused case `1/0/0`, skips `0`; hot repeat `5/0/0`, best warm `1.2ms`, speedup `20.33`; changed gate smoke `293/0/0` |
| Probed green | shared plus constant texture-to-surface active-lane side effects | previous probe combined constants with texture/surface active-lane lowering, but not workgroup vector state crossing the same barrier/return path | added `texture-surface:shared-constant-active-lane-return`, combining `__shared__ float4`, `__constant__` coefficients, helper texture vector read, pre-return surface vector write, post-barrier surface read, and pinned output | focused case `1/0/0`, skips `0`; hot repeat `5/0/0`, best warm `4.8ms`, speedup `31.81` |
| Probed green | constant texture-to-surface active-lane side effects | real WebGPU smoke covered constants, texture reads, surface writes, vector side effects, and active-lane returns separately, but not a corpus-shaped path combining `__constant__` coefficients, helper texture vector read, pre-return surface vector write, post-barrier surface read, and stable pinned output | added `texture-surface:constant-active-lane-return` to e2e and smoke case set | focused case `1/0/0`, skips `0`; hot repeat `5/0/0`, best warm `4.4ms`, speedup `19.39`; test-scope passed |
| Fixed | final real corpus pinned outputs | last 5 corpus fixtures still used dynamic generated CPU-reference comparison, and one `-Infinity` oracle showed fixture data passed through `JSON.stringify` would collapse non-finite numbers to `null` in browser e2e specs | pinned SobelTex, RoPE, packed RoPE, and llm.c attention query/key outputs; added fixture JS-literal serialization for `NaN`/`Infinity`/`-Infinity`; diff reports now include non-finite labels; raised expected-output baseline to `98` | focused remaining slice `5/0/0`, expected-output `5`, skips `0`; full corpus fixture gate `98/0/0`, expected-output `98`, skips `0`; `test:webgpu-fixtures` passed |
| Probed green | real corpus pinned outputs | 24 real WebGPU corpus fixtures were executing but only compared against generated CPU reference output, so fixture regressions could still move both dynamic paths together without a stable pinned oracle | pinned expected outputs for deterministic LeetCUDA activations/vector kernels, llm.c residual/permute/softmax/matmul/layernorm cases, and CUTE transpose fixtures; raised expected-output baseline to `71` | LeetCUDA slice `46/0/0`, expected-output `35`; llm.c slice `28/0/0`, expected-output `17`; full corpus fixture gate `98/0/0`, expected-output `71`; all skips `0` |
| Probed green | additional real corpus pinned outputs | 11 more simple deterministic fixtures still used only generated reference output: LeetCUDA nsight/interview kernels, cuda-samples `MatrixMulCUDA`, llm.c vectorized GELU and layernorm | pinned stable expected outputs and raised expected-output baseline to `82` | focused slice `11/0/0`, expected-output `11`; full corpus fixture gate `98/0/0`, expected-output `82`; all skips `0` |
| Probed green | mdspan/attention/optimizer pinned outputs | 11 more source-derived fixtures still used only generated reference output: cuda-samples Bezier/mdspan, llm.c AdamW/encoder-backward/attention value-path permutations | pinned stable expected outputs and raised expected-output baseline to `93` | focused slice `11/0/0`, expected-output `11`; full corpus fixture gate `98/0/0`, expected-output `93`; all skips `0` |
| Fixed | exact auto-corpus rerun baseline | profiling one exact auto-corpus case with `--auto-corpus-smoke-corpora llm.c` passed the selected case but still exited with `Auto corpus smoke baseline failed: 3/16 covered` because scoped-corpus coverage validation treated exact filters like a full corpus smoke | full auto-corpus coverage is enforced for unfiltered and broad `auto-corpus:<id>:` runs; exact case filters validate the selected case and report the filtered expected count | fail-first exact trimul run reproduced the false baseline failure; exact profile now exits `0` with `3/0/0`, skips `0`, and `autoCorpusSmokeExpectedCovered: 1` |
| Perf | auto-corpus synthetic reference loops | llm.c `trimul_global` WebGPU work was only about `5ms`, but CPU reference comparison took about `1.4s`, making focused iteration look like a WebGPU performance problem when it was synthetic reference size | synthetic smoke dimensions now keep sequence/channel/head/batch loops small for `T`, `C`, `channels`, `HS`, `NH`, and `B` in both node and browser input generators | synthetic-input unit passed; exact trimul profile now repeat `3/0/0`, skips `0`, best warm `113.1ms`; scoped llm.c verifier `44/0/0`, skips `0` |
| Fixed | scoped auto-corpus WebGPU coverage | scoped verifier requested `--auto-corpus-smoke-limit 8` for `leetcuda`, but generated auto-smoke fixtures round-robin from all corpora; case filters then removed every auto case, so the gate reported `autoCorpusSmokeExpectedCovered: 8` with `autoCorpusSmokeCovered: 0` and still passed | auto-smoke generation can now restrict candidate corpora, scoped verifier passes selected corpus ids, cache keys include corpus scope, and scoped auto-smoke coverage is validated even during filtered runs | fail-first scoped LeetCUDA verifier showed auto-corpus `0/8`; fixture and verifier CLI tests; scoped LeetCUDA verifier now `54/0/0` with auto-corpus `8/0/0`, skips `0` |
| Fixed | scoped verifier LeetCUDA fixture selection | scoped `--corpus leetcuda` generated `corpus:leetcuda:` filters, but explicit LeetCUDA fixture case names use `corpus:LeetCUDA:...`; browser phase failed with no matching cases despite 46 pinned fixtures | scoped verifier now derives exact corpus fixture case names from the registry and appends auto-corpus filters separately, avoiding case-label drift across corpora | fail-first scoped LeetCUDA verifier; CLI regression; scoped LeetCUDA verifier audit `293/293`, source WebGPU fixture slice `46/0/0`, skips `0` |
| Fixed | real-world verifier corpus scoping | `verify:real-world-cuda` could not run one corpus, so bugbash had to pay full-corpus feedback cost; wrapper also rejected `--forbid-skips` even though downstream always enforces it | wrapper now accepts `--only`/`--corpus`, validates corpus ids, passes audit `--only`, scopes browser fixtures with `--cases`, and treats wrapper-level `--forbid-skips` as accepted no-op | CLI unit; scoped cuda-samples verifier audit `357/357`, hard fails `0`; source WebGPU slice `19/0/0`, skips `0` |
| Probed green | real corpus constant array plus texture helper | corpus execution coverage did not include a real source combining `__constant__` array input, helper texture reads, and texture object input | added cuda-samples `convolutionRowsKernel` fixture from `convolutionTexture.cu` with `c_Kernel` identity constant array and texture input; raised corpus fixture baseline to lock it in | `corpus:cuda-samples:convolutionRowsKernel_texture` `1/0/0`; full corpus fixture gate `98/0/0`; hot repeat `5/0/0`, best warm `2.0ms` |
| Probed green | real corpus texture-to-surface write | corpus execution coverage had one texture fixture but no real corpus fixture combining `tex2D` reads with `surf2Dwrite` surface output | added CUDA-120 `imageConvolutionKernel` fixture from `day-28-Progress-Checkpoint.md`; validates texture input, surface output, and expected surface readback in real WebGPU; raised corpus fixture baseline to lock it in | `corpus:cuda-120:imageConvolutionKernel_surface` `1/0/0`; full corpus fixture gate `97/0/0`; hot repeat `5/0/0`, best warm `4.1ms` |
| Probed green | real corpus atomic pool allocation | corpus execution coverage did not include a real source using device-side atomic offset allocation into a pool-like pointer plus explicit offset readback | added CUDA-120 `useMemoryPoolKernel` fixture from `day-84-Progress-Checkpoint.md`; validates pool writes and `offset` readback through real WebGPU; raised corpus fixture baseline to lock it in | `corpus:cuda-120:useMemoryPoolKernel` `1/0/0`; full corpus fixture gate `96/0/0`; hot repeat `5/0/0`, best warm `3.3ms` |
| Probed green | real corpus constant memory | corpus execution coverage had buffer/scalar and one texture fixture, but no real corpus fixture proving scalar `__constant__` input materialization through WebGPU | added CUDA-120 `vectorScaleKernel` fixture from `day-26-Constant-Memory.md` with `constants.scaleFactor` and expected output; raised corpus fixture baseline to lock it in | `corpus:cuda-120:vectorScaleKernel_constant` `1/0/0`; changed gate corpus pair `2/0/0`; full corpus fixture gate `95/0/0`; hot repeat `5/0/0`, best warm `1.1ms` |
| Fixed | corpus WebGPU fixture harness | corpus execution fixture materialization only preserved buffers, scalars, and textures, so future corpus fixtures using constants, device globals, surfaces, memory pools, or explicit readback could silently run with incomplete inputs | shared fixture materializer now covers every supported WebGPU input family and the in-browser e2e materializer mirrors that behavior; fixture unit locks all families | `test:webgpu-fixtures` passed; changed WebGPU smoke `290/0/0`, skips `0` |
| Probed green | all-inactive device-global side effects | active-lane coverage had all-inactive guarded RHS and partial device-global side effects, but not every lane returning before a later barrier while pre-return device-global atomics were verified through direct global readback | existing active-lane lowering preserves device-global side effects before all lanes return and suppresses post-barrier active-lane work when no lanes remain active | `control:device-global-all-inactive-side-effect-return` `1/0/0`; hot repeat `5/0/0`, best warm `1.9ms` |
| Fixed | bugbash progress visibility | `bugbash:status` reported stale `Latest unit proof: compiler unit suite after signed byte-storage atomic fix: 421 passed` even though newer changed gates recorded `compiler unit 436/0`, making progress look older than reality | status parser now matches any `compiler unit` gate line, and regression locks that a newer changed gate supersedes an older suite-only line | `test:bugbash-status` passed; live `bugbash:status` now shows latest unit proof from half-global changed gate with `compiler unit 436/0` |
| Probed green | dynamic shared vector alias-chain active-lane side effects | dynamic shared vector pointer arrays had alias-chain coverage, but not inactive lanes writing vector lanes before an early return while active lanes read through shifted pointer arrays after a later barrier | existing active-lane lowering preserves pre-return dynamic shared writes and post-barrier vector pointer-array address scaling through chained aliases | `control:dynamic-shared-vector-alias-chain-active-lane-return` `1/0/0`; hot repeat `5/0/0`, best warm `4.7ms` |
| Probed green | device-global vector active-lane side effects | device-global vector lanes had scalar atomic and pointer-array coverage, but not inactive lanes writing/atomically updating device globals before an early return and active lanes reading them after a later barrier | existing active-lane lowering preserves pre-return device-global side effects, scalar atomic lane writes, and vector pointer-array reads across the barrier | `control:device-global-vector-scalar-atomic-active-lane-return,control:device-global-vector-pointer-array-active-lane-return` `2/0/0`; hot repeat `10/0/0`, best warm `1.4ms` / `1.4ms` |
| Fixed | unreferenced global half feature gating | selected-kernel compilation could fail with `missing-feature-shader-f16` solely because unused global `__constant__ half` / `__device__ half` declarations were declared before selected-kernel reachability filtering removed their IR/WGSL bindings | analyzer now computes global symbol reachability from the selected kernel plus reachable device functions, routes unreachable global feature requirements and initializer feature checks into a dead sink, and keeps f64 compatibility diagnostics tied to the same reachability bit | fail-first unit; focused unit slice passed; `runtime:unreachable-half-global-feature-binding` `1/0/0`; changed gate compiler unit `436/0`; WebGPU smoke `286/0/0` |
| Fixed | unreachable helper lowered-IR pruning | selected-kernel lowering kept unused device helper bodies in `ir.functions`, so downstream reference/WGSL feature and pointer-helper scans still had to inspect dead helper code even though WGSL emission skipped it | lowered IR now stores only selected-kernel reachable helper bodies; reachability also tracks function-valued vector `cg::reduce` operators so cooperative reductions stay available | fail-first unit; focused unit slice `451/0`; cooperative-reduce and dynamic-launch focused slices `451/0`; changed gate compiler unit `435/0` |
| Fixed | unreachable helper device-global binding | `__device__` globals referenced only from unreachable helpers still appeared in lowered IR and WGSL, adding dead storage bindings for selected kernels that never used them | lowered IR now keeps only device globals referenced from selected-kernel reachable bodies, matching constant and texture binding reachability | fail-first unit; focused unit slice `450/0`; `runtime:unreachable-device-global-binding` `1/0/0`; WebGPU smoke `285/0/0`; scoped gate passed |
| Fixed | unreachable helper constant binding | `__constant__` arrays referenced only from unreachable helpers still appeared in lowered IR and forced missing CPU-reference constant input for selected kernels that never used them | lowered IR now keeps only constants referenced from selected-kernel reachable bodies, matching texture binding reachability | fail-first unit; focused unit slice `449/0`; `runtime:unreachable-constant-binding` `1/0/0`; WebGPU smoke `284/0/0`; lint passed |
| Fixed | unreachable helper texture/surface diagnostics and texture binding | unused device helpers containing unsupported texture/surface calls failed selected-kernel compilation; after suppressing those diagnostics, an unused global texture still forced missing CPU-reference texture input | texture/surface compatibility diagnostics now honor selected-kernel reachability, and lowered IR keeps only textures referenced from selected-kernel reachable bodies | fail-first unit; focused unit slice `448/0`; `runtime:unreachable-texture-surface-compat-diagnostics` `1/0/0`; WebGPU smoke `283/0/0`; lint passed |
| Fixed | unreachable helper f64 and inline asm diagnostics | unused device helpers containing `double` params/locals/returns or unsupported inline PTX failed selected-kernel compilation with `unsupported-f64` / `unsupported-inline-asm`, even though selected kernels never called those helpers and no WGSL emitted them | f64 declaration/body checks and inline-asm compatibility diagnostics now honor selected-kernel reachability while still walking operands for ordinary lvalue/scalar validation | fail-first unit; focused unit slice `447/0`; `runtime:unreachable-feature-asm-compat-diagnostics` `1/0/0`; WebGPU smoke `282/0/0`; lint passed |
| Fixed | unreachable helper cooperative grid-sync diagnostics | unused device helpers containing `grid.sync()` or `cg::sync(grid)` failed selected-kernel compilation with `unsupported-cooperative-groups`, even though selected kernels never called those helpers and no WGSL emitted them | cooperative grid-sync compatibility diagnostics now honor selected-kernel reachability while keeping arity/group validation intact | fail-first unit; focused unit slice `446/0`; `runtime:unreachable-grid-sync-compat-diagnostics` `1/0/0`; WebGPU smoke `281/0/0`; lint passed |
| Fixed | unreachable helper CUDA runtime diagnostics | unused device helpers containing `cudaMemcpy` / `cudaDeviceSynchronize` failed selected-kernel compilation with `unsupported-cuda-runtime`, even though selected kernels never called those helpers and no WGSL emitted them | runtime/copy compatibility diagnostics now honor selected-kernel reachability while still walking arguments for ordinary validation | fail-first unit; focused unit slice `445/0`; `runtime:unreachable-runtime-compat-diagnostics` `1/0/0`; WebGPU smoke `280/0/0`; lint passed |
| Fixed | unreachable helper compatibility diagnostics | unused device helpers with unresolved `extern __shared__` storage or divergent return before `__syncthreads()` failed selected-kernel compile/WebGPU diagnostics even when the selected kernel never called those helpers | unreachable dynamic shared declarations now get a non-emitting placeholder while walking unreachable statements, and barrier-divergence validation runs only for selected-kernel reachable functions | fail-first unit; focused unit slice `444/0`; `runtime:unreachable-helper-compat-diagnostics` `1/0/0`; WebGPU smoke `279/0/0`; lint passed |
| Fixed | unreachable helper dynamic launches | unused device helpers containing `child<<<...>>>` failed compilation for selected kernels that never call those helpers, even though no emitted WGSL or runtime path used the launch | dynamic-launch compatibility diagnostics now only attach while walking selected-kernel reachable statements; unreachable helpers still validate ordinary expressions without poisoning selected-kernel compile | fail-first unit; focused unit slice `442/0`; `runtime:unreachable-dynamic-launch` `1/0/0`; WebGPU smoke `278/0/0`; lint passed |
| Perf | WebGPU smoke wrapper overhead | default smoke always enabled per-case profiling and duplicate timeout args, making routine scoped gates noisier and doing extra profile work when only pass/fail feedback was needed | smoke wrapper now builds args through a tested helper, defaults profiling off, keeps opt-in profiling through `CUDA_LITE_WEBGPU_SMOKE_PROFILE`, and emits one timeout arg | test-scope passed; WebGPU smoke `277/0/0` |
| Probed green | unreachable helper atomic real WebGPU fixture | unit coverage proved IR/WGSL strings, but real WebGPU smoke did not exercise selected-kernel reachable atomic pollution directly | WebGPU e2e fixture now supports `wgslContains` / `wgslNotContains`, and `atomic:unreachable-helper-plain-storage` locks plain storage/workgroup declarations plus runtime output | focused fixture `1/0/0`; WebGPU smoke `277/0/0`; fixture and test-scope harness tests passed |
| Fixed | unreachable helper atomic lowering | unused device helpers with direct atomics on device globals or helper-local `__shared__` storage could mark the selected kernel's IR/WGSL storage as atomic even when the selected kernel only performed plain reads/writes | validation still walks unreachable helpers for diagnostics, but unreachable helper atomic side effects now write to scratch sinks; exact pointer-atomic propagation and shared-name discovery use selected-kernel reachable helpers only | fail-first unit; focused atomic unit slice `441/0`; `verify:changed` typecheck + compiler unit `425/0`; lint passed |
| Fixed | unreachable helper shared declarations | unused device helpers with local `__shared__` declarations still contributed `ir.sharedDeclarations`, emitting extra WGSL `var<workgroup>` storage for kernels that never call the helper | shared declaration collection now uses the same selected-kernel reachable helper set as feature gating | fail-first unit; focused unit `439/0` |
| Fixed | unreachable half helper feature gating | compiling selected `bf162` active-lane kernel failed with `missing-feature-shader-f16` because an unused `half2` helper in the same CUDA source was scanned for required features | analyzer now tracks device-function reachability from the selected kernel and routes feature requirements from unreachable helper declarations/bodies into a dead feature sink while keeping diagnostics and overload metadata intact | focused unit `438/0`; `control:active-lane-shared-byte-half2-return-barrier,control:active-lane-shared-byte-bf162-return-barrier` `2/0/0` |
| Probed green | active-lane shared byte vector helper pointer views | packed shared-byte `half2`/`bf162` helper pointer coverage did not include divergent return before a later barrier, where inactive lanes write before return and active lanes read after barrier | existing active-lane lowering plus packed shared-byte vector pointer helpers preserve pre-return lane writes and later active-lane reads through `__shared__ uchar[]` | `control:active-lane-shared-byte-half2-return-barrier,control:active-lane-shared-byte-bf162-return-barrier` `2/0/0` |
| Probed green | shared byte vector helper pointer views | direct `half2`/`bf162` reinterpret over `__shared__ uchar[]` was green, but helper pointer read/write coverage was missing and could regress pointer helper storage compatibility for 16-bit vector carriers | existing packed shared-byte storage helpers preserve `half2` and `__nv_bfloat162` reads/writes through device helper pointer params | `storage:shared-byte-half2-reinterpret,storage:shared-byte-bf162-reinterpret` `2/0/0` |
| Probed green | shared signed atomics over byte-backed storage | storage signed byte atomics were fixed, but shared `uchar` byte roots needed real WebGPU coverage for helper signed add/sub/min/max/and/or/xor/CAS and direct CAS/and/or/xor/exchange at byte offset 4 | existing signed-u32 carrier helpers and shared byte-root address lowering preserve old/new signed values over packed workgroup storage | `storage:shared-byte-int-helper-atomic` `1/0/0`; earlier mixed guard group `3/0/0` |
| Fixed | int-view atomic helper over-emission | any unsigned atomic carrier plus an atomic call emitted the whole signed-u32 helper set, adding unused WGSL helper functions to kernels that needed only one or two signed view operations | int-view helper emission now derives used atomic kinds from IR calls and emits only those helpers for storage/workgroup address spaces | typecheck passed; focused WebGPU group `3/0/0` |
| Fixed | signed atomics over byte-backed storage | `int*` views over `uchar*` storage marked the root atomic but direct atomics emitted `atomicAdd(&scratch[...], -3)` against `atomic<u32>`, while helper atomics generated default-only `bg_ptr_atomicAdd_i32`; byte offset `&scratch[4]` also needed word addressing | signed atomic views over unsigned carriers now route through bitcast/CAS helpers for add/sub/min/max/bitwise/exchange/CAS, and direct/helper byte-root atomic addresses use `index >> 2` | fail-first ad hoc probes; focused unit `supports signed helper atomics over byte-backed storage views`; real WebGPU fixture `storage:param-byte-int-helper-atomic` plus uint guard `2/0/0` |
| Fixed | helper atomics over byte-backed storage | `add_word((uint *)&scratch[0], out)` compiled but generated `bg_ptr_atomicAdd_u32` with no `scratch` case, so helper atomics through byte-backed typed views returned the default and skipped the update | pointer atomic helpers now use the same byte-root compatibility/address policy as packed storage helpers, emitting `scratch[index >> 2]` for aligned typed atomic views over `uchar` storage/shared/global roots | fail-first ad hoc probe; focused unit `supports helper pointer atomics over byte-backed storage views`; real WebGPU fixture `storage:param-byte-uint-helper-atomic` `1/0/0` |
| Fixed | atomic packed-byte storage reads | After marking `uchar* scratch` atomic, normal `word[0]` reads through the helper path tried to shift `scratch[...]` where the element was `atomic<u32>`, causing WGSL pipeline creation failure | packed-byte storage read/write lowering now takes atomic root state, uses `atomicLoad` for packed reads, `atomicStore` for aligned typed writes, and atomic byte clear/or updates for sub-word writes | first real WebGPU run failed with `operator >> (atomic<u32>, u32)`; after fix focused fixture `1/0/0`; related storage helper group `3/0/0`; related atomic helper units `7/0` |
| Fixed | unaligned typed byte-storage overlays | `(float *)&scratch[1]` over `uchar*` storage compiled and self-read back, but raw byte placement was wrong because reference aliases scaled the base offset by element size and WGSL packed-byte helpers treated non-`uchar` views as whole-word aligned | byte-packed storage/shared reads now assemble scalar bits byte-by-byte for unaligned addresses, keep aligned word fast paths, writes update individual bytes only when needed, and reference aliases keep byte base offsets while only scaling pointer indexing by element byte size | fail-first ad hoc probe exposed wrong raw bytes; focused unit `preserves unaligned typed pointer views over byte storage`; real WebGPU fixture `storage:param-byte-unaligned-float-helper-reinterpret` `1/0/0` |
| Fixed | WMMA shared-memory alias shadowing | cuda-samples `compute_gemm` and `compute_gemm_imma` failed compile/codegen with `unsupported-wmma-pointer-operand` when same-named `tile_ptr` declarations mixed a storage/shared alias with a later pointer handle | pointer alias collection now understands structured shared/global/constant array roots and skips aliases by declaration span instead of by name, so an unrelated later handle no longer hides an earlier same-name alias | fail-first minimal reproduced the WMMA alias miss; focused unit `keeps same-named WMMA pointer aliases when only one declaration needs a handle`; cuda-samples audit `357/357`, hard fails `0` |
| Fixed | packed shared-byte bf16 views | `(__nv_bfloat16 *)&scratch[0]` over `__shared__ uchar scratch[]` was rejected as `unsupported-local-pointer`; packed shared-byte carrier also treated `bf16` like 32-bit float instead of 16-bit bfloat lanes | analyzer now allows `bf16` word-addressable explicit pointer aliases; packed shared-byte reads/writes store/read bfloat16 top-half f32 bits in 16-bit lanes; pointer helpers include `bf16` over `uchar` shared storage; CPU reference mirrors lane packing | fail-first real WebGPU case failed at compile; after fix `storage:shared-byte-bf16-reinterpret` `1/0/0`; compiler unit `412/0` |
| Fixed | scoped local pointer handles | two C block scopes reused the pointer variable name `value`; generated WGSL declared `value_buffer`/`value_base` twice, causing pipeline creation failure before the storage bug could run | local pointer handle declarations are resolved by source span and emitted with span-scoped backing names; device-pointer helper paths use the same scoped handle name lookup | fail-first `storage:shared-byte-half-reinterpret` failed with redeclared `value_buffer`/`value_base`; compiler unit regression `keeps same-named local pointer handles distinct across C block scopes`; full compiler unit `411/0` |
| Fixed | packed shared-byte half views | `(half *)&scratch[0]` over `__shared__ uchar scratch[]` wrote/read through `bg_ptr_write_f16/read_f16` with no shared-byte helper case, producing `0` instead of `[1, 2]`; direct packed byte logic also lacked 16-bit lane masking | packed byte shared reads/writes now handle `half` via `unpack2x16float`/`pack2x16float` lane selection; CPU reference mirrors 16-bit lane updates; pointer helpers include `half` over `uchar` shared storage | fail-first real WebGPU case failed compare; after fix `storage:shared-byte-half-reinterpret` `1/0/0`; focused unit `packs half pointer views over shared byte storage into 16-bit lanes`; full compiler unit `411/0` |
| Fixed | packed shared-byte float views | `(float *)&scratch[0]` over `__shared__ uchar scratch[]` emitted `bg_ptr_read_f32` returning raw `u32`, so WGSL pipeline creation failed with returned `u32`, expected `f32`; reference also treated float views as raw integer bits | packed byte scalar carrier conversion now bitcasts `float`/`double`/`bf16` from/to `u32`, keeps `int` bitcast, preserves `uint`, and handles bool carriers | fail-first real WebGPU case failed at pipeline creation; after fix `storage:shared-byte-float-reinterpret` `1/0/0`; focused unit `425/0`; compiler unit `409/0`; storage group `38/0/0`; smoke `269/0/0` |
| Fixed | concurrent shared-byte packed writes | `scratch[threadIdx.x] = uchar(threadIdx.x + 1)` over `__shared__ uchar scratch[4]` lowered each byte write as non-atomic read/modify/write of the same packed `u32`; real WebGPU produced `0x04000000` instead of `0x04030201` | packed `uchar` shared arrays now use `array<atomic<u32>>`; byte reads use `atomicLoad`; byte writes use `atomicAnd` to clear the byte and `atomicOr` to set it; typed word aliases use `atomicLoad`/`atomicStore`; direct `bytes[i]` reads route through packed shared reads | fail-first real WebGPU case failed at compare; after fix `storage:shared-byte-concurrent-writes` `1/0/0`; focused unit `424/0`; storage group `37/0/0`; smoke `268/0/0` |
| Fixed | shared byte storage reinterpret local pointers | cuda-samples `immaTensorCoreGemm` kernel `compute_gemm_imma` failed audit with `unsupported-local-pointer`, and local `int*` / `int4*` aliases over `uchar` storage could index bytes as scalar words or write whole packed slots incorrectly | analyzer allows word-addressable `uchar` pointer aliases; reference runtime packs shared `uchar` into byte-addressed words; WGSL pointer indexing scales typed aliases over byte roots by element byte size; direct shared `uchar` assignments route through packed byte writes | focused byte-reinterpret unit `424/0`; `storage:shared-byte-reinterpret` `1/0/0`; WebGPU smoke `267/0/0`; cuda-samples audit compile/codegen `357/357`, hard fails `0` |
| Fixed | loop-local divergent continue analysis | llm.c `fused_classifier_kernel2` and `fused_classifier_kernel4` were rejected with `divergent-continue-before-barrier` even though the `continue` only targeted an inner lane loop and did not skip the later helper barrier | analyzer now tracks barriers later in the current continue-target loop separately from barriers after enclosing loops/blocks; unsafe same-loop continues still error | focused unit run `423/0`; compiler unit file `407/0`; llm.c audit `148/0/0`; full real-world audit green; WebGPU smoke `266/0/0`; verifier src/dist `419/0/0` |
| Fixed | corpus audit C++ member helper extraction | cuda-samples `multiGpuConjugateGradient` became `missing-kernel` because class `PeerGroup` `__device__` member methods/constructors were collected as free device helpers and poisoned normalization until the requested kernel disappeared | audit device-function collection now filters constructor initializer signatures and `const`/override-style member methods; regression emits the class-adjacent kernel source and verifies member methods are excluded | corpus audit regression passed; cuda-samples compile/codegen `356/357`; hard fails `1`; full real-world audit green |
| Fixed | WebGPU smoke command echo | `e2e:webgpu:smoke` still printed a giant inner `pnpm run e2e:webgpu:case -- --cases ...` command before JSON, even after summary compaction | smoke wrapper now spawns `run-cuda-lite-tool.mjs` directly with the same WebGPU flags, bypassing nested pnpm lifecycle echo while preserving the lock/build-skip behavior | smoke `266/0/0`; output now starts with wrapper command then compact JSON |
| Fixed | WebGPU summary output size | focused/smoke runs with many `--cases` printed hundreds of full case-filter names, making iteration logs noisy and expensive to inspect | summary reports now compact large filter lists into count plus first/last samples while preserving full names for small focused runs | harness unit; 21-case real WebGPU sample `21/0/0` with compact `caseFilters` |
| Fixed | WebGPU case filter exactness | `--cases texture-surface:volume-vector-pointer-array-minmax-active-lane-return` also ran `texture-surface:volume-vector-pointer-array-minmax-active-lane-return-false-branch`, making focused reruns slower and less precise | shared case-name matcher now gives exact filters priority and only falls back to substring matching when no exact candidate exists | fail-first harness unit; exact real WebGPU rerun `1/0/0`; fuzzy fallback `minmax-active-lane-return` `12/0/0` |
| Fixed | divergent continue before later barrier | `if (tid >= N) continue;` before a later `__syncthreads()` had no diagnostic and could emit raw WGSL `continue`, letting some lanes skip the barrier | analyzer now treats thread-dependent `continue` before a later barrier as an error and the compatibility plan classifies it as an unsupported safety blocker until per-iteration active-lane continue lowering exists | fail-first compiler unit now `422/0`; safe canonical continue regression `422/0`; diagnostic `divergent-continue-before-barrier` |
| Fixed | nested divergent break before post-loop barrier | `if (tid >= N) { if (out[tid] >= 0u) break; }` inside a loop followed by `__syncthreads()` emitted a raw nested WGSL `break`, leaving non-uniform control flow before the barrier | active-lane break lowering now detects current-loop breaks nested through blocks/ifs, rewrites nested `break` to the loop active flag, and keeps breaks inside nested loops out of the current-loop rewrite | compiler focused unit `421/0`; real WebGPU break group `5/0/0`; `control:active-lane-nested-break-post-loop-barrier` `1/0/0` |
| Fixed | report-output parent directories | `e2e:webgpu:smoke -- --json .tmp/reports/smoke-profile.json` completed all WebGPU cases, then failed with `ENOENT` because report/progress writers assumed parent dirs already existed | e2e JSON, markdown, and progress writers now create parent dirs; CUDA-lite compiler/WebGPU benchmark report writers use same parent-dir creation behavior | smoke with nested JSON/markdown/progress output `265/0/0`; compiler benchmark nested markdown; WebGPU benchmark nested JSON/markdown |
| Probed green | 2D texture-to-layered-surface active-lane scalar pointer-array true/false branches | 2D `tex2D<uint4>` to `surf2DLayeredwrite/read` scalar pointer-array selection had select-only coverage and volume active-lane coverage, but no 2D active-lane true/false pair; this could regress pre-return layered-surface side effects, selected scalar storage handles, or false-branch shadow writes | existing active-lane lowering, texture vector conversion, layered surface vector read/write, pointer-array storage handle lowering, and scalar atomic lane writes preserve true and false selected targets | `texture-surface:uint4-atomic-pointer-array-active-lane-return,texture-surface:uint4-atomic-pointer-array-active-lane-return-false-branch` `2/0/0`; smoke `265/0/0`; hot gate `10/0/0`, speedups `1.35` / `1.13` |
| Fixed | stale last-failure status | a fail-first expectation miss wrote `.tmp/cuda-lite-last-failures.json`; the later green focused rerun left the artifact in place, so `bugbash:status` falsely reported an active failure | successful WebGPU e2e runs now clear the last-failure artifact after all failure/skip/warmup checks pass | focused green rerun `1/0/0`; `bugbash:status` reports `Active failure cases: 0`; smoke `263/0/0` |
| Probed green | texture-surface active-lane scalar pointer-array false branch | volume texture-to-surf3D scalar pointer-array active-lane case proved true target only; false-target selection could regress pre-return surface writes, selected scalar storage handles, or shadow-buffer atomics | existing active-lane side-effect lowering, volume texture conversion, surf3D vector read/write, pointer-array storage handle lowering, and scalar atomic lane writes preserve false-branch selected targets | `texture-surface:volume-atomic-pointer-array-active-lane-return-false-branch` `1/0/0`; smoke `263/0/0`; hot gate `5/0/0`, speedup `1.24` |
| Probed green | all-inactive guarded RHS | control, surface, atlas texture, and volume texture-to-surface guarded-RHS cases proved partial active lanes only; all lanes returning before a barrier could still evaluate side-effecting RHS helpers, atomics, texture reads, or surface writes if active-lane masks leaked | existing active-lane barrier lowering suppresses RHS side effects when every lane is inactive, including scalar atomics, vector atomic writes, surf3D reads, atlas texture reads, and volume texture-to-surface helpers | `control:active-lane-guarded-rhs-all-inactive,control:active-lane-assignment-guarded-rhs-all-inactive,control:active-lane-vector-atomic-guarded-rhs-all-inactive,control:active-lane-compound-assignment-guarded-rhs-all-inactive,surface:surf3d-active-lane-guarded-rhs-all-inactive,texture:atlas-active-lane-guarded-rhs-all-inactive,texture-surface:volume-active-lane-guarded-rhs-all-inactive` `7/0/0`; smoke `262/0/0`; hot gate `35/0/0`, speedups `1.22` / `1.11` / `1.21` / `1.11` / `1.07` / `1.26` / `1.15` |
| Probed green | compound false branch | surf3D, surf1D, layered surface, atlas texture, and non-atlas texture pointer-array compound active-lane cases only proved true-target selection; false-target branches could choose wrong vector storage base, skip compound member writes, or mis-route shadow-buffer updates | existing surface/texture vector reads, selected vector pointer-array storage handles, compound vector assignment, scalar atomic lane lowering, and active-lane side-effect guards preserve false-branch selected targets | `surface:surf3d-pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch,surface:surf1d-pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch,surface:pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch,texture:atlas-vector-atomic-pointer-array-compound-active-lane-return-false-branch,texture:pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch` `5/0/0`; smoke `255/0/0`; hot gate `25/0/0`, speedups `1.32` / `1.33` / `1.21` / `1.16` / `1.22` |
| Probed green | texture CAS/minmax false branch | atlas and non-atlas texture pointer-array CAS/exchange and min/max active-lane cases only proved true-target selection; false-target branches could choose wrong storage base, drop old-value writes, or mis-route shadow-buffer atomics | existing texture vector reads, selected pointer-array storage handles, CAS/exchange/minmax scalar lane lowering, and active-lane side-effect guards preserve false-branch selected targets | `texture:atlas-vector-atomic-pointer-array-cas-active-lane-return-false-branch,texture:atlas-vector-atomic-pointer-array-minmax-active-lane-return-false-branch,texture:pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch,texture:pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch` `4/0/0`; smoke `250/0/0`; hot gate `20/0/0`, speedups `1.37` / `1.16` / `1.38` / `1.00` |
| Probed green | surface CAS/minmax false branch | layered, surf1D, and surf3D pointer-array CAS/exchange and min/max active-lane cases only proved true-target selection; false-target branches could choose wrong storage base, drop old-value writes, or mis-route shadow-buffer atomics | existing surface vector reads, selected pointer-array storage handles, CAS/exchange/minmax scalar lane lowering, and active-lane side-effect guards preserve false-branch selected targets | `surface:surf3d-pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch,surface:surf3d-pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch,surface:surf1d-pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch,surface:surf1d-pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch,surface:pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch,surface:pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch` `6/0/0`; smoke `246/0/0`; hot gate `30/0/0`, speedups `1.23` / `2.87` / `2.59` / `2.39` / `1.03` / `1.14` |
| Probed green | volume vector-pointer false branch | volume texture-to-surf3D vector pointer-array CAS/exchange, min/max, compound, and atomic helper paths only proved true-target selection; false-target selection could regress vector pointer casts, scalar lane atomics, shadow-buffer writes, or active-lane surface side effects | existing volume texture conversion, surf3D vector read/write, vector pointer-array storage view, scalar-cast pointer arithmetic, CAS/exchange/minmax lowering, and active-lane side-effect guards preserve false-branch selected vector targets | `texture-surface:volume-vector-pointer-array-atomic-active-lane-return-false-branch,texture-surface:volume-vector-pointer-array-cas-active-lane-return-false-branch,texture-surface:volume-vector-pointer-array-compound-active-lane-return-false-branch,texture-surface:volume-vector-pointer-array-minmax-active-lane-return-false-branch` `4/0/0`; smoke `240/0/0`; hot gate `20/0/0`, speedups `1.33` / `2.25` / `1.23` / `1.26` |
| Probed green | surface/atlas pointer-array false branch | surface, surf1D, surf3D, and atlas texture selected pointer-array cases needed false-target coverage; true-target-only tests could miss wrong storage handle/base selection | existing surface/texture read conversion, pointer-array storage handle lowering, scalar atomic lane writes, and shadow-buffer selection preserve false-branch targets | `surface:pointer-alias-atomic-pointer-array-select-false-branch,surface:surf1d-pointer-alias-atomic-pointer-array-select-false-branch,surface:surf3d-pointer-alias-atomic-pointer-array-select-false-branch,texture:atlas-vector-atomic-pointer-array-select-false-branch` `4/0/0`; smoke `236/0/0`; hot gate `20/0/0`, speedups `2.76` / `1.16` / `1.17` / `1.12` |
| Probed green | texture/surface pointer-array false branch | existing pointer-array texture/surface cases mostly selected the true target buffer; false-branch selected storage handles could regress local pointer-array branch lowering, active-lane side effects, texture-to-surface readback, or shadow-buffer atomics | existing texture conversion, surface read/write, pointer-array storage handle lowering, active-lane guard, and scalar atomic lane writes preserve false-branch selected targets | `texture:pointer-alias-atomic-pointer-array-select-false-branch,texture:pointer-alias-atomic-pointer-array-active-lane-return-false-branch,texture-surface:uint4-atomic-pointer-array-select-false-branch,texture-surface:volume-atomic-pointer-array-select-false-branch` `4/0/0`; smoke `232/0/0`; hot gate `20/0/0`, speedups `2.18` / `1.17` / `1.26` / `1.09` |
| Probed green | atlas/volume texture guarded RHS active-lane paths | layered/3D texture reads inside side-effecting RHS helpers, including texture-to-3D-surface writes, could regress active-lane RHS guards, counter side effects, volume vector read conversion, or surf3D readback after a later barrier | existing active-lane guard, texture atlas/volume sampling, vector conversion, counter atomics, and surf3D vector read/write lowering preserve only active-lane side effects and correct readback | `texture:atlas-active-lane-guarded-rhs,texture-surface:volume-active-lane-guarded-rhs` `2/0/0`; smoke `228/0/0`; hot gate `10/0/0`, speedups `1.17` / `1.13` |
| Probed green | surf3D helper multi-surface and guarded RHS active-lane paths | 3D surface vector reads/writes passed through helper calls and surface-read RHS side effects after active-lane returns could regress multi-surface handle routing, z-linearized vector writes, or guarded RHS side-effect masks | existing helper call lowering, 3D surface read/write indexing, active-lane guard, and RHS side-effect predication preserve only active lanes and correct surface handles | `surface:surf3d-helper-vector-multi-surface-active-lane-return,surface:surf3d-active-lane-guarded-rhs` `2/0/0`; smoke `226/0/0`; hot gate `10/0/0`, speedups `1.14` / `1.45` |
| Probed green | surf3D surface pointer-alias active/atomic vector writes | 3D-surface scalar reads feeding float pointer-lane stores, atomic lane stores, atomic readback, and vector compound helper writes could regress z-linearized scalar reads, scalar-cast storage views, atomic lane writes, vector readback, or compound vector assignment | existing 3D surface scalar read, scalar storage view, atomic lane lowering, and vector compound assignment preserve side effects and readback | `surface:surf3d-pointer-alias-active-lane-store,surface:surf3d-pointer-alias-atomic-active-lane-store,surface:surf3d-pointer-alias-atomic-vector-readback,surface:surf3d-pointer-alias-atomic-vector-compound` `4/0/0`; smoke `224/0/0`; hot gate `20/0/0`, speedups `1.16` / `1.24` / `1.23` / `1.21` |
| Probed green | surf3D surface pointer-array select/basic active-lane return | 3D-surface-fed `uint4` selected scalar pointer-array atomic adds, both ordinary select and pre-barrier active-lane early return, could regress z-linearized vector reads, selected buffer handles, scalar-lane atomic writes, or side effects before a later barrier | existing 3D surface vector read, scalar pointer-array atomic lowering, and active-lane guard preserve selected-buffer writes before return | `surface:surf3d-pointer-alias-atomic-pointer-array-select,surface:surf3d-pointer-alias-atomic-pointer-array-active-lane-return` `2/0/0`; smoke `220/0/0`; hot gate `10/0/0`, speedups `1.18` / `1.31` |
| Probed green | surf3D surface pointer-array CAS/exchange active-lane return | 3D-surface-fed `uint4` selected scalar pointer-array CAS/exchange ops before an early `return` could regress z-linearized surface vector reads, compare/exchange old values, selected buffer handles, or active-lane side effects before a later barrier | existing 3D surface vector read, pointer-array CAS/exchange lowering, and active-lane guard preserve old-value side effects before return | `surface:surf3d-pointer-alias-atomic-pointer-array-cas-active-lane-return` `1/0/0`; smoke `218/0/0`; hot gate `5/0/0`, speedup `1.16` |
| Probed green | surf3D surface pointer-array min/max active-lane return | 3D-surface-fed `uint4` selected scalar pointer-array min/max ops before an early `return` could regress z-linearized surface vector reads, unsigned min/max old values, selected buffer handles, or active-lane side effects before a later barrier | existing 3D surface vector read, pointer-array min/max lowering, and active-lane guard preserve old-value side effects before return | `surface:surf3d-pointer-alias-atomic-pointer-array-minmax-active-lane-return` `1/0/0`; smoke `218/0/0`; hot gate `5/0/0`, speedup `1.09` |
| Probed green | surf3D surface pointer-array compound active-lane return | 3D-surface-fed `uint4` selected vector pointer-array compound writes before an early `return` could regress z-linearized surface vector reads, vector pointer selection, compound member writes, atomic scalar lane adds, or active-lane side effects before a later barrier | existing 3D surface vector read, vector pointer-array storage view, compound vector assignment, scalar atomic lane lowering, and active-lane guard preserve side effects before return | `surface:surf3d-pointer-alias-atomic-pointer-array-compound-active-lane-return` `1/0/0`; smoke `216/0/0`; hot gate `5/0/0`, speedup `1.35` |
| Probed green | layered surface pointer-array compound active-lane return | layered-surface-fed `uint4` selected vector pointer-array compound writes before an early `return` could regress layered surface vector reads, vector pointer selection, compound member writes, atomic scalar lane adds, or active-lane side effects before a later barrier | existing layered surface vector read, vector pointer-array storage view, compound vector assignment, scalar atomic lane lowering, and active-lane guard preserve side effects before return | `surface:pointer-alias-atomic-pointer-array-compound-active-lane-return` `1/0/0`; smoke `215/0/0`; hot gate `5/0/0`, speedup `1.19` |
| Probed green | surf1D pointer-array compound active-lane return | surf1D-fed `uint4` selected vector pointer-array compound writes before an early `return` could regress surf1D vector reads, vector pointer selection, compound member writes, atomic scalar lane adds, or active-lane side effects before a later barrier | existing surf1D vector read, vector pointer-array storage view, compound vector assignment, scalar atomic lane lowering, and active-lane guard preserve side effects before return | `surface:surf1d-pointer-alias-atomic-pointer-array-compound-active-lane-return` `1/0/0`; smoke `215/0/0`; hot gate `5/0/0`, speedup `1.19` |
| Probed green | atlas texture pointer-array compound active-lane return | atlas/3D texture-fed `uint4` selected vector pointer-array compound writes before an early `return` could regress atlas sampling, vector pointer selection, compound member writes, atomic scalar lane adds, or active-lane side effects before a later barrier | existing atlas/3D texture vector read, vector pointer-array storage view, compound vector assignment, scalar atomic lane lowering, and active-lane guard preserve side effects before return | `texture:atlas-vector-atomic-pointer-array-compound-active-lane-return` `1/0/0`; smoke `213/0/0`; hot gate `5/0/0`, speedup `1.33` |
| Probed green | texture pointer-array compound active-lane return | texture-fed `uint4` selected vector pointer-array compound writes before an early `return` could regress vector pointer selection, compound member writes, atomic scalar lane adds, or active-lane side effects before a later barrier | existing texture vector read, vector pointer-array storage view, compound vector assignment, scalar atomic lane lowering, and active-lane guard preserve side effects before return | `texture:pointer-alias-atomic-pointer-array-compound-active-lane-return` `1/0/0`; smoke `213/0/0`; hot gate `5/0/0`, speedup `1.15` |
| Probed green | atlas texture pointer-array CAS/exchange active-lane return | atlas/3D texture-fed `uint4` selected scalar pointer-array CAS/exchange ops before an early `return` could regress atlas sampling, compare/exchange old values, selected buffer handles, or active-lane side effects before a later barrier | existing atlas/3D texture vector read, pointer-array CAS/exchange lowering, and active-lane guard preserve old-value side effects before return | `texture:atlas-vector-atomic-pointer-array-cas-active-lane-return` `1/0/0`; smoke `211/0/0`; hot gate `5/0/0`, speedup `1.38` |
| Probed green | atlas texture pointer-array min/max active-lane return | atlas/3D texture-fed `uint4` selected scalar pointer-array min/max ops before an early `return` could regress atlas sampling, unsigned min/max old values, selected buffer handles, or active-lane side effects before a later barrier | existing atlas/3D texture vector read, pointer-array min/max lowering, and active-lane guard preserve old-value side effects before return | `texture:atlas-vector-atomic-pointer-array-minmax-active-lane-return` `1/0/0`; smoke `211/0/0`; hot gate `5/0/0`, speedup `1.11` |
| Probed green | texture pointer-array CAS/exchange active-lane return | texture-fed `uint4` selected scalar pointer-array CAS/exchange ops before an early `return` could regress compare/exchange old values, selected buffer handles, or active-lane side effects before a later barrier | existing texture vector read, pointer-array CAS/exchange lowering, and active-lane guard preserve old-value side effects before return | `texture:pointer-alias-atomic-pointer-array-cas-active-lane-return` `1/0/0`; smoke `209/0/0`; hot gate `5/0/0`, speedup `1.37` |
| Probed green | texture pointer-array min/max active-lane return | texture-fed `uint4` selected scalar pointer-array min/max ops before an early `return` could regress unsigned min/max old values, selected buffer handles, or active-lane side effects before a later barrier | existing texture vector read, pointer-array min/max lowering, and active-lane guard preserve old-value side effects before return | `texture:pointer-alias-atomic-pointer-array-minmax-active-lane-return` `1/0/0`; smoke `209/0/0`; hot gate `5/0/0`, speedup `1.26` |
| Probed green | layered surface pointer-array CAS/exchange active-lane return | layered-surface-fed `uint4` selected scalar pointer-array CAS/exchange ops before an early `return` could regress compare/exchange old values, selected buffer handles, or active-lane side effects before a later barrier | existing layered surface vector read, pointer-array CAS/exchange lowering, and active-lane guard preserve old-value side effects before return | `surface:pointer-alias-atomic-pointer-array-cas-active-lane-return` `1/0/0`; smoke `207/0/0`; hot gate `5/0/0`, speedup `1.28` |
| Probed green | layered surface pointer-array min/max active-lane return | layered-surface-fed `uint4` selected scalar pointer-array min/max ops before an early `return` could regress unsigned min/max old values, selected buffer handles, or active-lane side effects before a later barrier | existing layered surface vector read, pointer-array min/max lowering, and active-lane guard preserve old-value side effects before return | `surface:pointer-alias-atomic-pointer-array-minmax-active-lane-return` `1/0/0`; smoke `207/0/0`; hot gate `5/0/0`, speedup `1.25` |
| Probed green | surf1D pointer-array min/max active-lane return | surf1D-fed `uint4` selected scalar pointer-array min/max ops before an early `return` could regress unsigned min/max return values, selected buffer handles, or active-lane side effects before a later barrier | existing surf1D vector read, pointer-array scalar min/max lowering, and active-lane guard preserve old-value side effects before return | `surface:surf1d-pointer-alias-atomic-pointer-array-minmax-active-lane-return` `1/0/0`; smoke `205/0/0`; hot gate `5/0/0`, speedup `1.30` |
| Probed green | surf1D pointer-array CAS/exchange active-lane return | surf1D-fed `uint4` selected scalar pointer-array CAS/exchange ops before an early `return` could regress compare/exchange return values, selected buffer handles, or active-lane side effects before a later barrier | existing surf1D vector read, pointer-array scalar atomic return lowering, and active-lane guard preserve CAS/exchange side effects and old-value writes before return | `surface:surf1d-pointer-alias-atomic-pointer-array-cas-active-lane-return` `1/0/0`; smoke `204/0/0`; hot gate `5/0/0`, speedup `1.18` |
| Probed green | surf1D pointer-array active-lane return | surf1D-fed `uint4` selected scalar pointer-array atomics before an early `return` could regress flat byte-offset reads or active-lane side effects before a later barrier | existing active-lane guard plus surf1D vector read and pointer-array scalar atomic lowering preserves side effects before return | `surface:surf1d-pointer-alias-atomic-pointer-array-active-lane-return` `1/0/0`; smoke `203/0/0`; hot gate `5/0/0`, speedup `2.07` |
| Probed green | surface pointer-array active-lane return | layered surface-fed `uint4` selected scalar pointer-array atomics before an early `return` could regress active-lane side effects or selected storage handles before a later barrier | existing active-lane guard plus pointer-array scalar atomic lowering preserves layered-surface side effects before return | `surface:pointer-alias-atomic-pointer-array-active-lane-return` `1/0/0`; smoke `202/0/0`; hot gate `5/0/0`, speedup `1.24` |
| Probed green | texture pointer-array active-lane return | texture-fed `uint4` selected scalar pointer-array atomics before an early `return` could regress active-lane side effects or selected pointer-array storage handles before a later barrier | existing active-lane guard plus pointer-array scalar atomic lowering preserves non-atlas texture side effects before return | `texture:pointer-alias-atomic-pointer-array-active-lane-return` `1/0/0`; smoke `201/0/0`; hot gate `5/0/0`, speedup `1.26` |
| Probed green | texture pointer-alias atomic pointer-array select | texture-fed `uint4` atomic lane writes through selected scalar pointer-array slots could regress pointer-array base selection or flat lane writes outside atlas/surface paths | existing storage pointer-array and scalar atomic lane lowering handles non-atlas texture vector reads correctly | `texture:pointer-alias-atomic-pointer-array-select` `1/0/0`; smoke `200/0/0`; hot gate `5/0/0`, speedup `1.40` |
| Fixed | while/do-while divergent break before later barrier | `while` and `do-while` loops with thread-dependent `break` before a later `__syncthreads()` still emitted raw loop breaks even after the `for`-loop fix | active-lane break lowering now detects direct trailing breaks in `for`, `while`, and `do-while` loop bodies before later barriers/subgroup calls | unit guard `keeps post-loop barriers uniform after while and do-while divergent breaks`; WebGPU fixtures `2/0/0`; smoke `199/0/0`; hot gate `10/0/0`, speedups `1.17` / `1.07` |
| Fixed | post-loop divergent break before later barrier | `if (tid >= N) break;` inside a loop followed by `__syncthreads()` emitted raw WGSL `break` and only warned about unguarded storage writes, leaving barrier control flow non-uniform | analyzer now emits `divergent-break-before-barrier`; WGSL sequence lowering turns direct for-loop divergent breaks before later barriers into active-lane flag updates and predicates following statements | unit guard `keeps post-loop barriers uniform after divergent breaks`; `control:active-lane-break-post-loop-barrier` `1/0/0`; smoke `197/0/0`; hot gate `5/0/0`, speedup `3.83` |
| Probed green | volume vector pointer-array min/max before active-lane return | volume texture plus 3D surface read feeds `atomicMin`/`atomicMax` through a selected `uint4*` pointer array, then performs vector member compound writes before a later barrier | existing vector pointer helper min/max and member compound lowering preserves flat scalar lanes and active-lane side effects | `texture-surface:volume-vector-pointer-array-minmax-active-lane-return` `1/0/0`; smoke `196/0/0`; hot gate `5/0/0`, speedup `1.36` |
| Fixed | vector pointer lane offset helpers | non-atomic member writes through pointer helpers skipped the helper path, so `helpers:vector-lane-pointer-offset-helper` wrote the wrong scalar lane | pointer member assignment now routes lexical device pointer params through device pointer helper lowering before direct storage fallback | unit guard `preserves scalar-to-vector pointer alias byte offsets`; focused WebGPU helper case green; smoke `195/0/0` |
| Fixed | device-global vector pointer arrays | `&g_ptr_values[2]` for `__device__ float3[]` stored vector element index `2`, not flat scalar lane index `6`, causing helper-selected pointer arrays to read stale lanes | device-global pointer argument parts now scale vector element bases by lane count before helper dispatch | unit guard `keeps device-global vector pointer-array entries in flat lanes`; smoke `195/0/0` |
| Fixed | shared/local vector flat helper reads and writes | shared/local vector helper flat reads emitted whole vectors into scalar constructors, including invalid WGSL like `vec4<f32>(vec4<f32>, ...)` | storage vector flat read/write helpers now decompose flat lane indexes into vector element plus lane before scalar load/store | unit guard `reads shared vector pointer helpers through scalar lanes`; `storage:cross-space-vector-alias-consistency` green; smoke `195/0/0` |
| Fixed | shared vector pointer array bases | `&values[1]` for typed `__shared__ float3 values[]` used element base `1` instead of flat scalar base `3`, so pointer-array helpers wrote/read the wrong lanes | shared pointer argument parts now scale vector roots by lane count, matching device-global pointer argument semantics | `storage:shared-vector-pointer-array` green; smoke `195/0/0` |
| Fixed | dynamic shared vector pointer arithmetic | dynamic shared vector aliases had a special-case delta that suppressed vector pointer scaling, so `float3* values; &values[1]` addressed the wrong base | removed the dynamic-shared vector special-case; vector pointer arithmetic now consistently scales by lane count | updated dynamic shared vector pointer-array expectations; `storage:dynamic-shared-vector-pointer-array` and alias-chain variants green; smoke `195/0/0` |
| Fixed | lexical pointer param detection | generic storage params were incorrectly treated like device pointer helper params, producing unresolved helper bases such as `x_base` in normal storage vector helpers | device pointer param lookup is now lexical, and pointer helper fast paths only fire for real helper pointer params | `storage:shared-vector-helper` green; compiler unit `418/0`; smoke `195/0/0` |
| Fixed | vector pointer arithmetic atomics | `atomicAdd(reinterpret_cast<uint*>(vecPointer) + lane, value)` through helper-local scalar alias of a vector pointer-array target emitted invalid WGSL `atomicAdd(u32, u32)` | binary pointer arithmetic value-type resolution now follows the left pointer expression, preserving scalar casts over vector pointers before pointer-helper atomic lowering | unit guard `keeps casted vector pointer arithmetic atomics on pointer helpers`; `texture-surface:volume-vector-pointer-array-atomic-active-lane-return` `1/0/0`; smoke `193/0/0` |
| Fixed | surface helper params | `cudaSurfaceObject_t` helper params used name coincidence, not handles | surface dispatch helpers by handle | src/dist verifier `249/0/0` earlier |
| Fixed | vector surface write | `surf2Dwrite(float4, ...)` emitted invalid `...; 0;` WGSL | vector surface writes emit pure statements | `texture-surface:vector-helper-roundtrip` |
| Fixed | vector surface read | `surf2Dread(&float4, ...)` splatted lane 0 | lane-wise reads at byte offsets | `surface:vector-read` |
| Fixed | templated vector surface read | `surf2Dread<float4>` return cast as `f32(vec4)` | expression type inference honors template | `surface:helper-vector-read-multiple-surfaces` |
| Fixed | layered surface write | `surf2DLayeredwrite` wrote layer into Y, not Z | use Y as row, layer as Z | `surface:layered-write` |
| Fixed | layered/3D surface read/reference z | `surf2DLayeredread` and `surf3Dread` were unsupported, and reference layered writes flattened layer into Y instead of z-linearized storage | add layered/3D surface read analyzer/reference/WGSL lowering and share z-linearized surface read/write indexing | `surface:layered-read,surface:surf3d-read` `2/0/0`, compiler unit `395/0`, smoke `132/0/0` |
| Fixed | layered/3D vector surface read type inference | `surf3Dread<float4>` through helper returned `vec4<f32>` but WGSL value-type inference treated non-2D surface reads as scalar, emitting `f32(vec4<f32>)`; reference runtime also read only one lane for vector surface reads | surface read value inference now covers `surf2DLayeredread`/`surf3Dread`; reference reads vector lanes from z-linearized storage | `surface:layered-vector-read,surface:surf3d-vector-read` `2/0/0`, compiler unit `396/0`, smoke `134/0/0` |
| Probed green | multi-surface helper vector write before active-lane return | helper `cudaSurfaceObject_t` params across two surfaces plus vector read/write before return could reuse the wrong handle, drop lane-wise writes, or mis-order inactive-lane side effects before a later barrier | existing surface handle dispatch and active-lane lowering preserve vector helper read/write semantics across multiple surfaces | `surface:helper-vector-multi-surface-active-lane-return` `1/0/0`, smoke `181/0/0` |
| Probed green | 3D surface vector write before active-lane return | `surf3Dwrite(float4)` before return-and-barrier lowering could drop inactive-lane vector side effects, mis-scale z-linearized lanes, or fail later `surf3Dread<float4>` readback | existing active-lane lowering preserves 3D vector surface writes before lane deactivation and z-linearized vector readback | `surface:surf3d-vector-write-active-lane-return` `1/0/0`, smoke `180/0/0` |
| Probed green | surface vector read before active-lane return | layered/3D vector surface reads before deactivating a lane could be dropped, scalarized, or hidden behind non-uniform barrier lowering | existing active-lane lowering now preserves vector surface reads before lane deactivation and keeps later barriers uniform | `surface:vector-read-active-lane-return` `1/0/0`, smoke `135/0/0` |
| Probed green | surface vector write before active-lane return | vector `surf2DLayeredwrite` before deactivating a lane could drop lane-wise writes or fail barrier-uniform lowering | existing active-lane lowering preserves lane-wise vector surface writes before lane deactivation and keeps later barriers uniform | `surface:vector-write-active-lane-return` `1/0/0`, smoke `136/0/0` |
| Probed green | float3 surface vector write/read before active-lane return | `surf2DLayeredwrite(float3)` plus templated `surf2DLayeredread<float3>` before/after active-lane return could assume 4 lanes or mis-pack 3-lane vectors | existing vector storage/surface lowering preserves 3-lane surface writes and reads across active-lane barrier lowering | `surface:float3-vector-active-lane-return` `1/0/0`, smoke `139/0/0` |
| Probed green | uint3 surface vector write/read before active-lane return | `surf2DLayeredwrite(uint3)` plus templated `surf2DLayeredread<uint3>` before/after active-lane return could assume 4 lanes, mis-pack 3-lane uint vectors, or lose unsigned surface read casts | existing vector storage/surface lowering preserves 3-lane uint surface writes and reads across active-lane barrier lowering; surface fixtures remain `Float32Array`-backed by current runtime contract | `surface:uint3-vector-active-lane-return` `1/0/0`, smoke `143/0/0` |
| Probed green | int3 surface vector write/read before active-lane return | `surf2DLayeredwrite(int3)` plus templated `surf2DLayeredread<int3>` before/after active-lane return could lose signedness, assume 4 lanes, or mis-pack 3-lane int vectors | existing vector storage/surface lowering preserves signed 3-lane surface writes and reads across active-lane barrier lowering; surface fixtures remain `Float32Array`-backed by current runtime contract | `surface:int3-vector-active-lane-return` `1/0/0`, smoke `144/0/0` |
| Probed green | float4 surface vector write/read before active-lane return | `surf2DLayeredwrite(float4)` plus typed layered read before/after active-lane return could scalarize full-width float vectors or reorder the inactive-lane write before the barrier | existing vector storage/surface lowering preserves full-width float4 surface writes and reads across active-lane barrier lowering | `surface:layered-float4-vector-active-lane-return` `1/0/0`, smoke `166/0/0` |
| Probed green | mixed scalar/vector surface write/read before active-lane return | scalar `surf2DLayeredwrite(float)` plus vector `surf2DLayeredwrite(float4)` before active-lane return could alias layers, lose scalar/vector typed reads, or reorder inactive-lane writes before the barrier | existing typed surface read/write lowering preserves mixed scalar/vector layered surface side effects across active-lane barrier lowering | `surface:layered-mixed-scalar-vector-active-lane-return` `1/0/0`, smoke `167/0/0` |
| Probed green | surface read feeding pointer-alias store before return | typed layered surface scalar read feeding a scalar pointer alias over `float4*` storage before active-lane return could mis-address vector lanes, lose surface layer indexing, or reorder the alias write past lane deactivation | existing surface read lowering and scalar-view pointer alias writes preserve inactive-lane side effects across active-lane barrier lowering | `surface:pointer-alias-active-lane-store` `1/0/0`, smoke `168/0/0` |
| Probed green | surface read feeding atomic pointer-alias store before return | typed layered surface scalar read feeding an atomic scalar pointer alias over `uint4*` storage before active-lane return could mis-promote vector storage, mis-address scalar lanes, or lose later direct vector stores to atomic-promoted output | existing surface read lowering, atomic vector storage promotion, and scalar-view atomic pointer writes preserve inactive-lane side effects across active-lane barrier lowering | `surface:pointer-alias-atomic-active-lane-store` `1/0/0`, smoke `169/0/0` |
| Probed green | surface atomic vector readback after pointer alias atomic | `uint4 value = out[1]` after a surface-fed scalar atomic alias over `uint4*` storage could read scalar lanes `1..4` instead of vector element lanes `4..7` or lose atomic promotion on the readback path | existing atomic vector storage-view reads scale whole-vector indexes before lane-wise atomic loads for surface-fed pointer alias atomics | `surface:pointer-alias-atomic-vector-readback` `1/0/0`, smoke `170/0/0` |
| Probed green | surface atomic vector compound/member writes through helper | `vectorOut[lane] += value` and `vectorOut[lane].y += value` through pointer helpers targeting surface-fed atomic-promoted vector storage could update wrong scalar lanes or bypass atomic vector storage helpers | existing pointer helper lowering keeps vector-element indexes and atomic vector helpers scale read/write paths for surface-fed pointer alias atomics | `surface:pointer-alias-atomic-vector-compound` `1/0/0`, smoke `171/0/0` |
| Probed green | surface atomic pointer-array selection | layered surface `uint4` data selecting a scalar pointer from a local pointer array across two vector buffers could choose the wrong storage handle/base, drop dynamic pointer-array index lowering, or mis-scale later whole-vector readback | existing pointer-array storage handles and atomic vector storage-view reads preserve selected surface-fed scalar atomics across buffers | `surface:pointer-alias-atomic-pointer-array-select` `1/0/0`, smoke `188/0/0` |
| Probed green | uint4/int4 surface vector write/read before active-lane return | `surf2DLayeredwrite(uint4/int4)` plus templated layered reads before/after active-lane return could lose signedness/unsigned casts or fail full-width vector lane packing | existing vector storage/surface lowering preserves full-width signed and unsigned surface writes and reads across active-lane barrier lowering; surface fixtures remain `Float32Array`-backed by current runtime contract | `surface:uint4-vector-active-lane-return,surface:int4-vector-active-lane-return` `2/0/0`, smoke `146/0/0` |
| Probed green | float2/uint2/int2 surface vector write/read before active-lane return | `surf2DLayeredwrite(float2/uint2/int2)` plus templated layered reads before/after active-lane return could mis-scale 2-lane vectors, drop signedness/unsigned casts, or reuse 3/4-lane assumptions | existing vector storage/surface lowering preserves 2-lane float/signed/unsigned surface writes and reads across active-lane barrier lowering; surface fixtures remain `Float32Array`-backed by current runtime contract | `surface:float2-vector-active-lane-return,surface:uint2-vector-active-lane-return,surface:int2-vector-active-lane-return` `3/0/0`, smoke `149/0/0` |
| Probed green | helper layered vector write | vector `surf2DLayeredwrite` through `cudaSurfaceObject_t` helper param could lose handle/lane/layer semantics | existing surface dispatch + vector lane writes held | `surface:helper-vector-layered-write` |
| Probed green | 1D surface write | `surf1Dwrite` could share broken 2D/layered lowering path | existing Y=0/Z=0 lowering held | `surface:surf1d-write` |
| Probed green | 1D surface vector write | `surf1Dwrite(float4)` through helper could mis-scale x-byte offsets or emit invalid multi-statement vector writes for 1D surfaces | existing lane-wise vector surface write lowering preserves 1D helper param writes | `surface:surf1d-vector-write` `1/0/0`, smoke `174/0/0` |
| Fixed | 1D surface read | `surf1Dread` was missing from analyzer/reference/WGSL even though `surf1Dwrite` existed | x-only reads now lower as y=0/z=0 and support pointer and return forms through analyzer, CPU reference, and WGSL | `surface:surf1d-read` `1/0/0`, smoke `172/0/0` |
| Probed green | 1D surface vector read | `surf1Dread<float4>` and pointer-form `surf1Dread(&float4, ...)` could mis-scale x-byte offsets or collapse vector lanes after adding scalar 1D reads | existing lane-wise surface read lowering preserves 1D vector pointer and return forms | `surface:surf1d-vector-read` `1/0/0`, smoke `173/0/0` |
| Probed green | 1D surface vector write before active-lane return | `surf1Dwrite(float4)` before return-and-barrier lowering could drop inactive-lane side effects, mis-guard multi-statement vector writes, or mis-read lanes after the barrier | existing active-lane lowering preserves 1D vector surface writes before lane deactivation and later `surf1Dread<float4>` readback | `surface:surf1d-vector-active-lane-return` `1/0/0`, smoke `175/0/0` |
| Probed green | surf1Dread pointer-alias store before active-lane return | scalar `surf1Dread` feeding a scalar pointer alias over `float4*` storage before active-lane return could mis-address vector lanes or lose inactive-lane side effects | existing 1D surface read lowering and scalar-view pointer alias writes preserve inactive-lane writes across active-lane barrier lowering | `surface:surf1d-pointer-alias-active-lane-store` `1/0/0`, smoke `176/0/0` |
| Probed green | surf1Dread atomic pointer-alias store before active-lane return | unsigned `surf1Dread` feeding an atomic scalar pointer alias over `uint4*` storage before active-lane return could lose atomic vector storage promotion or mis-address scalar lanes | existing 1D surface read lowering, atomic vector storage promotion, and scalar-view atomic pointer writes preserve inactive-lane side effects | `surface:surf1d-pointer-alias-atomic-active-lane-store` `1/0/0`, smoke `177/0/0` |
| Probed green | surf1Dread atomic vector readback | `uint4 value = out[1]` after a surf1D-fed scalar atomic alias over `uint4*` storage could read scalar lanes or lose atomic vector promotion on whole-vector readback | existing atomic vector storage-view reads scale whole-vector indexes before lane-wise atomic loads for surf1D-fed pointer alias atomics | `surface:surf1d-pointer-alias-atomic-vector-readback` `1/0/0`, smoke `178/0/0` |
| Probed green | surf1Dread atomic vector compound/member writes through helper | `vectorOut[lane] += value` and `vectorOut[lane].y += value` through pointer helpers targeting surf1D-fed atomic-promoted vector storage could update wrong scalar lanes or bypass atomic vector storage helpers | existing pointer helper lowering keeps vector-element indexes and atomic vector helpers scale read/write paths for surf1D-fed pointer alias atomics | `surface:surf1d-pointer-alias-atomic-vector-compound` `1/0/0`, smoke `179/0/0` |
| Probed green | surf1Dread atomic pointer-array selection | `surf1Dread<uint4>` data selecting a scalar pointer from a local pointer array across two vector buffers could choose the wrong storage handle/base, drop dynamic pointer-array index lowering, or mis-scale later whole-vector readback | existing pointer-array storage handles and atomic vector storage-view reads preserve selected surf1D-fed scalar atomics across buffers | `surface:surf1d-pointer-alias-atomic-pointer-array-select` `1/0/0`, smoke `189/0/0` |
| Probed green | texture helper vector conversion | `tex2D<uint4>` through texture object helper could lose lane casts | existing vector cast path held | `texture:object-uint4-helper-read` |
| Probed green | nested texture helper chain | texture object could break through nested device helpers | existing texture object argument lowering held | `texture:nested-helper-vector-read` |
| Probed green | conditional vector lane pointer write | vector lane pointer through conditional rebind could target wrong lane/buffer | existing scalar-view pointer lowering held | `storage:conditional-vector-lane-pointer-write` |
| Probed green | pointer-array selected helper args | pointer array entries crossing helper call boundary could lose storage view | existing pointer handle lowering held | `storage:helper-pointer-array-selected-args` |
| Probed green | active-lane compound RHS | inactive lanes could evaluate side-effecting compound assignment RHS after return-before-barrier lowering | active-lane guard held | `control:active-lane-compound-assignment-guarded-rhs` |
| Probed green | active-lane uniform break barrier | loop break before barrier could corrupt active-lane barrier lowering | uniform break path held | `control:active-lane-uniform-break-barrier` |
| Probed green | scalar subgroup truthiness assignment | subgroup expression truthiness feeding assignment could miscast in scalar fallback | scalar subgroup lowering held | `control:subgroup-truthiness-assignment-scalar` |
| Fixed | packed shared vector scalar helpers | non-atomic `float*` helper view over shared `float4[]` returned/wrote whole vectors or used unscaled base | shared pointer helpers split packed element index from lane and preserve atomic shared flat-lane path | `storage:cross-space-vector-alias-consistency`, smoke `110/0/0` |
| Fixed | mixed local/storage helper pointer diagnostic | a helper called with both storage pointers and local pointer-array elements failed late with a generic storage-pointer error | `emitDevicePointerArgument` reports the local pointer-array boundary explicitly while preserving supported all-local helper lowering | compiler unit `373/0`, smoke `110/0/0` |
| Fixed | texture vector conversion constructor | `make_float4(uint4)` from a texture helper was rejected by analyzer and reference runtime even though WGSL emitter could cast lanes | vector constructors now accept CUDA vector args across scalar families in analyzer and reference runtime | `texture:helper-vector-cast-coercion`, compiler unit `374/0`, smoke `111/0/0` |
| Fixed | loop-internal return/barrier replay | real WebGPU lowering used loop-local active-lane guards, but e2e diagnostic gate only recognized top-level `bg_active_lane` and reported `non-uniform-return-before-barrier` | diagnostic gate now accepts proven `bg_barrier_loop_active_*` return lowering; replay cleanup removes stale last-failure file after green rerun | `control:active-lane-loop-internal-return-barrier` `1/0/0`, compiler unit `375/0`, smoke `112/0/0` |
| Fixed | alternate-branch return before barrier | `if (...) { active work } else { return; }` before a later barrier was not normalized into the existing active-lane return split | `splitIfTrailingVoidReturn` now supports return in either branch and preserves the still-active branch under the active-lane guard | `control:active-lane-alternate-return-barrier` `1/0/0`, compiler unit `376/0`, smoke `113/0/0` |
| Fixed | nested return before barrier | nested `return` inside a non-returning outer branch before a later barrier could survive as a raw WGSL `return` | early-return detection now recurses and nested return branches lower through active-lane guarded branch emission | `control:active-lane-nested-return-barrier` `1/0/0`, compiler unit `377/0`, smoke `114/0/0` |
| Probed green | loop alternate-branch return before barrier | loop-local `if (...) { work } else { return; }` could break active-lane barrier uniformity | existing loop-local active flag now covers alternate return split under bounded barrier loop lowering | `control:active-lane-loop-alternate-return-barrier` `1/0/0`, compiler unit `378/0`, smoke `115/0/0` |
| Probed green | side effect before loop return | side-effecting statements before a loop-local return could be dropped or evaluated after lane deactivation | existing `beforeReturn` lowering preserves the side effect before flipping loop-local active flag | `control:active-lane-loop-return-side-effect-barrier` `1/0/0`, compiler unit `379/0`, smoke `116/0/0` |
| Probed green | vector lane side effect before loop return | vector-lane writes before active-lane return could be lost or mis-addressed when later barriers force active-lane lowering | existing vector lane storage lowering preserves lane side effects before loop-local active flag flips | `control:active-lane-vector-return-side-effect-barrier` `1/0/0`, compiler unit `380/0`, smoke `117/0/0` |
| Probed green | pointer alias side effect before loop return | pointer-alias writes before active-lane return could write wrong storage view/index or be lost before lane deactivation | existing pointer helper lowering preserves alias side effects before loop-local active flag flips | `control:active-lane-pointer-alias-return-side-effect-barrier` `1/0/0`, compiler unit `381/0`, smoke `118/0/0` |
| Probed green | atomic side effect before loop return | atomic increments before active-lane return could be dropped or reordered past lane deactivation | existing atomic lowering preserves side effects before loop-local active flag flips | `control:active-lane-atomic-return-side-effect-barrier` `1/0/0`, compiler unit `382/0`, smoke `119/0/0` |
| Probed green | shared-memory side effect before return | shared-memory writes before active-lane return could be lost before a later barrier read by active lanes | existing top-level active-lane lowering preserves shared side effects and keeps later barriers uniform | `control:active-lane-shared-return-side-effect-barrier` `1/0/0`, compiler unit `383/0`, smoke `120/0/0` |
| Probed green | surface side effect before return | `surf2Dwrite` before active-lane return could be dropped or hidden behind non-uniform barrier diagnostics | active-lane lowering preserves surface writes before deactivating the lane and keeps later barriers uniform | `surface:active-lane-return-side-effect` `1/0/0`, compiler unit `384/0`, smoke `121/0/0` |
| Probed green | texture read side effect before return | helper `tex2D<float4>` before active-lane return could be dropped, miscast, or hidden behind non-uniform barrier diagnostics | active-lane lowering preserves texture helper reads and storage writes before deactivating the lane, while keeping later barriers uniform | `texture:active-lane-return-read-side-effect` `1/0/0`, compiler unit `385/0`, smoke `122/0/0` |
| Probed green | float3 texture read/store before return | `tex2D<float3>` feeding a `float3*` store before active-lane return could assume 4 lanes, mis-pack 3-lane output, or reorder deactivation | existing texture vector conversion and packed float3 storage preserve 3-lane texture reads across active-lane barrier lowering | `texture:float3-active-lane-store` `1/0/0`, smoke `140/0/0` |
| Probed green | float2/uint2/int2 texture read/store before return | `tex2D<float2/uint2/int2>` feeding 2-lane vector stores before active-lane return could mis-scale packed lanes, lose signedness/unsigned casts, or reuse 3/4-lane assumptions | existing texture vector conversion and packed 2-lane storage preserve float/signed/unsigned texture reads across active-lane barrier lowering | `texture:float2-active-lane-store,texture:uint2-active-lane-store,texture:int2-active-lane-store` `3/0/0`, smoke `152/0/0` |
| Probed green | uint3 texture read/store before return | `tex2D<uint3>` feeding a `uint3*` store before active-lane return could miscast lanes, assume 4 lanes, or mis-pack 3-lane uint output | existing texture vector conversion and packed uint3 storage preserve 3-lane uint texture reads across active-lane barrier lowering | `texture:uint3-active-lane-store` `1/0/0`, smoke `141/0/0` |
| Probed green | int3 texture read/store before return | `tex2D<int3>` feeding an `int3*` store before active-lane return could lose signedness, assume 4 lanes, or mis-pack 3-lane int output | existing texture vector conversion and packed int3 storage preserve signed 3-lane texture reads across active-lane barrier lowering | `texture:int3-active-lane-store` `1/0/0`, smoke `142/0/0` |
| Probed green | float4/uint4/int4 texture read/store before return | `tex2D<float4/uint4/int4>` feeding full-width vector stores before active-lane return could lose signedness/unsigned casts, scalarize lanes, or reorder deactivation | existing texture vector conversion and packed 4-lane storage preserve float/signed/unsigned texture reads across active-lane barrier lowering | `texture:float4-active-lane-store,texture:uint4-active-lane-store,texture:int4-active-lane-store` `3/0/0`, smoke `155/0/0` |
| Probed green | texture-to-surface side effect before return | helper `tex2D<float4>` feeding helper `surf2Dwrite` before active-lane return could lose either handle or reorder side effects | active-lane lowering preserves texture reads and surface writes across helper calls before deactivating the lane | `texture-surface:active-lane-return-side-effect` `1/0/0`, compiler unit `386/0`, smoke `123/0/0` |
| Probed green | texture-fed layered surface vector write before return | `tex2D<float4>` feeding vector `surf2DLayeredwrite` before active-lane return could lose texture value lanes, surface handle/layer, or deactivation order | active-lane lowering preserves texture vector read and lane-wise layered surface write before deactivating the lane | `texture-surface:vector-active-lane-return` `1/0/0`, smoke `137/0/0` |
| Probed green | 2-lane texture-fed layered surface vector write/read before return | `tex2D<float2/uint2/int2>` feeding typed `surf2DLayeredwrite` and post-barrier typed surface reads could mis-scale lanes, lose signedness/unsigned casts, or reorder active-lane deactivation | existing texture conversion, typed surface vector writes/reads, and active-lane lowering preserve 2-lane float/signed/unsigned values across texture-to-surface flow | `texture-surface:float2-vector-active-lane-return,texture-surface:uint2-vector-active-lane-return,texture-surface:int2-vector-active-lane-return` `3/0/0`, smoke `158/0/0` |
| Probed green | 3-lane texture-fed layered surface vector write/read before return | `tex2D<float3/uint3/int3>` feeding typed `surf2DLayeredwrite` and post-barrier typed surface reads could mis-pack 3 lanes, lose signedness/unsigned casts, or reorder active-lane deactivation | existing texture conversion, typed surface vector writes/reads, and active-lane lowering preserve 3-lane float/signed/unsigned values across texture-to-surface flow | `texture-surface:float3-vector-active-lane-return,texture-surface:uint3-vector-active-lane-return,texture-surface:int3-vector-active-lane-return` `3/0/0`, smoke `161/0/0` |
| Probed green | 4-lane texture-fed layered surface vector write/read before return | `tex2D<float4/uint4/int4>` feeding typed `surf2DLayeredwrite` and post-barrier typed surface reads could scalarize full-width vectors, lose signedness/unsigned casts, or reorder active-lane deactivation | existing texture conversion, typed surface vector writes/reads, and active-lane lowering preserve 4-lane float/signed/unsigned values across texture-to-surface flow | `texture-surface:float4-vector-active-lane-return,texture-surface:uint4-vector-active-lane-return,texture-surface:int4-vector-active-lane-return` `3/0/0`, smoke `164/0/0` |
| Probed green | texture-to-surface uint4 atomic pointer-array selection | `tex2D<uint4>` feeding `surf2DLayeredwrite`/`surf2DLayeredread<uint4>` then selecting a scalar pointer from a local pointer array across two vector buffers could lose texture/surface lane casts, choose the wrong storage handle/base, or mis-scale later whole-vector readback | existing texture conversion, typed surface vector writes/reads, pointer-array storage handles, and atomic vector storage-view reads preserve texture-to-surface selected scalar atomics across buffers | `texture-surface:uint4-atomic-pointer-array-select` `1/0/0`, smoke `190/0/0` |
| Probed green | volume texture-to-surface atomic pointer-array selection | `tex2DLayered<uint4>` plus `tex3D<uint4>` feeding `surf3Dwrite`/`surf3Dread<uint4>` then selecting a scalar pointer from a local pointer array across two vector buffers could mis-map atlas/volume lanes, lose 3D surface z indexing, choose the wrong storage handle/base, or mis-scale later whole-vector readback | existing atlas/3D texture conversion, 3D surface vector writes/reads, pointer-array storage handles, and atomic vector storage-view reads preserve volume texture-to-surface selected scalar atomics across buffers | `texture-surface:volume-atomic-pointer-array-select` `1/0/0`, smoke `191/0/0` |
| Probed green | volume texture-to-surface atomic pointer-array active-lane return | inactive lane `return` after `tex2DLayered<uint4>` plus `tex3D<uint4>` writes `surf3Dwrite`, then active lanes read `surf3Dread<uint4>` and select a scalar pointer array for atomics; this could drop pre-return side effects, mis-handle active-lane barriers, mis-map 3D surface z, or choose the wrong vector storage view | active-lane lowering preserves pre-return volume texture-to-surface side effects, then pointer-array atomic vector writes and readback stay coherent across buffers | `texture-surface:volume-atomic-pointer-array-active-lane-return` `1/0/0`, smoke `192/0/0` |
| Fixed | volume texture-to-surface vector-pointer array atomic active-lane return | inactive lane volume texture-to-3D-surface write then active-lane `surf3Dread<uint4>` selected a `uint4*` local pointer-array target and performed `atomicAdd(reinterpret_cast<uint*>(targets[0]) + 3, 9u)`; WGSL lowered the pointer arithmetic to a plain `u32` base and failed pipeline creation | scalar cast value type now survives binary pointer arithmetic over vector pointers, so pointer-helper atomic lowering emits `bg_ptr_atomicAdd_u32(buffer, base + lane, value)` | `texture-surface:volume-vector-pointer-array-atomic-active-lane-return` `1/0/0`, unit `413/0`, smoke `193/0/0` |
| Probed green | volume texture-to-surface vector-pointer array CAS/exchange active-lane return | inactive lane volume texture-to-3D-surface write then active-lane `surf3Dread<uint4>` selected a `uint4*` local pointer-array target and performed `atomicCAS`/`atomicExch`/`atomicAdd` through a scalar cast plus `atomicCAS(reinterpret_cast<uint*>(targets[0]) + 0, ...)`; this could regress the scalar-cast pointer arithmetic fix across CAS/exchange paths, lose compare/exchange return values, or reorder pre-return surface side effects | existing scalar-cast pointer arithmetic value typing, pointer-helper CAS/exchange lowering, and active-lane surface side-effect lowering preserve CAS/exchange semantics through vector pointer arrays | `texture-surface:volume-vector-pointer-array-cas-active-lane-return` `1/0/0`, unit `414/0`, smoke `194/0/0` |
| Probed green | mixed scalar/vector texture-fed layered surface write/read before return | scalar `tex2D<float>` plus vector `tex2D<uint4>` feeding typed `surf2DLayeredwrite(float4)` and post-barrier typed surface read could lose lane casts, scalar/vector conversion, surface handle/layer, or active-lane ordering | existing texture conversion, numeric casts, typed surface vector writes/reads, and active-lane lowering preserve mixed scalar/vector texture-to-surface flow | `texture-surface:mixed-vector-active-lane-return` `1/0/0`, smoke `165/0/0` |
| Probed green | layered/3D texture vector read into 3D surface vector write before return | `tex2DLayered<float4>` plus `tex3D<float4>` feeding vector `surf3Dwrite` before active-lane return could lose atlas/volume lanes, z indexing, or deactivation order | active-lane lowering preserves layered/3D texture vector reads, 3D surface vector write, and post-barrier vector readback | `texture-surface:volume-vector-active-lane-return` `1/0/0`, smoke `138/0/0` |
| Probed green | atlas/layered texture side effect before return | `tex2DLayered` and `tex3D` helper reads before active-lane return could mis-map atlas rows or be dropped before a later barrier | active-lane lowering preserves atlas/layered texture reads before lane deactivation | `texture:atlas-active-lane-return-read-side-effect` `1/0/0`, compiler unit `387/0`, smoke `124/0/0` |
| Probed green | atlas/3D texture vector store before return | `tex2DLayered<float4>` plus `tex3D<float4>` feeding vector storage before active-lane return could mis-map atlas rows, lose vector lanes, or reorder inactive-lane writes before the barrier | active-lane lowering preserves layered/3D texture vector reads and inactive-lane vector stores before later barriers | `texture:atlas-vector-active-lane-store` `1/0/0`, smoke `182/0/0` |
| Probed green | atlas/3D texture vector pointer-alias store before return | `tex2DLayered<float4>` plus `tex3D<float4>` feeding scalar pointer alias writes over `float4*` storage before active-lane return could mis-map atlas rows, lose vector lanes through scalar view writes, or reorder inactive-lane writes before the barrier | active-lane lowering preserves layered/3D texture vector reads and scalar-view pointer alias vector writes before later barriers | `texture:atlas-vector-pointer-alias-active-lane-store` `1/0/0`, smoke `183/0/0` |
| Probed green | atlas/3D texture vector atomic pointer-alias store before return | `tex2DLayered<uint4>` plus `tex3D<uint4>` feeding scalar atomic pointer alias writes over `uint4*` storage before active-lane return could mis-map atlas rows, lose vector lanes through atomic scalar-view writes, or reorder inactive-lane atomics before the barrier | active-lane lowering preserves layered/3D texture vector reads and scalar-view atomic vector writes before later barriers | `texture:atlas-vector-atomic-pointer-alias-active-lane-store` `1/0/0`, smoke `184/0/0` |
| Probed green | atlas/3D texture vector atomic pointer-alias readback after return | whole-vector `uint4 value = out[1]` after return-before-barrier atlas-fed scalar atomics over `uint4*` storage could read scalar lanes, miss atomic promotion, or observe pre-atomic values | active-lane lowering and atomic vector storage-view reads preserve atlas-fed scalar atomics before later whole-vector readback | `texture:atlas-vector-atomic-pointer-alias-readback` `1/0/0`, smoke `185/0/0` |
| Probed green | atlas/3D texture vector atomic pointer-alias compound writes | `tex2DLayered<uint4>` plus `tex3D<uint4>` feeding scalar atomics followed by `uint4*` compound/member helper writes could mis-scale vector indexes, bypass atomic-promoted storage helpers, or read wrong lanes in the summary | existing atomic vector storage helpers preserve atlas-fed scalar atomics and later vector compound/member writes through pointer aliases | `texture:atlas-vector-atomic-pointer-alias-compound` `1/0/0`, smoke `186/0/0` |
| Probed green | atlas/3D texture vector atomic pointer-array selection | atlas-fed `uint4` texture data selecting a scalar pointer from a local pointer array across two vector buffers could choose the wrong buffer/base, lose dynamic pointer-array indexes, or mis-scale later whole-vector readback | existing pointer-array storage handles and atomic vector storage-view reads preserve selected atlas-fed scalar atomics across buffers | `texture:atlas-vector-atomic-pointer-array-select` `1/0/0`, smoke `187/0/0` |
| Probed green | deep helper texture vector store before return | nested texture helper calls feeding `float4` storage writes before active-lane return could lose vector lane stores or deactivation order | active-lane lowering preserves deep texture helper calls and vector stores before lane deactivation | `texture:deep-helper-active-lane-vector-store` `1/0/0`, compiler unit `388/0`, smoke `125/0/0` |
| Probed green | mixed scalar/vector texture store before return | scalar `tex2D<float>` plus vector `tex2D<uint4>` feeding a `float4` store before active-lane return could miscast lanes or reorder deactivation | active-lane lowering preserves mixed scalar/vector texture reads and vector stores before lane deactivation | `texture:mixed-scalar-vector-active-lane-store` `1/0/0`, compiler unit `389/0`, smoke `126/0/0` |
| Probed green | texture-fed pointer alias store before return | texture scalar read feeding a scalar pointer alias over `float4*` storage before active-lane return could mis-address vector lanes or reorder deactivation | active-lane lowering preserves texture read and scalar pointer alias write before lane deactivation | `texture:pointer-alias-active-lane-store` `1/0/0`, compiler unit `390/0`, smoke `127/0/0` |
| Fixed | atomic vector direct store after pointer alias atomic | a buffer promoted to `array<atomic<u32>>` by pointer-alias atomics still used direct `out[...] = value` vector lane stores | direct vector assignments to atomic params/globals now emit lane-wise `atomicStore` | `texture:pointer-alias-atomic-active-lane-store` `1/0/0`, compiler unit `391/0`, smoke `128/0/0` |
| Fixed | atomic vector readback after pointer alias atomic | `uint4 value = out[1]` on an atomic-promoted vector buffer read scalar lanes `1..4` instead of vector element lanes `4..7` | atomic vector storage-view reads now scale whole-vector indexes before lane-wise `atomicLoad` for params/globals | `texture:pointer-alias-atomic-vector-readback` `1/0/0`, compiler unit `392/0`, smoke `129/0/0` |
| Fixed | atomic vector compound/member writes through device pointer helper | `vectorOut[lane] += value` and `vectorOut[lane].y += value` through pointer helpers targeting atomic-promoted vector storage either updated only one scalar lane or emitted unresolved direct `vectorOut[...]` WGSL | pointer helpers keep vector-element indexes; atomic vector storage helpers scale inside read/write helper bodies; atomic pointer-member assignment is emitted through pointer helpers before direct storage lowering | `texture:pointer-alias-atomic-vector-compound` `1/0/0`, compiler unit `393/0`, smoke `130/0/0` |

## Real WebGPU Fixture Counts

Current added surface/texture cases:

- `texture-surface:vector-helper-roundtrip`
- `surface:helper-dispatch-multiple-surfaces`
- `surface:vector-read`
- `surface:helper-vector-read-multiple-surfaces`
- `surface:helper-vector-multi-surface-active-lane-return`
- `surface:active-lane-return-side-effect`
- `surface:layered-write`
- `surface:layered-read`
- `surface:surf3d-read`
- `surface:layered-vector-read`
- `surface:surf3d-vector-read`
- `surface:surf3d-vector-write-active-lane-return`
- `surface:surf3d-helper-vector-multi-surface-active-lane-return`
- `surface:surf3d-active-lane-guarded-rhs`
- `surface:surf3d-active-lane-guarded-rhs-all-inactive`
- `surface:surf3d-pointer-alias-active-lane-store`
- `surface:surf3d-pointer-alias-atomic-active-lane-store`
- `surface:surf3d-pointer-alias-atomic-vector-readback`
- `surface:surf3d-pointer-alias-atomic-vector-compound`
- `surface:surf3d-pointer-alias-atomic-pointer-array-select`
- `surface:surf3d-pointer-alias-atomic-pointer-array-select-false-branch`
- `surface:surf3d-pointer-alias-atomic-pointer-array-active-lane-return`
- `surface:surf3d-pointer-alias-atomic-pointer-array-compound-active-lane-return`
- `surface:surf3d-pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch`
- `surface:surf3d-pointer-alias-atomic-pointer-array-cas-active-lane-return`
- `surface:surf3d-pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch`
- `surface:surf3d-pointer-alias-atomic-pointer-array-minmax-active-lane-return`
- `surface:surf3d-pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch`
- `surface:vector-read-active-lane-return`
- `surface:vector-write-active-lane-return`
- `surface:float2-vector-active-lane-return`
- `surface:uint2-vector-active-lane-return`
- `surface:int2-vector-active-lane-return`
- `surface:float3-vector-active-lane-return`
- `surface:uint3-vector-active-lane-return`
- `surface:int3-vector-active-lane-return`
- `surface:layered-float4-vector-active-lane-return`
- `surface:layered-mixed-scalar-vector-active-lane-return`
- `surface:pointer-alias-active-lane-store`
- `surface:pointer-alias-atomic-active-lane-store`
- `surface:pointer-alias-atomic-vector-readback`
- `surface:pointer-alias-atomic-vector-compound`
- `surface:pointer-alias-atomic-pointer-array-select`
- `surface:pointer-alias-atomic-pointer-array-select-false-branch`
- `surface:pointer-alias-atomic-pointer-array-active-lane-return`
- `surface:pointer-alias-atomic-pointer-array-compound-active-lane-return`
- `surface:pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch`
- `surface:pointer-alias-atomic-pointer-array-cas-active-lane-return`
- `surface:pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch`
- `surface:pointer-alias-atomic-pointer-array-minmax-active-lane-return`
- `surface:pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch`
- `surface:uint4-vector-active-lane-return`
- `surface:int4-vector-active-lane-return`
- `surface:helper-vector-layered-write`
- `surface:surf1d-write`
- `surface:surf1d-vector-write`
- `surface:surf1d-read`
- `surface:surf1d-vector-read`
- `surface:surf1d-vector-active-lane-return`
- `surface:surf1d-pointer-alias-active-lane-store`
- `surface:surf1d-pointer-alias-atomic-active-lane-store`
- `surface:surf1d-pointer-alias-atomic-vector-readback`
- `surface:surf1d-pointer-alias-atomic-vector-compound`
- `surface:surf1d-pointer-alias-atomic-pointer-array-select`
- `surface:surf1d-pointer-alias-atomic-pointer-array-select-false-branch`
- `surface:surf1d-pointer-alias-atomic-pointer-array-active-lane-return`
- `surface:surf1d-pointer-alias-atomic-pointer-array-compound-active-lane-return`
- `surface:surf1d-pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch`
- `surface:surf1d-pointer-alias-atomic-pointer-array-cas-active-lane-return`
- `surface:surf1d-pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch`
- `surface:surf1d-pointer-alias-atomic-pointer-array-minmax-active-lane-return`
- `surface:surf1d-pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch`
- `texture:object-uint4-helper-read`
- `texture:helper-vector-cast-coercion`
- `texture:nested-helper-vector-read`
- `texture:active-lane-return-read-side-effect`
- `texture:float2-active-lane-store`
- `texture:uint2-active-lane-store`
- `texture:int2-active-lane-store`
- `texture:float3-active-lane-store`
- `texture:uint3-active-lane-store`
- `texture:int3-active-lane-store`
- `texture:float4-active-lane-store`
- `texture:uint4-active-lane-store`
- `texture:int4-active-lane-store`
- `texture:atlas-active-lane-return-read-side-effect`
- `texture:atlas-active-lane-guarded-rhs`
- `texture:atlas-active-lane-guarded-rhs-all-inactive`
- `texture:atlas-vector-active-lane-store`
- `texture:atlas-vector-pointer-alias-active-lane-store`
- `texture:atlas-vector-atomic-pointer-alias-active-lane-store`
- `texture:atlas-vector-atomic-pointer-alias-readback`
- `texture:atlas-vector-atomic-pointer-alias-compound`
- `texture:atlas-vector-atomic-pointer-array-select`
- `texture:atlas-vector-atomic-pointer-array-select-false-branch`
- `texture:atlas-vector-atomic-pointer-array-compound-active-lane-return`
- `texture:atlas-vector-atomic-pointer-array-compound-active-lane-return-false-branch`
- `texture:atlas-vector-atomic-pointer-array-cas-active-lane-return`
- `texture:atlas-vector-atomic-pointer-array-cas-active-lane-return-false-branch`
- `texture:atlas-vector-atomic-pointer-array-minmax-active-lane-return`
- `texture:atlas-vector-atomic-pointer-array-minmax-active-lane-return-false-branch`
- `texture:deep-helper-active-lane-vector-store`
- `texture:mixed-scalar-vector-active-lane-store`
- `texture:pointer-alias-active-lane-store`
- `texture:pointer-alias-atomic-active-lane-store`
- `texture-surface:float2-vector-active-lane-return`
- `texture-surface:uint2-vector-active-lane-return`
- `texture-surface:int2-vector-active-lane-return`
- `texture-surface:float3-vector-active-lane-return`
- `texture-surface:uint3-vector-active-lane-return`
- `texture-surface:int3-vector-active-lane-return`
- `texture-surface:float4-vector-active-lane-return`
- `texture-surface:uint4-vector-active-lane-return`
- `texture-surface:uint4-atomic-pointer-array-select`
- `texture-surface:uint4-atomic-pointer-array-select-false-branch`
- `texture-surface:uint4-atomic-pointer-array-active-lane-return`
- `texture-surface:uint4-atomic-pointer-array-active-lane-return-false-branch`
- `texture-surface:int4-vector-active-lane-return`
- `texture-surface:mixed-vector-active-lane-return`
- `texture-surface:volume-active-lane-guarded-rhs`
- `texture-surface:volume-active-lane-guarded-rhs-all-inactive`
- `texture:pointer-alias-atomic-vector-readback`
- `texture:pointer-alias-atomic-vector-compound`
- `texture:pointer-alias-atomic-pointer-array-select`
- `texture:pointer-alias-atomic-pointer-array-select-false-branch`
- `texture:pointer-alias-atomic-pointer-array-active-lane-return`
- `texture:pointer-alias-atomic-pointer-array-active-lane-return-false-branch`
- `texture:pointer-alias-atomic-pointer-array-compound-active-lane-return`
- `texture:pointer-alias-atomic-pointer-array-compound-active-lane-return-false-branch`
- `texture:pointer-alias-atomic-pointer-array-cas-active-lane-return`
- `texture:pointer-alias-atomic-pointer-array-cas-active-lane-return-false-branch`
- `texture:pointer-alias-atomic-pointer-array-minmax-active-lane-return`
- `texture:pointer-alias-atomic-pointer-array-minmax-active-lane-return-false-branch`
- `texture-surface:active-lane-return-side-effect`
- `texture-surface:vector-active-lane-return`
- `texture-surface:volume-vector-active-lane-return`
- `texture-surface:volume-atomic-pointer-array-select`
- `texture-surface:volume-atomic-pointer-array-select-false-branch`
- `texture-surface:volume-atomic-pointer-array-active-lane-return`
- `texture-surface:volume-atomic-pointer-array-active-lane-return-false-branch`
- `texture-surface:volume-vector-pointer-array-atomic-active-lane-return`
- `texture-surface:volume-vector-pointer-array-atomic-active-lane-return-false-branch`
- `texture-surface:volume-vector-pointer-array-cas-active-lane-return`
- `texture-surface:volume-vector-pointer-array-cas-active-lane-return-false-branch`
- `texture-surface:volume-vector-pointer-array-compound-active-lane-return`
- `texture-surface:volume-vector-pointer-array-compound-active-lane-return-false-branch`
- `texture-surface:volume-vector-pointer-array-minmax-active-lane-return`
- `texture-surface:volume-vector-pointer-array-minmax-active-lane-return-false-branch`

Current added pointer/control cases:

- `storage:conditional-vector-lane-pointer-write`
- `storage:helper-pointer-array-selected-args`
- `storage:cross-space-vector-alias-consistency`
- `control:active-lane-compound-assignment-guarded-rhs`
- `control:active-lane-guarded-rhs-all-inactive`
- `control:active-lane-assignment-guarded-rhs-all-inactive`
- `control:active-lane-vector-atomic-guarded-rhs-all-inactive`
- `control:active-lane-compound-assignment-guarded-rhs-all-inactive`
- `control:active-lane-uniform-break-barrier`
- `control:active-lane-loop-internal-return-barrier`
- `control:active-lane-alternate-return-barrier`
- `control:active-lane-nested-return-barrier`
- `control:active-lane-loop-alternate-return-barrier`
- `control:active-lane-break-post-loop-barrier`
- `control:active-lane-while-break-post-loop-barrier`
- `control:active-lane-do-while-break-post-loop-barrier`
- `control:active-lane-loop-return-side-effect-barrier`
- `control:active-lane-vector-return-side-effect-barrier`
- `control:active-lane-pointer-alias-return-side-effect-barrier`
- `control:active-lane-atomic-return-side-effect-barrier`
- `control:active-lane-shared-return-side-effect-barrier`
- `control:active-lane-shared-byte-half2-return-barrier`
- `control:active-lane-shared-byte-bf162-return-barrier`
- `control:subgroup-truthiness-assignment-scalar`
- `storage:shared-byte-reinterpret`
- `storage:shared-byte-concurrent-writes`
- `storage:shared-byte-float-reinterpret`
- `storage:shared-byte-half2-reinterpret`
- `storage:shared-byte-bf162-reinterpret`
- `storage:shared-byte-int-helper-atomic`

Smoke current: `276/0/0`.

Full source e2e current: `453/0/0`.

Verifier current: src `453/0/0`, dist `453/0/0`.

## Remaining Probe Map

Probe these with fail-first real WebGPU fixtures:

- Surface family:
  - surface writes before active-lane return, layered writes, helper layered vector writes, layered reads, 3D reads, layered/3D vector reads, surface vector read/write before active-lane return, float2/uint2/int2/float3/uint3/int3/float4/uint4/int4 surface vector write/read before active-lane return, mixed scalar/vector layered surface side effects, surface-read pointer-alias side effects, surface-read atomic pointer-alias side effects, surface atomic vector readback, surface atomic vector compound helper writes, surf3D helper multi-surface and guarded RHS probes, surf3D all-inactive guarded RHS, surf3D pointer-alias active/atomic vector probes, surf3D pointer-array select/active/compound/CAS/minmax, surface/surf1D/surf3D pointer-array false-branch selection, surface/surf1D/surf3D compound false-branch selection, surface/surf1D/surf3D CAS/minmax false-branch selection, and 3D vector writes fed by layered/3D texture vectors are now green; keep probing next corpus-shaped surface/texture pattern
- Texture family:
  - vector helper return, cast/coercion, active-lane pre-return read, atlas/volume guarded RHS, atlas/volume all-inactive guarded RHS, atlas pointer-array false-branch selection, pointer-array false-branch selection, texture/atlas compound false-branch selection, texture/atlas CAS/minmax false-branch selection, 2D and volume scalar/vector pointer-array active-lane false-branch selection, float2/uint2/int2/float3/uint3/int3/float4/uint4/int4 texture active-lane stores, texture-to-surface pre-return side effects, constant/shared texture-to-surface active-lane reads/writes, 2-lane/3-lane/4-lane texture-fed layered surface vector writes/reads, mixed scalar/vector texture-fed layered surface vector writes/reads, texture-fed layered surface vector writes, layered/3D texture vector reads feeding 3D surface vector writes, atlas/layered active-lane reads, deep helper vector stores, mixed scalar/vector texture stores, texture-fed pointer alias writes, texture-fed pointer alias atomics, atomic vector readback, atomic vector compound helper writes, and atomic vector member helper writes are now green; keep probing next corpus-shaped texture/storage pattern
- Pointer/vector family:
  - shared-byte reinterpret local pointers, typed aliases over byte roots, direct packed-byte shared writes, unaligned typed byte-storage overlays, dynamic shared vector alias-chain pointer arrays after active-lane returns, cuda-samples WMMA shared-memory aliases, selected-kernel reachability filtering for unused constants/textures/device globals/global `half` feature requirements, device-global vector pointer-array reads after active-lane returns, and lowered-IR pruning for unreachable helper bodies are now green; mixed local pointer-param + generic storage pointer helper still has explicit diagnostic and implementation support remains future work
- Active-lane/control family:
  - loop-internal, alternate-branch, nested, loop+alternate, scalar side-effect, vector-lane side-effect, pointer-alias side-effect, atomic side-effect, shared-memory side-effect, dynamic shared vector alias-chain side effect, device-global vector scalar atomic/pointer-array/all-inactive side effects, surface side-effect, texture read side-effect, atlas/volume texture guarded RHS, all-inactive guarded RHS, surface/atlas pointer-array false-branch selection, 2D and volume texture-to-surface scalar/vector pointer-array active-lane false-branch selection, compound false-branch selection, surface CAS/minmax false-branch selection, texture CAS/minmax false-branch selection, float2/uint2/int2/float3/uint3/int3/float4/uint4/int4 texture-store side-effect, float2/uint2/int2/float3/uint3/int3/float4/uint4/int4 surface side-effect, mixed scalar/vector layered surface side-effect, surface-read pointer-alias side-effect, surface-read atomic pointer-alias side-effect, surface atomic vector readback, surface atomic vector compound helper, surf3D helper multi-surface and guarded RHS probes, surf3D pointer-alias active/atomic vector probes, surf3D pointer-array select/active/compound/CAS/minmax, texture-to-surface side-effect, 2-lane/3-lane/4-lane texture-fed layered surface vector side-effect, mixed scalar/vector texture-fed layered surface vector side-effect, texture-fed layered surface vector side-effect, layered/3D texture into 3D surface vector side-effect, atlas/layered texture return, deep helper vector-store, mixed scalar/vector texture-store, texture-fed pointer-alias, texture-fed pointer-alias atomic, atomic vector readback, atomic vector compound helper, and atomic vector member helper cases are now green in real WebGPU; keep probing next corpus-shaped texture/storage pattern
  - non-uniform break/return/continue should remain clear diagnostic or safe active-lane lowering, not silent miscompile; divergent continue before a barrier in the same target loop is blocked, while inner-loop continues that do not skip later barriers are allowed
- Perf/tooling:
  - keep `verify:changed` scoped and explain selected gates; exact `--cases` filters now run only the named case before falling back to fuzzy substring matching, large summary `caseFilters` now print compact count/sample output, and smoke wrapper avoids nested pnpm command echo
  - auto-corpus exact-case profiling no longer fails a full-corpus baseline check; filtered expected counts now match the selected case set
  - llm.c synthetic smoke reference loops are smaller, but layernorm reference cases still dominate the scoped run at about `234ms`; keep profiling reference hot spots separately from WebGPU dispatch time
  - corpus fixture expected-output baseline now covers `98/98` real WebGPU fixtures; source/dist real-world verifier and histogram/scalar hot perf gate are green after final oracle pinning
  - keep smoke real-WebGPU and fast enough for inner loop

## Gate Ladder

Use this order:

1. Target case: `pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:case -- --cases <case>`
2. Focused unit: `pnpm --filter @unlocalhosted/browsergrad-compiler test -- --runInBand -t <name>`
3. Type/lint: `pnpm --filter @unlocalhosted/browsergrad-compiler run typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler run lint`
4. Smoke: `pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:smoke`
5. Full source: `pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu -- --require-webgpu --forbid-skips --summary-only --case-timeout-ms 30000`
6. Verifier: `pnpm --filter @unlocalhosted/browsergrad-compiler run verify:real-world-cuda -- --skip-fetch --require-webgpu`

## Reporting Format

End each work chunk with:

- Found: bug/no bug
- Changed: files + root behavior
- Proof: exact counts
- Remaining: next 1-3 probes
- Git: staged/unstaged/dirty
