# Compiler Bugbash Progress

Last updated: 2026-07-01T15:13:23Z

Purpose: make compiler bugbash visible. Update this file whenever a new bug, fixture, gate, or remaining risk changes.

## Dashboard

| Field | Current |
| --- | --- |
| Overall status | Active bugbash, not complete |
| Fixed failure movement | Started from 87 failing real-world/audit cases; current verifier gate is green at src `253/0/0`, dist `253/0/0` |
| Current focus | Pointer/vector storage correctness, texture/vector conversion, active-lane/control semantics, and hot-loop test speed |
| Active work item | atlas/3D texture vector atomic pointer-alias compound real WebGPU fixture green; continue next corpus-shaped storage/texture/control probe |
| Skip policy | No added skips. WebGPU commands must use `--forbid-skips` |
| Worktree | Clean after latest fixture slice commit |
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

- `surface:helper-vector-read-multiple-surfaces`: `1 passed / 0 failed / 0 skipped`
- focused surf2Dread unit run: `386 passed`
- `surface:layered-write`: `1 passed / 0 failed / 0 skipped`
- focused surf2DLayeredwrite unit run: `387 passed`
- compiler typecheck: passed
- compiler lint: passed
- WebGPU smoke: `100 passed / 0 failed / 0 skipped`
- full source WebGPU e2e: `221 passed / 0 failed / 0 skipped`
- real-world CUDA verifier: src `253/0/0`, dist `253/0/0`
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

## Bugs Found During Current Run

| Status | Area | Symptom | Root Fix | Proof |
| --- | --- | --- | --- | --- |
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
| Probed green | mixed scalar/vector texture-fed layered surface write/read before return | scalar `tex2D<float>` plus vector `tex2D<uint4>` feeding typed `surf2DLayeredwrite(float4)` and post-barrier typed surface read could lose lane casts, scalar/vector conversion, surface handle/layer, or active-lane ordering | existing texture conversion, numeric casts, typed surface vector writes/reads, and active-lane lowering preserve mixed scalar/vector texture-to-surface flow | `texture-surface:mixed-vector-active-lane-return` `1/0/0`, smoke `165/0/0` |
| Probed green | layered/3D texture vector read into 3D surface vector write before return | `tex2DLayered<float4>` plus `tex3D<float4>` feeding vector `surf3Dwrite` before active-lane return could lose atlas/volume lanes, z indexing, or deactivation order | active-lane lowering preserves layered/3D texture vector reads, 3D surface vector write, and post-barrier vector readback | `texture-surface:volume-vector-active-lane-return` `1/0/0`, smoke `138/0/0` |
| Probed green | atlas/layered texture side effect before return | `tex2DLayered` and `tex3D` helper reads before active-lane return could mis-map atlas rows or be dropped before a later barrier | active-lane lowering preserves atlas/layered texture reads before lane deactivation | `texture:atlas-active-lane-return-read-side-effect` `1/0/0`, compiler unit `387/0`, smoke `124/0/0` |
| Probed green | atlas/3D texture vector store before return | `tex2DLayered<float4>` plus `tex3D<float4>` feeding vector storage before active-lane return could mis-map atlas rows, lose vector lanes, or reorder inactive-lane writes before the barrier | active-lane lowering preserves layered/3D texture vector reads and inactive-lane vector stores before later barriers | `texture:atlas-vector-active-lane-store` `1/0/0`, smoke `182/0/0` |
| Probed green | atlas/3D texture vector pointer-alias store before return | `tex2DLayered<float4>` plus `tex3D<float4>` feeding scalar pointer alias writes over `float4*` storage before active-lane return could mis-map atlas rows, lose vector lanes through scalar view writes, or reorder inactive-lane writes before the barrier | active-lane lowering preserves layered/3D texture vector reads and scalar-view pointer alias vector writes before later barriers | `texture:atlas-vector-pointer-alias-active-lane-store` `1/0/0`, smoke `183/0/0` |
| Probed green | atlas/3D texture vector atomic pointer-alias store before return | `tex2DLayered<uint4>` plus `tex3D<uint4>` feeding scalar atomic pointer alias writes over `uint4*` storage before active-lane return could mis-map atlas rows, lose vector lanes through atomic scalar-view writes, or reorder inactive-lane atomics before the barrier | active-lane lowering preserves layered/3D texture vector reads and scalar-view atomic vector writes before later barriers | `texture:atlas-vector-atomic-pointer-alias-active-lane-store` `1/0/0`, smoke `184/0/0` |
| Probed green | atlas/3D texture vector atomic pointer-alias readback after return | whole-vector `uint4 value = out[1]` after return-before-barrier atlas-fed scalar atomics over `uint4*` storage could read scalar lanes, miss atomic promotion, or observe pre-atomic values | active-lane lowering and atomic vector storage-view reads preserve atlas-fed scalar atomics before later whole-vector readback | `texture:atlas-vector-atomic-pointer-alias-readback` `1/0/0`, smoke `185/0/0` |
| Probed green | atlas/3D texture vector atomic pointer-alias compound writes | `tex2DLayered<uint4>` plus `tex3D<uint4>` feeding scalar atomics followed by `uint4*` compound/member helper writes could mis-scale vector indexes, bypass atomic-promoted storage helpers, or read wrong lanes in the summary | existing atomic vector storage helpers preserve atlas-fed scalar atomics and later vector compound/member writes through pointer aliases | `texture:atlas-vector-atomic-pointer-alias-compound` `1/0/0`, smoke `186/0/0` |
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
- `texture:atlas-vector-active-lane-store`
- `texture:atlas-vector-pointer-alias-active-lane-store`
- `texture:atlas-vector-atomic-pointer-alias-active-lane-store`
- `texture:atlas-vector-atomic-pointer-alias-readback`
- `texture:atlas-vector-atomic-pointer-alias-compound`
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
- `texture-surface:int4-vector-active-lane-return`
- `texture-surface:mixed-vector-active-lane-return`
- `texture:pointer-alias-atomic-vector-readback`
- `texture:pointer-alias-atomic-vector-compound`
- `texture-surface:active-lane-return-side-effect`
- `texture-surface:vector-active-lane-return`
- `texture-surface:volume-vector-active-lane-return`

Current added pointer/control cases:

- `storage:conditional-vector-lane-pointer-write`
- `storage:helper-pointer-array-selected-args`
- `storage:cross-space-vector-alias-consistency`
- `control:active-lane-compound-assignment-guarded-rhs`
- `control:active-lane-uniform-break-barrier`
- `control:active-lane-loop-internal-return-barrier`
- `control:active-lane-alternate-return-barrier`
- `control:active-lane-nested-return-barrier`
- `control:active-lane-loop-alternate-return-barrier`
- `control:active-lane-loop-return-side-effect-barrier`
- `control:active-lane-vector-return-side-effect-barrier`
- `control:active-lane-pointer-alias-return-side-effect-barrier`
- `control:active-lane-atomic-return-side-effect-barrier`
- `control:active-lane-shared-return-side-effect-barrier`
- `control:subgroup-truthiness-assignment-scalar`

Smoke current: `185/0/0`.

Full source e2e current: `221/0/0`.

Verifier current: src `253/0/0`, dist `253/0/0`.

## Remaining Probe Map

Probe these with fail-first real WebGPU fixtures:

- Surface family:
  - surface writes before active-lane return, layered writes, helper layered vector writes, layered reads, 3D reads, layered/3D vector reads, surface vector read/write before active-lane return, float2/uint2/int2/float3/uint3/int3/float4/uint4/int4 surface vector write/read before active-lane return, mixed scalar/vector layered surface side effects, surface-read pointer-alias side effects, surface-read atomic pointer-alias side effects, surface atomic vector readback, surface atomic vector compound helper writes, and 3D vector writes fed by layered/3D texture vectors are now green; keep probing next corpus-shaped surface/texture pattern
- Texture family:
  - vector helper return, cast/coercion, active-lane pre-return read, float2/uint2/int2/float3/uint3/int3/float4/uint4/int4 texture active-lane stores, texture-to-surface pre-return side effects, 2-lane/3-lane/4-lane texture-fed layered surface vector writes/reads, mixed scalar/vector texture-fed layered surface vector writes/reads, texture-fed layered surface vector writes, layered/3D texture vector reads feeding 3D surface vector writes, atlas/layered active-lane reads, deep helper vector stores, mixed scalar/vector texture stores, texture-fed pointer alias writes, texture-fed pointer alias atomics, atomic vector readback, atomic vector compound helper writes, and atomic vector member helper writes are now green; keep probing next corpus-shaped texture/storage pattern
- Pointer/vector family:
  - mixed local pointer-param + generic storage pointer helper now has explicit diagnostic; implementation support remains future work
- Active-lane/control family:
  - loop-internal, alternate-branch, nested, loop+alternate, scalar side-effect, vector-lane side-effect, pointer-alias side-effect, atomic side-effect, shared-memory side-effect, surface side-effect, texture read side-effect, float2/uint2/int2/float3/uint3/int3/float4/uint4/int4 texture-store side-effect, float2/uint2/int2/float3/uint3/int3/float4/uint4/int4 surface side-effect, mixed scalar/vector layered surface side-effect, surface-read pointer-alias side-effect, surface-read atomic pointer-alias side-effect, surface atomic vector readback, surface atomic vector compound helper, texture-to-surface side-effect, 2-lane/3-lane/4-lane texture-fed layered surface vector side-effect, mixed scalar/vector texture-fed layered surface vector side-effect, texture-fed layered surface vector side-effect, layered/3D texture into 3D surface vector side-effect, atlas/layered texture return, deep helper vector-store, mixed scalar/vector texture-store, texture-fed pointer-alias, texture-fed pointer-alias atomic, atomic vector readback, atomic vector compound helper, and atomic vector member helper cases are now green in real WebGPU; keep probing next corpus-shaped texture/storage pattern
  - non-uniform break/return should remain clear diagnostic, not silent miscompile
- Perf/tooling:
  - keep `verify:changed` scoped and explain selected gates
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
