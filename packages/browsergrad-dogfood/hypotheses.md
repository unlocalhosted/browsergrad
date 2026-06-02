# Adversarial Hypotheses — `@unlocalhosted/browsergrad-kernels@0.1.0`

Each hypothesis is a specific failure mode I expect the kernel library could
exhibit. For each, I describe the attack and the predicted outcome.
A test passes if the library survives the attack; a test fails if the
attack succeeds.

Status legend after running:
- 🟢 SAFE — library survived the attack
- 🔴 BUG — attack succeeded; package broke
- 🟡 PARTIAL — degraded but not broken (e.g. correct but slow, accepted but warned)
- ⚪ N/A — attack didn't apply (e.g. test of an op not in surface)

---

## Already discovered (from prior runs)

### H0a — `realizer.d.ts` lies: `materialize` returns Promise, declared Uint8Array
- Predicted: A TypeScript consumer following the `.d.ts` will get `undefined.buffer` at runtime.
- Status: 🔴 **CONFIRMED BUG**
- Evidence: my first dogfood run; the package's own test casts via `as unknown as Promise<Uint8Array>`.

### H0b — `kernels.attention` rejects different Q seq length vs K=V seq length
- Predicted: Cross-attention (Q seq ≠ K seq) is impossible; only self-attention works.
- Status: 🔴 **CONFIRMED BUG / DOCUMENTATION GAP**
- Evidence: `KernelError: attention: sequence lengths must match (Q: 3, K: 5, V: 5)`. Real attention should accept Q seq ≠ K=V seq.

### H0c — `kernels.softmax` numerical diff vs `reference.softmax` exceeds f32 tolerance
- Predicted: GPU softmax doesn't match CPU softmax within reasonable epsilon.
- Status: 🔴 **CONFIRMED — 0.38 max abs diff** (vs an expected < 1e-5).
- Evidence: prior run output.

---

## Numerical correctness adversarial hypotheses

### H1 — Softmax of all-equal inputs should be uniform 1/N
- Attack: `softmax([5,5,5,5])` should give `[0.25, 0.25, 0.25, 0.25]`.
- Prediction: Likely safe; the stable-softmax max-subtraction handles this.
- Why I'd test anyway: If subtraction order is wrong, all-zero numerator → 0/0 = NaN.

### H2 — Softmax of single-element row → `[1.0]`
- Attack: `softmax([42.0])` along single-element axis.
- Prediction: Likely safe but worth a probe; division by sum=1 should pass through.

### H3 — Softmax with extreme negative values (underflow)
- Attack: `softmax([-1000, -1000, 0])` — first two exp() underflow.
- Prediction: Safe IF stable; broken if naive (exp(-1000) = 0, then 0/sum is fine, but sum could be NaN).

### H4 — Softmax with Inf input
- Attack: `softmax([Infinity, 1, 1])`.
- Prediction: Likely returns NaN-filled output. Worth knowing whether kernel propagates or sanitizes.

### H5 — Softmax with NaN input
- Attack: `softmax([NaN, 1, 1])`.
- Prediction: Should produce NaN row. Failure mode: NaN poisons unrelated rows due to shared workgroup mem.

### H6 — Layernorm of all-zeros row
- Attack: variance = 0 → division by sqrt(0) → div-by-zero.
- Prediction: If eps not applied correctly, NaN. Reference likely uses `var + eps`, but does GPU?

### H7 — Layernorm of all-equal-nonzero row (e.g. all 5s)
- Attack: mean=5, var=0 → same div-by-zero path.
- Prediction: Same as H6.

### H8 — Relu of -0 (negative zero)
- Attack: `relu(-0)` — IEEE -0 has sign bit set.
- Prediction: Should return +0. Failure mode: kernel using `<` comparator might preserve the sign.

### H9 — Gelu of very large positive input
- Attack: `gelu(1e10)` — tanh approximation could overflow intermediates.
- Prediction: Should return ~input. Failure: NaN if exp() in intermediate overflows.

### H10 — Matmul accumulator precision at K=1024+
- Attack: matmul of all-ones (1, 4096) @ (4096, 1) → expect exactly 4096.
- Prediction: If fp32 accumulator, exact. If reduced precision, slight error.

### H11 — Matmul with one zero matrix
- Attack: A=zeros, B=ones → output should be all zeros.
- Prediction: Safe.

### H12 — Matmul with NaN propagation
- Attack: A has one NaN, rest zeros → only output rows touching that NaN should be NaN.
- Prediction: Safe but worth verifying NaN doesn't leak across workgroup boundaries.

---

## Shape and boundary adversarial hypotheses

### H13 — Empty tensor input (0 elements)
- Attack: `relu(tensor([0], new Float32Array(0)))`.
- Prediction: Likely crashes — many GPU runners assume length > 0.

### H14 — Single-element tensor
- Attack: `matmul([1,1] @ [1,1])` — dispatchCount might round to 0 workgroups.
- Prediction: Worth probing.

### H15 — Matmul at tile boundary minus 1 (M=15, K=15, N=15 for 16-tile kernel)
- Attack: Boundary code in tiled GEMM must handle partial tiles.
- Prediction: Existing tests show this works at 17×23×19; let me probe 1×1, 15×15×15, 1×16, 16×1.

### H16 — Non-2D matmul input (rank mismatch)
- Attack: `matmul([2,3,4] tensor, [4,2] tensor)`.
- Prediction: Should throw KernelError. Failure mode: silently treats first dim as broadcast.

### H17 — Matmul with inner-dim mismatch
- Attack: `matmul([2,3] @ [4,2])` — inner dims differ.
- Prediction: Should throw. Worth verifying error message.

### H18 — Layernorm with last axis = 1 (degenerate)
- Attack: tensor of shape [N, 1] — normalization over single element → all output should be (x - x) / 0 = 0/0.
- Prediction: Likely produces NaN row. eps should rescue but check.

### H19 — Attention with Sq=1 (single query, e.g. decoder step)
- Attack: Q has 1 row, K/V have many rows.
- Prediction: Safe — but if the kernel uses `@workgroup_size(16,16)` and dispatch gets fractional workgroups, could underflow.

---

## Bridge lifecycle and concurrency adversarial hypotheses

### H20 — Use a handle after release
- Attack: `b.release(h); b.materialize(h, ...)`.
- Prediction: Should throw or return garbage. Worst: silently returns the next allocation at the same handle ID.

### H21 — Release twice
- Attack: `b.release(h); b.release(h);`.
- Prediction: Should throw or no-op. Failure: double-free crashes the device.

### H22 — Handle ID reuse: release then allocate, does new alloc get same ID?
- Attack: probe whether handle IDs are reused.
- Prediction: Likely incrementing counter, never reused.

### H23 — Two bridges, share a device, interleave operations
- Attack: `b1.matmul` then `b2.matmul` on same device, both materialize.
- Prediction: Safe but worth verifying — pipeline cache might collide.

### H24 — 1000 uploads without release → memory growth
- Attack: Loop uploading 1000 small buffers without releasing.
- Prediction: `aliveHandleCount` should hit 1000. If GPU OOM, what happens?

### H25 — `aliveHandleCount` after device destruction
- Attack: Destroy the device, call `aliveHandleCount`.
- Prediction: Should return cached count or 0, not crash.

### H26 — `matmul` with dtype="int32" (unsupported)
- Attack: bridge.matmul(a, b, m, k, n, "int32").
- Prediction: Should throw — only float32 documented as supported.

---

## Codegen and cache adversarial hypotheses

### H27 — `generateFusedWgsl` with chain length 0
- Attack: `generateFusedWgsl([], 1)`.
- Prediction: Should throw.

### H28 — `generateFusedWgsl` with reference to nonexistent step
- Attack: `generateFusedWgsl([["ADD", -1, -2], ["DIV", 99, 0]], 2)` — step 99 doesn't exist.
- Prediction: Should throw with clear error.

### H29 — `generateFusedWgsl` with reference to nonexistent input
- Attack: `generateFusedWgsl([["ADD", -10, -20]], 2)` — inputs -10 and -20 don't exist for numInputs=2.
- Prediction: Should throw.

### H30 — `generateFusedWgsl` deterministic across calls
- Attack: call same args 100 times, hash output, expect single hash.
- Prediction: Safe per docs ("hash of ops list = cache key").

### H31 — Pipeline cache hit: repeated matmul of same shape → no recompile
- Attack: kernels.matmul same shape 10 times. Inspect device stats if exposed.
- Prediction: Should be fast (one compile, ten dispatches).

### H32 — `run_user_kernel` with WGSL syntax error
- Attack: pass invalid WGSL to bridge.run_user_kernel.
- Prediction: Should throw with compile error from browser. Failure: silent zero-output buffer.

### H33 — `run_user_kernel` with missing entry point `main`
- Attack: WGSL with `fn other()` not `fn main()`.
- Prediction: Should throw at pipeline creation.

### H34 — `run_user_kernel` with hash collision: same hash, different WGSL
- Attack: Two calls with same `hash` arg but different `wgsl`. Cache should key on hash, so second call gets first call's pipeline.
- Prediction: Cache uses hash → returns stale compiled WGSL.
- Risk: If users compute hash naively (e.g. hash of "name"), this is exploitable.

### H35 — `fused_elementwise` chain longer than supported
- Attack: 1000-op chain.
- Prediction: Should throw — likely a max length defined.

---

## Cross-cutting adversarial hypotheses

### H36 — Kernel run before adapter ready
- Attack: `await kernels.matmul()` called before `createDevice()` finishes.
- Prediction: TypeError on device.

### H37 — Operation against destroyed device
- Attack: `device.destroy(); kernels.matmul(device, ...);`
- Prediction: Should throw clean error. Failure: hangs or crashes browser tab.

### H38 — Buffer destroyed mid-operation
- Attack: `const r = matmulTiledDirect(...); inputBuffer.destroy(); await materialize(r.buffer, ...)`.
- Prediction: Either works (queue captures buffer) or throws. Crashing is unacceptable.

### H39 — Float64 input (downcasting)
- Attack: Pass `new Float64Array(...)` where Float32Array expected.
- Prediction: TypeScript catches at compile; runtime may silently produce wrong numbers.

### H40 — Tensor shape lies about data length
- Attack: `tensor([100, 100], new Float32Array(4))` — declared 10000 elts, actually 4.
- Prediction: Either throws on construction or silently reads garbage GPU memory.

---

---

## `@unlocalhosted/browsergrad-runtime` hypotheses

### R1 — `parseManifest` accepts a minimal valid manifest
### R2 — rejects non-kebab-case id (`Bad_ID` → fail)
### R3 — rejects malformed semver in version (`1.x` → fail)
### R4 — rejects oversized required_ops (>64 entries)
### R5 — rejects wrong-type required_ops (string vs array)
### R6 — `isSemverCompatible("^0.8.0", "0.8.5")` true, `"^0.8.0", "1.0.0"` false
### R7 — `isSemverCompatible("~0.8.0", "0.8.5")` true, `"~0.8.0", "0.9.0"` false
### R8 — `isSemverCompatible` exact match: only matches the exact version
### R9 — `assertCompatibleRuntime` throws `LabRuntimeMismatch` on incompatibility
### R10 — `LabRuntimeMismatch` exposes `.manifestPin` and `.runtimeVersion` fields
### R11 — Prereleases (e.g. `0.8.0-beta.1`) — actual behavior documented

---

## `@unlocalhosted/browsergrad-grad` hypotheses

### G1 — `installGrad` is idempotent (second call doesn't reinstall)
### G2 — `installGrad` exposes `browsergrad_grad` Python module
### G3 — `grad.__version__` matches the npm package version (`0.5.0`)
### G4 — Basic autograd: `(t*t).sum().backward()` → grad = 2t
### G5 — `nn.Linear` forward shape: `Linear(8,4)(zeros((16,8)))` → `(16,4)`
### G6 — `nn.Sequential + optim.SGD` converges on MSE in 20 steps
### G7 — `no_grad()` context disables autograd graph build
### G8 — `requires_grad=False` → backward leaves `.grad` as None
### G9 — `model.parameters()` walks recursively through nested Sequential
### G10 — `install_torch_alias()` wires `torch.nn` → `grad.nn`
### G11 — Cross-entropy loss decreases with training
### G12 — `state_dict` / `load_state_dict` roundtrip preserves weights
### G13 — `DataLoader(num_workers>0)` refuses with Pyodide-specific reason
### G14 — `torch.cuda.is_available()` returns `False` honestly (no fake GPU)
### G15 — `torch.compile`, `torch.fx.symbolic_trace` raise `NotImplementedError`

---

## `@unlocalhosted/browsergrad-jit` hypotheses (beyond craftingattention's 61-test coverage)

### J1 — `jit.__version__` matches npm version (`0.8.0`)
### J2 — `TensorProxy.shape` works without materializing
### J3 — `.numpy()` triggers realize and returns `np.ndarray`
### J4 — Trace cache stats are non-negative integers
### J5 — `use_fusion(False)` / `use_fusion(True)` toggle works mid-session
### J6 — Custom kernel hash is SHA-256 hex (64 chars, [0-9a-f])
### J7 — `realize_webgpu` without bridge throws with "bridge" in the error message
### J8 — `ShapeError` (not generic) raised on matmul inner-dim mismatch
### J9 — `save_safetensors` accepts dict with mixed TensorProxy + ndarray
### J10 — ONNX export bytes start with 0x08 (proto3 ir_version marker)
### J11 — ONNX `argmax` export raises `OnnxUnmappableOp`

---

## Cross-package integration hypotheses

### CP1 — `installJit` works on a bare-Pyodide-plus-numpy target
### CP2 — `grad` and `jit` coexist in the same Pyodide session (separate `install_torch_alias` namespaces)
### CP3 — `runtime.assertCompatibleRuntime` honors `jit.__version__` from the same session

---

## Hypothesis counts

**77 hypotheses across 9 categories:**

| Category | Range | Count |
|---|---|---|
| Kernels — numerical correctness | H1–H12 | 12 |
| Kernels — shape/boundary | H13–H19 | 7 |
| Kernels — bridge lifecycle | H20–H26 | 7 |
| Kernels — codegen/cache | H27–H35 | 9 |
| Kernels — cross-cutting | H36–H40 | 5 |
| Runtime — manifest + semver | R1–R11 | 11 |
| Grad — install + autograd + nn | G1–G15 | 15 |
| Jit — beyond craftingattention coverage | J1–J11 | 11 |
| Cross-package integration | CP1–CP3 | 3 |

Plus **3 already-confirmed bugs** (H0a, H0b, H0c) **and 3 new bugs surfaced by this dogfood**:
- GPU softmax returns all zeros
- GPU attention has 0.69 max diff (likely cascading from softmax)
- GPU softmax of all-equal inputs ignores the input
