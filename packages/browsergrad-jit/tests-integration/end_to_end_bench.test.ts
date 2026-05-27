/**
 * End-to-end benchmark + feedback harness.
 *
 * Exercises a realistic ML workflow across multiple PRDs in one Python
 * session:
 *
 *   1. Build a 2-layer MLP (PRD-005 nn.Module, PRD-006 fusion, PRD-007 VJP)
 *   2. Forward + backward + SGD step, times each phase (PRD-005, PRD-007)
 *   3. AMP autocast on the same model (PRD-010)
 *   4. Gradient checkpointing wraps a layer (PRD-009)
 *   5. Functional grad / vjp / vmap from bg.func (PRD-014)
 *   6. safetensors save/load round-trip (PRD-008)
 *   7. ONNX export of the inference graph (PRD-016)
 *   8. Custom WGSL kernel registration (PRD-015) — registration only
 *      since we have no GPUDevice in pyodide-in-node
 *
 * Output: a JSON report keyed by PRD, with success/failure + timing
 * data + any error context. The test asserts the *report exists*; the
 * actual assertions about results are advisory feedback, not pass/fail
 * gates — this is data collection, not regression checking.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { writeFileSync } from "node:fs";
import { clearNamespace, getJitTarget } from "./pyodide-host";

interface PRDReport {
  prd: string;
  scenario: string;
  status: "pass" | "fail" | "skip";
  duration_ms?: number;
  details?: Record<string, unknown>;
  error?: string;
}

interface FeedbackReport {
  results: PRDReport[];
  summary: {
    total: number;
    passed: number;
    failed: number;
    skipped: number;
  };
  observations: string[];
}

describe("end-to-end feedback harness", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  it("runs the full workflow and produces a feedback report", async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    const report = await target.run<FeedbackReport>(`
import browsergrad_jit as bg
import numpy as np
import time
import io
import sys

results = []
observations = []

def record(prd, scenario, fn):
    t0 = time.perf_counter()
    try:
        details = fn() or {}
        dt = (time.perf_counter() - t0) * 1000
        results.append({
            "prd": prd, "scenario": scenario, "status": "pass",
            "duration_ms": dt, "details": details,
        })
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        import traceback
        tb = traceback.format_exc()
        results.append({
            "prd": prd, "scenario": scenario, "status": "fail",
            "duration_ms": dt,
            "error": type(e).__name__ + ": " + str(e),
            "details": {"traceback_tail": tb[-500:]},
        })

# -------------------------------------------------------------------
# PRD-005 / PRD-007: build MLP, run forward + backward + SGD
# -------------------------------------------------------------------
bg.manual_seed(0)
model = bg.nn.Sequential(
    bg.nn.Linear(8, 16),
    bg.nn.ReLU(),
    bg.nn.Linear(16, 4),
)
opt = bg.optim.SGD([p for p in model.parameters()], lr=0.01)

x_np = np.random.RandomState(0).randn(32, 8).astype(np.float32)
y_np = np.random.RandomState(1).randn(32, 4).astype(np.float32)

def mlp_train_step():
    x = bg.from_numpy(x_np.copy())
    y = bg.from_numpy(y_np.copy())
    opt.zero_grad()
    pred = model(x)
    loss = ((pred - y) * (pred - y)).mean()
    loss.backward()
    opt.step()
    return {"loss": float(loss.numpy()), "param_count": sum(p.numel() for p in model.parameters())}

record("PRD-005/007", "mlp_train_step", mlp_train_step)
record("PRD-005/007", "mlp_train_step_2nd_call", mlp_train_step)  # tests trace cache

# -------------------------------------------------------------------
# PRD-006: fusion — count fused kernels on a softmax workload
# -------------------------------------------------------------------
def fusion_softmax():
    bg.jit.use_fusion(True)
    x = bg.from_numpy(np.random.randn(4, 8).astype(np.float32))
    y = bg.nn.functional.softmax(x, dim=-1)
    arr = y.numpy()
    fused = bg.jit.debug_fused_kernels()
    unfused = bg.jit.debug_unfused_reasons()
    return {
        "row_sums_close_to_1": bool(np.allclose(arr.sum(axis=-1), 1.0, atol=1e-5)),
        "fused_kernel_count": len(fused),
        "unfused_reason_count": len(unfused),
    }

record("PRD-006", "softmax_fusion", fusion_softmax)

# -------------------------------------------------------------------
# PRD-008: trace cache hit on repeated forward
# -------------------------------------------------------------------
def trace_cache():
    bg.clear_cache("trace")
    bg.jit.use_trace_cache(True)
    x = bg.from_numpy(np.zeros((4, 4), dtype=np.float32))
    for _ in range(5):
        _ = (x @ x).numpy()
    stats = bg.jit.trace_cache_stats()
    return {"trace_cache_stats": stats}

record("PRD-008", "trace_cache_repeated_call", trace_cache)

# -------------------------------------------------------------------
# PRD-008: safetensors save + load round-trip
# -------------------------------------------------------------------
def safetensors_roundtrip():
    state = {
        "w1": np.random.randn(8, 16).astype(np.float32),
        "b1": np.zeros((16,), dtype=np.float32),
    }
    blob = bg.save_safetensors(state)
    restored = bg.load_safetensors(blob)
    # load_safetensors returns TensorProxies; explicit .numpy() needed
    # to compare against ndarrays. (UX feedback: np.allclose on a
    # TensorProxy raises with a helpful message pointing to .numpy().)
    return {
        "w1_match": bool(np.allclose(state["w1"], restored["w1"].numpy())),
        "blob_bytes": len(blob),
    }

record("PRD-008", "safetensors_roundtrip", safetensors_roundtrip)

# -------------------------------------------------------------------
# PRD-009: gradient checkpointing parity
# -------------------------------------------------------------------
def checkpointed_grad():
    from browsergrad_jit.utils.checkpoint import checkpoint
    bg.manual_seed(7)
    x_np_ = np.random.randn(4, 8).astype(np.float32)
    target_np = np.random.randn(4, 4).astype(np.float32)

    def make_layer():
        np.random.seed(7)
        return bg.nn.Linear(8, 4)

    # Reference (no checkpoint)
    m1 = make_layer()
    x1 = bg.from_numpy(x_np_.copy())
    t1 = bg.from_numpy(target_np.copy())
    loss1 = ((m1(x1) - t1) * (m1(x1) - t1)).mean()
    loss1.backward()
    gw_ref = m1.weight.grad.numpy()

    # Checkpointed
    m2 = make_layer()
    x2 = bg.from_numpy(x_np_.copy())
    t2 = bg.from_numpy(target_np.copy())
    y2 = checkpoint(lambda xx: m2(xx), x2)
    loss2 = ((y2 - t2) * (y2 - t2)).mean()
    loss2.backward()
    gw_ckpt = m2.weight.grad.numpy()
    return {"max_grad_diff": float(np.max(np.abs(gw_ref - gw_ckpt)))}

record("PRD-009", "checkpoint_grad_parity", checkpointed_grad)

# -------------------------------------------------------------------
# PRD-010: AMP autocast + GradScaler
# -------------------------------------------------------------------
def amp_train_step():
    bg.manual_seed(0)
    model_a = bg.nn.Linear(8, 4)
    scaler = bg.amp.GradScaler()
    x = bg.from_numpy(np.random.randn(2, 8).astype(np.float32))
    y = bg.from_numpy(np.random.randn(2, 4).astype(np.float32))
    with bg.amp.autocast(device_type="webgpu", dtype="float16"):
        pred = model_a(x)
        loss = ((pred - y) * (pred - y)).mean()
    loss_scaled = scaler.scale(loss)
    loss_scaled.backward()
    return {
        "loss": float(loss.numpy()),
        "scale": scaler.get_scale(),
    }

record("PRD-010", "amp_autocast_basic", amp_train_step)

def amp_fp32_accumulator_matmul():
    # K=4096 all-ones — f16 accumulator would saturate, fp32 stays exact.
    K = 4096
    a = bg.from_numpy(np.ones((1, K), dtype=np.float16))
    b = bg.from_numpy(np.ones((K, 1), dtype=np.float16))
    out = (a @ b).numpy()
    return {"value": float(out[0, 0]), "exact": float(out[0, 0]) == 4096.0}

record("PRD-010", "fp32_accumulator_matmul_K4096", amp_fp32_accumulator_matmul)

# -------------------------------------------------------------------
# PRD-011.5: WebGPU realizer seam — register mock bridge, run one op
# -------------------------------------------------------------------
def webgpu_realizer_with_mock():
    class MockBridge:
        def __init__(self):
            self._h = {}; self._n = 0
            self.upload_count = 0
            self.matmul_count = 0
        def _mint(self, arr):
            self._n += 1; self._h[self._n] = np.array(arr, copy=True); return self._n
        def upload(self, data, shape, dtype):
            self.upload_count += 1
            arr = np.frombuffer(data, dtype=np.dtype(dtype))
            if shape and shape != (1,): arr = arr.reshape(shape)
            return self._mint(arr)
        def materialize(self, h, shape, dtype):
            return self._h[h].astype(np.dtype(dtype), copy=False).tobytes()
        def release(self, h): self._h.pop(h, None)
        def matmul(self, a, b, m, k, n, dtype):
            self.matmul_count += 1
            return self._mint((self._h[a] @ self._h[b]).astype(np.dtype(dtype)))
        def fused_elementwise(self, inputs, ops, shape, dtype):
            raise NotImplementedError
        def cast(self, h, src, dst, shape):
            return self._mint(self._h[h].astype(np.dtype(dst)))
        def flash_attention(self, *args, **kwargs): raise NotImplementedError
        def run_user_kernel(self, *args, **kwargs): raise NotImplementedError

    mb = MockBridge()
    bg.register_webgpu_bridge(mb)
    try:
        a = bg.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
        b = bg.from_numpy(np.array([[3.0], [4.0]], dtype=np.float32))
        out_gpu = bg.realize_webgpu(a @ b)
        out_np = (a @ b).numpy()
        return {
            "match": bool(np.allclose(out_gpu, out_np)),
            "upload_count": mb.upload_count,
            "matmul_count": mb.matmul_count,
        }
    finally:
        bg.unregister_webgpu_bridge()

record("PRD-011.5", "webgpu_realizer_mock_dispatch", webgpu_realizer_with_mock)

# -------------------------------------------------------------------
# PRD-014: functional grad + vmap
# -------------------------------------------------------------------
def functional_grad():
    x = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    g = bg.func.grad(lambda t: (t * t).sum())(x)
    return {"grad_values": g.numpy().tolist(), "grad_was_none": x.grad is None}

record("PRD-014", "functional_grad", functional_grad)

def functional_vmap():
    batched = bg.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
    mapped = bg.func.vmap(lambda x: x.sum())(batched)
    return {"per_sample_sums": mapped.numpy().tolist()}

record("PRD-014", "vmap_scalar_fn", functional_vmap)

# -------------------------------------------------------------------
# PRD-015: custom WGSL kernel registration
# -------------------------------------------------------------------
def custom_kernel_register():
    k = bg.custom_kernel(
        wgsl="@compute @workgroup_size(64, 1, 1) fn main() {}",
        name="placeholder",
        workgroup_size=(64, 1, 1),
        output_shape_fn=lambda s0: s0,
        dispatch_shape_fn=lambda s0: (max(int(np.prod(s0)), 1), 1, 1),
        num_inputs=1,
    )
    x = bg.from_numpy(np.zeros((4,), dtype=np.float32))
    u = k(x)
    return {
        "hash_prefix": k.hash[:8],
        "uop_op": u._uop.op,
        "uop_arg_op": u._uop.arg["op"],
    }

record("PRD-015", "custom_kernel_register", custom_kernel_register)

# -------------------------------------------------------------------
# PRD-016: ONNX export
# -------------------------------------------------------------------
def onnx_export():
    x = bg.from_numpy(np.array([[1.0, 2.0]], dtype=np.float32))
    w = bg.from_numpy(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32))
    b = bg.from_numpy(np.array([0.1, 0.1], dtype=np.float32))
    y = x @ w + b
    bts = bg.onnx.export_inference(y, input_buffers=(x,))
    return {"bytes": len(bts), "first_byte": bts[0]}

record("PRD-016", "onnx_export_x_at_w_plus_b", onnx_export)

# -------------------------------------------------------------------
# Cross-PRD: train + save state + load + infer
# -------------------------------------------------------------------
def full_train_save_load_infer():
    bg.manual_seed(42)
    model_t = bg.nn.Linear(4, 2)
    x = bg.from_numpy(np.random.randn(8, 4).astype(np.float32))
    y_target = bg.from_numpy(np.random.randn(8, 2).astype(np.float32))
    opt_t = bg.optim.SGD([p for p in model_t.parameters()], lr=0.01)

    initial_loss = float(((model_t(x) - y_target) * (model_t(x) - y_target)).mean().numpy())

    for _ in range(5):
        opt_t.zero_grad()
        loss = ((model_t(x) - y_target) * (model_t(x) - y_target)).mean()
        loss.backward()
        opt_t.step()

    final_loss = float(((model_t(x) - y_target) * (model_t(x) - y_target)).mean().numpy())

    # Save state
    state = {
        "weight": model_t.weight.numpy(),
        "bias": model_t.bias.numpy(),
    }
    blob = bg.save_safetensors(state)

    # Load into a fresh model
    model2 = bg.nn.Linear(4, 2)
    restored = bg.load_safetensors(blob)
    model2.weight = bg.from_numpy(restored["weight"])
    model2.bias = bg.from_numpy(restored["bias"])

    loaded_loss = float(((model2(x) - y_target) * (model2(x) - y_target)).mean().numpy())

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loaded_loss": loaded_loss,
        "loss_decreased": final_loss < initial_loss,
        "save_load_preserves_loss": abs(final_loss - loaded_loss) < 1e-5,
    }

record("X-PRD", "train_save_load_infer", full_train_save_load_infer)

# -------------------------------------------------------------------
# Observations: free-form notes the harness collects
# -------------------------------------------------------------------
n_fail = sum(1 for r in results if r["status"] == "fail")
n_pass = sum(1 for r in results if r["status"] == "pass")
observations.append(f"{n_pass}/{len(results)} scenarios passed; {n_fail} failed.")

# Cross-scenario timing observations
durations = sorted(((r["duration_ms"], r["prd"], r["scenario"]) for r in results if "duration_ms" in r), reverse=True)
if durations:
    top = durations[0]
    observations.append(f"Slowest: {top[1]}::{top[2]} at {top[0]:.0f}ms")

# Failure observations
for r in results:
    if r["status"] == "fail":
        observations.append(f"FAIL {r['prd']}::{r['scenario']}: {r['error']}")

# Summary
summary = {
    "total": len(results),
    "passed": n_pass,
    "failed": n_fail,
    "skipped": 0,
}

{"results": results, "summary": summary, "observations": observations}
`);

    // Write the report to disk for inspection.
    writeFileSync(
      "/tmp/bg-feedback-report.json",
      JSON.stringify(report, null, 2),
    );

    // Build a markdown summary the user can quote.
    const lines: string[] = [];
    lines.push("# Browsergrad-jit feedback report");
    lines.push("");
    lines.push(`**Total**: ${report.summary.total} | **Passed**: ${report.summary.passed} | **Failed**: ${report.summary.failed}`);
    lines.push("");
    lines.push("## Observations");
    for (const obs of report.observations) {
      lines.push(`- ${obs}`);
    }
    lines.push("");
    lines.push("## Per-scenario detail");
    lines.push("");
    lines.push("| PRD | Scenario | Status | Duration (ms) | Notes |");
    lines.push("|-----|----------|--------|---------------|-------|");
    for (const r of report.results) {
      const dt = r.duration_ms !== undefined ? r.duration_ms.toFixed(1) : "-";
      const sym = r.status === "pass" ? "PASS" : "FAIL";
      const notes = r.error
        ? `error: ${r.error}`
        : r.details
          ? Object.entries(r.details)
              .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
              .join(", ")
          : "";
      lines.push(`| ${r.prd} | ${r.scenario} | ${sym} | ${dt} | ${notes.slice(0, 200)} |`);
    }
    writeFileSync("/tmp/bg-feedback-report.md", lines.join("\n"));

    // Print the report so we have it in CI logs.
    console.log("\n=== END-TO-END FEEDBACK REPORT ===");
    console.log(`Total: ${report.summary.total} | Passed: ${report.summary.passed} | Failed: ${report.summary.failed}`);
    console.log("\n--- Observations ---");
    for (const obs of report.observations) {
      console.log(`  ${obs}`);
    }
    console.log("\n--- Per-scenario ---");
    for (const r of report.results) {
      const dt = r.duration_ms !== undefined ? `${r.duration_ms.toFixed(1)}ms` : "-";
      const sym = r.status === "pass" ? "✓" : "✗";
      console.log(`  ${sym} [${r.prd}] ${r.scenario} (${dt})`);
      if (r.details) {
        for (const [k, v] of Object.entries(r.details)) {
          console.log(`      ${k}: ${JSON.stringify(v)}`);
        }
      }
      if (r.error) {
        console.log(`      ERROR: ${r.error}`);
      }
    }
    console.log("=== END REPORT ===\n");

    // Sanity check: the report itself exists and ran every scenario.
    expect(report.summary.total).toBeGreaterThan(10);
    expect(report.results.length).toBe(report.summary.total);
  }, 120_000);
});
