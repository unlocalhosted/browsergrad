/**
 * Cross-package integration adversarial tests.
 *
 * Hypotheses: CP1, CP2, CP3.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { installGrad } from "@unlocalhosted/browsergrad-grad";
import { createNodePyodideTarget as gradTarget } from "@unlocalhosted/browsergrad-grad/node-adapter";
import { installJit } from "@unlocalhosted/browsergrad-jit";
import { createNodePyodideTarget as jitTarget } from "@unlocalhosted/browsergrad-jit/node-adapter";
import {
  parseManifest, assertCompatibleRuntime, LabRuntimeMismatch,
} from "@unlocalhosted/browsergrad-runtime";
import { getPyodide, pyBool, pyStr } from "../helpers.js";

describe("cross-package integration", () => {
  let installed = false;

  beforeAll(async () => {
    if (installed) return;
    const py = await getPyodide();
    await installGrad(gradTarget(py));
    await installJit(jitTarget(py));
    installed = true;
  }, 240_000);

  it("CP1: installJit works against a bare-Pyodide-plus-numpy target", async () => {
    // Already installed in beforeAll; verify importable.
    const ok = await pyBool(`
import browsergrad_jit
hasattr(browsergrad_jit, "TensorProxy")
`);
    expect(ok).toBe(true);
  });

  it("CP2: grad and jit coexist in same session (separate top-level modules)", async () => {
    const ok = await pyBool(`
import browsergrad_grad
import browsergrad_jit
# Each has its own version + own Tensor type.
(
    browsergrad_grad.__version__.startswith("0.5.") and
    browsergrad_jit.__version__.startswith("0.8.") and
    browsergrad_grad.Tensor is not browsergrad_jit.TensorProxy
)
`);
    expect(ok).toBe(true);
  });

  it("CP3: runtime.assertCompatibleRuntime against jit's actual version", async () => {
    const jitVersion = await pyStr(`import browsergrad_jit; browsergrad_jit.__version__`);

    const manifest = {
      id: "test-cross-pkg", version: "1.0.0",
      requires_browsergrad: `^${jitVersion}`,
      required_ops: ["MATMUL"], rubric_path: "r.py",
      starter_path: "s.py", reference_path: "ref.py", datasets: [],
    } as const;
    const r = parseManifest(manifest);
    if (!r.ok) throw new Error("parse failed");
    expect(() => assertCompatibleRuntime(r.manifest, jitVersion)).not.toThrow();
    // And an incompatibly old runtime version → throws.
    expect(() => assertCompatibleRuntime(r.manifest, "0.1.0")).toThrow(LabRuntimeMismatch);
  });

  it("grad and jit have independent torch_alias namespaces (no collision)", async () => {
    // If we install grad's torch alias, jit's TensorProxy should not be `torch.Tensor`.
    const ok = await pyBool(`
import browsergrad_grad as gr
import browsergrad_jit as jt
import sys
# Reset any prior install.
sys.modules.pop("torch", None)
gr.install_torch_alias()
import torch
# torch.Tensor should be grad's Tensor, not jit's TensorProxy.
torch.Tensor is gr.Tensor and torch.Tensor is not jt.TensorProxy
`);
    expect(ok).toBe(true);
  });
});
