/**
 * Surface tests — no Pyodide, no real Python. These tests verify the
 * codegen pipeline produces well-formed strings and the source registry
 * is internally consistent.
 *
 * Heavy lifting (real Pyodide execution, IR construction, leaf factories)
 * lives in `tests-integration/` and runs separately via
 * `pnpm test:integration`.
 */

import { describe, expect, it } from "vitest";
import { installJit } from "../src/index";
import { JitInstallError } from "../src/types";
import { SOURCE_FILES, MOUNT_ROOT } from "../src/python/index";

describe("@unlocalhosted/browsergrad-jit public surface", () => {
  it("exports installJit", () => {
    expect(typeof installJit).toBe("function");
  });

  it("exports JitInstallError as a real Error subclass", () => {
    const e = new JitInstallError("boom");
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe("JitInstallError");
    expect(e.message).toBe("boom");
  });
});

describe("Python source registry", () => {
  it("registers the expected modules in install order", () => {
    const paths = SOURCE_FILES.map((f) => f.path);
    expect(paths).toEqual([
      "browsergrad_jit/_errors.py",
      "browsergrad_jit/_ir.py",
      "browsergrad_jit/_buffer_table.py",
      "browsergrad_jit/_fusion_config.py",
      "browsergrad_jit/_realize.py",
      "browsergrad_jit/_fusion.py",
      "browsergrad_jit/_vjp.py",
      "browsergrad_jit/_trace_cache.py",
      "browsergrad_jit/_safetensors.py",
      "browsergrad_jit/_checkpoint.py",
      "browsergrad_jit/_utils_checkpoint.py",
      "browsergrad_jit/_amp.py",
      "browsergrad_jit/_bridge.py",
      "browsergrad_jit/_gpu_buffer_table.py",
      "browsergrad_jit/_realize_webgpu.py",
      "browsergrad_jit/_func.py",
      "browsergrad_jit/_custom_kernel.py",
      "browsergrad_jit/_onnx.py",
      "browsergrad_jit/_lab.py",
      "browsergrad_jit/_tensor_proxy.py",
      "browsergrad_jit/_functional.py",
      "browsergrad_jit/_nn.py",
      "browsergrad_jit/_optim.py",
      "browsergrad_jit/_torch_compat.py",
      "browsergrad_jit/__init__.py",
    ]);
  });

  it("ships non-empty content for every registered module", () => {
    for (const file of SOURCE_FILES) {
      expect(file.content.length).toBeGreaterThan(100);
    }
  });

  it("mounts under a distinct path from browsergrad-grad", () => {
    // The two packages must coexist; mount roots have to differ. This
    // guards against an accidental rename matching grad's mount.
    expect(MOUNT_ROOT).toBe("/lib/browsergrad_jit_src");
    expect(MOUNT_ROOT).not.toBe("/lib/browsergrad_grad_src");
  });

  it("interpolates the package version into __init__.py", () => {
    // The init template carries `__version__ = "${pkg.version}"`. After
    // codegen, the actual version string should be substituted in.
    const initFile = SOURCE_FILES.find((f) => f.path.endsWith("__init__.py"));
    expect(initFile).toBeDefined();
    expect(initFile!.content).toMatch(/__version__ = "0\.8\.0"/);
  });

  it("declares all 28 opcodes in _ir.py (23 core + 2 fusion + 2 autograd + 1 AMP)", () => {
    // Sanity check that the codegen bundled the IR with every opcode the
    // PRD-005 + PRD-006 + PRD-007 + PRD-010 surface needs.
    const irFile = SOURCE_FILES.find((f) => f.path.endsWith("_ir.py"));
    expect(irFile).toBeDefined();
    const ops = [
      "OP_BUFFER", "OP_LOAD", "OP_STORE", "OP_CONST", "OP_RANDOM",
      "OP_CAST", "OP_ADD", "OP_MUL", "OP_DIV", "OP_NEG",
      "OP_EXP", "OP_LOG", "OP_CMP", "OP_MATMUL", "OP_REDUCE",
      "OP_RESHAPE", "OP_PERMUTE", "OP_SLICE", "OP_PAD",
      "OP_WHERE", "OP_INDEX", "OP_MASK", "OP_CUSTOM",
      "OP_FUSED_ELEMENTWISE", "OP_FUSED_SOFTMAX",
      "OP_SCATTER_ADD", "OP_BROADCAST_TO",
      "OP_ISNAN",
    ];
    for (const op of ops) {
      expect(irFile!.content).toContain(op);
    }
  });

  it("declares the seven public error classes in _errors.py", () => {
    const errFile = SOURCE_FILES.find((f) => f.path.endsWith("_errors.py"));
    expect(errFile).toBeDefined();
    const classes = [
      "class JitError",
      "class ShapeError",
      "class TorchAliasConflict",
      "class NoBackwardError",
      "class JitNotImplementedError",
      "class RealizationError",
      "class BufferTableError",
    ];
    for (const cls of classes) {
      expect(errFile!.content).toContain(cls);
    }
  });
});
