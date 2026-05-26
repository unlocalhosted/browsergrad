import { describe, expect, it, vi } from "vitest";
import {
  installGrad,
  GradInstallError,
  type GradTarget,
  type InstallOptions,
} from "../src/index";
import { SOURCE_FILES, MOUNT_ROOT } from "../src/python/index";

/**
 * Surface + installer tests for the grad package.
 *
 * Python correctness tests need Pyodide — they live in the browser-integration
 * suite (planned). What we can verify in node:
 *   1. The Python source modules are present and non-empty.
 *   2. installGrad writes the expected files to a mock fs target.
 *   3. installGrad falls back to exec-only mode when no fs is provided.
 *   4. The skipImportCheck option suppresses the verification exec.
 *   5. Errors thrown by the target surface as GradInstallError or the
 *      target's own error (we don't re-wrap exec errors).
 */

describe("public surface", () => {
  it("exports installGrad and GradInstallError", () => {
    expect(typeof installGrad).toBe("function");
    expect(typeof GradInstallError).toBe("function");
    expect(new GradInstallError("x").name).toBe("GradInstallError");
  });
});

describe("Python source bundle", () => {
  it("ships the documented module set in load order", () => {
    const paths = SOURCE_FILES.map((s) => s.path);
    expect(paths).toEqual([
      "browsergrad_grad/tensor.py",
      "browsergrad_grad/functional.py",
      "browsergrad_grad/nn.py",
      "browsergrad_grad/optim.py",
      "browsergrad_grad/torch_compat.py",
      "browsergrad_grad/utils/__init__.py",
      "browsergrad_grad/utils/data.py",
      "browsergrad_grad/__init__.py",
    ]);
  });

  it("every module has non-trivial content", () => {
    // Tiny passthrough packages (utils/__init__.py) get a small floor;
    // real modules have to clear 100.
    for (const file of SOURCE_FILES) {
      const floor = file.path.endsWith("utils/__init__.py") ? 30 : 100;
      expect(file.content.length).toBeGreaterThan(floor);
    }
  });

  it("tensor.py defines the Tensor class", () => {
    const tensor = SOURCE_FILES.find((s) => s.path.endsWith("tensor.py"));
    expect(tensor?.content).toContain("class Tensor:");
    expect(tensor?.content).toContain("def backward");
  });

  it("nn.py defines Module, Linear, Sequential", () => {
    const nn = SOURCE_FILES.find((s) => s.path.endsWith("nn.py"));
    expect(nn?.content).toContain("class Module");
    expect(nn?.content).toContain("class Linear(Module)");
    expect(nn?.content).toContain("class Sequential(Module)");
  });

  it("optim.py defines SGD", () => {
    const optim = SOURCE_FILES.find((s) => s.path.endsWith("optim.py"));
    expect(optim?.content).toContain("class SGD(Optimizer)");
  });

  it("functional.py defines v0.2 op set", () => {
    const fn = SOURCE_FILES.find((s) => s.path.endsWith("functional.py"));
    // v0.1 ops still present
    expect(fn?.content).toContain("def relu");
    expect(fn?.content).toContain("def sigmoid");
    expect(fn?.content).toContain("def tanh");
    expect(fn?.content).toContain("def mse_loss");
    // v0.2 additions
    expect(fn?.content).toContain("def leaky_relu");
    expect(fn?.content).toContain("def gelu");
    expect(fn?.content).toContain("def softmax");
    expect(fn?.content).toContain("def log_softmax");
    expect(fn?.content).toContain("def cross_entropy_loss");
    expect(fn?.content).toContain("def nll_loss");
  });

  it("nn.py defines v0.2 additions (LayerNorm, Embedding, activations)", () => {
    const nn = SOURCE_FILES.find((s) => s.path.endsWith("nn.py"));
    expect(nn?.content).toContain("class LayerNorm(Module)");
    expect(nn?.content).toContain("class Embedding(Module)");
    expect(nn?.content).toContain("class ReLU(Module)");
    expect(nn?.content).toContain("class GELU(Module)");
  });

  it("optim.py defines Adam and AdamW", () => {
    const optim = SOURCE_FILES.find((s) => s.path.endsWith("optim.py"));
    expect(optim?.content).toContain("class Adam(Optimizer)");
    expect(optim?.content).toContain("class AdamW(Optimizer)");
  });

  it("__init__.py declares v0.4.17 and exports no_grad / cat / stack / install_torch_alias / top-level math", () => {
    const init = SOURCE_FILES.find((s) => s.path === "browsergrad_grad/__init__.py");
    expect(init?.content).toContain('__version__ = "0.4.17"');
    expect(init?.content).toContain("no_grad");
    expect(init?.content).toContain("cat");
    expect(init?.content).toContain("stack");
    expect(init?.content).toContain("install_torch_alias");
    expect(init?.content).toContain("from_numpy");
    expect(init?.content).toContain("manual_seed");
    expect(init?.content).toContain("matmul");
  });

  it("ships torch_compat.py with the install_torch_alias function", () => {
    const torchCompat = SOURCE_FILES.find((s) => s.path.endsWith("torch_compat.py"));
    expect(torchCompat).toBeDefined();
    expect(torchCompat?.content).toContain("def install_torch_alias");
    expect(torchCompat?.content).toContain('sys.modules["torch"]');
  });

  it("optim.py defines v0.4.5 LR schedulers", () => {
    const optim = SOURCE_FILES.find((s) => s.path.endsWith("optim.py"));
    expect(optim?.content).toContain("class StepLR(_LRScheduler)");
    expect(optim?.content).toContain("class CosineAnnealingLR(_LRScheduler)");
  });

  it("nn.py defines v0.4.1 Conv1d / BatchNorm1d / Flatten", () => {
    const nn = SOURCE_FILES.find((s) => s.path.endsWith("nn.py"));
    expect(nn?.content).toContain("class Conv1d(Module)");
    expect(nn?.content).toContain("class BatchNorm1d(Module)");
    expect(nn?.content).toContain("class Flatten(Module)");
  });

  it("nn.py defines v0.3.3 Dropout family", () => {
    const nn = SOURCE_FILES.find((s) => s.path.endsWith("nn.py"));
    expect(nn?.content).toContain("class Dropout(Module)");
    expect(nn?.content).toContain("class Dropout2d(Module)");
  });

  it("nn.py defines v0.4 AdaptiveAvgPool2d and MultiHeadAttention", () => {
    const nn = SOURCE_FILES.find((s) => s.path.endsWith("nn.py"));
    expect(nn?.content).toContain("class AdaptiveAvgPool2d(Module)");
    expect(nn?.content).toContain("class MultiHeadAttention(Module)");
  });

  it("nn.py defines v0.3 Conv2d", () => {
    const nn = SOURCE_FILES.find((s) => s.path.endsWith("nn.py"));
    expect(nn?.content).toContain("class Conv2d(Module)");
  });

  it("nn.py defines v0.3.1 pooling layers", () => {
    const nn = SOURCE_FILES.find((s) => s.path.endsWith("nn.py"));
    expect(nn?.content).toContain("class MaxPool2d(Module)");
    expect(nn?.content).toContain("class AvgPool2d(Module)");
  });

  it("nn.py defines v0.3.2 BatchNorm2d + Module.train/eval flags", () => {
    const nn = SOURCE_FILES.find((s) => s.path.endsWith("nn.py"));
    expect(nn?.content).toContain("class BatchNorm2d(Module)");
    expect(nn?.content).toContain("self.training");
    expect(nn?.content).toContain("def train(self, mode");
    expect(nn?.content).toContain("def eval(self)");
  });
});

describe("installGrad — fs-path mode", () => {
  function makeFsTarget(): {
    target: GradTarget;
    fsWrite: ReturnType<typeof vi.fn>;
    exec: ReturnType<typeof vi.fn>;
  } {
    const fsWrite = vi.fn(async (_path: string, _content: string) => {});
    const exec = vi.fn(async (_opts: { code: string }) => {});
    const target: GradTarget = {
      exec,
      fs: { write: fsWrite },
    };
    return { target, fsWrite, exec };
  }

  it("writes each module to mountRoot via fs.write", async () => {
    const { target, fsWrite } = makeFsTarget();
    await installGrad(target);
    expect(fsWrite).toHaveBeenCalledTimes(SOURCE_FILES.length);
    const paths = fsWrite.mock.calls.map((c) => c[0] as string);
    for (const file of SOURCE_FILES) {
      expect(paths).toContain(`${MOUNT_ROOT}/${file.path}`);
    }
  });

  it("respects a custom mountRoot", async () => {
    const { target, fsWrite } = makeFsTarget();
    const opts: InstallOptions = { mountRoot: "/custom/path" };
    await installGrad(target, opts);
    for (const call of fsWrite.mock.calls) {
      expect((call[0] as string).startsWith("/custom/path/")).toBe(true);
    }
  });

  it("inserts the mount root into sys.path", async () => {
    const { target, exec } = makeFsTarget();
    await installGrad(target);
    // First exec call adds to sys.path; last call is the import check.
    const firstCode = exec.mock.calls[0]![0].code;
    expect(firstCode).toContain("sys.path.insert(0,");
    expect(firstCode).toContain(MOUNT_ROOT);
  });

  it("runs the import smoke check by default", async () => {
    const { target, exec } = makeFsTarget();
    await installGrad(target);
    const lastCode = exec.mock.calls[exec.mock.calls.length - 1]![0].code;
    expect(lastCode).toContain("import browsergrad_grad");
  });

  it("skipImportCheck suppresses the verification exec", async () => {
    const { target, exec } = makeFsTarget();
    await installGrad(target, { skipImportCheck: true });
    for (const call of exec.mock.calls) {
      expect(call[0].code).not.toContain("import browsergrad_grad as _bg_check");
    }
  });

  it("wraps fs.write failures as GradInstallError", async () => {
    const target: GradTarget = {
      exec: vi.fn(),
      fs: {
        write: vi.fn(async () => {
          throw new Error("disk full");
        }),
      },
    };
    await expect(installGrad(target)).rejects.toThrow(GradInstallError);
    await expect(installGrad(target)).rejects.toThrow(/disk full/);
  });
});

describe("installGrad — exec-only fallback", () => {
  it("works without fs by inlining files in a single exec", async () => {
    const exec = vi.fn(async (_opts: { code: string }) => {});
    const target: GradTarget = { exec };
    await installGrad(target, { skipImportCheck: true });
    // Single bootstrap exec (no import-check exec because we skipped it).
    expect(exec).toHaveBeenCalledTimes(1);
    const code = exec.mock.calls[0]![0].code;
    expect(code).toContain("os.makedirs");
    expect(code).toContain("sys.path.insert");
    // Source is base64-embedded; verify the decode call is present.
    expect(code).toContain('__import__("base64")');
  });
});
