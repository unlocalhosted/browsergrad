/**
 * Lab harness primitive tests (PRD-013).
 *
 * The runtime's `browsergrad` module isn't available in pyodide-in-node
 * tests — the helpers fall through to structured-stdout output, which
 * we capture for verification.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("PRD-013 lab harness primitives", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("assert_pytorch_match returns True on a match", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
bg.lab.assert_pytorch_match("equal_arrays", a, b)
`);
    expect(ok).toBe(true);
  });

  it("assert_pytorch_match returns False on a mismatch", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
b = np.array([1.0, 5.0, 3.0], dtype=np.float32)
bg.lab.assert_pytorch_match("diverged", a, b, rtol=1e-6, atol=1e-6)
`);
    expect(ok).toBe(false);
  });

  it("assert_pytorch_match returns False on shape mismatch", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.zeros((3,), dtype=np.float32))
b = np.zeros((3, 1), dtype=np.float32)
bg.lab.assert_pytorch_match("shape_mismatch", a, b)
`);
    expect(ok).toBe(false);
  });

  it("assert_shape_match passes on correct shape", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit as bg
import numpy as np
a = bg.from_numpy(np.zeros((3, 4), dtype=np.float32))
bg.lab.assert_shape_match("ok_shape", a, (3, 4))
`);
    expect(ok).toBe(true);
  });

  it("assert_no_nan_inf returns False on NaN", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit as bg
import numpy as np
bad = bg.from_numpy(np.array([1.0, float("nan"), 3.0], dtype=np.float32))
bg.lab.assert_no_nan_inf("bad_tensor", bad)
`);
    expect(ok).toBe(false);
  });

  it("assert_no_nan_inf returns True on clean tensor", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit as bg
import numpy as np
good = bg.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
bg.lab.assert_no_nan_inf("clean_tensor", good)
`);
    expect(ok).toBe(true);
  });
});
