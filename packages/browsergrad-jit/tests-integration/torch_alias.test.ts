/**
 * install_torch_alias / uninstall_torch_alias integration tests.
 *
 * Verifies the owner-token protocol per the PRD-005 critique:
 *  - install() is idempotent (re-installing by the same owner is a no-op)
 *  - the torch namespace gets browsergrad_jit's full public surface
 *  - conflict detection refuses to overwrite another owner without force=True
 *  - uninstall() releases cleanly and is safe to call when no alias is set
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("install_torch_alias", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    // Always start each test from a clean slate.
    await target.run(`
import browsergrad_jit
browsergrad_jit.uninstall_torch_alias()
`);
  });

  it("registers torch / torch.nn / torch.nn.functional / torch.optim", async () => {
    const target = await getJitTarget();
    const result = await target.run<{
      hasTorch: boolean;
      hasNn: boolean;
      hasFunc: boolean;
      hasOptim: boolean;
      owner: string;
    }>(`
import browsergrad_jit
browsergrad_jit.install_torch_alias()
import sys
result = {
    "hasTorch": "torch" in sys.modules,
    "hasNn": "torch.nn" in sys.modules,
    "hasFunc": "torch.nn.functional" in sys.modules,
    "hasOptim": "torch.optim" in sys.modules,
    "owner": getattr(sys.modules["torch"], "__bg_owner__", "<none>"),
}
result
`);
    expect(result.hasTorch).toBe(true);
    expect(result.hasNn).toBe(true);
    expect(result.hasFunc).toBe(true);
    expect(result.hasOptim).toBe(true);
    expect(result.owner).toBe("browsergrad_jit");
  });

  it("import torch resolves to browsergrad_jit", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ ver: string; sumValue: number }>(`
import browsergrad_jit
browsergrad_jit.install_torch_alias()
import torch
import torch.nn as nn
import torch.nn.functional as F
t = torch.tensor([1.0, 2.0, 3.0, 4.0])
{"ver": torch.__version__, "sumValue": float(t.sum().item())}
`);
    expect(result.ver).toBe("0.6.0");
    expect(result.sumValue).toBe(10.0);
  });

  it("is idempotent — re-installing returns cleanly", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit
browsergrad_jit.install_torch_alias()
browsergrad_jit.install_torch_alias()
browsergrad_jit.install_torch_alias()
import sys
"torch" in sys.modules
`);
    expect(ok).toBe(true);
  });

  it("uninstall removes the alias and is safe to call twice", async () => {
    const target = await getJitTarget();
    const states = await target.run<{ after_install: boolean; after_uninstall: boolean }>(`
import browsergrad_jit
import sys
browsergrad_jit.install_torch_alias()
after_install = "torch" in sys.modules
browsergrad_jit.uninstall_torch_alias()
browsergrad_jit.uninstall_torch_alias()  # double-uninstall safe
after_uninstall = "torch" in sys.modules
{"after_install": after_install, "after_uninstall": after_uninstall}
`);
    expect(states.after_install).toBe(true);
    expect(states.after_uninstall).toBe(false);
  });

  it("refuses to shadow a foreign owner without force=True", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import browsergrad_jit
import sys
import types
# Plant a fake foreign torch ownership marker.
fake = types.ModuleType("torch")
fake.__bg_owner__ = "some_other_package"
sys.modules["torch"] = fake
try:
    browsergrad_jit.install_torch_alias()
    result = "no_error"
except browsergrad_jit.TorchAliasConflict as e:
    result = str(e)
finally:
    sys.modules.pop("torch", None)
result
`);
    expect(err).toMatch(/some_other_package/);
    expect(err).toMatch(/force=True/);
  });

  it("force=True overrides the conflict for tests", async () => {
    const target = await getJitTarget();
    const ok = await target.run<boolean>(`
import browsergrad_jit
import sys
import types
fake = types.ModuleType("torch")
fake.__bg_owner__ = "some_other_package"
sys.modules["torch"] = fake
browsergrad_jit.install_torch_alias(force=True)
ok = getattr(sys.modules["torch"], "__bg_owner__", None) == "browsergrad_jit"
sys.modules.pop("torch", None)
ok
`);
    expect(ok).toBe(true);
  });
});
