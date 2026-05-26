/**
 * Trace cache integration tests (PRD-008 v0).
 *
 * The cache fires on `nn.Module.__call__` when:
 *   - Every positional arg is a TensorProxy.
 *   - No arg has `requires_grad=True`.
 *   - The signature (id(module), training, shape+dtype-tuple) matches
 *     a prior call.
 *
 * Asserts:
 *   - Hits return numerically identical output.
 *   - Repeated calls with the same signature register as cache hits.
 *   - A new shape triggers a miss + new entry.
 *   - `use_trace_cache(False)` disables the path entirely.
 *   - `clear_trace_cache()` resets state.
 *   - requires_grad=True inputs bypass the cache (autograd integrity).
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("PRD-008 trace cache", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    // Reset cache between tests so each starts from a known state.
    await target.run(`
import browsergrad_jit as bg
bg.jit.clear_trace_cache()
bg.jit.use_trace_cache(True)
`);
  });

  // The trace cache fires for inference graphs (no _ctx on the output).
  // A model whose parameters require_grad will produce an output with
  // _ctx (the backward closure for the matmul/add). Caching such an
  // output and reusing it on a later call would point the closures at
  // stale input proxies. v0 refuses; PRD-014's torch.func transforms
  // make this safe by making backward a graph rewrite rather than a
  // closure. Tests below run "inference mode" — disable parameter
  // gradients up-front.
  function inferenceSetup(): string {
    return `for p in m.parameters(): p.requires_grad = False\n`;
  }

  it("a second forward with same input shape is a cache hit", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ hits: number; misses: number; entries: number }>(`
import browsergrad_jit as bg
import numpy as np
bg.manual_seed(0)

m = bg.nn.Linear(4, 3)
${inferenceSetup()}
x = bg.tensor(np.zeros((2, 4), dtype=np.float32))

# First call: miss; second + third: hits.
_ = m(x)
_ = m(x)
_ = m(x)
bg.jit.trace_cache_stats()
`);
    expect(result.misses).toBe(1);
    expect(result.hits).toBe(2);
    expect(result.entries).toBe(1);
  });

  it("a different input shape produces a separate cache entry", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ hits: number; misses: number; entries: number }>(`
import browsergrad_jit as bg
import numpy as np
bg.manual_seed(0)
m = bg.nn.Linear(4, 3)
${inferenceSetup()}
_ = m(bg.tensor(np.zeros((2, 4), dtype=np.float32)))
_ = m(bg.tensor(np.zeros((8, 4), dtype=np.float32)))
_ = m(bg.tensor(np.zeros((2, 4), dtype=np.float32)))
bg.jit.trace_cache_stats()
`);
    expect(result.misses).toBe(2);
    expect(result.hits).toBe(1);
    expect(result.entries).toBe(2);
  });

  it("cached output is numerically identical to a fresh forward", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ max_diff: number }>(`
import browsergrad_jit as bg
import numpy as np
bg.manual_seed(0)

m = bg.nn.Sequential(
    bg.nn.Linear(4, 8),
    bg.nn.ReLU(),
    bg.nn.Linear(8, 2),
)
for p in m.parameters(): p.requires_grad = False
x_np = np.random.RandomState(0).randn(3, 4).astype(np.float32)

# Forward with cache enabled — first call misses, second hits.
bg.jit.use_trace_cache(True)
y_first = m(bg.tensor(x_np.copy())).numpy()
y_second = m(bg.tensor(x_np.copy())).numpy()

# Same forward with cache off — fresh trace each time.
bg.jit.use_trace_cache(False)
y_uncached = m(bg.tensor(x_np.copy())).numpy()
bg.jit.use_trace_cache(True)

{
    "max_diff": float(max(
        np.max(np.abs(y_first - y_uncached)),
        np.max(np.abs(y_second - y_uncached)),
    )),
}
`);
    expect(result.max_diff).toBeLessThan(1e-6);
  });

  it("disabling the cache stops counting hits", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ hits: number; misses: number }>(`
import browsergrad_jit as bg
import numpy as np
m = bg.nn.Linear(4, 3)
x = bg.tensor(np.zeros((2, 4), dtype=np.float32))

bg.jit.use_trace_cache(False)
_ = m(x); _ = m(x); _ = m(x)
bg.jit.use_trace_cache(True)
bg.jit.trace_cache_stats()
`);
    expect(result.hits).toBe(0);
    expect(result.misses).toBe(0);
  });

  it("requires_grad=True inputs bypass the cache (autograd integrity)", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ entries: number; hits: number }>(`
import browsergrad_jit as bg
import numpy as np
m = bg.nn.Linear(4, 3)
x = bg.from_numpy(np.zeros((2, 4), dtype=np.float32), requires_grad=True)
_ = m(x)
_ = m(x)
bg.jit.trace_cache_stats()
`);
    // Two calls, both bypass the cache → 0 hits, 0 misses, 0 entries.
    expect(result.entries).toBe(0);
    expect(result.hits).toBe(0);
  });

  it("clear_trace_cache resets entries + counters", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ before_entries: number; after_entries: number; after_hits: number }>(`
import browsergrad_jit as bg
import numpy as np
m = bg.nn.Linear(4, 3)
${inferenceSetup()}
x = bg.tensor(np.zeros((2, 4), dtype=np.float32))
_ = m(x); _ = m(x)
before = bg.jit.trace_cache_stats()
bg.jit.clear_trace_cache()
after = bg.jit.trace_cache_stats()
{
    "before_entries": before["entries"],
    "after_entries": after["entries"],
    "after_hits": after["hits"],
}
`);
    expect(result.before_entries).toBe(1);
    expect(result.after_entries).toBe(0);
    expect(result.after_hits).toBe(0);
  });

  it("train/eval toggles produce separate cache entries", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ before_first_repeat: number; after_first_repeat: number }>(`
import browsergrad_jit as bg
import numpy as np

# Use a plain Linear (no Dropout) so we can reason about a single
# module's cache behavior without composition-time recursion noise.
m = bg.nn.Linear(4, 3)
for p in m.parameters(): p.requires_grad = False
x = bg.tensor(np.zeros((2, 4), dtype=np.float32))

m.train()
_ = m(x)         # miss: train+(2,4)
m.eval()
_ = m(x)         # miss: eval+(2,4)
before_repeat = bg.jit.trace_cache_stats()["misses"]

m.train()
_ = m(x)         # hit
m.eval()
_ = m(x)         # hit
after_repeat = bg.jit.trace_cache_stats()["misses"]
{
    "before_first_repeat": before_repeat,
    "after_first_repeat": after_repeat,
}
`);
    expect(result.before_first_repeat).toBe(2);
    expect(result.after_first_repeat).toBe(2); // no new misses on hit pass
  });

  it("cache_stats aggregates trace cache state", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, any>>(`
import browsergrad_jit as bg
import numpy as np
m = bg.nn.Linear(4, 3)
${inferenceSetup()}
x = bg.tensor(np.zeros((2, 4), dtype=np.float32))
_ = m(x); _ = m(x)
bg.cache_stats()
`);
    expect(result.trace).toBeDefined();
    expect(result.trace.entries).toBe(1);
    expect(result.trace.hits).toBe(1);
  });

  it("refuses to cache when any parameter requires_grad (training mode safety)", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ entries: number }>(`
import browsergrad_jit as bg
import numpy as np
m = bg.nn.Linear(4, 3)
# Parameters require_grad by default — output will carry _ctx for backward.
# The cache must refuse, since the closures would point at stale inputs.
x = bg.tensor(np.zeros((2, 4), dtype=np.float32))
_ = m(x); _ = m(x); _ = m(x)
bg.jit.trace_cache_stats()
`);
    expect(result.entries).toBe(0);
  });
});
