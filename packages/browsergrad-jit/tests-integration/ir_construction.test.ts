/**
 * IR construction integration tests.
 *
 * Verifies the Python IR module loads cleanly under real Pyodide, the
 * opcode catalog is complete, UOp construction enforces the documented
 * invariants (shape validation, leaf/non-leaf arity), structural hashing
 * is stable, and toposort produces a leaves-first ordering.
 *
 * These are the foundational tests that Week 1 of PRD-005 ships. Every
 * subsequent week (Realizer, autograd, nn.Module, etc.) layers on top
 * of behavior asserted here. If these fail, no later test can pass.
 */

import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("browsergrad_jit._ir under real Pyodide", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("imports browsergrad_jit and reports the package version", async () => {
    const target = await getJitTarget();
    const version = await target.run<string>(`
import browsergrad_jit
browsergrad_jit.__version__
`);
    expect(version).toBe("0.3.0");
  });

  it("exposes 27 opcodes in ALL_OPS (23 core + 2 fusion + 2 autograd)", async () => {
    const target = await getJitTarget();
    const count = await target.run<number>(`
from browsergrad_jit._ir import ALL_OPS
len(ALL_OPS)
`);
    expect(count).toBe(27);
  });

  it("constructs a BUFFER leaf with the documented signature", async () => {
    const target = await getJitTarget();
    const summary = await target.run<{ op: string; shape: number[]; dtype: string; arg: string }>(`
from browsergrad_jit._ir import buffer
u = buffer("test:buf_0", (32, 128), "float32")
{"op": u.op, "shape": list(u.shape), "dtype": u.dtype, "arg": u.arg}
`);
    expect(summary.op).toBe("BUFFER");
    expect(summary.shape).toEqual([32, 128]);
    expect(summary.dtype).toBe("float32");
    expect(summary.arg).toBe("test:buf_0");
  });

  it("rejects unknown opcode strings at construction", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
from browsergrad_jit._ir import UOp
try:
    UOp(op="NOT_AN_OPCODE", inputs=(), shape=(), dtype="float32", arg=None)
    result = "no_error"
except ValueError as e:
    result = str(e)
result
`);
    expect(err).toMatch(/unknown opcode 'NOT_AN_OPCODE'/);
  });

  it("rejects leaf opcodes that carry inputs", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
from browsergrad_jit._ir import UOp, buffer
fake_input = buffer("test:x", (4,), "float32")
try:
    UOp(op="CONST", inputs=(fake_input,), shape=(), dtype="float32", arg={"value": 1.0})
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/ShapeError/);
    expect(err).toMatch(/leaf opcode/);
  });

  it("rejects non-leaf opcodes that have zero inputs", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
from browsergrad_jit._ir import UOp
try:
    UOp(op="ADD", inputs=(), shape=(4,), dtype="float32", arg=None)
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/ShapeError/);
    expect(err).toMatch(/at least one input/);
  });

  it("rejects negative dimensions in shape", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
from browsergrad_jit._ir import buffer
try:
    buffer("test:bad", (32, -1), "float32")
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/ShapeError/);
    expect(err).toMatch(/non-negative/);
  });

  it("refuses preposterously large allocations at construction", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
from browsergrad_jit._ir import buffer
try:
    # 10**12 elements — orders of magnitude past the 2^30 ceiling.
    buffer("test:huge", (10**12,), "float32")
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/ShapeError/);
    expect(err).toMatch(/ceiling/);
  });

  it("rejects unknown dtypes", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
from browsergrad_jit._ir import buffer
try:
    buffer("test:bad", (4,), "complex128")
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/unknown dtype/);
    expect(err).toMatch(/complex128/);
  });

  it("hashes structurally — two BUFFER UOps with the same id/shape/dtype compare equal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ eq: boolean; hash_eq: boolean }>(`
from browsergrad_jit._ir import buffer
a = buffer("test:b", (4,), "float32")
b = buffer("test:b", (4,), "float32")
{"eq": a == b, "hash_eq": hash(a) == hash(b)}
`);
    expect(result.eq).toBe(true);
    expect(result.hash_eq).toBe(true);
  });

  it("hashes structurally — UOps differing in arg compare unequal", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ eq: boolean }>(`
from browsergrad_jit._ir import buffer
a = buffer("test:b1", (4,), "float32")
b = buffer("test:b2", (4,), "float32")
{"eq": a == b}
`);
    expect(result.eq).toBe(false);
  });

  it("toposort produces a leaves-first ordering for a linear chain", async () => {
    const target = await getJitTarget();
    const ops = await target.run<string[]>(`
from browsergrad_jit._ir import UOp, buffer, load, toposort
buf = buffer("test:x", (4,), "float32")
lo = load(buf)
neg = UOp(op="NEG", inputs=(lo,), shape=(4,), dtype="float32", arg=None)
order = toposort(neg)
[u.op for u in order]
`);
    expect(ops).toEqual(["BUFFER", "LOAD", "NEG"]);
  });

  it("toposort handles a diamond DAG without duplicating shared subexpression", async () => {
    const target = await getJitTarget();
    const opsAndCount = await target.run<{ ops: string[]; n: number }>(`
from browsergrad_jit._ir import UOp, buffer, load, toposort
# x → load → (neg, exp) → add        ← classic diamond
buf = buffer("test:x", (4,), "float32")
lo = load(buf)
neg = UOp(op="NEG", inputs=(lo,), shape=(4,), dtype="float32", arg=None)
exp = UOp(op="EXP", inputs=(lo,), shape=(4,), dtype="float32", arg=None)
add = UOp(op="ADD", inputs=(neg, exp), shape=(4,), dtype="float32", arg=None)
order = toposort(add)
{"ops": [u.op for u in order], "n": len(order)}
`);
    // 5 nodes total: BUFFER, LOAD, NEG, EXP, ADD — the diamond LOAD must
    // appear exactly once even though two children reference it.
    expect(opsAndCount.n).toBe(5);
    expect(opsAndCount.ops[0]).toBe("BUFFER");
    expect(opsAndCount.ops[1]).toBe("LOAD");
    expect(opsAndCount.ops[opsAndCount.ops.length - 1]).toBe("ADD");
    expect(opsAndCount.ops.filter((o) => o === "LOAD").length).toBe(1);
  });

  it("toposort handles a deep linear chain without hitting recursion limits", async () => {
    const target = await getJitTarget();
    const n = await target.run<number>(`
from browsergrad_jit._ir import UOp, buffer, load, toposort
# 5000-deep chain of NEGs. Python's default recursion limit is 1000;
# the iterative toposort must clear this without raising RecursionError.
buf = buffer("test:x", (4,), "float32")
cur = load(buf)
for _ in range(5000):
    cur = UOp(op="NEG", inputs=(cur,), shape=(4,), dtype="float32", arg=None)
len(toposort(cur))
`);
    expect(n).toBe(5002);  // BUFFER + LOAD + 5000 NEGs
  });
});

describe("browsergrad_jit BufferTable lifecycle", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("registers an array and returns a session-scoped buffer_id", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ id: string; session_token: string }>(`
import numpy as np
from browsergrad_jit._buffer_table import BufferTable
bt = BufferTable()
arr = np.zeros((4,), dtype=np.float32)
bid = bt.new_buffer(arr, name="my_buf")
{"id": bid, "session_token": bt.session_token}
`);
    // Buffer id is "{session_token}:my_buf"
    expect(result.id).toBe(`${result.session_token}:my_buf`);
  });

  it("refuses lookups from a different session", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import numpy as np
from browsergrad_jit._buffer_table import BufferTable
a = BufferTable()
b = BufferTable()
arr = np.zeros((4,), dtype=np.float32)
bid = a.new_buffer(arr, name="x")
try:
    b.get(bid)  # buffer_id was minted by session a; b refuses
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/BufferTableError/);
    expect(err).toMatch(/belongs to session/);
  });

  it("refuses to register the same name twice in one session", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import numpy as np
from browsergrad_jit._buffer_table import BufferTable
bt = BufferTable()
bt.new_buffer(np.zeros((4,), dtype=np.float32), name="x")
try:
    bt.new_buffer(np.zeros((4,), dtype=np.float32), name="x")
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/BufferTableError/);
    expect(err).toMatch(/already registered/);
  });

  it("refuses shape-changing updates", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import numpy as np
from browsergrad_jit._buffer_table import BufferTable
bt = BufferTable()
bid = bt.new_buffer(np.zeros((4,), dtype=np.float32))
try:
    bt.update(bid, np.zeros((8,), dtype=np.float32))
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/BufferTableError/);
    expect(err).toMatch(/shape mismatch/);
  });
});

describe("browsergrad_jit.Session per-loop isolation", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("new_session() returns Sessions with distinct buffer tokens", async () => {
    const target = await getJitTarget();
    const distinct = await target.run<boolean>(`
import browsergrad_jit as jit
a = jit.new_session()
b = jit.new_session()
a.buffer_table.session_token != b.buffer_table.session_token
`);
    expect(distinct).toBe(true);
  });

  it("the default session is stable across get_default_session calls", async () => {
    const target = await getJitTarget();
    const same = await target.run<boolean>(`
import browsergrad_jit as jit
a = jit.get_default_session()
b = jit.get_default_session()
a is b
`);
    expect(same).toBe(true);
  });

  it("set_default_session swaps the implicit session", async () => {
    const target = await getJitTarget();
    const swapped = await target.run<boolean>(`
import browsergrad_jit as jit
original = jit.get_default_session()
fresh = jit.new_session()
jit.set_default_session(fresh)
ok = jit.get_default_session() is fresh and jit.get_default_session() is not original
jit.set_default_session(original)  # restore for hygiene
ok
`);
    expect(swapped).toBe(true);
  });
});

describe("browsergrad_jit.TensorProxy stub", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
  });

  it("exposes shape, dtype, ndim, len, repr without realization", async () => {
    const target = await getJitTarget();
    const meta = await target.run<{
      shape: number[];
      dtype: string;
      ndim: number;
      len: number;
      numel: number;
      repr: string;
    }>(`
from browsergrad_jit import TensorProxy
from browsergrad_jit._ir import buffer, load
buf = buffer("test:x", (3, 4), "float32")
proxy = TensorProxy(load(buf))
{
    "shape": list(proxy.shape),
    "dtype": proxy.dtype,
    "ndim": proxy.ndim,
    "len": len(proxy),
    "numel": proxy.numel(),
    "repr": repr(proxy),
}
`);
    expect(meta.shape).toEqual([3, 4]);
    expect(meta.dtype).toBe("float32");
    expect(meta.ndim).toBe(2);
    expect(meta.len).toBe(3);
    expect(meta.numel).toBe(12);
    expect(meta.repr).toMatch(/TensorProxy\(shape=\(3, 4\), dtype='float32'/);
    expect(meta.repr).toMatch(/op=LOAD/);
  });

  it(".data raises AttributeError with the migration hint", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
from browsergrad_jit import TensorProxy
from browsergrad_jit._ir import buffer, load
proxy = TensorProxy(load(buffer("test:x", (4,), "float32")))
try:
    _ = proxy.data
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/AttributeError/);
    expect(err).toMatch(/\.data is not available/);
    expect(err).toMatch(/Use \.numpy\(\)/);
  });

  it("__array__ raises RuntimeError when numpy tries to silently convert", async () => {
    const target = await getJitTarget();
    const err = await target.run<string>(`
import numpy as np
from browsergrad_jit import TensorProxy
from browsergrad_jit._ir import buffer, load
proxy = TensorProxy(load(buffer("test:x", (4,), "float32")))
try:
    np.asarray(proxy)
    result = "no_error"
except Exception as e:
    result = type(e).__name__ + ": " + str(e)
result
`);
    expect(err).toMatch(/RuntimeError/);
    expect(err).toMatch(/np\.ndarray implicitly/);
  });

  it("realization triggers return values for tensors built via factories", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ scalar: number; list: number[] }>(`
import browsergrad_jit as bg
t = bg.tensor([1.0, 2.0, 3.0])
{"scalar": t.sum().item(), "list": t.tolist()}
`);
    expect(result.scalar).toBeCloseTo(6.0, 5);
    expect(result.list).toEqual([1.0, 2.0, 3.0]);
  });

  it("size() returns a tuple when dim is None, an int when dim is provided", async () => {
    const target = await getJitTarget();
    const result = await target.run<{ all: number[]; first: number; last: number }>(`
from browsergrad_jit import TensorProxy
from browsergrad_jit._ir import buffer, load
proxy = TensorProxy(load(buffer("test:x", (3, 4, 5), "float32")))
{
    "all": list(proxy.size()),
    "first": proxy.size(0),
    "last": proxy.size(-1),
}
`);
    expect(result.all).toEqual([3, 4, 5]);
    expect(result.first).toBe(3);
    expect(result.last).toBe(5);
  });
});
