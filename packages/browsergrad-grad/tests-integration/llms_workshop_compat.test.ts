/**
 * Workshop-compatibility tests: every primitive needed by the
 * LLMs-from-scratch workshop (rasbt/LLMs-from-scratch) must work without
 * any adaptation from the student.
 *
 * Covers the 12 additions made for workshop readiness:
 *  1. Tensor.var(dim, keepdim, unbiased)
 *  2. Tensor.masked_fill(mask, value) / masked_fill_
 *  3. torch.triu / torch.tril
 *  4. torch.arange
 *  5. nn.ModuleList
 *  6. Module.register_buffer
 *  7. torch.multinomial
 *  8. Tensor.topk(k, dim, largest) — multi-dim
 *  9. Tensor.flatten(start_dim, end_dim)
 * 10. Tensor.contiguous()
 * 11. Tensor.numel()
 * 12. torch.pi / torch.inf constants
 */

import { beforeAll, describe, expect, it } from "vitest";
import { clearNamespace, getGradTarget } from "./pyodide-host";

let target: Awaited<ReturnType<typeof getGradTarget>>;

beforeAll(async () => {
  target = await getGradTarget();
}, 120_000);

async function reset(): Promise<void> {
  await clearNamespace(target);
}

const PRELUDE = `
import browsergrad_grad as grad
from browsergrad_grad.torch_compat import install_torch_alias
torch = install_torch_alias()
import numpy as np
`;

// ─── 1. var ────────────────────────────────────────────────

describe("Tensor.var", () => {
  beforeAll(reset);

  it("unbiased=False (population var) matches numpy", async () => {
    const r = await target.run<{ val: number; grad_ok: boolean }>(`
${PRELUDE}
x = grad.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
v = x.var(dim=-1, keepdim=True, unbiased=False)
v.sum().backward()
{
  "val": float(v.numpy()[0, 0]),
  "grad_ok": x.grad is not None,
}
`);
    expect(r.val).toBeCloseTo(2 / 3, 5);
    expect(r.grad_ok).toBe(true);
  });

  it("unbiased=True (Bessel-corrected) matches numpy", async () => {
    const r = await target.run<{ val: number }>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0])
{"val": float(x.var(unbiased=True).item())}
`);
    expect(r.val).toBeCloseTo(1.0, 5);
  });

  it("torch.Tensor.var works via torch alias", async () => {
    const r = await target.run<{ val: number }>(`
${PRELUDE}
x = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
{"val": float(x.var(unbiased=True).item())}
`);
    expect(r.val).toBeCloseTo(4.571428, 4);
  });
});

// ─── 2. masked_fill ────────────────────────────────────────

describe("Tensor.masked_fill", () => {
  beforeAll(reset);

  it("fills masked positions with -inf (causal mask pattern)", async () => {
    const r = await target.run<{ shape: number[]; has_inf: boolean }>(`
${PRELUDE}
scores = grad.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
mask = grad.triu(grad.ones(2, 3), diagonal=1).bool()
out = scores.masked_fill(mask, float("-inf"))
{"shape": list(out.shape), "has_inf": float(out.numpy()[0, 1]) == float("-inf")}
`);
    expect(r.shape).toEqual([2, 3]);
    expect(r.has_inf).toBe(true);
  });

  it("gradient does not flow through masked positions", async () => {
    const r = await target.run<{ grad_at_masked: number; grad_at_unmasked: number }>(`
${PRELUDE}
x = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
mask = grad.Tensor([False, True, False])
out = x.masked_fill(mask, 0.0)
out.sum().backward()
{"grad_at_masked": float(x.grad.numpy()[1]), "grad_at_unmasked": float(x.grad.numpy()[0])}
`);
    expect(r.grad_at_masked).toBe(0);
    expect(r.grad_at_unmasked).toBe(1);
  });
});

// ─── 3. triu / tril ────────────────────────────────────────

describe("torch.triu / torch.tril", () => {
  beforeAll(reset);

  it("triu creates upper-triangular mask", async () => {
    const r = await target.run<{ values: number[][] }>(`
${PRELUDE}
m = torch.triu(torch.ones(3, 3), diagonal=1)
{"values": m.tolist()}
`);
    expect(r.values).toEqual([
      [0, 1, 1],
      [0, 0, 1],
      [0, 0, 0],
    ]);
  });

  it("tril creates lower-triangular causal mask", async () => {
    const r = await target.run<{ v00: number; v01: number; v11: number }>(`
${PRELUDE}
m = torch.tril(torch.ones(3, 3))
a = m.tolist()
{"v00": a[0][0], "v01": a[0][1], "v11": a[1][1]}
`);
    expect(r.v00).toBe(1);
    expect(r.v01).toBe(0);
    expect(r.v11).toBe(1);
  });
});

// ─── 4. arange ─────────────────────────────────────────────

describe("torch.arange", () => {
  beforeAll(reset);

  it("returns int64 for integer range", async () => {
    const r = await target.run<{ vals: number[]; dtype: string }>(`
${PRELUDE}
t = torch.arange(5)
{"vals": t.tolist(), "dtype": t.dtype}
`);
    expect(r.vals).toEqual([0, 1, 2, 3, 4]);
    expect(r.dtype).toBe("int64");
  });

  it("works as embedding indices", async () => {
    const r = await target.run<{ shape: number[] }>(`
${PRELUDE}
emb = torch.nn.Embedding(10, 4)
pos = torch.arange(5)
out = emb(pos)
{"shape": list(out.shape)}
`);
    expect(r.shape).toEqual([5, 4]);
  });
});

// ─── 5. ModuleList ─────────────────────────────────────────

describe("nn.ModuleList", () => {
  beforeAll(reset);

  it("parameters() yields from all child modules", async () => {
    const r = await target.run<{ n_params: number }>(`
${PRELUDE}
heads = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
{"n_params": sum(1 for _ in heads.parameters())}
`);
    // 3 Linear heads × 2 params (weight + bias) = 6
    expect(r.n_params).toBe(6);
  });

  it("supports iteration and indexing", async () => {
    const r = await target.run<{ len: number; shape: number[] }>(`
${PRELUDE}
heads = torch.nn.ModuleList([torch.nn.Linear(4, 2), torch.nn.Linear(4, 2)])
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
out = heads[0](x)
{"len": len(heads), "shape": list(out.shape)}
`);
    expect(r.len).toBe(2);
    expect(r.shape).toEqual([1, 2]);
  });

  it("append works after construction", async () => {
    const r = await target.run<{ n: number }>(`
${PRELUDE}
ml = torch.nn.ModuleList()
ml.append(torch.nn.Linear(2, 2))
ml.append(torch.nn.Linear(2, 2))
{"n": len(ml)}
`);
    expect(r.n).toBe(2);
  });
});

// ─── 6. register_buffer ────────────────────────────────────

describe("Module.register_buffer", () => {
  beforeAll(reset);

  it("buffer accessible as attribute but not in parameters()", async () => {
    const r = await target.run<{ param_count: number; mask_shape: number[] }>(`
${PRELUDE}
class MyAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
        self.register_buffer("mask", mask)
    def forward(self, x):
        return x

m = MyAttn()
{
  "param_count": sum(1 for _ in m.parameters()),
  "mask_shape": list(m.mask.shape),
}
`);
    // proj has weight + bias = 2 params; mask is NOT a param
    expect(r.param_count).toBe(2);
    expect(r.mask_shape).toEqual([5, 5]);
  });

  it("buffer survives state_dict / load_state_dict round-trip", async () => {
    const r = await target.run<{ mask_sum: number }>(`
${PRELUDE}
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias_buf", torch.ones(3))
    def forward(self, x): return x

m = M()
sd = m.state_dict()
m2 = M()
m2.load_state_dict(sd)
{"mask_sum": float(m2.bias_buf.sum().item())}
`);
    expect(r.mask_sum).toBeCloseTo(3.0, 5);
  });
});

// ─── 7. multinomial ────────────────────────────────────────

describe("torch.multinomial", () => {
  beforeAll(reset);

  it("samples from a 1-D probability tensor", async () => {
    const r = await target.run<{ shape: number[]; in_range: boolean }>(`
${PRELUDE}
torch.manual_seed(0)
probs = torch.tensor([0.1, 0.5, 0.4])
idx = torch.multinomial(probs, num_samples=10, replacement=True)
{"shape": list(idx.shape), "in_range": all(0 <= v <= 2 for v in idx.tolist())}
`);
    expect(r.shape).toEqual([10]);
    expect(r.in_range).toBe(true);
  });

  it("temperature-sampling pattern (logits → softmax → multinomial)", async () => {
    const r = await target.run<{ shape: number[] }>(`
${PRELUDE}
torch.manual_seed(42)
logits = torch.randn(1, 50257)
temperature = 1.0
probs = torch.nn.functional.softmax(logits[0] / temperature, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
{"shape": list(idx_next.shape)}
`);
    expect(r.shape).toEqual([1]);
  });
});

// ─── 8. topk multi-dim ─────────────────────────────────────

describe("Tensor.topk — multi-dim", () => {
  beforeAll(reset);

  it("top-k along last dim of 2-D tensor", async () => {
    const r = await target.run<{ val_shape: number[]; idx_shape: number[] }>(`
${PRELUDE}
logits = torch.randn(4, 50257)
values, indices = logits.topk(10, dim=-1)
{"val_shape": list(values.shape), "idx_shape": list(indices.shape)}
`);
    expect(r.val_shape).toEqual([4, 10]);
    expect(r.idx_shape).toEqual([4, 10]);
  });

  it("torch.topk functional form", async () => {
    const r = await target.run<{ vals: number[] }>(`
${PRELUDE}
x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
vals, _ = torch.topk(x, 3)
{"vals": vals.tolist()}
`);
    expect(r.vals[0]).toBeCloseTo(5, 4);
    expect(r.vals[1]).toBeCloseTo(4, 4);
    expect(r.vals[2]).toBeCloseTo(3, 4);
  });
});

// ─── 9. flatten ────────────────────────────────────────────

describe("Tensor.flatten", () => {
  beforeAll(reset);

  it("flattens batch and seq dims for cross-entropy (workshop pattern)", async () => {
    const r = await target.run<{ shape: number[] }>(`
${PRELUDE}
# logits shape: (batch=2, seq=4, vocab=100)
logits = torch.randn(2, 4, 100)
flat = logits.flatten(0, 1)
{"shape": list(flat.shape)}
`);
    expect(r.shape).toEqual([8, 100]);
  });

  it("default flatten collapses all dims except batch", async () => {
    const r = await target.run<{ shape: number[] }>(`
${PRELUDE}
x = torch.randn(3, 4, 5)
{"shape": list(x.flatten(1).shape)}
`);
    expect(r.shape).toEqual([3, 20]);
  });
});

// ─── 10. contiguous ────────────────────────────────────────

describe("Tensor.contiguous", () => {
  beforeAll(reset);

  it("is a no-op that preserves shape (multi-head view pattern)", async () => {
    const r = await target.run<{ shape: number[] }>(`
${PRELUDE}
x = torch.randn(2, 4, 8)
# Workshop pattern: transpose then contiguous then view
out = x.transpose(1, 2).contiguous().view(2, 4, 8)
{"shape": list(out.shape)}
`);
    expect(r.shape).toEqual([2, 4, 8]);
  });
});

// ─── 11. numel ─────────────────────────────────────────────

describe("Tensor.numel", () => {
  beforeAll(reset);

  it("returns total element count", async () => {
    const r = await target.run<{ n: number }>(`
${PRELUDE}
x = torch.randn(3, 4, 5)
{"n": x.numel()}
`);
    expect(r.n).toBe(60);
  });
});

// ─── 12. torch.pi / torch.inf ──────────────────────────────

describe("torch.pi and torch.inf", () => {
  beforeAll(reset);

  it("torch.pi matches math.pi", async () => {
    const r = await target.run<{ pi: number }>(`
${PRELUDE}
import math
{"pi": torch.pi}
`);
    expect(r.pi).toBeCloseTo(Math.PI, 10);
  });

  it("torch.inf is positive infinity", async () => {
    const r = await target.run<{ is_inf: boolean }>(`
${PRELUDE}
{"is_inf": torch.inf == float("inf")}
`);
    expect(r.is_inf).toBe(true);
  });

  it("GELU formula using torch.pi compiles correctly", async () => {
    const r = await target.run<{ shape: number[]; finite: boolean }>(`
${PRELUDE}
import math
x = torch.tensor([-1.0, 0.0, 1.0])
# Exact formula from LLMs-from-scratch ch04
gelu = 0.5 * x * (1 + torch.nn.functional.tanh(
    math.sqrt(2.0 / torch.pi) * (x + 0.044715 * x ** 3)
))
arr = gelu.numpy()
{"shape": list(gelu.shape), "finite": bool(np.all(np.isfinite(arr)))}
`);
    expect(r.shape).toEqual([3]);
    expect(r.finite).toBe(true);
  });
});

// ─── End-to-end: mini GPT forward pass ─────────────────────

describe("end-to-end: mini GPT forward pass", () => {
  beforeAll(reset);

  it("CausalAttention + LayerNorm + FeedForward forward pass runs without error", async () => {
    const r = await target.run<{ loss_finite: boolean; loss_positive: boolean }>(`
${PRELUDE}
import math

# Minimal GPT config
vocab_size = 256
ctx_len    = 16
emb_dim    = 32
n_heads    = 4
head_dim   = emb_dim // n_heads

class CausalAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_k = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_v = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_proj = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = torch.nn.Dropout(p=0.0)
        mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, t, d = x.shape
        q = self.W_q(x).view(b, t, n_heads, head_dim).transpose(1, 2)
        k = self.W_k(x).view(b, t, n_heads, head_dim).transpose(1, 2)
        v = self.W_v(x).view(b, t, n_heads, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(2, 3)) * scale
        attn = attn.masked_fill(self.mask[:t, :t], float("-inf"))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)

class GPTBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(emb_dim)
        self.attn = CausalAttention()
        self.ln2 = torch.nn.LayerNorm(emb_dim)
        self.ff  = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4 * emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = torch.nn.Embedding(ctx_len, emb_dim)
        self.blocks  = torch.nn.ModuleList([GPTBlock() for _ in range(2)])
        self.ln_f    = torch.nn.LayerNorm(emb_dim)
        self.head    = torch.nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

torch.manual_seed(0)
model = MiniGPT()
opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 2
idx     = grad.Tensor(np.random.randint(0, vocab_size, (batch_size, ctx_len)), dtype="int64")
targets = grad.Tensor(np.random.randint(0, vocab_size, (batch_size, ctx_len)), dtype="int64")

opt.zero_grad()
logits = model(idx)
loss   = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), targets.flatten(0, 1)
)
loss.backward()
opt.step()

loss_val = float(loss.item())
import numpy as _np
{
  "loss_finite":   bool(_np.isfinite(loss_val)),
  "loss_positive": loss_val > 0,
}
`);
    expect(r.loss_finite).toBe(true);
    expect(r.loss_positive).toBe(true);
  });
});
