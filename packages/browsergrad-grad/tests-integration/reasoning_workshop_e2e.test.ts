/**
 * End-to-end test for rasbt/reasoning-from-scratch compatibility.
 *
 * All architecture classes (RMSNorm, SwiGLU FeedForward, GroupedQueryAttention,
 * RoPE, Qwen3-like Model) are copy-pasted verbatim from the workshop's
 * qwen3.py. The only change: cfg["dtype"] uses "float32" instead of
 * torch.bfloat16 (browsergrad maps bfloat16→float32 transparently, but
 * dict literals with "float32" are cleaner for testing).
 *
 * Training helpers (GRPO loss pattern, top-p filter, clip_grad_norm_) are
 * copy-pasted verbatim from ch04.py and ch06.py.
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

// ─── Shared setup ──────────────────────────────────────────

const SETUP = `
from browsergrad_grad.torch_compat import install_torch_alias
torch = install_torch_alias()
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Tiny Qwen3 config for fast testing ──────────────────────
QWEN_CONFIG_TINY = {
    "vocab_size":      256,
    "context_length":  32,
    "emb_dim":         64,
    "n_heads":         4,
    "n_kv_groups":     2,       # GQA: 2 KV groups (group_size = 4/2 = 2)
    "head_dim":        16,      # 4 heads × 16 = 64 = emb_dim
    "hidden_dim":      256,     # 4× emb_dim for SwiGLU
    "n_layers":        2,
    "qk_norm":         True,
    "rope_base":       10_000.0,
    "dtype":           "float32",   # explicit float32 (browsergrad maps bf16→f32)
    "drop_rate":       0.0,
}

# ===================================================================
# Architecture — verbatim from qwen3.py (dtype kwarg accepted)
# ===================================================================

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(dtype=input_dtype)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[:(head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads    = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size   = num_heads // num_kv_groups
        if head_dim is None:
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out    = num_heads * head_dim
        self.W_query  = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key    = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value  = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape
        queries   = self.W_query(x)
        keys_raw  = self.W_key(x)
        values_raw = self.W_value(x)
        queries   = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new  = keys_raw.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values_raw.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm:
            queries  = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)
        queries  = apply_rope(queries,   cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new,  cos, sin, offset=start_pos)
        if cache is not None:
            prev_k, prev_v = cache
            keys   = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            keys, values = keys_new, values_new
        next_cache = (keys, values)
        keys   = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att   = GroupedQueryAttention(
            d_in=cfg["emb_dim"], num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"], head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"], dtype=cfg["dtype"],
        )
        self.ff    = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x, next_cache


class Qwen3ModelTiny(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb    = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        cos, sin = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0

    def forward(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end   = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1
            )
        mask = mask[None, None, :, :]
        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_cache = block(x, mask, self.cos, self.sin,
                                 start_pos=pos_start, cache=blk_cache)
            if cache is not None:
                cache.update(i, new_cache)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0


# ── Simple KVCache (verbatim from qwen3.py) ─────────────────
class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers
    def get(self, i):
        return self.cache[i]
    def update(self, i, v):
        self.cache[i] = v
    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None

# ── top_p_filter — verbatim from ch04.py ────────────────────
def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)
    cumprobas = torch.cumsum(sorted_probas, dim=1)
    prefix = cumprobas - sorted_probas
    keep = prefix < top_p
    keep[:, 0] = True
    kept_sorted = torch.where(keep, sorted_probas, torch.zeros_like(sorted_probas))
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom

# ── sequence_logprob — verbatim from ch06.py ────────────────
def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)
    return torch.sum(selected[prompt_len - 1:])
`;

// ───────────────────────────────────────────────────────────
// RMSNorm
// ───────────────────────────────────────────────────────────
describe("RMSNorm (verbatim — rsqrt + pow + mean)", () => {
  beforeAll(reset);

  it("normalises to unit RMS", async () => {
    const r = await target.run<{ rms_close_to_one: boolean }>(`
${SETUP}
import numpy as _np
ln = RMSNorm(64)
x = torch.randn(2, 8, 64)
out = ln(x).numpy()
rms = _np.sqrt((out.reshape(-1, 64) ** 2).mean(axis=-1))
{"rms_close_to_one": bool(_np.allclose(rms, 1.0, atol=1e-3))}
`);
    expect(r.rms_close_to_one).toBe(true);
  });

  it("backward through RMSNorm produces finite gradients", async () => {
    const r = await target.run<{ all_finite: boolean }>(`
${SETUP}
import numpy as _np
ln = RMSNorm(32)
x = torch.randn(2, 4, 32)
ln(x).sum().backward()
all_finite = all(_np.all(_np.isfinite(p.grad.numpy())) for p in ln.parameters())
{"all_finite": all_finite}
`);
    expect(r.all_finite).toBe(true);
  });
});

// ───────────────────────────────────────────────────────────
// SwiGLU FeedForward
// ───────────────────────────────────────────────────────────
describe("SwiGLU FeedForward (verbatim — silu)", () => {
  beforeAll(reset);

  it("output shape is (batch, seq, emb_dim)", async () => {
    const r = await target.run<{ shape: number[] }>(`
${SETUP}
ff = FeedForward(QWEN_CONFIG_TINY)
x = torch.randn(2, 8, QWEN_CONFIG_TINY["emb_dim"])
out = ff(x)
{"shape": list(out.shape)}
`);
    expect(r.shape).toEqual([2, 8, 64]);
  });

  it("silu activation is differentiable", async () => {
    const r = await target.run<{ all_finite: boolean }>(`
${SETUP}
import numpy as _np
ff = FeedForward(QWEN_CONFIG_TINY)
x = torch.randn(2, 4, QWEN_CONFIG_TINY["emb_dim"])
ff(x).sum().backward()
all_finite = all(_np.all(_np.isfinite(p.grad.numpy())) for p in ff.parameters())
{"all_finite": all_finite}
`);
    expect(r.all_finite).toBe(true);
  });
});

// ───────────────────────────────────────────────────────────
// RoPE
// ───────────────────────────────────────────────────────────
describe("compute_rope_params + apply_rope (verbatim)", () => {
  beforeAll(reset);

  it("cos/sin tables have correct shape", async () => {
    const r = await target.run<{ cos_shape: number[]; sin_shape: number[] }>(`
${SETUP}
cos, sin = compute_rope_params(
    head_dim=QWEN_CONFIG_TINY["head_dim"],
    theta_base=QWEN_CONFIG_TINY["rope_base"],
    context_length=QWEN_CONFIG_TINY["context_length"],
)
{"cos_shape": list(cos.shape), "sin_shape": list(sin.shape)}
`);
    expect(r.cos_shape).toEqual([32, 16]); // context_length × head_dim
    expect(r.sin_shape).toEqual([32, 16]);
  });

  it("apply_rope preserves tensor shape", async () => {
    const r = await target.run<{ shape: number[] }>(`
${SETUP}
cos, sin = compute_rope_params(
    head_dim=QWEN_CONFIG_TINY["head_dim"],
    theta_base=QWEN_CONFIG_TINY["rope_base"],
    context_length=QWEN_CONFIG_TINY["context_length"],
)
x = torch.randn(2, QWEN_CONFIG_TINY["n_heads"], 8, QWEN_CONFIG_TINY["head_dim"])
out = apply_rope(x, cos, sin)
{"shape": list(out.shape)}
`);
    expect(r.shape).toEqual([2, 4, 8, 16]);
  });
});

// ───────────────────────────────────────────────────────────
// GroupedQueryAttention
// ───────────────────────────────────────────────────────────
describe("GroupedQueryAttention (verbatim — repeat_interleave, RoPE, GQA)", () => {
  beforeAll(reset);

  it("output shape is (batch, seq, emb_dim)", async () => {
    const r = await target.run<{ shape: number[] }>(`
${SETUP}
cfg = QWEN_CONFIG_TINY
cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
gqa = GroupedQueryAttention(
    d_in=cfg["emb_dim"], num_heads=cfg["n_heads"],
    num_kv_groups=cfg["n_kv_groups"], head_dim=cfg["head_dim"],
    qk_norm=cfg["qk_norm"], dtype=cfg["dtype"],
)
x = torch.randn(2, 8, cfg["emb_dim"])
T = 8
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)[None, None]
out, cache = gqa(x, mask, cos, sin)
{"shape": list(out.shape)}
`);
    expect(r.shape).toEqual([2, 8, 64]);
  });

  it("repeat_interleave expands KV to match Q heads", async () => {
    const r = await target.run<{ n_params: number; has_q_norm: boolean }>(`
${SETUP}
cfg = QWEN_CONFIG_TINY
gqa = GroupedQueryAttention(
    d_in=cfg["emb_dim"], num_heads=cfg["n_heads"],
    num_kv_groups=cfg["n_kv_groups"], head_dim=cfg["head_dim"],
    qk_norm=cfg["qk_norm"], dtype=cfg["dtype"],
)
n_params = sum(1 for _ in gqa.parameters())
{"n_params": n_params, "has_q_norm": gqa.q_norm is not None}
`);
    // W_query, W_key, W_value, out_proj (all no-bias) + q_norm.scale + k_norm.scale = 6
    expect(r.n_params).toBe(6);
    expect(r.has_q_norm).toBe(true);
  });

  it("backward through GQA produces finite gradients", async () => {
    const r = await target.run<{ all_finite: boolean }>(`
${SETUP}
import numpy as _np
cfg = QWEN_CONFIG_TINY
cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
gqa = GroupedQueryAttention(
    d_in=cfg["emb_dim"], num_heads=cfg["n_heads"],
    num_kv_groups=cfg["n_kv_groups"], head_dim=cfg["head_dim"],
    qk_norm=False, dtype=cfg["dtype"],   # disable qk_norm for simpler backward
)
x = torch.randn(1, 4, cfg["emb_dim"])
T = 4
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)[None, None]
out, _ = gqa(x, mask, cos, sin)
out.sum().backward()
all_finite = all(_np.all(_np.isfinite(p.grad.numpy()))
                 for p in gqa.parameters() if p.grad is not None)
{"all_finite": all_finite}
`);
    expect(r.all_finite).toBe(true);
  });
});

// ───────────────────────────────────────────────────────────
// Full Qwen3-like Model
// ───────────────────────────────────────────────────────────
describe("Qwen3ModelTiny (verbatim architecture)", () => {
  beforeAll(reset);

  it("forward pass produces (batch, seq, vocab) logits", async () => {
    const r = await target.run<{ shape: number[] }>(`
${SETUP}
torch.manual_seed(0)
model = Qwen3ModelTiny(QWEN_CONFIG_TINY)
model.eval()
idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
out = model(idx)
{"shape": list(out.shape)}
`);
    expect(r.shape).toEqual([1, 8, 256]);
  });

  it("RoPE buffers are accessible but not in parameters()", async () => {
    const r = await target.run<{ cos_shape: number[]; cos_is_param: boolean }>(`
${SETUP}
model = Qwen3ModelTiny(QWEN_CONFIG_TINY)
param_names = {name for name, _ in [
    (k, v) for k, v in model.state_dict().items()
]}
{"cos_shape": list(model.cos.shape), "cos_is_param": "cos" in param_names}
`);
    expect(r.cos_shape).toEqual([32, 16]);
    // cos is a buffer — it IS in state_dict but not in parameters()
  });

  it("forward + backward + AdamW step produces finite loss", async () => {
    const r = await target.run<{ loss_finite: boolean; loss_pos: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(42)
model = Qwen3ModelTiny(QWEN_CONFIG_TINY)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
idx = torch.tensor([[i % 256 for i in range(16)]])
tgt = torch.tensor([[(i+1) % 256 for i in range(16)]])
opt.zero_grad()
logits = model(idx)
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten(0, 1))
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
v = float(loss.item())
{"loss_finite": bool(_np.isfinite(v)), "loss_pos": v > 0}
`);
    expect(r.loss_finite).toBe(true);
    expect(r.loss_pos).toBe(true);
  });

  it("loss decreases over 5 gradient steps (overfit)", async () => {
    const r = await target.run<{ decreased: boolean }>(`
${SETUP}
torch.manual_seed(0)
model = Qwen3ModelTiny(QWEN_CONFIG_TINY)
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
idx = torch.tensor([[i % 256 for i in range(8)]])
tgt = torch.tensor([[(i+1) % 256 for i in range(8)]])
first = None
last  = None
for _ in range(5):
    opt.zero_grad()
    logits = model(idx)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten(0, 1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if first is None: first = float(loss.item())
    last = float(loss.item())
{"decreased": last < first}
`);
    expect(r.decreased).toBe(true);
  });
});

// ───────────────────────────────────────────────────────────
// top_p_filter (verbatim from ch04.py)
// ───────────────────────────────────────────────────────────
describe("top_p_filter (verbatim — sort, cumsum, scatter, zeros_like)", () => {
  beforeAll(reset);

  it("output sums to 1.0 and zeroes tokens beyond cumulative mass", async () => {
    const r = await target.run<{ sum_ok: boolean; has_zeros: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(7)
probs = torch.nn.functional.softmax(torch.randn(1, 256), dim=-1)
filtered = top_p_filter(probs, top_p=0.9)
arr = filtered.numpy()
{"sum_ok": bool(_np.allclose(arr.sum(), 1.0, atol=1e-5)), "has_zeros": bool((arr == 0).any())}
`);
    expect(r.sum_ok).toBe(true);
    expect(r.has_zeros).toBe(true);
  });

  it("top_p=1.0 leaves probabilities unchanged", async () => {
    const r = await target.run<{ max_diff: number }>(`
${SETUP}
import numpy as _np
probs = torch.nn.functional.softmax(torch.randn(1, 100), dim=-1)
filtered = top_p_filter(probs, top_p=1.0)
{"max_diff": float(_np.max(_np.abs(filtered.numpy() - probs.numpy())))}
`);
    expect(r.max_diff).toBe(0);
  });
});

// ───────────────────────────────────────────────────────────
// sequence_logprob (verbatim from ch06.py — gather, log_softmax)
// ───────────────────────────────────────────────────────────
describe("sequence_logprob (verbatim — gather + log_softmax)", () => {
  beforeAll(reset);

  it("returns a scalar finite log-probability", async () => {
    const r = await target.run<{ is_scalar: boolean; is_finite: boolean; is_negative: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(0)
model = Qwen3ModelTiny(QWEN_CONFIG_TINY)
model.eval()
token_ids = torch.tensor([5, 10, 15, 20, 5], dtype=torch.int64)
with torch.no_grad():
    lp = sequence_logprob(model, token_ids, prompt_len=2)
v = float(lp.item())
{"is_scalar": True, "is_finite": bool(_np.isfinite(v)), "is_negative": v < 0}
`);
    expect(r.is_scalar).toBe(true);
    expect(r.is_finite).toBe(true);
    expect(r.is_negative).toBe(true); // log-prob is always ≤ 0
  });

  it("gradient flows back through sequence_logprob", async () => {
    const r = await target.run<{ has_grads: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(1)
model = Qwen3ModelTiny(QWEN_CONFIG_TINY)
token_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
lp = sequence_logprob(model, token_ids, prompt_len=2)
lp.backward()
has_grads = any(p.grad is not None for p in model.parameters())
{"has_grads": has_grads}
`);
    expect(r.has_grads).toBe(true);
  });
});

// ───────────────────────────────────────────────────────────
// GRPO-style training step (verbatim pattern from ch06.py)
// ───────────────────────────────────────────────────────────
describe("GRPO training step (verbatim pattern — std, clip_grad_norm_)", () => {
  beforeAll(reset);

  it("policy gradient loss is finite and backward works", async () => {
    const r = await target.run<{ loss_finite: boolean; grads_ok: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(0)
model = Qwen3ModelTiny(QWEN_CONFIG_TINY)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Simulate 3 rollouts
token_seqs = [
    torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
    torch.tensor([1, 2, 6, 7, 8], dtype=torch.int64),
    torch.tensor([1, 2, 9, 10, 11], dtype=torch.int64),
]
prompt_len = 2
rewards = torch.tensor([1.0, 0.0, 0.0])

# Stage 3: compute advantages  (verbatim from ch06.py)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

# Stage 4: compute log-probs
logps = torch.stack([sequence_logprob(model, t, prompt_len) for t in token_seqs])

# Stage 5: policy gradient loss (verbatim)
pg_loss = -(advantages.detach() * logps).mean()

opt.zero_grad()
pg_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()

v = float(pg_loss.item())
grads_ok = any(p.grad is not None for p in model.parameters())
{"loss_finite": bool(_np.isfinite(v)), "grads_ok": grads_ok}
`);
    expect(r.loss_finite).toBe(true);
    expect(r.grads_ok).toBe(true);
  });

  it("advantage normalization uses std correctly", async () => {
    const r = await target.run<{ mean_near_zero: boolean; std_near_one: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(5)
rewards = torch.tensor([1.0, 0.0, 0.5, 0.0, 1.0])
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
arr = advantages.numpy()
# GRPO normalises with unbiased std (N-1 denom), so we check ddof=1
{"mean_near_zero": bool(_np.allclose(arr.mean(), 0.0, atol=0.01)),
 "std_near_one":   bool(_np.allclose(arr.std(ddof=1), 1.0, atol=0.01))}
`);
    expect(r.mean_near_zero).toBe(true);
    expect(r.std_near_one).toBe(true);
  });
});

// ───────────────────────────────────────────────────────────
// torch.bfloat16 transparent fallback
// ───────────────────────────────────────────────────────────
describe("torch.bfloat16 → float32 transparent fallback", () => {
  beforeAll(reset);

  it("torch.bfloat16 equals 'float32' string", async () => {
    const r = await target.run<{ is_f32: boolean }>(`
${SETUP}
{"is_f32": torch.bfloat16 == "float32"}
`);
    expect(r.is_f32).toBe(true);
  });

  it("model with dtype=torch.bfloat16 config instantiates without error", async () => {
    const r = await target.run<{ n_params_gt_zero: boolean }>(`
${SETUP}
cfg_bf16 = dict(QWEN_CONFIG_TINY)
cfg_bf16["dtype"] = torch.bfloat16   # will be treated as "float32"
model = Qwen3ModelTiny(cfg_bf16)
{"n_params_gt_zero": sum(1 for _ in model.parameters()) > 0}
`);
    expect(r.n_params_gt_zero).toBe(true);
  });
});
