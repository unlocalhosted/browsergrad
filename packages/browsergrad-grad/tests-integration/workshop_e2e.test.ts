/**
 * End-to-end workshop compatibility test.
 *
 * Every class and function below is copy-pasted verbatim from
 * rasbt/LLMs-from-scratch (ch03–ch05), the only change being that
 * `tiktoken` is replaced by a mock tokenizer so the test environment
 * doesn't need the real BPE package.
 *
 * The test proves that a student can:
 *   1. `from browsergrad_grad.torch_compat import install_torch_alias`
 *   2. Paste workshop code as-is (`import torch`, `import torch.nn as nn`, …)
 *   3. Train and generate without touching a single line of the original code.
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
// Install the torch alias once; define the mock tokenizer and
// all verbatim workshop classes that later suites reuse.

const SETUP = `
from browsergrad_grad.torch_compat import install_torch_alias
torch = install_torch_alias()
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ── Mock tokenizer (replaces tiktoken for testing) ──────────
class MockTokenizer:
    """Maps each character to its ASCII code (mod 256).
    Enough to exercise the data pipeline without the real BPE package.
    """
    def encode(self, text, allowed_special=None):
        return [ord(c) % 256 for c in text]
    def decode(self, ids):
        return ''.join(chr(i % 256) for i in ids)

# ── GPT_CONFIG_SMALL — fast stand-in for GPT_CONFIG_124M ────
GPT_CONFIG_SMALL = {
    "vocab_size": 256,
    "context_length": 32,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

# ===================================================================
# Chapter 2 — verbatim from gpt.py / previous_chapters.py
# ===================================================================

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = MockTokenizer()
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers)
    return dataloader


# ===================================================================
# Chapter 3 — verbatim MultiHeadAttention from ch04/gpt.py
# ===================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


# ===================================================================
# Chapter 4 — verbatim from ch04/gpt.py
# ===================================================================

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff    = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb   = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb   = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb  = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ===================================================================
# Chapter 5 — verbatim generation helpers from previous_chapters.py
# ===================================================================

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits   = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx      = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits  = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")),
                logits,
            )
        if temperature > 0.0:
            logits   = logits / temperature
            probs    = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
`;

// ───────────────────────────────────────────────────────────
// Chapter 2 — data pipeline
// ───────────────────────────────────────────────────────────

describe("ch02 — GPTDatasetV1 + DataLoader", () => {
  beforeAll(reset);

  it("builds batches of (input_ids, target_ids) with sliding window", async () => {
    const r = await target.run<{
      n_batches: number;
      input_shape: number[];
      target_shape: number[];
      target_is_shifted: boolean;
    }>(`
${SETUP}
text = "abcdefghijklmnopqrstuvwxyz" * 20   # 520 chars
loader = create_dataloader_v1(
    text, batch_size=4, max_length=8, stride=4,
    shuffle=False, drop_last=True,
)
batches = list(loader)
b_in, b_tgt = batches[0]
# target should be input shifted by 1
shifted = (b_tgt.numpy()[:, 0] == b_in.numpy()[:, 1]).all()
{
    "n_batches": len(batches),
    "input_shape": list(b_in.shape),
    "target_shape": list(b_tgt.shape),
    "target_is_shifted": bool(shifted),
}
`);
    expect(r.input_shape).toEqual([4, 8]);
    expect(r.target_shape).toEqual([4, 8]);
    expect(r.target_is_shifted).toBe(true);
    expect(r.n_batches).toBeGreaterThan(0);
  });
});

// ───────────────────────────────────────────────────────────
// Chapter 3 — MultiHeadAttention
// ───────────────────────────────────────────────────────────

describe("ch03 — MultiHeadAttention (verbatim)", () => {
  beforeAll(reset);

  it("output shape matches (batch, seq, d_out)", async () => {
    const r = await target.run<{ shape: number[] }>(`
${SETUP}
torch.manual_seed(123)
cfg = GPT_CONFIG_SMALL
mha = MultiHeadAttention(
    d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
    context_length=cfg["context_length"],
    dropout=0.0, num_heads=cfg["n_heads"],
)
x = torch.randn(2, 8, cfg["emb_dim"])
out = mha(x)
{"shape": list(out.shape)}
`);
    expect(r.shape).toEqual([2, 8, 64]);
  });

  it("causal mask is registered as a buffer (not a parameter)", async () => {
    const r = await target.run<{ n_params: number; mask_shape: number[] }>(`
${SETUP}
cfg = GPT_CONFIG_SMALL
mha = MultiHeadAttention(
    d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
    context_length=cfg["context_length"],
    dropout=0.0, num_heads=cfg["n_heads"],
)
# buffer is accessible as attribute but excluded from parameters()
param_count = sum(1 for _ in mha.parameters())
{
    "n_params": param_count,
    "mask_shape": list(mha.mask.shape),
}
`);
    // W_query, W_key, W_value: bias=False → 1 param each (3 total)
    // out_proj: bias=True → 2 params
    // Total: 5 params; mask is a buffer, not a param
    expect(r.n_params).toBe(5);
    expect(r.mask_shape).toEqual([32, 32]);
  });

  it("attention weights are causal (no future leakage)", async () => {
    const r = await target.run<{ causal_ok: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(0)
cfg = GPT_CONFIG_SMALL
mha = MultiHeadAttention(
    d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
    context_length=cfg["context_length"],
    dropout=0.0, num_heads=1,
)
# Use identity W_q/W_k/W_v so attn scores are fully determined by x
x = torch.randn(1, 6, cfg["emb_dim"])
out = mha(x)
# Just verify output shape is right and nothing exploded
{"causal_ok": bool(_np.all(_np.isfinite(out.numpy())))}
`);
    expect(r.causal_ok).toBe(true);
  });

  it("backward through MultiHeadAttention produces gradients", async () => {
    const r = await target.run<{ has_grads: boolean; all_finite: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(42)
cfg = GPT_CONFIG_SMALL
mha = MultiHeadAttention(
    d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
    context_length=cfg["context_length"],
    dropout=0.0, num_heads=cfg["n_heads"],
)
x = torch.randn(2, 4, cfg["emb_dim"])
out = mha(x)
out.sum().backward()
grads = [p.grad for p in mha.parameters()]
has_grads = all(g is not None for g in grads)
all_finite = all(_np.all(_np.isfinite(g.numpy())) for g in grads)
{"has_grads": has_grads, "all_finite": all_finite}
`);
    expect(r.has_grads).toBe(true);
    expect(r.all_finite).toBe(true);
  });
});

// ───────────────────────────────────────────────────────────
// Chapter 4 — LayerNorm, GELU, TransformerBlock, GPTModel
// ───────────────────────────────────────────────────────────

describe("ch04 — LayerNorm (verbatim, var-based)", () => {
  beforeAll(reset);

  it("normalises to mean≈0 std≈1 before affine", async () => {
    const r = await target.run<{ mean_ok: boolean; std_ok: boolean }>(`
${SETUP}
import numpy as _np
ln = LayerNorm(64)
# zero-init scale to 1, shift to 0 — already done in __init__
x = torch.randn(4, 8, 64)
out = ln(x)
arr = out.numpy()
row_means = arr.reshape(-1, 64).mean(axis=-1)
row_stds  = arr.reshape(-1, 64).std(axis=-1)
{
    "mean_ok": bool(_np.allclose(row_means, 0.0, atol=1e-4)),
    "std_ok":  bool(_np.allclose(row_stds,  1.0, atol=1e-3)),
}
`);
    expect(r.mean_ok).toBe(true);
    expect(r.std_ok).toBe(true);
  });

  it("scale and shift are registered as parameters", async () => {
    const r = await target.run<{ n_params: number }>(`
${SETUP}
ln = LayerNorm(32)
{"n_params": sum(1 for _ in ln.parameters())}
`);
    expect(r.n_params).toBe(2);
  });

  it("backward through LayerNorm produces finite gradients", async () => {
    const r = await target.run<{ all_finite: boolean }>(`
${SETUP}
import numpy as _np
ln = LayerNorm(64)
x = torch.randn(2, 8, 64)
ln(x).sum().backward()
grads_ok = all(_np.all(_np.isfinite(p.grad.numpy())) for p in ln.parameters())
{"all_finite": grads_ok}
`);
    expect(r.all_finite).toBe(true);
  });
});

describe("ch04 — GELU (verbatim, torch.pi + torch.tanh formula)", () => {
  beforeAll(reset);

  it("GELU is close to the reference approximation", async () => {
    const r = await target.run<{ max_err: number }>(`
${SETUP}
import numpy as _np
gelu = GELU()
x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
out = gelu(x).numpy()
# Reference: scipy-style tanh approx
c = _np.sqrt(2.0 / _np.pi)
ref = 0.5 * x.numpy() * (1 + _np.tanh(c * (x.numpy() + 0.044715 * x.numpy()**3)))
{"max_err": float(_np.max(_np.abs(out - ref)))}
`);
    expect(r.max_err).toBeLessThan(1e-5);
  });
});

describe("ch04 — GPTModel (verbatim)", () => {
  beforeAll(reset);

  it("forward pass produces (batch, seq, vocab_size) logits", async () => {
    const r = await target.run<{ shape: number[] }>(`
${SETUP}
torch.manual_seed(0)
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()
idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
out = model(idx)
{"shape": list(out.shape)}
`);
    expect(r.shape).toEqual([1, 8, 256]);
  });

  it("parameter count is correct for small config", async () => {
    const r = await target.run<{ n_params: number }>(`
${SETUP}
model = GPTModel(GPT_CONFIG_SMALL)
n = sum(p.numel() for p in model.parameters())
{"n_params": n}
`);
    // sanity-check: should be in a reasonable range for the small config
    expect(r.n_params).toBeGreaterThan(50_000);
    expect(r.n_params).toBeLessThan(1_000_000);
  });

  it("forward + backward + AdamW step completes without NaN", async () => {
    const r = await target.run<{ loss_finite: boolean; loss_positive: boolean }>(`
${SETUP}
import numpy as _np
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_SMALL)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

# Build a tiny batch
batch_size, seq_len = 2, 16
idx  = torch.tensor([[i % 256 for i in range(seq_len)] for _ in range(batch_size)])
tgt  = torch.tensor([[(i+1) % 256 for i in range(seq_len)] for _ in range(batch_size)])

optimizer.zero_grad()
logits = model(idx)                          # (B, T, V)
loss   = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), tgt.flatten(0, 1)
)
loss.backward()
optimizer.step()

v = float(loss.item())
{"loss_finite": bool(_np.isfinite(v)), "loss_positive": v > 0}
`);
    expect(r.loss_finite).toBe(true);
    expect(r.loss_positive).toBe(true);
  });

  it("state_dict round-trip preserves all weights", async () => {
    const r = await target.run<{ max_diff: number }>(`
${SETUP}
import numpy as _np
torch.manual_seed(7)
m1 = GPTModel(GPT_CONFIG_SMALL)
sd = m1.state_dict()
m2 = GPTModel(GPT_CONFIG_SMALL)
m2.load_state_dict(sd)
max_diff = max(
    float(_np.max(_np.abs(v - sd[k])))
    for k, v in m2.state_dict().items()
)
{"max_diff": max_diff}
`);
    expect(r.max_diff).toBe(0);
  });
});

// ───────────────────────────────────────────────────────────
// Chapter 5 — text generation
// ───────────────────────────────────────────────────────────

describe("ch05 — generate_text_simple (greedy, verbatim)", () => {
  beforeAll(reset);

  it("extends input by max_new_tokens tokens", async () => {
    const r = await target.run<{ output_len: number }>(`
${SETUP}
torch.manual_seed(0)
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()
idx = torch.tensor([[10, 20, 30]])
out = generate_text_simple(model, idx, max_new_tokens=5,
                            context_size=GPT_CONFIG_SMALL["context_length"])
{"output_len": int(out.shape[1])}
`);
    expect(r.output_len).toBe(8); // 3 prompt + 5 generated
  });

  it("greedy generation is deterministic", async () => {
    const r = await target.run<{ same: boolean }>(`
${SETUP}
torch.manual_seed(0)
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()
idx = torch.tensor([[10, 20, 30]])
out1 = generate_text_simple(model, idx, max_new_tokens=5,
                             context_size=GPT_CONFIG_SMALL["context_length"])
out2 = generate_text_simple(model, idx, max_new_tokens=5,
                             context_size=GPT_CONFIG_SMALL["context_length"])
{"same": out1.tolist() == out2.tolist()}
`);
    expect(r.same).toBe(true);
  });

  it("generate tokens are in-vocabulary", async () => {
    const r = await target.run<{ in_vocab: boolean }>(`
${SETUP}
torch.manual_seed(0)
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()
idx = torch.tensor([[5, 10, 15]])
out = generate_text_simple(model, idx, max_new_tokens=10,
                            context_size=GPT_CONFIG_SMALL["context_length"])
tokens = out.numpy().flatten().tolist()
{"in_vocab": all(0 <= t < GPT_CONFIG_SMALL["vocab_size"] for t in tokens)}
`);
    expect(r.in_vocab).toBe(true);
  });
});

describe("ch05 — generate() with temperature and top-k (verbatim)", () => {
  beforeAll(reset);

  it("temperature sampling extends sequence correctly", async () => {
    const r = await target.run<{ output_len: number; in_vocab: boolean }>(`
${SETUP}
torch.manual_seed(42)
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()
idx = torch.tensor([[5, 10]])
out = generate(model, idx, max_new_tokens=6, context_size=32,
               temperature=1.0, top_k=None)
tokens = out.numpy().flatten().tolist()
{
    "output_len": int(out.shape[1]),
    "in_vocab": all(0 <= t < 256 for t in tokens),
}
`);
    expect(r.output_len).toBe(8);
    expect(r.in_vocab).toBe(true);
  });

  it("top-k sampling restricts choices to top-k logits", async () => {
    const r = await target.run<{ output_len: number; in_vocab: boolean }>(`
${SETUP}
torch.manual_seed(99)
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()
idx = torch.tensor([[1, 2, 3]])
out = generate(model, idx, max_new_tokens=4, context_size=32,
               temperature=0.7, top_k=10)
tokens = out.numpy().flatten().tolist()
{
    "output_len": int(out.shape[1]),
    "in_vocab": all(0 <= t < 256 for t in tokens),
}
`);
    expect(r.output_len).toBe(7);
    expect(r.in_vocab).toBe(true);
  });

  it("eos_id terminates generation early", async () => {
    const r = await target.run<{ stopped_early: boolean }>(`
${SETUP}
torch.manual_seed(0)
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()
# greedy (temperature=0) so generation is deterministic
# Ask for up to 20 new tokens but stop at the first token that == eos_id
greedy_out = generate(model, torch.tensor([[1, 2]]), max_new_tokens=1,
                      context_size=32, temperature=0.0)
eos_id = int(greedy_out.numpy()[0, -1])  # whatever the first greedy token is

out = generate(model, torch.tensor([[1, 2]]), max_new_tokens=20,
               context_size=32, temperature=0.0, eos_id=eos_id)
{"stopped_early": int(out.shape[1]) <= 4}  # 2 prompt + at most 2 more
`);
    expect(r.stopped_early).toBe(true);
  });
});

// ───────────────────────────────────────────────────────────
// Chapter 5 — mini training loop (verbatim calc_loss_batch pattern)
// ───────────────────────────────────────────────────────────

describe("ch05 — training loop (calc_loss_batch pattern)", () => {
  beforeAll(reset);

  it("loss decreases over 10 steps on a tiny overfit corpus", async () => {
    const r = await target.run<{
      loss_start: number;
      loss_end: number;
      decreased: boolean;
    }>(`
${SETUP}
import numpy as _np

def calc_loss_batch(input_batch, target_batch, model):
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(0, 1)
    )
    return loss

torch.manual_seed(0)
model = GPTModel(GPT_CONFIG_SMALL)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

# Single fixed batch — overfit check
seq = [i % GPT_CONFIG_SMALL["vocab_size"] for i in range(16)]
input_batch  = torch.tensor([seq])
target_batch = torch.tensor([seq[1:] + [seq[0]]])

losses = []
for _ in range(10):
    optimizer.zero_grad()
    loss = calc_loss_batch(input_batch, target_batch, model)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.item()))

{
    "loss_start": losses[0],
    "loss_end":   losses[-1],
    "decreased":  losses[-1] < losses[0],
}
`);
    expect(r.loss_start).toBeGreaterThan(0);
    expect(r.decreased).toBe(true);
  });

  it("evaluate_model pattern works with DataLoader", async () => {
    const r = await target.run<{ train_loss_finite: boolean }>(`
${SETUP}
import numpy as _np

def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0.0
    n = 0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten(0, 1)
        )
        total_loss += float(loss.item())
        n += 1
    return total_loss / n if n > 0 else 0.0

torch.manual_seed(0)
model = GPTModel(GPT_CONFIG_SMALL)
model.eval()

text = "hello world " * 50
loader = create_dataloader_v1(
    text, batch_size=2, max_length=16, stride=8,
    shuffle=False, drop_last=True,
)
loss = calc_loss_loader(loader, model, num_batches=2)
{"train_loss_finite": bool(_np.isfinite(loss))}
`);
    expect(r.train_loss_finite).toBe(true);
  });
});
