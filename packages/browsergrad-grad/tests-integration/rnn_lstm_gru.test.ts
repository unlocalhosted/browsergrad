/**
 * Pile A #15 — nn.RNN, nn.LSTM, nn.GRU + BPTT.
 *
 * Oracle: same recurrence, hand-unrolled in NumPy using the same parameter
 * values the layer was initialized with. BPTT verified via finite differences.
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
import browsergrad_grad.nn as nn
import numpy as np
`;

describe("nn.RNN", () => {
  beforeAll(reset);

  it("forward matches the hand-unrolled tanh recurrence", async () => {
    const result = await target.run<{ ok: boolean; max_diff: number }>(`
${PRELUDE}
np.random.seed(0)
T, B, I, H = 4, 2, 3, 5  # seq_len, batch, input_size, hidden_size
rnn = nn.RNN(input_size=I, hidden_size=H, batch_first=False, nonlinearity='tanh')
x_np = np.random.randn(T, B, I).astype(np.float32)
x = grad.Tensor(x_np)
out, h_n = rnn(x)
# Hand-rolled oracle
W_ih = rnn.weight_ih_l0.data
W_hh = rnn.weight_hh_l0.data
b_ih = rnn.bias_ih_l0.data
b_hh = rnn.bias_hh_l0.data
h = np.zeros((B, H), dtype=np.float32)
oracle_out = np.zeros((T, B, H), dtype=np.float32)
for t in range(T):
    h = np.tanh(x_np[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
    oracle_out[t] = h
out_data = np.asarray(out.tolist(), dtype=np.float32)
hn_data = np.asarray(h_n.tolist(), dtype=np.float32)
max_diff_out = float(np.max(np.abs(out_data - oracle_out)))
max_diff_hn = float(np.max(np.abs(hn_data - h)))
{"ok": max_diff_out < 1e-4 and max_diff_hn < 1e-4, "max_diff": max(max_diff_out, max_diff_hn)}
`);
    expect(result.ok).toBe(true);
  });

  it("backward flows gradients through time (BPTT) — finite-diff sanity", async () => {
    const result = await target.run<{ closest: number; eps: number }>(`
${PRELUDE}
np.random.seed(1)
T, B, I, H = 3, 1, 2, 3
rnn = nn.RNN(input_size=I, hidden_size=H, batch_first=False)
x_np = np.random.randn(T, B, I).astype(np.float32)
x = grad.Tensor(x_np, requires_grad=True)
out, _ = rnn(x)
loss = out.sum()
loss.backward()
ours = np.asarray(x.grad).copy()

# Finite diff against the same recurrence
def fwd(xp):
    h = np.zeros((B, H), dtype=np.float32)
    s = 0.0
    W_ih = rnn.weight_ih_l0.data; W_hh = rnn.weight_hh_l0.data
    b_ih = rnn.bias_ih_l0.data;   b_hh = rnn.bias_hh_l0.data
    for t in range(T):
        h = np.tanh(xp[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        s += float(h.sum())
    return s
eps = 1e-3
fd = np.zeros_like(x_np)
for t in range(T):
    for b in range(B):
        for i in range(I):
            xp = x_np.copy(); xp[t, b, i] += eps
            xm = x_np.copy(); xm[t, b, i] -= eps
            fd[t, b, i] = (fwd(xp) - fwd(xm)) / (2 * eps)
diff = float(np.max(np.abs(ours - fd)))
{"closest": diff, "eps": eps}
`);
    expect(result.closest).toBeLessThan(1e-2);
  });
});

describe("nn.LSTM", () => {
  beforeAll(reset);

  it("forward matches the hand-unrolled LSTM recurrence", async () => {
    const result = await target.run<{ ok: boolean }>(`
${PRELUDE}
np.random.seed(2)
T, B, I, H = 3, 2, 4, 5
lstm = nn.LSTM(input_size=I, hidden_size=H, batch_first=False)
x_np = np.random.randn(T, B, I).astype(np.float32)
out, (h_n, c_n) = lstm(grad.Tensor(x_np))

W_ih = lstm.weight_ih_l0.data  # shape (4H, I)
W_hh = lstm.weight_hh_l0.data  # shape (4H, H)
b_ih = lstm.bias_ih_l0.data; b_hh = lstm.bias_hh_l0.data
h = np.zeros((B, H), dtype=np.float32)
c = np.zeros((B, H), dtype=np.float32)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

oracle = np.zeros((T, B, H), dtype=np.float32)
for t in range(T):
    gates = x_np[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh
    i_g = sigmoid(gates[:, 0:H])
    f_g = sigmoid(gates[:, H:2*H])
    g_g = np.tanh(gates[:, 2*H:3*H])
    o_g = sigmoid(gates[:, 3*H:4*H])
    c = f_g * c + i_g * g_g
    h = o_g * np.tanh(c)
    oracle[t] = h

out_np = np.asarray(out.tolist(), dtype=np.float32)
ok = float(np.max(np.abs(out_np - oracle))) < 1e-4
{"ok": ok}
`);
    expect(result.ok).toBe(true);
  });
});

describe("nn.GRU", () => {
  beforeAll(reset);

  it("forward matches the hand-unrolled GRU recurrence", async () => {
    const result = await target.run<{ ok: boolean }>(`
${PRELUDE}
np.random.seed(3)
T, B, I, H = 3, 2, 4, 5
gru = nn.GRU(input_size=I, hidden_size=H, batch_first=False)
x_np = np.random.randn(T, B, I).astype(np.float32)
out, h_n = gru(grad.Tensor(x_np))

W_ih = gru.weight_ih_l0.data  # (3H, I)
W_hh = gru.weight_hh_l0.data  # (3H, H)
b_ih = gru.bias_ih_l0.data
b_hh = gru.bias_hh_l0.data
h = np.zeros((B, H), dtype=np.float32)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

oracle = np.zeros((T, B, H), dtype=np.float32)
for t in range(T):
    ih = x_np[t] @ W_ih.T + b_ih   # (B, 3H)
    hh = h        @ W_hh.T + b_hh
    r = sigmoid(ih[:, 0:H]   + hh[:, 0:H])
    z = sigmoid(ih[:, H:2*H] + hh[:, H:2*H])
    n = np.tanh(ih[:, 2*H:3*H] + r * hh[:, 2*H:3*H])
    h = (1.0 - z) * n + z * h
    oracle[t] = h

out_np = np.asarray(out.tolist(), dtype=np.float32)
ok = float(np.max(np.abs(out_np - oracle))) < 1e-4
{"ok": ok}
`);
    expect(result.ok).toBe(true);
  });
});

describe("RNN: batch_first=True", () => {
  beforeAll(reset);

  it("batch_first=True transposes input/output appropriately", async () => {
    const result = await target.run<{ shapes_equal: boolean }>(`
${PRELUDE}
T, B, I, H = 3, 2, 4, 5
rnn_seq = nn.RNN(I, H, batch_first=False)
rnn_bf  = nn.RNN(I, H, batch_first=True)
# Force same weights
rnn_bf.weight_ih_l0.data[...] = rnn_seq.weight_ih_l0.data
rnn_bf.weight_hh_l0.data[...] = rnn_seq.weight_hh_l0.data
rnn_bf.bias_ih_l0.data[...]   = rnn_seq.bias_ih_l0.data
rnn_bf.bias_hh_l0.data[...]   = rnn_seq.bias_hh_l0.data
x_seq = np.random.randn(T, B, I).astype(np.float32)
x_bf  = np.transpose(x_seq, (1, 0, 2))
out_seq, _ = rnn_seq(grad.Tensor(x_seq))
out_bf, _  = rnn_bf(grad.Tensor(x_bf))
seq_arr = np.asarray(out_seq.tolist())
bf_arr  = np.asarray(out_bf.tolist())
ok = bool(np.allclose(seq_arr, np.transpose(bf_arr, (1, 0, 2)), atol=1e-5))
{"shapes_equal": ok}
`);
    expect(result.shapes_equal).toBe(true);
  });
});

describe("nn.RNN/LSTM/GRU: multi-layer and bidirectional", () => {
  beforeAll(reset);

  it("RNN bidirectional output and h_n follow PyTorch direction ordering", async () => {
    const result = await target.run<{ ok: boolean; max_diff: number }>(`
${PRELUDE}
np.random.seed(4)
T, B, I, H = 4, 2, 3, 5
rnn = nn.RNN(I, H, bidirectional=True, batch_first=False)
x_np = np.random.randn(T, B, I).astype(np.float32)
out, h_n = rnn(grad.Tensor(x_np))

def run_direction(seq, W_ih, W_hh, b_ih, b_hh):
    h = np.zeros((B, H), dtype=np.float32)
    ys = []
    for t in range(seq.shape[0]):
        h = np.tanh(seq[t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
        ys.append(h.copy())
    return np.stack(ys, axis=0), h

fwd, h_fwd = run_direction(
    x_np,
    rnn.weight_ih_l0.data, rnn.weight_hh_l0.data,
    rnn.bias_ih_l0.data, rnn.bias_hh_l0.data,
)
rev_raw, h_rev = run_direction(
    x_np[::-1],
    rnn.weight_ih_l0_reverse.data, rnn.weight_hh_l0_reverse.data,
    rnn.bias_ih_l0_reverse.data, rnn.bias_hh_l0_reverse.data,
)
oracle = np.concatenate([fwd, rev_raw[::-1]], axis=2)
oracle_h = np.stack([h_fwd, h_rev], axis=0)
out_np = np.asarray(out.tolist(), dtype=np.float32)
hn_np = np.asarray(h_n.tolist(), dtype=np.float32)
max_diff = float(max(np.max(np.abs(out_np - oracle)), np.max(np.abs(hn_np - oracle_h))))
{"ok": max_diff < 1e-4, "max_diff": max_diff}
`);
    expect(result.ok).toBe(true);
  });

  it("LSTM stacked bidirectional state shape and state_dict keys match PyTorch layout", async () => {
    const result = await target.run<{ ok: boolean; keys_ok: boolean; shapes_ok: boolean; grads_ok: boolean }>(`
${PRELUDE}
np.random.seed(5)
B, T, I, H = 2, 3, 4, 6
lstm = nn.LSTM(I, H, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
lstm.eval()
x = grad.Tensor(np.random.randn(B, T, I).astype(np.float32), requires_grad=True)
h0 = grad.Tensor(np.random.randn(4, B, H).astype(np.float32))
c0 = grad.Tensor(np.random.randn(4, B, H).astype(np.float32))
out, (h_n, c_n) = lstm(x, (h0, c0))
loss = out.sum() + h_n.sum() + c_n.sum()
loss.backward()
sd = lstm.state_dict()
expected = [
    "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
    "weight_ih_l0_reverse", "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse",
    "weight_ih_l1", "weight_hh_l1", "bias_ih_l1", "bias_hh_l1",
    "weight_ih_l1_reverse", "weight_hh_l1_reverse", "bias_ih_l1_reverse", "bias_hh_l1_reverse",
]
keys_ok = sorted(sd.keys()) == sorted(expected)
shapes_ok = (
    out.shape == (B, T, 2 * H)
    and h_n.shape == (4, B, H)
    and c_n.shape == (4, B, H)
    and sd["weight_ih_l1"].shape == (4 * H, 2 * H)
    and sd["weight_ih_l1_reverse"].shape == (4 * H, 2 * H)
)
grads_ok = x.grad is not None and x.grad.shape == (B, T, I)
{"ok": bool(keys_ok and shapes_ok and grads_ok), "keys_ok": bool(keys_ok), "shapes_ok": bool(shapes_ok), "grads_ok": bool(grads_ok)}
`);
    expect(result.ok).toBe(true);
  });

  it("GRU stacked bidirectional supports bias=False, custom h_0, and backward", async () => {
    const result = await target.run<{ ok: boolean }>(`
${PRELUDE}
np.random.seed(6)
T, B, I, H = 3, 2, 4, 5
gru = nn.GRU(I, H, num_layers=2, bias=False, bidirectional=True)
x = grad.Tensor(np.random.randn(T, B, I).astype(np.float32), requires_grad=True)
h0 = grad.Tensor(np.random.randn(4, B, H).astype(np.float32))
out, h_n = gru(x, h0)
loss = (out * out).mean() + h_n.mean()
loss.backward()
sd = gru.state_dict()
no_bias = all("bias" not in k for k in sd.keys())
shape_ok = out.shape == (T, B, 2 * H) and h_n.shape == (4, B, H)
keys_ok = "weight_ih_l1_reverse" in sd and sd["weight_ih_l1"].shape == (3 * H, 2 * H)
grad_ok = x.grad is not None and x.grad.shape == x.shape
{"ok": bool(no_bias and shape_ok and keys_ok and grad_ok)}
`);
    expect(result.ok).toBe(true);
  });

  it("RNN dropout applies between stacked layers in train mode only", async () => {
    const result = await target.run<{ ok: boolean; train_eval_diff: number; eval_repeat_diff: number }>(`
${PRELUDE}
np.random.seed(7)
T, B, I, H = 5, 2, 3, 4
rnn = nn.RNN(I, H, num_layers=2, dropout=0.9)
x = grad.Tensor(np.ones((T, B, I), dtype=np.float32))
rnn.eval()
eval_a, _ = rnn(x)
eval_b, _ = rnn(x)
rnn.train()
np.random.seed(123)
train_a, _ = rnn(x)
eval_arr_a = np.asarray(eval_a.tolist())
eval_arr_b = np.asarray(eval_b.tolist())
train_arr = np.asarray(train_a.tolist())
eval_repeat_diff = float(np.max(np.abs(eval_arr_a - eval_arr_b)))
train_eval_diff = float(np.max(np.abs(train_arr - eval_arr_a)))
{"ok": eval_repeat_diff == 0.0 and train_eval_diff > 1e-5, "train_eval_diff": train_eval_diff, "eval_repeat_diff": eval_repeat_diff}
`);
    expect(result.ok).toBe(true);
  });
});
