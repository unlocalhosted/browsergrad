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
