/**
 * Kitchen-sink end-to-end test: every major library feature in one model.
 *
 * Trains a CNN with Conv2d + BatchNorm2d + ReLU + MaxPool2d + Dropout +
 * Flatten + Linear + ReLU + Linear, optimized with Adam, on a synthetic
 * 4-class image classification task. Evaluates via `no_grad` + `argmax`.
 *
 * If any of the following has a bug, this test fails:
 *   - Conv2d im2col forward/backward
 *   - BatchNorm2d train/eval branching + running stats
 *   - MaxPool2d argmax routing
 *   - Dropout train/eval branching
 *   - Flatten reshape backward
 *   - Linear + bias broadcasting
 *   - cross_entropy_loss fused formula
 *   - Adam bias correction
 *   - Module.train()/eval() recursive propagation
 *   - no_grad context manager
 *   - Tensor.argmax + int conversion
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
import browsergrad_grad.functional as F
import browsergrad_grad.optim as optim
import numpy as np
`;

describe("kitchen-sink CNN classifier", () => {
  beforeAll(reset);

  it("4-class CNN with BN + Dropout trains and infers cleanly", async () => {
    const result = await target.run<{
      initial_acc: number;
      final_acc: number;
      train_eval_diff: boolean;
    }>(`
${PRELUDE}
np.random.seed(101)

# Four classes of 8x8 single-channel images — each class has a bright
# 4x4 quadrant in a different corner.
def synth(n_per, quadrant):
    out = np.random.randn(n_per, 1, 8, 8).astype(np.float32) * 0.2
    if quadrant == 0:   out[:, 0, :4, :4] += 1.5  # top-left
    elif quadrant == 1: out[:, 0, :4, 4:] += 1.5  # top-right
    elif quadrant == 2: out[:, 0, 4:, :4] += 1.5  # bottom-left
    else:               out[:, 0, 4:, 4:] += 1.5  # bottom-right
    return out

per_class = 24
X = np.concatenate([synth(per_class, q) for q in range(4)], axis=0)
y_labels = np.concatenate([np.full(per_class, q, dtype=np.int64) for q in range(4)])

class KitchenSinkCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.drop2d = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16 * 2 * 2, 32)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 4)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop2d(self.pool(x))           # 8x8 → 4x4
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)                         # 4x4 → 2x2
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

model = KitchenSinkCNN()
opt = optim.Adam(model.parameters(), lr=0.01)

def accuracy():
    model.eval()
    with grad.no_grad():
        logits = model(grad.Tensor(X))
        preds = logits.argmax(dim=-1).data.astype(np.int64)
    model.train()
    return float((preds == y_labels).mean())

initial = accuracy()
for _ in range(80):
    opt.zero_grad()
    loss = F.cross_entropy_loss(model(grad.Tensor(X)), y_labels)
    loss.backward()
    opt.step()
final = accuracy()

# Sanity: train-mode forward (with Dropout active) differs from eval-mode
# forward — confirms the cross-cutting train/eval system is wired up.
model.eval()
with grad.no_grad():
    eval_logits = model(grad.Tensor(X)).data
model.train()
with grad.no_grad():
    train_logits = model(grad.Tensor(X)).data
diff = bool(np.abs(eval_logits - train_logits).max() > 0)

{"initial_acc": initial, "final_acc": final, "train_eval_diff": diff}
`);
    // 4-class problem, random init → ~25% accuracy.
    expect(result.initial_acc).toBeLessThan(0.5);
    // A correctly-wired pipeline reaches near-perfect on this trivial task.
    expect(result.final_acc).toBeGreaterThan(0.9);
    // Dropout / BatchNorm should produce different outputs in train vs eval.
    expect(result.train_eval_diff).toBe(true);
  });
});
