"""Generate PyTorch reference fixtures for browsergrad-grad conformance tests.

Run with a real torch install:

    python scripts/gen_pytorch_fixtures.py

Writes JSON fixtures into tests-integration/fixtures/. Each fixture captures:
  - inputs (numpy arrays as nested lists)
  - parameter values (so the test loads the SAME weights)
  - expected forward output
  - expected backward grads where applicable

The browsergrad-grad conformance test loads these and asserts agreement
within 1e-4 (relaxed to 1e-3 for layers with long accumulation chains).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit(
        "This script requires numpy and torch.\n"
        "Install with: pip install numpy torch"
    )

torch.manual_seed(0)
FIX_DIR = Path(__file__).resolve().parent.parent / "tests-integration" / "fixtures"
FIX_DIR.mkdir(parents=True, exist_ok=True)


def t(x):
    """Tensor → list of plain Python floats."""
    return x.detach().cpu().numpy().astype(np.float32).tolist()


def make_linear() -> dict:
    """nn.Linear forward + backward."""
    torch.manual_seed(1)
    layer = nn.Linear(3, 4)
    x = torch.randn(2, 3, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    return {
        "name": "linear_2x3_4",
        "kind": "linear",
        "in_features": 3,
        "out_features": 4,
        "weight": t(layer.weight),
        "bias": t(layer.bias),
        "input": t(x),
        "forward": t(y),
        "x_grad": t(x.grad),
    }


def make_cross_entropy() -> dict:
    """F.cross_entropy forward + grad through logits."""
    torch.manual_seed(2)
    logits = torch.randn(5, 3, requires_grad=True)
    targets = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    return {
        "name": "cross_entropy_5x3",
        "kind": "cross_entropy",
        "logits": t(logits),
        "targets": targets.tolist(),
        "loss": float(loss.item()),
        "logits_grad": t(logits.grad),
    }


def make_layernorm() -> dict:
    torch.manual_seed(3)
    layer = nn.LayerNorm(normalized_shape=4)
    x = torch.randn(2, 4, requires_grad=True)
    y = layer(x)
    return {
        "name": "layernorm_2x4",
        "kind": "layernorm",
        "normalized_shape": [4],
        "weight": t(layer.weight),
        "bias": t(layer.bias),
        "input": t(x),
        "forward": t(y),
    }


def make_softmax() -> dict:
    torch.manual_seed(4)
    x = torch.randn(3, 5)
    y = F.softmax(x, dim=-1)
    return {
        "name": "softmax_3x5_dim_minus1",
        "kind": "softmax",
        "input": t(x),
        "dim": -1,
        "forward": t(y),
    }


def make_relu() -> dict:
    torch.manual_seed(5)
    x = torch.randn(4, 5, requires_grad=True)
    y = F.relu(x)
    y.sum().backward()
    return {
        "name": "relu_4x5",
        "kind": "relu",
        "input": t(x),
        "forward": t(y),
        "x_grad": t(x.grad),
    }


def main():
    fixtures = {
        "version": "1",
        "torch_version": torch.__version__,
        "cases": [
            make_linear(),
            make_cross_entropy(),
            make_layernorm(),
            make_softmax(),
            make_relu(),
        ],
    }
    out_path = FIX_DIR / "pytorch_conformance.json"
    out_path.write_text(json.dumps(fixtures, indent=2))
    print(f"Wrote {out_path} ({len(fixtures['cases'])} cases)")


if __name__ == "__main__":
    main()
