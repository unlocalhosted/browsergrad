/**
 * Pile A ★ — PyTorch-conformance fixture suite.
 *
 * The strongest possible oracle: actual outputs captured from real PyTorch
 * (via scripts/gen_pytorch_fixtures.py), compared against browsergrad-grad
 * forward/backward results within 1e-4.
 *
 * Regenerate fixtures with:
 *   /tmp/bgvenv/bin/python scripts/gen_pytorch_fixtures.py
 * (or any Python env with `torch` installed)
 */

import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";

import { beforeAll, describe, expect, it } from "vitest";
import { clearNamespace, getGradTarget } from "./pyodide-host";

let target: Awaited<ReturnType<typeof getGradTarget>>;

beforeAll(async () => {
  target = await getGradTarget();
}, 120_000);

async function reset(): Promise<void> {
  await clearNamespace(target);
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = resolve(__dirname, "fixtures", "pytorch_conformance.json");

type FixtureCase =
  | {
      kind: "linear";
      name: string;
      in_features: number;
      out_features: number;
      weight: number[][];
      bias: number[];
      input: number[][];
      forward: number[][];
      x_grad: number[][];
    }
  | {
      kind: "cross_entropy";
      name: string;
      logits: number[][];
      targets: number[];
      loss: number;
      logits_grad: number[][];
    }
  | {
      kind: "layernorm";
      name: string;
      normalized_shape: number[];
      weight: number[];
      bias: number[];
      input: number[][];
      forward: number[][];
    }
  | {
      kind: "softmax";
      name: string;
      input: number[][];
      dim: number;
      forward: number[][];
    }
  | {
      kind: "relu";
      name: string;
      input: number[][];
      forward: number[][];
      x_grad: number[][];
    };

interface Fixtures {
  version: string;
  torch_version: string;
  cases: FixtureCase[];
}

const fixtures: Fixtures = JSON.parse(readFileSync(FIXTURE_PATH, "utf8"));

function flatAbsMaxDiff(a: number[] | number[][], b: number[] | number[][]): number {
  const flatA = (a as unknown[]).flat(Infinity) as number[];
  const flatB = (b as unknown[]).flat(Infinity) as number[];
  let m = 0;
  for (let i = 0; i < flatA.length; i++) {
    m = Math.max(m, Math.abs(flatA[i]! - flatB[i]!));
  }
  return m;
}

describe(`PyTorch conformance suite (torch ${fixtures.torch_version})`, () => {
  beforeAll(reset);

  for (const c of fixtures.cases) {
    if (c.kind === "linear") {
      it(`${c.name}: nn.Linear forward + backward agrees with torch within 1e-4`, async () => {
        const res = await target.run<{ forward: number[][]; x_grad: number[][] }>(`
import browsergrad_grad as grad
import browsergrad_grad.nn as nn
import numpy as np
fc = nn.Linear(${c.in_features}, ${c.out_features})
fc.weight.data[...] = np.array(${JSON.stringify(c.weight)}, dtype=np.float32)
fc.bias.data[...]   = np.array(${JSON.stringify(c.bias)}, dtype=np.float32)
x = grad.Tensor(np.array(${JSON.stringify(c.input)}, dtype=np.float32), requires_grad=True)
y = fc(x)
y.sum().backward()
{"forward": y.tolist(), "x_grad": np.asarray(x.grad).tolist()}
`);
        expect(flatAbsMaxDiff(res.forward, c.forward)).toBeLessThan(1e-4);
        expect(flatAbsMaxDiff(res.x_grad, c.x_grad)).toBeLessThan(1e-4);
      });
    }

    if (c.kind === "cross_entropy") {
      it(`${c.name}: F.cross_entropy_loss matches torch loss + grad within 1e-4`, async () => {
        const res = await target.run<{ loss: number; logits_grad: number[][] }>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import numpy as np
logits = grad.Tensor(np.array(${JSON.stringify(c.logits)}, dtype=np.float32), requires_grad=True)
loss = F.cross_entropy_loss(logits, np.array(${JSON.stringify(c.targets)}, dtype=np.int64))
loss.backward()
{"loss": float(loss.data), "logits_grad": np.asarray(logits.grad).tolist()}
`);
        expect(Math.abs(res.loss - c.loss)).toBeLessThan(1e-4);
        expect(flatAbsMaxDiff(res.logits_grad, c.logits_grad)).toBeLessThan(1e-4);
      });
    }

    if (c.kind === "layernorm") {
      it(`${c.name}: nn.LayerNorm forward matches torch within 1e-4`, async () => {
        const res = await target.run<{ forward: number[][] }>(`
import browsergrad_grad as grad
import browsergrad_grad.nn as nn
import numpy as np
ln = nn.LayerNorm(${JSON.stringify(c.normalized_shape)})
ln.weight.data[...] = np.array(${JSON.stringify(c.weight)}, dtype=np.float32)
ln.bias.data[...]   = np.array(${JSON.stringify(c.bias)}, dtype=np.float32)
x = grad.Tensor(np.array(${JSON.stringify(c.input)}, dtype=np.float32))
y = ln(x)
{"forward": y.tolist()}
`);
        expect(flatAbsMaxDiff(res.forward, c.forward)).toBeLessThan(1e-4);
      });
    }

    if (c.kind === "softmax") {
      it(`${c.name}: F.softmax matches torch within 1e-5`, async () => {
        const res = await target.run<{ forward: number[][] }>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import numpy as np
x = grad.Tensor(np.array(${JSON.stringify(c.input)}, dtype=np.float32))
y = F.softmax(x, dim=${c.dim})
{"forward": y.tolist()}
`);
        expect(flatAbsMaxDiff(res.forward, c.forward)).toBeLessThan(1e-5);
      });
    }

    if (c.kind === "relu") {
      it(`${c.name}: F.relu forward + backward matches torch exactly`, async () => {
        const res = await target.run<{ forward: number[][]; x_grad: number[][] }>(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import numpy as np
x = grad.Tensor(np.array(${JSON.stringify(c.input)}, dtype=np.float32), requires_grad=True)
y = F.relu(x)
y.sum().backward()
{"forward": y.tolist(), "x_grad": np.asarray(x.grad).tolist()}
`);
        // ReLU is exact bit-for-bit at non-zero entries; tolerate zero.
        expect(flatAbsMaxDiff(res.forward, c.forward)).toBe(0);
        expect(flatAbsMaxDiff(res.x_grad, c.x_grad)).toBe(0);
      });
    }
  }
});
