import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_CROSS_ENTROPY_CONFORMANCE } from "../../../test-support/framework-cross-entropy-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Grad torch.nn.functional.cross_entropy conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("matches typed weighted index and probability targets with stable snapshot gradients", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CROSS_ENTROPY_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as bg
import browsergrad_grad.functional as F
import browsergrad_grad.nn as nn
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
base = fixture["base"]

def tensor(shape, values, dtype="float32", requires_grad=False):
    return bg.Tensor(
        np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape)),
        dtype=dtype,
        requires_grad=requires_grad,
    )

index_input = tensor(base["inputShape"], base["input"], requires_grad=True)
index_target = tensor([2], base["indexTarget"], "int64")
weight = tensor([3], base["weight"])
index_none = F.cross_entropy(
    index_input,
    index_target,
    weight=weight,
    reduction="none",
    label_smoothing=base["labelSmoothing"],
)
index_sum = F.cross_entropy(
    tensor(base["inputShape"], base["input"]),
    index_target,
    weight=weight,
    reduction="sum",
    label_smoothing=base["labelSmoothing"],
)
index_mean = F.cross_entropy(
    index_input,
    index_target,
    weight=weight,
    label_smoothing=base["labelSmoothing"],
)
index_input.data[...] = 100
index_target.data[...] = 1
weight.data[...] = 100
index_mean.backward()

probability_input = tensor(
    base["inputShape"], base["input"], requires_grad=True
)
probability_target = tensor(
    base["inputShape"],
    base["probabilityTarget"],
    requires_grad=True,
)
probability_weight = tensor([3], base["weight"])
probability_none = F.cross_entropy(
    probability_input,
    probability_target,
    weight=probability_weight,
    reduction="none",
    label_smoothing=base["labelSmoothing"],
)
probability_sum = F.cross_entropy(
    tensor(base["inputShape"], base["input"]),
    tensor(base["inputShape"], base["probabilityTarget"]),
    weight=probability_weight,
    reduction="sum",
    label_smoothing=base["labelSmoothing"],
)
probability_mean = F.cross_entropy(
    probability_input,
    probability_target,
    weight=probability_weight,
    label_smoothing=base["labelSmoothing"],
)
probability_input.data[...] = -100
probability_target.data[...] = 0
probability_weight.data[...] = 50
probability_mean.backward()

ignore_input = tensor(base["inputShape"], base["input"], requires_grad=True)
all_ignore = F.cross_entropy(
    ignore_input,
    tensor([2], [-100, -100], "int64"),
)
all_ignore.backward()
empty_input = tensor([0, 3], [], requires_grad=True)
empty_target = tensor([0], [], "int64")
empty_mean = F.cross_entropy(empty_input, empty_target)
empty_mean.backward()

dtype_results = []
for dtype in fixture["mixedDtypes"]:
    source = tensor(base["inputShape"], base["input"], dtype, True)
    output = F.cross_entropy(
        source,
        tensor([2], base["indexTarget"], "int64"),
        reduction="sum",
    )
    output.backward()
    dtype_results.append({
        "output": output.dtype,
        "gradient": source.grad.dtype,
    })

legacy_array_target = F.cross_entropy_loss(
    tensor(base["inputShape"], base["input"]),
    np.asarray(base["indexTarget"], dtype=np.int64),
)
module = nn.CrossEntropyLoss(
    weight=tensor([3], base["weight"]),
    reduction="sum",
    label_smoothing=base["labelSmoothing"],
)
module_value = module(
    tensor(base["inputShape"], base["input"]),
    tensor([2], base["indexTarget"], "int64"),
)

{
    "indexNone": index_none.tolist(),
    "indexSum": float(index_sum.item()),
    "indexMean": float(index_mean.item()),
    "indexGradient": index_input.grad.tolist(),
    "probabilityNone": probability_none.tolist(),
    "probabilitySum": float(probability_sum.item()),
    "probabilityMean": float(probability_mean.item()),
    "probabilityInputGradient": probability_input.grad.tolist(),
    "probabilityTargetGradient": probability_target.grad.tolist(),
    "allIgnoreNan": bool(np.isnan(all_ignore.item())),
    "allIgnoreGradient": ignore_input.grad.tolist(),
    "emptyMeanNan": bool(np.isnan(empty_mean.item())),
    "emptyGradientShape": list(empty_input.grad.shape),
    "dtypes": dtype_results,
    "legacyArrayTarget": float(legacy_array_target.item()),
    "module": float(module_value.item()),
    "moduleState": {
        key: np.asarray(value).reshape(-1).tolist()
        for key, value in module.state_dict().items()
    },
}
`);

    const fixture = FRAMEWORK_CROSS_ENTROPY_CONFORMANCE;
    expectCloseArray(result.indexNone, fixture.base.indexNone);
    expect(result.indexSum).toBeCloseTo(fixture.base.indexSum, 5);
    expect(result.indexMean).toBeCloseTo(fixture.base.indexMean, 5);
    expectCloseArray(
      (result.indexGradient as number[][]).flat(),
      fixture.base.indexMeanGradient,
    );
    expectCloseArray(result.probabilityNone, fixture.base.probabilityNone);
    expect(result.probabilitySum).toBeCloseTo(fixture.base.probabilitySum, 5);
    expect(result.probabilityMean).toBeCloseTo(fixture.base.probabilityMean, 5);
    expectCloseArray(
      (result.probabilityInputGradient as number[][]).flat(),
      fixture.base.probabilityMeanInputGradient,
    );
    expectCloseArray(
      (result.probabilityTargetGradient as number[][]).flat(),
      fixture.base.probabilityMeanTargetGradient,
    );
    expect(result.allIgnoreNan).toBe(true);
    expectCloseArray(
      (result.allIgnoreGradient as number[][]).flat(),
      [0, 0, 0, 0, 0, 0],
    );
    expect(result.emptyMeanNan).toBe(true);
    expect(result.emptyGradientShape).toEqual([0, 3]);
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map((dtype) => ({
      output: dtype,
      gradient: dtype,
    })));
    expect(result.legacyArrayTarget).toBeCloseTo(0.288726, 5);
    expect(result.module).toBeCloseTo(fixture.base.indexSum, 5);
    expect(result.moduleState).toEqual({ weight: fixture.base.weight });
  });

  it("rejects invalid target modes, weights, reductions, smoothing, ranges, and budgets", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CROSS_ENTROPY_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_grad as bg
import browsergrad_grad.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

def tensor(values, dtype="float32", requires_grad=False):
    return bg.Tensor(
        np.asarray(values, dtype=np.dtype(dtype)),
        dtype=dtype,
        requires_grad=requires_grad,
    )

source = tensor([[1, 2, 3], [2, 0, -1]])
indices = tensor([2, 0], "int64")
probabilities = tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
errors = {
    "input-type": error(lambda: F.cross_entropy([[1.0]], indices)),
    "target-type": error(lambda: F.cross_entropy(source, [2, 0])),
    "input-rank": error(lambda: F.cross_entropy(tensor(1.0), tensor(0, "int64"))),
    "class-empty": error(lambda: F.cross_entropy(
        tensor(np.empty((2, 0), dtype=np.float32)),
        tensor([0, 0], "int64"),
    )),
    "index-shape": error(lambda: F.cross_entropy(source, tensor([[2], [0]], "int64"))),
    "index-dtype": error(lambda: F.cross_entropy(source, tensor([2, 0], "int32"))),
    "probability-dtype": error(lambda: F.cross_entropy(
        source, tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], "float64")
    )),
    "probability-ignore": error(lambda: F.cross_entropy(
        source, probabilities, ignore_index=0
    )),
    "weight-type": error(lambda: F.cross_entropy(source, indices, weight=[1, 2, 4])),
    "weight-shape": error(lambda: F.cross_entropy(source, indices, weight=tensor([1, 2]))),
    "weight-dtype": error(lambda: F.cross_entropy(
        source, indices, weight=tensor([1, 2, 4], "float64")
    )),
    "weight-grad": error(lambda: F.cross_entropy(
        source, indices, weight=tensor([1, 2, 4], requires_grad=True)
    )),
    "reduction": error(lambda: F.cross_entropy(source, indices, reduction="median")),
    "reduction-type": error(lambda: F.cross_entropy(source, indices, reduction=object())),
    "ignore-type": error(lambda: F.cross_entropy(source, indices, ignore_index=True)),
    "ignore-range": error(lambda: F.cross_entropy(source, indices, ignore_index=1 << 70)),
    "smoothing-type": error(lambda: F.cross_entropy(source, indices, label_smoothing=True)),
    "smoothing-range": error(lambda: F.cross_entropy(source, indices, label_smoothing=-0.1)),
    "size-average-type": error(lambda: F.cross_entropy(source, indices, size_average=1)),
    "reduce-type": error(lambda: F.cross_entropy(source, indices, reduce=1)),
    "target-range": error(lambda: F.cross_entropy(source, tensor([2, 3], "int64"))),
}

class HostileArray(np.ndarray):
    pass
errors["array-subclass"] = error(lambda: F.cross_entropy(
    source,
    bg.Tensor(np.asarray([2, 0], dtype=np.int64).view(HostileArray), dtype="int64"),
))

work = fixture["limits"]["workExtent"]
workspace = fixture["limits"]["workspaceExtent"]
errors["work"] = error(lambda: F.cross_entropy(
    tensor(np.empty((1, work), dtype=np.float32)),
    tensor([0], "int64"),
))
errors["workspace"] = error(lambda: F.cross_entropy(
    tensor(np.empty((workspace, 1), dtype=np.float64), "float64"),
    tensor(np.zeros(workspace, dtype=np.int64), "int64"),
))
errors["zero-hidden-work"] = error(lambda: F.cross_entropy(
    tensor(np.empty((0, 1, work), dtype=np.float32)),
    tensor(np.empty((0, work), dtype=np.int64), "int64"),
))
errors
`);

    const messages: Record<string, string> = {
      "input-type": "input must be",
      "target-type": "target must be",
      "input-rank": "no user input dimension",
      "class-empty": "at least one class",
      "index-shape": "index target shape",
      "index-dtype": "index target dtype",
      "probability-dtype": "probability target dtype",
      "probability-ignore": "ignore_index is not supported",
      "weight-type": "weight must be",
      "weight-shape": "weight shape",
      "weight-dtype": "weight dtype",
      "weight-grad": "must not require grad",
      reduction: "reduction must be 'none', 'sum', or 'mean'",
      "reduction-type": "reduction must be a string",
      "ignore-type": "ignore_index must be an exact integer",
      "ignore-range": "ignore_index must fit signed int64",
      "smoothing-type": "label_smoothing must be an exact real scalar",
      "smoothing-range": "label_smoothing must be finite and in [0, 1]",
      "size-average-type": "size_average must be an exact bool or None",
      "reduce-type": "reduce must be an exact bool or None",
      "target-range": "target values must be in",
    };
    for (const [key, message] of Object.entries(messages)) {
      expect(errors[key]).toContain(message);
    }
    expect(errors["array-subclass"]).toContain("exact ndarray");
    expect(errors.work).toContain("work");
    expect(errors.workspace).toContain("workspace");
    expect(errors["zero-hidden-work"]).toContain("work");
  });
});
