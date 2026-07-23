import { beforeAll, describe, expect, it } from "vitest";

import { FRAMEWORK_BATCH_NORM_1D_CONFORMANCE } from "../../../test-support/framework-batch-norm1d-conformance";
import { getGradTarget } from "./pyodide-host";

describe("shared torch.nn.BatchNorm1d conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  it("uses biased batch normalization and unbiased persistent variance", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BATCH_NORM_1D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
values = np.asarray(fixture["input2d"], dtype=np.float32)
module = bg.nn.BatchNorm1d(2, momentum=1.0, affine=False)
mean_id = id(module.running_mean)
var_id = id(module.running_var)
tracked_id = id(module.num_batches_tracked)
output = module(bg.Tensor(values))
state = module.state_dict()

{
    "batchMean": output.data.mean(axis=0).round(6).tolist(),
    "batchVariance": output.data.var(axis=0).round(5).tolist(),
    "runningMean": module.running_mean.data.round(6).tolist(),
    "runningVariance": module.running_var.data.round(6).tolist(),
    "tracked": int(module.num_batches_tracked.data),
    "identityStable": (
        mean_id == id(module.running_mean)
        and var_id == id(module.running_var)
        and tracked_id == id(module.num_batches_tracked)
    ),
    "stateKeys": sorted(state.keys()),
}
`);

    expect(result).toEqual({
      batchMean: [0, 0],
      batchVariance: [1, 1],
      runningMean: FRAMEWORK_BATCH_NORM_1D_CONFORMANCE.expectedBatchMean2d,
      runningVariance:
        FRAMEWORK_BATCH_NORM_1D_CONFORMANCE.expectedUnbiasedRunningVariance2d,
      tracked: 1,
      identityStable: true,
      stateKeys: [
        "num_batches_tracked",
        "running_mean",
        "running_var",
      ],
    });
  });

  it("matches affine gradients, 3D geometry, eval, untracked, and cumulative modes", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BATCH_NORM_1D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
values = np.asarray(fixture["input2d"], dtype=np.float32)
upstream = np.asarray(fixture["upstream2d"], dtype=np.float32)
source = bg.Tensor(values, requires_grad=True)
module = bg.nn.BatchNorm1d(2, affine=True, track_running_stats=False)
module.weight.data[...] = np.asarray(fixture["affineWeight"], dtype=np.float32)
module.bias.data[...] = np.asarray(fixture["affineBias"], dtype=np.float32)
output = module(source)
output.backward(bg.Tensor(upstream))

three_d_source = bg.Tensor(
    np.asarray(fixture["input3d"], dtype=np.float32),
    requires_grad=True,
)
three_d_module = bg.nn.BatchNorm1d(2, affine=False)
three_d_output = three_d_module(three_d_source)
three_d_output.sum().backward()

tracked = bg.nn.BatchNorm1d(2, momentum=1.0, affine=False)
tracked(bg.Tensor(values))
tracked.eval()
eval_values = values + 2.0
eval_source = bg.Tensor(eval_values, requires_grad=True)
eval_output = tracked(eval_source)
eval_output.sum().backward()
expected_eval = (
    (eval_values - np.asarray(fixture["expectedBatchMean2d"], dtype=np.float32))
    / np.sqrt(
        np.asarray(fixture["expectedUnbiasedRunningVariance2d"], dtype=np.float32)
        + fixture["eps"]
    )
)

cumulative = bg.nn.BatchNorm1d(2, momentum=None, affine=False)
cumulative(bg.Tensor(values))
cumulative(bg.Tensor(values + 4.0))
expected_cumulative = np.stack((values, values + 4.0)).mean(axis=(0, 1))

untracked = bg.nn.BatchNorm1d(
    2, affine=False, track_running_stats=False
).eval()
untracked_output = untracked(bg.Tensor(values))

{
    "inputGradientFinite": bool(np.isfinite(source.grad.data).all()),
    "weightGradient": module.weight.grad.data.round(6).tolist(),
    "biasGradient": module.bias.grad.data.round(6).tolist(),
    "threeDShape": list(three_d_output.shape),
    "threeDGradientFinite": bool(np.isfinite(three_d_source.grad.data).all()),
    "evalMatches": bool(np.allclose(eval_output.data, expected_eval, atol=1e-6)),
    "evalGradientFinite": bool(np.isfinite(eval_source.grad.data).all()),
    "untrackedEvalUsesBatch": bool(abs(float(untracked_output.data.mean())) < 1e-6),
    "cumulativeTracked": int(cumulative.num_batches_tracked.data),
    "cumulativeMean": cumulative.running_mean.data.round(6).tolist(),
    "expectedCumulativeMean": expected_cumulative.round(6).tolist(),
}
`);

    expect(result).toMatchObject({
      inputGradientFinite: true,
      threeDShape: [2, 2, 2],
      threeDGradientFinite: true,
      evalMatches: true,
      evalGradientFinite: true,
      untrackedEvalUsesBatch: true,
      cumulativeTracked: 2,
    });
    expect(result.cumulativeMean).toEqual(result.expectedCumulativeMean);
    expect(result.weightGradient).toHaveLength(2);
    expect(result.biasGradient).toEqual([4, 7]);
  });

  it("rejects coercive constructor, shape, dtype, and singleton-stat requests", async () => {
    const target = await getGradTarget();
    const result = await target.run<Record<string, string>>(`
import browsergrad_grad as bg
import numpy as np

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "numFeatures": error(lambda: bg.nn.BatchNorm1d(True)),
    "eps": error(lambda: bg.nn.BatchNorm1d(2, eps=float("nan"))),
    "momentum": error(lambda: bg.nn.BatchNorm1d(2, momentum=1.1)),
    "affine": error(lambda: bg.nn.BatchNorm1d(2, affine=1)),
    "tracking": error(lambda: bg.nn.BatchNorm1d(2, track_running_stats=1)),
    "rank": error(lambda: bg.nn.BatchNorm1d(2)(
        bg.Tensor(np.ones((2, 2, 1, 1), dtype=np.float32))
    )),
    "channels": error(lambda: bg.nn.BatchNorm1d(3)(
        bg.Tensor(np.ones((2, 2), dtype=np.float32))
    )),
    "dtype": error(lambda: bg.nn.BatchNorm1d(2)(
        bg.Tensor(np.ones((2, 2), dtype=np.float64), dtype="float64")
    )),
    "sampleCount": error(lambda: bg.nn.BatchNorm1d(2)(
        bg.Tensor(np.ones((1, 2), dtype=np.float32))
    )),
}
`);

    for (const [name, value] of Object.entries(result)) {
      expect(value, name).not.toBe("no_error");
    }
  });
});
