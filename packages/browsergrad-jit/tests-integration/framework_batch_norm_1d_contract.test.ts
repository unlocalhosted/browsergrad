import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_BATCH_NORM_1D_CONFORMANCE } from "../../../test-support/framework-batch-norm1d-conformance";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.nn.BatchNorm1d contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("owns typed forward/state IR and commits unbiased running statistics exactly once", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BATCH_NORM_1D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
x = bg.from_numpy(
    np.asarray(fixture["input2d"], dtype=np.float32),
    requires_grad=True,
)
bn = bg.nn.BatchNorm1d(
    2,
    eps=fixture["eps"],
    momentum=fixture["momentum"],
    affine=False,
)
running_mean_id = id(bn.running_mean)
running_var_id = id(bn.running_var)
tracked_id = id(bn.num_batches_tracked)
y = bn(x)
first = y.numpy()
second = y.numpy()
after_replay = int(bn.num_batches_tracked)
y.sum().backward()

# Two lazy forwards must preserve construction order even if the later output
# is the first one realized.
ordered = bg.nn.BatchNorm1d(2, momentum=1.0, affine=False)
first_input = bg.from_numpy(np.asarray(fixture["input2d"], dtype=np.float32))
second_values = np.asarray(fixture["input2d"], dtype=np.float32) + 10.0
second_input = bg.from_numpy(second_values)
first_output = ordered(first_input)
second_output = ordered(second_input)
second_output.numpy()
first_output.numpy()

stream_count_before = len(x._get_session().buffer_table._effect_streams)
bounded = bg.nn.BatchNorm1d(2, affine=False)
for _ in range(32):
    bounded(bg.from_numpy(
        np.asarray(fixture["input2d"], dtype=np.float32)
    )).numpy()
stream_count_after = len(x._get_session().buffer_table._effect_streams)

{
    "opcode": y._uop.op,
    "statsOpcode": y._uop.inputs[1].op,
    "argFields": sorted(y._uop.arg.keys()),
    "valuesReplay": bool(np.array_equal(first, second)),
    "batchMean": first.mean(axis=0).round(6).tolist(),
    "batchVariance": first.var(axis=0).round(5).tolist(),
    "runningMean": bn.running_mean.round(6).tolist(),
    "runningVariance": bn.running_var.round(6).tolist(),
    "trackedAfterReplay": after_replay,
    "trackedAfterBackward": int(bn.num_batches_tracked),
    "bufferIdentityStable": (
        running_mean_id == id(bn.running_mean)
        and running_var_id == id(bn.running_var)
        and tracked_id == id(bn.num_batches_tracked)
    ),
    "orderedTracked": int(ordered.num_batches_tracked),
    "orderedMean": ordered.running_mean.round(6).tolist(),
    "expectedOrderedMean": second_values.mean(axis=0).round(6).tolist(),
    "effectStreamDelta": stream_count_after - stream_count_before,
    "boundedTracked": int(bounded.num_batches_tracked),
}
`);

    expect(result).toMatchObject({
      opcode: "BATCH_NORM_1D",
      statsOpcode: "BATCH_NORM_1D_STATS_UPDATE",
      argFields: ["affine", "eps", "stats_mode"],
      valuesReplay: true,
      batchMean: [0, 0],
      batchVariance: [1, 1],
      runningMean: FRAMEWORK_BATCH_NORM_1D_CONFORMANCE.expectedBatchMean2d,
      runningVariance:
        FRAMEWORK_BATCH_NORM_1D_CONFORMANCE.expectedUnbiasedRunningVariance2d,
      trackedAfterReplay: 1,
      trackedAfterBackward: 1,
      bufferIdentityStable: true,
      orderedTracked: 2,
      effectStreamDelta: 1,
      boundedTracked: 32,
    });
    expect(result.orderedMean).toEqual(result.expectedOrderedMean);
  });

  it("matches closure and symbolic gradients across batch, running, affine, and untracked modes", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BATCH_NORM_1D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import json
import numpy as np
from browsergrad_jit._realize import realize

fixture = json.loads(${JSON.stringify(fixtureJson)})
values = np.asarray(fixture["input2d"], dtype=np.float32)
upstream_values = np.asarray(fixture["upstream2d"], dtype=np.float32)
x = bg.from_numpy(values, requires_grad=True)
bn = bg.nn.BatchNorm1d(2, affine=True, track_running_stats=False)
bn.weight._get_session().buffer_table.update(
    bn.weight._uop.inputs[0].arg,
    np.asarray(fixture["affineWeight"], dtype=np.float32),
)
bn.bias._get_session().buffer_table.update(
    bn.bias._uop.inputs[0].arg,
    np.asarray(fixture["affineBias"], dtype=np.float32),
)
y = bn(x)
upstream = bg.from_numpy(upstream_values)
session = x._get_session()
closure_inputs = tuple(
    realize(proxy._uop, session.buffer_table)
    for proxy in y._ctx.input_proxies
)
closure = y._ctx.fn(upstream_values, closure_inputs)
y.backward(upstream)

functional_source = bg.from_numpy(values)
functional_input_gradient = bg.func.grad(
    lambda value: bg.nn.BatchNorm1d(
        2, affine=False, track_running_stats=False
    )(value).sum()
)(functional_source).numpy()

three_d = bg.from_numpy(
    np.asarray(fixture["input3d"], dtype=np.float32),
    requires_grad=True,
)
three_d_module = bg.nn.BatchNorm1d(2, affine=False)
three_d_output = three_d_module(three_d)
three_d_output.sum().backward()

tracked = bg.nn.BatchNorm1d(2, momentum=1.0, affine=False)
tracked(bg.from_numpy(values)).numpy()
tracked.eval()
eval_input = bg.from_numpy(values + 2.0, requires_grad=True)
eval_output = tracked(eval_input)
expected_eval = (
    (values + 2.0 - np.asarray(fixture["expectedBatchMean2d"], dtype=np.float32))
    / np.sqrt(
        np.asarray(fixture["expectedUnbiasedRunningVariance2d"], dtype=np.float32)
        + fixture["eps"]
    )
)
# The graph owns an immutable running-stat snapshot.
tracked.running_mean[...] = np.float32(100.0)
eval_values = eval_output.numpy()
eval_output.sum().backward()

{
    "closureInputMatches": bool(np.allclose(closure[0], x.grad.numpy(), atol=1e-6)),
    "closureWeightMatches": bool(np.allclose(closure[1], bn.weight.grad.numpy(), atol=1e-6)),
    "closureBiasMatches": bool(np.allclose(closure[2], bn.bias.grad.numpy(), atol=1e-6)),
    "functionalFinite": bool(np.isfinite(functional_input_gradient).all()),
    "functionalSumNearZero": bool(abs(float(functional_input_gradient.sum())) < 1e-5),
    "threeDShape": list(three_d_output.shape),
    "threeDGradientFinite": bool(np.isfinite(three_d.grad.numpy()).all()),
    "evalSnapshotStable": bool(np.allclose(eval_values, expected_eval, atol=1e-6)),
    "evalGradientFinite": bool(np.isfinite(eval_input.grad.numpy()).all()),
    "untrackedEvalUsesBatch": bool(abs(float(
        bg.nn.BatchNorm1d(
            2, affine=False, track_running_stats=False
        ).eval()(bg.from_numpy(values)).numpy().mean()
    )) < 1e-6),
}
`);

    expect(result).toEqual({
      closureInputMatches: true,
      closureWeightMatches: true,
      closureBiasMatches: true,
      functionalFinite: true,
      functionalSumNearZero: true,
      threeDShape: [2, 2, 2],
      threeDGradientFinite: true,
      evalSnapshotStable: true,
      evalGradientFinite: true,
      untrackedEvalUsesBatch: true,
    });
  });

  it("preserves state through checkpoints and cumulative averaging", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BATCH_NORM_1D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import json
import numpy as np
from browsergrad_jit.utils.checkpoint import checkpoint

fixture = json.loads(${JSON.stringify(fixtureJson)})
values = np.asarray(fixture["input2d"], dtype=np.float32)
module = bg.nn.BatchNorm1d(2, momentum=None, affine=False)
source = bg.from_numpy(values, requires_grad=True)
output = checkpoint(lambda value: module(value), source)
output.numpy()
before_backward = int(module.num_batches_tracked)
output.sum().backward()
after_backward = int(module.num_batches_tracked)

module(bg.from_numpy(values + 4.0)).numpy()
expected_cumulative_mean = np.stack((values, values + 4.0)).mean(axis=(0, 1))
state = module.state_dict()

{
    "beforeBackward": before_backward,
    "afterBackward": after_backward,
    "afterSecond": int(module.num_batches_tracked),
    "cumulativeMean": module.running_mean.round(6).tolist(),
    "expectedCumulativeMean": expected_cumulative_mean.round(6).tolist(),
    "stateKeys": sorted(state.keys()),
    "gradientFinite": bool(np.isfinite(source.grad.numpy()).all()),
}
`);

    expect(result).toMatchObject({
      beforeBackward: 1,
      afterBackward: 1,
      afterSecond: 2,
      stateKeys: [
        "num_batches_tracked",
        "running_mean",
        "running_var",
      ],
      gradientFinite: true,
    });
    expect(result.cumulativeMean).toEqual(result.expectedCumulativeMean);
  });

  it("fails closed for malformed contracts, forged effects, and unevidenced routes", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_BATCH_NORM_1D_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import gc
import json
import numpy as np
from browsergrad_jit import _vmap
from browsergrad_jit._framework_contracts import (
    BATCH_NORM_1D_OUTPUT_BYTE_MAX,
    BATCH_NORM_1D_OUTPUT_EXTENT_MAX,
    BATCH_NORM_1D_WORK_ELEMENT_MAX,
    BATCH_NORM_1D_WORKSPACE_BYTE_MAX,
    BATCH_NORM_1D_WORK_VISIT_FACTOR,
    infer_batch_norm_1d_contract,
)

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

values = np.asarray(fixture["input2d"], dtype=np.float32)
source = bg.from_numpy(values)
module = bg.nn.BatchNorm1d(2, affine=False)
output = module(source)
state_node = output._uop.inputs[1]
original_effect = state_node.arg["effect_id"]
state_node.arg["effect_id"] = "forged:effect"
forged_effect = error(output.numpy)
state_node.arg["effect_id"] = original_effect

other_module = bg.nn.BatchNorm1d(2, affine=False)
other_output = other_module(source)
state_node.arg["effect_id"] = other_output._uop.inputs[1].arg["effect_id"]
wrong_effect_targets = error(output.numpy)
state_node.arg["effect_id"] = original_effect

eval_module = bg.nn.BatchNorm1d(2, affine=False)
eval_module.eval()
eval_output = eval_module(source)

class Spec:
    def __init__(self, shape, dtype="float32"):
        self.shape = shape
        self.dtype = dtype
        self.op = "LOAD"

errors = {
    "rank": error(lambda: bg.nn.BatchNorm1d(2)(
        bg.from_numpy(np.ones((2, 2, 1, 1), dtype=np.float32))
    )),
    "channels": error(lambda: bg.nn.BatchNorm1d(3)(source)),
    "dtype": error(lambda: bg.nn.BatchNorm1d(2)(
        bg.from_numpy(np.ones((2, 2), dtype=np.float64))
    )),
    "sampleCount": error(lambda: bg.nn.BatchNorm1d(2)(
        bg.from_numpy(np.ones((1, 2), dtype=np.float32))
    )),
    "numFeatures": error(lambda: bg.nn.BatchNorm1d(True)),
    "eps": error(lambda: bg.nn.BatchNorm1d(2, eps=float("nan"))),
    "momentum": error(lambda: bg.nn.BatchNorm1d(2, momentum=1.1)),
    "affine": error(lambda: bg.nn.BatchNorm1d(2, affine=1)),
    "tracking": error(lambda: bg.nn.BatchNorm1d(2, track_running_stats=1)),
    "extent": error(lambda: infer_batch_norm_1d_contract(
        (Spec((2, fixture["limits"]["outputExtent"] + 1)),),
        fixture["eps"],
        False,
        "batch",
    )),
    "forgedEffect": forged_effect,
    "wrongEffectTargets": wrong_effect_targets,
    "vmap": error(lambda: _vmap._VMAP_RULES["BATCH_NORM_1D"](
        eval_output._uop, {}, 2
    )),
    "onnx": error(lambda: bg.onnx.export_inference(
        eval_output,
        input_buffers=(source,),
    )),
    "plan": error(lambda: bg.gpu_plan_summary(eval_output)),
}

# A long-lived session retains buffers after short-lived modules are collected.
# Fresh module storage must never depend on a recyclable Python object id.
lifecycle_buffer_ids = []
for _ in range(32):
    candidate = bg.nn.BatchNorm1d(2, affine=False)
    candidate_output = candidate(source)
    lifecycle_buffer_ids.append(candidate_output._uop.inputs[1].inputs[-3].arg)
    del candidate_output
    del candidate
    gc.collect()

original_eps = eval_output._uop.arg["eps"]
eval_output._uop.arg["eps"] = True
errors["mutation"] = error(eval_output.numpy)
eval_output._uop.arg["eps"] = original_eps

{
    "errors": errors,
    "uniqueLifecycleBuffers": len(set(lifecycle_buffer_ids)),
    "limits": {
        "outputBytes": BATCH_NORM_1D_OUTPUT_BYTE_MAX,
        "outputExtent": BATCH_NORM_1D_OUTPUT_EXTENT_MAX,
        "workElements": BATCH_NORM_1D_WORK_ELEMENT_MAX,
        "workspaceBytes": BATCH_NORM_1D_WORKSPACE_BYTE_MAX,
        "workVisitFactor": BATCH_NORM_1D_WORK_VISIT_FACTOR,
    },
}
`);

    expect(result.limits).toEqual(
      FRAMEWORK_BATCH_NORM_1D_CONFORMANCE.limits,
    );
    expect(result.uniqueLifecycleBuffers).toBe(32);
    for (const value of Object.values(result.errors as Record<string, string>)) {
      expect(value).not.toBe("no_error");
    }
    expect((result.errors as Record<string, string>).forgedEffect).toMatch(
      /RealizationError/u,
    );
    expect((result.errors as Record<string, string>).wrongEffectTargets).toMatch(
      /RealizationError/u,
    );
    expect((result.errors as Record<string, string>).vmap).toMatch(
      /JitNotImplementedError/u,
    );
    expect((result.errors as Record<string, string>).onnx).toMatch(
      /OnnxUnmappableOp/u,
    );
    expect((result.errors as Record<string, string>).plan).toMatch(
      /GpuPlanUnsupported/u,
    );
  });
});
