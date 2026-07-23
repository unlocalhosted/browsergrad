import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_CROSS_ENTROPY_CONFORMANCE } from "../../../test-support/framework-cross-entropy-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 typed torch.nn.functional.cross_entropy contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches weighted index and probability targets, smoothing, reductions, modules, and empties", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CROSS_ENTROPY_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
base = fixture["base"]

def tensor(shape, values, dtype="float32", requires_grad=False, session=None):
    return bg.from_numpy(
        np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape)),
        requires_grad=requires_grad,
        session=session,
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
index_mean.backward()

probability_input = tensor(
    base["inputShape"], base["input"], requires_grad=True
)
probability_target = tensor(
    base["inputShape"],
    base["probabilityTarget"],
    requires_grad=True,
)
probability_none = F.cross_entropy(
    probability_input,
    probability_target,
    weight=weight,
    reduction="none",
    label_smoothing=base["labelSmoothing"],
)
probability_sum = F.cross_entropy(
    tensor(base["inputShape"], base["input"]),
    tensor(base["inputShape"], base["probabilityTarget"]),
    weight=weight,
    reduction="sum",
    label_smoothing=base["labelSmoothing"],
)
probability_mean = F.cross_entropy(
    probability_input,
    probability_target,
    weight=weight,
    label_smoothing=base["labelSmoothing"],
)
closure = probability_mean._ctx.fn(
    np.asarray(1.0, dtype=np.float32),
    (
        probability_input.numpy(),
        probability_target.numpy(),
        weight.numpy(),
    ),
)
probability_mean.backward()

ignore_input = tensor(base["inputShape"], base["input"], requires_grad=True)
ignore_target = tensor([2], [2, -100], "int64")
ignore_loss = F.cross_entropy(
    ignore_input,
    ignore_target,
    weight=weight,
    reduction="sum",
    label_smoothing=base["labelSmoothing"],
)
ignore_loss.backward()
all_ignore_input = tensor(base["inputShape"], base["input"], requires_grad=True)
all_ignore = F.cross_entropy(
    all_ignore_input,
    tensor([2], [-100, -100], "int64"),
)
all_ignore.backward()

spatial = F.cross_entropy(
    tensor([1, 3, 2], [1, 0, 2, -1, 3, 1]),
    tensor([1, 2], [2, 0], "int64"),
    reduction="none",
)
unbatched_input = tensor([3], [1, 2, 3], requires_grad=True)
unbatched = F.cross_entropy(
    unbatched_input,
    tensor([], [2], "int64"),
)
unbatched.backward()

empty_input = tensor([0, 3], [], requires_grad=True)
empty_target = tensor([0], [], "int64")
empty_none = F.cross_entropy(empty_input, empty_target, reduction="none")
empty_sum = F.cross_entropy(empty_input, empty_target, reduction="sum")
empty_mean = F.cross_entropy(empty_input, empty_target)
empty_mean.backward()

dtype_results = []
for dtype in fixture["mixedDtypes"]:
    source = tensor(base["inputShape"], base["input"], dtype, True)
    labels = tensor([2], base["indexTarget"], "int64")
    output = F.cross_entropy(source, labels, reduction="sum")
    output.backward()
    dtype_results.append({
        "output": output.dtype,
        "gradient": source.grad.dtype,
    })

legacy_sum = F.cross_entropy(
    tensor(base["inputShape"], base["input"]),
    index_target,
    weight=weight,
    size_average=False,
    label_smoothing=base["labelSmoothing"],
)
legacy_none = F.cross_entropy(
    tensor(base["inputShape"], base["input"]),
    index_target,
    reduce=False,
)

bg.install_torch_alias()
import torch
module = torch.nn.CrossEntropyLoss(
    weight=weight,
    reduction="sum",
    label_smoothing=base["labelSmoothing"],
)
module_value = module(
    tensor(base["inputShape"], base["input"]),
    index_target,
)
other = bg.new_session()
module_other_value = module(
    tensor(base["inputShape"], base["input"], session=other),
    tensor([2], base["indexTarget"], "int64", session=other),
)

with bg.no_grad():
    detached = F.cross_entropy(
        tensor(base["inputShape"], base["input"], requires_grad=True),
        index_target,
    )

{
    "op": index_none._uop.op,
    "arg": index_none._uop.arg,
    "indexNone": index_none.numpy().tolist(),
    "indexSum": float(index_sum.item()),
    "indexMean": float(index_mean.item()),
    "indexGradient": index_input.grad.numpy().reshape(-1).tolist(),
    "probabilityNone": probability_none.numpy().tolist(),
    "probabilitySum": float(probability_sum.item()),
    "probabilityMean": float(probability_mean.item()),
    "probabilityInputGradient": probability_input.grad.numpy().reshape(-1).tolist(),
    "probabilityTargetGradient": probability_target.grad.numpy().reshape(-1).tolist(),
    "closureInput": closure[0].reshape(-1).tolist(),
    "closureTarget": closure[1].reshape(-1).tolist(),
    "closureWeightNone": closure[2] is None,
    "ignore": float(ignore_loss.item()),
    "allIgnoreNan": bool(np.isnan(all_ignore.item())),
    "allIgnoreGradient": all_ignore_input.grad.numpy().reshape(-1).tolist(),
    "spatialShape": list(spatial.shape),
    "unbatched": float(unbatched.item()),
    "unbatchedGradient": unbatched_input.grad.numpy().tolist(),
    "empty": {
        "noneShape": list(empty_none.shape),
        "sum": float(empty_sum.item()),
        "meanNan": bool(np.isnan(empty_mean.item())),
        "gradientShape": list(empty_input.grad.shape),
    },
    "dtypes": dtype_results,
    "legacySum": float(legacy_sum.item()),
    "legacyNoneShape": list(legacy_none.shape),
    "module": float(module_value.item()),
    "moduleOther": float(module_other_value.item()),
    "moduleState": {
        key: value.reshape(-1).tolist()
        for key, value in module.state_dict().items()
    },
    "ownsData": bool(index_none.numpy().flags["OWNDATA"]),
    "detached": detached.requires_grad,
}
`);

    const fixture = FRAMEWORK_CROSS_ENTROPY_CONFORMANCE;
    expect(result.op).toBe("CROSS_ENTROPY");
    expect(result.arg).toEqual({
      reduction: "none",
      batch_rank: 0,
      ignore_index: -100,
      has_weight: true,
      label_smoothing: fixture.base.labelSmoothing,
      target_mode: "indices",
    });
    expectCloseArray(result.indexNone, fixture.base.indexNone);
    expect(result.indexSum).toBeCloseTo(fixture.base.indexSum, 5);
    expect(result.indexMean).toBeCloseTo(fixture.base.indexMean, 5);
    expectCloseArray(result.indexGradient, fixture.base.indexMeanGradient);
    expectCloseArray(result.probabilityNone, fixture.base.probabilityNone);
    expect(result.probabilitySum).toBeCloseTo(fixture.base.probabilitySum, 5);
    expect(result.probabilityMean).toBeCloseTo(fixture.base.probabilityMean, 5);
    expectCloseArray(
      result.probabilityInputGradient,
      fixture.base.probabilityMeanInputGradient,
    );
    expectCloseArray(
      result.probabilityTargetGradient,
      fixture.base.probabilityMeanTargetGradient,
    );
    expectCloseArray(result.closureInput, fixture.base.probabilityMeanInputGradient);
    expectCloseArray(result.closureTarget, fixture.base.probabilityMeanTargetGradient);
    expect(result.closureWeightNone).toBe(true);
    expect(result.ignore).toBeCloseTo(fixture.base.indexNone[0]!, 5);
    expect(result.allIgnoreNan).toBe(true);
    expectCloseArray(result.allIgnoreGradient, [0, 0, 0, 0, 0, 0]);
    expect(result.spatialShape).toEqual([1, 2]);
    expect(result.unbatched).toBeCloseTo(0.4076059644, 5);
    expectCloseArray(result.unbatchedGradient, [
      0.0900305732,
      0.2447284711,
      -0.3347590443,
    ]);
    expect(result.empty).toEqual({
      noneShape: [0],
      sum: 0,
      meanNan: true,
      gradientShape: [0, 3],
    });
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map((dtype) => ({
      output: dtype,
      gradient: dtype,
    })));
    expect(result.legacySum).toBeCloseTo(fixture.base.indexSum, 5);
    expect(result.legacyNoneShape).toEqual([2]);
    expect(result.module).toBeCloseTo(fixture.base.indexSum, 5);
    expect(result.moduleOther).toBeCloseTo(fixture.base.indexSum, 5);
    expect(result.moduleState).toEqual({ weight: fixture.base.weight });
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
  });

  it("supports symbolic VJP and vmap, exports index ONNX, and refuses unsupported lowering", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CROSS_ENTROPY_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import (
    OP_CROSS_ENTROPY,
    OP_CROSS_ENTROPY_VJP,
    toposort,
)
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})
base = fixture["base"]
inputs = np.asarray(base["input"], dtype=np.float32).reshape(base["inputShape"])
indices = np.asarray(base["indexTarget"], dtype=np.int64)
probabilities = np.asarray(
    base["probabilityTarget"], dtype=np.float32
).reshape(base["inputShape"])
weights = np.asarray(base["weight"], dtype=np.float32)

source = bg.from_numpy(inputs)
labels = bg.from_numpy(indices)
weight = bg.from_numpy(weights)
relu_source = bg.from_numpy(
    np.asarray([-1.0, 2.0], dtype=np.float32)
)
relu_gradient = bg.func.grad(
    lambda value: F.relu(value).sum()
)(relu_source)
index_loss = F.cross_entropy(
    source,
    labels,
    weight=weight,
    label_smoothing=base["labelSmoothing"],
)
index_gradient = bg.func.grad(lambda value: F.cross_entropy(
    value,
    labels,
    weight=weight,
    label_smoothing=base["labelSmoothing"],
))(source)
probability_target = bg.from_numpy(probabilities)
target_gradient = bg.func.grad(lambda value: F.cross_entropy(
    source,
    value,
    weight=weight,
    label_smoothing=base["labelSmoothing"],
))(probability_target)
symbolic = get_rule(OP_CROSS_ENTROPY)(
    index_loss._uop,
    index_loss._uop.inputs,
    bg.ones()._uop,
)

batched_source = bg.from_numpy(np.stack([inputs, inputs + 0.5]))
batched_indices = bg.from_numpy(np.stack([indices, indices]))
mapped_index = bg.func.vmap(lambda value, target: F.cross_entropy(
    value,
    target,
    weight=weight,
    reduction="mean",
    label_smoothing=base["labelSmoothing"],
))(batched_source, batched_indices)
batched_probabilities = bg.from_numpy(
    np.stack([probabilities, probabilities])
)
mapped_probability = bg.func.vmap(lambda value, target: F.cross_entropy(
    value,
    target,
    weight=weight,
    reduction="none",
    label_smoothing=base["labelSmoothing"],
))(batched_source, batched_probabilities)
batched_weights = bg.from_numpy(np.stack([weights, weights * 2]))
mapped_weight = bg.func.vmap(lambda value, target, class_weight: F.cross_entropy(
    value,
    target,
    weight=class_weight,
    reduction="sum",
    label_smoothing=base["labelSmoothing"],
))(batched_source, batched_indices, batched_weights)

export_loss = F.cross_entropy(
    source,
    labels,
    weight=weight,
    reduction="sum",
)
model = bg.onnx.export_inference(
    export_loss,
    input_buffers=(source, labels, weight),
)
rank_one_source = bg.from_numpy(np.asarray([1, 2, 3], dtype=np.float32))
scalar_label = bg.from_numpy(np.asarray(2, dtype=np.int64))
rank_one_model = bg.onnx.export_inference(
    F.cross_entropy(rank_one_source, scalar_label, reduction="none"),
    input_buffers=(rank_one_source, scalar_label),
)

${ONNX_PROTOBUF_TEST_HELPERS}

def op_types(model):
    graph = next(
        payload for number, wire, payload in fields(model)
        if number == 7 and wire == 2
    )
    return [
        next(
            value.decode("utf-8")
            for field, kind, value in fields(payload)
            if field == 4 and kind == 2
        )
        for number, wire, payload in fields(graph)
        if number == 1 and wire == 2
    ]

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

support = next(
    item for item in bg.framework_operation_support()["operations"]
    if item["opcode"] == OP_CROSS_ENTROPY
)

{
    "reluDtype": F.relu(relu_source).dtype,
    "reluGradient": relu_gradient.numpy().tolist(),
    "indexGradient": index_gradient.numpy().reshape(-1).tolist(),
    "targetGradient": target_gradient.numpy().reshape(-1).tolist(),
    "symbolicOps": [node.op for node in toposort(symbolic[0])],
    "symbolicNondifferentiable": [value is None for value in symbolic[1:]],
    "mappedIndex": mapped_index.numpy().tolist(),
    "mappedIndexBatchRank": mapped_index._uop.arg["batch_rank"],
    "mappedProbabilityShape": list(mapped_probability.shape),
    "mappedProbabilityMode": mapped_probability._uop.arg["target_mode"],
    "mappedWeight": mapped_weight.numpy().tolist(),
    "onnx": op_types(model),
    "rankOneOnnx": op_types(rank_one_model),
    "probabilityOnnxError": error(lambda: bg.onnx.export_inference(
        F.cross_entropy(source, probability_target)
    )),
    "smoothingOnnxError": error(lambda: bg.onnx.export_inference(index_loss)),
    "mappedOnnxError": error(lambda: bg.onnx.export_inference(mapped_index)),
    "planError": error(lambda: bg.gpu_plan_summary(index_loss)),
    "webgpuSupported": (
        OP_CROSS_ENTROPY in supported_opcodes()
        or OP_CROSS_ENTROPY_VJP in supported_opcodes()
    ),
    "support": support,
}
`);

    const fixture = FRAMEWORK_CROSS_ENTROPY_CONFORMANCE;
    expect(result.reluDtype).toBe("float32");
    expectCloseArray(result.reluGradient, [0, 1]);
    expectCloseArray(result.indexGradient, fixture.base.indexMeanGradient);
    expectCloseArray(
      result.targetGradient,
      fixture.base.probabilityMeanTargetGradient,
    );
    expect(result.symbolicOps).toContain("CROSS_ENTROPY_VJP");
    expect(result.symbolicOps).not.toContain("CUSTOM");
    expect(result.symbolicNondifferentiable).toEqual([true, true]);
    expectCloseArray(result.mappedIndex, [
      fixture.base.indexMean,
      fixture.base.indexMean,
    ]);
    expect(result.mappedIndexBatchRank).toBe(1);
    expect(result.mappedProbabilityShape).toEqual([2, 2]);
    expect(result.mappedProbabilityMode).toBe("probabilities");
    expectCloseArray(result.mappedWeight, [
      fixture.base.indexSum,
      2 * fixture.base.indexSum,
    ]);
    expect(result.onnx).toEqual(["SoftmaxCrossEntropyLoss", "Identity"]);
    expect(result.rankOneOnnx).toEqual([
      "Unsqueeze",
      "Unsqueeze",
      "SoftmaxCrossEntropyLoss",
      "Squeeze",
      "Identity",
    ]);
    expect(result.probabilityOnnxError).toMatch(
      /^OnnxUnmappableOp: .*probability targets/u,
    );
    expect(result.smoothingOnnxError).toMatch(
      /^OnnxUnmappableOp: .*label_smoothing/u,
    );
    expect(result.mappedOnnxError).toMatch(/^OnnxUnmappableOp: .*vmap/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*CROSS_ENTROPY/u);
    expect(result.webgpuSupported).toBe(false);
    expect(result.support).toMatchObject({
      contractId: "browsergrad.jit.framework.functional.cross-entropy.v1",
      opcode: "CROSS_ENTROPY",
      semanticState: "typed",
      retiredOpaqueOperationId: "jit.custom.cross-entropy.v0",
    });
  });

  it("rejects malformed, mutated, cross-session, and over-budget requests", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_CROSS_ENTROPY_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import OP_BUFFER, OP_CROSS_ENTROPY, UOp
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

def tensor(values, dtype="float32", session=None, requires_grad=False):
    return bg.from_numpy(
        np.asarray(values, dtype=np.dtype(dtype)),
        session=session,
        requires_grad=requires_grad,
    )

source = tensor([[1, 2, 3], [2, 0, -1]])
indices = tensor([2, 0], "int64")
probabilities = tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
weight = tensor([1, 2, 4])
errors = {
    "input-type": error(lambda: F.cross_entropy([[1.0]], indices)),
    "target-type": error(lambda: F.cross_entropy(source, [2, 0])),
    "session": error(lambda: F.cross_entropy(
        source, tensor([2, 0], "int64", bg.new_session())
    )),
    "input-rank": error(lambda: F.cross_entropy(tensor(1.0), tensor(0, "int64"))),
    "class-empty": error(lambda: F.cross_entropy(
        bg.from_numpy(np.empty((2, 0), dtype=np.float32)),
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
    "weight-session": error(lambda: F.cross_entropy(
        source, indices, weight=tensor([1, 2, 4], session=bg.new_session())
    )),
    "reduction": error(lambda: F.cross_entropy(source, indices, reduction="median")),
    "reduction-type": error(lambda: F.cross_entropy(source, indices, reduction=object())),
    "ignore-type": error(lambda: F.cross_entropy(source, indices, ignore_index=True)),
    "ignore-range": error(lambda: F.cross_entropy(source, indices, ignore_index=1 << 70)),
    "smoothing-type": error(lambda: F.cross_entropy(source, indices, label_smoothing=True)),
    "smoothing-range": error(lambda: F.cross_entropy(source, indices, label_smoothing=1.1)),
    "size-average-type": error(lambda: F.cross_entropy(source, indices, size_average=1)),
    "reduce-type": error(lambda: F.cross_entropy(source, indices, reduce=1)),
    "target-range": error(lambda: F.cross_entropy(
        source, tensor([2, 3], "int64")
    ).numpy()),
}

class HostileArray(np.ndarray):
    pass
hostile = bg.from_numpy(
    np.asarray([2, 0], dtype=np.int64).view(HostileArray)
)
errors["array-subclass"] = error(lambda: F.cross_entropy(source, hostile).numpy())

valid = F.cross_entropy(source, indices, weight=weight, reduction="sum")
valid._uop.arg["target_mode"] = "probabilities"
dy = UOp(OP_BUFFER, (), (), "float32", arg="dy")
errors["mutated-cpu"] = error(valid.numpy)
errors["mutated-vjp"] = error(lambda: get_rule(OP_CROSS_ENTROPY)(
    valid._uop, valid._uop.inputs, dy
))
errors["mutated-vmap"] = error(lambda: get_vmap_rule(OP_CROSS_ENTROPY)(
    valid._uop, {id(node): node for node in valid._uop.inputs}, 2
))
errors["mutated-onnx"] = error(lambda: bg.onnx.export_inference(valid))
errors["mutated-plan"] = error(lambda: bg.gpu_plan_summary(valid))

def forged(input_shape, target_shape, dtype="float32"):
    logits = UOp(OP_BUFFER, (), input_shape, dtype, arg="input-" + str(input_shape))
    target = UOp(OP_BUFFER, (), target_shape, "int64", arg="target-" + str(target_shape))
    return UOp(
        OP_CROSS_ENTROPY,
        (logits, target),
        (),
        dtype,
        arg={
            "reduction": "sum",
            "batch_rank": 0,
            "ignore_index": -100,
            "has_weight": False,
            "label_smoothing": 0.0,
            "target_mode": "indices",
        },
    )

work = fixture["limits"]["workExtent"]
workspace = fixture["limits"]["workspaceExtent"]
errors["rank"] = error(lambda: forged((1,) * 33, (1,) * 32))
errors["work"] = error(lambda: forged((1, work), (1,)))
errors["workspace"] = error(lambda: forged((workspace, 1), (workspace,), "float64"))
errors["zero-hidden-work"] = error(lambda: forged((0, 1, work), (0, work)))
errors
`);

    const messages: Record<string, string> = {
      "input-type": "input must be",
      "target-type": "target must be",
      session: "same session",
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
      "weight-session": "same session",
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
    expect(errors["array-subclass"]).toMatch(
      /^RealizationError: .*exact ndarray/u,
    );
    for (const key of [
      "mutated-cpu",
      "mutated-vjp",
      "mutated-vmap",
      "mutated-onnx",
      "mutated-plan",
    ]) {
      expect(errors[key]).toMatch(
        /^(RealizationError|ShapeError): .*target_mode/u,
      );
    }
    expect(errors.rank).toMatch(/^ShapeError: .*rank/u);
    expect(errors.work).toMatch(/^ShapeError: .*work/u);
    expect(errors.workspace).toMatch(/^ShapeError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ShapeError: .*work/u);
  });
});
