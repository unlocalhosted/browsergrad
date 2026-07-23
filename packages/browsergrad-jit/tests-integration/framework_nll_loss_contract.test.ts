import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_NLL_LOSS_CONFORMANCE } from "../../../test-support/framework-nll-loss-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

function expectCloseArray(actual: unknown, expected: readonly number[], digits = 5): void {
  expect(Array.isArray(actual)).toBe(true);
  const values = actual as number[];
  expect(values).toHaveLength(expected.length);
  values.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, digits));
}

describe("Gate 6 typed torch.nn.functional.nll_loss contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("matches weighted class-index reductions, gradients, empties, dtypes, and aliases", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_NLL_LOSS_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
base = fixture["base"]

def tensor(shape, values, dtype="float32", requires_grad=False):
    return bg.from_numpy(
        np.asarray(values, dtype=np.dtype(dtype)).reshape(tuple(shape)),
        requires_grad=requires_grad,
    )

source = tensor(base["inputShape"], base["input"], requires_grad=True)
target = tensor(base["targetShape"], base["target"], "int64")
weight = tensor([3], base["weight"])
none = F.nll_loss(
    source,
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduction="none",
)
none_array = none.numpy()
upstream = tensor(base["targetShape"], base["upstream"])
closure = none._ctx.fn(
    upstream.numpy(),
    (source.numpy(), target.numpy(), weight.numpy()),
)
none.backward(upstream)

summed = F.nll_loss(
    tensor(base["inputShape"], base["input"]),
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduction="sum",
)
mean_source = tensor(base["inputShape"], base["input"], requires_grad=True)
mean = F.nll_loss(
    mean_source,
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
)
mean.backward()
unweighted = F.nll_loss(
    tensor(base["inputShape"], base["input"]),
    target,
    ignore_index=base["ignoreIndex"],
)

unbatched_fixture = fixture["unbatched"]
unbatched_source = tensor([3], unbatched_fixture["input"], requires_grad=True)
unbatched_target = tensor([], [unbatched_fixture["target"]], "int64")
unbatched = F.nll_loss(unbatched_source, unbatched_target)
unbatched.backward()

empty = fixture["empty"]
zero_batch_input = tensor(empty["zeroBatchInputShape"], [])
zero_batch_target = tensor(empty["zeroBatchTargetShape"], [], "int64")
zero_support_input = tensor(empty["zeroSupportInputShape"], [])
zero_support_target = tensor(empty["zeroSupportTargetShape"], [], "int64")
empty_results = {
    "zeroBatchNoneShape": list(F.nll_loss(
        zero_batch_input, zero_batch_target, reduction="none"
    ).shape),
    "zeroBatchSum": float(F.nll_loss(
        zero_batch_input, zero_batch_target, reduction="sum"
    ).item()),
    "zeroBatchMeanNan": bool(np.isnan(F.nll_loss(
        zero_batch_input, zero_batch_target, reduction="mean"
    ).item())),
    "zeroSupportMeanNan": bool(np.isnan(F.nll_loss(
        zero_support_input, zero_support_target, reduction="mean"
    ).item())),
}

dtype_results = []
for case in fixture["mixedDtypes"]:
    dtype = case["dtype"]
    dtype_source = tensor([2, 3], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], dtype, True)
    dtype_target = tensor([2], [0, 2], "int64")
    dtype_output = F.nll_loss(dtype_source, dtype_target, reduction="sum")
    dtype_output.backward()
    dtype_results.append({
        "output": dtype_output.dtype,
        "gradient": dtype_source.grad.dtype,
    })

legacy_sum = F.nll_loss(
    tensor(base["inputShape"], base["input"]),
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    size_average=False,
)
legacy_none = F.nll_loss(
    tensor(base["inputShape"], base["input"]),
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduce=False,
)
with bg.no_grad():
    detached = F.nll_loss(
        tensor([3], unbatched_fixture["input"], requires_grad=True),
        unbatched_target,
    )

bg.install_torch_alias()
import torch
module = torch.nn.NLLLoss(
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduction="sum",
)
module_value = module(
    tensor(base["inputShape"], base["input"]),
    target,
)
module_session = bg.new_session()
module_other_session_value = module(
    bg.from_numpy(
        np.asarray(base["input"], dtype=np.float32).reshape(base["inputShape"]),
        session=module_session,
    ),
    bg.from_numpy(
        np.asarray(base["target"], dtype=np.int64).reshape(base["targetShape"]),
        session=module_session,
    ),
)

{
    "op": none._uop.op,
    "arg": none._uop.arg,
    "none": none_array.reshape(-1).tolist(),
    "noneGradient": source.grad.numpy().reshape(-1).tolist(),
    "closure": closure[0].reshape(-1).tolist(),
    "nonInputClosureGradients": [value is None for value in closure[1:]],
    "sum": float(summed.item()),
    "mean": float(mean.item()),
    "meanGradient": mean_source.grad.numpy().reshape(-1).tolist(),
    "unweightedMean": float(unweighted.item()),
    "unbatched": float(unbatched.item()),
    "unbatchedGradient": unbatched_source.grad.numpy().tolist(),
    "empty": empty_results,
    "dtypes": dtype_results,
    "legacySum": float(legacy_sum.item()),
    "legacyNoneShape": list(legacy_none.shape),
    "ownsData": bool(none_array.flags["OWNDATA"]),
    "detached": detached.requires_grad,
    "module": float(module_value.item()),
    "moduleOtherSession": float(module_other_session_value.item()),
    "moduleState": {
        key: value.reshape(-1).tolist()
        for key, value in module.state_dict().items()
    },
}
`);

    const fixture = FRAMEWORK_NLL_LOSS_CONFORMANCE;
    expect(result.op).toBe("NLL_LOSS");
    expect(result.arg).toEqual({
      reduction: "none",
      batch_rank: 0,
      ignore_index: fixture.base.ignoreIndex,
      has_weight: true,
    });
    expectCloseArray(result.none, fixture.base.weightedNone);
    expectCloseArray(result.noneGradient, fixture.base.weightedNoneGradient);
    expectCloseArray(result.closure, fixture.base.weightedNoneGradient);
    expect(result.nonInputClosureGradients).toEqual([true, true]);
    expect(result.sum).toBeCloseTo(fixture.base.weightedSum, 5);
    expect(result.mean).toBeCloseTo(fixture.base.weightedMean, 5);
    expectCloseArray(result.meanGradient, fixture.base.weightedMeanGradient);
    expect(result.unweightedMean).toBeCloseTo(fixture.base.unweightedMean, 5);
    expect(result.unbatched).toBeCloseTo(fixture.unbatched.value, 5);
    expectCloseArray(result.unbatchedGradient, fixture.unbatched.gradient);
    expect(result.empty).toEqual({
      zeroBatchNoneShape: fixture.empty.zeroBatchTargetShape,
      zeroBatchSum: 0,
      zeroBatchMeanNan: true,
      zeroSupportMeanNan: true,
    });
    expect(result.dtypes).toEqual(fixture.mixedDtypes.map(({ dtype }) => ({
      output: dtype,
      gradient: dtype,
    })));
    expect(result.legacySum).toBeCloseTo(fixture.base.weightedSum, 5);
    expect(result.legacyNoneShape).toEqual(fixture.base.targetShape);
    expect(result.ownsData).toBe(true);
    expect(result.detached).toBe(false);
    expect(result.module).toBeCloseTo(fixture.base.weightedSum, 5);
    expect(result.moduleOtherSession).toBeCloseTo(fixture.base.weightedSum, 5);
    expect(result.moduleState).toEqual({ weight: fixture.base.weight });
  });

  it("supports symbolic VJP and vmap, exports exact ONNX NLL, and refuses device lowering", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_NLL_LOSS_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import OP_NLL_LOSS, OP_NLL_LOSS_VJP, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})
base = fixture["base"]
shape = tuple(base["inputShape"])
target_shape = tuple(base["targetShape"])
inputs = np.asarray(base["input"], dtype=np.float32).reshape(shape)
targets = np.asarray(base["target"], dtype=np.int64).reshape(target_shape)
weights = np.asarray(base["weight"], dtype=np.float32)

source = bg.from_numpy(inputs, requires_grad=True)
target = bg.from_numpy(targets)
weight = bg.from_numpy(weights)
loss = F.nll_loss(
    source, target, weight=weight, ignore_index=base["ignoreIndex"], reduction="mean"
)
gradient = bg.func.grad(lambda value: F.nll_loss(
    value,
    target,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduction="mean",
))(source)
symbolic = get_rule(OP_NLL_LOSS)(loss._uop, loss._uop.inputs, bg.ones()._uop)

batched_source = bg.from_numpy(np.stack([inputs, inputs - 0.2]))
batched_target = bg.from_numpy(np.stack([targets, targets]))
mapped_mean = bg.func.vmap(lambda value, labels: F.nll_loss(
    value,
    labels,
    weight=weight,
    ignore_index=base["ignoreIndex"],
    reduction="mean",
))(batched_source, batched_target)
batched_weight = bg.from_numpy(np.stack([weights, weights * 2]))
mapped_weight = bg.func.vmap(lambda value, labels, class_weight: F.nll_loss(
    value,
    labels,
    weight=class_weight,
    ignore_index=base["ignoreIndex"],
    reduction="sum",
))(batched_source, batched_target, batched_weight)
mapped_none = bg.func.vmap(lambda value, labels: F.nll_loss(
    value,
    labels,
    ignore_index=base["ignoreIndex"],
    reduction="none",
))(batched_source, batched_target)
nested_source = bg.from_numpy(np.stack([
    batched_source.numpy(),
    batched_source.numpy(),
]))
nested_target = bg.from_numpy(np.stack([
    batched_target.numpy(),
    batched_target.numpy(),
]))
nested_mean = bg.func.vmap(bg.func.vmap(
    lambda value, labels: F.nll_loss(
        value,
        labels,
        weight=weight,
        ignore_index=base["ignoreIndex"],
        reduction="mean",
    )
))(nested_source, nested_target)

model = bg.onnx.export_inference(
    F.nll_loss(source, target, weight=weight, ignore_index=base["ignoreIndex"], reduction="sum"),
    input_buffers=(source, target, weight),
)
unbatched = fixture["unbatched"]
rank_one_source = bg.from_numpy(np.asarray(unbatched["input"], dtype=np.float32))
scalar_target = bg.from_numpy(np.asarray(unbatched["target"], dtype=np.int64))
rank_one_model = bg.onnx.export_inference(
    F.nll_loss(rank_one_source, scalar_target, reduction="none"),
    input_buffers=(rank_one_source, scalar_target),
)

${ONNX_PROTOBUF_TEST_HELPERS}

def signed_i64(value):
    return value - (1 << 64) if value >= (1 << 63) else value

def parse_model(model):
    graph = next(
        payload for number, wire, payload in fields(model)
        if number == 7 and wire == 2
    )
    nodes = []
    for number, wire, payload in fields(graph):
        if number != 1 or wire != 2:
            continue
        node_fields = list(fields(payload))
        op_type = next(
            value.decode("utf-8") for field, kind, value in node_fields
            if field == 4 and kind == 2
        )
        attributes = {}
        for field, kind, attribute_payload in node_fields:
            if field != 5 or kind != 2:
                continue
            attribute_fields = list(fields(attribute_payload))
            name = next(
                value.decode("utf-8") for child, child_kind, value in attribute_fields
                if child == 1 and child_kind == 2
            )
            string_value = next(
                (value.decode("utf-8") for child, child_kind, value in attribute_fields
                 if child == 4 and child_kind == 2),
                None,
            )
            integer_value = next(
                (value for child, child_kind, value in attribute_fields
                 if child == 3 and child_kind == 0),
                None,
            )
            attributes[name] = (
                string_value
                if string_value is not None
                else signed_i64(integer_value)
            )
        nodes.append({"op": op_type, "attributes": attributes})
    return nodes

support = next(
    item for item in bg.framework_operation_support()["operations"]
    if item["opcode"] == OP_NLL_LOSS
)

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "gradient": gradient.numpy().reshape(-1).tolist(),
    "symbolicOps": [node.op for node in toposort(symbolic[0])],
    "symbolicNondifferentiable": [value is None for value in symbolic[1:]],
    "mappedMean": mapped_mean.numpy().tolist(),
    "mappedMeanBatchRank": mapped_mean._uop.arg["batch_rank"],
    "mappedWeight": mapped_weight.numpy().tolist(),
    "mappedNoneShape": list(mapped_none.shape),
    "nestedMean": nested_mean.numpy().reshape(-1).tolist(),
    "nestedBatchRank": nested_mean._uop.arg["batch_rank"],
    "onnx": parse_model(model),
    "rankOneOnnx": parse_model(rank_one_model),
    "vmapOnnxError": error(lambda: bg.onnx.export_inference(mapped_mean)),
    "planError": error(lambda: bg.gpu_plan_summary(loss)),
    "webgpuSupported": OP_NLL_LOSS in supported_opcodes() or OP_NLL_LOSS_VJP in supported_opcodes(),
    "support": support,
}
`);

    const fixture = FRAMEWORK_NLL_LOSS_CONFORMANCE;
    expectCloseArray(result.gradient, fixture.base.weightedMeanGradient);
    expect(result.symbolicOps).toContain("NLL_LOSS_VJP");
    expect(result.symbolicOps).not.toContain("CUSTOM");
    expect(result.symbolicNondifferentiable).toEqual([true, true]);
    expectCloseArray(result.mappedMean, [
      fixture.base.weightedMean,
      fixture.base.weightedMean + 0.2,
    ]);
    expect(result.mappedMeanBatchRank).toBe(1);
    expectCloseArray(result.mappedWeight, [
      fixture.base.weightedSum,
      2 * (fixture.base.weightedSum + 0.8),
    ]);
    expect(result.mappedNoneShape).toEqual([2, ...fixture.base.targetShape]);
    expectCloseArray(result.nestedMean, [
      fixture.base.weightedMean,
      fixture.base.weightedMean + 0.2,
      fixture.base.weightedMean,
      fixture.base.weightedMean + 0.2,
    ]);
    expect(result.nestedBatchRank).toBe(2);
    expect(result.onnx).toEqual([
      {
        op: "NegativeLogLikelihoodLoss",
        attributes: { reduction: "sum", ignore_index: fixture.base.ignoreIndex },
      },
      { op: "Identity", attributes: {} },
    ]);
    expect((result.rankOneOnnx as { op: string }[]).map(({ op }) => op)).toEqual([
      "Unsqueeze",
      "Unsqueeze",
      "NegativeLogLikelihoodLoss",
      "Squeeze",
      "Identity",
    ]);
    expect(result.vmapOnnxError).toMatch(/^OnnxUnmappableOp: .*vmap/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*NLL_LOSS/u);
    expect(result.webgpuSupported).toBe(false);
    expect(result.support).toMatchObject({
      contractId: "browsergrad.jit.framework.functional.nll-loss.v1",
      opcode: "NLL_LOSS",
      semanticState: "typed",
      retiredOpaqueOperationId: "jit.custom.nll-loss.v0",
    });
  });

  it("rejects malformed, mutated, out-of-range, cross-session, and over-budget requests", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_NLL_LOSS_CONFORMANCE);
    const errors = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import browsergrad_jit.nn.functional as F
import json
import numpy as np
from browsergrad_jit._ir import OP_BUFFER, OP_NLL_LOSS, UOp
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})

def error(fn):
    try:
        fn()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

def tensor(values, dtype="float32", session=None):
    return bg.from_numpy(np.asarray(values, dtype=np.dtype(dtype)), session=session)

source = tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]])
target = tensor([0, 2], "int64")
weight = tensor([0.5, 2.0, 1.5])
other_target = tensor([0, 2], "int64", bg.new_session())
other_weight = tensor([0.5, 2.0, 1.5], session=bg.new_session())
errors = {
    "input-type": error(lambda: F.nll_loss([[-0.1]], target)),
    "target-type": error(lambda: F.nll_loss(source, [0, 2])),
    "target-shape": error(lambda: F.nll_loss(source, tensor([[0], [2]], "int64"))),
    "session": error(lambda: F.nll_loss(source, other_target)),
    "input-rank": error(lambda: F.nll_loss(tensor(1.0), tensor(0, "int64"))),
    "reduction": error(lambda: F.nll_loss(source, target, reduction="median")),
    "reduction-type": error(lambda: F.nll_loss(source, target, reduction=object())),
    "input-dtype": error(lambda: F.nll_loss(tensor([[1, 2], [3, 4]], "int32"), target)),
    "target-dtype": error(lambda: F.nll_loss(source, tensor([0, 1], "int32"))),
    "weight-type": error(lambda: F.nll_loss(source, target, weight=[1, 2, 3])),
    "weight-shape": error(lambda: F.nll_loss(source, target, weight=tensor([1, 2]))),
    "weight-dtype": error(lambda: F.nll_loss(source, target, weight=tensor([1, 2, 3], "float64"))),
    "weight-grad": error(lambda: F.nll_loss(
        source, target, weight=bg.from_numpy(np.ones(3, dtype=np.float32), requires_grad=True)
    )),
    "weight-session": error(lambda: F.nll_loss(source, target, weight=other_weight)),
    "ignore-type": error(lambda: F.nll_loss(source, target, ignore_index=True)),
    "ignore-range": error(lambda: F.nll_loss(source, target, ignore_index=1 << 70)),
    "size-average-type": error(lambda: F.nll_loss(source, target, size_average=1)),
    "reduce-type": error(lambda: F.nll_loss(source, target, reduce=1)),
    "target-range": error(lambda: F.nll_loss(source, tensor([0, 3], "int64")).numpy()),
}

class HostileArray(np.ndarray):
    pass
hostile_target = bg.from_numpy(np.asarray([0, 2], dtype=np.int64).view(HostileArray))
errors["array-subclass"] = error(lambda: F.nll_loss(source, hostile_target).numpy())

valid = F.nll_loss(source, target, weight=weight, reduction="sum")
valid._uop.arg["has_weight"] = 1
dy = UOp(OP_BUFFER, (), (), "float32", arg="dy")
errors["mutated-cpu"] = error(valid.numpy)
errors["mutated-vjp"] = error(lambda: get_rule(OP_NLL_LOSS)(
    valid._uop, valid._uop.inputs, dy
))
errors["mutated-vmap"] = error(lambda: get_vmap_rule(OP_NLL_LOSS)(
    valid._uop, {id(node): node for node in valid._uop.inputs}, 2
))
errors["mutated-onnx"] = error(lambda: bg.onnx.export_inference(valid))
errors["mutated-plan"] = error(lambda: bg.gpu_plan_summary(valid))

def forged(input_shape, target_shape, dtype="float32"):
    left = UOp(OP_BUFFER, (), input_shape, dtype, arg="left-" + str(input_shape))
    labels = UOp(OP_BUFFER, (), target_shape, "int64", arg="target-" + str(target_shape))
    return UOp(
        OP_NLL_LOSS,
        (left, labels),
        (),
        dtype,
        arg={
            "reduction": "sum",
            "batch_rank": 0,
            "ignore_index": -100,
            "has_weight": False,
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

    for (const invalid of FRAMEWORK_NLL_LOSS_CONFORMANCE.invalid) {
      expect(errors[invalid.id]).toContain(invalid.message);
    }
    expect(errors["array-subclass"]).toMatch(/^RealizationError: .*exact ndarray/u);
    for (const key of ["mutated-cpu", "mutated-vjp", "mutated-vmap", "mutated-onnx", "mutated-plan"] as const) {
      expect(errors[key]).toMatch(/^(RealizationError|ShapeError): .*has_weight/u);
    }
    expect(errors.rank).toMatch(/^ShapeError: .*rank/u);
    expect(errors.work).toMatch(/^ShapeError: .*work/u);
    expect(errors.workspace).toMatch(/^ShapeError: .*workspace/u);
    expect(errors["zero-hidden-work"]).toMatch(/^ShapeError: .*work/u);
  });
});
