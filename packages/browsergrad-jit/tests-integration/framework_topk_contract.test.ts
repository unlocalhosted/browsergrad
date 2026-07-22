import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_TOPK_CONFORMANCE } from "../../../test-support/framework-topk-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.topk contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("shares selection, dtype, ownership, gradient, resource, and refusal semantics", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_TOPK_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source_array = np.asarray(valid["inputValues"], dtype=np.float32)
source = bg.from_numpy(source_array, requires_grad=True)
largest_values, largest_indices = bg.topk(
    source,
    np.int32(valid["k"]),
    dim=None,
    largest=True,
    sorted=True,
)
smallest_values, smallest_indices = source.topk(
    2,
    dim=1,
    largest=False,
    sorted=True,
)
largest_snapshot = largest_values.numpy()
largest_index_snapshot = largest_indices.numpy()
largest_values.backward(bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32)))

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed_array = np.asarray(case["input"], dtype=np.dtype(case["dtype"]))
    typed = bg.from_numpy(typed_array)
    values, indices = bg.topk(typed, len(case["input"]), sorted=True)
    values_array = values.numpy()
    indices_array = indices.numpy()
    dtype_cases.append({
        "valueDtype": values.dtype,
        "indexDtype": indices.dtype,
        "values": values_array.tolist(),
        "gatherMatches": bool(np.array_equal(
            values_array,
            np.take_along_axis(typed_array, indices_array, axis=0),
        )),
        "indicesUnique": len(set(indices_array.tolist())) == len(case["input"]),
    })

empty_values, empty_indices = bg.topk(
    bg.from_numpy(np.empty(tuple(fixture["empty"]["inputShape"]), dtype=np.float32)),
    fixture["empty"]["k"],
)
zero_values, zero_indices = bg.topk(
    bg.from_numpy(np.ones(tuple(fixture["zeroK"]["inputShape"]), dtype=np.float32)),
    0,
)

unsorted_spec = fixture["unsorted"]
unsorted_source = bg.from_numpy(np.asarray(unsorted_spec["input"], dtype=np.float32))
unsorted_values, unsorted_indices = bg.topk(
    unsorted_source,
    unsorted_spec["k"],
    largest=False,
    sorted=False,
)

tie_source = bg.from_numpy(np.asarray([5, 1, 5, 5, 2], dtype=np.float32))
tie_values_a, tie_indices_a = bg.topk(tie_source, 2)
tie_values_b, tie_indices_b = bg.topk(tie_source, 2)
tie_a = tie_indices_a.numpy()
tie_b = tie_indices_b.numpy()

uint_values, uint_indices = bg.topk(
    bg.from_numpy(np.asarray([0, 255, 1], dtype=np.uint8)),
    2,
)
signed_min_values, signed_min_indices = bg.topk(
    bg.from_numpy(np.asarray([np.iinfo(np.int64).min, 1, 0], dtype=np.int64)),
    2,
)
nan_source = bg.from_numpy(np.asarray([np.nan, 1.0, np.nan, 2.0], dtype=np.float32))
nan_largest_values, _ = bg.topk(nan_source, 3)
nan_smallest_values, _ = bg.topk(nan_source, 2, largest=False)

class HostileIndex:
    calls = 0
    def __index__(self):
        HostileIndex.calls += 1
        return 1

class HostileBool:
    calls = 0
    def __bool__(self):
        HostileBool.calls += 1
        return True

errors = {}
attempts = {
    "non-tensor": lambda: bg.topk([[1.0]], 1),
    "bool-k": lambda: bg.topk(source, True),
    "float-k": lambda: bg.topk(source, 1.0),
    "negative-k": lambda: bg.topk(source, -1),
    "oversized-k": lambda: bg.topk(source, 7),
    "bool-dim": lambda: bg.topk(source, 1, dim=True),
    "float-dim": lambda: bg.topk(source, 1, dim=1.0),
    "out-of-range-dim": lambda: bg.topk(source, 1, dim=2),
    "largest-type": lambda: bg.topk(source, 1, largest=1),
    "sorted-type": lambda: bg.topk(source, 1, sorted=1),
    "out-mutation": lambda: bg.topk(source, 1, out=(source, source)),
    "unsupported-dtype": lambda: bg.topk(bg.from_numpy(np.ones((1,), dtype=np.uint16)), 1),
    "scalar-input": lambda: bg.topk(bg.tensor(1.0), 1),
    "hostile-k": lambda: bg.topk(source, HostileIndex()),
    "hostile-dim": lambda: bg.topk(source, 1, dim=HostileIndex()),
    "hostile-largest": lambda: bg.topk(source, 1, largest=HostileBool()),
    "hostile-sorted": lambda: bg.topk(source, 1, sorted=HostileBool()),
    "zero-size-oversized-axis": lambda: bg.topk(
        bg.from_numpy(np.empty((0, (1 << 20) + 1), dtype=np.float32)), 0, dim=1
    ),
    "zero-size-oversized-extent": lambda: bg.topk(
        bg.from_numpy(np.empty((0, (1 << 28) + 1), dtype=np.float32)), 0, dim=0
    ),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

values_own = bool(largest_snapshot.flags["OWNDATA"])
indices_own = bool(largest_index_snapshot.flags["OWNDATA"])
largest_snapshot[0, 0] = 999
largest_index_snapshot[0, 0] = 999
rerun_values, rerun_indices = bg.topk(source, valid["k"])

large = bg.from_numpy(np.arange(4 * 50257, dtype=np.float32).reshape(4, 50257))
large_values, large_indices = bg.topk(large, 10)

{
    "valueOp": largest_values._uop.op,
    "indexOp": largest_indices._uop.op,
    "valueArg": dict(largest_values._uop.arg),
    "indexArg": dict(largest_indices._uop.arg),
    "largestValues": rerun_values.numpy().tolist(),
    "largestIndices": rerun_indices.numpy().tolist(),
    "smallestValues": smallest_values.numpy().tolist(),
    "smallestIndices": smallest_indices.numpy().tolist(),
    "gradient": source.grad.numpy().tolist(),
    "indicesRequireGrad": largest_indices.requires_grad,
    "dtypeCases": dtype_cases,
    "empty": [list(empty_values.shape), list(empty_indices.shape)],
    "zeroK": [list(zero_values.shape), list(zero_indices.shape)],
    "unsorted": {
        "values": sorted(unsorted_values.numpy().tolist()),
        "indices": sorted(unsorted_indices.numpy().tolist()),
        "arg": dict(unsorted_values._uop.arg),
    },
    "ties": {
        "values": tie_values_a.numpy().tolist(),
        "indicesValid": bool(all(int(index) in (0, 2, 3) for index in tie_a)),
        "indicesUnique": len(set(tie_a.tolist())) == 2,
        "deterministic": bool(np.array_equal(tie_a, tie_b)),
    },
    "edgeOrdering": {
        "uintValues": uint_values.numpy().tolist(),
        "uintIndices": uint_indices.numpy().tolist(),
        "signedMinValuesCorrect": bool(np.array_equal(
            signed_min_values.numpy(), np.asarray([1, 0], dtype=np.int64)
        )),
        "signedMinIndices": signed_min_indices.numpy().tolist(),
        "nanLargest": [bool(np.isnan(value)) for value in nan_largest_values.numpy()],
        "nanSmallest": nan_smallest_values.numpy().tolist(),
    },
    "largeShapes": [list(large_values.shape), list(large_indices.shape)],
    "ownsData": [values_own, indices_own],
    "errors": errors,
    "hostileCalls": {"index": HostileIndex.calls, "bool": HostileBool.calls},
}
`);

    expect(result).toMatchObject({
      valueOp: "TOPK_VALUES",
      indexOp: "TOPK_INDICES",
      valueArg: { axis: 1, k: 3, largest: true, sorted: true },
      indexArg: { axis: 1, k: 3, largest: true, sorted: true },
      largestValues: FRAMEWORK_TOPK_CONFORMANCE.valid.largestValues,
      largestIndices: FRAMEWORK_TOPK_CONFORMANCE.valid.largestIndices,
      smallestValues: FRAMEWORK_TOPK_CONFORMANCE.valid.smallestValues,
      smallestIndices: FRAMEWORK_TOPK_CONFORMANCE.valid.smallestIndices,
      gradient: FRAMEWORK_TOPK_CONFORMANCE.valid.expectedGradient,
      indicesRequireGrad: false,
      empty: [
        FRAMEWORK_TOPK_CONFORMANCE.empty.expectedShape,
        FRAMEWORK_TOPK_CONFORMANCE.empty.expectedShape,
      ],
      zeroK: [
        FRAMEWORK_TOPK_CONFORMANCE.zeroK.expectedShape,
        FRAMEWORK_TOPK_CONFORMANCE.zeroK.expectedShape,
      ],
      unsorted: {
        values: FRAMEWORK_TOPK_CONFORMANCE.unsorted.expectedValueSet,
        indices: FRAMEWORK_TOPK_CONFORMANCE.unsorted.expectedIndexSet,
        arg: { axis: 0, k: 3, largest: false, sorted: false },
      },
      ties: {
        values: [5, 5],
        indicesValid: true,
        indicesUnique: true,
        deterministic: true,
      },
      edgeOrdering: {
        uintValues: [255, 1],
        uintIndices: [1, 2],
        signedMinValuesCorrect: true,
        signedMinIndices: [1, 2],
        nanLargest: [true, true, false],
        nanSmallest: [1, 2],
      },
      largeShapes: [[4, 10], [4, 10]],
      ownsData: [true, true],
      hostileCalls: { index: 0, bool: 0 },
    });
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_TOPK_CONFORMANCE.dtypeCases.map((testCase) => ({
        valueDtype: testCase.dtype,
        indexDtype: "int64",
        values: testCase.values,
        gatherMatches: true,
        indicesUnique: true,
      })),
    );
    const errors = result.errors as Record<string, string>;
    for (const invalid of FRAMEWORK_TOPK_CONFORMANCE.invalid) {
      expect(errors[invalid.id]).not.toBe("no_error");
    }
  });

  it("provides selection VJP, batch-safe vmap, exact ONNX, and device refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_TOPK_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import json
import numpy as np
from browsergrad_jit._ir import OP_TOPK_INDICES, OP_TOPK_VALUES, OP_SCATTER_ADD, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32))
cotangent = bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32))
gradient = bg.func.grad(
    lambda value: (bg.topk(value, valid["k"])[0] * cotangent).sum()
)(source)

vmap_spec = fixture["vmap"]
mapped_source = bg.from_numpy(np.asarray(vmap_spec["inputValues"], dtype=np.float32))
mapped_values = bg.func.vmap(lambda row: bg.topk(row, vmap_spec["k"])[0])(mapped_source)
mapped_indices = bg.func.vmap(lambda row: bg.topk(row, vmap_spec["k"])[1])(mapped_source)

${ONNX_PROTOBUF_TEST_HELPERS}

def parse_topk(model):
    graph = next(payload for number, wire, payload in fields(model) if number == 7 and wire == 2)
    op_types = []
    attrs = {}
    topk_output_count = None
    k = None
    for number, wire, payload in fields(graph):
        if number == 1 and wire == 2:
            node_fields = list(fields(payload))
            op_type = next(
                item.decode("utf-8") for field, kind, item in node_fields
                if field == 4 and kind == 2
            )
            op_types.append(op_type)
            if op_type == "TopK":
                topk_output_count = sum(1 for field, kind, _ in node_fields if field == 2 and kind == 2)
                for field, kind, item in node_fields:
                    if field != 5 or kind != 2:
                        continue
                    attr_fields = list(fields(item))
                    name = next(
                        value.decode("utf-8") for attr_field, attr_kind, value in attr_fields
                        if attr_field == 1 and attr_kind == 2
                    )
                    integer = next(
                        value for attr_field, attr_kind, value in attr_fields
                        if attr_field == 3 and attr_kind == 0
                    )
                    attrs[name] = integer
        elif number == 5 and wire == 2:
            tensor_fields = list(fields(payload))
            name = next(
                item.decode("utf-8") for field, kind, item in tensor_fields
                if field == 8 and kind == 2
            )
            if name.startswith("const_topk_k_"):
                raw = next(item for field, kind, item in tensor_fields if field == 9 and kind == 2)
                k = int.from_bytes(raw, "little", signed=True)
    return {
        "opTypes": op_types,
        "k": k,
        "axis": attrs.get("axis"),
        "largest": attrs.get("largest"),
        "sorted": attrs.get("sorted"),
        "topkOutputCount": topk_output_count,
    }

values, indices = bg.topk(source, valid["k"])
values_onnx = parse_topk(bg.onnx.export_inference(values, input_buffers=(source,)))
indices_onnx = parse_topk(bg.onnx.export_inference(indices, input_buffers=(source,)))
unsorted_indices = bg.topk(source, 2, largest=False, sorted=False)[1]
unsorted_onnx = parse_topk(bg.onnx.export_inference(unsorted_indices, input_buffers=(source,)))
bool_source = bg.from_numpy(np.asarray([True, False], dtype=np.bool_))
zero_k = bg.topk(source, 0)[0]

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "registered": [get_rule(OP_TOPK_VALUES) is not None, get_rule(OP_TOPK_INDICES) is not None],
    "gradient": gradient.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mappedValues": mapped_values.numpy().tolist(),
    "mappedIndices": mapped_indices.numpy().tolist(),
    "mappedArgs": [dict(mapped_values._uop.arg), dict(mapped_indices._uop.arg)],
    "valuesOnnx": values_onnx,
    "indicesOnnx": indices_onnx,
    "unsortedOnnx": unsorted_onnx,
    "boolOnnxError": error(lambda: bg.onnx.export_inference(bg.topk(bool_source, 1)[0], input_buffers=(bool_source,))),
    "zeroKOnnxError": error(lambda: bg.onnx.export_inference(zero_k, input_buffers=(source,))),
    "planError": error(lambda: bg.gpu_plan_summary(values)),
    "webgpuSupported": [OP_TOPK_VALUES in supported_opcodes(), OP_TOPK_INDICES in supported_opcodes()],
}
`);

    expect(result).toMatchObject({
      registered: [true, true],
      gradient: FRAMEWORK_TOPK_CONFORMANCE.valid.expectedGradient,
      mappedValues: FRAMEWORK_TOPK_CONFORMANCE.vmap.expectedValues,
      mappedIndices: FRAMEWORK_TOPK_CONFORMANCE.vmap.expectedIndices,
      mappedArgs: [
        { axis: 1, k: 2, largest: true, sorted: true },
        { axis: 1, k: 2, largest: true, sorted: true },
      ],
      valuesOnnx: {
        opTypes: FRAMEWORK_TOPK_CONFORMANCE.onnx.valuesOpTypes,
        k: FRAMEWORK_TOPK_CONFORMANCE.onnx.k,
        axis: FRAMEWORK_TOPK_CONFORMANCE.onnx.axis,
        largest: FRAMEWORK_TOPK_CONFORMANCE.onnx.largest,
        sorted: FRAMEWORK_TOPK_CONFORMANCE.onnx.sorted,
        topkOutputCount: FRAMEWORK_TOPK_CONFORMANCE.onnx.topkOutputCount,
      },
      indicesOnnx: {
        opTypes: FRAMEWORK_TOPK_CONFORMANCE.onnx.indicesOpTypes,
        k: FRAMEWORK_TOPK_CONFORMANCE.onnx.k,
        axis: FRAMEWORK_TOPK_CONFORMANCE.onnx.axis,
        largest: FRAMEWORK_TOPK_CONFORMANCE.onnx.largest,
        sorted: FRAMEWORK_TOPK_CONFORMANCE.onnx.sorted,
        topkOutputCount: FRAMEWORK_TOPK_CONFORMANCE.onnx.topkOutputCount,
      },
      unsortedOnnx: { k: 2, axis: 1, largest: 0, sorted: 0 },
      webgpuSupported: [false, false],
    });
    expect(result.gradientOps).toContain("SCATTER_ADD");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.boolOnnxError).toMatch(/^OnnxUnmappableOp:/u);
    expect(result.zeroKOnnxError).toMatch(/^OnnxUnmappableOp:/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported:/u);
  });

  it("rejects malformed pairs, mutated arguments, and oversized workspace at every boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_TOPK_INDICES, OP_TOPK_VALUES
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

source = bg.from_numpy(np.asarray([[3.0, 1.0, 2.0]], dtype=np.float32))
other = bg.from_numpy(np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32))
values, indices = bg.topk(source, 2)
dy = bg.ones(1, 2)._uop
wrong_source_indices = UOp(OP_TOPK_INDICES, (other._uop,), (1, 2), "int64", arg={
    "axis": 1, "k": 2, "largest": True, "sorted": True,
})
mutated_values, _ = bg.topk(source, 2)
mutated_values._uop.arg["k"] = 1
large = UOp(OP_BUFFER, (), (64, 1 << 19), "float32", arg="topk-workspace")
oversized_axis = UOp(OP_BUFFER, (), (0, (1 << 20) + 1), "float32", arg="topk-axis")

{
    "missingIndexInput": error(lambda: bg.Tensor(UOp(OP_TOPK_VALUES, (source._uop,), (1, 2), "float32", arg={
        "axis": 1, "k": 2, "largest": True, "sorted": True,
    }))),
    "wrongIndexSource": error(lambda: bg.Tensor(UOp(OP_TOPK_VALUES, (source._uop, wrong_source_indices), (1, 2), "float32", arg={
        "axis": 1, "k": 2, "largest": True, "sorted": True,
    }))),
    "wrongIndexDtype": error(lambda: bg.Tensor(UOp(OP_TOPK_INDICES, (source._uop,), (1, 2), "int32", arg={
        "axis": 1, "k": 2, "largest": True, "sorted": True,
    }))),
    "wrongOutputShape": error(lambda: bg.Tensor(UOp(OP_TOPK_INDICES, (source._uop,), (1, 3), "int64", arg={
        "axis": 1, "k": 2, "largest": True, "sorted": True,
    }))),
    "workspaceBound": error(lambda: UOp(OP_TOPK_INDICES, (large,), (64, 1), "int64", arg={
        "axis": 1, "k": 1, "largest": True, "sorted": True,
    })),
    "axisBound": error(lambda: UOp(OP_TOPK_INDICES, (oversized_axis,), (0, 1), "int64", arg={
        "axis": 1, "k": 1, "largest": True, "sorted": True,
    })),
    "openArgRealize": error(lambda: mutated_values.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_TOPK_VALUES)(mutated_values._uop, mutated_values._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_TOPK_VALUES)(
        mutated_values._uop,
        {id(mutated_values._uop.inputs[0]): mutated_values._uop.inputs[0], id(mutated_values._uop.inputs[1]): mutated_values._uop.inputs[1]},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated_values, input_buffers=(source,))),
    "openArgPlan": error(lambda: bg.gpu_plan_summary(mutated_values)),
}
`);

    expect(result.missingIndexInput).toMatch(/^ShapeError: .*source and TOPK_INDICES/u);
    expect(result.wrongIndexSource).toMatch(/^ShapeError: .*exact same source/u);
    expect(result.wrongIndexDtype).toMatch(/^ShapeError: .*int64/u);
    expect(result.wrongOutputShape).toMatch(/^ShapeError: .*output shape/u);
    expect(result.workspaceBound).toMatch(/^ShapeError: .*workspace/u);
    expect(result.axisBound).toMatch(/^ShapeError: .*selection ceiling/u);
    expect(result.openArgRealize).toMatch(/^RealizationError:/u);
    expect(result.openArgVjp).toMatch(/^ShapeError:/u);
    expect(result.openArgVmap).toMatch(/^ShapeError:/u);
    expect(result.openArgOnnx).toMatch(/^ShapeError:/u);
    expect(result.openArgPlan).toMatch(/^ShapeError:/u);
  });
});
