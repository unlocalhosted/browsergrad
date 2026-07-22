import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_SORT_CONFORMANCE } from "../../../test-support/framework-sort-conformance";
import { ONNX_PROTOBUF_TEST_HELPERS } from "../../../test-support/onnx-protobuf-test-helpers";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("Gate 6 typed torch.sort contract", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getJitTarget());
  });

  it("shares ordering, tie, dtype, ownership, gradient, and refusal semantics", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SORT_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32), requires_grad=True)
ascending_values, ascending_indices = bg.sort(source, dim=np.int32(valid["dim"]), stable=True)
descending_values, descending_indices = source.sort(dim=1, descending=True, stable=True)
ascending_snapshot = ascending_values.numpy()
ascending_index_snapshot = ascending_indices.numpy()
ascending_values.backward(bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32)))

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed = bg.from_numpy(np.asarray(case["input"], dtype=np.dtype(case["dtype"])))
    values, indices = bg.sort(typed, stable=True)
    dtype_cases.append({
        "valueDtype": values.dtype,
        "indexDtype": indices.dtype,
        "values": values.numpy().tolist(),
        "indices": indices.numpy().tolist(),
    })

scalar_values, scalar_indices = bg.sort(
    bg.from_numpy(np.asarray(fixture["scalar"]["value"], dtype=np.float32)),
    dim=0,
    stable=True,
)
empty_values, empty_indices = bg.sort(
    bg.from_numpy(np.empty(tuple(fixture["empty"]["inputShape"]), dtype=np.float32)),
    stable=True,
)
uint_values, uint_indices = bg.sort(
    bg.from_numpy(np.asarray([0, 255, 1, 255], dtype=np.uint8)),
    descending=True,
    stable=True,
)
nan_source = bg.from_numpy(np.asarray([np.nan, 1.0, np.nan], dtype=np.float32))
_, nan_ascending_indices = bg.sort(nan_source, stable=True)
_, nan_descending_indices = bg.sort(nan_source, descending=True, stable=True)
signed_min_source = bg.from_numpy(
    np.asarray([np.iinfo(np.int64).min, 1, 0], dtype=np.int64)
)
signed_min_values, signed_min_indices = bg.sort(
    signed_min_source,
    descending=True,
    stable=True,
)

class HostileDim:
    calls = 0
    def __index__(self):
        HostileDim.calls += 1
        return 0

class HostileBool:
    calls = 0
    def __bool__(self):
        HostileBool.calls += 1
        return False

errors = {}
attempts = {
    "non-tensor": lambda: bg.sort([[1.0]]),
    "bool-dim": lambda: bg.sort(source, dim=True),
    "float-dim": lambda: bg.sort(source, dim=1.0),
    "out-of-range-dim": lambda: bg.sort(source, dim=2),
    "descending-type": lambda: bg.sort(source, descending=1),
    "stable-type": lambda: bg.sort(source, stable=1),
    "out-mutation": lambda: bg.sort(source, out=(source, source)),
    "unsupported-dtype": lambda: bg.sort(bg.from_numpy(np.ones((1,), dtype=np.uint16))),
    "hostile-dim": lambda: bg.sort(source, dim=HostileDim()),
    "hostile-descending": lambda: bg.sort(source, descending=HostileBool()),
    "hostile-stable": lambda: bg.sort(source, stable=HostileBool()),
    "zero-size-oversized-axis": lambda: bg.sort(
        bg.from_numpy(np.empty((0, (1 << 28) + 1), dtype=np.float32)), dim=0
    ),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

values_own = bool(ascending_snapshot.flags["OWNDATA"])
indices_own = bool(ascending_index_snapshot.flags["OWNDATA"])
ascending_snapshot[0, 0] = 999
ascending_index_snapshot[0, 0] = 999
rerun_values, rerun_indices = bg.sort(source, stable=True)

{
    "valueOp": ascending_values._uop.op,
    "indexOp": ascending_indices._uop.op,
    "valueArg": dict(ascending_values._uop.arg),
    "indexArg": dict(ascending_indices._uop.arg),
    "ascendingValues": rerun_values.numpy().tolist(),
    "ascendingIndices": rerun_indices.numpy().tolist(),
    "descendingValues": descending_values.numpy().tolist(),
    "descendingIndices": descending_indices.numpy().tolist(),
    "gradient": source.grad.numpy().tolist(),
    "indicesRequireGrad": ascending_indices.requires_grad,
    "dtypeCases": dtype_cases,
    "scalar": {"value": scalar_values.item(), "index": scalar_indices.item()},
    "empty": {"valueShape": list(empty_values.shape), "indexShape": list(empty_indices.shape)},
    "edgeOrdering": {
        "uintValues": uint_values.numpy().tolist(),
        "uintIndices": uint_indices.numpy().tolist(),
        "nanAscendingIndices": nan_ascending_indices.numpy().tolist(),
        "nanDescendingIndices": nan_descending_indices.numpy().tolist(),
        "signedMinValuesCorrect": bool(np.array_equal(
            signed_min_values.numpy(),
            np.asarray([1, 0, np.iinfo(np.int64).min], dtype=np.int64),
        )),
        "signedMinIndices": signed_min_indices.numpy().tolist(),
    },
    "ownsData": [values_own, indices_own],
    "errors": errors,
    "hostileCalls": {"dim": HostileDim.calls, "bool": HostileBool.calls},
}
`);

    expect(result).toMatchObject({
      valueOp: "SORT_VALUES",
      indexOp: "SORT_INDICES",
      valueArg: { axis: 1, descending: false, stable: true },
      indexArg: { axis: 1, descending: false, stable: true },
      ascendingValues: FRAMEWORK_SORT_CONFORMANCE.valid.ascendingValues,
      ascendingIndices: FRAMEWORK_SORT_CONFORMANCE.valid.ascendingIndices,
      descendingValues: FRAMEWORK_SORT_CONFORMANCE.valid.descendingValues,
      descendingIndices: FRAMEWORK_SORT_CONFORMANCE.valid.descendingIndices,
      gradient: FRAMEWORK_SORT_CONFORMANCE.valid.expectedGradient,
      indicesRequireGrad: false,
      scalar: {
        value: FRAMEWORK_SORT_CONFORMANCE.scalar.value,
        index: FRAMEWORK_SORT_CONFORMANCE.scalar.expectedIndex,
      },
      empty: {
        valueShape: FRAMEWORK_SORT_CONFORMANCE.empty.expectedShape,
        indexShape: FRAMEWORK_SORT_CONFORMANCE.empty.expectedShape,
      },
      edgeOrdering: {
        uintValues: [255, 255, 1, 0],
        uintIndices: [1, 3, 2, 0],
        nanAscendingIndices: [1, 0, 2],
        nanDescendingIndices: [0, 2, 1],
        signedMinValuesCorrect: true,
        signedMinIndices: [1, 2, 0],
      },
      ownsData: [true, true],
      hostileCalls: { dim: 0, bool: 0 },
    });
    expect(result.dtypeCases).toEqual(
      FRAMEWORK_SORT_CONFORMANCE.dtypeCases.map((testCase) => ({
        valueDtype: testCase.dtype,
        indexDtype: "int64",
        values: testCase.values,
        indices: testCase.indices,
      })),
    );
    const errors = result.errors as Record<string, string>;
    for (const invalid of FRAMEWORK_SORT_CONFORMANCE.invalid) {
      expect(errors[invalid.id]).not.toBe("no_error");
    }
  });

  it("provides permutation VJP, batch-safe vmap, exact ONNX, and device refusal", async () => {
    const target = await getJitTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_SORT_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_jit as bg
import json
import numpy as np
from browsergrad_jit._ir import OP_SORT_INDICES, OP_SORT_VALUES, OP_SCATTER_ADD, toposort
from browsergrad_jit._realize_webgpu import supported_opcodes
from browsergrad_jit._vjp import get_rule

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source = bg.from_numpy(np.asarray(valid["inputValues"], dtype=np.float32))
cotangent = bg.from_numpy(np.asarray(valid["cotangent"], dtype=np.float32))
gradient = bg.func.grad(
    lambda value: (bg.sort(value, stable=True)[0] * cotangent).sum()
)(source)

vmap_spec = fixture["vmap"]
mapped_source = bg.from_numpy(np.asarray(vmap_spec["inputValues"], dtype=np.float32))
mapped_values = bg.func.vmap(lambda row: bg.sort(row, stable=True)[0])(mapped_source)
mapped_indices = bg.func.vmap(lambda row: bg.sort(row, stable=True)[1])(mapped_source)

${ONNX_PROTOBUF_TEST_HELPERS}

def parse_sort(model):
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
            if name.startswith("const_sort_k_"):
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

values, indices = bg.sort(source, stable=True)
values_onnx = parse_sort(bg.onnx.export_inference(values, input_buffers=(source,)))
indices_onnx = parse_sort(bg.onnx.export_inference(indices, input_buffers=(source,)))
bool_source = bg.from_numpy(np.asarray([True, False], dtype=np.bool_))

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

{
    "registered": [get_rule(OP_SORT_VALUES) is not None, get_rule(OP_SORT_INDICES) is not None],
    "gradient": gradient.numpy().tolist(),
    "gradientOps": sorted({node.op for node in toposort(gradient._uop)}),
    "mappedValues": mapped_values.numpy().tolist(),
    "mappedIndices": mapped_indices.numpy().tolist(),
    "mappedArgs": [dict(mapped_values._uop.arg), dict(mapped_indices._uop.arg)],
    "valuesOnnx": values_onnx,
    "indicesOnnx": indices_onnx,
    "boolOnnxError": error(lambda: bg.onnx.export_inference(bg.sort(bool_source)[0], input_buffers=(bool_source,))),
    "scalarOnnxError": error(lambda: bg.onnx.export_inference(bg.sort(bg.tensor(1.0))[0])),
    "planError": error(lambda: bg.gpu_plan_summary(values)),
    "webgpuSupported": [OP_SORT_VALUES in supported_opcodes(), OP_SORT_INDICES in supported_opcodes()],
}
`);

    expect(result).toMatchObject({
      registered: [true, true],
      gradient: FRAMEWORK_SORT_CONFORMANCE.valid.expectedGradient,
      mappedValues: FRAMEWORK_SORT_CONFORMANCE.vmap.expectedValues,
      mappedIndices: FRAMEWORK_SORT_CONFORMANCE.vmap.expectedIndices,
      mappedArgs: [
        { axis: 1, descending: false, stable: true },
        { axis: 1, descending: false, stable: true },
      ],
      valuesOnnx: {
        opTypes: FRAMEWORK_SORT_CONFORMANCE.onnx.valuesOpTypes,
        k: FRAMEWORK_SORT_CONFORMANCE.onnx.k,
        axis: FRAMEWORK_SORT_CONFORMANCE.onnx.axis,
        largest: FRAMEWORK_SORT_CONFORMANCE.onnx.largest,
        sorted: FRAMEWORK_SORT_CONFORMANCE.onnx.sorted,
        topkOutputCount: FRAMEWORK_SORT_CONFORMANCE.onnx.topkOutputCount,
      },
      indicesOnnx: {
        opTypes: FRAMEWORK_SORT_CONFORMANCE.onnx.indicesOpTypes,
        k: FRAMEWORK_SORT_CONFORMANCE.onnx.k,
        axis: FRAMEWORK_SORT_CONFORMANCE.onnx.axis,
        largest: FRAMEWORK_SORT_CONFORMANCE.onnx.largest,
        sorted: FRAMEWORK_SORT_CONFORMANCE.onnx.sorted,
        topkOutputCount: FRAMEWORK_SORT_CONFORMANCE.onnx.topkOutputCount,
      },
      webgpuSupported: [false, false],
    });
    expect(result.gradientOps).toContain("SCATTER_ADD");
    expect(result.gradientOps).not.toContain("CUSTOM");
    expect(result.boolOnnxError).toMatch(/^OnnxUnmappableOp: .*bool/u);
    expect(result.scalarOnnxError).toMatch(/^OnnxUnmappableOp: .*non-scalar/u);
    expect(result.planError).toMatch(/^GpuPlanUnsupported: .*SORT/u);
  });

  it("rejects malformed or mutated paired sort IR at every admitted boundary", async () => {
    const target = await getJitTarget();
    const result = await target.run<Record<string, string>>(`
import browsergrad_jit as bg
import numpy as np
from browsergrad_jit._ir import UOp, OP_BUFFER, OP_SORT_INDICES, OP_SORT_VALUES
from browsergrad_jit._vjp import get_rule
from browsergrad_jit._vmap import get_vmap_rule

source = bg.from_numpy(np.asarray([[3.0, 1.0, 2.0]], dtype=np.float32))
values, indices = bg.sort(source, stable=True)
mutated_values, mutated_indices = bg.sort(source, stable=True)
mutated_values._uop.arg["descending"] = True
mutated_index_values, mutated_index = bg.sort(source, stable=True)
mutated_index._uop.arg["stable"] = 1
dy = bg.from_numpy(np.ones(values.shape, dtype=np.float32))._uop
other = bg.from_numpy(np.asarray([[9.0, 8.0, 7.0]], dtype=np.float32))
large = UOp(OP_BUFFER, (), (0, (1 << 28) + 1), "float32", arg="sort:large")

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

wrong_source_indices = UOp(OP_SORT_INDICES, (other._uop,), other.shape, "int64", arg={
    "axis": 1, "descending": False, "stable": True
})

{
    "missingIndexInput": error(lambda: bg.Tensor(UOp(OP_SORT_VALUES, (source._uop,), source.shape, "float32", arg={
        "axis": 1, "descending": False, "stable": True
    }), session=source._get_session()).numpy()),
    "wrongIndexSource": error(lambda: bg.Tensor(UOp(OP_SORT_VALUES, (source._uop, wrong_source_indices), source.shape, "float32", arg={
        "axis": 1, "descending": False, "stable": True
    }), session=source._get_session()).numpy()),
    "wrongIndexDtype": error(lambda: bg.Tensor(UOp(OP_SORT_INDICES, (source._uop,), source.shape, "int32", arg={
        "axis": 1, "descending": False, "stable": True
    }), session=source._get_session()).numpy()),
    "resourceBound": error(lambda: UOp(OP_SORT_INDICES, (large,), large.shape, "int64", arg={
        "axis": 0, "descending": False, "stable": True
    })),
    "openArgCpu": error(lambda: mutated_values.numpy()),
    "openArgVjp": error(lambda: get_rule(OP_SORT_VALUES)(mutated_values._uop, mutated_values._uop.inputs, dy)),
    "openArgVmap": error(lambda: get_vmap_rule(OP_SORT_VALUES)(
        mutated_values._uop,
        {id(source._uop): source._uop, id(mutated_indices._uop): mutated_indices._uop},
        1,
    )),
    "openArgOnnx": error(lambda: bg.onnx.export_inference(mutated_values, input_buffers=(source,))),
    "openArgPlan": error(lambda: bg.gpu_plan_summary(mutated_values)),
    "indexArgCpu": error(lambda: mutated_index.numpy()),
}
`);

    expect(result.missingIndexInput).toMatch(/^ShapeError: .*source and SORT_INDICES/u);
    expect(result.wrongIndexSource).toMatch(/^ShapeError: .*exact same source/u);
    expect(result.wrongIndexDtype).toMatch(/^ShapeError: .*int64/u);
    expect(result.resourceBound).toMatch(/^ShapeError: .*per-axis ceiling/u);
    expect(result.openArgCpu).toMatch(/^RealizationError: .*ordering arguments/u);
    expect(result.openArgVjp).toMatch(/^ShapeError: .*ordering arguments/u);
    expect(result.openArgVmap).toMatch(/^ShapeError: .*ordering arguments/u);
    expect(result.openArgOnnx).toMatch(/^ShapeError: .*ordering arguments/u);
    expect(result.openArgPlan).toMatch(/^ShapeError: .*ordering arguments/u);
    expect(result.indexArgCpu).toMatch(/^RealizationError: .*booleans/u);
  });
});
