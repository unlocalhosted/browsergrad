import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { FRAMEWORK_TOPK_CONFORMANCE } from "../../../test-support/framework-topk-conformance";
import { clearNamespace, getGradTarget } from "./pyodide-host";

describe("Gate 6 eager torch.topk conformance", () => {
  beforeAll(async () => {
    await getGradTarget();
  }, 120_000);

  beforeEach(async () => {
    await clearNamespace(await getGradTarget());
  });

  it("shares selection, dtype, ownership, gradient, resource, and refusal semantics", async () => {
    const target = await getGradTarget();
    const fixtureJson = JSON.stringify(FRAMEWORK_TOPK_CONFORMANCE);
    const result = await target.run<Record<string, unknown>>(`
import browsergrad_grad as grad
import json
import numpy as np

fixture = json.loads(${JSON.stringify(fixtureJson)})
valid = fixture["valid"]
source_array = np.asarray(valid["inputValues"], dtype=np.float32)
source = grad.Tensor(source_array, requires_grad=True, dtype="float32")
largest_values, largest_indices = grad.topk(
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
largest_values.backward(
    grad.Tensor(np.asarray(valid["cotangent"], dtype=np.float32), dtype="float32")
)

dtype_cases = []
for case in fixture["dtypeCases"]:
    typed_array = np.asarray(case["input"], dtype=np.dtype(case["dtype"]))
    typed = grad.Tensor(typed_array, dtype=case["dtype"])
    values, indices = grad.topk(typed, len(case["input"]), sorted=True)
    dtype_cases.append({
        "valueDtype": values.dtype,
        "indexDtype": indices.dtype,
        "values": values.data.tolist(),
        "gatherMatches": bool(np.array_equal(
            values.data,
            np.take_along_axis(typed_array, indices.data, axis=0),
        )),
        "indicesUnique": len(set(indices.data.tolist())) == len(case["input"]),
    })

empty_values, empty_indices = grad.topk(
    grad.Tensor(
        np.empty(tuple(fixture["empty"]["inputShape"]), dtype=np.float32),
        dtype="float32",
    ),
    fixture["empty"]["k"],
)
zero_values, zero_indices = grad.topk(
    grad.Tensor(
        np.ones(tuple(fixture["zeroK"]["inputShape"]), dtype=np.float32),
        dtype="float32",
    ),
    0,
)

unsorted_spec = fixture["unsorted"]
unsorted_source = grad.Tensor(
    np.asarray(unsorted_spec["input"], dtype=np.float32),
    dtype="float32",
)
unsorted_values, unsorted_indices = grad.topk(
    unsorted_source,
    unsorted_spec["k"],
    largest=False,
    sorted=False,
)

tie_source = grad.Tensor(np.asarray([5, 1, 5, 5, 2], dtype=np.float32), dtype="float32")
tie_values_a, tie_indices_a = grad.topk(tie_source, 2)
_, tie_indices_b = grad.topk(tie_source, 2)

mutable_source = grad.Tensor(source_array, requires_grad=True, dtype="float32")
mutable_values, mutable_indices = grad.topk(mutable_source, valid["k"])
mutable_indices.data[...] = 0
mutable_values.backward(
    grad.Tensor(np.asarray(valid["cotangent"], dtype=np.float32), dtype="float32")
)

uint_values, uint_indices = grad.topk(
    grad.Tensor(np.asarray([0, 255, 1], dtype=np.uint8), dtype="uint8"),
    2,
)
signed_min_values, signed_min_indices = grad.topk(
    grad.Tensor(
        np.asarray([np.iinfo(np.int64).min, 1, 0], dtype=np.int64),
        dtype="int64",
    ),
    2,
)
nan_source = grad.Tensor(
    np.asarray([np.nan, 1.0, np.nan, 2.0], dtype=np.float32),
    dtype="float32",
)
nan_largest_values, _ = grad.topk(nan_source, 3)
nan_smallest_values, _ = grad.topk(nan_source, 2, largest=False)

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
    "non-tensor": lambda: grad.topk([[1.0]], 1),
    "bool-k": lambda: grad.topk(source, True),
    "float-k": lambda: grad.topk(source, 1.0),
    "negative-k": lambda: grad.topk(source, -1),
    "oversized-k": lambda: grad.topk(source, 7),
    "bool-dim": lambda: grad.topk(source, 1, dim=True),
    "float-dim": lambda: grad.topk(source, 1, dim=1.0),
    "out-of-range-dim": lambda: grad.topk(source, 1, dim=2),
    "largest-type": lambda: grad.topk(source, 1, largest=1),
    "sorted-type": lambda: grad.topk(source, 1, sorted=1),
    "out-mutation": lambda: grad.topk(source, 1, out=(source, source)),
    "unsupported-dtype": lambda: grad.topk(
        grad.Tensor(np.ones((1,), dtype=np.uint16), dtype="uint16"), 1
    ),
    "scalar-input": lambda: grad.topk(grad.Tensor(np.asarray(1.0), dtype="float32"), 1),
    "hostile-k": lambda: grad.topk(source, HostileIndex()),
    "hostile-dim": lambda: grad.topk(source, 1, dim=HostileIndex()),
    "hostile-largest": lambda: grad.topk(source, 1, largest=HostileBool()),
    "hostile-sorted": lambda: grad.topk(source, 1, sorted=HostileBool()),
    "zero-size-oversized-axis": lambda: grad.topk(
        grad.Tensor(np.empty((0, (1 << 20) + 1), dtype=np.float32), dtype="float32"),
        0,
        dim=1,
    ),
    "zero-size-oversized-extent": lambda: grad.topk(
        grad.Tensor(np.empty((0, (1 << 28) + 1), dtype=np.float32), dtype="float32"),
        0,
        dim=0,
    ),
}
for case in fixture["invalid"]:
    try:
        attempts[case["id"]]()
        errors[case["id"]] = "no_error"
    except Exception as exc:
        errors[case["id"]] = type(exc).__name__ + ": " + str(exc)

values_array = largest_values.data
indices_array = largest_indices.data
values_own = bool(values_array.flags["OWNDATA"])
indices_own = bool(indices_array.flags["OWNDATA"])
values_array[0, 0] = 999
indices_array[0, 0] = 999
rerun_values, rerun_indices = grad.topk(source, valid["k"])

large = grad.Tensor(
    np.arange(4 * 50257, dtype=np.float32).reshape(4, 50257),
    dtype="float32",
)
large_values, large_indices = grad.topk(large, 10)

tie_a = tie_indices_a.data
tie_b = tie_indices_b.data
{
    "largestValues": rerun_values.data.tolist(),
    "largestIndices": rerun_indices.data.tolist(),
    "smallestValues": smallest_values.data.tolist(),
    "smallestIndices": smallest_indices.data.tolist(),
    "gradient": source.grad.data.tolist(),
    "mutationSafeGradient": mutable_source.grad.data.tolist(),
    "dtypeCases": dtype_cases,
    "empty": [list(empty_values.shape), list(empty_indices.shape)],
    "zeroK": [list(zero_values.shape), list(zero_indices.shape)],
    "unsorted": {
        "values": sorted(unsorted_values.data.tolist()),
        "indices": sorted(unsorted_indices.data.tolist()),
    },
    "ties": {
        "values": tie_values_a.data.tolist(),
        "indicesValid": bool(all(int(index) in (0, 2, 3) for index in tie_a)),
        "indicesUnique": len(set(tie_a.tolist())) == 2,
        "deterministic": bool(np.array_equal(tie_a, tie_b)),
    },
    "edgeOrdering": {
        "uintValues": uint_values.data.tolist(),
        "uintIndices": uint_indices.data.tolist(),
        "signedMinValuesCorrect": bool(np.array_equal(
            signed_min_values.data, np.asarray([1, 0], dtype=np.int64)
        )),
        "signedMinIndices": signed_min_indices.data.tolist(),
        "nanLargest": [bool(np.isnan(value)) for value in nan_largest_values.data],
        "nanSmallest": nan_smallest_values.data.tolist(),
    },
    "largeShapes": [list(large_values.shape), list(large_indices.shape)],
    "ownsData": [values_own, indices_own],
    "errors": errors,
    "hostileCalls": {"index": HostileIndex.calls, "bool": HostileBool.calls},
}
`);

    expect(result).toMatchObject({
      largestValues: FRAMEWORK_TOPK_CONFORMANCE.valid.largestValues,
      largestIndices: FRAMEWORK_TOPK_CONFORMANCE.valid.largestIndices,
      smallestValues: FRAMEWORK_TOPK_CONFORMANCE.valid.smallestValues,
      smallestIndices: FRAMEWORK_TOPK_CONFORMANCE.valid.smallestIndices,
      gradient: FRAMEWORK_TOPK_CONFORMANCE.valid.expectedGradient,
      mutationSafeGradient: FRAMEWORK_TOPK_CONFORMANCE.valid.expectedGradient,
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
});
