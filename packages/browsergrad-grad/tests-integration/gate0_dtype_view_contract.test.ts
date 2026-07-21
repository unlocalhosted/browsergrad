import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { beforeAll, describe, expect, it } from "vitest";

import { clearNamespace, getGradTarget } from "./pyodide-host";

interface BehaviorCase {
  readonly id: string;
  readonly expected: Readonly<Record<string, unknown>>;
}

interface Fixture {
  readonly schemaVersion: 1;
  readonly adapterId: "grad.view-bf16-compat.v0";
  readonly environment: {
    readonly pyodide: string;
    readonly numpy: string;
  };
  readonly cases: readonly BehaviorCase[];
}

const fixture = JSON.parse(readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "fixtures/grad-view-bf16.v0.json"),
  "utf8",
)) as Fixture;

let target: Awaited<ReturnType<typeof getGradTarget>>;

describe("Gate 0 Grad dtype/view/materialization contract", () => {
  beforeAll(async () => {
    target = await getGradTarget();
    await clearNamespace(target);
  });

  it("matches the versioned compatibility behavior fixture", async () => {
    const actual = await target.run<{
      environment: { pyodide: string; numpy: string };
      cases: Record<string, Record<string, unknown>>;
    }>(`
import numpy as np
import pyodide
import browsergrad_grad as grad
grad.install_torch_alias()
import torch

cases = {}

dtype_aliases = [
    "bf16", "bfloat16", "bool", "double", "float", "float16", "float32",
    "float64", "fp16", "half", "int", "int16", "int32", "int64", "int8",
    "long", "short", "uint8",
]
cases["grad.dtype.alias-registry.v0"] = {
    alias: [grad.Tensor([0], dtype=alias).dtype, int(grad.Tensor([0], dtype=alias).data.dtype.itemsize)]
    for alias in dtype_aliases
}
numpy_fallback = grad.Tensor([1 + 2j], dtype="complex64")
cases["grad.dtype.numpy-string-fallback.v0"] = {
    "dtype": numpy_fallback.dtype,
    "itemsize": int(numpy_fallback.data.dtype.itemsize),
}
cases["grad.dtype.constructor-defaults.v0"] = {
    "tensorInteger": grad.Tensor([1, 2]).dtype,
    "tensorFloat": grad.Tensor([1.0, 2.0]).dtype,
    "torchInteger": torch.tensor([1, 2]).dtype,
    "torchFloat": torch.tensor([1.0, 2.0]).dtype,
}

bf16_tensor = grad.Tensor(np.array([1.00390625, 2.0], dtype=np.float32), dtype="bf16")
cases["grad.dtype.bf16-is-f32.v0"] = {
    "dtype": bf16_tensor.dtype,
    "itemsize": int(bf16_tensor.data.dtype.itemsize),
    "storedValue": float(bf16_tensor.data[0]),
}
cases["grad.dtype.torch-bfloat16-token-is-f32.v0"] = {
    "token": torch.bfloat16,
    "distinctFromFloat32": bool(torch.bfloat16 != torch.float32),
}

source_f32 = np.array([1.0, 2.0], dtype=np.float32)
from_f32 = grad.from_numpy(source_f32)
source_f32[0] = 7.0
source_mutation_visible = float(from_f32.data[0]) == 7.0
from_f32.data[1] = 9.0
cases["grad.interop.from-numpy-f32-alias.v0"] = {
    "sharesMemory": bool(np.shares_memory(source_f32, from_f32.data)),
    "sourceMutationVisible": bool(source_mutation_visible),
    "tensorMutationVisible": bool(float(source_f32[1]) == 9.0),
    "dtype": from_f32.dtype,
}

source_f64 = np.array([1.0, 2.0], dtype=np.float64)
from_f64 = grad.from_numpy(source_f64)
source_f64[0] = 7.0
cases["grad.interop.from-numpy-f64-materializes.v0"] = {
    "sharesMemory": bool(np.shares_memory(source_f64, from_f64.data)),
    "sourceMutationVisible": bool(float(from_f64.data[0]) == 7.0),
    "dtype": from_f64.dtype,
}

base = grad.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
reshape_contiguous = base.reshape(3, 2)
base.data[0, 0] = 17.0
reshape_source_mutation_visible = float(reshape_contiguous.data[0, 0]) == 17.0
reshape_contiguous.data[0, 1] = 19.0
reshape_result_mutation_visible = float(base.data[0, 1]) == 19.0
reshape_contiguous.sum().backward()
cases["grad.view.reshape-contiguous-alias.v0"] = {
    "sharesMemory": bool(np.shares_memory(base.data, reshape_contiguous.data)),
    "cContiguous": bool(reshape_contiguous.data.flags.c_contiguous),
    "requiresGrad": bool(reshape_contiguous.requires_grad),
    "graphEdge": bool(reshape_contiguous._ctx is not None),
    "parentIdentity": bool(reshape_contiguous._ctx[0][0] is base),
    "sourceMutationVisible": bool(reshape_source_mutation_visible),
    "resultMutationVisible": bool(reshape_result_mutation_visible),
    "backwardGradient": base.grad.tolist(),
}
view_source = grad.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
view_direct = view_source.view(3, 2)
view_direct.sum().backward()
cases["grad.view.view-direct-alias.v0"] = {
    "sharesMemory": bool(np.shares_memory(view_source.data, view_direct.data)),
    "parentIdentity": bool(view_direct._ctx[0][0] is view_source),
    "backwardGradient": view_source.grad.tolist(),
}

transposed_for_reshape = base.transpose(0, 1)
reshape_noncontiguous = transposed_for_reshape.reshape(6)
cases["grad.view.reshape-noncontiguous-materializes.v0"] = {
    "sharesMemory": bool(np.shares_memory(transposed_for_reshape.data, reshape_noncontiguous.data)),
    "cContiguous": bool(reshape_noncontiguous.data.flags.c_contiguous),
    "requiresGrad": bool(reshape_noncontiguous.requires_grad),
    "graphEdge": bool(reshape_noncontiguous._ctx is not None),
}
float16_matrix = grad.Tensor(np.arange(6, dtype=np.float16).reshape(2, 3), dtype="float16")
float16_reshape = float16_matrix.reshape(3, 2)
cases["grad.view.reshape-float16-materializes-f32.v0"] = {
    "sharesMemory": bool(np.shares_memory(float16_matrix.data, float16_reshape.data)),
    "sourceDtype": float16_matrix.dtype,
    "resultDtype": float16_reshape.dtype,
}
transpose_source = grad.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
transpose_probe = transpose_source.transpose(0, 1)
transpose_source.data[0, 0] = 17.0
transpose_source_mutation_visible = float(transpose_probe.data[0, 0]) == 17.0
transpose_probe.data[1, 0] = 19.0
transpose_result_mutation_visible = float(transpose_source.data[0, 1]) == 19.0
transpose_probe.sum().backward()
cases["grad.view.transpose-alias.v0"] = {
    "sharesMemory": bool(np.shares_memory(transpose_source.data, transpose_probe.data)),
    "cContiguous": bool(transpose_probe.data.flags.c_contiguous),
    "requiresGrad": bool(transpose_probe.requires_grad),
    "graphEdge": bool(transpose_probe._ctx is not None),
    "parentIdentity": bool(transpose_probe._ctx[0][0] is transpose_source),
    "sourceMutationVisible": bool(transpose_source_mutation_visible),
    "resultMutationVisible": bool(transpose_result_mutation_visible),
    "backwardGradient": transpose_source.grad.tolist(),
}
float16_transpose = float16_matrix.transpose(0, 1)
cases["grad.view.transpose-float16-materializes-f32.v0"] = {
    "sharesMemory": bool(np.shares_memory(float16_matrix.data, float16_transpose.data)),
    "sourceDtype": float16_matrix.dtype,
    "resultDtype": float16_transpose.dtype,
}
bool_matrix = grad.Tensor(np.array([[True, False], [False, True]], dtype=np.bool_), dtype="bool")
bool_transpose = bool_matrix.transpose(0, 1)
cases["grad.view.transpose-bool-materializes-f32.v0"] = {
    "sharesMemory": bool(np.shares_memory(bool_matrix.data, bool_transpose.data)),
    "sourceDtype": bool_matrix.dtype,
    "resultDtype": bool_transpose.dtype,
}

base_rank3 = grad.Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), requires_grad=True)
permuted = base_rank3.permute(2, 0, 1)
base_rank3.data[0, 0, 0] = 31.0
permute_source_mutation_visible = float(permuted.data[0, 0, 0]) == 31.0
permuted.data[1, 0, 0] = 37.0
permute_result_mutation_visible = float(base_rank3.data[0, 0, 1]) == 37.0
permuted.sum().backward()
cases["grad.view.permute-alias.v0"] = {
    "sharesMemory": bool(np.shares_memory(base_rank3.data, permuted.data)),
    "cContiguous": bool(permuted.data.flags.c_contiguous),
    "requiresGrad": bool(permuted.requires_grad),
    "graphEdge": bool(permuted._ctx is not None),
    "parentIdentity": bool(permuted._ctx[0][0] is base_rank3),
    "sourceMutationVisible": bool(permute_source_mutation_visible),
    "resultMutationVisible": bool(permute_result_mutation_visible),
    "backwardGradientSum": int(base_rank3.grad.data.sum()),
}
float16_rank3 = grad.Tensor(np.arange(24, dtype=np.float16).reshape(2, 3, 4), dtype="float16")
float16_permute = float16_rank3.permute(2, 0, 1)
cases["grad.view.permute-float16-materializes-f32.v0"] = {
    "sharesMemory": bool(np.shares_memory(float16_rank3.data, float16_permute.data)),
    "sourceDtype": float16_rank3.dtype,
    "resultDtype": float16_permute.dtype,
}

expand_source = grad.Tensor(np.array([[1.0], [2.0]], dtype=np.float32), requires_grad=True)
expanded = expand_source.expand(2, 3)
expanded.sum().backward()
def expand_error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__
cases["grad.materialization.expand-copy.v0"] = {
    "sharesMemory": bool(np.shares_memory(expand_source.data, expanded.data)),
    "cContiguous": bool(expanded.data.flags.c_contiguous),
    "requiresGrad": bool(expanded.requires_grad),
    "graphEdge": bool(expanded._ctx is not None),
    "parentIdentity": bool(expanded._ctx[0][0] is expand_source),
    "backwardGradient": expand_source.grad.tolist(),
    "validMinusOneValues": expand_source.expand(-1, 3).tolist(),
    "invalidShapeErrors": {
        "float": expand_error(lambda: expand_source.expand(2, 3.0)),
        "bool": expand_error(lambda: expand_source.expand(2, True)),
        "leadingMinusOne": expand_error(lambda: grad.Tensor(np.array([1.0, 2.0], dtype=np.float32)).expand(-1, 2)),
        "negative": expand_error(lambda: expand_source.expand(2, -2)),
        "incompatible": expand_error(lambda: expand_source.expand(3, 3)),
        "fewerDimensions": expand_error(lambda: expand_source.expand(2)),
    },
}
float16_expand_source = grad.Tensor(np.array([[1.0], [2.0]], dtype=np.float16), dtype="float16")
float16_expanded = float16_expand_source.expand(2, 3)
int32_expand_source = grad.Tensor(np.array([[1], [2]], dtype=np.int32), dtype="int32")
int32_expanded = int32_expand_source.expand(2, 3)
cases["grad.materialization.expand-dtype-preservation.v0"] = {
    "float16SourceDtype": float16_expand_source.dtype,
    "float16ResultDtype": float16_expanded.dtype,
    "int32SourceDtype": int32_expand_source.dtype,
    "int32ResultDtype": int32_expanded.dtype,
}

index_source = grad.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
basic_slice = index_source[:, 1:]
index_source.data[0, 1] = 17.0
slice_source_mutation_visible = float(basic_slice.data[0, 0]) == 17.0
basic_slice.data[0, 1] = 19.0
slice_result_mutation_visible = float(index_source.data[0, 2]) == 19.0
basic_slice.sum().backward()
cases["grad.view.basic-slice-alias.v0"] = {
    "sharesMemory": bool(np.shares_memory(index_source.data, basic_slice.data)),
    "requiresGrad": bool(basic_slice.requires_grad),
    "graphEdge": bool(basic_slice._ctx is not None),
    "parentIdentity": bool(basic_slice._ctx[0][0] is index_source),
    "sourceMutationVisible": bool(slice_source_mutation_visible),
    "resultMutationVisible": bool(slice_result_mutation_visible),
    "backwardGradient": index_source.grad.tolist(),
}
fancy_source = grad.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
fancy_index = fancy_source[[0, 0]]
fancy_index.sum().backward()
cases["grad.view.fancy-index-materializes.v0"] = {
    "sharesMemory": bool(np.shares_memory(fancy_source.data, fancy_index.data)),
    "requiresGrad": bool(fancy_index.requires_grad),
    "graphEdge": bool(fancy_index._ctx is not None),
    "parentIdentity": bool(fancy_index._ctx[0][0] is fancy_source),
    "backwardGradient": fancy_source.grad.tolist(),
}
int64_source = grad.Tensor(np.arange(6, dtype=np.int64).reshape(2, 3), dtype="int64")
int64_slice = int64_source[:, 1:]
cases["grad.view.basic-slice-int64-materializes-f32.v0"] = {
    "sharesMemory": bool(np.shares_memory(int64_source.data, int64_slice.data)),
    "sourceDtype": int64_source.dtype,
    "resultDtype": int64_slice.dtype,
}

contiguous_result = transposed_for_reshape.contiguous()
cases["grad.materialization.contiguous-noncontiguous-noop.v0"] = {
    "sameObject": bool(contiguous_result is transposed_for_reshape),
    "cContiguousBefore": bool(transposed_for_reshape.data.flags.c_contiguous),
    "cContiguousAfter": bool(contiguous_result.data.flags.c_contiguous),
    "graphPreserved": bool(contiguous_result._ctx is transposed_for_reshape._ctx),
}

detached = base.detach()
cases["grad.materialization.detach-copy.v0"] = {
    "sharesMemory": bool(np.shares_memory(base.data, detached.data)),
    "requiresGrad": bool(detached.requires_grad),
    "graphEdge": bool(detached._ctx is not None),
}
float16_detached = float16_matrix.detach()
cases["grad.materialization.detach-float16-to-f32.v0"] = {
    "sourceDtype": float16_matrix.dtype,
    "resultDtype": float16_detached.dtype,
    "requiresGrad": bool(float16_detached.requires_grad),
}

numpy_source = grad.Tensor(np.array([1.0, 2.0], dtype=np.float32))
numpy_copy = numpy_source.numpy()
numpy_source.data[0] = 7.0
source_mutation_reached_copy = float(numpy_copy[0]) == 7.0
numpy_copy[1] = 9.0
cases["grad.interop.numpy-copy.v0"] = {
    "sharesMemory": bool(np.shares_memory(numpy_source.data, numpy_copy)),
    "sourceMutationVisible": bool(source_mutation_reached_copy),
    "arrayMutationVisible": bool(float(numpy_source.data[1]) == 9.0),
}

array_source = grad.Tensor(np.array([1.0, 2.0], dtype=np.float32))
array_alias = np.asarray(array_source)
array_source.data[0] = 7.0
source_mutation_reached_alias = float(array_alias[0]) == 7.0
array_alias[1] = 9.0
cases["grad.interop.array-protocol-alias.v0"] = {
    "sharesMemory": bool(np.shares_memory(array_source.data, array_alias)),
    "sourceMutationVisible": bool(source_mutation_reached_alias),
    "arrayMutationVisible": bool(float(array_source.data[1]) == 9.0),
}

same_dtype = base.to("float32")
cases["grad.conversion.to-same-dtype-identity.v0"] = {
    "sameObject": bool(same_dtype is base),
    "requiresGrad": bool(same_dtype.requires_grad),
    "graphPreserved": bool(same_dtype._ctx is base._ctx),
}
cross_dtype = base.to("float64")
cases["grad.conversion.to-cross-dtype-detaches.v0"] = {
    "sharesMemory": bool(np.shares_memory(base.data, cross_dtype.data)),
    "dtype": cross_dtype.dtype,
    "requiresGrad": bool(cross_dtype.requires_grad),
    "graphEdge": bool(cross_dtype._ctx is not None),
}
noncontiguous_source = grad.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3).T)
noncontiguous_cross_dtype = noncontiguous_source.to("float64")
cases["grad.conversion.to-cross-dtype-preserves-layout-order.v0"] = {
    "cContiguous": bool(noncontiguous_cross_dtype.data.flags.c_contiguous),
    "fContiguous": bool(noncontiguous_cross_dtype.data.flags.f_contiguous),
    "dtype": noncontiguous_cross_dtype.dtype,
}
unrecognized_to = base.to("definitely-not-a-browsergrad-dtype")
cases["grad.conversion.to-unrecognized-string-noop.v0"] = {
    "sameObject": bool(unrecognized_to is base),
    "dtype": unrecognized_to.dtype,
    "requiresGrad": bool(unrecognized_to.requires_grad),
}

{
    "environment": {"pyodide": pyodide.__version__, "numpy": np.__version__},
    "cases": cases,
}
`);

    expect(actual.environment).toEqual(fixture.environment);
    expect(Object.keys(actual.cases).sort()).toEqual(fixture.cases.map((testCase) => testCase.id).sort());
    for (const testCase of fixture.cases) {
      expect(actual.cases[testCase.id], testCase.id).toEqual(testCase.expected);
    }
  });
});
