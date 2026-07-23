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

def error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__ + ": " + str(exc)

dtype_aliases = [
    "bool", "double", "float", "float16", "float32", "float64", "fp16",
    "half", "int", "int16", "int32", "int64", "int8", "long", "short",
    "uint8", "uint16", "uint32", "uint64",
]
cases["grad.dtype.alias-registry.v1"] = {
    alias: [grad.Tensor([0], dtype=alias).dtype, int(grad.Tensor([0], dtype=alias).data.dtype.itemsize)]
    for alias in dtype_aliases
}
numpy_dtype_specs = [
    np.dtype("bool"), np.float16, np.dtype("float32"), np.float64,
    np.int8, np.dtype("int16"), np.int32, np.dtype("int64"),
    np.uint8, np.uint16, np.dtype("uint32"), np.uint64,
]
cases["grad.dtype.numpy-spec-registry.v1"] = {
    "dtypes": [
        grad.Tensor([0], dtype=spec).dtype
        for spec in numpy_dtype_specs
    ],
}
cases["grad.dtype.torch-unsigned-tokens.v1"] = {
    name: [
        getattr(torch, name),
        torch.tensor([0], dtype=getattr(torch, name)).dtype,
    ]
    for name in ["uint8", "uint16", "uint32", "uint64"]
}
cases["grad.dtype.unsupported-rejected.v1"] = {
    "complexString": error(lambda: grad.Tensor([1 + 2j], dtype="complex64")),
    "numpyAbbreviation": error(lambda: grad.Tensor([1.0], dtype="f4")),
    "complexType": error(lambda: grad.Tensor([1 + 2j], dtype=np.complex64)),
    "objectDtype": error(lambda: grad.Tensor([object()], dtype=np.dtype("object"))),
    "datetimeDtype": error(
        lambda: grad.Tensor([0], dtype=np.dtype("datetime64[D]"))
    ),
    "structuredDtype": error(
        lambda: grad.Tensor([(1,)], dtype=np.dtype([("value", np.int32)]))
    ),
}
direct_default_source = np.array([1.0, 2.0], dtype=np.float32)
direct_default_tensor = grad.Tensor(direct_default_source)
cases["grad.dtype.tensor-constructor-default.v1"] = {
    "tensorInteger": grad.Tensor([1, 2]).dtype,
    "tensorFloat": grad.Tensor([1.0, 2.0]).dtype,
    "tensorBoolean": grad.Tensor([True, False]).dtype,
    "tensorNumpyFloat64": grad.Tensor(
        np.array([1.0], dtype=np.float64)
    ).dtype,
    "float32ArrayAliases": bool(np.shares_memory(
        direct_default_source,
        direct_default_tensor.data,
    )),
}
torch_numpy_source = np.arange(6, dtype=np.float64).reshape(2, 3).T
torch_numpy_tensor = torch.tensor(torch_numpy_source)
torch_existing_source = grad.Tensor(
    np.array([1, 2], dtype=np.int32),
    dtype="int32",
    requires_grad=True,
)
torch_existing_tensor = torch.tensor(torch_existing_source)
torch_explicit_source = np.array([1.0, 2.0], dtype=np.float64)
torch_explicit_tensor = torch.tensor(torch_explicit_source, dtype=torch.float16)
torch_numpy_source[0, 0] = 17.0
torch_numpy_source_mutation_visible = float(torch_numpy_tensor.data[0, 0]) == 17.0
torch_numpy_tensor.data[1, 0] = 19.0
cases["grad.dtype.torch-constructor-inference.v1"] = {
    "pythonDtypes": {
        "boolean": torch.tensor([True, False]).dtype,
        "integer": torch.tensor([1, 2]).dtype,
        "floating": torch.tensor([1.0, 2.0]).dtype,
        "mixed": torch.tensor([1, 2.0]).dtype,
        "empty": torch.tensor([]).dtype,
    },
    "preservedDtypes": {
        "numpyFloat64": torch_numpy_tensor.dtype,
        "numpyUint16": torch.tensor(np.array([1], dtype=np.uint16)).dtype,
        "numpyScalarFloat16": torch.tensor(np.float16(1.0)).dtype,
        "tensorInt32": torch_existing_tensor.dtype,
        "explicitFloat16": torch_explicit_tensor.dtype,
    },
    "ownership": {
        "numpyAliases": bool(np.shares_memory(
            torch_numpy_source,
            torch_numpy_tensor.data,
        )),
        "tensorAliases": bool(np.shares_memory(
            torch_existing_source.data,
            torch_existing_tensor.data,
        )),
        "explicitAliases": bool(np.shares_memory(
            torch_explicit_source,
            torch_explicit_tensor.data,
        )),
        "numpySourceMutationVisible": bool(
            torch_numpy_source_mutation_visible
        ),
        "numpyOutputMutationVisible": bool(
            float(torch_numpy_source[1, 0]) == 19.0
        ),
        "numpyFContiguous": bool(torch_numpy_tensor.data.flags.f_contiguous),
    },
    "autograd": {
        "existingRequiresGrad": bool(torch_existing_source.requires_grad),
        "copyRequiresGrad": bool(torch_existing_tensor.requires_grad),
        "copyIsLeaf": bool(torch_existing_tensor._ctx is None),
        "requestedFloat": bool(torch.tensor(
            [1.0],
            dtype=torch.float16,
            requires_grad=True,
        ).requires_grad),
    },
    "errors": {
        "complex": error(lambda: torch.tensor([1 + 2j])),
        "object": error(lambda: torch.tensor(np.array([object()], dtype=np.object_))),
        "string": error(lambda: torch.tensor(["1"])),
        "structured": error(
            lambda: torch.tensor(np.array([(1,)], dtype=[("value", np.int32)]))
        ),
        "integerRequiresGrad": error(
            lambda: torch.tensor([1], requires_grad=True)
        ),
        "requiresGradType": error(
            lambda: torch.tensor([1.0], requires_grad=1)
        ),
    },
}

cases["grad.dtype.bf16-rejected.v1"] = {
    "bf16Constructor": error(lambda: grad.Tensor([1.0], dtype="bf16")),
    "bfloat16Constructor": error(lambda: grad.Tensor([1.0], dtype="bfloat16")),
    "toBfloat16": error(lambda: grad.Tensor([1.0]).to("bfloat16")),
}
cases["grad.dtype.torch-bfloat16-token-distinct-unsupported.v1"] = {
    "token": torch.bfloat16,
    "distinctFromFloat32": bool(torch.bfloat16 != torch.float32),
    "constructor": error(lambda: torch.tensor([1.0], dtype=torch.bfloat16)),
}

from_numpy_sources = {
    name: np.array([0, 1], dtype=np.dtype(name))
    for name in [
        "bool", "float16", "float32", "float64",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
    ]
}
from_numpy_tensors = {
    name: grad.from_numpy(source)
    for name, source in from_numpy_sources.items()
}
from_numpy_sources["float64"][0] = 7.0
from_numpy_tensors["int64"].data[1] = 9
cases["grad.interop.from-numpy-alias.v1"] = {
    "dtypes": {
        name: tensor.dtype
        for name, tensor in from_numpy_tensors.items()
    },
    "allShareMemory": bool(all(
        np.shares_memory(from_numpy_sources[name], tensor.data)
        for name, tensor in from_numpy_tensors.items()
    )),
    "sourceMutationVisible": bool(
        float(from_numpy_tensors["float64"].data[0]) == 7.0
    ),
    "tensorMutationVisible": bool(
        int(from_numpy_sources["int64"][1]) == 9
    ),
}
noncontiguous_numpy_source = np.arange(
    12,
    dtype=np.float16,
).reshape(3, 4).T[:, ::-1]
noncontiguous_numpy_tensor = grad.from_numpy(noncontiguous_numpy_source)
noncontiguous_numpy_source[0, 0] = 17.0
noncontiguous_numpy_tensor.data[1, 1] = 19.0
cases["grad.interop.from-numpy-noncontiguous-alias.v1"] = {
    "sameArrayObject": bool(
        noncontiguous_numpy_tensor.data is noncontiguous_numpy_source
    ),
    "sharesMemory": bool(np.shares_memory(
        noncontiguous_numpy_source,
        noncontiguous_numpy_tensor.data,
    )),
    "dtype": noncontiguous_numpy_tensor.dtype,
    "stridesPreserved": bool(
        noncontiguous_numpy_source.strides
        == noncontiguous_numpy_tensor.data.strides
    ),
    "cContiguous": bool(noncontiguous_numpy_tensor.data.flags.c_contiguous),
    "fContiguous": bool(noncontiguous_numpy_tensor.data.flags.f_contiguous),
    "sourceMutationVisible": bool(
        float(noncontiguous_numpy_tensor.data[0, 0]) == 17.0
    ),
    "tensorMutationVisible": bool(
        float(noncontiguous_numpy_source[1, 1]) == 19.0
    ),
}
readonly_numpy_source = np.array([1.0], dtype=np.float32)
readonly_numpy_source.flags.writeable = False
cases["grad.interop.from-numpy-invalid-rejected.v1"] = {
    "list": error(lambda: grad.from_numpy([1.0])),
    "readonly": error(lambda: grad.from_numpy(readonly_numpy_source)),
    "complex64": error(
        lambda: grad.from_numpy(np.array([1 + 2j], dtype=np.complex64))
    ),
    "object": error(
        lambda: grad.from_numpy(np.array([object()], dtype=np.object_))
    ),
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
cases["grad.view.reshape-float16-alias-preserves-dtype.v1"] = {
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
cases["grad.view.transpose-float16-alias-preserves-dtype.v1"] = {
    "sharesMemory": bool(np.shares_memory(float16_matrix.data, float16_transpose.data)),
    "sourceDtype": float16_matrix.dtype,
    "resultDtype": float16_transpose.dtype,
}
bool_matrix = grad.Tensor(np.array([[True, False], [False, True]], dtype=np.bool_), dtype="bool")
bool_transpose = bool_matrix.transpose(0, 1)
cases["grad.view.transpose-bool-alias-preserves-dtype.v1"] = {
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
cases["grad.view.permute-float16-alias-preserves-dtype.v1"] = {
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
cases["grad.view.basic-slice-int64-alias-preserves-dtype.v1"] = {
    "sharesMemory": bool(np.shares_memory(int64_source.data, int64_slice.data)),
    "sourceDtype": int64_source.dtype,
    "resultDtype": int64_slice.dtype,
}

contiguous_source = grad.Tensor(
    np.arange(6, dtype=np.float32).reshape(2, 3).T,
    dtype="float32",
    requires_grad=True,
)
contiguous_result = contiguous_source.contiguous()
contiguous_source.data[0, 0] = 17.0
contiguous_source_mutation_visible = float(contiguous_result.data[0, 0]) == 17.0
contiguous_result.data[1, 0] = 19.0
contiguous_result_mutation_visible = float(contiguous_source.data[1, 0]) == 19.0
contiguous_result.sum().backward()
cases["grad.materialization.contiguous-copy.v1"] = {
    "sameObject": bool(contiguous_result is contiguous_source),
    "sharesMemory": bool(np.shares_memory(contiguous_source.data, contiguous_result.data)),
    "cContiguousBefore": bool(contiguous_source.data.flags.c_contiguous),
    "cContiguousAfter": bool(contiguous_result.data.flags.c_contiguous),
    "requiresGrad": bool(contiguous_result.requires_grad),
    "graphEdge": bool(contiguous_result._ctx is not None),
    "parentIdentity": bool(contiguous_result._ctx[0][0] is contiguous_source),
    "sourceMutationVisible": bool(contiguous_source_mutation_visible),
    "resultMutationVisible": bool(contiguous_result_mutation_visible),
    "backwardGradient": contiguous_source.grad.tolist(),
}
contiguous_identity_source = grad.Tensor(
    np.arange(6, dtype=np.float32).reshape(2, 3),
    requires_grad=True,
)
contiguous_identity = contiguous_identity_source.contiguous()
cases["grad.materialization.contiguous-identity.v1"] = {
    "sameObject": bool(contiguous_identity is contiguous_identity_source),
    "cContiguous": bool(contiguous_identity.data.flags.c_contiguous),
    "graphPreserved": bool(contiguous_identity._ctx is contiguous_identity_source._ctx),
}
float16_contiguous_source = grad.Tensor(
    np.arange(6, dtype=np.float16).reshape(2, 3).T,
    dtype="float16",
)
float16_contiguous = float16_contiguous_source.contiguous()
cases["grad.materialization.contiguous-float16-copy.v1"] = {
    "sharesMemory": bool(np.shares_memory(
        float16_contiguous_source.data,
        float16_contiguous.data,
    )),
    "sourceDtype": float16_contiguous_source.dtype,
    "resultDtype": float16_contiguous.dtype,
    "cContiguous": bool(float16_contiguous.data.flags.c_contiguous),
}

detach_source = grad.Tensor(
    np.array([1.0, 2.0], dtype=np.float32),
    requires_grad=True,
)
detached = detach_source.detach()
detach_source.data[0] = 7.0
source_mutation_visible = float(detached.data[0]) == 7.0
detached.data[1] = 9.0
cases["grad.view.detach-alias.v1"] = {
    "sameObject": bool(detached is detach_source),
    "sharesMemory": bool(np.shares_memory(detach_source.data, detached.data)),
    "requiresGrad": bool(detached.requires_grad),
    "graphEdge": bool(detached._ctx is not None),
    "isLeaf": bool(detached._is_leaf),
    "sourceMutationVisible": bool(source_mutation_visible),
    "resultMutationVisible": bool(float(detach_source.data[1]) == 9.0),
}
float16_detach_source = grad.Tensor(
    np.arange(6, dtype=np.float16).reshape(2, 3),
    dtype="float16",
    requires_grad=True,
)
float16_detached = float16_detach_source.detach()
cases["grad.view.detach-float16-alias.v1"] = {
    "sharesMemory": bool(np.shares_memory(
        float16_detach_source.data,
        float16_detached.data,
    )),
    "sourceDtype": float16_detach_source.dtype,
    "resultDtype": float16_detached.dtype,
    "requiresGrad": bool(float16_detached.requires_grad),
}
noncontiguous_detach_source = grad.Tensor(
    np.arange(6, dtype=np.float32).reshape(2, 3).T,
    dtype="float32",
    requires_grad=True,
)
noncontiguous_detached = noncontiguous_detach_source.detach()
cases["grad.view.detach-noncontiguous-alias.v1"] = {
    "sharesMemory": bool(np.shares_memory(
        noncontiguous_detach_source.data,
        noncontiguous_detached.data,
    )),
    "cContiguousBefore": bool(
        noncontiguous_detach_source.data.flags.c_contiguous
    ),
    "cContiguousAfter": bool(noncontiguous_detached.data.flags.c_contiguous),
    "stridesPreserved": bool(
        noncontiguous_detach_source.data.strides
        == noncontiguous_detached.data.strides
    ),
}

numpy_source = grad.Tensor(
    np.arange(6, dtype=np.float16).reshape(2, 3).T,
    dtype="float16",
)
numpy_snapshot = numpy_source.numpy()
array_snapshot = np.asarray(numpy_source)
typed_array_snapshot = np.asarray(numpy_source, dtype=np.float64)
numpy_source.data[0, 0] = 17.0
source_mutation_reached_numpy = float(numpy_snapshot[0, 0]) == 17.0
source_mutation_reached_array = float(array_snapshot[0, 0]) == 17.0
numpy_snapshot[1, 0] = 19.0
array_snapshot[2, 0] = 23.0
cases["grad.interop.numpy-export-snapshot.v1"] = {
    "numpySharesMemory": bool(np.shares_memory(
        numpy_source.data,
        numpy_snapshot,
    )),
    "arraySharesMemory": bool(np.shares_memory(
        numpy_source.data,
        array_snapshot,
    )),
    "sourceMutationReachedNumpy": bool(source_mutation_reached_numpy),
    "sourceMutationReachedArray": bool(source_mutation_reached_array),
    "numpyMutationReachedSource": bool(
        float(numpy_source.data[1, 0]) == 19.0
    ),
    "arrayMutationReachedSource": bool(
        float(numpy_source.data[2, 0]) == 23.0
    ),
    "sourceDtype": numpy_source.dtype,
    "numpyDtype": numpy_snapshot.dtype.name,
    "arrayDtype": array_snapshot.dtype.name,
    "typedArrayDtype": typed_array_snapshot.dtype.name,
    "numpyFContiguous": bool(numpy_snapshot.flags.f_contiguous),
    "arrayFContiguous": bool(array_snapshot.flags.f_contiguous),
    "copyFalseError": error(
        lambda: numpy_source.__array__(copy=False)
    ),
    "invalidCopyError": error(
        lambda: numpy_source.__array__(copy="yes")
    ),
}

same_dtype = base.to("float32")
cases["grad.conversion.to-same-dtype-identity.v0"] = {
    "sameObject": bool(same_dtype is base),
    "requiresGrad": bool(same_dtype.requires_grad),
    "graphPreserved": bool(same_dtype._ctx is base._ctx),
}
cross_dtype_source = grad.Tensor(
    np.array([1.5, -2.25], dtype=np.float32),
    dtype="float32",
    requires_grad=True,
)
cross_dtype = cross_dtype_source.to("float64")
cross_dtype.backward(grad.Tensor(
    np.array([1.0, 2.0], dtype=np.float64),
    dtype="float64",
))
cases["grad.conversion.to-cross-floating-autograd.v1"] = {
    "sharesMemory": bool(np.shares_memory(
        cross_dtype_source.data,
        cross_dtype.data,
    )),
    "dtype": cross_dtype.dtype,
    "requiresGrad": bool(cross_dtype.requires_grad),
    "graphEdge": bool(cross_dtype._ctx is not None),
    "parentIdentity": bool(cross_dtype._ctx[0][0] is cross_dtype_source),
    "sourceGradientDtype": cross_dtype_source.grad.dtype,
    "sourceGradient": cross_dtype_source.grad.tolist(),
}
noncontiguous_source = grad.Tensor(
    np.arange(6, dtype=np.float16).reshape(2, 3).T,
    dtype="float16",
    requires_grad=True,
)
noncontiguous_cross_dtype = noncontiguous_source.to("float64")
noncontiguous_cross_dtype.backward(grad.Tensor(
    np.ones((3, 2), dtype=np.float64),
    dtype="float64",
))
cases["grad.conversion.to-cross-floating-float16-layout.v1"] = {
    "cContiguous": bool(noncontiguous_cross_dtype.data.flags.c_contiguous),
    "fContiguous": bool(noncontiguous_cross_dtype.data.flags.f_contiguous),
    "sourceDtype": noncontiguous_source.dtype,
    "resultDtype": noncontiguous_cross_dtype.dtype,
    "requiresGrad": bool(noncontiguous_cross_dtype.requires_grad),
    "sourceGradientDtype": noncontiguous_source.grad.dtype,
    "sourceGradient": noncontiguous_source.grad.tolist(),
}
no_grad_source = grad.Tensor(
    np.array([1.0, 2.0], dtype=np.float32),
    requires_grad=True,
)
with grad.no_grad():
    no_grad_cross_dtype = no_grad_source.to("float64")
cases["grad.conversion.to-cross-floating-no-grad.v1"] = {
    "dtype": no_grad_cross_dtype.dtype,
    "requiresGrad": bool(no_grad_cross_dtype.requires_grad),
    "graphEdge": bool(no_grad_cross_dtype._ctx is not None),
}
nonfloating_source = grad.Tensor(
    np.array([1.5, -2.25], dtype=np.float32),
    requires_grad=True,
)
nonfloating_cross_dtype = nonfloating_source.to("int64")
cases["grad.conversion.to-cross-nonfloating-detached.v1"] = {
    "dtype": nonfloating_cross_dtype.dtype,
    "values": nonfloating_cross_dtype.tolist(),
    "requiresGrad": bool(nonfloating_cross_dtype.requires_grad),
    "graphEdge": bool(nonfloating_cross_dtype._ctx is not None),
}
def to_error(call):
    try:
        call()
        return "no_error"
    except Exception as exc:
        return type(exc).__name__
cases["grad.conversion.to-invalid-request-rejected.v1"] = {
    "invalidPositional": to_error(
        lambda: base.to("definitely-not-a-browsergrad-dtype")
    ),
    "invalidKeyword": to_error(
        lambda: base.to(dtype="definitely-not-a-browsergrad-dtype")
    ),
    "unsupportedKeyword": to_error(lambda: base.to(copy=True)),
    "duplicateDtype": to_error(
        lambda: base.to("float64", dtype="float32")
    ),
    "duplicateDtypeNone": to_error(
        lambda: base.to("float64", dtype=None)
    ),
    "duplicateDeviceNone": to_error(
        lambda: base.to("cpu", device=None)
    ),
    "tooManyArguments": to_error(
        lambda: base.to("cpu", "float64", False)
    ),
    "ambiguousTwoPositional": to_error(
        lambda: base.to("float64", "cpu")
    ),
    "invalidDeviceType": to_error(lambda: base.to(device=object())),
}
cpu_and_dtype_source = grad.Tensor(
    np.array([1.0, 2.0], dtype=np.float32),
    requires_grad=True,
)
cpu_and_dtype = cpu_and_dtype_source.to("cpu", dtype="float64")
other_dtype = grad.Tensor(
    np.array([0.0], dtype=np.float64),
    dtype="float64",
)
to_other = cpu_and_dtype_source.to(other_dtype)
cases["grad.device.tensor-cpu-identity.v1"] = {
    "positionalSameObject": bool(base.to("cpu") is base),
    "keywordSameObject": bool(base.to(device="cpu") is base),
    "cpuMethodSameObject": bool(base.cpu() is base),
    "combinedDtype": cpu_and_dtype.dtype,
    "combinedRequiresGrad": bool(cpu_and_dtype.requires_grad),
    "combinedParentIdentity": bool(
        cpu_and_dtype._ctx[0][0] is cpu_and_dtype_source
    ),
    "otherDtype": to_other.dtype,
    "otherParentIdentity": bool(
        to_other._ctx[0][0] is cpu_and_dtype_source
    ),
}
cases["grad.device.tensor-unsupported-rejected.v1"] = {
    "toCuda": to_error(lambda: base.to("cuda")),
    "toCudaIndex": to_error(lambda: base.to(device="cuda:0")),
    "toMps": to_error(lambda: base.to("mps")),
    "toIndexedCpu": to_error(lambda: base.to("cpu:0")),
    "cudaMethod": to_error(lambda: base.cuda()),
}
cases["grad.device.torch-tensor-constructor.v1"] = {
    "cpuDtype": torch.tensor([1, 2], device="cpu").dtype,
    "cudaError": to_error(
        lambda: torch.tensor([1.0], device="cuda")
    ),
    "cudaIndexError": to_error(
        lambda: torch.tensor([1.0], device="cuda:0")
    ),
    "invalidDeviceType": to_error(
        lambda: torch.tensor([1.0], device=object())
    ),
}
module = torch.nn.Linear(2, 2)
cases["grad.device.module-to.v1"] = {
    "noArgumentSameObject": bool(module.to() is module),
    "cpuSameObject": bool(module.to("cpu") is module),
    "cpuKeywordSameObject": bool(module.to(device="cpu") is module),
    "cudaError": to_error(lambda: module.to("cuda:0")),
    "dtypeError": to_error(lambda: module.to(dtype=torch.float64)),
    "unsupportedKeyword": to_error(lambda: module.to(non_blocking=True)),
    "duplicateDevice": to_error(
        lambda: module.to("cpu", device="cpu")
    ),
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
