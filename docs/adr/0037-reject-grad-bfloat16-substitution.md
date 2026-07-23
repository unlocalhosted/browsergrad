# ADR-0037 — Reject Grad Bfloat16 Substitution

- **Status:** Accepted
- **Date:** 2026-07-23
- **Owners:** `@unlocalhosted/browsergrad-grad`
- **Retires:** `grad.dtype.bf16-is-f32.v0`,
  `grad.dtype.torch-bfloat16-token-is-f32.v0`

## Context

`browsergrad-grad` mapped both `bf16` spellings and `torch.bfloat16` to NumPy
`float32`. That behavior used four-byte storage, retained float32 values, and
performed no bfloat16 rounding or conversion. It was a silent semantic
substitution, not bfloat16 support.

`nn.Linear` and `nn.Embedding` also accepted a `dtype` argument but ignored it,
always allocating float32 parameters. This hid the substitution in models that
forwarded `torch.bfloat16` through configuration.

## Decision

Keep `torch.bfloat16` as the distinct string token `"bfloat16"`. Reject
`bf16` and `bfloat16` with `NotImplementedError` before tensor allocation or
conversion. The diagnostic states that no real bfloat16 storage or conversion
exists and requires callers to choose an explicitly supported dtype.

`nn.Linear` and `nn.Embedding` now resolve their parameter dtype. They accept
their exact current float32 storage/computation profile, reject other floating
profiles as unimplemented, reject non-floating parameter dtypes, and reach the
same bfloat16 refusal instead of ignoring the request.

The Grad compatibility inventory advances to schema version 2. It removes
bfloat16 from the alias map and records the two spellings as explicitly
unsupported dtypes. The executable fixture covers direct construction,
`torch.tensor`, and `Tensor.to`; the reasoning-workshop integration covers a
model-level parameter request.

## Consequences

Existing callers that relied on fake bfloat16 must request `float32`
explicitly. BrowserGrad no longer claims bfloat16 through renamed float32
values. Real bfloat16 support remains future work requiring distinct storage,
conversion, numerical, serialization, and backend contracts.

This closes the bfloat16 half of the frozen Grad view/dtype adapter. Correct
`contiguous`, detach, conversion, and interop materialization behavior remain
separate Gate 6 work.
