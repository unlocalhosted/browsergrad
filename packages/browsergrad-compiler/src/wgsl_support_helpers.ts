import { poolDataName, poolOffsetName } from "./pool_bindings.js";

export interface WgslNameContext {
  nameFor(name: string): string;
}

export interface WgslRawPoolAllocator {
  readonly baseName: string;
  readonly offsetName: string;
}

export function rawPoolHelperName(baseName: string, offsetName: string): string {
  return `bg_raw_pool_alloc_${baseName}_${offsetName}`;
}

export function emitPoolHelper(name: string, context: WgslNameContext): string[] {
  const dataName = context.nameFor(poolDataName(name));
  const offsetName = context.nameFor(poolOffsetName(name));
  return [
    `fn bg_pool_alloc_${name}(size_bytes: u32) -> u32 {`,
    `  let old = atomicAdd(&${offsetName}, size_bytes);`,
    `  let capacity = arrayLength(&${dataName}) * 4u;`,
    "  if ((old + size_bytes) > capacity) {",
    "    return 0u;",
    "  }",
    "  return old + 1u;",
    "}",
  ];
}

export function emitRawPoolHelper(allocator: WgslRawPoolAllocator): string[] {
  return [
    `fn ${rawPoolHelperName(allocator.baseName, allocator.offsetName)}(pool_size_bytes: u32, size_bytes: u32) -> u32 {`,
    `  let old = atomicAdd(&${allocator.offsetName}[0], size_bytes);`,
    "  if ((old + size_bytes) > pool_size_bytes) {",
    "    return 0u;",
    "  }",
    "  return old + 1u;",
    "}",
  ];
}

export function emitCurandHelpers(): string[] {
  return [
    "fn bg_curand_next(state: ptr<function, u32>) -> u32 {",
    "  var x = *state;",
    "  x = (x * 1664525u) + 1013904223u;",
    "  *state = x;",
    "  return x;",
    "}",
    "fn bg_curand_init(seed: u32, sequence: u32, offset: u32, state: ptr<function, u32>) {",
    "  *state = seed ^ (sequence * 747796405u) ^ offset ^ 2891336453u;",
    "  _ = bg_curand_next(state);",
    "}",
    "fn bg_curand_uniform(state: ptr<function, u32>) -> f32 {",
    "  let bits = bg_curand_next(state);",
    "  return (f32(bits) + 1.0) * 2.3283064365386963e-10;",
    "}",
    "fn bg_curand_normal(state: ptr<function, u32>) -> f32 {",
    "  let u1 = max(bg_curand_uniform(state), 1.1754943508222875e-38);",
    "  let u2 = bg_curand_uniform(state);",
    "  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);",
    "}",
    "fn bg_curand_next_storage(state: ptr<storage, u32, read_write>) -> u32 {",
    "  var x = *state;",
    "  x = (x * 1664525u) + 1013904223u;",
    "  *state = x;",
    "  return x;",
    "}",
    "fn bg_curand_init_storage(seed: u32, sequence: u32, offset: u32, state: ptr<storage, u32, read_write>) {",
    "  *state = seed ^ (sequence * 747796405u) ^ offset ^ 2891336453u;",
    "  _ = bg_curand_next_storage(state);",
    "}",
    "fn bg_curand_uniform_storage(state: ptr<storage, u32, read_write>) -> f32 {",
    "  let bits = bg_curand_next_storage(state);",
    "  return (f32(bits) + 1.0) * 2.3283064365386963e-10;",
    "}",
    "fn bg_curand_normal_storage(state: ptr<storage, u32, read_write>) -> f32 {",
    "  let u1 = max(bg_curand_uniform_storage(state), 1.1754943508222875e-38);",
    "  let u2 = bg_curand_uniform_storage(state);",
    "  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);",
    "}",
  ];
}

export function emitFrexpHelpers(): string[] {
  return [
    "fn bg_frexp(value: f32, exponent_out: ptr<function, i32>) -> f32 {",
    "  if (value == 0.0 || value != value || abs(value) > 3.4028234663852886e38) {",
    "    *exponent_out = 0;",
    "    return value;",
    "  }",
    "  let exponent = i32(floor(log2(abs(value)))) + 1;",
    "  *exponent_out = exponent;",
    "  return value / exp2(f32(exponent));",
    "}",
  ];
}

export function emitSpecialFloatConstantHelpers(): readonly string[] {
  return [
    "fn bg_f32_inf() -> f32 {",
    "  var bits: u32 = 0x7f800000u;",
    "  return bitcast<f32>(bits);",
    "}",
    "fn bg_f32_nan() -> f32 {",
    "  var bits: u32 = 0x7fc00000u;",
    "  return bitcast<f32>(bits);",
    "}",
  ];
}

export function emitFp8Helpers(): readonly string[] {
  return [
    "fn bg_fp8_inf(sign: f32) -> f32 {",
    "  var bits: u32 = 0x7f800000u;",
    "  return sign * bitcast<f32>(bits);",
    "}",
    "",
    "fn bg_fp8_nan() -> f32 {",
    "  var bits: u32 = 0x7fc00000u;",
    "  return bitcast<f32>(bits);",
    "}",
    "",
    "fn bg_fp8_to_f32(bits_raw: u32, mode: u32) -> f32 {",
    "  let bits = bits_raw & 0xffu;",
    "  let sign = select(1.0, -1.0, (bits & 0x80u) != 0u);",
    "  if (mode == 1u) {",
    "    let exp_bits = (bits >> 2u) & 0x1fu;",
    "    let mant = bits & 0x03u;",
    "    if (exp_bits == 0u && mant == 0u) { return sign * 0.0; }",
    "    if (exp_bits == 0u) { return sign * f32(mant) * exp2(-16.0); }",
    "    if (exp_bits == 0x1fu) { return select(bg_fp8_inf(sign), bg_fp8_nan(), mant != 0u); }",
    "    return sign * (1.0 + f32(mant) / 4.0) * exp2(f32(i32(exp_bits) - 15));",
    "  }",
    "  let exp_bits = (bits >> 3u) & 0x0fu;",
    "  let mant = bits & 0x07u;",
    "  if (exp_bits == 0u && mant == 0u) { return sign * 0.0; }",
    "  if (exp_bits == 0u) { return sign * f32(mant) * exp2(-9.0); }",
    "  if (exp_bits == 0x0fu && mant == 0x07u) { return bg_fp8_nan(); }",
    "  return sign * (1.0 + f32(mant) / 8.0) * exp2(f32(i32(exp_bits) - 7));",
    "}",
    "",
    "fn bg_round_even(x: f32) -> u32 {",
    "  let base = floor(x);",
    "  let diff = x - base;",
    "  if (diff < 0.5) { return u32(base); }",
    "  if (diff > 0.5) { return u32(base + 1.0); }",
    "  let even = (u32(base) & 1u) == 0u;",
    "  return select(u32(base + 1.0), u32(base), even);",
    "}",
    "",
    "fn bg_f32_to_fp8_format(value: f32, saturate: u32, mantissa_bits: u32, bias: i32, max_exponent: u32, max_mantissa: u32, nan_bits: u32, inf_bits: u32) -> u32 {",
    "  if (value != value) { return nan_bits; }",
    "  let sign_bit = select(0u, 0x80u, bitcast<u32>(value) >> 31u != 0u);",
    "  var magnitude = abs(value);",
    "  if (magnitude == 0.0) { return sign_bit; }",
    "  let mantissa_scale = f32(1u << mantissa_bits);",
    "  let max_finite = (1.0 + f32(max_mantissa) / mantissa_scale) * exp2(f32(i32(max_exponent) - bias));",
    "  if (magnitude > max_finite) {",
    "    if (saturate == 1u) { magnitude = max_finite; }",
    "    else { return sign_bit | inf_bits; }",
    "  }",
    "  let raw_exp = i32(floor(log2(magnitude)));",
    "  var exp_bits = raw_exp + bias;",
    "  if (exp_bits <= 0) {",
    "    let mant = min(max_mantissa, bg_round_even(magnitude / exp2(f32(1 - bias)) * mantissa_scale));",
    "    return sign_bit | mant;",
    "  }",
    "  var mant = bg_round_even((magnitude / exp2(f32(raw_exp)) - 1.0) * mantissa_scale);",
    "  if (mant == (1u << mantissa_bits)) {",
    "    exp_bits = exp_bits + 1;",
    "    mant = 0u;",
    "  }",
    "  if (exp_bits > i32(max_exponent) || (exp_bits == i32(max_exponent) && mant > max_mantissa)) {",
    "    if (saturate != 1u) { return sign_bit | inf_bits; }",
    "    exp_bits = i32(max_exponent);",
    "    mant = max_mantissa;",
    "  }",
    "  return sign_bit | (u32(exp_bits) << mantissa_bits) | mant;",
    "}",
    "",
    "fn bg_f32_to_fp8(value: f32, saturate: u32, mode: u32) -> u32 {",
    "  if (mode == 1u) { return bg_f32_to_fp8_format(value, saturate, 2u, 15, 30u, 3u, 0x7fu, 0x7cu); }",
    "  return bg_f32_to_fp8_format(value, saturate, 3u, 7, 15u, 6u, 0x7fu, 0x7fu);",
    "}",
  ];
}
