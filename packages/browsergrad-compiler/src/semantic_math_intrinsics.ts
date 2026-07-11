import type { SemanticExpression } from "./semantic_ir.js";
import { isSemanticFloatVectorType, semanticExpressionValueType } from "./semantic_vector_intrinsics.js";
import type { CudaLiteScalarType } from "./types.js";

export const SEMANTIC_FP8_CALLS: ReadonlySet<string> = new Set(["__nv_cvt_fp8_to_halfraw", "__nv_cvt_float_to_fp8"]);

export const SEMANTIC_HALF_CONVERSION_CALLS: ReadonlySet<string> = new Set([
  "__float2half", "__float2half_rn", "__float2half_rz", "__float2half_ru", "__float2half_rd",
  "__int2half_rn", "__int2half_rz", "__int2half_ru", "__int2half_rd",
  "__uint2half_rn", "__uint2half_rz", "__uint2half_ru", "__uint2half_rd",
  "__short2half_rn", "__short2half_rz", "__short2half_ru", "__short2half_rd",
  "__ushort2half_rn", "__ushort2half_rz", "__ushort2half_ru", "__ushort2half_rd",
]);

export const SEMANTIC_BFLOAT_CONVERSION_CALLS: ReadonlySet<string> = new Set([
  "__float2bfloat16", "__float2bfloat16_rn", "__float2bfloat16_rz", "__float2bfloat16_ru", "__float2bfloat16_rd", "__double2bfloat16",
  "__int2bfloat16_rn", "__int2bfloat16_rz", "__int2bfloat16_ru", "__int2bfloat16_rd",
  "__ll2bfloat16_rn", "__ll2bfloat16_rz", "__ll2bfloat16_ru", "__ll2bfloat16_rd",
  "__uint2bfloat16_rn", "__uint2bfloat16_rz", "__uint2bfloat16_ru", "__uint2bfloat16_rd",
  "__ull2bfloat16_rn", "__ull2bfloat16_rz", "__ull2bfloat16_ru", "__ull2bfloat16_rd",
  "__short2bfloat16_rn", "__short2bfloat16_rz", "__short2bfloat16_ru", "__short2bfloat16_rd",
  "__ushort2bfloat16_rn", "__ushort2bfloat16_rz", "__ushort2bfloat16_ru", "__ushort2bfloat16_rd",
  "__bfloat162short_rn", "__bfloat162short_rz", "__bfloat162short_ru", "__bfloat162short_rd",
  "__bfloat162ushort_rn", "__bfloat162ushort_rz", "__bfloat162ushort_ru", "__bfloat162ushort_rd",
  "__bfloat162ll_rn", "__bfloat162ll_rz", "__bfloat162ll_ru", "__bfloat162ll_rd",
  "__bfloat162ull_rn", "__bfloat162ull_rz", "__bfloat162ull_ru", "__bfloat162ull_rd",
  "__bfloat162char_rz", "__bfloat162uchar_rz",
]);

export const SEMANTIC_BFLOAT_HELPER_CALLS: ReadonlySet<string> = new Set([
  ...SEMANTIC_BFLOAT_CONVERSION_CALLS,
  "__habs", "__hceil", "__hfloor", "__hrcp", "__hrsqrt", "hrsqrt", "__hsqrt", "__htrunc", "__hneg", "hexp",
  "__hadd", "__hadd_rn", "__hadd_sat", "__hsub", "__hsub_rn", "__hsub_sat",
  "__hmul", "__hmul_rn", "__hmul_sat", "__hdiv", "__hdiv_rn", "__hfma", "__hfma_rn", "__hfma_sat", "__hfma_relu",
  "__hmin", "__hmax", "__hmin_nan", "__hmax_nan",
  "__halves2bfloat162", "__float22bfloat162_rn", "__float2bfloat162_rn", "__floats2bfloat162_rn",
  "h2ceil", "h2floor", "h2rcp", "h2rsqrt", "h2sqrt", "h2trunc",
  "h2exp", "h2exp2", "h2exp10", "h2log", "h2log2", "h2log10",
  "h2sin", "h2cos", "h2tanh", "h2tanh_approx", "h2rint",
  "__hadd2", "__hadd2_rn", "__hadd2_sat", "__hsub2", "__hsub2_rn", "__hsub2_sat",
  "__hmul2", "__hmul2_rn", "__hmul2_sat", "__h2div", "__hfma2", "__hfma2_rn", "__hfma2_sat", "__hfma2_relu", "__hcmadd",
  "__hmin2", "__hmax2", "__hmin2_nan", "__hmax2_nan",
]);

export const SEMANTIC_MATH_CALLS: ReadonlyMap<string, string> = new Map([
  ["clock", "clock"],
  ["clock64", "clock"],
  ["sqrt", "sqrt"],
  ["sqrtf", "sqrt"],
  ["__fsqrt_rn", "sqrt"],
  ["rsqrt", "inverseSqrt"],
  ["rsqrtf", "inverseSqrt"],
  ["__frsqrt_rn", "inverseSqrt"],
  ["exp", "exp"],
  ["expf", "exp"],
  ["__expf", "exp"],
  ["exp2", "exp2"],
  ["exp2f", "exp2"],
  ["__exp2f", "exp2"],
  ["exp10", "exp10"],
  ["exp10f", "exp10"],
  ["__exp10f", "exp10"],
  ["expm1", "expm1"],
  ["expm1f", "expm1"],
  ["erf", "erf"],
  ["erff", "erf"],
  ["erfc", "erfc"],
  ["erfcf", "erfc"],
  ["erfcx", "erfcx"],
  ["erfcxf", "erfcx"],
  ["erfinv", "erfinv"],
  ["erfinvf", "erfinv"],
  ["erfcinv", "erfcinv"],
  ["erfcinvf", "erfcinv"],
  ["normcdf", "normcdf"],
  ["normcdff", "normcdf"],
  ["normcdfinv", "normcdfinv"],
  ["normcdfinvf", "normcdfinv"],
  ["tgamma", "tgamma"],
  ["tgammaf", "tgamma"],
  ["lgamma", "lgamma"],
  ["lgammaf", "lgamma"],
  ["log", "log"],
  ["logf", "log"],
  ["__logf", "log"],
  ["log2", "log2"],
  ["log2f", "log2"],
  ["__log2f", "log2"],
  ["log10", "log10"],
  ["log10f", "log10"],
  ["__log10f", "log10"],
  ["log1p", "log1p"],
  ["log1pf", "log1p"],
  ["fabs", "abs"],
  ["fabsf", "abs"],
  ["abs", "abs"],
  ["floor", "floor"],
  ["floorf", "floor"],
  ["ceil", "ceil"],
  ["ceilf", "ceil"],
  ["trunc", "trunc"],
  ["truncf", "trunc"],
  ["round", "round_away"],
  ["roundf", "round_away"],
  ["rint", "round_even"],
  ["rintf", "round_even"],
  ["nearbyint", "round_even"],
  ["nearbyintf", "round_even"],
  ["sin", "sin"],
  ["sinf", "sin"],
  ["__sinf", "sin"],
  ["sinpi", "sinpi"],
  ["sinpif", "sinpi"],
  ["cos", "cos"],
  ["cosf", "cos"],
  ["__cosf", "cos"],
  ["cospi", "cospi"],
  ["cospif", "cospi"],
  ["tan", "tan"],
  ["tanf", "tan"],
  ["__tanf", "tan"],
  ["asin", "asin"],
  ["asinf", "asin"],
  ["acos", "acos"],
  ["acosf", "acos"],
  ["atan", "atan"],
  ["atanf", "atan"],
  ["atan2", "atan2"],
  ["atan2f", "atan2"],
  ["sinh", "sinh"],
  ["sinhf", "sinh"],
  ["cosh", "cosh"],
  ["coshf", "cosh"],
  ["tanh", "tanh"],
  ["tanhf", "tanh"],
  ["__tanhf", "tanh"],
  ["asinh", "asinh"],
  ["asinhf", "asinh"],
  ["acosh", "acosh"],
  ["acoshf", "acosh"],
  ["atanh", "atanh"],
  ["atanhf", "atanh"],
  ["cbrt", "cbrt"],
  ["cbrtf", "cbrt"],
  ["rcbrt", "rcbrt"],
  ["rcbrtf", "rcbrt"],
  ["ldexp", "ldexp"],
  ["ldexpf", "ldexp"],
  ["scalbn", "ldexp"],
  ["scalbnf", "ldexp"],
  ["scalbln", "ldexp"],
  ["scalblnf", "ldexp"],
  ["fmod", "fmod"],
  ["fmodf", "fmod"],
  ["remainder", "remainder"],
  ["remainderf", "remainder"],
  ["logb", "logb"],
  ["logbf", "logb"],
  ["ilogb", "ilogb"],
  ["ilogbf", "ilogb"],
  ["fdim", "fdim"],
  ["fdimf", "fdim"],
  ["nextafter", "nextafter"],
  ["nextafterf", "nextafter"],
  ["nexttoward", "nextafter"],
  ["nexttowardf", "nextafter"],
  ["hypot", "hypot"],
  ["hypotf", "hypot"],
  ["rhypot", "rhypot"],
  ["rhypotf", "rhypot"],
  ["norm3df", "norm"],
  ["norm4df", "norm"],
  ["rnorm3df", "rnorm"],
  ["rnorm4df", "rnorm"],
  ["lrint", "float_to_int_rn"],
  ["lrintf", "float_to_int_rn"],
  ["llrint", "float_to_int_rn"],
  ["llrintf", "float_to_int_rn"],
  ["lround", "float_to_int_round"],
  ["lroundf", "float_to_int_round"],
  ["llround", "float_to_int_round"],
  ["llroundf", "float_to_int_round"],
  ["__float2int_rn", "float_to_int_rn"],
  ["__float2int_rz", "float_to_int_rz"],
  ["__float2int_ru", "float_to_int_ru"],
  ["__float2int_rd", "float_to_int_rd"],
  ["__float2uint_rn", "float_to_uint_rn"],
  ["__float2uint_rz", "float_to_uint_rz"],
  ["__float2uint_ru", "float_to_uint_ru"],
  ["__float2uint_rd", "float_to_uint_rd"],
  ["__int2float_rn", "int_to_float"],
  ["__int2float_rz", "int_to_float"],
  ["__int2float_ru", "int_to_float"],
  ["__int2float_rd", "int_to_float"],
  ["__uint2float_rn", "uint_to_float"],
  ["__uint2float_rz", "uint_to_float"],
  ["__uint2float_ru", "uint_to_float"],
  ["__uint2float_rd", "uint_to_float"],
  ["wmma::__float_to_tf32", "tf32"],
  ["__half2float", "half_to_float"],
  ["__float2half", "to_half"],
  ["__float2half_rn", "to_half"],
  ["__float2half_rz", "float_to_half_rz"],
  ["__float2half_ru", "float_to_half_ru"],
  ["__float2half_rd", "float_to_half_rd"],
  ["__int2half_rn", "int_to_half"],
  ["__int2half_rz", "int_to_half_rz"],
  ["__int2half_ru", "int_to_half_ru"],
  ["__int2half_rd", "int_to_half_rd"],
  ["__uint2half_rn", "uint_to_half"],
  ["__uint2half_rz", "uint_to_half_rz"],
  ["__uint2half_ru", "uint_to_half_ru"],
  ["__uint2half_rd", "uint_to_half_rd"],
  ["__short2half_rn", "short_to_half_rn"],
  ["__short2half_rz", "short_to_half_rz"],
  ["__short2half_ru", "short_to_half_ru"],
  ["__short2half_rd", "short_to_half_rd"],
  ["__ushort2half_rn", "ushort_to_half_rn"],
  ["__ushort2half_rz", "ushort_to_half_rz"],
  ["__ushort2half_ru", "ushort_to_half_ru"],
  ["__ushort2half_rd", "ushort_to_half_rd"],
  ["__half_as_short", "half_as_short"],
  ["__half_as_ushort", "half_as_ushort"],
  ["__short_as_half", "short_as_half"],
  ["__ushort_as_half", "ushort_as_half"],
  ["__half2int_rn", "half_to_int_rn"],
  ["__half2int_rz", "half_to_int_rz"],
  ["__half2int_ru", "half_to_int_ru"],
  ["__half2int_rd", "half_to_int_rd"],
  ["__half2short_rn", "half_to_short_rn"],
  ["__half2short_rz", "half_to_short_rz"],
  ["__half2short_ru", "half_to_short_ru"],
  ["__half2short_rd", "half_to_short_rd"],
  ["__half2uint_rn", "half_to_uint_rn"],
  ["__half2uint_rz", "half_to_uint_rz"],
  ["__half2uint_ru", "half_to_uint_ru"],
  ["__half2uint_rd", "half_to_uint_rd"],
  ["__half2ushort_rn", "half_to_ushort_rn"],
  ["__half2ushort_rz", "half_to_ushort_rz"],
  ["__half2ushort_ru", "half_to_ushort_ru"],
  ["__half2ushort_rd", "half_to_ushort_rd"],
  ["__nv_cvt_fp8_to_halfraw", "fp8_to_half"],
  ["__nv_cvt_float_to_fp8", "float_to_fp8"],
  ["__habs", "half_abs"],
  ["__hceil", "half_ceil"],
  ["__hfloor", "half_floor"],
  ["__hrcp", "half_rcp"],
  ["__hrsqrt", "half_rsqrt"],
  ["hrsqrt", "half_rsqrt"],
  ["__hsqrt", "half_sqrt"],
  ["__htrunc", "half_trunc"],
  ["__hneg", "half_neg"],
  ["__hadd_rn", "half_add"],
  ["__hadd_sat", "half_add_sat"],
  ["__hsub", "half_sub"],
  ["__hsub_rn", "half_sub"],
  ["__hsub_sat", "half_sub_sat"],
  ["__hmul", "half_mul"],
  ["__hmul_rn", "half_mul"],
  ["__hmul_sat", "half_mul_sat"],
  ["__hdiv", "half_div"],
  ["__hdiv_rn", "half_div"],
  ["__hfma", "half_fma"],
  ["__hfma_rn", "half_fma"],
  ["__hfma_sat", "half_fma_sat"],
  ["__hfma_relu", "half_fma_relu"],
  ["hexp", "half_exp"],
  ["__hmin", "half_min"],
  ["__hmax", "half_max"],
  ["__hmin_nan", "half_min_nan"],
  ["__hmax_nan", "half_max_nan"],
  ["__hisnan", "half_isnan"],
  ["__hisinf", "half_isinf"],
  ["__heq", "half_eq"],
  ["__hne", "half_ne"],
  ["__hgt", "half_gt"],
  ["__hge", "half_ge"],
  ["__hlt", "half_lt"],
  ["__hle", "half_le"],
  ["__hequ", "half_equ"],
  ["__hneu", "half_neu"],
  ["__hgtu", "half_gtu"],
  ["__hgeu", "half_geu"],
  ["__hltu", "half_ltu"],
  ["__hleu", "half_leu"],
  ["__bfloat162float", "bf16_to_float"],
  ["__float2bfloat16", "to_bf16"],
  ["__float2bfloat16_rn", "to_bf16"],
  ["__float2bfloat16_rz", "float_to_bf16_rz"],
  ["__float2bfloat16_ru", "float_to_bf16_ru"],
  ["__float2bfloat16_rd", "float_to_bf16_rd"],
  ["__double2bfloat16", "double_to_bf16"],
  ["__int2bfloat16_rn", "int_to_bf16"],
  ["__int2bfloat16_rz", "int_to_bf16_rz"],
  ["__int2bfloat16_ru", "int_to_bf16_ru"],
  ["__int2bfloat16_rd", "int_to_bf16_rd"],
  ["__ll2bfloat16_rn", "int_to_bf16"],
  ["__ll2bfloat16_rz", "int_to_bf16_rz"],
  ["__ll2bfloat16_ru", "int_to_bf16_ru"],
  ["__ll2bfloat16_rd", "int_to_bf16_rd"],
  ["__uint2bfloat16_rn", "uint_to_bf16"],
  ["__uint2bfloat16_rz", "uint_to_bf16_rz"],
  ["__uint2bfloat16_ru", "uint_to_bf16_ru"],
  ["__uint2bfloat16_rd", "uint_to_bf16_rd"],
  ["__ull2bfloat16_rn", "uint_to_bf16"],
  ["__ull2bfloat16_rz", "uint_to_bf16_rz"],
  ["__ull2bfloat16_ru", "uint_to_bf16_ru"],
  ["__ull2bfloat16_rd", "uint_to_bf16_rd"],
  ["__short2bfloat16_rn", "short_to_bf16_rn"],
  ["__short2bfloat16_rz", "short_to_bf16_rz"],
  ["__short2bfloat16_ru", "short_to_bf16_ru"],
  ["__short2bfloat16_rd", "short_to_bf16_rd"],
  ["__ushort2bfloat16_rn", "ushort_to_bf16_rn"],
  ["__ushort2bfloat16_rz", "ushort_to_bf16_rz"],
  ["__ushort2bfloat16_ru", "ushort_to_bf16_ru"],
  ["__ushort2bfloat16_rd", "ushort_to_bf16_rd"],
  ["__bfloat16_as_short", "bf16_as_short"],
  ["__bfloat16_as_ushort", "bf16_as_ushort"],
  ["__nv_bfloat16_as_ushort", "bf16_as_ushort"],
  ["__short_as_bfloat16", "short_as_bf16"],
  ["__ushort_as_bfloat16", "ushort_as_bf16"],
  ["__bfloat162int_rn", "bf16_to_int_rn"],
  ["__bfloat162int_rz", "bf16_to_int_rz"],
  ["__bfloat162int_ru", "bf16_to_int_ru"],
  ["__bfloat162int_rd", "bf16_to_int_rd"],
  ["__bfloat162ll_rn", "bf16_to_int_rn"],
  ["__bfloat162ll_rz", "bf16_to_int_rz"],
  ["__bfloat162ll_ru", "bf16_to_int_ru"],
  ["__bfloat162ll_rd", "bf16_to_int_rd"],
  ["__bfloat162uint_rn", "bf16_to_uint_rn"],
  ["__bfloat162uint_rz", "bf16_to_uint_rz"],
  ["__bfloat162uint_ru", "bf16_to_uint_ru"],
  ["__bfloat162uint_rd", "bf16_to_uint_rd"],
  ["__bfloat162ull_rn", "bf16_to_uint_rn"],
  ["__bfloat162ull_rz", "bf16_to_uint_rz"],
  ["__bfloat162ull_ru", "bf16_to_uint_ru"],
  ["__bfloat162ull_rd", "bf16_to_uint_rd"],
  ["__bfloat162short_rn", "bf16_to_short_rn"],
  ["__bfloat162short_rz", "bf16_to_short_rz"],
  ["__bfloat162short_ru", "bf16_to_short_ru"],
  ["__bfloat162short_rd", "bf16_to_short_rd"],
  ["__bfloat162ushort_rn", "bf16_to_ushort_rn"],
  ["__bfloat162ushort_rz", "bf16_to_ushort_rz"],
  ["__bfloat162ushort_ru", "bf16_to_ushort_ru"],
  ["__bfloat162ushort_rd", "bf16_to_ushort_rd"],
  ["__bfloat162char_rz", "bf16_to_char_rz"],
  ["__bfloat162uchar_rz", "bf16_to_uchar_rz"],
  ["__clz", "clz"],
  ["__clzll", "clzll"],
  ["__ffs", "ffs"],
  ["__ffsll", "ffs"],
  ["__popc", "popc"],
  ["__popcll", "popc"],
  ["__brev", "brev"],
  ["__brevll", "brev"],
  ["__mul24", "mul24"],
  ["__umul24", "umul24"],
  ["__mulhi", "mulhi"],
  ["__umulhi", "umulhi"],
  ["__mul64hi", "mulhi"],
  ["__umul64hi", "umulhi"],
  ["__byte_perm", "byte_perm"],
  ["__funnelshift_l", "funnelshift_l"],
  ["__funnelshift_lc", "funnelshift_lc"],
  ["__funnelshift_r", "funnelshift_r"],
  ["__funnelshift_rc", "funnelshift_rc"],
  ["__rhadd", "rhadd"],
  ["__uhadd", "uhadd"],
  ["__urhadd", "urhadd"],
  ["__hadd", "hadd"],
  ["__float_as_int", "float_as_int"],
  ["__float_as_uint", "float_as_uint"],
  ["__sad", "sad"],
  ["__usad", "usad"],
  ["__usad4", "usad4"],
  ["__viaddmax_s32", "viaddmax_s32"],
  ["__viaddmax_s32_relu", "viaddmax_s32_relu"],
  ["__viaddmin_s32", "viaddmin_s32"],
  ["__viaddmin_s32_relu", "viaddmin_s32_relu"],
  ["__viaddmax_u32", "viaddmax_u32"],
  ["__viaddmin_u32", "viaddmin_u32"],
  ["__viaddmax_s16x2", "viaddmax_s16x2"],
  ["__viaddmax_s16x2_relu", "viaddmax_s16x2_relu"],
  ["__viaddmin_s16x2", "viaddmin_s16x2"],
  ["__viaddmin_s16x2_relu", "viaddmin_s16x2_relu"],
  ["__viaddmax_u16x2", "viaddmax_u16x2"],
  ["__viaddmin_u16x2", "viaddmin_u16x2"],
  ["__vimax_s32_relu", "vimax_s32_relu"],
  ["__vimin_s32_relu", "vimin_s32_relu"],
  ["__vimax_s16x2_relu", "vimax_s16x2_relu"],
  ["__vimin_s16x2_relu", "vimin_s16x2_relu"],
  ["__vimax3_s32", "vimax3_s32"],
  ["__vimax3_s32_relu", "vimax3_s32_relu"],
  ["__vimin3_s32", "vimin3_s32"],
  ["__vimin3_s32_relu", "vimin3_s32_relu"],
  ["__vimax3_u32", "vimax3_u32"],
  ["__vimin3_u32", "vimin3_u32"],
  ["__vimax3_s16x2", "vimax3_s16x2"],
  ["__vimax3_s16x2_relu", "vimax3_s16x2_relu"],
  ["__vimin3_s16x2", "vimin3_s16x2"],
  ["__vimin3_s16x2_relu", "vimin3_s16x2_relu"],
  ["__vimax3_u16x2", "vimax3_u16x2"],
  ["__vimin3_u16x2", "vimin3_u16x2"],
  ["__vibmax_s32", "vibmax_s32"],
  ["__vibmin_s32", "vibmin_s32"],
  ["__vibmax_u32", "vibmax_u32"],
  ["__vibmin_u32", "vibmin_u32"],
  ["__vibmax_s16x2", "vibmax_s16x2"],
  ["__vibmin_s16x2", "vibmin_s16x2"],
  ["__vibmax_u16x2", "vibmax_u16x2"],
  ["__vibmin_u16x2", "vibmin_u16x2"],
  ["__vadd2", "vadd2"],
  ["__vsub2", "vsub2"],
  ["__vabs2", "vabs2"],
  ["__vabsss2", "vabsss2"],
  ["__vneg2", "vneg2"],
  ["__vnegss2", "vnegss2"],
  ["__vaddss2", "vaddss2"],
  ["__vsubss2", "vsubss2"],
  ["__vaddus2", "vaddus2"],
  ["__vsubus2", "vsubus2"],
  ["__vabsdiffu2", "vabsdiffu2"],
  ["__vabsdiffs2", "vabsdiffs2"],
  ["__vsads2", "vsads2"],
  ["__vsadu2", "vsadu2"],
  ["__vhaddu2", "vhaddu2"],
  ["__vavgs2", "vavgs2"],
  ["__vavgu2", "vavgu2"],
  ["__vminu2", "vminu2"],
  ["__vmaxu2", "vmaxu2"],
  ["__vmins2", "vmins2"],
  ["__vmaxs2", "vmaxs2"],
  ["__vcmpeq2", "vcmpeq2"],
  ["__vcmpne2", "vcmpne2"],
  ["__vcmpges2", "vcmpges2"],
  ["__vcmpgeu2", "vcmpgeu2"],
  ["__vcmpgts2", "vcmpgts2"],
  ["__vcmpgtu2", "vcmpgtu2"],
  ["__vcmples2", "vcmples2"],
  ["__vcmpleu2", "vcmpleu2"],
  ["__vcmplts2", "vcmplts2"],
  ["__vcmpltu2", "vcmpltu2"],
  ["__vseteq2", "vseteq2"],
  ["__vsetne2", "vsetne2"],
  ["__vsetges2", "vsetges2"],
  ["__vsetgeu2", "vsetgeu2"],
  ["__vsetgts2", "vsetgts2"],
  ["__vsetgtu2", "vsetgtu2"],
  ["__vsetles2", "vsetles2"],
  ["__vsetleu2", "vsetleu2"],
  ["__vsetlts2", "vsetlts2"],
  ["__vsetltu2", "vsetltu2"],
  ["__vadd4", "vadd4"],
  ["__vsub4", "vsub4"],
  ["__vabs4", "vabs4"],
  ["__vabsss4", "vabsss4"],
  ["__vneg4", "vneg4"],
  ["__vnegss4", "vnegss4"],
  ["__vaddss4", "vaddss4"],
  ["__vsubss4", "vsubss4"],
  ["__vaddus4", "vaddus4"],
  ["__vsubus4", "vsubus4"],
  ["__vabsdiffu4", "vabsdiffu4"],
  ["__vabsdiffs4", "vabsdiffs4"],
  ["__vsads4", "vsads4"],
  ["__vsadu4", "vsadu4"],
  ["__vhaddu4", "vhaddu4"],
  ["__vavgs4", "vavgs4"],
  ["__vavgu4", "vavgu4"],
  ["__vminu4", "vminu4"],
  ["__vmaxu4", "vmaxu4"],
  ["__vmins4", "vmins4"],
  ["__vmaxs4", "vmaxs4"],
  ["__vcmpeq4", "vcmpeq4"],
  ["__vcmpne4", "vcmpne4"],
  ["__vcmpges4", "vcmpges4"],
  ["__vcmpgeu4", "vcmpgeu4"],
  ["__vcmpgts4", "vcmpgts4"],
  ["__vcmpgtu4", "vcmpgtu4"],
  ["__vcmples4", "vcmples4"],
  ["__vcmpleu4", "vcmpleu4"],
  ["__vcmplts4", "vcmplts4"],
  ["__vcmpltu4", "vcmpltu4"],
  ["__vseteq4", "vseteq4"],
  ["__vsetne4", "vsetne4"],
  ["__vsetges4", "vsetges4"],
  ["__vsetgeu4", "vsetgeu4"],
  ["__vsetgts4", "vsetgts4"],
  ["__vsetgtu4", "vsetgtu4"],
  ["__vsetles4", "vsetles4"],
  ["__vsetleu4", "vsetleu4"],
  ["__vsetlts4", "vsetlts4"],
  ["__vsetltu4", "vsetltu4"],
  ["__dp4a", "dp4a"],
  ["__dp2a_lo", "dp2a_lo"],
  ["__dp2a_hi", "dp2a_hi"],
  ["IMAD", "imad"],
  ["UMUL", "umul"],
  ["UMAD", "umad"],
  ["umin", "umin"],
  ["assert", "assert"],
  ["fmin", "min"],
  ["fminf", "min"],
  ["min", "min"],
  ["fmax", "max"],
  ["fmaxf", "max"],
  ["max", "max"],
  ["pow", "pow"],
  ["powf", "pow"],
  ["__powf", "pow"],
  ["__fdividef", "divide"],
  ["fdividef", "divide"],
  ["__fadd_rn", "add"],
  ["__fsub_rn", "sub"],
  ["__fmul_rn", "mul"],
  ["__fdiv_rn", "divide"],
  ["__frcp_rn", "reciprocal"],
  ["__builtin_inff", "builtin_inf"],
  ["__builtin_huge_valf", "builtin_inf"],
  ["__uint_as_float", "uint_as_float"],
  ["__int_as_float", "int_as_float"],
  ["__saturatef", "saturate"],
  ["copysign", "copysign"],
  ["copysignf", "copysign"],
  ["isinf", "isinf"],
  ["isinff", "isinf"],
  ["__isinff", "isinf"],
  ["isfinite", "isfinite"],
  ["isfinitef", "isfinite"],
  ["finite", "isfinite"],
  ["finitef", "isfinite"],
  ["__finitef", "isfinite"],
  ["isnan", "isnan"],
  ["isnanf", "isnan"],
  ["__isnanf", "isnan"],
  ["isNan", "isnan"],
  ["signbit", "signbit"],
  ["signbitf", "signbit"],
  ["isnormal", "isnormal"],
  ["isgreater", "isgreater"],
  ["isgreaterequal", "isgreaterequal"],
  ["isless", "isless"],
  ["islessequal", "islessequal"],
  ["islessgreater", "islessgreater"],
  ["isunordered", "isunordered"],
  ["fma", "fma"],
  ["fmaf", "fma"],
  ["__fmaf_rn", "fma"],
  ["lerp", "lerp"],
  ["div_ceil", "div_ceil"],
  ["ceil_div", "div_ceil"],
  ["__bg_modf_intpart", "modf_intpart"],
  ["__bg_modf_fraction", "modf_fraction"],
  ["__bg_frexp_exponent", "frexp_exponent"],
  ["__bg_frexp_mantissa", "frexp_mantissa"],
  ["__bg_remquo_quotient", "remquo_quotient"],
  ["__bg_remquo_remainder", "remquo_remainder"],
  ["__bg_i16_lane", "i16_lane"],
  ["__bg_u16_lane", "u16_lane"],
]);

export function isSemanticMathCallName(name: string): boolean {
  return SEMANTIC_MATH_CALLS.has(name);
}

const SEMANTIC_VECTOR_MIN_MAX_CALLS: ReadonlySet<string> = new Set([
  "fmin", "fminf", "min", "fmax", "fmaxf", "max",
]);

/** Returns the vector overload selected by a CUDA min/max call, if any. */
export function semanticVectorMinMaxCallValueType(
  name: string | undefined,
  args: readonly SemanticExpression[],
): CudaLiteScalarType | undefined {
  if (name === undefined || !SEMANTIC_VECTOR_MIN_MAX_CALLS.has(name) || args.length !== 2) return undefined;
  const left = semanticExpressionValueType(args[0]!);
  const right = semanticExpressionValueType(args[1]!);
  if (isSemanticFloatVectorType(left) && (right === undefined || !isSemanticFloatVectorType(right) || right === left)) return left;
  if (isSemanticFloatVectorType(right) && (left === undefined || !isSemanticFloatVectorType(left) || left === right)) return right;
  return undefined;
}

export function semanticMathCallArgumentsSupported(
  name: string | undefined,
  args: readonly SemanticExpression[],
  expressionSupported: (expression: SemanticExpression) => boolean,
): boolean {
  return name !== undefined &&
    isSemanticMathCallName(name) &&
    (name === "__usad4" ? args.length === 2 || args.length === 3 : args.length === semanticMathCallArity(name)) &&
    args.every(expressionSupported);
}

export function semanticMathCallArity(name: string): number {
  return name === "clock" ||
    name === "clock64" ||
    name === "__builtin_inff" ||
    name === "__builtin_huge_valf"
    ? 0
    : name === "fmin" ||
    name === "fminf" ||
    name === "min" ||
    name === "fmax" ||
    name === "fmaxf" ||
    name === "max" ||
    name === "pow" ||
    name === "powf" ||
    name === "__powf" ||
    name === "__fdividef" ||
    name === "fdividef" ||
    name === "__fadd_rn" ||
    name === "__fsub_rn" ||
    name === "__fmul_rn" ||
    name === "__fdiv_rn" ||
    name === "__hadd_rn" ||
    name === "__hadd_sat" ||
    name === "__hsub" ||
    name === "__hsub_rn" ||
    name === "__hsub_sat" ||
    name === "__hmul" ||
    name === "__hmul_rn" ||
    name === "__hmul_sat" ||
    name === "__hdiv" ||
    name === "__hdiv_rn" ||
    name === "__hmin" ||
    name === "__hmax" ||
    name === "__hmin_nan" ||
    name === "__hmax_nan" ||
    name === "__heq" ||
    name === "__hne" ||
    name === "__hgt" ||
    name === "__hge" ||
    name === "__hlt" ||
    name === "__hle" ||
    name === "__hequ" ||
    name === "__hneu" ||
    name === "__hgtu" ||
    name === "__hgeu" ||
    name === "__hltu" ||
    name === "__hleu" ||
    name === "__nv_cvt_fp8_to_halfraw" ||
    name === "copysign" ||
    name === "copysignf" ||
    name === "isgreater" ||
    name === "isgreaterequal" ||
    name === "isless" ||
    name === "islessequal" ||
    name === "islessgreater" ||
    name === "isunordered" ||
    name === "div_ceil" ||
    name === "ceil_div" ||
    name === "ldexp" ||
    name === "ldexpf" ||
    name === "scalbn" ||
    name === "scalbnf" ||
    name === "scalbln" ||
    name === "scalblnf" ||
    name === "fmod" ||
    name === "fmodf" ||
    name === "remainder" ||
    name === "remainderf" ||
    name === "fdim" ||
    name === "fdimf" ||
    name === "nextafter" ||
    name === "nextafterf" ||
    name === "nexttoward" ||
    name === "nexttowardf" ||
    name === "hypot" ||
    name === "hypotf" ||
    name === "rhypot" ||
    name === "rhypotf" ||
    name === "__bg_remquo_quotient" ||
    name === "__bg_remquo_remainder" ||
    name === "atan2" ||
    name === "atan2f" ||
    name === "__mul24" ||
    name === "__umul24" ||
    name === "__mulhi" ||
    name === "__umulhi" ||
    name === "__mul64hi" ||
    name === "__umul64hi" ||
    name === "__rhadd" ||
    name === "__uhadd" ||
    name === "__urhadd" ||
    name === "__hadd" ||
    name === "__vadd2" ||
    name === "__vsub2" ||
    name === "__vaddss2" ||
    name === "__vsubss2" ||
    name === "__vaddus2" ||
    name === "__vsubus2" ||
    name === "__vabsdiffu2" ||
    name === "__vabsdiffs2" ||
    name === "__vsads2" ||
    name === "__vsadu2" ||
    name === "__vhaddu2" ||
    name === "__vavgs2" ||
    name === "__vavgu2" ||
    name === "__vminu2" ||
    name === "__vmaxu2" ||
    name === "__vmins2" ||
    name === "__vmaxs2" ||
    name === "__vcmpeq2" ||
    name === "__vcmpne2" ||
    name === "__vcmpges2" ||
    name === "__vcmpgeu2" ||
    name === "__vcmpgts2" ||
    name === "__vcmpgtu2" ||
    name === "__vcmples2" ||
    name === "__vcmpleu2" ||
    name === "__vcmplts2" ||
    name === "__vcmpltu2" ||
    name === "__vseteq2" ||
    name === "__vsetne2" ||
    name === "__vsetges2" ||
    name === "__vsetgeu2" ||
    name === "__vsetgts2" ||
    name === "__vsetgtu2" ||
    name === "__vsetles2" ||
    name === "__vsetleu2" ||
    name === "__vsetlts2" ||
    name === "__vsetltu2" ||
    name === "__vadd4" ||
    name === "__vsub4" ||
    name === "__vaddss4" ||
    name === "__vsubss4" ||
    name === "__vaddus4" ||
    name === "__vsubus4" ||
    name === "__vabsdiffu4" ||
    name === "__vabsdiffs4" ||
    name === "__vsads4" ||
    name === "__vsadu4" ||
    name === "__vhaddu4" ||
    name === "__vavgs4" ||
    name === "__vavgu4" ||
    name === "__vminu4" ||
    name === "__vmaxu4" ||
    name === "__vmins4" ||
    name === "__vmaxs4" ||
    name === "__vcmpeq4" ||
    name === "__vcmpne4" ||
    name === "__vcmpges4" ||
    name === "__vcmpgeu4" ||
    name === "__vcmpgts4" ||
    name === "__vcmpgtu4" ||
    name === "__vcmples4" ||
    name === "__vcmpleu4" ||
    name === "__vcmplts4" ||
    name === "__vcmpltu4" ||
    name === "__vseteq4" ||
    name === "__vsetne4" ||
    name === "__vsetges4" ||
    name === "__vsetgeu4" ||
    name === "__vsetgts4" ||
    name === "__vsetgtu4" ||
    name === "__vsetles4" ||
    name === "__vsetleu4" ||
    name === "__vsetlts4" ||
    name === "__vsetltu4" ||
    name === "__vimax_s32_relu" ||
    name === "__vimin_s32_relu" ||
    name === "__vimax_s16x2_relu" ||
    name === "__vimin_s16x2_relu" ||
    name === "__vibmax_s32" ||
    name === "__vibmin_s32" ||
    name === "__vibmax_u32" ||
    name === "__vibmin_u32" ||
    name === "__vibmax_s16x2" ||
    name === "__vibmin_s16x2" ||
    name === "__vibmax_u16x2" ||
    name === "__vibmin_u16x2" ||
    name === "__bg_i16_lane" ||
    name === "__bg_u16_lane" ||
    name === "UMUL" ||
    name === "umin"
    ? 2
    : name === "__hisnan" ||
    name === "__hisinf"
    ? 1
    : name === "fma" ||
      name === "fmaf" ||
      name === "__fmaf_rn" ||
      name === "__hfma" ||
      name === "__hfma_rn" ||
      name === "__hfma_sat" ||
      name === "__hfma_relu" ||
      name === "lerp" ||
      name === "norm3df" ||
      name === "rnorm3df" ||
      name === "__byte_perm" ||
      name === "__funnelshift_l" ||
      name === "__funnelshift_lc" ||
      name === "__funnelshift_r" ||
      name === "__funnelshift_rc" ||
      name === "__sad" ||
      name === "__usad" ||
      name === "__usad4" ||
      name === "__viaddmax_s32" ||
      name === "__viaddmax_s32_relu" ||
      name === "__viaddmin_s32" ||
      name === "__viaddmin_s32_relu" ||
      name === "__viaddmax_u32" ||
      name === "__viaddmin_u32" ||
      name === "__viaddmax_s16x2" ||
      name === "__viaddmax_s16x2_relu" ||
      name === "__viaddmin_s16x2" ||
      name === "__viaddmin_s16x2_relu" ||
      name === "__viaddmax_u16x2" ||
      name === "__viaddmin_u16x2" ||
      name === "__vimax3_s32" ||
      name === "__vimax3_s32_relu" ||
      name === "__vimin3_s32" ||
      name === "__vimin3_s32_relu" ||
      name === "__vimax3_u32" ||
      name === "__vimin3_u32" ||
      name === "__vimax3_s16x2" ||
      name === "__vimax3_s16x2_relu" ||
      name === "__vimin3_s16x2" ||
      name === "__vimin3_s16x2_relu" ||
      name === "__vimax3_u16x2" ||
      name === "__vimin3_u16x2" ||
      name === "__dp4a" ||
      name === "__dp2a_lo" ||
      name === "__dp2a_hi" ||
      name === "__nv_cvt_float_to_fp8" ||
      name === "IMAD" ||
      name === "UMAD"
    ? 3
    : name === "norm4df" ||
      name === "rnorm4df"
    ? 4
    : 1;
}
