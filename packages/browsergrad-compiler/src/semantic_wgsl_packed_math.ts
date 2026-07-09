type SemanticPackedLaneWidth = 8 | 16;

export function emitSemanticVSetExpression(lhs: string, rhs: string, laneWidth: SemanticPackedLaneWidth, signed: boolean, operator: string): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const comparisons: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : leftBits;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : rightBits;
    comparisons.push(`(${left} ${operator} ${right})`);
  }
  return `select(0u, 1u, ${comparisons.join(" && ")})`;
}

export function emitSemanticVCompareExpression(lhs: string, rhs: string, laneWidth: SemanticPackedLaneWidth, signed: boolean, operator: string): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : leftBits;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : rightBits;
    lanes.push(`(select(0u, ${mask}, (${left} ${operator} ${right})) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

export function emitSemanticVPackedUnaryExpression(value: string, laneWidth: SemanticPackedLaneWidth, op: "abs" | "sat_abs" | "neg" | "sat_neg"): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const minValue = laneWidth === 8 ? "-128" : "-32768";
  const maxValue = laneWidth === 8 ? "127" : "32767";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const bits = `((${value} >> ${shift}) & ${mask})`;
    const signed = `(i32(${bits}) - select(0, ${signSub}, ${bits} >= ${signBit}))`;
    const result =
      op === "abs" ? `abs(${signed})` :
      op === "sat_abs" ? `min(${maxValue}, abs(${signed}))` :
      op === "neg" ? `(-${signed})` :
      `clamp(-${signed}, ${minValue}, ${maxValue})`;
    lanes.push(`((u32(${result}) & ${mask}) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

export function emitSemanticVPackedAbsDiffExpression(lhs: string, rhs: string, laneWidth: SemanticPackedLaneWidth): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    const left = `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))`;
    const right = `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))`;
    lanes.push(`((u32(abs(${left} - ${right})) & ${mask}) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

export function emitSemanticVPackedSadExpression(lhs: string, rhs: string, laneWidth: SemanticPackedLaneWidth, signed: boolean): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    const left = signed ? `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))` : `i32(${leftBits})`;
    const right = signed ? `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))` : `i32(${rightBits})`;
    lanes.push(`u32(abs(${left} - ${right}))`);
  }
  return `(${lanes.join(" + ")})`;
}

export function emitSemanticVPackedAverageExpression(lhs: string, rhs: string, laneWidth: SemanticPackedLaneWidth, signedRounded: boolean): string {
  const laneCount = 32 / laneWidth;
  const mask = laneWidth === 8 ? "0xffu" : "0xffffu";
  const signBit = laneWidth === 8 ? "0x80u" : "0x8000u";
  const signSub = laneWidth === 8 ? "256" : "65536";
  const lanes: string[] = [];
  for (let lane = 0; lane < laneCount; lane++) {
    const shift = `${lane * laneWidth}u`;
    const leftBits = `((${lhs} >> ${shift}) & ${mask})`;
    const rightBits = `((${rhs} >> ${shift}) & ${mask})`;
    if (signedRounded) {
      const left = `(i32(${leftBits}) - select(0, ${signSub}, ${leftBits} >= ${signBit}))`;
      const right = `(i32(${rightBits}) - select(0, ${signSub}, ${rightBits} >= ${signBit}))`;
      lanes.push(`((u32((${left} + ${right} + 1) >> 1u) & ${mask}) << ${shift})`);
    } else {
      lanes.push(`(((${leftBits} + ${rightBits}) >> 1u) << ${shift})`);
    }
  }
  return `(${lanes.join(" | ")})`;
}

export function emitSemanticViadd16x2Expression(lhs: string, rhs: string, cmpValue: string, signed: boolean, choose: "max" | "min", relu: boolean): string {
  const lanes: string[] = [];
  for (const shift of ["0u", "16u"]) {
    const leftBits = `((${lhs} >> ${shift}) & 0xffffu)`;
    const rightBits = `((${rhs} >> ${shift}) & 0xffffu)`;
    const cmpBits = `((${cmpValue} >> ${shift}) & 0xffffu)`;
    const left = signed ? `(i32(${leftBits}) - select(0, 65536, ${leftBits} >= 0x8000u))` : `i32(${leftBits})`;
    const right = signed ? `(i32(${rightBits}) - select(0, 65536, ${rightBits} >= 0x8000u))` : `i32(${rightBits})`;
    const cmp = signed ? `(i32(${cmpBits}) - select(0, 65536, ${cmpBits} >= 0x8000u))` : `i32(${cmpBits})`;
    const selected = `${choose}((${left} + ${right}), ${cmp})`;
    const value = relu ? `max(${selected}, 0)` : selected;
    lanes.push(`((u32(${value}) & 0xffffu) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

export function emitSemanticViMinMax16x2Expression(inputs: readonly string[], signed: boolean, choose: "max" | "min", relu: boolean): string {
  const lanes: string[] = [];
  for (const shift of ["0u", "16u"]) {
    const values = inputs.map((input) => {
      const bits = `((${input} >> ${shift}) & 0xffffu)`;
      return signed ? `(i32(${bits}) - select(0, 65536, ${bits} >= 0x8000u))` : `i32(${bits})`;
    });
    const selected = values.slice(1).reduce((acc, value) => `${choose}(${acc}, ${value})`, values[0] ?? "0");
    const value = relu ? `max(${selected}, 0)` : selected;
    lanes.push(`((u32(${value}) & 0xffffu) << ${shift})`);
  }
  return `(${lanes.join(" | ")})`;
}

export function emitRoundEvenWgsl(emitted: string): string {
  return `bg_semantic_round_even_f32(${emitted})`;
}

export function halfConversionModeLiteral(callee: string): "0u" | "1u" | "2u" | "3u" {
  if (callee.endsWith("_rn")) return "0u";
  if (callee.endsWith("_rz")) return "1u";
  if (callee.endsWith("_ru")) return "2u";
  return "3u";
}

export function wgslRoundBfloat16(value: string, mode: "0u" | "1u" | "2u" | "3u" = "0u"): string {
  return `bitcast<f32>(bg_f32_to_bf16_bits_mode(f32(${value}), ${mode}) << 16u)`;
}

export function wgslSaturateHalf(value: string): string {
  return `select(clamp(${value}, f16(0.0), f16(1.0)), f16(0.0), (${value}) != (${value}))`;
}

export function wgslSaturateHalf2(value: string): string {
  return `select(clamp(${value}, vec2<f16>(0.0), vec2<f16>(1.0)), vec2<f16>(0.0), (${value}) != (${value}))`;
}

export function wgslSaturateBfloat16(value: string): string {
  return wgslRoundBfloat16(`select(clamp(${value}, 0.0, 1.0), 0.0, (${value}) != (${value}))`);
}

export function wgslReluBfloat16(value: string): string {
  return wgslRoundBfloat16(`select(max(${value}, 0.0), ${value}, (${value}) != (${value}))`);
}

export function wgslSaturateBf162(value: string): string {
  return `select(clamp(${value}, vec2<f32>(0.0), vec2<f32>(1.0)), vec2<f32>(0.0), (${value}) != (${value}))`;
}

export function wgslReluBf162(value: string): string {
  return `select(max(${value}, vec2<f32>(0.0)), ${value}, (${value}) != (${value}))`;
}
