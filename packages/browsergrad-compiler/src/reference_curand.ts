export function referenceCurandNext(state: number): number {
  return (Math.imul(state >>> 0, 1664525) + 1013904223) >>> 0;
}

export function referenceCurandAdvance(state: number, count: number): number {
  let accMult = 1;
  let accPlus = 0;
  let curMult = 1664525;
  let curPlus = 1013904223;
  let delta = count >>> 0;
  while (delta > 0) {
    if ((delta & 1) !== 0) {
      accMult = Math.imul(accMult, curMult) >>> 0;
      accPlus = (Math.imul(accPlus, curMult) + curPlus) >>> 0;
    }
    curPlus = Math.imul((curMult + 1) >>> 0, curPlus) >>> 0;
    curMult = Math.imul(curMult, curMult) >>> 0;
    delta >>>= 1;
  }
  return (Math.imul(accMult, state >>> 0) + accPlus) >>> 0;
}

export function referenceCurandNormalPair(first: number, second: number): [number, number] {
  const u1 = Math.max((first + 1) * 2.3283064365386963e-10, 1.1754943508222875e-38);
  const u2 = (second + 1) * 2.3283064365386963e-10;
  const radius = Math.sqrt(-2 * Math.log(u1));
  const angle = 6.283185307179586 * u2;
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
}

export function referenceCurandPoissonDraw(state: number, lambda: number): [number, number] {
  if (lambda <= 0) return [0, state >>> 0];
  if (lambda < 64) {
    const limit = Math.fround(Math.exp(Math.fround(-lambda)));
    let product = Math.fround(1);
    let count = 0;
    let current = state >>> 0;
    while (count < 512 && product > limit) {
      current = referenceCurandNext(current);
      product = Math.fround(product * Math.fround(Math.fround(current + 1) * Math.fround(2.3283064365386963e-10)));
      count++;
    }
    return [Math.max(0, count - 1) >>> 0, current];
  }
  const first = referenceCurandNext(state);
  const second = referenceCurandNext(first);
  const u1 = Math.fround(Math.max(Math.fround(Math.fround(first + 1) * Math.fround(2.3283064365386963e-10)), Math.fround(1.1754943508222875e-38)));
  const u2 = Math.fround(Math.fround(second + 1) * Math.fround(2.3283064365386963e-10));
  const normal = Math.fround(Math.fround(Math.sqrt(Math.fround(Math.fround(-2) * Math.fround(Math.log(u1))))) * Math.fround(Math.cos(Math.fround(Math.fround(6.283185307179586) * u2))));
  const value = Math.fround(lambda + Math.fround(Math.fround(Math.sqrt(lambda)) * normal));
  return [Math.max(0, Math.floor(Math.fround(value + Math.fround(0.5)))) >>> 0, second];
}
