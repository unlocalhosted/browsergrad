/**
 * Compile-time CuTe layout values supported by CUDA-lite.
 *
 * This is deliberately not a general CuTe object model.  It models the
 * smallest useful layout value: a rank-one affine map with static shape and
 * stride.  Parser lowering erases the descriptor to scalar expressions, so
 * the semantic reference and WGSL emitters execute the same resulting IR.
 */
export interface CuteStaticRank1Layout {
  readonly kind: "cute-static-rank-1-layout";
  readonly extent: number;
  readonly stride: number;
}

export type CuteStaticLayoutQuery = "size" | "rank" | "cosize";

export function createCuteStaticRank1Layout(extent: number, stride = 1): CuteStaticRank1Layout {
  return { kind: "cute-static-rank-1-layout", extent, stride };
}

export function cuteStaticRank1LayoutQuery(layout: CuteStaticRank1Layout, query: CuteStaticLayoutQuery): number {
  switch (query) {
    case "size":
      return layout.extent;
    case "rank":
      return 1;
    case "cosize":
      return layout.extent === 0 ? 0 : 1 + ((layout.extent - 1) * layout.stride);
  }
}
