// 2D value colouring for the neural-net visualizer.
//   positive -> red, zero -> grey, negative -> blue.
// The gradient is LINEAR and normalized per-matrix against the largest
// absolute value, so even small values keep a visible tint.

const GREY: [number, number, number] = [128, 128, 128];
const RED: [number, number, number] = [225, 55, 55];
const BLUE: [number, number, number] = [55, 105, 225];

export function valueToColor(v: number, maxAbs: number): string {
  if (maxAbs <= 1e-12 || v === 0) return `rgb(${GREY[0]},${GREY[1]},${GREY[2]})`;
  // linear normalization, clamped to [-1, 1]
  const t = Math.max(-1, Math.min(1, v / maxAbs));
  const target = t > 0 ? RED : BLUE;
  const a = Math.abs(t);
  const c = GREY.map((g, i) => Math.round(g + (target[i] - g) * a));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function maxAbsOf(m: number[][]): number {
  let max = 0;
  for (const row of m) for (const v of row) max = Math.max(max, Math.abs(v));
  return max;
}
