// SVG chart renderers for the matrix-training demo, returned as strings and
// injected via dangerouslySetInnerHTML. Themed for a dark background: the
// diverging blue→grey→red value palette is kept (it's the didactic core),
// but chrome (titles, grids, ticks) uses light-on-dark colors.

import { Matrix, Vec, SPARSE_DIM, maxAbs, flatten2D } from "./sparseTraining";

const BLUE = [56, 132, 214];
const GREY = [148, 156, 168];
const RED = [224, 78, 92];

const INK = "#e5e7eb"; // titles
const MUTED = "#9aa4b2"; // tick / axis labels
const GRID = "rgba(255,255,255,0.07)";
const ZERO_LINE = "rgba(255,255,255,0.28)";

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function mixColor(c1: number[], c2: number[], t: number): string {
  return `rgb(${Math.round(lerp(c1[0], c2[0], t))}, ${Math.round(
    lerp(c1[1], c2[1], t)
  )}, ${Math.round(lerp(c1[2], c2[2], t))})`;
}
export function divergingColor(value: number, maxAbsValue: number): string {
  const t = Math.max(-1, Math.min(1, value / maxAbsValue));
  if (t < 0) return mixColor(BLUE, GREY, t + 1);
  return mixColor(GREY, RED, t);
}
function positiveColor(value: number, maxValue: number): string {
  const t = Math.max(0, Math.min(1, value / Math.max(maxValue, 1e-9)));
  return mixColor([70, 82, 96], RED, t);
}
function fmt(value: number, digits = 2, signed = false): string {
  const out = Number(value).toFixed(digits);
  if (!signed) return out;
  return (value >= 0 ? "+" : "") + out;
}
interface BarOpts {
  width?: number;
  height?: number;
  title?: string;
  digits?: number;
  maxAbs?: number;
  positiveOnly?: boolean;
  selectedIndex?: number;
  selectedLabel?: string;
}

export function renderBarChart(values: Vec, labels: string[], opts: BarOpts = {}): string {
  const width = opts.width || 420;
  const height = opts.height || 300;
  const title = opts.title || "";
  const digits = opts.digits ?? 2;
  const maxAbsValue = opts.maxAbs || maxAbs(values);
  const positiveOnly = opts.positiveOnly || false;
  const { selectedIndex, selectedLabel } = opts;
  const margin = { top: 34, right: 18, bottom: 42, left: 28 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const minY = positiveOnly ? 0 : Math.min(0, ...values) * 1.18 || -1;
  const maxY = positiveOnly
    ? Math.max(...values) * 1.18 || 1
    : Math.max(0, ...values) * 1.18 || 1;
  const y = (v: number) => margin.top + ((maxY - v) / (maxY - minY)) * innerH;
  const zeroY = y(0);
  const slot = innerW / values.length;
  const barW = slot * 0.72;

  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${title}">`;
  svg += `<text x="${margin.left}" y="20" font-size="15" font-weight="700" fill="${INK}">${title}</text>`;
  for (let g = 0; g < 4; g++) {
    const yy = margin.top + (g / 3) * innerH;
    svg += `<line x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}" stroke="${GRID}" stroke-width="1"/>`;
  }
  svg += `<line x1="${margin.left}" y1="${zeroY}" x2="${width - margin.right}" y2="${zeroY}" stroke="${ZERO_LINE}" stroke-width="1.4"/>`;

  values.forEach((v, i) => {
    const x = margin.left + i * slot + (slot - barW) / 2;
    const yv = y(v);
    const h = Math.abs(zeroY - yv);
    const rectY = v >= 0 ? yv : zeroY;
    const fill = positiveOnly ? positiveColor(v, maxY) : divergingColor(v, maxAbsValue);
    const selected = selectedIndex === i;
    const stroke = selected ? "#fff" : "rgba(255,255,255,0.18)";
    const strokeWidth = selected ? 2.2 : 0.8;
    svg += `<rect x="${x.toFixed(2)}" y="${rectY.toFixed(2)}" width="${barW.toFixed(2)}" height="${h.toFixed(2)}" rx="3" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
    const labelY = v >= 0 ? yv - 8 : zeroY + h + 16;
    svg += `<text x="${(x + barW / 2).toFixed(2)}" y="${labelY.toFixed(2)}" font-size="12" font-weight="700" text-anchor="middle" fill="${INK}">${fmt(v, digits, positiveOnly)}</text>`;
    svg += `<text x="${(x + barW / 2).toFixed(2)}" y="${height - 14}" font-size="11" fill="${MUTED}" text-anchor="middle">${labels[i]}</text>`;
  });
  if (selectedLabel) {
    svg += `<text x="${width - 16}" y="20" font-size="11" fill="${MUTED}" text-anchor="end">${selectedLabel}</text>`;
  }
  svg += `</svg>`;
  return svg;
}

interface HeatOpts {
  width?: number;
  height?: number;
  title?: string;
  digits?: number;
  maxAbs?: number;
}

export function renderHeatmap(
  matrix: Matrix,
  rowLabels: string[],
  colLabels: string[],
  opts: HeatOpts = {}
): string {
  const width = opts.width || 420;
  const height = opts.height || 300;
  const title = opts.title || "";
  const maxAbsValue = opts.maxAbs || maxAbs(flatten2D(matrix));
  const margin = { top: 34, right: 22, bottom: 42, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const rows = matrix.length;
  const cols = matrix[0].length;
  const cw = innerW / cols;
  const ch = innerH / rows;

  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${title}">`;
  svg += `<text x="${margin.left}" y="20" font-size="15" font-weight="700" fill="${INK}">${title}</text>`;
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const v = matrix[i][j];
      const x = margin.left + j * cw;
      const yy = margin.top + i * ch;
      svg += `<rect x="${x.toFixed(2)}" y="${yy.toFixed(2)}" width="${cw.toFixed(2)}" height="${ch.toFixed(2)}" fill="${divergingColor(v, maxAbsValue)}" stroke="rgba(0,0,0,0.25)" stroke-width="0.5"/>`;
    }
  rowLabels.forEach((label, i) => {
    const yy = margin.top + i * ch + ch / 2 + 5;
    svg += `<text x="${margin.left - 8}" y="${yy.toFixed(2)}" font-size="11" fill="${MUTED}" text-anchor="end">${label}</text>`;
  });
  colLabels.forEach((label, j) => {
    const x = margin.left + j * cw + cw / 2;
    svg += `<text x="${x.toFixed(2)}" y="${height - 14}" font-size="11" fill="${MUTED}" text-anchor="middle">${label}</text>`;
  });
  svg += `</svg>`;
  return svg;
}

interface SparseOpts extends HeatOpts {
  size?: number;
  activePositions?: [number, number][];
  grown?: [number, number] | null;
  pruned?: [number, number] | null;
}

// Max cells drawn per side for a fully-dense matrix; larger matrices are
// block-averaged down to this so we never emit tens of thousands of rects.
const DISPLAY_CAP = 32;

export function renderSparseHeatmap(matrix: Matrix, opts: SparseOpts = {}): string {
  const size = opts.size || SPARSE_DIM;
  const width = opts.width || 420;
  const height = opts.height || 320;
  const title = opts.title || "";
  const maxAbsValue = opts.maxAbs || maxAbs(flatten2D(matrix));
  const activePositions = opts.activePositions || [];
  const margin = { top: 34, right: 18, bottom: 20, left: 18 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const cw = innerW / size;
  const ch = innerH / size;

  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${title}">`;
  svg += `<text x="${margin.left}" y="20" font-size="15" font-weight="700" fill="${INK}">${title}</text>`;
  // faint backdrop so the grid extent is visible
  svg += `<rect x="${margin.left}" y="${margin.top}" width="${innerW.toFixed(2)}" height="${innerH.toFixed(2)}" fill="rgba(255,255,255,0.03)"/>`;

  if (activePositions.length > 0) {
    // sparse: draw only the active cells (cheap regardless of matrix size)
    for (const [i, j] of activePositions) {
      const v = matrix[i][j];
      const x = margin.left + j * cw;
      const yy = margin.top + i * ch;
      svg += `<rect x="${x.toFixed(2)}" y="${yy.toFixed(2)}" width="${Math.max(cw, 2).toFixed(2)}" height="${Math.max(ch, 2).toFixed(2)}" fill="${divergingColor(v, maxAbsValue)}" stroke="rgba(255,255,255,0.5)" stroke-width="0.4"/>`;
    }
    const marker = (pos: [number, number], color: string, dashed: boolean) => {
      const [i, j] = pos;
      const x = margin.left + j * cw;
      const yy = margin.top + i * ch;
      const dash = dashed ? ` stroke-dasharray="3 2"` : "";
      const w = Math.max(cw, 2);
      const h = Math.max(ch, 2);
      return `<rect x="${(x - 1.5).toFixed(2)}" y="${(yy - 1.5).toFixed(2)}" width="${(w + 3).toFixed(2)}" height="${(h + 3).toFixed(2)}" fill="none" stroke="${color}" stroke-width="1.6"${dash}/>`;
    };
    if (opts.pruned) svg += marker(opts.pruned, "#f87171", true);
    if (opts.grown) svg += marker(opts.grown, "#4ade80", false);
  } else {
    // dense: draw every cell, block-averaging down if the matrix is large
    const D = Math.min(size, DISPLAY_CAP);
    const dcw = innerW / D;
    const dch = innerH / D;
    for (let di = 0; di < D; di++)
      for (let dj = 0; dj < D; dj++) {
        const i0 = Math.floor((di * size) / D);
        const i1 = Math.max(i0 + 1, Math.floor(((di + 1) * size) / D));
        const j0 = Math.floor((dj * size) / D);
        const j1 = Math.max(j0 + 1, Math.floor(((dj + 1) * size) / D));
        let sum = 0;
        let cnt = 0;
        for (let i = i0; i < i1; i++)
          for (let j = j0; j < j1; j++) {
            sum += matrix[i][j];
            cnt++;
          }
        const v = sum / cnt;
        svg += `<rect x="${(margin.left + dj * dcw).toFixed(2)}" y="${(margin.top + di * dch).toFixed(2)}" width="${dcw.toFixed(2)}" height="${dch.toFixed(2)}" fill="${divergingColor(v, maxAbsValue)}"/>`;
      }
  }

  svg += `</svg>`;
  return svg;
}

export function renderLossChart(
  fullLosses: number[],
  batchLosses: number[],
  currentStep: number,
  opts: { width?: number; height?: number; title?: string } = {}
): string {
  const width = opts.width || 920;
  const height = opts.height || 300;
  const title = opts.title || "";
  const margin = { top: 34, right: 20, bottom: 42, left: 54 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const maxStep = Math.max(1, fullLosses.length - 1);
  const allLosses = fullLosses.concat(batchLosses);
  const top = Math.max(...allLosses) * 1.08;
  const bottom = Math.min(0, Math.min(...allLosses) * 0.92);
  const x = (step: number) => margin.left + (step / maxStep) * innerW;
  const y = (val: number) => margin.top + ((top - val) / (top - bottom)) * innerH;
  const pointsFull = fullLosses.map((loss, i) => `${x(i).toFixed(2)},${y(loss).toFixed(2)}`);
  const pointsBatch = batchLosses.map((loss, i) => `${x(i).toFixed(2)},${y(loss).toFixed(2)}`);
  const currentX = x(currentStep);
  const currentYFull = y(fullLosses[currentStep]);
  const currentYBatch = y(batchLosses[currentStep]);

  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Loss curve">`;
  svg += `<text x="${margin.left}" y="20" font-size="15" font-weight="700" fill="${INK}">${title}</text>`;
  for (let t = 0; t < 5; t++) {
    const yy = margin.top + (t / 4) * innerH;
    svg += `<line x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}" stroke="${GRID}" stroke-width="1"/>`;
  }
  for (let t = 0; t <= 5; t++) {
    const idx = Math.round((maxStep / 5) * t);
    const xx = x(idx);
    svg += `<line x1="${xx}" y1="${margin.top}" x2="${xx}" y2="${height - margin.bottom}" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>`;
    svg += `<text x="${xx}" y="${height - 14}" font-size="11" fill="${MUTED}" text-anchor="middle">${idx}</text>`;
  }
  [top, top * 0.75, top * 0.5, top * 0.25, bottom].forEach((tick) => {
    const yy = y(tick);
    svg += `<text x="${margin.left - 8}" y="${yy + 4}" font-size="11" fill="${MUTED}" text-anchor="end">${fmt(tick, 3)}</text>`;
  });
  svg += `<polyline points="${pointsFull.join(" ")}" fill="none" stroke="rgb(56,132,214)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`;
  svg += `<polyline points="${pointsBatch.join(" ")}" fill="none" stroke="rgb(224,78,92)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>`;
  svg += `<line x1="${currentX.toFixed(2)}" y1="${margin.top}" x2="${currentX.toFixed(2)}" y2="${height - margin.bottom}" stroke="rgba(255,255,255,0.5)" stroke-dasharray="6 5" stroke-width="1.5"/>`;
  svg += `<circle cx="${currentX.toFixed(2)}" cy="${currentYFull.toFixed(2)}" r="6" fill="rgb(56,132,214)" stroke="#0a0f1a" stroke-width="2.5"/>`;
  svg += `<circle cx="${currentX.toFixed(2)}" cy="${currentYBatch.toFixed(2)}" r="5" fill="rgb(224,78,92)" stroke="#0a0f1a" stroke-width="2.5"/>`;
  svg += `<rect x="${width - 190}" y="${margin.top - 2}" width="14" height="14" rx="4" fill="rgb(56,132,214)"/>`;
  svg += `<text x="${width - 170}" y="${margin.top + 11}" font-size="12" fill="${MUTED}">Full dataset loss</text>`;
  svg += `<rect x="${width - 190}" y="${margin.top + 20}" width="14" height="14" rx="4" fill="rgb(224,78,92)"/>`;
  svg += `<text x="${width - 170}" y="${margin.top + 33}" font-size="12" fill="${MUTED}">Batch loss</text>`;
  svg += `<text x="${currentX + 12}" y="${currentYFull - 12}" font-size="12" font-weight="700" fill="${INK}">${fmt(fullLosses[currentStep], 4)}</text>`;
  svg += `<text x="${currentX + 12}" y="${currentYBatch + 18}" font-size="12" font-weight="700" fill="${INK}">${fmt(batchLosses[currentStep], 4)}</text>`;
  svg += `</svg>`;
  return svg;
}
