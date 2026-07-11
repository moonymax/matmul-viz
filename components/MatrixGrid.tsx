"use client";

import { Matrix } from "@/lib/autograd";
import { valueToColor, maxAbsOf } from "@/lib/nnColor";

interface Props {
  m: Matrix;
  // "grad" grids get a subtly different frame so you can tell them apart
  variant?: "value" | "grad";
  dim?: boolean; // not yet revealed -> faded
}

export default function MatrixGrid({ m, variant = "value", dim = false }: Props) {
  const maxAbs = maxAbsOf(m);
  const cols = m[0]?.length ?? 0;
  // shrink cells a touch for wide matrices so cards stay compact
  const cell = cols > 4 ? 26 : 32;

  return (
    <div
      className="inline-grid gap-[2px] rounded transition-opacity duration-300"
      style={{
        gridTemplateColumns: `repeat(${cols}, ${cell}px)`,
        opacity: dim ? 0.18 : 1,
      }}
    >
      {m.flatMap((row, i) =>
        row.map((v, j) => (
          <div
            key={`${i}-${j}`}
            title={v.toFixed(4)}
            className="flex items-center justify-center rounded-[3px] font-mono"
            style={{
              width: cell,
              height: cell,
              background: valueToColor(v, maxAbs),
              fontSize: cols > 4 ? 8 : 9,
              color: "rgba(255,255,255,0.92)",
              textShadow: "0 1px 1px rgba(0,0,0,0.55)",
              outline: variant === "grad" ? "1px dashed rgba(255,255,255,0.25)" : "none",
            }}
          >
            {formatVal(v)}
          </div>
        ))
      )}
    </div>
  );
}

function formatVal(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
