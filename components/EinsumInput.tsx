"use client";

import { EinsumParsed } from "@/lib/einsum";

interface Props {
  notation: string;
  setNotation: (s: string) => void;
  parsed: EinsumParsed | null;
  dimSizes: Record<string, number>;
  setDimSizes: (d: Record<string, number>) => void;
}

export default function EinsumInput({ notation, setNotation, parsed, dimSizes, setDimSizes }: Props) {
  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium">Einsum Notation</label>
      <input
        className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded font-mono text-lg"
        value={notation}
        onChange={(e) => setNotation(e.target.value)}
        placeholder="ij,jk->ik"
      />
      {parsed && (
        <div className="flex flex-wrap gap-3">
          {parsed.allDims.map((d) => (
            <div key={d} className="flex items-center gap-2">
              <label className="font-mono text-sm">{d} =</label>
              <input
                type="number"
                min={1}
                max={16}
                className="w-16 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-center"
                value={dimSizes[d] ?? 2}
                onChange={(e) =>
                  setDimSizes({ ...dimSizes, [d]: Math.max(1, Math.min(16, parseInt(e.target.value) || 1)) })
                }
              />
            </div>
          ))}
          <span className="text-xs text-gray-400 self-center">
            Sum over: {parsed.sumDims.join(", ") || "none"}
          </span>
        </div>
      )}
      {!parsed && notation.length > 0 && (
        <p className="text-red-400 text-sm">Invalid einsum notation. Use format like: ij,jk-&gt;ik</p>
      )}
    </div>
  );
}
