"use client";

interface Props {
  label: string;
  rows: number;
  cols: number;
  data: number[][];
  setData: (d: number[][]) => void;
}

export default function MatrixInput({ label, rows, cols, data, setData }: Props) {
  const fillRandom = () => {
    const d = Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.round((Math.random() * 20 - 10) * 10) / 10)
    );
    setData(d);
  };

  const clear = () => {
    setData(Array.from({ length: rows }, () => Array(cols).fill(0)));
  };

  const updateCell = (r: number, c: number, val: string) => {
    const next = data.map((row) => [...row]);
    next[r][c] = parseFloat(val) || 0;
    setData(next);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <span className="font-medium text-sm">{label}</span>
        <span className="text-xs text-gray-400">
          {rows}×{cols}
        </span>
        <button onClick={fillRandom} className="text-xs px-2 py-1 bg-gray-700 rounded hover:bg-gray-600">
          Random
        </button>
        <button onClick={clear} className="text-xs px-2 py-1 bg-gray-700 rounded hover:bg-gray-600">
          Clear
        </button>
      </div>
      <div className="overflow-auto max-h-48">
        <table className="border-collapse">
          <tbody>
            {Array.from({ length: rows }, (_, r) => (
              <tr key={r}>
                {Array.from({ length: cols }, (_, c) => (
                  <td key={c} className="p-0">
                    <div className="flex flex-col items-center px-1 py-0.5">
                      <span className="text-xs text-gray-400">{data[r]?.[c] ?? 0}</span>
                      <input
                        type="range"
                        min="-10"
                        max="10"
                        step="0.1"
                        className="w-16 h-2 accent-blue-500"
                        value={data[r]?.[c] ?? 0}
                        onChange={(e) => updateCell(r, c, e.target.value)}
                      />
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
