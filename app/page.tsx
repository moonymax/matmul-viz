"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import { parseEinsum, computeEinsum } from "@/lib/einsum";
import EinsumInput from "@/components/EinsumInput";
import MatrixInput from "@/components/MatrixInput";

const Scene = dynamic(() => import("@/components/Scene"), { ssr: false });

function makeMatrix(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

export default function Home() {
  const [notation, setNotation] = useState("ij,jk->ik");
  const [dimSizes, setDimSizes] = useState<Record<string, number>>({ i: 2, j: 4, k: 3 });
  const [matA, setMatA] = useState<number[][]>(() => makeMatrix(2, 4));
  const [matB, setMatB] = useState<number[][]>(() => makeMatrix(4, 3));

  const parsed = useMemo(() => parseEinsum(notation), [notation]);

  const aDims = useMemo(() => {
    if (!parsed) return { rows: 2, cols: 2 };
    return {
      rows: dimSizes[parsed.inputSubscripts[0][0]] ?? 2,
      cols: dimSizes[parsed.inputSubscripts[0][1]] ?? 2,
    };
  }, [parsed, dimSizes]);

  const bDims = useMemo(() => {
    if (!parsed) return { rows: 2, cols: 2 };
    return {
      rows: dimSizes[parsed.inputSubscripts[1][0]] ?? 2,
      cols: dimSizes[parsed.inputSubscripts[1][1]] ?? 2,
    };
  }, [parsed, dimSizes]);

  const resizeMatrix = useCallback((prev: number[][], rows: number, cols: number) => {
    return Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols }, (_, c) => prev[r]?.[c] ?? 0)
    );
  }, []);

  useEffect(() => {
    setMatA((prev) => resizeMatrix(prev, aDims.rows, aDims.cols));
  }, [aDims.rows, aDims.cols, resizeMatrix]);

  useEffect(() => {
    setMatB((prev) => resizeMatrix(prev, bDims.rows, bDims.cols));
  }, [bDims.rows, bDims.cols, resizeMatrix]);

  useEffect(() => {
    if (!parsed) return;
    setDimSizes((prev) => {
      const next = { ...prev };
      for (const d of parsed.allDims) {
        if (!(d in next)) next[d] = 2;
      }
      return next;
    });
  }, [parsed]);

  const { intermediate, result } = useMemo(() => {
    if (!parsed) return { intermediate: [] as number[][][], result: [] as number[][] };
    return computeEinsum(parsed, matA, matB, dimSizes);
  }, [parsed, matA, matB, dimSizes]);

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      <div className="w-80 flex-shrink-0 p-4 space-y-6 overflow-y-auto border-r border-gray-700">
        <h1 className="text-xl font-bold">Matrix Multiply Viz</h1>
        <EinsumInput
          notation={notation}
          setNotation={setNotation}
          parsed={parsed}
          dimSizes={dimSizes}
          setDimSizes={setDimSizes}
        />
        {parsed && (
          <>
            <MatrixInput label="Matrix A" rows={aDims.rows} cols={aDims.cols} data={matA} setData={setMatA} />
            <MatrixInput label="Matrix B" rows={bDims.rows} cols={bDims.cols} data={matB} setData={setMatB} />
            <div className="space-y-2">
              <span className="font-medium text-sm">Result</span>
              <div className="overflow-auto max-h-48">
                <table className="border-collapse">
                  <tbody>
                    {result.map((row, r) => (
                      <tr key={r}>
                        {row.map((v, c) => (
                          <td key={c} className="w-14 h-8 text-center text-sm bg-gray-800 border border-gray-700">
                            {v}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="flex-1">
        {parsed ? (
          <Scene
            parsed={parsed}
            dimSizes={dimSizes}
            matA={matA}
            matB={matB}
            intermediate={intermediate}
            result={result}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            Enter a valid einsum notation to visualize
          </div>
        )}
      </div>
    </div>
  );
}
