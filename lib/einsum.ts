export interface EinsumParsed {
  inputSubscripts: [string, string];
  outputSubscripts: string;
  allDims: string[];
  sumDims: string[];
}

export function parseEinsum(notation: string): EinsumParsed | null {
  const match = notation.replace(/\s/g, "").match(/^([a-z]+),([a-z]+)->([a-z]+)$/);
  if (!match) return null;
  const [, a, b, out] = match;
  const allSet = new Set([...a, ...b]);
  const outSet = new Set([...out]);
  const sumDims = [...allSet].filter((d) => !outSet.has(d));
  return {
    inputSubscripts: [a, b],
    outputSubscripts: out,
    allDims: [...allSet],
    sumDims,
  };
}

export function computeEinsum(
  parsed: EinsumParsed,
  matA: number[][],
  matB: number[][],
  dimSizes: Record<string, number>
): { intermediate: number[][][]; result: number[][] } {
  const [subsA, subsB] = parsed.inputSubscripts;
  const outSubs = parsed.outputSubscripts;

  // Output dims order from outputSubscripts
  const outDim0 = outSubs[0];
  const outDim1 = outSubs[1];
  const size0 = dimSizes[outDim0] ?? 1;
  const size1 = dimSizes[outDim1] ?? 1;

  // Sum dims
  const sumDimSizes = parsed.sumDims.map((d) => dimSizes[d] ?? 1);
  const totalSumSize = sumDimSizes.reduce((a, b) => a * b, 1);

  // intermediate[i][k][flatSumIdx] = product of A and B entries
  const intermediate: number[][][] = [];
  const result: number[][] = [];

  for (let i = 0; i < size0; i++) {
    intermediate[i] = [];
    result[i] = [];
    for (let k = 0; k < size1; k++) {
      intermediate[i][k] = [];
      let sum = 0;
      for (let s = 0; s < totalSumSize; s++) {
        // Decode multi-dim sum index
        const sumIndices: Record<string, number> = {};
        let rem = s;
        for (let d = parsed.sumDims.length - 1; d >= 0; d--) {
          const sz = sumDimSizes[d];
          sumIndices[parsed.sumDims[d]] = rem % sz;
          rem = Math.floor(rem / sz);
        }

        const allIndices: Record<string, number> = {
          ...sumIndices,
          [outDim0]: i,
          [outDim1]: k,
        };

        // Look up A[row][col] based on subscript mapping
        const aRow = allIndices[subsA[0]] ?? 0;
        const aCol = allIndices[subsA[1]] ?? 0;
        const bRow = allIndices[subsB[0]] ?? 0;
        const bCol = allIndices[subsB[1]] ?? 0;

        const aVal = matA[aRow]?.[aCol] ?? 0;
        const bVal = matB[bRow]?.[bCol] ?? 0;
        const prod = aVal * bVal;
        intermediate[i][k][s] = prod;
        sum += prod;
      }
      result[i][k] = sum;
    }
  }

  return { intermediate, result };
}
