"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import ColoredCell from "./ColoredCell";
import { EinsumParsed } from "@/lib/einsum";

interface Props {
  parsed: EinsumParsed;
  dimSizes: Record<string, number>;
  matA: number[][];
  matB: number[][];
  intermediate: number[][][];
  result: number[][];
}

export default function Scene({ parsed, dimSizes, matA, matB, intermediate, result }: Props) {
  const outDim0 = parsed.outputSubscripts[0];
  const outDim1 = parsed.outputSubscripts[1];
  const [subsA, subsB] = parsed.inputSubscripts;

  const size0 = dimSizes[outDim0] ?? 1; // rows of result (i)
  const size1 = dimSizes[outDim1] ?? 1; // cols of result (k)
  const sumSize = parsed.sumDims.reduce((a, d) => a * (dimSizes[d] ?? 1), 1); // depth (j)

  const aRows = dimSizes[subsA[0]] ?? 1;
  const aCols = dimSizes[subsA[1]] ?? 1;
  const bRows = dimSizes[subsB[0]] ?? 1;
  const bCols = dimSizes[subsB[1]] ?? 1;

  // Layout: Matrix A on left, cube in center, Matrix B on top, Result on right
  const gap = 2;
  const cubeOriginX = 0;
  const cubeOriginY = 0;
  const cubeOriginZ = 0;

  return (
    <Canvas
      camera={{ position: [size0 + 8, size1 + 5, sumSize + 8], fov: 50 }}
      gl={{ powerPreference: "high-performance", antialias: false }}
      onCreated={({ gl }) => {
        const canvas = gl.domElement;
        canvas.addEventListener("webglcontextlost", (e) => {
          e.preventDefault();
          console.warn("WebGL context lost — will attempt restore");
        });
        canvas.addEventListener("webglcontextrestored", () => {
          console.log("WebGL context restored");
        });
      }}
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 10]} intensity={0.8} />
      <OrbitControls />

      {/* Label */}
      <Html position={[cubeOriginX + size0 / 2, size1 + gap + 2, cubeOriginZ]} center>
        <span style={{ color: "white", fontSize: 14, whiteSpace: "nowrap" }}>
          {`${parsed.inputSubscripts[0]},${parsed.inputSubscripts[1]}->${parsed.outputSubscripts}`}
        </span>
      </Html>

      {/* Matrix A — laid flat on top of the intermediate cube (XZ plane) */}
      <group position={[cubeOriginX, cubeOriginY + size1 + gap, cubeOriginZ]}>
        <Html position={[aRows / 2 - 0.5, 1.2, aCols / 2 - 0.5]} center>
          <span style={{ color: "#aaa", fontSize: 12, whiteSpace: "nowrap" }}>
            Matrix A ({aRows}x{aCols})
          </span>
        </Html>
        {matA.map((row, r) =>
          row.map((v, c) => (
            <ColoredCell key={`a-${r}-${c}`} position={[r, 0, c]} value={v} size={0.85} />
          ))
        )}
      </group>

      {/* Matrix B — to the left of the intermediate, aligned in YZ plane */}
      <group position={[cubeOriginX - gap, cubeOriginY, cubeOriginZ]}>
        <Html position={[-1.2, bCols / 2 - 0.5, bRows / 2 - 0.5]} center>
          <span style={{ color: "#aaa", fontSize: 12, whiteSpace: "nowrap" }}>
            Matrix B ({bRows}x{bCols})
          </span>
        </Html>
        {matB.map((row, r) =>
          row.map((v, c) => (
            <ColoredCell key={`b-${r}-${c}`} position={[0, c, r]} value={v} size={0.85} />
          ))
        )}
      </group>

      {/* Intermediate cube — element-wise products */}
      <group position={[cubeOriginX, cubeOriginY, cubeOriginZ]}>
        {intermediate.map((plane, i) =>
          plane.map((row, k) =>
            row.map((v, s) => (
              <ColoredCell
                key={`c-${i}-${k}-${s}`}
                position={[i, k, s]}
                value={v}
                size={0.85}
                opacity={0.5}
              />
            ))
          )
        )}
      </group>

      {/* Result matrix — on the right side */}
      <group position={[cubeOriginX, cubeOriginY, sumSize + gap - 1]}>
        <Html position={[size0 / 2 - 0.5, -1.2, 0]} center>
          <span style={{ color: "#aaa", fontSize: 12, whiteSpace: "nowrap" }}>
            Result ({size0}x{size1})
          </span>
        </Html>
        {result.map((row, i) =>
          row.map((v, k) => (
            <ColoredCell key={`r-${i}-${k}`} position={[i, k, 0]} value={v} size={0.85} />
          ))
        )}
      </group>
    </Canvas>
  );
}
