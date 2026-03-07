"use client";

import { useRef } from "react";
import * as THREE from "three";
import { valueToColor } from "@/lib/color";

interface Props {
  position: [number, number, number];
  value: number;
  size?: number;
  opacity?: number;
}

export default function ColoredCell({ position, value, size = 0.9, opacity = 0.85 }: Props) {
  const meshRef = useRef<THREE.Mesh>(null);
  const color = valueToColor(value);

  return (
    <mesh ref={meshRef} position={position}>
      <boxGeometry args={[size, size, size]} />
      <meshStandardMaterial color={color} transparent opacity={opacity} />
    </mesh>
  );
}
