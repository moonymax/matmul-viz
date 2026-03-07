import * as THREE from "three";

const GREY = 0.4;
const THRESHOLD = 2;

export function valueToColor(v: number): THREE.Color {
  if (v === 0) return new THREE.Color(GREY, GREY, GREY);

  const absV = Math.abs(v);

  // 0..2: linear ramp from 0 to 1
  // >2: continues logarithmically beyond 1
  let t: number;
  if (absV <= THRESHOLD) {
    t = absV / THRESHOLD;
  } else {
    t = 1 + Math.log(absV / THRESHOLD);
  }

  // Map t to color intensity: 0 = grey, 1+ = full color
  const intensity = Math.min(t / (1 + t), 1); // asymptotically approaches 1
  const r = GREY * (1 - intensity);

  if (v > 0) {
    return new THREE.Color(GREY + (1 - GREY) * intensity, r, r);
  } else {
    return new THREE.Color(r, r, GREY + (1 - GREY) * intensity);
  }
}
