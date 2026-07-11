// A fixed 2-layer neural net wired up with the autograd engine, plus a
// "frame" builder that turns one training iteration into a scrubbable
// timeline: forward pass -> backward pass -> gradient-descent step.

import {
  Matrix,
  Tensor,
  cloneM,
  matmul,
  addBias,
  tanhT,
  mse,
  topoSort,
} from "./autograd";

export interface NetParams {
  X: Matrix;
  Y: Matrix;
  W1: Matrix;
  b1: Matrix;
  W2: Matrix;
  b2: Matrix;
}

export type Phase = "forward" | "backward" | "update";

export interface Frame {
  phase: Phase;
  title: string;
  detail: string;
  active: string[]; // labels currently highlighted
  dataShown: string[]; // labels whose value grid is revealed
  gradShown: string[]; // labels whose gradient grid is revealed
  data: Record<string, Matrix>;
  grad: Record<string, Matrix>;
}

export interface Epoch {
  frames: Frame[];
  loss: number;
  updated: Pick<NetParams, "W1" | "b1" | "W2" | "b2">;
}

// Layout of the computation graph, left-to-right. Each inner array is a
// column; params sit stacked in their own columns.
export const LAYOUT: string[][] = [
  ["X"],
  ["W1", "b1"],
  ["XW1"],
  ["Z1"],
  ["A1"],
  ["W2", "b2"],
  ["A1W2"],
  ["Z2"],
  ["Y"],
  ["Loss"],
];

export const NODE_META: Record<string, { title: string; sub: string }> = {
  X: { title: "X", sub: "input" },
  Y: { title: "Y", sub: "target" },
  W1: { title: "W₁", sub: "weights" },
  b1: { title: "b₁", sub: "bias" },
  W2: { title: "W₂", sub: "weights" },
  b2: { title: "b₂", sub: "bias" },
  XW1: { title: "XW₁", sub: "X · W₁" },
  Z1: { title: "Z₁", sub: "XW₁ + b₁" },
  A1: { title: "A₁", sub: "tanh(Z₁)" },
  A1W2: { title: "A₁W₂", sub: "A₁ · W₂" },
  Z2: { title: "Ŷ", sub: "A₁W₂ + b₂" },
  Loss: { title: "L", sub: "MSE(Ŷ, Y)" },
};

export function buildEpoch(net: NetParams, lr: number): Epoch {
  // --- build the graph (forward is computed eagerly on construction) ---
  const X = new Tensor(cloneM(net.X), { label: "X", formula: "X (input)" });
  const Y = new Tensor(cloneM(net.Y), { label: "Y", formula: "Y (target)" });
  const W1 = new Tensor(cloneM(net.W1), { label: "W1", isParam: true, formula: "W₁" });
  const b1 = new Tensor(cloneM(net.b1), { label: "b1", isParam: true, formula: "b₁" });
  const W2 = new Tensor(cloneM(net.W2), { label: "W2", isParam: true, formula: "W₂" });
  const b2 = new Tensor(cloneM(net.b2), { label: "b2", isParam: true, formula: "b₂" });

  const XW1 = matmul(X, W1, "XW1", "X · W₁");
  const Z1 = addBias(XW1, b1, "Z1", "XW₁ + b₁");
  const A1 = tanhT(Z1, "A1", "tanh(Z₁)");
  const A1W2 = matmul(A1, W2, "A1W2", "A₁ · W₂");
  const Z2 = addBias(A1W2, b2, "Z2", "A₁W₂ + b₂");
  const Loss = mse(Z2, Y, "Loss", "MSE(Ŷ, Y)");

  const order = topoSort(Loss);
  const nodes = order; // includes leaves + computed
  const leaves = order.filter((t) => t.prev.length === 0);
  const computed = order.filter((t) => t.prev.length > 0);

  const dataSnapshot = (): Record<string, Matrix> =>
    Object.fromEntries(nodes.map((n) => [n.label, cloneM(n.data)]));
  const gradSnapshot = (): Record<string, Matrix> =>
    Object.fromEntries(nodes.map((n) => [n.label, cloneM(n.grad)]));

  const frames: Frame[] = [];
  const baseData = dataSnapshot();

  // --- FORWARD ---
  const shownData = new Set(leaves.map((l) => l.label));
  frames.push({
    phase: "forward",
    title: "Setup",
    detail: "Inputs X, target Y and parameters W₁, b₁, W₂, b₂ are given.",
    active: [],
    dataShown: [...shownData],
    gradShown: [],
    data: baseData,
    grad: gradSnapshot(),
  });

  for (const node of computed) {
    shownData.add(node.label);
    frames.push({
      phase: "forward",
      title: `Forward — ${NODE_META[node.label]?.title ?? node.label}`,
      detail: `Compute ${node.label} = ${node.formula}.`,
      active: [node.label],
      dataShown: [...shownData],
      gradShown: [],
      data: baseData,
      grad: gradSnapshot(),
    });
  }

  // --- BACKWARD ---
  for (const n of nodes) n.zeroGrad();
  Loss.grad = [[1]];
  const shownGrad = new Set<string>(["Loss"]);
  frames.push({
    phase: "backward",
    title: "Backward — seed",
    detail: "Seed the gradient: ∂L/∂L = 1. Now flow it back through the graph.",
    active: ["Loss"],
    dataShown: [...shownData],
    gradShown: [...shownGrad],
    data: baseData,
    grad: gradSnapshot(),
  });

  const reversed = [...computed].reverse();
  for (const node of reversed) {
    node._backward();
    for (const p of node.prev) shownGrad.add(p.label);
    frames.push({
      phase: "backward",
      title: `Backward — ${NODE_META[node.label]?.title ?? node.label}`,
      detail: `Backprop through ${node.label} = ${node.formula}; push gradients into ${node.prev
        .map((p) => p.label)
        .join(", ")}.`,
      active: [node.label, ...node.prev.map((p) => p.label)],
      dataShown: [...shownData],
      gradShown: [...shownGrad],
      data: baseData,
      grad: gradSnapshot(),
    });
  }

  // --- GRADIENT DESCENT STEP ---
  const params = [W1, b1, W2, b2];
  const updated = {
    W1: cloneM(W1.data),
    b1: cloneM(b1.data),
    W2: cloneM(W2.data),
    b2: cloneM(b2.data),
  };
  const updatedData: Record<string, Matrix> = { ...baseData };
  for (const p of params) {
    const nd = p.data.map((row, i) => row.map((v, j) => v - lr * p.grad[i][j]));
    updatedData[p.label] = nd;
    (updated as Record<string, Matrix>)[p.label] = nd;
  }
  frames.push({
    phase: "update",
    title: "Gradient descent step",
    detail: `Update every parameter: θ ← θ − η·∇θ  (η = ${lr}). Smaller loss next round.`,
    active: params.map((p) => p.label),
    dataShown: [...shownData],
    gradShown: [...shownGrad],
    data: updatedData,
    grad: gradSnapshot(),
  });

  return { frames, loss: Loss.data[0][0], updated };
}

// Deterministic seeded RNG so server and client render the same initial net.
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

export function randomParams(
  seed: number,
  inF = 3,
  hidden = 4,
  outF = 2,
  samples = 4
): NetParams {
  const rng = makeRng(seed);
  const rand = () => Math.round((rng() * 2 - 1) * 100) / 100;
  const mat = (r: number, c: number, f: () => number): Matrix =>
    Array.from({ length: r }, () => Array.from({ length: c }, f));

  return {
    X: [
      [0, 0, 1],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
    ].slice(0, samples),
    Y: [
      [0, 1],
      [1, 0],
      [1, 0],
      [0, 1],
    ].slice(0, samples),
    W1: mat(inF, hidden, rand),
    b1: mat(1, hidden, () => 0),
    W2: mat(hidden, outF, rand),
    b2: mat(1, outF, () => 0),
  };
}
