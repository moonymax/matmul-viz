// A tiny reverse-mode autograd over 2D matrices (tensors).
// Deliberately unoptimized and explicit — every op knows how to push
// gradients into its inputs, so we can "play" the backward pass op by op.

export type Matrix = number[][];

export const zeros = (rows: number, cols: number): Matrix =>
  Array.from({ length: rows }, () => Array(cols).fill(0));

export const cloneM = (m: Matrix): Matrix => m.map((row) => row.slice());

export function matMul(a: Matrix, b: Matrix): Matrix {
  const n = a.length;
  const m = b[0].length;
  const k = b.length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += a[i][p] * b[p][j];
      out[i][j] = s;
    }
  }
  return out;
}

export function transpose(a: Matrix): Matrix {
  const out = zeros(a[0].length, a.length);
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
  return out;
}

// Accumulate src into dst in place (gradients accumulate).
function addInto(dst: Matrix, src: Matrix) {
  for (let i = 0; i < dst.length; i++)
    for (let j = 0; j < dst[0].length; j++) dst[i][j] += src[i][j];
}

export interface TensorOpts {
  label?: string;
  op?: string;
  formula?: string;
  prev?: Tensor[];
  isParam?: boolean;
}

export class Tensor {
  data: Matrix;
  grad: Matrix;
  rows: number;
  cols: number;
  label: string;
  op: string;
  formula: string;
  prev: Tensor[];
  isParam: boolean;
  _backward: () => void = () => {};

  constructor(data: Matrix, opts: TensorOpts = {}) {
    this.data = data;
    this.rows = data.length;
    this.cols = data[0].length;
    this.grad = zeros(this.rows, this.cols);
    this.label = opts.label ?? "";
    this.op = opts.op ?? "input";
    this.formula = opts.formula ?? this.label;
    this.prev = opts.prev ?? [];
    this.isParam = opts.isParam ?? false;
  }

  zeroGrad() {
    this.grad = zeros(this.rows, this.cols);
  }
}

// out = A · B
export function matmul(a: Tensor, b: Tensor, label: string, formula: string): Tensor {
  const out = new Tensor(matMul(a.data, b.data), {
    label,
    formula,
    op: "matmul",
    prev: [a, b],
  });
  out._backward = () => {
    addInto(a.grad, matMul(out.grad, transpose(b.data)));
    addInto(b.grad, matMul(transpose(a.data), out.grad));
  };
  return out;
}

// out = A + b, where b is a (1 x cols) bias broadcast over rows.
export function addBias(a: Tensor, b: Tensor, label: string, formula: string): Tensor {
  const data = zeros(a.rows, a.cols);
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < a.cols; j++) data[i][j] = a.data[i][j] + b.data[0][j];
  const out = new Tensor(data, { label, formula, op: "add", prev: [a, b] });
  out._backward = () => {
    addInto(a.grad, out.grad);
    for (let j = 0; j < a.cols; j++)
      for (let i = 0; i < a.rows; i++) b.grad[0][j] += out.grad[i][j];
  };
  return out;
}

// out = tanh(A) elementwise
export function tanhT(a: Tensor, label: string, formula: string): Tensor {
  const data = a.data.map((row) => row.map((v) => Math.tanh(v)));
  const out = new Tensor(data, { label, formula, op: "tanh", prev: [a] });
  out._backward = () => {
    for (let i = 0; i < a.rows; i++)
      for (let j = 0; j < a.cols; j++)
        a.grad[i][j] += (1 - out.data[i][j] * out.data[i][j]) * out.grad[i][j];
  };
  return out;
}

// scalar mean-squared-error loss; target is treated as a constant.
export function mse(pred: Tensor, target: Tensor, label: string, formula: string): Tensor {
  const n = pred.rows * pred.cols;
  let s = 0;
  for (let i = 0; i < pred.rows; i++)
    for (let j = 0; j < pred.cols; j++) {
      const d = pred.data[i][j] - target.data[i][j];
      s += d * d;
    }
  const out = new Tensor([[s / n]], { label, formula, op: "mse", prev: [pred] });
  out._backward = () => {
    const g = out.grad[0][0];
    for (let i = 0; i < pred.rows; i++)
      for (let j = 0; j < pred.cols; j++)
        pred.grad[i][j] += ((2 * (pred.data[i][j] - target.data[i][j])) / n) * g;
  };
  return out;
}

// Depth-first topological order: inputs first, loss last.
export function topoSort(root: Tensor): Tensor[] {
  const order: Tensor[] = [];
  const seen = new Set<Tensor>();
  const visit = (t: Tensor) => {
    if (seen.has(t)) return;
    seen.add(t);
    for (const p of t.prev) visit(p);
    order.push(t);
  };
  visit(root);
  return order;
}
