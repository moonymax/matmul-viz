"use client";

import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import {
  buildSimulations,
  sparsePredict,
  matVecPublic,
  msePublic,
  maxAbs,
  flatten2D,
  MODE_META,
  DATASET_SIZE,
  SPARSE_DIM,
  Mode,
  Frame,
  Simulation,
} from "@/lib/sparseTraining";
import {
  renderBarChart,
  renderHeatmap,
  renderSparseHeatmap,
  renderLossChart,
} from "@/lib/trainingCharts";

const SPEED_INTERVAL = 150; // ms at 1×

function Svg({ html, className = "" }: { html: string; className?: string }) {
  return (
    <div
      className={`[&_svg]:block [&_svg]:w-full [&_svg]:h-auto ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function Card({
  title,
  sub,
  children,
  full = false,
}: {
  title: string;
  sub?: string;
  children: React.ReactNode;
  full?: boolean;
}) {
  return (
    <div
      className={`rounded-xl bg-gray-900/70 border border-gray-800 p-4 ${
        full ? "col-span-full" : ""
      }`}
    >
      <h2 className="text-base font-semibold text-gray-100">{title}</h2>
      {sub && <p className="text-xs text-gray-400 mt-0.5 mb-3 leading-snug min-h-[2.2em]">{sub}</p>}
      {children}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-gray-900 border border-gray-800 px-3 py-2 min-w-[110px]">
      <div className="text-[10px] uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-sm font-semibold font-mono text-gray-100">{value}</div>
    </div>
  );
}

export default function TrainingApp() {
  const sims = useMemo(() => buildSimulations(), []);
  const [mode, setMode] = useState<Mode>("nonlinear");
  const [steps, setSteps] = useState<Record<Mode, number>>({
    nonlinear: 0,
    sparse: 0,
    dense16: 0,
  });
  const [example, setExample] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);

  const sim: Simulation = sims[mode];
  const step = Math.min(steps[mode], sim.frames.length - 1);
  const frame = sim.frames[step];
  const meta = MODE_META[mode];

  const setStep = useCallback(
    (updater: number | ((s: number) => number)) => {
      setSteps((prev) => {
        const cur = prev[mode];
        const next = typeof updater === "function" ? updater(cur) : updater;
        return { ...prev, [mode]: Math.max(0, Math.min(sim.frames.length - 1, next)) };
      });
    },
    [mode, sim.frames.length]
  );

  // playback
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => {
      setSteps((prev) => {
        const cur = prev[mode];
        if (cur >= sim.frames.length - 1) {
          setPlaying(false);
          return prev;
        }
        return { ...prev, [mode]: cur + 1 };
      });
    }, SPEED_INTERVAL / speed);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, speed, mode, sim.frames.length]);

  const usesLift = mode !== "nonlinear";

  const predOf = useCallback(
    (idx: number, f: Frame) =>
      mode !== "nonlinear"
        ? sparsePredict(f.W, sim.X[idx], sim.routing!).pred
        : matVecPublic(f.W, sim.X[idx]),
    [mode, sim]
  );

  // ---- chart html ----
  const charts = useMemo(() => {
    const x = sim.X[example];
    const target = sim.targets[example];
    const liftView = usesLift ? sparsePredict(frame.W, x, sim.routing!) : null;
    const pred = liftView ? liftView.pred : matVecPublic(frame.W, x);
    const vecMax = maxAbs(target.concat(pred));

    const input = usesLift
      ? renderBarChart(
          liftView!.lifted,
          Array.from({ length: SPARSE_DIM }, (_, i) => `z${i + 1}`),
          { title: "Dense lifted vector z = U·x", maxAbs: maxAbs(liftView!.lifted), width: 520, height: 300 }
        )
      : renderBarChart(x, ["x1", "x2", "x3", "x4"], {
          title: "Input vector",
          maxAbs: maxAbs(x),
        });

    const rowL = ["y1", "y2", "y3", "y4"];
    const colL = ["x1", "x2", "x3", "x4"];
    // sparse tracks a moving active set; dense16 has every cell active
    const activePos =
      mode === "dense16" ? [] : frame.activePositions ?? sim.routing?.positions ?? [];
    const heat = (
      m: number[][],
      title: string,
      digits: number,
      markers = false
    ) =>
      usesLift
        ? renderSparseHeatmap(m, {
            title,
            digits,
            maxAbs: maxAbs(flatten2D(m)),
            activePositions: activePos,
            grown: markers && mode === "sparse" ? frame.grown : null,
            pruned: markers && mode === "sparse" ? frame.pruned : null,
          })
        : renderHeatmap(m, rowL, colL, { title, digits, maxAbs: maxAbs(flatten2D(m)) });

    const batchLossValues = frame.batchIndices.map((i) => msePublic(predOf(i, frame), sim.targets[i]));
    const selectedInBatch = frame.batchIndices.indexOf(example);

    return {
      input,
      weight: heat(
        frame.W,
        mode === "sparse"
          ? "Dynamic 16×16 scaffold"
          : mode === "dense16"
          ? "Dense 16×16 scaffold"
          : "Learned weight matrix",
        2,
        true
      ),
      grad: heat(frame.grad, "Gradient  dL/dW", 3),
      m1: heat(frame.m, "Adam 1st moment (m)", 4),
      m2: heat(frame.v, "Adam 2nd moment (v)", 4),
      target: renderBarChart(target, rowL, { title: "Target vector", maxAbs: vecMax }),
      pred: renderBarChart(pred, rowL, { title: "Predicted vector", maxAbs: vecMax }),
      batch: renderBarChart(
        batchLossValues,
        frame.batchIndices.map((i) => `#${i + 1}`),
        {
          title: "Mini-batch data losses",
          maxAbs: Math.max(...batchLossValues),
          positiveOnly: true,
          digits: 4,
          selectedIndex: selectedInBatch >= 0 ? selectedInBatch : undefined,
          selectedLabel: selectedInBatch >= 0 ? `selected #${example + 1}` : "",
        }
      ),
      loss: renderLossChart(
        sim.frames.map((f) => f.fullLoss),
        sim.frames.map((f) => f.batchLoss),
        step,
        { title: "Loss over training" }
      ),
    };
  }, [sim, frame, mode, usesLift, example, step, predOf]);

  const num = (v: number, d = 6) => v.toFixed(d);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-[1600px] mx-auto p-6 space-y-5">
        {/* header */}
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 text-xs text-gray-500 mb-1">
              <Link href="/" className="hover:text-sky-400">matmul</Link>
              <span>·</span>
              <Link href="/nn" className="hover:text-sky-400">autograd nn</Link>
              <span>·</span>
              <span className="text-gray-300">training</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight">Matrix Training with Adam</h1>
            <p className="text-sm text-gray-400 max-w-3xl mt-1 leading-snug">
              Train a 4×4 linear model against a nonlinear target, then compare a sparse
              reparameterization of the same model. Both use Adam — inspect the gradient alongside
              its running moments as you scrub through every mini-batch.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 justify-end">
            {[
              ["rgb(56,132,214)", "negative"],
              ["rgb(148,156,168)", "≈ zero"],
              ["rgb(224,78,92)", "positive"],
            ].map(([c, label]) => (
              <span
                key={label}
                className="inline-flex items-center gap-1.5 text-xs text-gray-300 bg-gray-900 border border-gray-800 rounded-full px-3 py-1"
              >
                <span className="w-2.5 h-2.5 rounded-sm" style={{ background: c }} />
                {label}
              </span>
            ))}
          </div>
        </header>

        {/* tabs */}
        <div className="inline-flex bg-gray-900 border border-gray-800 rounded-lg p-1 gap-1">
          {(["nonlinear", "sparse", "dense16"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => {
                setPlaying(false);
                setMode(m);
              }}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
                mode === m ? "bg-sky-600 text-white" : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {m === "nonlinear"
                ? "Nonlinear data"
                : m === "sparse"
                ? "Sparse matrix"
                : "Dense 16×16"}
            </button>
          ))}
        </div>

        <div className="rounded-lg bg-gray-900/60 border border-gray-800 px-4 py-2.5">
          <div className="text-xs font-semibold text-sky-300">{meta.badge}</div>
          <div className="text-xs text-gray-400 mt-1 leading-snug">{meta.note}</div>
        </div>

        {/* controls */}
        <div className="rounded-xl bg-gray-900/70 border border-gray-800 p-4 space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => setPlaying((p) => !p)}
              className="px-4 py-2 text-sm font-semibold bg-sky-600 rounded-lg hover:bg-sky-500"
            >
              {playing ? "❚❚ Pause" : "▶ Play"}
            </button>
            <button
              onClick={() => {
                setPlaying(false);
                setStep(0);
              }}
              className="px-4 py-2 text-sm font-medium bg-gray-800 rounded-lg hover:bg-gray-700"
            >
              Reset
            </button>

            <div className="flex items-center gap-2 flex-1 min-w-[240px]">
              <label className="text-xs text-gray-400 whitespace-nowrap">Step</label>
              <input
                type="range"
                min={0}
                max={sim.frames.length - 1}
                value={step}
                onChange={(e) => {
                  setPlaying(false);
                  setStep(Number(e.target.value));
                }}
                className="w-full accent-sky-500"
              />
            </div>

            <label className="flex items-center gap-2 text-xs text-gray-400">
              Example
              <select
                value={example}
                onChange={(e) => setExample(Number(e.target.value))}
                className="bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-gray-200"
              >
                {Array.from({ length: DATASET_SIZE }, (_, i) => (
                  <option key={i} value={i}>
                    #{i + 1}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex items-center gap-2 text-xs text-gray-400">
              Speed
              <select
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                className="bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-gray-200"
              >
                <option value={1}>1×</option>
                <option value={2}>2×</option>
                <option value={4}>4×</option>
              </select>
            </label>
          </div>

          <div className="flex flex-wrap gap-2">
            <Stat label="Step" value={String(step)} />
            <Stat label="Epoch / batch" value={`E${frame.epoch + 1} · B${frame.batchInEpoch + 1}`} />
            <Stat label="Batch loss" value={num(frame.dataBatchLoss ?? frame.batchLoss)} />
            <Stat label="L1 penalty" value={num(frame.l1Penalty ?? 0)} />
            <Stat label="Total loss" value={num(frame.fullLoss)} />
            <Stat label="Final loss" value={num(sim.frames[sim.frames.length - 1].fullLoss)} />
            <Stat label="Grad norm" value={num(frame.gradNorm)} />
          </div>
        </div>

        {/* charts grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card
            title={usesLift ? "Dense lifted input" : "Input vector"}
            sub={
              usesLift
                ? `Example #${example + 1}: the 4-vector is lifted into a dense 16D vector before the scaffold.`
                : `Selected example #${example + 1} of ${DATASET_SIZE}.`
            }
          >
            <Svg html={charts.input} />
          </Card>

          <Card
            title={mode === "nonlinear" ? "Learned weight matrix" : "Scaffold weights"}
            sub={
              mode === "sparse"
                ? "Updated by Adam. Active cells move over training: grown from zero, pruned when near-zero and stable."
                : mode === "dense16"
                ? "All 256 cells trainable — same fixed lift/unlift as the sparse tab, but no sparsity."
                : "Updated by gradients from the current mini-batch."
            }
          >
            <Svg html={charts.weight} />
            {mode === "sparse" && (
              <div className="flex items-center gap-4 text-[11px] text-gray-400 mt-1">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm border-2 border-green-400" /> grown
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm border-2 border-dashed border-red-400" /> pruned
                </span>
                {frame.grown && frame.pruned ? (
                  <span className="text-gray-300 font-mono">
                    +({frame.grown[0]},{frame.grown[1]}) −({frame.pruned[0]},{frame.pruned[1]})
                  </span>
                ) : (
                  <span className="text-gray-600">no swap this step</span>
                )}
              </div>
            )}
          </Card>

          <Card title="Target vector" sub={meta.targetSub}>
            <Svg html={charts.target} />
          </Card>

          <Card title="Gradient & Adam moments" sub="The gradient drives the update; Adam keeps a first-moment (m) and second-moment (v) EMA beside it." full>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Svg html={charts.grad} />
              <Svg html={charts.m1} />
              <Svg html={charts.m2} />
            </div>
          </Card>

          <Card title="Predicted vector" sub="Model output for the selected example at this step.">
            <Svg html={charts.pred} />
          </Card>

          <Card title="Mini-batch losses" sub={meta.batchSub}>
            <Svg html={charts.batch} />
          </Card>

          <Card
            title="Loss over training"
            sub="Blue: full-dataset loss. Red: current mini-batch objective. Sparse mode includes the L1 penalty."
            full
          >
            <Svg html={charts.loss} />
          </Card>
        </div>
      </div>
    </div>
  );
}
