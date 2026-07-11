"use client";

import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import {
  buildSimulations,
  forwardModel,
  maxAbs,
  flatten2D,
  MODE_META,
  PARAM_COUNT,
  LOWRANK_R,
  DATASET_SIZE,
  Mode,
  Simulation,
} from "@/lib/sparseTraining";
import { renderBarChart, renderSparseHeatmap, renderLossChart } from "@/lib/trainingCharts";

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
  className = "",
}: {
  title: string;
  sub?: string;
  children: React.ReactNode;
  full?: boolean;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl bg-gray-900/70 border border-gray-800 p-4 ${
        full ? "col-span-full" : ""
      } ${className}`}
    >
      <h2 className="text-base font-semibold text-gray-100">{title}</h2>
      {sub && <p className="text-xs text-gray-400 mt-0.5 mb-3 leading-snug">{sub}</p>}
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
  const [mode, setMode] = useState<Mode>("sparse");
  const [steps, setSteps] = useState<Record<Mode, number>>({ dense16: 0, sparse: 0, lowrank: 0 });
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

  const charts = useMemo(() => {
    const dim = sim.cfg.dim;
    const x = sim.X[example];
    const y = sim.targets[example];
    const pred = forwardModel(frame.layers[0].W, frame.layers[1].W, x, sim.cfg).pred;
    const vecMax = maxAbs(y.concat(pred));
    const yL = ["y1", "y2", "y3", "y4"];

    const target = renderBarChart(y, yL, { title: "Target", maxAbs: vecMax, width: 300, height: 210 });
    const predChart = renderBarChart(pred, yL, {
      title: "Predicted",
      maxAbs: vecMax,
      width: 300,
      height: 210,
    });

    const layers = frame.layers.map((L, li) => {
      const active = L.active ?? [];
      const hm = (m: number[][], title: string, markers = false) =>
        renderSparseHeatmap(m, {
          size: dim,
          title,
          maxAbs: maxAbs(flatten2D(m)),
          activePositions: active,
          grown: markers ? L.grown : null,
          pruned: markers ? L.pruned : null,
          width: 300,
          height: 250,
        });
      return {
        sparse: L.active !== undefined,
        grown: L.grown,
        pruned: L.pruned,
        weights: hm(L.W, `W${li + 1} — weights`, true),
        grad: hm(L.grad, `∂L/∂W${li + 1} — gradient`),
        mom: hm(L.m, `momentum (Adam m)`),
      };
    });

    const loss = renderLossChart(
      sim.frames.map((f) => f.fullLoss),
      sim.frames.map((f) => f.batchLoss),
      step,
      { title: "Loss over training" }
    );

    return { target, pred: predChart, layers, loss };
  }, [sim, frame, example, step]);

  const num = (v: number, d = 6) => v.toFixed(d);
  // data loss only (L1 penalty excluded) at the last training step, so the
  // number is comparable across tabs regardless of the L1 term.
  const finalDataLoss = sim.frames[sim.frames.length - 1].dataFullLoss;

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
            <h1 className="text-2xl font-bold tracking-tight">Approximating a Big Dense Net</h1>
            <p className="text-sm text-gray-400 max-w-3xl mt-1 leading-snug">
              The data comes from a randomly-initialised big dense 2-layer network. Three students
              try to reproduce it: the full dense net (realizable reference), a sparse net, and a
              low-rank net — the last two parameter-matched, to see which structure approximates a
              big dense network better at equal budget.
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
          {(["dense16", "sparse", "lowrank"] as Mode[]).map((m) => (
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
              {m === "dense16" ? "Dense 16×16" : m === "sparse" ? "Sparse" : "Low-rank"}
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
            <Stat label="Params" value={String(PARAM_COUNT[mode])} />
            <Stat label="Step" value={String(step)} />
            <Stat label="Epoch / batch" value={`E${frame.epoch + 1} · B${frame.batchInEpoch + 1}`} />
            <Stat label="Batch loss" value={num(frame.dataBatchLoss)} />
            <Stat label="L1 penalty" value={num(frame.l1Penalty)} />
            <Stat label="Total loss" value={num(frame.fullLoss)} />
            <Stat label="Final data loss" value={num(finalDataLoss)} />
            <Stat label="Grad norm" value={num(frame.gradNorm)} />
          </div>
        </div>

        {/* target + predicted, small, side by side */}
        <div className="flex flex-wrap gap-4">
          <Card title="Target vector" sub="Teacher output for the selected example." className="w-72">
            <Svg html={charts.target} />
          </Card>
          <Card
            title="Predicted vector"
            sub="Student output at this training step."
            className="w-72"
          >
            <Svg html={charts.pred} />
          </Card>
        </div>

        {/* whole-network weights / gradient / momentum */}
        <Card
          title="Network weights · gradient · momentum"
          sub="Both layers of the network. Per layer: the weights, the gradient ∂L/∂W, and Adam's momentum (first moment m)."
        >
          <div className="space-y-4">
            {charts.layers.map((L, i) => (
              <div key={i}>
                <div className="flex items-center gap-3 mb-1.5">
                  <span className="text-sm font-semibold text-gray-200">Layer {i + 1}</span>
                  <span className="text-[11px] text-gray-500">
                    {mode === "sparse" ? "sparse" : mode === "lowrank" ? `rank ${LOWRANK_R}` : "dense"}
                  </span>
                  {L.sparse && (
                    <span className="flex items-center gap-3 text-[11px] text-gray-400">
                      <span className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-sm border-2 border-green-400" /> grown
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-sm border-2 border-dashed border-red-400" /> pruned
                      </span>
                      {L.grown && L.pruned ? (
                        <span className="text-gray-300 font-mono">
                          +({L.grown[0]},{L.grown[1]}) −({L.pruned[0]},{L.pruned[1]})
                        </span>
                      ) : (
                        <span className="text-gray-600">no swap this step</span>
                      )}
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <Svg html={L.weights} />
                  <Svg html={L.grad} />
                  <Svg html={L.mom} />
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* loss curve */}
        <Card
          title="Loss over training"
          sub="Blue: full-dataset loss. Red: current mini-batch objective (sparse includes L1)."
        >
          <Svg html={charts.loss} />
        </Card>
      </div>
    </div>
  );
}
