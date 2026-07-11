"use client";

import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import MatrixGrid from "@/components/MatrixGrid";
import {
  NetParams,
  buildEpoch,
  randomParams,
  LAYOUT,
  NODE_META,
  Phase,
} from "@/lib/nnDemo";

const PHASE_COLOR: Record<Phase, string> = {
  forward: "#22c55e",
  backward: "#f59e0b",
  update: "#38bdf8",
};

export default function NNPage() {
  const [net, setNet] = useState<NetParams>(() => randomParams(1234));
  const [lr, setLr] = useState(0.5);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600); // ms per frame
  const [autoLoop, setAutoLoop] = useState(true);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [epochCount, setEpochCount] = useState(0);

  const epoch = useMemo(() => buildEpoch(net, lr), [net, lr]);
  const frames = epoch.frames;
  const frame = frames[Math.min(step, frames.length - 1)];

  // record starting loss once per net
  useEffect(() => {
    setLossHistory((h) => (h.length === 0 ? [epoch.loss] : h));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const commitEpoch = useCallback(() => {
    setNet((prev) => ({ ...prev, ...epoch.updated }));
    setLossHistory((h) => [...h, buildEpoch({ ...net, ...epoch.updated }, lr).loss]);
    setEpochCount((c) => c + 1);
    setStep(0);
  }, [epoch.updated, net, lr]);

  // playback loop
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!playing) return;
    timer.current = setTimeout(() => {
      setStep((s) => {
        if (s < frames.length - 1) return s + 1;
        // reached the end (the update frame)
        if (autoLoop) {
          commitEpoch();
          return 0;
        }
        setPlaying(false);
        return s;
      });
    }, speed);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [playing, step, speed, frames.length, autoLoop, commitEpoch]);

  const reset = () => {
    setPlaying(false);
    setNet(randomParams(Math.floor(1 + Math.random() * 1e6)));
    setLossHistory([]);
    setEpochCount(0);
    setStep(0);
  };

  const dataShown = new Set(frame.dataShown);
  const gradShown = new Set(frame.gradShown);
  const active = new Set(frame.active);

  return (
    <div className="flex h-screen bg-gray-950 text-white overflow-hidden">
      {/* ---------------- sidebar ---------------- */}
      <div className="w-80 flex-shrink-0 p-4 space-y-5 overflow-y-auto border-r border-gray-800">
        <div>
          <h1 className="text-lg font-bold">Autograd NN Playground</h1>
          <p className="text-xs text-gray-400 mt-1">
            A 2-layer net, one operation at a time: forward → backward → step.
          </p>
          <div className="flex gap-3">
            <Link href="/" className="text-xs text-sky-400 hover:underline">
              ← matmul
            </Link>
            <Link href="/train" className="text-xs text-sky-400 hover:underline">
              training demo →
            </Link>
          </div>
        </div>

        {/* phase legend */}
        <div className="space-y-1.5">
          {(["forward", "backward", "update"] as Phase[]).map((p) => (
            <div key={p} className="flex items-center gap-2 text-xs">
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ background: PHASE_COLOR[p] }}
              />
              <span
                className={frame.phase === p ? "font-semibold" : "text-gray-400"}
              >
                {p === "forward"
                  ? "Forward pass"
                  : p === "backward"
                  ? "Backward pass"
                  : "Gradient descent"}
              </span>
            </div>
          ))}
        </div>

        {/* color legend */}
        <div className="space-y-1">
          <span className="text-xs font-medium">Value colour</span>
          <div className="h-3 rounded" style={{
            background: "linear-gradient(90deg, rgb(55,105,225), rgb(128,128,128), rgb(225,55,55))",
          }} />
          <div className="flex justify-between text-[10px] text-gray-400">
            <span>− negative</span>
            <span>0</span>
            <span>+ positive</span>
          </div>
          <p className="text-[10px] text-gray-500">
            Normalized per matrix by max |value| (linear).
          </p>
        </div>

        {/* controls */}
        <div className="space-y-3">
          <div className="flex gap-2">
            <button
              onClick={() => setStep((s) => Math.max(0, s - 1))}
              className="flex-1 text-sm px-2 py-1.5 bg-gray-800 rounded hover:bg-gray-700"
            >
              ◀ Prev
            </button>
            <button
              onClick={() => setPlaying((p) => !p)}
              className="flex-1 text-sm px-2 py-1.5 bg-sky-600 rounded hover:bg-sky-500 font-medium"
            >
              {playing ? "❚❚ Pause" : "▶ Play"}
            </button>
            <button
              onClick={() =>
                setStep((s) => {
                  if (s < frames.length - 1) return s + 1;
                  commitEpoch();
                  return 0;
                })
              }
              className="flex-1 text-sm px-2 py-1.5 bg-gray-800 rounded hover:bg-gray-700"
            >
              Next ▶
            </button>
          </div>

          <label className="block text-xs">
            Speed: {speed} ms/op
            <input
              type="range"
              min={120}
              max={1600}
              step={20}
              value={1720 - speed}
              onChange={(e) => setSpeed(1720 - Number(e.target.value))}
              className="w-full accent-sky-500"
            />
          </label>

          <label className="block text-xs">
            Learning rate η: {lr}
            <input
              type="range"
              min={0.05}
              max={2}
              step={0.05}
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
              className="w-full accent-sky-500"
            />
          </label>

          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={autoLoop}
              onChange={(e) => setAutoLoop(e.target.checked)}
              className="accent-sky-500"
            />
            Loop into next epoch automatically
          </label>

          <div className="flex gap-2">
            <button
              onClick={commitEpoch}
              className="flex-1 text-xs px-2 py-1.5 bg-gray-800 rounded hover:bg-gray-700"
            >
              Apply step → new epoch
            </button>
            <button
              onClick={reset}
              className="flex-1 text-xs px-2 py-1.5 bg-gray-800 rounded hover:bg-gray-700"
            >
              Randomize
            </button>
          </div>
        </div>

        {/* loss history */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="font-medium">Loss</span>
            <span className="font-mono text-amber-300">{epoch.loss.toFixed(5)}</span>
          </div>
          <LossSparkline history={lossHistory} />
          <div className="text-[10px] text-gray-500">
            epoch {epochCount} · {lossHistory.length} recorded
          </div>
        </div>
      </div>

      {/* ---------------- main board ---------------- */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* step banner */}
        <div
          className="px-5 py-3 border-b border-gray-800 flex items-center gap-4"
          style={{ borderLeft: `4px solid ${PHASE_COLOR[frame.phase]}` }}
        >
          <div className="flex-1">
            <div className="text-sm font-semibold" style={{ color: PHASE_COLOR[frame.phase] }}>
              {frame.title}
            </div>
            <div className="text-xs text-gray-300">{frame.detail}</div>
          </div>
          <div className="text-xs text-gray-500 font-mono whitespace-nowrap">
            op {step + 1}/{frames.length}
          </div>
        </div>

        {/* timeline scrubber */}
        <div className="px-5 py-2 border-b border-gray-800">
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={step}
            onChange={(e) => {
              setPlaying(false);
              setStep(Number(e.target.value));
            }}
            className="w-full accent-sky-500"
          />
        </div>

        {/* graph */}
        <div className="flex-1 overflow-auto p-6">
          <div className="flex items-start gap-3 min-w-max">
            {LAYOUT.map((column, ci) => (
              <div key={ci} className="flex items-center gap-3">
                <div className="flex flex-col gap-4">
                  {column.map((label) => (
                    <NodeCard
                      key={label}
                      label={label}
                      data={frame.data[label]}
                      grad={frame.grad[label]}
                      showData={dataShown.has(label)}
                      showGrad={gradShown.has(label) && frame.phase !== "forward"}
                      active={active.has(label)}
                      phase={frame.phase}
                    />
                  ))}
                </div>
                {ci < LAYOUT.length - 1 && (
                  <div className="text-gray-600 text-xl">→</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function NodeCard({
  label,
  data,
  grad,
  showData,
  showGrad,
  active,
  phase,
}: {
  label: string;
  data: number[][];
  grad: number[][];
  showData: boolean;
  showGrad: boolean;
  active: boolean;
  phase: Phase;
}) {
  const meta = NODE_META[label] ?? { title: label, sub: "" };
  if (!data) return null;
  const ring = active ? PHASE_COLOR[phase] : "transparent";

  return (
    <div
      className="rounded-lg bg-gray-900 border border-gray-800 p-2.5 transition-all duration-200"
      style={{
        boxShadow: active ? `0 0 0 2px ${ring}, 0 0 18px ${ring}66` : "none",
      }}
    >
      <div className="flex items-baseline justify-between mb-1.5 gap-3">
        <span className="text-sm font-semibold">{meta.title}</span>
        <span className="text-[10px] text-gray-500">{meta.sub}</span>
      </div>
      <MatrixGrid m={data} dim={!showData} />
      {showGrad && (
        <div className="mt-2">
          <div className="text-[10px] text-amber-400/80 mb-1">∇ (gradient)</div>
          <MatrixGrid m={grad} variant="grad" />
        </div>
      )}
    </div>
  );
}

function LossSparkline({ history }: { history: number[] }) {
  if (history.length < 2) {
    return (
      <div className="h-16 rounded bg-gray-900 border border-gray-800 flex items-center justify-center text-[10px] text-gray-600">
        train to see the curve
      </div>
    );
  }
  const w = 260;
  const h = 64;
  const max = Math.max(...history);
  const min = Math.min(...history);
  const range = max - min || 1;
  const pts = history
    .map((v, i) => {
      const x = (i / (history.length - 1)) * (w - 6) + 3;
      const y = h - 4 - ((v - min) / range) * (h - 8);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="rounded bg-gray-900 border border-gray-800">
      <polyline points={pts} fill="none" stroke="#f59e0b" strokeWidth={1.5} />
    </svg>
  );
}
