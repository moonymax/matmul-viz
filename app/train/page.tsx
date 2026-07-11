"use client";

import dynamic from "next/dynamic";

// Client-only: the simulation uses transcendental math (tanh/sin) whose last-ULP
// results differ between the Node SSR runtime and the browser, which would cause
// hydration mismatches. Rendering only on the client avoids that entirely.
const TrainingApp = dynamic(() => import("@/components/TrainingApp"), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-gray-950 text-gray-500 flex items-center justify-center">
      Building simulation…
    </div>
  ),
});

export default function TrainPage() {
  return <TrainingApp />;
}
