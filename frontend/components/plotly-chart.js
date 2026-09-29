"use client";

import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js").then((module) => module.default), {
  ssr: false,
  loading: () => <div className="plot-loading"><span /> Rendering visualization…</div>,
});

export default function PlotlyChart({ data, layout, className }) {
  return (
    <Plot
      data={data}
      layout={{ ...layout, autosize: true }}
      config={{ displaylogo: false, responsive: true, scrollZoom: true }}
      className={className}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  );
}
