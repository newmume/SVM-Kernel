"use client";

import { useEffect, useMemo, useState } from "react";

import PlotlyChart from "./plotly-chart";
import { make2DPlot, make3DPlot } from "../lib/plots";


const DEFAULT_MODEL = {
  dataset: "rings",
  kernel: "rbf",
  C: 10,
  gamma: 1,
  degree: 3,
  coef0: 0,
  balanced: false,
  points: 160,
  noise: 0.1,
  label_noise: 0,
  class_ratio: 0.45,
  separation: 1.25,
  inner_radius: 0.9,
  outer_radius: 2.05,
  seed: 7,
  resolution: 80,
};

const DEFAULT_VIEW = {
  heatmap: true,
  boundary: true,
  margins: true,
  support: true,
  surface: true,
  wireframe: false,
  projection: true,
  zeroPlane: true,
  opacity: 0.82,
  colorscale: "Turbo",
  azimuth: -42,
  elevation: 27,
  zoom: 1,
};

const DATASET_LABELS = {
  rings: "同心圓環",
  moons: "交錯月牙",
  xor: "XOR 四象限",
  linear: "線性可分",
};

const KERNEL_LABELS = {
  linear: "Linear",
  poly: "Polynomial",
  rbf: "RBF · Gaussian",
  sigmoid: "Sigmoid",
};

const KERNEL_FORMULAS = {
  linear: "K(x,z) = xᵀz",
  poly: "K(x,z) = (γxᵀz + r)ᵈ",
  rbf: "K(x,z) = exp(−γ‖x−z‖²)",
  sigmoid: "K(x,z) = tanh(γxᵀz + r)",
};

function SectionLabel({ children }) {
  return <div className="section-label">{children}</div>;
}

function SelectControl({ label, value, options, onChange }) {
  return (
    <label className="control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map(([optionValue, optionLabel]) => (
          <option value={optionValue} key={optionValue}>{optionLabel}</option>
        ))}
      </select>
    </label>
  );
}

function RangeControl({ label, value, min, max, step, onChange, format = (item) => item }) {
  return (
    <label className="control range-control">
      <span><span>{label}</span><b>{format(value)}</b></span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <small><span>{min}</span><span>{max}</span></small>
    </label>
  );
}

function ToggleControl({ label, checked, onChange }) {
  return (
    <label className="toggle-control">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <i aria-hidden="true" />
    </label>
  );
}

function Metric({ label, value, detail }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </div>
  );
}

function InfoCard({ kicker, children, warning = false }) {
  return (
    <div className={`info-card${warning ? " warning-card" : ""}`}>
      <div className="info-kicker">{kicker}</div>
      {children}
    </div>
  );
}

function downloadCsv(result, seed) {
  if (!result) return;
  const supports = new Set(result.points.supportIndices);
  const header = "x1,x2,label,decision_score,is_support_vector";
  const rows = result.points.x.map((x, index) => [
    x,
    result.points.y[index],
    result.points.labels[index],
    result.points.scores[index],
    supports.has(index),
  ].join(","));
  const blob = new Blob(["\ufeff" + [header, ...rows].join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `svm-demo-seed-${seed}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function Dashboard() {
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [view, setView] = useState(DEFAULT_VIEW);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const modelKey = useMemo(() => JSON.stringify(model), [model]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setLoading(true);
      setError("");
      try {
        const response = await fetch("/api/model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(model),
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Model computation failed.");
        setResult(payload);
      } catch (requestError) {
        if (requestError.name !== "AbortError") setError(requestError.message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 280);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [modelKey]);

  const plot2d = useMemo(() => result ? make2DPlot(result, view) : null, [result, view]);
  const plot3d = useMemo(() => result ? make3DPlot(result, view) : null, [result, view]);
  const updateModel = (key, value) => setModel((current) => ({ ...current, [key]: value }));
  const updateView = (key, value) => setView((current) => ({ ...current, [key]: value }));

  const reset = () => {
    setModel({ ...DEFAULT_MODEL });
    setView({ ...DEFAULT_VIEW });
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><span>◈</span></div>
          <div><b>SVM Kernel Lab</b><small>INTERACTIVE · VERCEL V2</small></div>
        </div>
        <p className="sidebar-copy">即時模型控制台 · 每次調整都會重新訓練 sklearn SVC</p>
        <div className="sidebar-actions">
          <button type="button" onClick={reset}>↺ Reset</button>
          <button type="button" onClick={() => updateModel("seed", Math.floor(Math.random() * 10000))}>⤨ Random</button>
        </div>

        <SectionLabel>MODEL PARAMETERS</SectionLabel>
        <SelectControl
          label="Kernel function"
          value={model.kernel}
          onChange={(value) => updateModel("kernel", value)}
          options={Object.entries(KERNEL_LABELS)}
        />
        <SelectControl
          label="C · regularization"
          value={String(model.C)}
          onChange={(value) => updateModel("C", Number(value))}
          options={[0.1, 0.3, 1, 3, 10, 30, 100].map((value) => [String(value), String(value)])}
        />
        {model.kernel !== "linear" && (
          <SelectControl
            label="Gamma γ · influence radius"
            value={String(model.gamma)}
            onChange={(value) => updateModel("gamma", Number(value))}
            options={[0.01, 0.03, 0.1, 0.3, 1, 3, 10].map((value) => [String(value), String(value)])}
          />
        )}
        {model.kernel === "poly" && (
          <RangeControl label="Degree · polynomial order" value={model.degree} min={2} max={6} step={1} onChange={(value) => updateModel("degree", value)} />
        )}
        {(model.kernel === "poly" || model.kernel === "sigmoid") && (
          <RangeControl label="Coef0 · independent term" value={model.coef0} min={-2} max={2} step={0.1} onChange={(value) => updateModel("coef0", value)} format={(value) => value.toFixed(1)} />
        )}
        <ToggleControl label="Class weight · balanced" checked={model.balanced} onChange={(value) => updateModel("balanced", value)} />

        <div className="divider" />
        <SectionLabel>DATA GENERATOR</SectionLabel>
        <SelectControl
          label="Dataset geometry"
          value={model.dataset}
          onChange={(value) => updateModel("dataset", value)}
          options={Object.entries(DATASET_LABELS)}
        />
        <RangeControl label="Number of points" value={model.points} min={40} max={400} step={20} onChange={(value) => updateModel("points", value)} />
        <RangeControl label="Feature noise" value={model.noise} min={0} max={0.5} step={0.01} onChange={(value) => updateModel("noise", value)} format={(value) => value.toFixed(2)} />
        <RangeControl label="Label noise · flipped labels" value={model.label_noise} min={0} max={0.25} step={0.01} onChange={(value) => updateModel("label_noise", value)} format={(value) => value.toFixed(2)} />
        <RangeControl label="Class −1 ratio" value={model.class_ratio} min={0.2} max={0.8} step={0.05} onChange={(value) => updateModel("class_ratio", value)} format={(value) => `${Math.round(value * 100)}%`} />
        <RangeControl label="Separation / scale" value={model.separation} min={0.6} max={2} step={0.05} onChange={(value) => updateModel("separation", value)} format={(value) => value.toFixed(2)} />
        {model.dataset === "rings" && (
          <>
            <RangeControl label="Inner cluster radius" value={model.inner_radius} min={0.4} max={1.3} step={0.05} onChange={(value) => updateModel("inner_radius", value)} format={(value) => value.toFixed(2)} />
            <RangeControl label="Outer ring radius" value={model.outer_radius} min={1.4} max={3} step={0.05} onChange={(value) => updateModel("outer_radius", value)} format={(value) => value.toFixed(2)} />
          </>
        )}
        <label className="control">
          <span>Random seed</span>
          <input type="number" min="0" max="9999" value={model.seed} onChange={(event) => updateModel("seed", Math.max(0, Math.min(9999, Number(event.target.value))))} />
        </label>

        <details>
          <summary>◫ View options</summary>
          <div className="detail-content">
            <RangeControl label="Grid resolution" value={model.resolution} min={50} max={120} step={10} onChange={(value) => updateModel("resolution", value)} />
            {[['heatmap', 'Decision heatmap'], ['boundary', 'Decision boundary'], ['margins', 'Margin lines ±1'], ['support', 'Support vectors'], ['surface', '3D decision surface'], ['wireframe', '3D wireframe'], ['projection', '2D floor projection'], ['zeroPlane', 'f(x,y) = 0 plane']].map(([key, label]) => (
              <ToggleControl key={key} label={label} checked={view[key]} onChange={(value) => updateView(key, value)} />
            ))}
            <RangeControl label="Surface opacity" value={view.opacity} min={0.2} max={1} step={0.05} onChange={(value) => updateView("opacity", value)} format={(value) => value.toFixed(2)} />
            <SelectControl label="Surface palette" value={view.colorscale} onChange={(value) => updateView("colorscale", value)} options={['Turbo', 'RdBu', 'Viridis', 'Plasma', 'Cividis'].map((value) => [value, value])} />
          </div>
        </details>
        <details>
          <summary>⌁ 3D camera</summary>
          <div className="detail-content">
            <RangeControl label="Azimuth" value={view.azimuth} min={-180} max={180} step={1} onChange={(value) => updateView("azimuth", value)} />
            <RangeControl label="Elevation" value={view.elevation} min={8} max={70} step={1} onChange={(value) => updateView("elevation", value)} />
            <RangeControl label="Zoom" value={view.zoom} min={0.65} max={1.55} step={0.05} onChange={(value) => updateView("zoom", value)} format={(value) => `${value.toFixed(2)}×`} />
          </div>
        </details>
      </aside>

      <main className="workspace">
        <header className="hero">
          <div>
            <div className="eyebrow">NONLINEAR CLASSIFICATION / LIVE DECISION FUNCTION</div>
            <h1>SVM Kernel Trick · 3D Explorer</h1>
            <p>從 2D 決策邊界到 3D decision function，一次看懂 C、γ、margin 與 support vectors 的連動。</p>
          </div>
          <div className="hero-actions">
            <span className={`live-pill ${error ? "offline" : ""}`}><i />{error ? " API ERROR" : loading ? " COMPUTING" : " MODEL ONLINE"}</span>
            <button type="button" onClick={() => downloadCsv(result, model.seed)} disabled={!result}>⇩ Export CSV</button>
          </div>
        </header>

        {error && <div className="error-banner"><b>模型計算失敗</b><span>{error}</span></div>}

        {result ? (
          <>
            <section className="metric-grid">
              <Metric label="TRAINING ACCURACY" value={`${(result.metrics.accuracy * 100).toFixed(1)}%`} />
              <Metric label="SUPPORT VECTORS" value={`${result.metrics.supportVectors} / ${result.points.x.length}`} detail={`${Math.round(result.metrics.supportRatio * 100)}% of data`} />
              <Metric label="MARGIN REGION" value={`${(result.metrics.marginRatio * 100).toFixed(1)}%`} detail="| f(x) | ≤ 1" />
              <Metric label="BOUNDARY COMPLEXITY" value={`${Math.round(result.metrics.complexity)} / 100`} />
            </section>

            <section className="chart-shell chart-2d">
              <div className="chart-heading"><span>2D DATA · INPUT SPACE</span><small>YELLOW = f(x,y) 0 · DOTTED = MARGINS ±1</small></div>
              <div className="plot-wrap short"><PlotlyChart data={plot2d.data} layout={plot2d.layout} /></div>
              {loading && <div className="chart-refresh">Updating model…</div>}
            </section>

            <section className="lower-grid">
              <div className="chart-shell chart-3d">
                <div className="chart-heading"><span>3D DECISION FUNCTION · MODEL SPACE</span><small>DRAG ROTATE · SCROLL ZOOM · DOUBLE-CLICK RESET</small></div>
                <div className="plot-wrap tall"><PlotlyChart data={plot3d.data} layout={plot3d.layout} /></div>
                {loading && <div className="chart-refresh">Updating model…</div>}
              </div>

              <aside className="insight-column">
                <InfoCard kicker="MODEL STATUS">
                  <div className="info-row"><span>Kernel</span><b>{model.kernel.toUpperCase()}</b></div>
                  <div className="info-row"><span>C</span><b>{model.C}</b></div>
                  <div className="info-row"><span>Gamma γ</span><b>{model.kernel === "linear" ? "N/A" : model.gamma}</b></div>
                  <div className="info-row"><span>Dataset</span><b>{DATASET_LABELS[model.dataset]}</b></div>
                  <div className="info-row"><span>Seed</span><b>{model.seed}</b></div>
                </InfoCard>
                <InfoCard kicker="DECISION FUNCTION">
                  <p className="formula">f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b</p>
                  <p className="class-rule"><span className="positive">Class +1</span> if f(x) &gt; 0<br /><span className="negative">Class −1</span> if f(x) &lt; 0</p>
                </InfoCard>
                <InfoCard kicker="KERNEL FUNCTION"><p className="formula kernel-formula">{KERNEL_FORMULAS[model.kernel]}</p></InfoCard>
                {result.notes.map((note) => <InfoCard kicker={note.tag} key={`${note.tag}-${note.text}`}><p>{note.text}</p></InfoCard>)}
                <InfoCard kicker="IMPORTANT NOTE" warning>
                  <p>此 3D 圖是 decision function <b>f(x,y)</b> 的高度，不是 RBF 真正的特徵空間。RBF 對應高維、甚至無限維映射。</p>
                </InfoCard>
              </aside>
            </section>

            <details className="teaching-guide">
              <summary>教學導覽：怎麼用這個 Demo？</summary>
              <div className="guide-grid">
                <div><b>01 · 先改 Gamma</b><p>選 RBF，從 0.1 → 1 → 10。看黃色邊界從平滑變得貼近每個資料點。</p></div>
                <div><b>02 · 再比較 C</b><p>加入 label noise，再比較 C = 0.3 與 C = 100，觀察錯分、margin 與 overfitting。</p></div>
                <div><b>03 · 換 Kernel</b><p>在圓環資料切到 Linear，直接看出線性模型限制；再用 RBF 恢復非線性邊界。</p></div>
              </div>
            </details>
          </>
        ) : (
          <div className="initial-loading"><span /><b>Initializing sklearn model</b><small>FastAPI is preparing the first decision surface…</small></div>
        )}

        <footer>SVM Kernel Lab V2 · Next.js + FastAPI + scikit-learn + Plotly · 所有指標皆由目前資料即時計算</footer>
      </main>
    </div>
  );
}
