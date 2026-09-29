const CLASS_COLORS = { 0: "#55c2ff", 1: "#ff5c7a" };
const TEXT = "#c8d8ea";
const GRID = "rgba(126,166,210,.12)";

function classPoints(result, target) {
  const output = { x: [], y: [], scores: [] };
  result.points.labels.forEach((label, index) => {
    if (label === target) {
      output.x.push(result.points.x[index]);
      output.y.push(result.points.y[index]);
      output.scores.push(result.points.scores[index]);
    }
  });
  return output;
}

function supportPoints(result) {
  return result.points.supportIndices.reduce(
    (output, index) => {
      output.x.push(result.points.x[index]);
      output.y.push(result.points.y[index]);
      output.scores.push(result.points.scores[index]);
      return output;
    },
    { x: [], y: [], scores: [] },
  );
}

function contourLine(result, level, name, color, dash, width) {
  return {
    type: "contour",
    x: result.grid.x,
    y: result.grid.y,
    z: result.grid.z,
    contours: { start: level, end: level, size: 1, coloring: "lines" },
    line: { color, width, dash },
    colorscale: [[0, color], [1, color]],
    showscale: false,
    hoverinfo: "skip",
    name,
  };
}

const axis2d = (title) => ({
  title,
  color: TEXT,
  gridcolor: GRID,
  zerolinecolor: "rgba(126,166,210,.22)",
  linecolor: "rgba(126,166,210,.25)",
  mirror: true,
});

export function make2DPlot(result, view) {
  const traces = [];
  if (view.heatmap) {
    traces.push({
      type: "contour",
      x: result.grid.x,
      y: result.grid.y,
      z: result.grid.z,
      colorscale: [[0, "#092b55"], [0.49, "#153b54"], [0.5, "#39213a"], [1, "#571a32"]],
      contours: { coloring: "heatmap", showlines: false },
      opacity: 0.74,
      showscale: false,
      hoverinfo: "skip",
      name: "Decision regions",
    });
  }
  if (view.margins) {
    traces.push(contourLine(result, -1, "Margin −1", "rgba(255,255,255,.58)", "dot", 1.5));
    traces.push(contourLine(result, 1, "Margin +1", "rgba(255,255,255,.58)", "dot", 1.5));
  }
  if (view.boundary) {
    traces.push(contourLine(result, 0, "f(x,y) = 0", "#f8e16c", "solid", 3));
  }

  [0, 1].forEach((target) => {
    const points = classPoints(result, target);
    const label = target === 0 ? "Class −1" : "Class +1";
    traces.push({
      type: "scatter",
      mode: "markers",
      x: points.x,
      y: points.y,
      customdata: points.scores,
      name: label,
      marker: {
        size: 7,
        color: CLASS_COLORS[target],
        opacity: 0.94,
        line: { color: "rgba(255,255,255,.72)", width: 0.8 },
      },
      hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{customdata:.3f}<extra>" + label + "</extra>",
    });
  });

  if (view.support) {
    const support = supportPoints(result);
    traces.push({
      type: "scatter",
      mode: "markers",
      x: support.x,
      y: support.y,
      name: `Support vectors (${support.x.length})`,
      marker: { size: 14, color: "rgba(0,0,0,0)", symbol: "circle", line: { color: "#f8e16c", width: 2 } },
      hoverinfo: "skip",
    });
  }

  return {
    data: traces,
    layout: {
      height: 320,
      margin: { l: 48, r: 18, t: 20, b: 42 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(5,14,28,.58)",
      font: { color: TEXT, family: "Inter, Segoe UI, sans-serif", size: 12 },
      xaxis: axis2d("Feature X₁"),
      yaxis: { ...axis2d("Feature X₂"), scaleanchor: "x", scaleratio: 1 },
      legend: { orientation: "h", yanchor: "bottom", y: 1.01, xanchor: "right", x: 1, bgcolor: "rgba(4,12,24,.55)" },
      hovermode: "closest",
      uirevision: "svm-2d",
    },
  };
}

function matrix(rows, columns, value) {
  return Array.from({ length: rows }, () => Array(columns).fill(value));
}

function wireframe(result) {
  const traces = [];
  const rows = result.grid.y.length;
  const columns = result.grid.x.length;
  const step = Math.max(4, Math.floor(rows / 14));
  for (let row = 0; row < rows; row += step) {
    traces.push({
      type: "scatter3d", mode: "lines", x: result.grid.x,
      y: Array(columns).fill(result.grid.y[row]), z: result.grid.z[row],
      line: { color: "rgba(218,235,255,.22)", width: 1 }, hoverinfo: "skip", showlegend: false,
    });
  }
  for (let column = 0; column < columns; column += step) {
    traces.push({
      type: "scatter3d", mode: "lines", x: Array(rows).fill(result.grid.x[column]),
      y: result.grid.y, z: result.grid.z.map((row) => row[column]),
      line: { color: "rgba(218,235,255,.18)", width: 1 }, hoverinfo: "skip", showlegend: false,
    });
  }
  return traces;
}

export function make3DPlot(result, view) {
  const traces = [];
  const rows = result.grid.y.length;
  const columns = result.grid.x.length;
  const zSpan = result.grid.max - result.grid.min;
  const floor = result.grid.min - Math.max(0.3, zSpan * 0.13);

  if (view.projection) {
    traces.push({
      type: "surface", x: result.grid.x, y: result.grid.y,
      z: matrix(rows, columns, floor), surfacecolor: result.grid.z,
      colorscale: view.colorscale, cmin: result.grid.min, cmax: result.grid.max,
      opacity: 0.34, showscale: false, hoverinfo: "skip", name: "2D projection",
    });
  }
  if (view.zeroPlane) {
    traces.push({
      type: "surface", x: result.grid.x, y: result.grid.y,
      z: matrix(rows, columns, 0), surfacecolor: matrix(rows, columns, 0),
      colorscale: [[0, "#f8e16c"], [1, "#f8e16c"]],
      opacity: 0.13, showscale: false, hoverinfo: "skip", name: "f = 0 plane",
    });
  }
  if (view.surface) {
    traces.push({
      type: "surface", x: result.grid.x, y: result.grid.y, z: result.grid.z,
      colorscale: view.colorscale, opacity: view.opacity,
      cmin: result.grid.min, cmax: result.grid.max,
      colorbar: { title: { text: "f(x,y)", font: { color: TEXT } }, thickness: 12, len: 0.72, tickfont: { color: TEXT } },
      lighting: { ambient: 0.72, diffuse: 0.8, roughness: 0.6, specular: 0.18 },
      contours: { z: { show: true, usecolormap: false, color: "rgba(255,255,255,.2)", width: 1 } },
      name: "Decision surface",
    });
  }
  if (view.wireframe) traces.push(...wireframe(result));

  [0, 1].forEach((target) => {
    const points = classPoints(result, target);
    const label = target === 0 ? "Class −1" : "Class +1";
    traces.push({
      type: "scatter3d", mode: "markers", x: points.x, y: points.y, z: points.scores,
      name: label,
      marker: { size: 4.4, color: CLASS_COLORS[target], opacity: 0.96, line: { color: "#07111f", width: 0.8 } },
      hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{z:.3f}<extra>" + label + "</extra>",
    });
  });
  if (view.support) {
    const support = supportPoints(result);
    traces.push({
      type: "scatter3d", mode: "markers", x: support.x, y: support.y, z: support.scores,
      name: `Support vectors (${support.x.length})`, hoverinfo: "skip",
      marker: { size: 7.5, color: "rgba(0,0,0,0)", symbol: "circle-open", line: { color: "#f8e16c", width: 2.2 } },
    });
  }

  const radius = 2.25 / view.zoom;
  const azimuth = (view.azimuth * Math.PI) / 180;
  const elevation = (view.elevation * Math.PI) / 180;
  const eye = {
    x: radius * Math.cos(elevation) * Math.cos(azimuth),
    y: radius * Math.cos(elevation) * Math.sin(azimuth),
    z: radius * Math.sin(elevation),
  };
  const sceneAxis = {
    backgroundcolor: "rgba(3,10,21,.72)", gridcolor: GRID,
    zerolinecolor: "rgba(126,166,210,.28)", color: TEXT, showspikes: false,
  };

  return {
    data: traces,
    layout: {
      height: 620,
      margin: { l: 0, r: 0, t: 6, b: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      font: { color: TEXT, family: "Inter, Segoe UI, sans-serif" },
      scene: {
        xaxis: { ...sceneAxis, title: "Feature X₁" },
        yaxis: { ...sceneAxis, title: "Feature X₂" },
        zaxis: { ...sceneAxis, title: "Decision score  f(x,y)" },
        camera: { eye },
        aspectmode: "manual",
        aspectratio: { x: 1.12, y: 1, z: 0.72 },
      },
      legend: { x: 0.01, y: 0.98, bgcolor: "rgba(4,12,24,.68)" },
      uirevision: `camera-${view.azimuth}-${view.elevation}-${view.zoom}`,
    },
  };
}
