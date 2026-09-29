"""Computation and Plotly chart helpers for the SVM V2 classroom demo."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.svm import SVC


CLASS_COLORS = {0: "#55C2FF", 1: "#FF5C7A"}
GRID_COLOR = "rgba(126, 166, 210, 0.12)"
TEXT_COLOR = "#C8D8EA"


def _split_count(total: int, class_0_ratio: float) -> tuple[int, int]:
    n0 = max(2, int(round(total * class_0_ratio)))
    return n0, max(2, total - n0)


def _cluster_samples(rng, centers, counts, spread):
    chunks = [rng.normal(center, spread, size=(count, 2)) for center, count in zip(centers, counts)]
    return np.vstack(chunks)


def generate_dataset(
    dataset: str,
    n_points: int,
    noise: float,
    label_noise: float,
    class_0_ratio: float,
    separation: float,
    inner_radius: float,
    outer_radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one of four demo datasets with consistent controls."""
    rng = np.random.default_rng(seed)
    n0, n1 = _split_count(n_points, class_0_ratio)

    if dataset == "同心圓環":
        a0 = rng.uniform(0, 2 * np.pi, n0)
        a1 = rng.uniform(0, 2 * np.pi, n1)
        r0 = inner_radius * np.sqrt(rng.uniform(0.08, 1.0, n0))
        ring_width = max(0.18, 0.34 * separation)
        r1 = rng.uniform(outer_radius - ring_width, outer_radius + ring_width, n1)
        x0 = np.column_stack((r0 * np.cos(a0), r0 * np.sin(a0)))
        x1 = np.column_stack((r1 * np.cos(a1), r1 * np.sin(a1)))
        X = np.vstack((x0, x1))
        X += rng.normal(0, noise, X.shape)
        y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]

    elif dataset == "交錯月牙":
        X, y = make_moons(n_samples=(n0, n1), noise=noise * 0.72, random_state=seed)
        X = (X - X.mean(axis=0)) * separation

    elif dataset == "XOR 四象限":
        n00, n01 = _split_count(n0, 0.5)
        n10, n11 = _split_count(n1, 0.5)
        d = separation
        spread = 0.28 + noise * 0.85
        x0 = _cluster_samples(rng, [(-d, -d), (d, d)], [n00, n01], spread)
        x1 = _cluster_samples(rng, [(-d, d), (d, -d)], [n10, n11], spread)
        X = np.vstack((x0, x1))
        y = np.r_[np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)]

    else:  # 線性可分
        d = 0.72 * separation
        spread = 0.42 + noise * 0.8
        x0 = rng.normal((-d, -d * 0.55), spread, size=(n0, 2))
        x1 = rng.normal((d, d * 0.55), spread, size=(n1, 2))
        X = np.vstack((x0, x1))
        y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]

    flip_count = int(round(len(y) * label_noise))
    if flip_count:
        flip_indices = rng.choice(len(y), size=flip_count, replace=False)
        y[flip_indices] = 1 - y[flip_indices]

    order = rng.permutation(len(y))
    return X[order], y[order]


def fit_svm(X, y, kernel, C, gamma, degree, coef0, class_weight):
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        class_weight="balanced" if class_weight else None,
    )
    model.fit(X, y)
    return model


def decision_grid(X, model, resolution):
    span = np.ptp(X, axis=0)
    pad = np.maximum(span * 0.18, 0.55)
    x_axis = np.linspace(X[:, 0].min() - pad[0], X[:, 0].max() + pad[0], resolution)
    y_axis = np.linspace(X[:, 1].min() - pad[1], X[:, 1].max() + pad[1], resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    zz = model.decision_function(grid).reshape(xx.shape)
    return xx, yy, zz


def model_metrics(model, X, y, zz):
    prediction = model.predict(X)
    accuracy = float(np.mean(prediction == y))
    ambiguous = float(np.mean(np.abs(model.decision_function(X)) <= 1.0))
    regions = zz >= 0
    transitions = np.count_nonzero(regions[:, 1:] != regions[:, :-1])
    transitions += np.count_nonzero(regions[1:, :] != regions[:-1, :])
    complexity = min(100.0, transitions / max(1, zz.shape[0] * 1.6))
    return {
        "accuracy": accuracy,
        "support_vectors": int(len(model.support_)),
        "support_ratio": len(model.support_) / len(X),
        "margin_ratio": ambiguous,
        "complexity": complexity,
    }


def _axis_style(title):
    return dict(
        title=title,
        color=TEXT_COLOR,
        gridcolor=GRID_COLOR,
        zerolinecolor="rgba(126, 166, 210, 0.22)",
        linecolor="rgba(126, 166, 210, 0.25)",
        mirror=True,
    )


def make_2d_figure(
    X,
    y,
    model,
    xx,
    yy,
    zz,
    show_heatmap,
    show_boundary,
    show_margins,
    show_support,
):
    fig = go.Figure()

    if show_heatmap:
        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=zz,
                colorscale=[[0, "#092B55"], [0.49, "#153B54"], [0.5, "#39213A"], [1, "#571A32"]],
                contours=dict(coloring="heatmap", showlines=False),
                opacity=0.72,
                showscale=False,
                hoverinfo="skip",
                name="Decision regions",
            )
        )

    if show_margins:
        for level, name in [(-1, "Margin −1"), (1, "Margin +1")]:
            fig.add_trace(
                go.Contour(
                    x=xx[0], y=yy[:, 0], z=zz,
                    contours=dict(start=level, end=level, size=1),
                    line=dict(color="rgba(255,255,255,.55)", width=1.5, dash="dot"),
                    showscale=False, hoverinfo="skip", name=name,
                )
            )

    if show_boundary:
        fig.add_trace(
            go.Contour(
                x=xx[0], y=yy[:, 0], z=zz,
                contours=dict(start=0, end=0, size=1),
                line=dict(color="#F8E16C", width=3),
                showscale=False, hoverinfo="skip", name="f(x,y) = 0",
            )
        )

    for cls, label in [(0, "Class −1"), (1, "Class +1")]:
        mask = y == cls
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers", name=label,
                marker=dict(size=7, color=CLASS_COLORS[cls], opacity=0.92,
                            line=dict(color="rgba(255,255,255,.72)", width=0.8)),
                customdata=np.column_stack((model.decision_function(X[mask]),)),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{customdata[0]:.3f}<extra>" + label + "</extra>",
            )
        )

    if show_support:
        sv = model.support_vectors_
        fig.add_trace(
            go.Scatter(
                x=sv[:, 0], y=sv[:, 1], mode="markers",
                marker=dict(size=13, color="rgba(0,0,0,0)", symbol="circle",
                            line=dict(color="#F8E16C", width=2)),
                name=f"Support vectors ({len(sv)})",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=315,
        margin=dict(l=24, r=18, t=22, b=26),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,14,28,.58)",
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, sans-serif", size=12),
        xaxis=_axis_style("Feature X₁"),
        yaxis={**_axis_style("Feature X₂"), "scaleanchor": "x", "scaleratio": 1},
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(4,12,24,.55)"),
        hovermode="closest",
        uirevision="svm-2d",
    )
    return fig


def _wireframe_traces(xx, yy, zz):
    traces = []
    step = max(4, xx.shape[0] // 14)
    for i in range(0, xx.shape[0], step):
        traces.append(go.Scatter3d(x=xx[i], y=yy[i], z=zz[i], mode="lines",
                                   line=dict(color="rgba(218,235,255,.22)", width=1),
                                   hoverinfo="skip", showlegend=False))
    for j in range(0, xx.shape[1], step):
        traces.append(go.Scatter3d(x=xx[:, j], y=yy[:, j], z=zz[:, j], mode="lines",
                                   line=dict(color="rgba(218,235,255,.18)", width=1),
                                   hoverinfo="skip", showlegend=False))
    return traces


def make_3d_figure(
    X,
    y,
    model,
    xx,
    yy,
    zz,
    colorscale,
    surface_opacity,
    show_surface,
    show_wireframe,
    show_support,
    show_projection,
    show_zero_plane,
    azimuth,
    elevation,
    zoom,
):
    fig = go.Figure()
    z_min, z_max = float(zz.min()), float(zz.max())

    if show_projection:
        floor = z_min - max(0.3, (z_max - z_min) * 0.13)
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=np.full_like(zz, floor), surfacecolor=zz,
                colorscale=colorscale, cmin=z_min, cmax=z_max,
                opacity=0.34, showscale=False, hoverinfo="skip", name="2D projection",
            )
        )

    if show_zero_plane:
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=np.zeros_like(zz),
                surfacecolor=np.zeros_like(zz), colorscale=[[0, "#F8E16C"], [1, "#F8E16C"]],
                opacity=0.13, showscale=False, hoverinfo="skip", name="f = 0 plane",
            )
        )

    if show_surface:
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=zz, colorscale=colorscale,
                opacity=surface_opacity, cmin=z_min, cmax=z_max,
                colorbar=dict(title=dict(text="f(x,y)", font=dict(color=TEXT_COLOR)),
                              thickness=12, len=0.72, tickfont=dict(color=TEXT_COLOR)),
                lighting=dict(ambient=0.72, diffuse=0.8, roughness=0.6, specular=0.18),
                contours=dict(z=dict(show=True, usecolormap=False, color="rgba(255,255,255,.2)", width=1)),
                name="Decision surface",
            )
        )

    if show_wireframe:
        for trace in _wireframe_traces(xx, yy, zz):
            fig.add_trace(trace)

    scores = model.decision_function(X)
    for cls, label in [(0, "Class −1"), (1, "Class +1")]:
        mask = y == cls
        fig.add_trace(
            go.Scatter3d(
                x=X[mask, 0], y=X[mask, 1], z=scores[mask], mode="markers", name=label,
                marker=dict(size=4.4, color=CLASS_COLORS[cls], opacity=0.96,
                            line=dict(color="#07111F", width=0.8)),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{z:.3f}<extra>" + label + "</extra>",
            )
        )

    if show_support:
        sv = model.support_vectors_
        sv_scores = model.decision_function(sv)
        fig.add_trace(
            go.Scatter3d(
                x=sv[:, 0], y=sv[:, 1], z=sv_scores, mode="markers",
                marker=dict(size=7.5, color="rgba(0,0,0,0)", symbol="circle",
                            line=dict(color="#F8E16C", width=2.2)),
                name=f"Support vectors ({len(sv)})", hoverinfo="skip",
            )
        )

    radius = 2.25 / zoom
    az = math.radians(azimuth)
    el = math.radians(elevation)
    eye = dict(x=radius * math.cos(el) * math.cos(az),
               y=radius * math.cos(el) * math.sin(az),
               z=radius * math.sin(el))

    scene_axis = dict(backgroundcolor="rgba(3,10,21,.72)", gridcolor=GRID_COLOR,
                      zerolinecolor="rgba(126,166,210,.28)", color=TEXT_COLOR, showspikes=False)
    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, sans-serif"),
        scene=dict(
            xaxis={**scene_axis, "title": "Feature X₁"},
            yaxis={**scene_axis, "title": "Feature X₂"},
            zaxis={**scene_axis, "title": "Decision score  f(x,y)"},
            camera=dict(eye=eye),
            aspectmode="manual", aspectratio=dict(x=1.12, y=1.0, z=0.72),
        ),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(4,12,24,.68)"),
        uirevision=f"camera-{azimuth}-{elevation}-{zoom}",
    )
    return fig


def teaching_notes(kernel, C, gamma, degree, label_noise, metrics):
    notes = []
    if kernel == "rbf" and gamma < 0.2:
        notes.append(("LOW GAMMA", "單點影響範圍較廣，邊界傾向平滑，可能低擬合。"))
    elif kernel == "rbf" and gamma > 3:
        notes.append(("HIGH GAMMA", "單點影響範圍很窄，邊界更彎曲，也更容易追著雜訊跑。"))
    if C < 1:
        notes.append(("SOFT MARGIN", "模型容許較多錯分，換取更寬、較穩健的 margin。"))
    elif C > 20:
        notes.append(("HARDER MARGIN", "模型強烈懲罰錯分，訓練準確率高不代表泛化一定更好。"))
    if kernel == "poly":
        notes.append((f"DEGREE {degree}", "多項式階數越高，能形成的曲線越複雜；請同時觀察 support vectors。"))
    if label_noise > 0.08:
        notes.append(("LABEL NOISE", "目前有明顯標籤翻轉；比較小 C 與大 C，最容易看出正規化作用。"))
    if metrics["support_ratio"] > 0.55:
        notes.append(("MANY SUPPORT VECTORS", "超過一半資料成為 support vectors，代表邊界需要依賴較多樣本。"))
    if not notes:
        notes.append(("BALANCED SETTING", "目前參數落在中段。移動 C 或 gamma，觀察黃色邊界與 margin 如何改變。"))
    return notes[:3]
