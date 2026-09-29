"""Numerical SVM engine shared by the Vercel FastAPI service."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC


def _split_count(total: int, class_0_ratio: float) -> tuple[int, int]:
    n0 = max(2, int(round(total * class_0_ratio)))
    return n0, max(2, total - n0)


def _cluster_samples(rng, centers, counts, spread):
    chunks = [rng.normal(center, spread, size=(count, 2)) for center, count in zip(centers, counts)]
    return np.vstack(chunks)


def generate_dataset(params):
    rng = np.random.default_rng(params.seed)
    n0, n1 = _split_count(params.points, params.class_ratio)

    if params.dataset == "rings":
        angle0 = rng.uniform(0, 2 * np.pi, n0)
        angle1 = rng.uniform(0, 2 * np.pi, n1)
        radius0 = params.inner_radius * np.sqrt(rng.uniform(0.08, 1.0, n0))
        ring_width = max(0.18, 0.34 * params.separation)
        radius1 = rng.uniform(
            params.outer_radius - ring_width,
            params.outer_radius + ring_width,
            n1,
        )
        x0 = np.column_stack((radius0 * np.cos(angle0), radius0 * np.sin(angle0)))
        x1 = np.column_stack((radius1 * np.cos(angle1), radius1 * np.sin(angle1)))
        X = np.vstack((x0, x1))
        X += rng.normal(0, params.noise, X.shape)
        y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]

    elif params.dataset == "moons":
        X, y = make_moons(
            n_samples=(n0, n1),
            noise=params.noise * 0.72,
            random_state=params.seed,
        )
        X = (X - X.mean(axis=0)) * params.separation

    elif params.dataset == "xor":
        n00, n01 = _split_count(n0, 0.5)
        n10, n11 = _split_count(n1, 0.5)
        distance = params.separation
        spread = 0.28 + params.noise * 0.85
        x0 = _cluster_samples(
            rng,
            [(-distance, -distance), (distance, distance)],
            [n00, n01],
            spread,
        )
        x1 = _cluster_samples(
            rng,
            [(-distance, distance), (distance, -distance)],
            [n10, n11],
            spread,
        )
        X = np.vstack((x0, x1))
        y = np.r_[np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)]

    else:
        distance = 0.72 * params.separation
        spread = 0.42 + params.noise * 0.8
        x0 = rng.normal((-distance, -distance * 0.55), spread, size=(n0, 2))
        x1 = rng.normal((distance, distance * 0.55), spread, size=(n1, 2))
        X = np.vstack((x0, x1))
        y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]

    flip_count = int(round(len(y) * params.label_noise))
    if flip_count:
        indices = rng.choice(len(y), size=flip_count, replace=False)
        y[indices] = 1 - y[indices]

    order = rng.permutation(len(y))
    return X[order], y[order]


def _make_grid(X, model, resolution):
    span = np.ptp(X, axis=0)
    pad = np.maximum(span * 0.18, 0.55)
    x_axis = np.linspace(X[:, 0].min() - pad[0], X[:, 0].max() + pad[0], resolution)
    y_axis = np.linspace(X[:, 1].min() - pad[1], X[:, 1].max() + pad[1], resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    zz = model.decision_function(grid).reshape(xx.shape)
    return x_axis, y_axis, zz


def _metrics(model, X, y, zz):
    scores = model.decision_function(X)
    accuracy = float(np.mean(model.predict(X) == y))
    margin_ratio = float(np.mean(np.abs(scores) <= 1.0))
    regions = zz >= 0
    transitions = np.count_nonzero(regions[:, 1:] != regions[:, :-1])
    transitions += np.count_nonzero(regions[1:, :] != regions[:-1, :])
    complexity = min(100.0, transitions / max(1, zz.shape[0] * 1.6))
    return scores, {
        "accuracy": accuracy,
        "supportVectors": int(len(model.support_)),
        "supportRatio": float(len(model.support_) / len(X)),
        "marginRatio": margin_ratio,
        "complexity": float(complexity),
    }


def _teaching_notes(params, metrics):
    notes = []
    if params.kernel == "rbf" and params.gamma < 0.2:
        notes.append({"tag": "LOW GAMMA", "text": "單點影響範圍較廣，邊界傾向平滑，可能低擬合。"})
    elif params.kernel == "rbf" and params.gamma > 3:
        notes.append({"tag": "HIGH GAMMA", "text": "單點影響範圍很窄，邊界更彎曲，也更容易追著雜訊跑。"})
    if params.C < 1:
        notes.append({"tag": "SOFT MARGIN", "text": "模型容許較多錯分，換取更寬、較穩健的 margin。"})
    elif params.C > 20:
        notes.append({"tag": "HARDER MARGIN", "text": "模型強烈懲罰錯分；訓練準確率高不代表泛化一定更好。"})
    if params.kernel == "poly":
        notes.append({"tag": f"DEGREE {params.degree}", "text": "多項式階數越高，能形成的曲線越複雜。"})
    if params.label_noise > 0.08:
        notes.append({"tag": "LABEL NOISE", "text": "目前有明顯標籤翻轉；比較小 C 與大 C，最容易看出正規化作用。"})
    if metrics["supportRatio"] > 0.55:
        notes.append({"tag": "MANY SUPPORT VECTORS", "text": "超過一半資料成為 support vectors，邊界需要依賴較多樣本。"})
    if not notes:
        notes.append({"tag": "BALANCED SETTING", "text": "移動 C 或 gamma，觀察黃色邊界與 margin 如何改變。"})
    return notes[:3]


def build_model_response(params):
    X, y = generate_dataset(params)
    model = SVC(
        kernel=params.kernel,
        C=params.C,
        gamma=params.gamma,
        degree=params.degree,
        coef0=params.coef0,
        class_weight="balanced" if params.balanced else None,
    )
    model.fit(X, y)
    x_axis, y_axis, zz = _make_grid(X, model, params.resolution)
    scores, metrics = _metrics(model, X, y, zz)

    return {
        "points": {
            "x": X[:, 0].tolist(),
            "y": X[:, 1].tolist(),
            "labels": y.astype(int).tolist(),
            "scores": scores.tolist(),
            "supportIndices": model.support_.astype(int).tolist(),
        },
        "grid": {
            "x": x_axis.tolist(),
            "y": y_axis.tolist(),
            "z": zz.tolist(),
            "min": float(zz.min()),
            "max": float(zz.max()),
        },
        "metrics": metrics,
        "notes": _teaching_notes(params, metrics),
    }
