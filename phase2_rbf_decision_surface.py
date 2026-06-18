import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False
from utils.data_generator import generate_ring_dataset
from utils.svm_utils import train_svm, make_decision_grid, compute_decision_surface


def plot_2d_decision_boundary(ax, X, y, model, xx, yy, Z):
    inner = y == 0
    outer = y == 1
    ax.scatter(X[inner, 0], X[inner, 1], c="blue", edgecolors="black", s=40, label="類別 0（內圈）")
    ax.scatter(X[outer, 0], X[outer, 1], c="red", edgecolors="black", s=40, label="類別 1（外圈）")

    Z_grid = Z.reshape(xx.shape)
    levels = [-1, 0, 1]
    linestyles = ["--", "-", "--"]
    colors = ["orange", "yellow", "orange"]
    for level, ls, col in zip(levels, linestyles, colors):
        ax.contour(
            xx, yy, Z_grid,
            levels=[level],
            linestyles=ls,
            colors=col,
            linewidths=2,
        )

    sv = model.support_vectors_
    ax.scatter(
        sv[:, 0], sv[:, 1],
        s=120,
        facecolors="none",
        edgecolors="green",
        linewidths=2,
        label="支援向量",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("二維決策邊界 f(x,y)=0 與邊界線")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")


def plot_3d_decision_surface(X, y, model, xx, yy, Z):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    Z_grid = Z.reshape(xx.shape)
    surf = ax.plot_surface(
        xx, yy, Z_grid,
        cmap="coolwarm",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    fig.colorbar(surf, ax=ax, label="決策函數值 f(x,y)")

    Z_pts = model.decision_function(X)
    inner = y == 0
    outer = y == 1
    ax.scatter(
        X[inner, 0], X[inner, 1],
        Z_pts[inner],
        c="blue",
        edgecolors="black",
        s=50,
        label="類別 0（內圈）",
    )
    ax.scatter(
        X[outer, 0], X[outer, 1],
        Z_pts[outer],
        c="red",
        edgecolors="black",
        s=50,
        label="類別 1（外圈）",
    )

    sv = model.support_vectors_
    sv_idx = model.support_
    ax.scatter(
        sv[:, 0], sv[:, 1],
        Z_pts[sv_idx],
        s=150,
        facecolors="none",
        edgecolors="green",
        linewidths=2,
        label="支援向量",
    )

    ax.contour(
        xx, yy, Z_grid,
        levels=[0],
        colors="yellow",
        linewidths=3,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("三維決策函數曲面 z = f(x, y)")
    ax.legend(loc="upper right")
    return fig


def main():
    X, y = generate_ring_dataset(random_seed=7, noise=0.08)
    model = train_svm(X, y, kernel="rbf", C=10, gamma=1)

    x_range = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    y_range = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    xx, yy, grid_points = make_decision_grid(x_range, y_range, resolution=80)
    Z = compute_decision_surface(model, grid_points)

    fig_2d, ax_2d = plt.subplots(figsize=(8, 7))
    plot_2d_decision_boundary(ax_2d, X, y, model, xx, yy, Z)
    plt.tight_layout()
    plt.savefig("outputs/rbf_2d_decision_boundary.png", dpi=150)
    print("Saved: outputs/rbf_2d_decision_boundary.png")

    fig_3d = plot_3d_decision_surface(X, y, model, xx, yy, Z)
    plt.tight_layout()
    plt.savefig("outputs/rbf_3d_decision_surface.png", dpi=150)
    print("Saved: outputs/rbf_3d_decision_surface.png")

    print(f"支援向量數量: {len(model.support_vectors_)}")
    print(f"訓練準確率: {model.score(X, y):.3f}")


if __name__ == "__main__":
    main()
