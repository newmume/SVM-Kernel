import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.data_generator import generate_ring_dataset
from utils.svm_utils import train_svm, make_decision_grid, compute_decision_surface


def teaching_note(gamma, C):
    notes = []
    if gamma < 0.2:
        notes.append("Gamma 值較小：決策邊界較平滑，每個點的影響範圍較廣。")
    if gamma > 3:
        notes.append("Gamma 值較大：決策邊界非常靈活，可能產生過擬合（overfitting）。")
    if C < 1:
        notes.append("C 值較小：模型允許較多錯誤，以換取更寬的邊界（margin）。")
    if C > 20:
        notes.append("C 值較大：模型會更努力地正確分類所有訓練資料，可能導致過擬合。")
    if not notes:
        notes.append("當前參數產生平衡的決策邊界，效果良好。")
    return notes


@st.cache_data
def train_and_compute(kernel, C, gamma, degree, noise, n_points, seed):
    X, y = generate_ring_dataset(
        n_inner=int(n_points * 0.45),
        n_outer=int(n_points * 0.55),
        noise=noise,
        random_seed=seed,
    )
    kwargs = dict(kernel=kernel, C=C, gamma=gamma, degree=degree)
    model = train_svm(X, y, **kwargs)
    pad = 0.5
    x_range = (X[:, 0].min() - pad, X[:, 0].max() + pad)
    y_range = (X[:, 1].min() - pad, X[:, 1].max() + pad)
    xx, yy, grid_points = make_decision_grid(x_range, y_range, resolution=80)
    Z = compute_decision_surface(model, grid_points)
    return X, y, model, xx, yy, Z


def make_2d_plot(X, y, model, xx, yy, Z):
    Z_grid = Z.reshape(xx.shape)
    fig = go.Figure()

    inner = y == 0
    outer = y == 1

    fig.add_trace(go.Scatter(
        x=X[inner, 0], y=X[inner, 1],
        mode="markers",
        marker=dict(color="blue", size=6, line=dict(color="black", width=1)),
        name="類別 0（內圈）",
    ))
    fig.add_trace(go.Scatter(
        x=X[outer, 0], y=X[outer, 1],
        mode="markers",
        marker=dict(color="red", size=6, line=dict(color="black", width=1)),
        name="類別 1（外圈）",
    ))

    contour_levels = [-1, 0, 1]
    contour_colors = ["orange", "yellow", "orange"]
    contour_dash = ["dash", "solid", "dash"]
    for level, color, dash in zip(contour_levels, contour_colors, contour_dash):
        fig.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0],
            z=Z_grid,
            contours=dict(start=level, end=level, size=0.01),
            line=dict(color=color, width=3, dash=dash),
            contours_coloring="lines",
            showscale=False,
            name=f"f(x,y) = {level}",
        ))

    sv = model.support_vectors_
    fig.add_trace(go.Scatter(
        x=sv[:, 0], y=sv[:, 1],
        mode="markers",
        marker=dict(
            color="green",
            size=12,
            symbol="circle-open",
            line=dict(width=3),
        ),
        name=f"支援向量（{len(sv)}）",
    ))

    fig.update_layout(
        title="二維決策邊界",
        xaxis_title="x",
        yaxis_title="y",
        width=500,
        height=500,
        showlegend=True,
    )
    return fig


def make_3d_plot(X, y, model, xx, yy, Z):
    Z_grid = Z.reshape(xx.shape)
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=xx[0], y=yy[:, 0], z=Z_grid,
        colorscale="RdBu",
        opacity=0.75,
        showscale=True,
        colorbar=dict(title="f(x,y)"),
        name="決策曲面",
    ))

    inner = y == 0
    outer = y == 1
    Z_pts = model.decision_function(X)

    fig.add_trace(go.Scatter3d(
        x=X[inner, 0], y=X[inner, 1], z=Z_pts[inner],
        mode="markers",
        marker=dict(color="blue", size=4, line=dict(color="black", width=1)),
        name="類別 0（內圈）",
    ))
    fig.add_trace(go.Scatter3d(
        x=X[outer, 0], y=X[outer, 1], z=Z_pts[outer],
        mode="markers",
        marker=dict(color="red", size=4, line=dict(color="black", width=1)),
        name="類別 1（外圈）",
    ))

    sv = model.support_vectors_
    sv_idx = model.support_
    fig.add_trace(go.Scatter3d(
        x=sv[:, 0], y=sv[:, 1], z=Z_pts[sv_idx],
        mode="markers",
        marker=dict(
            color="green",
            size=8,
            symbol="circle-open",
            line=dict(width=3),
        ),
        name=f"支援向量（{len(sv)}）",
    ))

    fig.update_layout(
        title="三維決策函數曲面",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=600,
        height=500,
    )
    return fig


def page_home():
    st.title("SVM 核技巧 3D 互動展示")
    st.markdown("---")

    col_logo, col_desc = st.columns([1, 2])
    with col_logo:
        st.image("SVM_核函數視覺化指南.png", width=280)
    with col_desc:
        st.markdown("""
        ### 什麼是 SVM 核技巧？
        支援向量機（SVM）是一種強大的機器學習分類演算法。
        **核技巧（Kernel Trick）** 讓 SVM 能夠在**原始低維空間**中，
        透過隱式的特徵映射，學習**非線性決策邊界**。

        ### 教育故事
        1. **二維困境** — 中心藍點與外圈紅點無法用直線分開
        2. **特徵映射** — 使用 $\\phi(x, y) = (x, y, x^2 + y^2)$ 提升到 3D
        3. **超平面** — 在 3D 空間中用水平平面分開資料
        4. **回到 2D** — 投影回二維形成圓形決策邊界
        5. **真實 RBF** — 真正的 RBF 核對應無限維特徵空間
        """)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📐 Phase 1")
        st.markdown("**Manim 動畫**\n\n展示特徵映射 $\\phi(x, y) = (x, y, x^2 + y^2)$ 將 2D 點提升到 3D 的過程，以及超平面如何分離資料。")
        st.code("manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D", language="bash")
    with col2:
        st.subheader("📊 Phase 2")
        st.markdown("**RBF 決策曲面**\n\n使用 sklearn SVC 訓練真實 RBF 核 SVM，繪製二維決策邊界與三維決策函數曲面。")
        st.code("python phase2_rbf_decision_surface.py", language="bash")
        st.image("outputs/rbf_2d_decision_boundary.png", width=250)
    with col3:
        st.subheader("🎮 Phase 3")
        st.markdown("**互動展示**\n\n點擊左側導覽前往「互動展示」頁面，調整 Kernel、C、Gamma 等參數，即時觀察決策邊界變化。")
        st.markdown("---\n👉 **前往左側選單 → 互動展示**")


def page_demo():
    st.title("🎮 SVM 互動展示")
    st.markdown("在左側面板調整參數，觀察決策邊界的即時變化。")

    with st.sidebar:
        st.header("參數設定")
        kernel = st.selectbox("核函數 (Kernel)", ["linear", "poly", "rbf", "sigmoid"], index=2)
        C = st.slider("C（正規化參數）", 0.1, 100.0, 10.0, step=0.1)
        gamma = st.slider("Gamma（核寬度）", 0.01, 10.0, 1.0, step=0.01)
        degree = st.slider("多項式次數 (Degree)", 2, 6, 3, step=1)
        noise = st.slider("雜訊 (Noise)", 0.0, 0.5, 0.08, step=0.01)
        n_points = st.slider("資料點數量", 40, 300, 120, step=10)
        seed = st.number_input("隨機種子", value=7, step=1)

    X, y, model, xx, yy, Z = train_and_compute(
        kernel, C, gamma, degree, noise, n_points, seed,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_2d = make_2d_plot(X, y, model, xx, yy, Z)
        st.plotly_chart(fig_2d, width="stretch")

    with col2:
        fig_3d = make_3d_plot(X, y, model, xx, yy, Z)
        st.plotly_chart(fig_3d, width="stretch")

    with st.expander("教學提示", expanded=True):
        notes = teaching_note(gamma, C)
        for note in notes:
            st.markdown(f"- {note}")
        st.markdown(f"**支援向量數量：** {len(model.support_vectors_)}")
        st.markdown(f"**訓練準確率：** {model.score(X, y):.3f}")

    with st.expander("概念說明與公式"):
        st.markdown(r"""
        **核心問題：** 二維空間中的環狀資料無法用一條直線分開。

        **核心函數（Kernel Function）：**
        - 線性核：$K(x, x') = x \cdot x'$
        - 多項式核：$K(x, x') = (x \cdot x' + 1)^d$
        - RBF 核：$K(x, x') = \exp(-\gamma \|x - x'\|^2)$

        **決策函數：**
        $$f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$$

        **教學用的特徵映射（Phase 1）：**
        $$\phi(x, y) = (x, y, x^2 + y^2)$$
        - 將 2D 點 $(x, y)$ 提升到 3D：$z = x^2 + y^2$
        - 在 3D 特徵空間中，可用水平超平面 $z = c$ 分開資料
        - 投影回 2D 後形成圓形決策邊界 $x^2 + y^2 = c$

        **真正的 RBF 核：**
        - RBF 核對應的是**無限維**特徵空間，並非簡單的 3D 映射
        - 上方的 3D 曲面顯示的是 **決策函數** $z = f(x, y)$，而非特徵空間本身
        - 透過調整 Gamma 和 C 可以觀察到邊界從平滑到過擬合的變化
        """)


def main():
    st.set_page_config(
        page_title="SVM 核技巧 3D 互動展示",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.markdown("# 🧠 SVM 核技巧")
        st.markdown("---")
        page = st.radio(
            "導覽選單",
            ["🏠 首頁", "🎮 互動展示"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("© 2026 SVM-Kernel Demo")

    if "互動展示" in page:
        page_demo()
    else:
        page_home()


if __name__ == "__main__":
    main()
