"""SVM Kernel Lab V2 — a technology-styled Streamlit teaching dashboard."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from svm_engine import (
    decision_grid,
    fit_svm,
    generate_dataset,
    make_2d_figure,
    make_3d_figure,
    model_metrics,
    teaching_notes,
)


st.set_page_config(
    page_title="SVM Kernel Lab · V2",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULTS = {
    "dataset": "同心圓環",
    "kernel": "rbf",
    "C": 10.0,
    "gamma": 1.0,
    "degree": 3,
    "coef0": 0.0,
    "balanced": False,
    "points": 160,
    "noise": 0.10,
    "label_noise": 0.00,
    "class_ratio": 0.45,
    "separation": 1.25,
    "inner_radius": 0.90,
    "outer_radius": 2.05,
    "seed": 7,
    "resolution": 90,
    "show_heatmap": True,
    "show_boundary": True,
    "show_margins": True,
    "show_support": True,
    "show_surface": True,
    "show_wireframe": False,
    "show_projection": True,
    "show_zero_plane": True,
    "surface_opacity": 0.82,
    "colorscale": "Turbo",
    "azimuth": -42,
    "elevation": 27,
    "zoom": 1.00,
}


CSS = r"""
<style>
:root {
  --ink: #eaf4ff; --muted: #8ca6bf; --line: rgba(118, 180, 232, .18);
  --cyan: #43d7ff; --violet: #8b7cff; --panel: rgba(7, 18, 35, .74);
}
.stApp {
  color: var(--ink);
  background:
    radial-gradient(circle at 78% 0%, rgba(43, 89, 156, .20), transparent 32rem),
    radial-gradient(circle at 24% 100%, rgba(79, 45, 140, .14), transparent 28rem),
    linear-gradient(135deg, #030812 0%, #071321 52%, #040a13 100%);
}
.stApp:before {
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.25;
  background-image: linear-gradient(rgba(80,170,220,.04) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(80,170,220,.04) 1px, transparent 1px);
  background-size:32px 32px;
}
[data-testid="stSidebar"] { background: rgba(4, 12, 24, .94); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1680px; }
h1, h2, h3 { letter-spacing: -.025em; }
.brand { display:flex; gap:12px; align-items:center; margin-bottom:.35rem; }
.brand-mark { width:38px; height:38px; border:1px solid rgba(67,215,255,.6); transform:rotate(45deg);
  display:grid; place-items:center; box-shadow:0 0 24px rgba(67,215,255,.18); }
.brand-mark span { transform:rotate(-45deg); color:#62ddff; font-size:18px; }
.brand-copy b { font-size:1.02rem; letter-spacing:.04em; }.brand-copy small{display:block;color:var(--muted);font-size:.68rem;letter-spacing:.18em}
.eyebrow { color:#55d9ff; font-size:.68rem; letter-spacing:.22em; font-weight:700; margin-bottom:.25rem; }
.hero-title { font-size:clamp(1.55rem,3vw,2.35rem); line-height:1.08; font-weight:750; margin:0; }
.hero-sub { color:var(--muted); margin:.42rem 0 0; max-width:780px; font-size:.91rem; }
.live-pill { display:inline-flex; align-items:center; gap:7px; border:1px solid rgba(86,214,255,.24); border-radius:999px;
  padding:5px 10px; color:#9ae9ff; font-size:.68rem; letter-spacing:.08em; background:rgba(45,160,205,.08); }
.live-dot {width:6px;height:6px;border-radius:50%;background:#40efb1;box-shadow:0 0 11px #40efb1;}
.section-label { color:#9bb2c9; font-size:.68rem; letter-spacing:.17em; font-weight:700; margin:.15rem 0 .45rem; }
.chart-shell { border:1px solid var(--line); border-radius:14px; background:linear-gradient(150deg,rgba(9,23,43,.82),rgba(5,13,26,.68));
  box-shadow:0 16px 44px rgba(0,0,0,.18), inset 0 1px rgba(255,255,255,.025); padding:9px 12px 2px; }
.chart-title { display:flex;align-items:center;justify-content:space-between;font-weight:650;font-size:.88rem;margin:2px 3px 0; }
.chart-title small{font-weight:500;color:#6f8da7;font-size:.65rem;letter-spacing:.08em}
[data-testid="stMetric"] { border:1px solid var(--line); border-radius:12px; padding:11px 14px;
  background:linear-gradient(145deg,rgba(11,29,52,.78),rgba(5,14,27,.72)); box-shadow:inset 3px 0 #24bfe8; }
[data-testid="stMetricLabel"] { color:#86a1bb; font-size:.72rem; letter-spacing:.06em; }
[data-testid="stMetricValue"] { color:#edf8ff; font-size:1.42rem; }
.info-card { border:1px solid var(--line); border-radius:12px; background:rgba(7,18,35,.68); padding:14px 15px; margin-bottom:10px; }
.info-card .kicker {font-size:.61rem;color:#54d8ff;letter-spacing:.17em;font-weight:750;margin-bottom:7px}
.info-row{display:flex;justify-content:space-between;gap:12px;color:#92aac1;font-size:.78rem;padding:4px 0;border-bottom:1px solid rgba(116,167,209,.08)}
.info-row:last-child{border:0}.info-row b{color:#e7f4ff;font-weight:600}.note{color:#afc3d6;font-size:.76rem;line-height:1.55;margin:0}
.warning-note{border-left:2px solid #f7d65a;padding-left:10px;color:#b8cadd;font-size:.73rem;line-height:1.52}
.stButton > button, .stDownloadButton > button { border:1px solid rgba(80,195,235,.25); background:rgba(11,35,58,.72); color:#dff7ff; border-radius:9px; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#4ddaff; color:white; box-shadow:0 0 18px rgba(77,218,255,.12); }
div[data-testid="stExpander"] { border:1px solid var(--line); background:rgba(6,17,32,.55); border-radius:10px; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background:rgba(8,23,41,.72); border-color:rgba(110,169,215,.24); }
hr { border-color:var(--line); }
#MainMenu, footer {visibility:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def reset_controls():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def randomize_data():
    st.session_state["seed"] = random.randint(0, 9999)


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


@st.cache_data(show_spinner=False)
def compute_model(dataset, points, noise, label_noise, class_ratio, separation,
                  inner_radius, outer_radius, seed, kernel, C, gamma, degree,
                  coef0, balanced, resolution):
    X, y = generate_dataset(
        dataset, points, noise, label_noise, class_ratio, separation,
        inner_radius, outer_radius, seed,
    )
    model = fit_svm(X, y, kernel, C, gamma, degree, coef0, balanced)
    xx, yy, zz = decision_grid(X, model, resolution)
    return X, y, model, xx, yy, zz


with st.sidebar:
    st.markdown("""
    <div class="brand"><div class="brand-mark"><span>◈</span></div>
    <div class="brand-copy"><b>SVM Kernel Lab</b><small>INTERACTIVE · V2</small></div></div>
    """, unsafe_allow_html=True)
    st.caption("即時模型控制台 · 所有控制項皆會重新計算模型")

    action_a, action_b = st.columns(2)
    action_a.button("↺ Reset", width="stretch", on_click=reset_controls)
    action_b.button("⤨ Random", width="stretch", on_click=randomize_data)

    st.markdown('<div class="section-label">MODEL PARAMETERS</div>', unsafe_allow_html=True)
    st.selectbox("Kernel function", ["linear", "poly", "rbf", "sigmoid"], key="kernel",
                 format_func={"linear": "Linear", "poly": "Polynomial", "rbf": "RBF · Gaussian", "sigmoid": "Sigmoid"}.get)
    st.select_slider("C · regularization", options=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0], key="C")
    if st.session_state.kernel != "linear":
        st.select_slider("Gamma γ · influence radius", options=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], key="gamma")
    if st.session_state.kernel == "poly":
        st.slider("Degree · polynomial order", 2, 6, key="degree")
    if st.session_state.kernel in ("poly", "sigmoid"):
        st.slider("Coef0 · independent term", -2.0, 2.0, step=0.1, key="coef0")
    st.toggle("Class weight · balanced", key="balanced", help="類別不平衡時，自動提高少數類別的權重。")

    st.divider()
    st.markdown('<div class="section-label">DATA GENERATOR</div>', unsafe_allow_html=True)
    st.selectbox("Dataset geometry", ["同心圓環", "交錯月牙", "XOR 四象限", "線性可分"], key="dataset")
    st.slider("Number of points", 40, 400, step=20, key="points")
    st.slider("Feature noise", 0.0, 0.5, step=0.01, key="noise")
    st.slider("Label noise · flipped labels", 0.0, 0.25, step=0.01, key="label_noise")
    st.slider("Class −1 ratio", 0.20, 0.80, step=0.05, key="class_ratio")
    st.slider("Separation / scale", 0.60, 2.00, step=0.05, key="separation")
    if st.session_state.dataset == "同心圓環":
        st.slider("Inner cluster radius", 0.40, 1.30, step=0.05, key="inner_radius")
        st.slider("Outer ring radius", 1.40, 3.00, step=0.05, key="outer_radius")
        if st.session_state.outer_radius <= st.session_state.inner_radius + 0.2:
            st.warning("外圈半徑需明顯大於內圈；目前資料可能高度重疊。")
    st.number_input("Random seed", min_value=0, max_value=9999, step=1, key="seed")

    st.divider()
    with st.expander("◫ View options", expanded=False):
        st.slider("Grid resolution", 50, 140, step=10, key="resolution")
        st.toggle("Decision heatmap", key="show_heatmap")
        st.toggle("Decision boundary", key="show_boundary")
        st.toggle("Margin lines ±1", key="show_margins")
        st.toggle("Support vectors", key="show_support")
        st.toggle("3D decision surface", key="show_surface")
        st.toggle("3D wireframe", key="show_wireframe")
        st.toggle("2D floor projection", key="show_projection")
        st.toggle("f(x,y) = 0 plane", key="show_zero_plane")
        st.slider("Surface opacity", 0.20, 1.00, step=0.05, key="surface_opacity")
        st.selectbox("Surface palette", ["Turbo", "RdBu", "Viridis", "Plasma", "Cividis"], key="colorscale")
    with st.expander("⌁ 3D camera", expanded=False):
        st.slider("Azimuth", -180, 180, key="azimuth")
        st.slider("Elevation", 8, 70, key="elevation")
        st.slider("Zoom", 0.65, 1.55, step=0.05, key="zoom")


X, y, model, xx, yy, zz = compute_model(
    st.session_state.dataset, st.session_state.points, st.session_state.noise,
    st.session_state.label_noise, st.session_state.class_ratio, st.session_state.separation,
    st.session_state.inner_radius, st.session_state.outer_radius, st.session_state.seed,
    st.session_state.kernel, st.session_state.C, st.session_state.gamma,
    st.session_state.degree, st.session_state.coef0, st.session_state.balanced,
    st.session_state.resolution,
)
metrics = model_metrics(model, X, y, zz)


head_left, head_right = st.columns([5, 1.15], vertical_alignment="center")
with head_left:
    st.markdown('<div class="eyebrow">NONLINEAR CLASSIFICATION / LIVE DECISION FUNCTION</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">SVM Kernel Trick · 3D Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">從 2D 決策邊界到 3D decision function，一次看懂 C、γ、margin 與 support vectors 的連動。</p>', unsafe_allow_html=True)
with head_right:
    st.markdown('<div class="live-pill"><span class="live-dot"></span> MODEL ONLINE</div>', unsafe_allow_html=True)
    export_df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y,
                              "decision_score": model.decision_function(X),
                              "is_support_vector": False})
    export_df.loc[model.support_, "is_support_vector"] = True
    st.download_button("⇩ Export CSV", export_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"svm_demo_seed_{st.session_state.seed}.csv", mime="text/csv",
                       width="stretch")

st.write("")
m1, m2, m3, m4 = st.columns(4)
m1.metric("TRAINING ACCURACY", f"{metrics['accuracy']:.1%}")
m2.metric("SUPPORT VECTORS", f"{metrics['support_vectors']} / {len(X)}", f"{metrics['support_ratio']:.0%} of data", delta_color="off")
m3.metric("MARGIN REGION", f"{metrics['margin_ratio']:.1%}", "| f(x) | ≤ 1", delta_color="off")
m4.metric("BOUNDARY COMPLEXITY", f"{metrics['complexity']:.0f} / 100")

st.write("")
st.markdown('<div class="chart-shell"><div class="chart-title"><span>2D DATA · INPUT SPACE</span><small>YELLOW = f(x,y) 0 · DOTTED = MARGINS ±1</small></div>', unsafe_allow_html=True)
fig_2d = make_2d_figure(
    X, y, model, xx, yy, zz,
    st.session_state.show_heatmap, st.session_state.show_boundary,
    st.session_state.show_margins, st.session_state.show_support,
)
st.plotly_chart(fig_2d, width="stretch", config={"displaylogo": False, "scrollZoom": True})
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
plot_col, info_col = st.columns([4.3, 1.15], gap="medium")
with plot_col:
    st.markdown('<div class="chart-shell"><div class="chart-title"><span>3D DECISION FUNCTION · MODEL SPACE</span><small>DRAG ROTATE · SCROLL ZOOM · DOUBLE-CLICK RESET</small></div>', unsafe_allow_html=True)
    fig_3d = make_3d_figure(
        X, y, model, xx, yy, zz,
        st.session_state.colorscale, st.session_state.surface_opacity,
        st.session_state.show_surface, st.session_state.show_wireframe,
        st.session_state.show_support, st.session_state.show_projection,
        st.session_state.show_zero_plane, st.session_state.azimuth,
        st.session_state.elevation, st.session_state.zoom,
    )
    st.plotly_chart(fig_3d, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    st.markdown('</div>', unsafe_allow_html=True)

with info_col:
    gamma_text = f"{st.session_state.gamma:g}" if st.session_state.kernel != "linear" else "N/A"
    st.markdown(f"""
    <div class="info-card"><div class="kicker">MODEL STATUS</div>
      <div class="info-row"><span>Kernel</span><b>{st.session_state.kernel.upper()}</b></div>
      <div class="info-row"><span>C</span><b>{st.session_state.C:g}</b></div>
      <div class="info-row"><span>Gamma γ</span><b>{gamma_text}</b></div>
      <div class="info-row"><span>Dataset</span><b>{st.session_state.dataset}</b></div>
      <div class="info-row"><span>Seed</span><b>{st.session_state.seed}</b></div>
    </div>
    <div class="info-card"><div class="kicker">DECISION FUNCTION</div>
      <p class="note">f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b</p>
      <p class="note" style="margin-top:9px"><span style="color:#ff6b87">Class +1</span> if f(x) &gt; 0<br>
      <span style="color:#65caff">Class −1</span> if f(x) &lt; 0</p>
    </div>
    """, unsafe_allow_html=True)

    """
    if st.session_state.kernel == "rbf":
        st.markdown(r"$K(x,z)=exp(-gamma lVert x-zVert^2)$")
    elif st.session_state.kernel == "poly":
        st.markdown(r"$K(x,z)=(gamma x^Tz+r)^d$")
    elif st.session_state.kernel == "sigmoid":
        st.markdown(r"$K(x,z)=\tanh(gamma x^Tz+r)$")
    else:
        st.markdown(r"$K(x,z)=x^Tz$")

    """
    if st.session_state.kernel == "rbf":
        st.latex(r"K(x,z)=\exp(-\gamma \lVert x-z\rVert^2)")
    elif st.session_state.kernel == "poly":
        st.latex(r"K(x,z)=(\gamma x^Tz+r)^d")
    elif st.session_state.kernel == "sigmoid":
        st.latex(r"K(x,z)=\tanh(\gamma x^Tz+r)")
    else:
        st.latex(r"K(x,z)=x^Tz")

    for tag, note in teaching_notes(
        st.session_state.kernel, st.session_state.C, st.session_state.gamma,
        st.session_state.degree, st.session_state.label_noise, metrics,
    ):
        st.markdown(f'<div class="info-card"><div class="kicker">{tag}</div><p class="note">{note}</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card"><div class="kicker">IMPORTANT NOTE</div>
      <div class="warning-note">此 3D 圖是 decision function <b>f(x,y)</b> 的高度，不是 RBF 真正的特徵空間。RBF 對應的是高維、甚至無限維映射。</div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("教學導覽：怎麼用這個 Demo？"):
    guide_a, guide_b, guide_c = st.columns(3)
    guide_a.markdown("**01 · 先改 Gamma**\n\n選 RBF，從 `0.1 → 1 → 10`。看黃色邊界從平滑變得貼近每個資料點。")
    guide_b.markdown("**02 · 再比較 C**\n\n加入 label noise，再比較 `C = 0.3` 與 `C = 100`。觀察錯分、margin 與 overfitting。")
    guide_c.markdown("**03 · 換 Kernel**\n\n在圓環資料切到 Linear，直接看出線性模型的限制；再用 RBF 恢復非線性邊界。")
    st.info("教學用映射 z = x² + y² 能直觀說明『升維後線性可分』；本頁的 3D surface 則是實際 SVC 的 decision function，兩者概念相關但不是同一件事。")

st.caption("SVM Kernel Lab V2 · scikit-learn SVC + Plotly · 本頁所有模型指標皆由目前資料即時計算")
