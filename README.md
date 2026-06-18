# SVM Kernel Trick 3D Interactive Demo

An educational 3‑phase demonstration of the Support Vector Machine kernel trick.

![SVM 核函數視覺化指南](SVM_核函數視覺化指南.png)

## Educational Story

1. **2D Limitation:** Blue points in the center and red points in an outer ring cannot be separated by a straight line.
2. **Feature Mapping:** Using `φ(x, y) = (x, y, x² + y²)`, points are lifted into 3D where they become linearly separable.
3. **Hyperplane:** A horizontal plane separates the lifted points in the 3D feature space.
4. **Back to 2D:** Projecting the hyperplane back gives a circular decision boundary.
5. **Real RBF:** A true RBF SVM decision function surface is shown in Phase 2 and Phase 3.
6. **Interactive:** Phase 3 lets students adjust `C`, `gamma`, kernel, and noise interactively.

## Phase 1: Manim Kernel Trick Animation

Animates the conceptual mapping `z = x² + y²`.

```bash
# Low quality preview
manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D

# High quality render
manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D
```

## Phase 2: Real RBF SVM Decision Surface

Trains an sklearn `SVC(kernel='rbf')` and visualises both the 2D decision boundary and the 3D decision function surface.

```bash
python phase2_rbf_decision_surface.py
```

## Phase 3: Interactive Streamlit Demo

A web app where students explore how kernel, C, gamma, degree, noise, and point count affect the SVM decision boundary.

```bash
streamlit run phase3_streamlit_app.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Important Mathematical Note

The mapping `z = x² + y²` is a **teaching‑only** visualisation. A real RBF kernel corresponds to an infinite‑dimensional feature space. Phase 2 and Phase 3 show the `decision_function` surface, not the feature space itself.
