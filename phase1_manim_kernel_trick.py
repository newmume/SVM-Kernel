import sys
import numpy as np
from manim import *
from utils.data_generator import generate_ring_dataset


class SVMKernelTrick3D(ThreeDScene):
    def setup(self):
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)

    def construct(self):
        rng = np.random.default_rng(7)
        X, y = generate_ring_dataset(random_seed=7, noise=0.0)
        inner_mask = y == 0
        outer_mask = y == 1

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-0.5, 6, 1],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        axes.add(axes.get_axis_labels(x_label="x", y_label="y", z_label="z"))
        axes.shift(OUT * 0.5)

        self.wait(0.5)

        title = Text("SVM 核技巧", font_size=48, color=YELLOW)
        subtitle = Text("從 2D 到 3D", font_size=36, color=WHITE)
        subtitle.next_to(title, DOWN)
        title_group = VGroup(title, subtitle)
        self.play(Write(title_group))
        self.wait(1)
        self.play(FadeOut(title_group))
        self.wait(0.3)

        blue_dots_2d = VGroup()
        for x, y_coord in X[inner_mask]:
            dot = Dot3D(point=[x, y_coord, 0], radius=0.06, color=BLUE)
            blue_dots_2d.add(dot)
        red_dots_2d = VGroup()
        for x, y_coord in X[outer_mask]:
            dot = Dot3D(point=[x, y_coord, 0], radius=0.06, color=RED)
            red_dots_2d.add(dot)

        self.play(
            Create(axes),
            LaggedStart(
                *[Create(d) for d in blue_dots_2d],
                lag_ratio=0.02,
            ),
            LaggedStart(
                *[Create(d) for d in red_dots_2d],
                lag_ratio=0.02,
            ),
        )
        text_2d = Text(
            "二維空間中，無法用直線分開藍點與紅點",
            font_size=28,
            color=WHITE,
        )
        text_2d.to_corner(UP + LEFT)
        self.play(Write(text_2d))
        self.wait(1.5)
        self.play(FadeOut(text_2d))

        # Step 3: Show mapping formula
        formula = MathTex(
            r"\phi(x, y) = (x, y, x^2 + y^2)",
            font_size=48,
            color=YELLOW,
        )
        formula.to_corner(UP + LEFT)
        self.play(Write(formula))
        self.wait(1)

        # Step 4: Animate lifting to 3D
        blue_dots_3d = VGroup()
        for x, y_coord in X[inner_mask]:
            z_val = x**2 + y_coord**2
            dot = Dot3D(point=[x, y_coord, z_val], radius=0.06, color=BLUE)
            blue_dots_3d.add(dot)
        red_dots_3d = VGroup()
        for x, y_coord in X[outer_mask]:
            z_val = x**2 + y_coord**2
            dot = Dot3D(point=[x, y_coord, z_val], radius=0.06, color=RED)
            red_dots_3d.add(dot)

        self.play(
            ReplacementTransform(blue_dots_2d, blue_dots_3d),
            ReplacementTransform(red_dots_2d, red_dots_3d),
            run_time=2,
        )
        self.wait(0.5)

        # Step 5: Show paraboloid surface
        surface = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            checkerboard_colors=[BLUE_D, BLUE_E],
            fill_opacity=0.22,
            stroke_width=0.5,
        )
        self.play(Create(surface), run_time=2)
        self.wait(0.5)

        # Step 6: Show separating hyperplane
        c = 1.3
        hyperplane = Surface(
            lambda u, v: axes.c2p(u, v, c),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            checkerboard_colors=[YELLOW, YELLOW_D],
            fill_opacity=0.35,
            stroke_width=0.5,
        )
        hyperplane_label = Text(
            "特徵空間中的超平面",
            font_size=24,
            color=YELLOW,
        )
        hyperplane_label.to_corner(UP + RIGHT)
        self.play(Create(hyperplane), Write(hyperplane_label), run_time=1.5)
        self.wait(1)

        # Step 7: Project back to 2D - show decision circle
        circle = Circle(
            radius=np.sqrt(c),
            color=YELLOW,
            stroke_width=6,
        )
        plane_circle = Circle(
            radius=np.sqrt(c),
            color=YELLOW,
            stroke_width=6,
        )
        plane_circle.rotate(PI / 2, RIGHT)
        plane_circle.shift(OUT * 0.5)

        self.begin_ambient_camera_rotation(rate=0.18)
        self.play(
            FadeOut(hyperplane),
            FadeOut(hyperplane_label),
            FadeOut(formula),
            run_time=1,
        )

        circle_3d = Circle(
            radius=np.sqrt(c),
            color=YELLOW,
            stroke_width=6,
        )
        circle_3d.rotate(PI / 2, RIGHT)
        circle_3d.shift(axes.c2p(0, 0, c))
        self.play(Create(circle_3d))
        self.wait(1)
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=2)
        self.wait(0.5)

        text_projection = Text(
            "x² + y² = c  ⇒  二維中的圓形決策邊界",
            font_size=28,
            color=YELLOW,
        )
        text_projection.to_corner(UP + LEFT)
        circle_2d = Circle(radius=np.sqrt(c), color=YELLOW, stroke_width=6)
        circle_2d.shift(axes.c2p(0, 0, 0))
        circle_2d.rotate(PI / 2, RIGHT)
        self.play(
            Transform(circle_3d, circle_2d),
            Write(text_projection),
            run_time=1.5,
        )
        self.wait(1)

        # Step 8 & 9: Camera rotation + final summary
        self.move_camera(phi=65 * DEGREES, theta=-45 * DEGREES, run_time=2)
        self.wait(1)
        self.stop_ambient_camera_rotation()

        summary_1 = Text(
            "3D 空間中：線性超平面可分類",
            font_size=30,
            color=GREEN,
        )
        summary_2 = Text(
            "2D 空間中：非線性決策邊界",
            font_size=30,
            color=YELLOW,
        )
        summary_3 = Text(
            "這就是核技巧（Kernel Trick）的核心直覺",
            font_size=26,
            color=WHITE,
        )
        summary_1.to_corner(UP + LEFT)
        summary_2.next_to(summary_1, DOWN)
        summary_3.next_to(summary_2, DOWN, buff=0.5)

        self.play(
            FadeOut(text_projection),
            FadeOut(circle_3d),
        )
        self.play(
            Write(summary_1),
            Write(summary_2),
            Write(summary_3),
        )
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(3)
