from manim import *

class SimpleEmbedding(Scene):
    def construct(self):
        # Axes
        axes = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-1, 1, 0.5],
            axis_config={"include_numbers": True}
        )
        self.play(Create(axes))

        # Vectors
        v_jay = Arrow(start=ORIGIN, end=[-0.4, 0.8, 0], buff=0, color=BLUE)
        v_1 = Arrow(start=ORIGIN, end=[-0.3, 0.2, 0], buff=0, color=GREEN)
        v_2 = Arrow(start=ORIGIN, end=[-0.5, -0.4, 0], buff=0, color=RED)

        # Labels
        label_jay = Text("Jay", font_size=24).next_to(v_jay.get_end(), UP)
        label_1 = Text("P1", font_size=24).next_to(v_1.get_end(), RIGHT)
        label_2 = Text("P2", font_size=24).next_to(v_2.get_end(), DOWN)

        self.play(GrowArrow(v_jay), Write(label_jay))
        self.play(GrowArrow(v_1), Write(label_1))
        self.play(GrowArrow(v_2), Write(label_2))

        self.wait(2)
