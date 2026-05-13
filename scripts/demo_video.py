"""
MedSimplify - Demo Video (Manim CE)
Generates animated explainer sections for the 3-minute YouTube demo.

Sections:
1. ProblemScene - Shows the problem (complex document, statistics)
2. SolutionScene - Shows the transformation (before/after)
3. ArchitectureScene - Shows how it works
4. MetricsScene - Shows results (grade levels, improvement)
5. CTAScene - Call to action with URL

Render: manim -pqh scripts/demo_video.py AllScenes
Preview: manim -pql scripts/demo_video.py AllScenes
"""

from manim import *

class ProblemScene(Scene):
    def construct(self):
        # Title
        title = Text("The Problem", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Statistics
        stat1 = Text("1 in 5 adults cannot read\nstandard documents", font_size=32, color=YELLOW)
        stat1.next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(stat1))
        self.wait(1.5)
        self.play(FadeOut(stat1))

        # Show a complex document (represented as text block)
        doc_text = Text(
            "Dear Patient,\nYour glycated haemoglobin\n(HbA1c) at 58 mmol/mol\nexceeds the threshold of\n48 mmol/mol...",
            font_size=24, color=RED_A
        )
        doc_label = Text("Grade 14 Reading Level", font_size=20, color=RED)
        doc_group = VGroup(doc_text, doc_label.next_to(doc_text, DOWN))
        doc_group.move_to(ORIGIN)

        self.play(FadeIn(doc_group))
        self.wait(2)

        # Confused person (represented by question marks)
        confused = Text("???", font_size=72, color=RED)
        confused.next_to(doc_group, RIGHT, buff=1)
        self.play(FadeIn(confused, scale=0.5))
        self.wait(1)

        # Consequences
        consequences = VGroup(
            Text("Missed medications", font_size=24, color=RED_B),
            Text("Missed appointments", font_size=24, color=RED_B),
            Text("Lost benefits", font_size=24, color=RED_B),
        ).arrange(DOWN, buff=0.3)
        consequences.to_edge(DOWN)
        self.play(FadeIn(consequences, shift=UP))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


class SolutionScene(Scene):
    def construct(self):
        title = Text("MedSimplify", font_size=56, color=BLUE)
        subtitle = Text("Making documents accessible for everyone", font_size=28, color=BLUE_B)
        VGroup(title, subtitle).arrange(DOWN).to_edge(UP)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)

        # Before/After split
        before_title = Text("BEFORE", font_size=20, color=RED)
        before_text = Text(
            "Your glycated\nhaemoglobin (HbA1c)\nat 58 mmol/mol\nexceeds threshold...",
            font_size=18, color=RED_A
        )
        before_grade = Text("Grade 14", font_size=16, color=RED)
        before = VGroup(before_title, before_text, before_grade).arrange(DOWN, buff=0.3)
        before.to_edge(LEFT, buff=1).shift(DOWN * 0.5)

        after_title = Text("AFTER", font_size=20, color=GREEN)
        after_text = Text(
            "[INFO] Your blood test:\n\nYour blood sugar\nis too high.\n\n[MEDICINE] Take Metformin\n1 tablet, 2x daily",
            font_size=18, color=GREEN_A
        )
        after_grade = Text("Grade 3", font_size=16, color=GREEN)
        after = VGroup(after_title, after_text, after_grade).arrange(DOWN, buff=0.3)
        after.to_edge(RIGHT, buff=1).shift(DOWN * 0.5)

        arrow = Arrow(before.get_right(), after.get_left(), color=YELLOW, buff=0.3)
        arrow_label = Text("Gemma 4", font_size=16, color=YELLOW)
        arrow_label.next_to(arrow, UP, buff=0.1)

        self.play(FadeIn(before))
        self.wait(1)
        self.play(GrowArrow(arrow), FadeIn(arrow_label))
        self.wait(0.5)
        self.play(FadeIn(after))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


class ArchitectureScene(Scene):
    def construct(self):
        title = Text("How It Works", font_size=40, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))

        # Pipeline boxes
        boxes = []
        labels = ["Document\nPhoto/Text", "Gemma 4\nVision", "Fine-tuned\nGemma 4", "Easy Read\nOutput"]
        colors = [BLUE, PURPLE, GREEN, YELLOW]

        for i, (label, color) in enumerate(zip(labels, colors)):
            box = RoundedRectangle(width=2.2, height=1.4, corner_radius=0.2, color=color)
            text = Text(label, font_size=16, color=color)
            text.move_to(box)
            group = VGroup(box, text)
            boxes.append(group)

        pipeline = VGroup(*boxes).arrange(RIGHT, buff=0.5)
        pipeline.move_to(ORIGIN)

        for i, box in enumerate(boxes):
            self.play(FadeIn(box, scale=0.8), run_time=0.5)
            if i < len(boxes) - 1:
                arrow = Arrow(
                    boxes[i].get_right(),
                    boxes[i+1].get_left(),
                    color=WHITE, buff=0.1
                )
                self.play(GrowArrow(arrow), run_time=0.3)

        self.wait(1)

        # Key features
        features = VGroup(
            Text("Runs locally via Ollama", font_size=20, color=TEAL),
            Text("20+ languages supported", font_size=20, color=TEAL),
            Text("Fine-tuned with Unsloth", font_size=20, color=TEAL),
        ).arrange(DOWN, buff=0.3)
        features.to_edge(DOWN)
        self.play(FadeIn(features, shift=UP))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


class MetricsScene(Scene):
    def construct(self):
        title = Text("Results", font_size=40, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))

        # Animated counter: Grade 14 -> Grade 3
        grade_label = Text("Reading Level:", font_size=28, color=WHITE)
        grade_label.shift(UP * 1)

        grade_tracker = ValueTracker(14.2)
        grade_number = always_redraw(
            lambda: DecimalNumber(
                grade_tracker.get_value(),
                num_decimal_places=1,
                font_size=72,
                color=interpolate_color(RED, GREEN, (14.2 - grade_tracker.get_value()) / 10.4)
            ).next_to(grade_label, DOWN, buff=0.5)
        )
        grade_text = always_redraw(
            lambda: Text(
                "Grade",
                font_size=24,
                color=WHITE
            ).next_to(grade_number, LEFT, buff=0.2)
        )

        self.play(FadeIn(grade_label), FadeIn(grade_number), FadeIn(grade_text))
        self.wait(0.5)

        # Animate the decrease
        self.play(
            grade_tracker.animate.set_value(3.8),
            run_time=3,
            rate_func=smooth
        )
        self.wait(1)

        # Show improvement
        improvement = Text("10.4 grade levels improvement!", font_size=28, color=GREEN)
        improvement.shift(DOWN * 1.5)
        self.play(FadeIn(improvement, scale=0.5))
        self.wait(1)

        # Additional metrics
        metrics = VGroup(
            Text("94% information preserved", font_size=22, color=BLUE_B),
            Text("< 8 second processing", font_size=22, color=BLUE_B),
            Text("Works offline (Ollama)", font_size=22, color=BLUE_B),
        ).arrange(DOWN, buff=0.2)
        metrics.to_edge(DOWN)
        self.play(FadeIn(metrics))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


class CTAScene(Scene):
    def construct(self):
        # Final call to action
        quote = Text(
            "Every person deserves to\nunderstand their own\nhealth information.",
            font_size=36, color=WHITE
        )
        self.play(Write(quote), run_time=2)
        self.wait(2)
        self.play(FadeOut(quote))

        # Project info
        name = Text("MedSimplify", font_size=56, color=BLUE)
        url = Text("huggingface.co/spaces/kaushikss/medsimplify", font_size=22, color=YELLOW)
        built = Text("Built with Gemma 4 + Unsloth + Ollama", font_size=20, color=GRAY)
        hackathon = Text("Gemma 4 Good Hackathon 2026", font_size=18, color=GRAY)

        group = VGroup(name, url, built, hackathon).arrange(DOWN, buff=0.4)
        self.play(FadeIn(group, scale=0.8))
        self.wait(3)


class AllScenes(Scene):
    """Render all scenes in sequence."""
    def construct(self):
        # We render each scene separately and concatenate with ffmpeg
        # This class exists as a placeholder for the full video
        pass
