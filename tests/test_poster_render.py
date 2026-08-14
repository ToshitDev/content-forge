"""Tests for src/poster_render.py — the pure-code poster rendering layer.

No network access, no API key: PosterOutput is built by hand here rather
than via PosterAgent, since this module only cares about turning an
already-parsed plan into pixels.
"""

from unittest.mock import patch

from PIL import Image

from src.models import PosterOutput, StyleOutput
from src.poster_render import _extract_hex, _wrap_text, render_poster

SAMPLE_SPEC = PosterOutput(
    headline="Fall Career Fair 2026",
    subtext="Oct 15, 6-9pm at the Student Union",
    background_color="warm terracotta (#C97C5D)",
    text_color="cream (#F5F1ED)",
    accent_color="sage green (#7A9B8E)",
    layout="headline top-center, details bottom-third",
)


def test_extract_hex_pulls_hex_out_of_a_description():
    assert _extract_hex("warm terracotta (#C97C5D)") == "#C97C5D"


def test_extract_hex_falls_back_when_no_hex_present():
    """LLM output isn't guaranteed to include a hex code — a plain color
    name shouldn't crash rendering, just fall back to a safe default."""
    assert _extract_hex("a nice shade of blue") == "#333333"


def test_wrap_text_splits_long_text_into_multiple_lines():
    """A word-wrap sanity check: given a narrow max_width, a multi-word
    string is split across more than one line, and every line individually
    fits within that width."""
    image = Image.new("RGB", (10, 10))
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)

    lines = _wrap_text(draw, "one two three four five six seven eight", font, max_width=80)

    assert len(lines) > 1
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        assert bbox[2] - bbox[0] <= 80


def test_render_poster_writes_a_valid_png_and_svg(tmp_path):
    """End-to-end: render_poster produces a real, openable PNG at the
    requested size, plus a companion SVG with the same stem."""
    output_path = tmp_path / "poster.png"

    png_path, svg_path = render_poster(SAMPLE_SPEC, str(output_path), size=(400, 500))

    assert png_path == output_path
    assert png_path.exists()
    assert svg_path == output_path.with_suffix(".svg")
    assert svg_path.exists()

    with Image.open(png_path) as image:
        image.verify()
        assert Image.open(png_path).size == (400, 500)

    svg_text = svg_path.read_text()
    assert svg_text.startswith("<?xml")
    # At this narrow width the headline wraps word-by-word across
    # separate <tspan>s, so check for a word rather than the full phrase.
    assert "Fall" in svg_text
    assert "#C97C5D" in svg_text


def test_render_poster_falls_back_to_procedural_background_on_ai_failure(tmp_path):
    """use_ai_background=True with a style kit, but the AI call raises —
    render_poster must still succeed by falling back to the procedural
    gradient+texture background, never crashing poster generation over
    the optional AI feature (missing key, timeout, moderation, network
    error — this covers "any exception" generically)."""
    style_kit = StyleOutput(
        colors=["warm terracotta (#C97C5D)"],
        font_mood="bold",
        layout_tendency="dense, edge-to-edge",
        vibe="energetic",
    )
    output_path = tmp_path / "poster.png"

    def always_fails(*args, **kwargs):
        raise RuntimeError("simulated BFL failure")

    with patch("src.poster_render.generate_ai_background", side_effect=always_fails):
        png_path, svg_path = render_poster(
            SAMPLE_SPEC,
            str(output_path),
            size=(400, 500),
            style_kit=style_kit,
            use_ai_background=True,
        )

    assert png_path.exists()
    assert svg_path.exists()
    with Image.open(png_path) as image:
        image.verify()
        assert Image.open(png_path).size == (400, 500)
