"""Tests for src/poster_render.py — the pure-code poster rendering layer.

No network access, no API key: PosterOutput is built by hand here rather
than via PosterAgent, since this module only cares about turning an
already-parsed plan into pixels.
"""

from unittest.mock import patch

from PIL import Image

from src import poster_render
from src.models import PosterOutput, StyleOutput
from src.poster_render import (
    ANTON_FONT_PATH,
    BALOO2_FONT_PATH,
    INTER_FONT_PATH,
    MIN_CONTRAST_RATIO,
    PLAYFAIR_FONT_PATH,
    _contrast_ratio,
    _extract_hex,
    _resolve_text_color,
    _split_long_word,
    _wrap_text,
    render_poster,
    select_font,
)

SAMPLE_SPEC = PosterOutput(
    headline="Fall Career Fair 2026",
    subtext="Oct 15, 6-9pm at the Student Union",
    background_color="warm terracotta (#C97C5D)",
    text_color="cream (#F5F1ED)",
    accent_color="sage green (#7A9B8E)",
    layout="headline top-center, details bottom-third",
)


def test_select_font_matches_real_style_kit_phrasing():
    """The exact phrasing a real Style Kit has produced for a bold,
    technical font_mood must resolve to Anton, not just a synthetic
    "bold" one-word test case."""
    assert select_font("bold and technical, sans-serif industrial") == ANTON_FONT_PATH


def test_select_font_matches_playful_mood():
    assert select_font("playful and rounded") == BALOO2_FONT_PATH


def test_select_font_matches_elegant_mood():
    assert select_font("elegant and refined, classic serif") == PLAYFAIR_FONT_PATH


def test_select_font_falls_back_to_inter_for_clean_minimal_mood():
    assert select_font("clean and minimal") == INTER_FONT_PATH


def test_select_font_does_not_misfire_on_sans_serif():
    """"sans-serif" contains "serif" as a substring but means the
    opposite — a mood naming it without any other elegant/serif/classic
    wording must NOT resolve to Playfair Display."""
    assert select_font("clean, sans-serif, minimal") == INTER_FONT_PATH


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


def test_split_long_word_hard_wraps_a_word_wider_than_max_width():
    """Safety net for the case plain word-wrapping can't help with: a
    single unbreakable token (long URL, run-on string) wider than the
    canvas even alone must still come out as lines that each fit,
    instead of overflowing past the margin uncut."""
    image = Image.new("RGB", (10, 10))
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    long_word = "supercalifragilisticexpialidocious" * 3

    chunks = _split_long_word(draw, long_word, font, max_width=80)

    assert "".join(chunks) == long_word
    for chunk in chunks:
        bbox = draw.textbbox((0, 0), chunk, font=font)
        assert bbox[2] - bbox[0] <= 80


def test_render_poster_writes_a_valid_png_and_svg(tmp_path):
    """End-to-end: render_poster produces a real, openable PNG at the
    requested size, plus a companion SVG with the same stem."""
    output_path = tmp_path / "poster.png"

    result = render_poster(SAMPLE_SPEC, str(output_path), size=(400, 500))

    assert result.png_path == output_path
    assert result.png_path.exists()
    assert result.svg_path == output_path.with_suffix(".svg")
    assert result.svg_path.exists()
    assert result.contrast_warning is None or isinstance(result.contrast_warning, str)
    assert result.assets.background_image is not None

    with Image.open(result.png_path) as image:
        image.verify()
        assert Image.open(result.png_path).size == (400, 500)

    svg_text = result.svg_path.read_text()
    assert svg_text.startswith("<?xml")
    # At this narrow width the headline wraps word-by-word across
    # separate <tspan>s, so check for a word rather than the full phrase.
    assert "Fall" in svg_text
    assert "#C97C5D" in svg_text


def test_render_poster_subtext_uses_the_selected_headline_font(tmp_path):
    """Regression test: the subtext used to stay hardcoded to a default
    font regardless of headline_font_path/style-kit mood — every
    _load_font call (headline's and subtext's) must use the SAME
    selected font family, not a fixed one that never changes."""
    output_path = tmp_path / "poster.png"

    with patch(
        "src.poster_render._load_font", wraps=poster_render._load_font
    ) as mock_load_font:
        render_poster(
            SAMPLE_SPEC,
            str(output_path),
            size=(400, 500),
            headline_font_path=PLAYFAIR_FONT_PATH,
        )

    font_paths_used = {call.args[0] for call in mock_load_font.call_args_list}
    assert font_paths_used == {PLAYFAIR_FONT_PATH}


def test_render_poster_subtext_font_changes_when_a_different_font_is_selected(tmp_path):
    """Same render, two different font selections — the subtext's chosen
    font family must differ between them, proving it actually tracks the
    selector instead of staying fixed."""
    used_font_paths_by_render: list[set] = []

    for font_path in (ANTON_FONT_PATH, PLAYFAIR_FONT_PATH):
        output_path = tmp_path / f"poster-{font_path.stem}.png"
        with patch(
            "src.poster_render._load_font", wraps=poster_render._load_font
        ) as mock_load_font:
            render_poster(
                SAMPLE_SPEC,
                str(output_path),
                size=(400, 500),
                headline_font_path=font_path,
            )
        used_font_paths_by_render.append({call.args[0] for call in mock_load_font.call_args_list})

    assert used_font_paths_by_render[0] == {ANTON_FONT_PATH}
    assert used_font_paths_by_render[1] == {PLAYFAIR_FONT_PATH}
    assert used_font_paths_by_render[0] != used_font_paths_by_render[1]


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
        result = render_poster(
            SAMPLE_SPEC,
            str(output_path),
            size=(400, 500),
            style_kit=style_kit,
            use_ai_background=True,
        )

    assert result.png_path.exists()
    assert result.svg_path.exists()
    with Image.open(result.png_path) as image:
        image.verify()
        assert Image.open(result.png_path).size == (400, 500)


def test_contrast_ratio_flags_a_known_bad_pairing():
    """Light gray text on white is a classic low-contrast pairing —
    well under the 4.5 AA threshold."""
    ratio = _contrast_ratio((200, 200, 200), (255, 255, 255))
    assert ratio < MIN_CONTRAST_RATIO


def test_contrast_ratio_passes_a_known_good_pairing():
    """Pure black on pure white is the maximum possible contrast (21:1),
    comfortably above the 4.5 AA threshold."""
    ratio = _contrast_ratio((0, 0, 0), (255, 255, 255))
    assert ratio >= MIN_CONTRAST_RATIO


def test_resolve_text_color_auto_corrects_an_unlocked_low_contrast_color():
    """A near-white text color against a near-white background fails
    contrast; since it's not locked (came from the AI plan, not a manual
    color-picker edit), it should be swapped for whichever of pure
    black/white contrasts better — here, black."""
    image = Image.new("RGB", (100, 100), (245, 245, 245))
    box = (0, 0, 100, 100)

    resolved_hex, warning = _resolve_text_color(image, box, "#F0F0F0", locked=False)

    assert resolved_hex == "#000000"
    assert warning is False


def test_resolve_text_color_leaves_a_locked_color_untouched_but_warns():
    """A manually-picked color (locked=True) is never overridden, even
    if it fails contrast — only flagged so the UI can show a warning."""
    image = Image.new("RGB", (100, 100), (245, 245, 245))
    box = (0, 0, 100, 100)

    resolved_hex, warning = _resolve_text_color(image, box, "#F0F0F0", locked=True)

    assert resolved_hex == "#F0F0F0"
    assert warning is True


def test_resolve_text_color_leaves_a_passing_color_unchanged():
    """Contrast already clears the threshold — no correction, no warning."""
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    box = (0, 0, 100, 100)

    resolved_hex, warning = _resolve_text_color(image, box, "#000000", locked=False)

    assert resolved_hex == "#000000"
    assert warning is False


def test_render_poster_composites_accent_elements_when_requested(tmp_path):
    """use_accent_elements=True fetches accent images (mocked here) and
    the resulting PosterRenderResult carries them back in .assets for
    caching — the same mechanism a free re-render reuses."""
    from PIL import Image as PILImage

    style_kit = StyleOutput(
        colors=["warm terracotta (#C97C5D)"],
        font_mood="bold",
        layout_tendency="dense, edge-to-edge",
        vibe="energetic",
    )
    output_path = tmp_path / "poster.png"

    def fake_accent_element(*args, **kwargs):
        buffer_image = PILImage.new("RGBA", (64, 64), (255, 0, 0, 255))
        import io

        buffer = io.BytesIO()
        buffer_image.save(buffer, format="PNG")
        return buffer.getvalue()

    with patch("src.poster_render.generate_accent_element", side_effect=fake_accent_element):
        result = render_poster(
            SAMPLE_SPEC,
            str(output_path),
            size=(400, 500),
            style_kit=style_kit,
            use_accent_elements=True,
        )

    assert result.png_path.exists()
    assert len(result.assets.accent_images) == 2
