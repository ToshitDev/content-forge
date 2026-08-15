"""Renders a PosterOutput plan into an actual poster image.

Pure code, no AI-generated pixels: PosterAgent (src/agents/poster.py)
only decides WHAT the poster should say and which style-kit colors to
use; everything here — the gradient, the font, where each line of text
lands — is deterministic drawing code. Same plan in, same pixels out,
every time.

VISUAL SANITY NOTE FOR FUTURE PHASES: headline, the accent divider, and
subtext are always drawn as ONE grouped block, not three independently
positioned elements. "Grouped" means: measure the block's total height
first (headline lines + gap + divider + gap + subtext lines), pick ONE
vertical anchor for the whole block (top/center/bottom) from
spec.layout, then stack the three pieces contiguously starting there.
A layout description that names two different positions (e.g. "headline
top, subtext bottom") does NOT split them across the canvas anymore —
see _block_anchor's docstring for why, and change that function (not
the drawing code) if a future phase genuinely needs independent
per-element placement.
"""

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape as _escape_xml

from PIL import Image, ImageColor, ImageDraw, ImageFont

from src.models import PosterOutput, StyleOutput
from src.poster_background import generate_accent_element, generate_ai_background

logger = logging.getLogger(__name__)

# Pillow's own type is a union: a bundled TTF loads as FreeTypeFont, but
# the ultimate fallback (Pillow's built-in font) is the plainer
# ImageFont type. Every function here treats both the same way.
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

# Headline font family, picked per-poster by select_font() below from
# the Style Kit's font_mood — see MOOD_FONT_RULES. Inter and Playfair
# Display and Baloo 2 are variable fonts (one file, a weight axis) with
# a "Bold" named instance; Anton is a single static weight and is
# already heavy/blocky by design, so it needs no variation selection —
# see _load_font's `bold` handling for how that split is handled.
ANTON_FONT_PATH = FONTS_DIR / "Anton-Regular.ttf"
INTER_FONT_PATH = FONTS_DIR / "Inter-Variable.ttf"
BALOO2_FONT_PATH = FONTS_DIR / "Baloo2-Variable.ttf"
PLAYFAIR_FONT_PATH = FONTS_DIR / "PlayfairDisplay-Variable.ttf"

# (font path, keywords) pairs, checked in order — first match wins.
# Simple substring keyword matching, NOT exhaustive: this only covers
# the handful of moods the Style Agent actually tends to describe (e.g.
# "bold and technical, sans-serif industrial", "playful and rounded",
# "clean and minimal"). Expand this list with more (path, keywords)
# entries as new moods turn up in real Style Kit output, rather than
# trying to enumerate every possible adjective up front.
MOOD_FONT_RULES: list[tuple[Path, tuple[str, ...]]] = [
    (ANTON_FONT_PATH, ("bold", "technical", "industrial", "blocky")),
    (BALOO2_FONT_PATH, ("playful", "rounded", "fun")),
    (PLAYFAIR_FONT_PATH, ("elegant", "serif", "classic")),
]

HEX_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")
FALLBACK_COLOR = "#333333"

MARGIN_RATIO = 0.08  # horizontal text margin, as a fraction of canvas width
HEADLINE_START_SIZE = 90
HEADLINE_MIN_SIZE = 36
SUBTEXT_START_SIZE = 40
SUBTEXT_MIN_SIZE = 20
LINE_SPACING = 1.25

# Where the grouped block sits when anchored to the top/bottom, as a
# fraction of canvas height — not the block's own position, just how
# much breathing room to leave at that edge.
TOP_ANCHOR_RATIO = 0.10
BOTTOM_ANCHOR_MARGIN_RATIO = 0.10

# Gap above and below the divider (between it and the headline/subtext),
# as a fraction of canvas height, so it scales with poster size instead
# of a fixed pixel count.
BLOCK_GAP_RATIO = 0.035
DIVIDER_WIDTH_RATIO = 0.22  # divider line length, as a fraction of canvas width
DIVIDER_THICKNESS = 4  # pixels — a hairline stays a hairline regardless of canvas size

# Background texture — grid lines and/or corner brackets, chosen by
# _texture_style() from the style kit's layout_tendency and scaled by
# _texture_intensity() from its vibe, so two different style kits don't
# produce the same-looking poster. All still pure Pillow shapes drawn at
# low opacity, composited onto the gradient — no AI imagery.
GRID_SPACING_RATIO = 0.07  # grid cell size, as a fraction of canvas width
GRID_BASE_OPACITY = 0.12  # before the vibe-driven intensity multiplier
CORNER_INSET_RATIO = 0.05  # corner bracket distance from the edge
CORNER_ARM_RATIO = 0.05  # corner bracket arm length
CORNER_THICKNESS = 3
CORNER_BASE_OPACITY = 0.35  # brighter than the grid — there are only 8 short lines total

# The whole-poster border frame (task-required, always drawn — separate
# from the corner brackets above, which are a conditional texture choice).
BORDER_INSET_RATIO = 0.025
BORDER_THICKNESS = 3

# Decorative AI-generated accent elements (badge/icon/corner flourish/
# geometric shape) — a separate, optional, separately-toggled layer from
# the AI background. Sized and inset so they land in the canvas corners,
# which the horizontally-centered headline/subtext block never reaches
# regardless of whether it's anchored top/center/bottom.
ACCENT_SIZE_RATIO = 0.22  # each element's size, as a fraction of the shorter canvas dimension
ACCENT_INSET_RATIO = 0.06
ACCENT_CORNER_ANCHORS = ("top-right", "bottom-left")  # up to 2 elements, opposite corners

# (element type, keywords) pairs, checked in order — same simple,
# expandable keyword-matching spirit as MOOD_FONT_RULES/_texture_style.
ACCENT_TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("badge", ("bold", "energetic", "vibrant", "loud", "dynamic")),
    ("icon", ("playful", "fun", "whimsical", "rounded")),
    ("corner flourish", ("elegant", "classic", "refined", "sophisticated", "serif")),
    ("geometric accent shape", ("calm", "minimal", "understated", "clean", "modern")),
]

# WCAG 2.x contrast ratio (1:1 to 21:1). 4.5 is the standard AA
# threshold for normal text; applied uniformly to both headline and
# subtext here rather than branching on WCAG's large-vs-normal-text
# split, since poster type sizes vary a lot and a single conservative
# threshold is simpler to reason about than picking the "right" one per
# font size.
MIN_CONTRAST_RATIO = 4.5


@dataclass
class PosterAssets:
    """Already-fetched AI images (background and/or accent elements),
    so a text/color/font-only edit can be re-rendered by reusing these
    instead of hitting the paid BFL APIs again. Round-trips through
    PosterRenderResult.assets -> caller stores it (e.g. in Streamlit
    session_state) -> passed back in as render_poster's `assets` arg on
    the next call.
    """

    background_image: Image.Image | None = None
    accent_images: list[Image.Image] = field(default_factory=list)


@dataclass
class PosterRenderResult:
    """Everything render_poster() produces: the two saved files, the
    resolved image objects (for the caller to cache via PosterAssets),
    and whether text contrast needed a warning."""

    png_path: Path
    svg_path: Path
    assets: PosterAssets
    contrast_warning: str | None


def select_font(font_mood: str) -> Path:
    """Pick a headline font family from the Style Kit's font_mood.

    Simple substring keyword matching against MOOD_FONT_RULES, not
    exhaustive — see that list's docstring for how to extend it. Falls
    back to Inter (clean/minimal/modern, and anything that doesn't hit
    a more specific mood) when nothing matches.
    """
    text = font_mood.lower()
    # "sans-serif"/"sans serif" contains "serif" as a substring but
    # means the opposite — strip it out first so a mood description
    # like "clean, sans-serif, minimal" doesn't misfire into the
    # serif/elegant rule below.
    text = text.replace("sans-serif", "").replace("sans serif", "")
    for path, keywords in MOOD_FONT_RULES:
        if any(keyword in text for keyword in keywords):
            return path
    return INTER_FONT_PATH


def render_poster(
    spec: PosterOutput,
    output_path: str,
    size: tuple[int, int] = (1080, 1350),
    style_kit: StyleOutput | None = None,
    use_ai_background: bool = False,
    reference_image_bytes: bytes | None = None,
    use_accent_elements: bool = False,
    assets: PosterAssets | None = None,
    headline_font_path: Path | None = None,
    lock_colors: bool = False,
) -> PosterRenderResult:
    """Render `spec` to a PNG and a companion SVG.

    The PNG is written to `output_path`; the SVG is saved alongside it
    with the same stem and a ".svg" extension, so a caller that wants
    both just needs to pick one path.

    ASSET REUSE (why a text/color/font edit doesn't cost a new API
    call): `assets`, when given, is used as-is for the background
    and/or accent elements instead of building or fetching them again —
    only text layout, colors, and the font are recomputed. Pass back the
    `assets` field of a previous PosterRenderResult to get this. Without
    it, the background is built fresh (AI if `use_ai_background`, else
    procedural) and accent elements are fetched fresh if
    `use_accent_elements` — each independently, so e.g. `assets` with
    only a background image still fetches fresh accents if requested.

    Args:
        spec: The Poster Agent's plan — headline, subtext, three colors,
            and a free-text layout description.
        output_path: Where to save the PNG.
        size: Canvas size in pixels. Defaults to a 4:5 portrait poster.
        style_kit: The Style Kit used to generate `spec` (if any).
            Optional — its layout_tendency and vibe pick the procedural
            background texture style/intensity (see _texture_style /
            _texture_intensity) and the accent element type(s) (see
            _select_accent_element_types), its font_mood picks the
            default headline font (see select_font, overridden by
            `headline_font_path` if given), and its full contents also
            feed the AI background/accent prompts. None falls back to a
            neutral default texture, the default headline font, and
            disables both AI features (there's nothing to build a
            prompt from).
        use_ai_background: If True (and no background in `assets`),
            generate the background via the Black Forest Labs Flux API
            (src/poster_background.py) instead of the procedural
            gradient + texture. Any failure — missing key, timeout,
            moderation, network error — is caught and logged, and
            rendering falls back to the procedural background instead
            of ever crashing poster generation over an optional, paid
            feature.
        reference_image_bytes: Raw bytes of a reference image (if the
            user uploaded one in the Reference section) to pass to Flux
            2 for real image-based style transfer. Only used when
            `use_ai_background` is True and there's no cached background
            in `assets`; None falls back to the text-only AI background
            path.
        use_accent_elements: If True (and no accents in `assets`),
            generate 1-2 small decorative graphics and composite them
            into opposite corners. Separately toggled from the
            background — an additional cost per element — and just as
            failure-tolerant: any generation failure is caught, logged,
            and simply means fewer (or zero) accents, never a crash.
        assets: Already-fetched background/accent images to reuse — see
            "ASSET REUSE" above.
        headline_font_path: Overrides select_font(style_kit.font_mood)
            when the user has picked a specific font explicitly (e.g.
            via a font selector in the UI) rather than leaving it to the
            mood-based default.
        lock_colors: True once the user has manually picked text/accent
            colors (e.g. via color pickers in the UI) rather than using
            the Poster Agent's originals — see _resolve_text_color for
            what this changes about contrast auto-correction.

    Returns:
        A PosterRenderResult — png_path, svg_path, the resolved assets
        (cache these for a free re-render), and contrast_warning (None
        if headline/subtext both read clearly against their background).
    """
    background_hex = _extract_hex(spec.background_color)
    text_hex = _extract_hex(spec.text_color)
    accent_hex = _extract_hex(spec.accent_color)

    layout_tendency = style_kit.layout_tendency if style_kit else ""
    vibe = style_kit.vibe if style_kit else ""
    font_mood = style_kit.font_mood if style_kit else ""
    texture_style = _texture_style(layout_tendency)
    intensity = _texture_intensity(vibe)
    resolved_headline_font_path = headline_font_path or select_font(font_mood)

    image = _resolve_background(
        size, background_hex, accent_hex, texture_style, intensity, style_kit,
        use_ai_background, reference_image_bytes, assets,
    )
    background_for_result = image.copy()

    accent_images = _resolve_accents(style_kit, use_accent_elements, assets)
    image = _composite_accents(image, accent_images, size)
    draw = ImageDraw.Draw(image)

    width, height = size
    margin = int(width * MARGIN_RATIO)
    max_text_width = width - 2 * margin
    gap = int(height * BLOCK_GAP_RATIO)

    headline_font, headline_lines, headline_size = _wrap_and_fit(
        draw, spec.headline, resolved_headline_font_path, max_text_width, HEADLINE_START_SIZE,
        HEADLINE_MIN_SIZE, bold=True,
    )
    subtext_font, subtext_lines, subtext_size = _wrap_and_fit(
        draw, spec.subtext, resolved_headline_font_path, max_text_width, SUBTEXT_START_SIZE,
        SUBTEXT_MIN_SIZE,
    )

    # Measure the whole block before drawing anything, so we know where
    # to start — see the module note on why this is one block, not two
    # independently anchored pieces.
    headline_block_height = _measure_lines_height(draw, headline_lines, headline_font)
    subtext_block_height = _measure_lines_height(draw, subtext_lines, subtext_font)
    total_block_height = headline_block_height + gap + DIVIDER_THICKNESS + gap + subtext_block_height

    headline_top = _block_top_y(spec.layout, height, total_block_height)
    divider_y = headline_top + headline_block_height + gap
    subtext_top = divider_y + DIVIDER_THICKNESS + gap

    # Contrast is checked (and, if unlocked, auto-corrected) against
    # each region's actual background pixels BEFORE that text is drawn —
    # sampling after would just measure the text ink itself.
    headline_box = (0, headline_top, width, headline_top + max(headline_block_height, 1))
    text_hex, headline_warning = _resolve_text_color(image, headline_box, text_hex, lock_colors)
    _draw_centered_lines(draw, headline_lines, headline_font, text_hex, width, headline_top)

    _draw_divider(draw, width, divider_y, accent_hex)

    subtext_box = (0, subtext_top, width, subtext_top + max(subtext_block_height, 1))
    accent_hex, subtext_warning = _resolve_text_color(image, subtext_box, accent_hex, lock_colors)
    _draw_centered_lines(draw, subtext_lines, subtext_font, accent_hex, width, subtext_top)

    _draw_border(draw, size, accent_hex)

    png_path = Path(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG")

    svg_path = png_path.with_suffix(".svg")
    svg_path.write_text(
        _build_svg(
            size,
            background_hex,
            text_hex,
            accent_hex,
            headline_lines,
            headline_top,
            headline_size,
            divider_y,
            subtext_lines,
            subtext_top,
            subtext_size,
            texture_style,
            intensity,
            accent_images,
        )
    )

    contrast_warning = None
    if headline_warning or subtext_warning:
        contrast_warning = (
            "Text may be hard to read against the background — consider a "
            "different text/accent color or background."
        )

    return PosterRenderResult(
        png_path=png_path,
        svg_path=svg_path,
        assets=PosterAssets(background_image=background_for_result, accent_images=accent_images),
        contrast_warning=contrast_warning,
    )


def _resolve_background(
    size: tuple[int, int],
    background_hex: str,
    accent_hex: str,
    texture_style: str,
    intensity: float,
    style_kit: StyleOutput | None,
    use_ai_background: bool,
    reference_image_bytes: bytes | None,
    assets: PosterAssets | None,
) -> Image.Image:
    """Get the poster's background image: reuse a cached one from
    `assets` if given, otherwise build fresh (AI or procedural).

    A cached image is resized to `size` in case it was generated (or
    re-fetched) at a different canvas size than this render is using.
    """
    if assets is not None and assets.background_image is not None:
        return assets.background_image.convert("RGB").resize(size)
    return _build_background(
        size, background_hex, accent_hex, texture_style, intensity, style_kit,
        use_ai_background, reference_image_bytes,
    )


def _build_background(
    size: tuple[int, int],
    background_hex: str,
    accent_hex: str,
    texture_style: str,
    intensity: float,
    style_kit: StyleOutput | None,
    use_ai_background: bool,
    reference_image_bytes: bytes | None,
) -> Image.Image:
    """Build a fresh background image (not reused from cached assets).

    Tries the AI background first when requested (and a style kit is
    available to build a prompt from) — with the reference image for
    real style transfer if one was given, otherwise the text-only path.
    On ANY failure there — missing key, timeout, moderation, network
    error, anything — logs a warning and falls through to the
    procedural gradient + texture instead of raising. That fallback is
    deliberately unconditional: this is an optional, paid feature, and
    generating a poster should never depend on a third-party API being
    up.
    """
    if use_ai_background and style_kit is not None:
        try:
            ai_bytes = generate_ai_background(style_kit, size, reference_image_bytes)
            return Image.open(io.BytesIO(ai_bytes)).convert("RGB").resize(size)
        except Exception as error:  # noqa: BLE001 - optional feature, must always fall back
            logger.warning(
                "AI background generation failed, falling back to procedural background: %s",
                error,
            )

    image = _gradient_background(size, background_hex)
    return _apply_texture(image, size, accent_hex, texture_style, intensity)


def _select_accent_element_types(vibe: str) -> list[str]:
    """Pick 1-2 accent element types ("badge", "icon", "corner
    flourish", "geometric accent shape") from the style kit's vibe.

    Simple keyword matching against ACCENT_TYPE_RULES, not exhaustive —
    same spirit as _texture_style/select_font. The first rule whose
    keywords appear sets the primary type; a second, different type
    rounds out the pair for a touch of contrast between the two accents
    rather than the identical graphic in both corners.
    """
    text = vibe.lower()
    matched = [
        element_type
        for element_type, keywords in ACCENT_TYPE_RULES
        if any(keyword in text for keyword in keywords)
    ]
    if not matched:
        matched = ["geometric accent shape"]
    primary = matched[0]
    secondary = next((t for t, _ in ACCENT_TYPE_RULES if t != primary), primary)
    return [primary, secondary]


def _resolve_accents(
    style_kit: StyleOutput | None, use_accent_elements: bool, assets: PosterAssets | None
) -> list[Image.Image]:
    """Get the poster's accent images: reuse cached ones from `assets`
    if given, otherwise fetch fresh ones if requested, otherwise none."""
    if assets is not None and assets.accent_images:
        return assets.accent_images
    if use_accent_elements and style_kit is not None:
        return _fetch_accent_elements(style_kit)
    return []


def _fetch_accent_elements(style_kit: StyleOutput) -> list[Image.Image]:
    """Generate 1-2 accent images via the BFL API (never reused from
    cache — see _resolve_accents for that).

    Never raises — a generation failure here just means fewer (or zero)
    accents, the same "optional feature never crashes generation"
    contract as the background.
    """
    accent_images: list[Image.Image] = []
    for element_type in _select_accent_element_types(style_kit.vibe):
        try:
            element_bytes = generate_accent_element(style_kit, element_type)
            accent_images.append(Image.open(io.BytesIO(element_bytes)).convert("RGBA"))
        except Exception as error:  # noqa: BLE001 - optional feature, must always degrade gracefully
            logger.warning("Accent element generation failed, skipping: %s", error)
    return accent_images


def _composite_accents(
    image: Image.Image, accent_images: list[Image.Image], size: tuple[int, int]
) -> Image.Image:
    """Paste up to 2 accent images into opposite corners of the canvas
    (see ACCENT_CORNER_ANCHORS), sized and inset so they never reach the
    horizontally-centered headline/subtext block in the middle."""
    if not accent_images:
        return image
    image = image.convert("RGBA")
    width, height = size
    accent_size = int(min(width, height) * ACCENT_SIZE_RATIO)
    inset = int(min(width, height) * ACCENT_INSET_RATIO)

    for accent_image, corner in zip(accent_images[:2], ACCENT_CORNER_ANCHORS, strict=False):
        resized = accent_image.convert("RGBA").resize((accent_size, accent_size))
        if corner == "top-right":
            position = (width - inset - accent_size, inset)
        else:  # bottom-left
            position = (inset, height - inset - accent_size)
        image.paste(resized, position, resized)

    return image.convert("RGB")


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG relative luminance of an sRGB color, in [0, 1]."""

    def channel(value: int) -> float:
        normalized = value / 255
        return normalized / 12.92 if normalized <= 0.03928 else ((normalized + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast_ratio(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    """WCAG contrast ratio between two sRGB colors: 1.0 (identical) to
    21.0 (pure black against pure white)."""
    luminance1 = _relative_luminance(rgb1)
    luminance2 = _relative_luminance(rgb2)
    lighter, darker = max(luminance1, luminance2), min(luminance1, luminance2)
    return (lighter + 0.05) / (darker + 0.05)


def _sample_average_color(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Average RGB color of `image` within `box` — a cheap stand-in for
    "the actual pixels behind the text", since the background can be a
    gradient, procedural texture, or AI image, not a flat color.
    Resizing the cropped region down to a single pixel with box-filter
    resampling computes that average using Pillow's own (fast, C-level)
    resampling rather than a manual per-pixel loop.
    """
    region = image.convert("RGB").crop(box)
    pixel = region.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    return pixel  # type: ignore[return-value]


def _resolve_text_color(
    image: Image.Image, box: tuple[int, int, int, int], color_hex: str, locked: bool
) -> tuple[str, bool]:
    """Check `color_hex`'s WCAG contrast against the pixels actually
    behind `box`, in the poster as rendered so far (background + any
    accents, but not yet this text) — not just against the flat
    background_color/accent_color from the Poster Agent's plan, which
    may not reflect a busy AI background or texture at all.

    If it fails and `locked` is False (the color still came from the
    Poster Agent, not a manual color-picker edit), tries switching to
    pure white or pure black — whichever contrasts better — and uses
    that if it clears the threshold. A manually-picked color (`locked`)
    is never overridden, only flagged.

    Returns (final_color_hex, warning_triggered) — warning_triggered is
    True when contrast is still below MIN_CONTRAST_RATIO after this
    (whether because the color is locked, or because even the
    auto-correction didn't clear the bar).
    """
    background_rgb = _sample_average_color(image, box)
    text_rgb = ImageColor.getrgb(color_hex)[:3]
    ratio = _contrast_ratio(text_rgb, background_rgb)
    if ratio >= MIN_CONTRAST_RATIO:
        return color_hex, False
    if locked:
        return color_hex, True

    white_ratio = _contrast_ratio((255, 255, 255), background_rgb)
    black_ratio = _contrast_ratio((0, 0, 0), background_rgb)
    corrected_hex, corrected_ratio = (
        ("#FFFFFF", white_ratio) if white_ratio >= black_ratio else ("#000000", black_ratio)
    )
    return corrected_hex, corrected_ratio < MIN_CONTRAST_RATIO


def _extract_hex(color_description: str) -> str:
    """Pull a #RRGGBB hex code out of a color description string.

    Style Kit colors (and the Poster Agent's copies of them) look like
    "warm terracotta (#C97C5D)". Falls back to a neutral dark gray if a
    hex code isn't found — this is LLM output, not guaranteed to be
    perfectly formatted.
    """
    match = HEX_COLOR_PATTERN.search(color_description)
    return match.group(0) if match else FALLBACK_COLOR


def _gradient_background(size: tuple[int, int], base_hex: str) -> Image.Image:
    """A simple vertical gradient: a lightened version of base_hex at the
    top, fading to base_hex itself at the bottom. Enough depth that it
    doesn't read as a flat color swatch, without needing anything fancier.
    """
    width, height = size
    base_rgb = ImageColor.getrgb(base_hex)
    light_rgb = tuple(min(255, channel + 35) for channel in base_rgb)

    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y / max(height - 1, 1)
        row_rgb = tuple(
            int(light_rgb[i] + (base_rgb[i] - light_rgb[i]) * t) for i in range(3)
        )
        draw.line([(0, y), (width, y)], fill=row_rgb)
    return image


def _texture_style(layout_tendency: str) -> str:
    """Decide the background texture motif from the Style Kit's
    layout_tendency: "grid" (thin lines across the whole canvas) for a
    structured/technical/dense tendency, "frame" (small corner brackets
    only) for a soft/minimal/whitespace-heavy one, or "both" when the
    wording doesn't clearly lean either way (including an empty string).
    """
    text = layout_tendency.lower()
    structured_signals = ("grid", "dense", "edge-to-edge", "structured", "geometric", "modular")
    soft_signals = ("whitespace", "minimal", "soft", "airy", "open", "centered", "organic")
    structured_score = sum(1 for signal in structured_signals if signal in text)
    soft_score = sum(1 for signal in soft_signals if signal in text)
    if structured_score > soft_score:
        return "grid"
    if soft_score > structured_score:
        return "frame"
    return "both"


def _texture_intensity(vibe: str) -> float:
    """Scale how visible the background texture is from the Style Kit's
    vibe — a bold/energetic vibe gets more visible texture, a calm/
    understated one stays closer to invisible. This is the "parameterize
    it, don't hardcode it" half of the texture system: _texture_style
    picks WHAT gets drawn, this picks HOW STRONG it looks, so two style
    kits with the same layout_tendency but different vibes still don't
    render identical posters.
    """
    text = vibe.lower()
    bold_signals = ("bold", "energetic", "vibrant", "loud", "dynamic", "playful")
    calm_signals = ("calm", "understated", "quiet", "subtle", "minimal", "soft", "gentle")
    bold_score = sum(1 for signal in bold_signals if signal in text)
    calm_score = sum(1 for signal in calm_signals if signal in text)
    if bold_score > calm_score:
        return 1.5
    if calm_score > bold_score:
        return 0.6
    return 1.0


def _apply_texture(
    image: Image.Image,
    size: tuple[int, int],
    accent_hex: str,
    texture_style: str,
    intensity: float,
) -> Image.Image:
    """Draw the chosen background texture onto `image` at low opacity.

    Drawn on a separate RGBA overlay and alpha-composited onto the
    gradient, rather than drawn directly — Pillow's plain ImageDraw
    doesn't blend partial-opacity fills against whatever gradient color
    is already at that pixel, an overlay does.
    """
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    # getrgb()'s return type also covers RGBA (4-tuple) inputs; a #RRGGBB
    # hex string is always 3 channels, but the declared type is broader.
    r, g, b = ImageColor.getrgb(accent_hex)[:3]
    accent_rgb = (r, g, b)

    if texture_style in ("grid", "both"):
        _draw_grid(overlay_draw, size, accent_rgb, intensity)
    if texture_style in ("frame", "both"):
        _draw_corner_brackets(overlay_draw, size, accent_rgb, intensity)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _draw_grid(
    draw: ImageDraw.ImageDraw, size: tuple[int, int], accent_rgb: tuple[int, int, int], intensity: float
) -> None:
    """Thin, low-opacity grid lines across the whole canvas — a
    "structured/technical" texture cue."""
    width, height = size
    spacing = max(20, int(width * GRID_SPACING_RATIO))
    alpha = max(0, min(255, int(255 * GRID_BASE_OPACITY * intensity)))
    color = (*accent_rgb, alpha)
    for x in range(0, width, spacing):
        draw.line([(x, 0), (x, height)], fill=color, width=1)
    for y in range(0, height, spacing):
        draw.line([(0, y), (width, y)], fill=color, width=1)


def _draw_corner_brackets(
    draw: ImageDraw.ImageDraw, size: tuple[int, int], accent_rgb: tuple[int, int, int], intensity: float
) -> None:
    """Small L-shaped marks at each of the 4 corners — a light "soft/
    minimal" texture cue that doesn't fill the canvas the way a grid does.
    """
    width, height = size
    alpha = max(0, min(255, int(255 * CORNER_BASE_OPACITY * intensity)))
    color = (*accent_rgb, alpha)
    inset = int(min(width, height) * CORNER_INSET_RATIO)
    arm = int(min(width, height) * CORNER_ARM_RATIO)
    # (x, y, horizontal arm direction, vertical arm direction) per corner.
    corners = [
        (inset, inset, 1, 1),
        (width - inset, inset, -1, 1),
        (inset, height - inset, 1, -1),
        (width - inset, height - inset, -1, -1),
    ]
    for x, y, dx, dy in corners:
        draw.line([(x, y), (x + arm * dx, y)], fill=color, width=CORNER_THICKNESS)
        draw.line([(x, y), (x, y + arm * dy)], fill=color, width=CORNER_THICKNESS)


def _draw_border(draw: ImageDraw.ImageDraw, size: tuple[int, int], accent_hex: str) -> None:
    """A thin accent-colored rule just inside the poster's edge — a
    simple frame around the whole piece. Always drawn (unlike the
    texture above, this isn't conditional on the style kit)."""
    width, height = size
    inset = int(min(width, height) * BORDER_INSET_RATIO)
    draw.rectangle(
        [(inset, inset), (width - inset - 1, height - inset - 1)],
        outline=accent_hex,
        width=BORDER_THICKNESS,
    )


def _block_anchor(layout_text: str) -> str:
    """Decide where the combined headline+divider+subtext block sits
    vertically ("top", "center", or "bottom"), from the Poster Agent's
    free-text layout description.

    Headline and subtext are always drawn as one tightly-stacked unit
    (see the module docstring) — this only decides where that whole
    unit sits, never where each piece sits independently. A layout that
    names BOTH "top" and "bottom" (e.g. "headline top-center, details
    bottom-third") reads as two conflicting anchors for a single block —
    that's exactly the case that used to split headline and subtext to
    opposite edges with a dead gap between them, so it now falls back to
    "center", same as a layout that names neither.
    """
    text = layout_text.lower()
    if "center" in text or "middle" in text:
        return "center"
    mentions_top = "top" in text
    mentions_bottom = "bottom" in text
    if mentions_top and not mentions_bottom:
        return "top"
    if mentions_bottom and not mentions_top:
        return "bottom"
    return "center"


def _block_top_y(layout_text: str, height: int, block_height: int) -> int:
    """Turn a layout description + the block's measured height into the
    y position where the block should start."""
    anchor = _block_anchor(layout_text)
    if anchor == "top":
        return int(height * TOP_ANCHOR_RATIO)
    if anchor == "bottom":
        return int(height * (1 - BOTTOM_ANCHOR_MARGIN_RATIO) - block_height)
    return int((height - block_height) / 2)


def _draw_divider(draw: ImageDraw.ImageDraw, canvas_width: int, y: int, color_hex: str) -> None:
    """A thin horizontal accent-colored rule between headline and
    subtext, so the gap between them reads as an intentional design
    element rather than dead space, even when the block leaves some
    breathing room above or below it."""
    line_length = int(canvas_width * DIVIDER_WIDTH_RATIO)
    x_start = (canvas_width - line_length) / 2
    x_end = x_start + line_length
    y_center = y + DIVIDER_THICKNESS / 2
    draw.line([(x_start, y_center), (x_end, y_center)], fill=color_hex, width=DIVIDER_THICKNESS)


def _load_font(path: Path, size: int, bold: bool = False) -> Font:
    """Load a bundled TTF at `size`, falling back to Pillow's built-in
    default font if the file is missing or unreadable.

    When `bold` is set and the font is a variable font with a "Bold"
    named instance — Inter, Baloo 2, and Playfair Display are all
    shipped this way, one file covering every weight via a weight axis
    — that instance is selected. Anton has no variation axes (it's a
    single static weight, already heavy/blocky by design) and
    get_variation_names() raises OSError for it, same as for any other
    static font; that's expected, not an error worth logging.
    """
    try:
        font = ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size=size)

    if bold and isinstance(font, ImageFont.FreeTypeFont):
        try:
            variation_names = font.get_variation_names()
        except OSError:
            variation_names = []  # static font — no weight axis to select
        if b"Bold" in variation_names:
            font.set_variation_by_name("Bold")
    return font


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: Font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: Font, max_width: int) -> list[str]:
    """Word-wrap `text` to fit within max_width, balancing line lengths
    so a short word doesn't get orphaned alone on the last line (e.g.
    wrapping "...it will not go to plan. NO" as ["...it will not go", "to
    plan. NO"] rather than ["...it will not go to plan.", "NO"]).

    A pure greedy wrap (pack words until the next one doesn't fit) finds
    the minimum possible number of lines, but often dumps a short
    remainder alone on the final line. This first finds that minimum
    line count via a plain greedy wrap at max_width, then binary-searches
    for the NARROWEST width that still wraps into that same number of
    lines — a narrower target width forces words to redistribute more
    evenly across all of them instead of front-loading every line.
    """
    words = text.split()
    if not words:
        return []

    def wrap_at(width: int) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if not current or _text_width(draw, candidate, font) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    greedy_lines = wrap_at(max_width)
    if len(greedy_lines) <= 1:
        return greedy_lines

    target_line_count = len(greedy_lines)
    lo, hi = 1, max_width
    best = greedy_lines
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = wrap_at(mid)
        if len(candidate) == target_line_count:
            best = candidate
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def _split_overflowing_lines(
    draw: ImageDraw.ImageDraw, lines: list[str], font: Font, max_width: int
) -> list[str]:
    """Last-resort safety net used by _wrap_and_fit once its font-shrink
    loop has already tried every size down to min_size: word-wrapping
    can only break on spaces, so a single word with no space to break on
    (a long URL, a run-on string) that's still wider than max_width even
    at min_size comes out of _wrap_text as its own overflowing line.
    Re-measures each line's ACTUAL rendered width for the active font
    (never assumed from character count) and hard-wraps any line that's
    still too wide character by character, so headline text never
    silently overflows or gets cut off at the canvas edge regardless of
    which font is selected.
    """
    result: list[str] = []
    for line in lines:
        if _text_width(draw, line, font) <= max_width:
            result.append(line)
        else:
            result.extend(_split_long_word(draw, line, font, max_width))
    return result


def _split_long_word(draw: ImageDraw.ImageDraw, word: str, font: Font, max_width: int) -> list[str]:
    """Hard-wrap `word` character by character into chunks that each fit
    within max_width. Only reached when a whole word is wider than the
    canvas even alone — always makes forward progress (each chunk gets
    at least one character), so this terminates even if a single
    character's rendered width exceeds max_width."""
    chunks: list[str] = []
    current = ""
    for char in word:
        candidate = current + char
        if current and _text_width(draw, candidate, font) > max_width:
            chunks.append(current)
            current = char
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _wrap_and_fit(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    max_width: int,
    start_size: int,
    min_size: int,
    bold: bool = False,
) -> tuple[Font, list[str], int]:
    """Find the largest font size (within [min_size, start_size], step 2)
    at which `text` word-wraps into lines that all fit within max_width,
    checked by re-measuring each candidate line's ACTUAL rendered width
    for the font actually being tried (never assumed from character
    count) — so this stays correct across every headline font the style
    kit or a manual picker can select, not just whichever font happened
    to be in place when this was first written.

    Falls back to min_size if even that doesn't fit cleanly (e.g. one
    very long word wider than max_width even alone) — and even then,
    hard-wraps that word character by character via
    _split_overflowing_lines rather than letting it silently overflow
    past the canvas margin, so long headline text is never cut off
    regardless of font.

    Returns the chosen size alongside the font object (rather than
    reading it back off the font later) since Pillow's fallback
    ImageFont type doesn't expose a `.size` attribute. `bold` is passed
    straight through to _load_font — see there for what it does.
    """
    for size in range(start_size, min_size - 1, -2):
        font = _load_font(font_path, size, bold=bold)
        lines = _wrap_text(draw, text, font, max_width)
        if all(_text_width(draw, line, font) <= max_width for line in lines):
            return font, lines, size
    font = _load_font(font_path, min_size, bold=bold)
    lines = _split_overflowing_lines(
        draw, _wrap_text(draw, text, font, max_width), font, max_width
    )
    return font, lines, min_size


def _measure_lines_height(draw: ImageDraw.ImageDraw, lines: list[str], font: Font) -> int:
    """Total vertical space `lines` will occupy when drawn by
    _draw_centered_lines, computed without drawing anything — needed to
    figure out where the grouped block should start before committing to
    a position (see the module note)."""
    total = 0.0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        total += (bbox[3] - bbox[1]) * LINE_SPACING
    return int(total)


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: Font,
    color_hex: str,
    canvas_width: int,
    top_y: int,
) -> None:
    """Draw each line horizontally centered, stacked top to bottom from top_y."""
    y = float(top_y)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        x = (canvas_width - line_width) / 2
        draw.text((x, y), line, font=font, fill=color_hex)
        y += line_height * LINE_SPACING


def _build_svg(
    size: tuple[int, int],
    background_hex: str,
    text_hex: str,
    accent_hex: str,
    headline_lines: list[str],
    headline_top: int,
    headline_size: int,
    divider_y: int,
    subtext_lines: list[str],
    subtext_top: int,
    subtext_size: int,
    texture_style: str,
    intensity: float,
    accent_images: list[Image.Image],
) -> str:
    """Render the same poster as a simple, hand-editable SVG.

    Same content, colors, texture, and approximate layout as the PNG,
    but text stays real <text>/<tspan> elements rather than rasterized
    pixels — still editable (font, color, wording) in any vector tool
    afterward. Deliberate exception: an AI-generated background is
    fundamentally a raster photo, not something a "simple template" SVG
    can represent — the SVG background stays a flat background_hex rect
    for that case, same as when there's no AI background at all. Accent
    elements ARE embedded (base64 data-URI <image>s) since they're small
    and keeping them out would leave the SVG visibly incomplete next to
    the PNG.
    """
    width, height = size
    headline_tspans = _build_tspans(headline_lines, width / 2, headline_size)
    subtext_tspans = _build_tspans(subtext_lines, width / 2, subtext_size)
    divider_half_width = width * DIVIDER_WIDTH_RATIO / 2
    texture_svg = _build_texture_svg(size, accent_hex, texture_style, intensity)
    border_svg = _build_border_svg(size, accent_hex)
    accents_svg = _build_accents_svg(size, accent_images)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="{background_hex}" />
  {texture_svg}
  {accents_svg}
  <text x="{width / 2}" y="{headline_top + headline_size}" text-anchor="middle" \
font-family="DejaVu Sans, sans-serif" font-weight="bold" font-size="{headline_size}" \
fill="{text_hex}">{headline_tspans}</text>
  <line x1="{width / 2 - divider_half_width}" y1="{divider_y}" x2="{width / 2 + divider_half_width}" \
y2="{divider_y}" stroke="{accent_hex}" stroke-width="{DIVIDER_THICKNESS}" />
  <text x="{width / 2}" y="{subtext_top + subtext_size}" text-anchor="middle" \
font-family="DejaVu Sans, sans-serif" font-size="{subtext_size}" \
fill="{accent_hex}">{subtext_tspans}</text>
  {border_svg}
</svg>
"""


def _build_accents_svg(size: tuple[int, int], accent_images: list[Image.Image]) -> str:
    """SVG markup for the same accent elements _composite_accents draws
    onto the PNG, embedded as base64 data-URI <image> elements — small
    enough to embed directly, unlike a full AI background photo (see
    _build_svg's docstring)."""
    if not accent_images:
        return ""
    width, height = size
    accent_size = int(min(width, height) * ACCENT_SIZE_RATIO)
    inset = int(min(width, height) * ACCENT_INSET_RATIO)
    parts = []
    for accent_image, corner in zip(accent_images[:2], ACCENT_CORNER_ANCHORS, strict=False):
        x, y = (
            (width - inset - accent_size, inset)
            if corner == "top-right"
            else (inset, height - inset - accent_size)
        )
        buffer = io.BytesIO()
        accent_image.convert("RGBA").resize((accent_size, accent_size)).save(buffer, format="PNG")
        data_uri = base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append(
            f'<image x="{x}" y="{y}" width="{accent_size}" height="{accent_size}" '
            f'href="data:image/png;base64,{data_uri}" />'
        )
    return "".join(parts)


def _build_tspans(lines: list[str], x: float, font_size: int) -> str:
    """One <tspan> per wrapped line, each on its own row below the first."""
    return "".join(
        f'<tspan x="{x}" dy="{0 if i == 0 else font_size * LINE_SPACING}">{_escape_xml(line)}</tspan>'
        for i, line in enumerate(lines)
    )


def _build_texture_svg(
    size: tuple[int, int], accent_hex: str, texture_style: str, intensity: float
) -> str:
    """SVG markup for the same background texture _apply_texture draws
    onto the PNG (grid lines and/or corner brackets), at matching
    opacity — so the SVG isn't a stripped-down copy of the PNG."""
    width, height = size
    parts = []

    if texture_style in ("grid", "both"):
        spacing = max(20, int(width * GRID_SPACING_RATIO))
        opacity = max(0.0, min(1.0, GRID_BASE_OPACITY * intensity))
        grid_lines = [f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" />' for x in range(0, width, spacing)]
        grid_lines += [f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" />' for y in range(0, height, spacing)]
        parts.append(
            f'<g stroke="{accent_hex}" stroke-width="1" stroke-opacity="{opacity:.3f}">'
            + "".join(grid_lines)
            + "</g>"
        )

    if texture_style in ("frame", "both"):
        opacity = max(0.0, min(1.0, CORNER_BASE_OPACITY * intensity))
        inset = int(min(width, height) * CORNER_INSET_RATIO)
        arm = int(min(width, height) * CORNER_ARM_RATIO)
        corners = [
            (inset, inset, 1, 1),
            (width - inset, inset, -1, 1),
            (inset, height - inset, 1, -1),
            (width - inset, height - inset, -1, -1),
        ]
        bracket_lines = []
        for x, y, dx, dy in corners:
            bracket_lines.append(f'<line x1="{x}" y1="{y}" x2="{x + arm * dx}" y2="{y}" />')
            bracket_lines.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + arm * dy}" />')
        parts.append(
            f'<g stroke="{accent_hex}" stroke-width="{CORNER_THICKNESS}" stroke-opacity="{opacity:.3f}">'
            + "".join(bracket_lines)
            + "</g>"
        )

    return "".join(parts)


def _build_border_svg(size: tuple[int, int], accent_hex: str) -> str:
    """SVG markup for the same whole-poster border _draw_border draws
    onto the PNG."""
    width, height = size
    inset = int(min(width, height) * BORDER_INSET_RATIO)
    return (
        f'<rect x="{inset}" y="{inset}" width="{width - 2 * inset}" '
        f'height="{height - 2 * inset}" fill="none" stroke="{accent_hex}" '
        f'stroke-width="{BORDER_THICKNESS}" />'
    )
