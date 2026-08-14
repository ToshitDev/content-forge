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

import re
from pathlib import Path
from xml.sax.saxutils import escape as _escape_xml

from PIL import Image, ImageColor, ImageDraw, ImageFont

from src.models import PosterOutput

# Pillow's own type is a union: a bundled TTF loads as FreeTypeFont, but
# the ultimate fallback (Pillow's built-in font) is the plainer
# ImageFont type. Every function here treats both the same way.
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
HEADLINE_FONT_PATH = FONTS_DIR / "DejaVuSans-Bold.ttf"
SUBTEXT_FONT_PATH = FONTS_DIR / "DejaVuSans.ttf"

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


def render_poster(
    spec: PosterOutput, output_path: str, size: tuple[int, int] = (1080, 1350)
) -> tuple[Path, Path]:
    """Render `spec` to a PNG and a companion SVG.

    The PNG is written to `output_path`; the SVG is saved alongside it
    with the same stem and a ".svg" extension, so a caller that wants
    both just needs to pick one path.

    Args:
        spec: The Poster Agent's plan — headline, subtext, three colors,
            and a free-text layout description.
        output_path: Where to save the PNG.
        size: Canvas size in pixels. Defaults to a 4:5 portrait poster.

    Returns:
        (png_path, svg_path)
    """
    background_hex = _extract_hex(spec.background_color)
    text_hex = _extract_hex(spec.text_color)
    accent_hex = _extract_hex(spec.accent_color)

    image = _gradient_background(size, background_hex)
    draw = ImageDraw.Draw(image)

    width, height = size
    margin = int(width * MARGIN_RATIO)
    max_text_width = width - 2 * margin
    gap = int(height * BLOCK_GAP_RATIO)

    headline_font, headline_lines, headline_size = _wrap_and_fit(
        draw, spec.headline, HEADLINE_FONT_PATH, max_text_width, HEADLINE_START_SIZE,
        HEADLINE_MIN_SIZE,
    )
    subtext_font, subtext_lines, subtext_size = _wrap_and_fit(
        draw, spec.subtext, SUBTEXT_FONT_PATH, max_text_width, SUBTEXT_START_SIZE,
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

    _draw_centered_lines(draw, headline_lines, headline_font, text_hex, width, headline_top)
    _draw_divider(draw, width, divider_y, accent_hex)
    _draw_centered_lines(draw, subtext_lines, subtext_font, accent_hex, width, subtext_top)

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
        )
    )

    return png_path, svg_path


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


def _load_font(path: Path, size: int) -> Font:
    """Load a bundled TTF at `size`, falling back to Pillow's built-in
    default font if the file is missing or unreadable."""
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size=size)


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


def _wrap_and_fit(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    max_width: int,
    start_size: int,
    min_size: int,
) -> tuple[Font, list[str], int]:
    """Find the largest font size (within [min_size, start_size], step 2)
    at which `text` word-wraps into lines that all fit within max_width.

    Falls back to min_size if even that doesn't fit cleanly (e.g. one
    very long word) — Pillow just draws past the margin rather than
    raising, which is an acceptable edge case for a best-effort layout.
    Returns the chosen size alongside the font object (rather than
    reading it back off the font later) since Pillow's fallback
    ImageFont type doesn't expose a `.size` attribute.
    """
    for size in range(start_size, min_size - 1, -2):
        font = _load_font(font_path, size)
        lines = _wrap_text(draw, text, font, max_width)
        if all(_text_width(draw, line, font) <= max_width for line in lines):
            return font, lines, size
    font = _load_font(font_path, min_size)
    return font, _wrap_text(draw, text, font, max_width), min_size


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
) -> str:
    """Render the same poster as a simple, hand-editable SVG.

    Same content, colors, and approximate layout as the PNG, but text
    stays real <text>/<tspan> elements rather than rasterized pixels —
    still editable (font, color, wording) in any vector tool afterward.
    """
    width, height = size
    headline_tspans = _build_tspans(headline_lines, width / 2, headline_size)
    subtext_tspans = _build_tspans(subtext_lines, width / 2, subtext_size)
    divider_half_width = width * DIVIDER_WIDTH_RATIO / 2
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="{background_hex}" />
  <text x="{width / 2}" y="{headline_top + headline_size}" text-anchor="middle" \
font-family="DejaVu Sans, sans-serif" font-weight="bold" font-size="{headline_size}" \
fill="{text_hex}">{headline_tspans}</text>
  <line x1="{width / 2 - divider_half_width}" y1="{divider_y}" x2="{width / 2 + divider_half_width}" \
y2="{divider_y}" stroke="{accent_hex}" stroke-width="{DIVIDER_THICKNESS}" />
  <text x="{width / 2}" y="{subtext_top + subtext_size}" text-anchor="middle" \
font-family="DejaVu Sans, sans-serif" font-size="{subtext_size}" \
fill="{accent_hex}">{subtext_tspans}</text>
</svg>
"""


def _build_tspans(lines: list[str], x: float, font_size: int) -> str:
    """One <tspan> per wrapped line, each on its own row below the first."""
    return "".join(
        f'<tspan x="{x}" dy="{0 if i == 0 else font_size * LINE_SPACING}">{_escape_xml(line)}</tspan>'
        for i, line in enumerate(lines)
    )
