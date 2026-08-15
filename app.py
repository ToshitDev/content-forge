"""Streamlit UI for ContentForge.

Purely presentational: collects inputs, calls run_pipeline(), and
renders whatever it returns. No prompt text or API calls live here —
all of that stays in src/agents/ and src/pipeline.py.
"""

import asyncio
import logging
import re
import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path

import pymupdf
import streamlit as st

from src import history
from src.agents.poster import PosterAgent
from src.agents.style import StyleAgent
from src.agents.suggest import SuggestAgent
from src.agents.voice import VoiceAgent, estimate_cost
from src.logging_config import configure_logging
from src.models import PosterOutput, StyleOutput, SuggestOutput
from src.pipeline import run_pipeline
from src.poster_render import (
    ANTON_FONT_PATH,
    BALOO2_FONT_PATH,
    BEBAS_NEUE_FONT_PATH,
    INTER_FONT_PATH,
    MONTSERRAT_FONT_PATH,
    ORBITRON_FONT_PATH,
    OSWALD_FONT_PATH,
    PLAYFAIR_FONT_PATH,
    PosterAssets,
    PosterVariant,
    _extract_hex,
    render_poster,
    render_poster_variants,
)

configure_logging()
logger = logging.getLogger(__name__)

PLATFORMS = ["Instagram", "YouTube", "LinkedIn", "X"]
FORMATS = ["reel", "carousel", "post"]
PERSON_PREFERENCES = ["Face visible", "Person shown, no face", "No person at all"]
VOICE_MODES = ["Use generic voice", "Clone from a sample"]
LAYOUT_OPTIONS = [
    "headline top-center, details bottom-third",
    "headline centered, subtext directly below",
    "headline bottom-third, details top-center",
]
# Font selector options for the poster's "Edit and re-render" section.
# None means "auto" — render_poster falls back to select_font(font_mood)
# from the style kit, same as when no font override is given at all.
FONT_CHOICES: dict[str, Path | None] = {
    "Auto (from style kit mood)": None,
    "Anton (bold, technical)": ANTON_FONT_PATH,
    "Inter (clean, minimal)": INTER_FONT_PATH,
    "Baloo 2 (playful, rounded)": BALOO2_FONT_PATH,
    "Playfair Display (elegant, classic)": PLAYFAIR_FONT_PATH,
    "Bebas Neue (tall, condensed)": BEBAS_NEUE_FONT_PATH,
    "Orbitron (futuristic, sci-fi)": ORBITRON_FONT_PATH,
    "Oswald (modern, condensed)": OSWALD_FONT_PATH,
    "Montserrat (clean, professional)": MONTSERRAT_FONT_PATH,
}
# Reverse lookup — used when a poster variant is selected (see
# _activate_poster_variant) to pre-select the font dropdown to match
# whatever font that variant actually used, rather than leaving it on
# "Auto", which would silently re-resolve via select_font() and could
# land on a DIFFERENT font than the one the user just saw and chose
# (variants 2 and 3 deliberately use a different font than "Auto" would).
FONT_PATH_TO_LABEL: dict[Path, str] = {
    path: label for label, path in FONT_CHOICES.items() if path is not None
}
HEX_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")
POSTERS_DIR = Path(__file__).resolve().parent / "posters"

VERDICT_BANNERS = {"POST": st.success, "REWORK": st.warning, "SKIP": st.error}


def render_header() -> None:
    """Render the page title and subtitle."""
    st.title("ContentForge")
    st.caption("Real audience input in, reviewed content package out")


def convert_pdf_page_to_image(pdf_bytes: bytes) -> bytes:
    """Render a PDF's first page to PNG bytes, for the same vision flow
    an uploaded image goes through."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pixmap = doc.load_page(0).get_pixmap()
        return pixmap.tobytes("png")


def extract_style_kit(image_bytes: bytes, media_type: str, theme: str) -> StyleOutput:
    """Call the Style Agent and return the parsed style kit.

    Bridges the agent's async call into Streamlit's synchronous script
    execution, same pattern as fetch_suggested_comments.
    """
    agent = StyleAgent()
    raw = asyncio.run(
        agent.run_parsed({"theme": theme}, image_bytes=image_bytes, media_type=media_type)
    )
    return StyleOutput.from_dict(raw)


def render_style_kit(style_kit: StyleOutput) -> None:
    """Show the extracted style kit: color swatches, font mood, vibe."""
    st.write("**Detected style kit**")
    if style_kit.colors:
        swatch_cols = st.columns(len(style_kit.colors))
        for col, color in zip(swatch_cols, style_kit.colors, strict=True):
            match = HEX_COLOR_PATTERN.search(color)
            col.color_picker(color, value=match.group(0) if match else "#888888", disabled=True)
    st.caption(f"Font mood: {style_kit.font_mood}")
    st.caption(f"Vibe: {style_kit.vibe}")


def render_style_reference() -> None:
    """Render the optional reference upload/theme field.

    On upload, extracts a style kit (image directly, or a PDF's first
    page rendered to an image first) and stashes it in
    st.session_state["style_kit"] for later phases (posters, visuals)
    to reuse. Also stashes the raw image bytes themselves (not just the
    text-derived style kit) in st.session_state["style_reference_image"]
    — the Poster section's AI background can pass this straight to
    Flux 2 as an input_image for real style transfer, not just a
    paraphrased text description of it.
    """
    uploaded = st.file_uploader(
        "Reference (image, PDF, or just describe a theme)",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
    )
    theme = st.text_input("Or describe a theme")
    # Stashed so the Poster section (which also takes an optional theme)
    # can reuse it without asking the user to type it twice.
    st.session_state["style_theme"] = theme

    if uploaded is None:
        # No upload (yet, or anymore — e.g. the user cleared it): nothing
        # to pass as an input_image, and a stale image from an earlier
        # upload this session would be actively wrong to reuse.
        st.session_state["style_reference_image"] = None
        return

    file_bytes = uploaded.getvalue()
    if uploaded.type == "application/pdf":
        image_bytes = convert_pdf_page_to_image(file_bytes)
        media_type = "image/png"
    else:
        image_bytes = file_bytes
        media_type = uploaded.type or "image/png"

    try:
        style_kit = extract_style_kit(image_bytes, media_type, theme)
    except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.error(f"Couldn't extract a style kit: {error}")
        return

    st.session_state["style_kit"] = style_kit
    st.session_state["style_reference_image"] = image_bytes
    render_style_kit(style_kit)


def fetch_poster_spec(event_details: str, style_kit: StyleOutput, theme: str) -> PosterOutput:
    """Call the Poster Agent and return its parsed plan.

    Bridges the agent's async call into Streamlit's synchronous script
    execution, same pattern as fetch_suggested_comments/extract_style_kit.
    """
    agent = PosterAgent()
    raw = asyncio.run(
        agent.run_parsed(
            {
                "event_details": event_details,
                "colors": ", ".join(style_kit.colors),
                "font_mood": style_kit.font_mood,
                "layout_tendency": style_kit.layout_tendency,
                "theme": theme,
            }
        )
    )
    return PosterOutput.from_dict(raw)


def render_poster_generation(style_kit: StyleOutput) -> None:
    """Collect event details + the two AI toggles, then generate 3
    poster variants (render_poster_variants — src/poster_render.py) over
    ONE shared background and stash them for render_poster_variant_picker
    to display. The AI toggle checkboxes are widgets keyed into
    session_state ("poster_use_ai_background"/"poster_use_accent_elements"),
    so render_poster_editor further down can read the same values back
    without needing them passed around as parameters.
    """
    name = st.text_input("Event name", key="poster_event_name")
    date = st.text_input("Date", key="poster_event_date")
    event_time = st.text_input("Time", key="poster_event_time")
    location = st.text_input("Location", key="poster_event_location")
    cta = st.text_input("Call to action", key="poster_event_cta")

    use_ai_background = st.checkbox(
        "Use AI-generated background (~$0.02-0.06 per poster)",
        value=False,
        key="poster_use_ai_background",
        help="Generates the background via the Black Forest Labs Flux "
        "API instead of the built-in gradient. If a reference image was "
        "uploaded above, it's used directly for style transfer instead "
        "of just a text description of it. Falls back to the built-in "
        "background automatically if generation fails or BFL_API_KEY "
        "isn't set. Generated ONCE and shared by all 3 variants below — "
        "never billed 3 times.",
    )
    use_accent_elements = st.checkbox(
        "Add AI-generated decorative accents (~$0.01-0.03 per element, 1-2 elements)",
        value=False,
        key="poster_use_accent_elements",
        help="Generates 1-2 small decorative graphics (badge, icon, "
        "corner flourish, or geometric shape, picked from the style "
        "kit's vibe) and composites them into opposite corners. Billed "
        "separately from the background above; also shared by all 3 "
        "variants rather than regenerated per variant.",
    )

    if st.button("Generate poster"):
        if not all(field.strip() for field in (name, date, event_time, location, cta)):
            st.warning("Fill in all five event details first.")
        else:
            event_details = (
                f"Event name: {name}\n"
                f"Date: {date}\n"
                f"Time: {event_time}\n"
                f"Location: {location}\n"
                f"Call to action: {cta}"
            )
            try:
                spec = fetch_poster_spec(
                    event_details, style_kit, st.session_state.get("style_theme", "")
                )
                variants = render_poster_variants(
                    spec,
                    str(POSTERS_DIR),
                    style_kit=style_kit,
                    use_ai_background=use_ai_background,
                    reference_image_bytes=st.session_state.get("style_reference_image"),
                    use_accent_elements=use_accent_elements,
                )
            except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
                st.error(f"Couldn't generate poster: {error}")
            else:
                st.session_state["poster_variants"] = variants
                # Default to the first ("Bold") variant so there's
                # always an active poster to edit right away — the user
                # can switch to a different one below without regenerating.
                _activate_poster_variant(variants[0])


def _activate_poster_variant(variant: PosterVariant) -> None:
    """Make `variant` the active poster feeding into render_poster_editor
    below — the same session_state fields a plain render_poster() call
    populates, plus resetting every edit-widget key so the editor's
    inputs (headline/subtext/layout/colors/font) repopulate from THIS
    variant's spec instead of stale values left over from whichever
    variant (or previous poster) was active before — same "pop before
    the widget is re-instantiated" rule the original single-poster flow
    already followed for a fresh "Generate poster" click.
    """
    st.session_state["poster_spec"] = variant.spec
    st.session_state["poster_png_path"] = str(variant.result.png_path)
    st.session_state["poster_svg_path"] = str(variant.result.svg_path)
    st.session_state["poster_assets"] = variant.result.assets
    st.session_state["poster_contrast_warning"] = variant.result.contrast_warning
    st.session_state["active_variant_label"] = variant.label
    st.session_state.pop("poster_headline_edit", None)
    st.session_state.pop("poster_subtext_edit", None)
    st.session_state.pop("poster_layout_edit", None)
    st.session_state.pop("poster_bg_color_edit", None)
    st.session_state.pop("poster_text_color_edit", None)
    st.session_state.pop("poster_accent_color_edit", None)
    # Pre-select the font dropdown to match this variant's ACTUAL font —
    # leaving it on "Auto" would silently re-resolve via select_font()
    # and could land on a different font than the one the user just saw
    # and picked (see FONT_PATH_TO_LABEL's docstring comment above).
    st.session_state["poster_font_edit"] = FONT_PATH_TO_LABEL.get(
        variant.font_path, "Auto (from style kit mood)"
    )


def render_poster_variant_picker() -> None:
    """Show the 3 generated variants side by side (st.columns), each
    with a "Use this one" button — whichever is clicked becomes the
    active poster via _activate_poster_variant, which the editor below
    immediately reflects since both run in the same Streamlit script
    pass. Renders nothing if no variants have been generated yet.
    """
    variants: list[PosterVariant] = st.session_state.get("poster_variants", [])
    if not variants:
        return

    st.write("**Pick a treatment**")
    active_label = st.session_state.get("active_variant_label")
    columns = st.columns(len(variants))
    for column, variant in zip(columns, variants, strict=True):
        with column:
            st.image(str(variant.result.png_path))
            is_active = variant.label == active_label
            st.caption(f"{variant.label}{' (active)' if is_active else ''}")
            if st.button(
                "Use this one",
                key=f"use_variant_{variant.label}",
                disabled=is_active,
            ):
                _activate_poster_variant(variant)


def render_poster_editor(style_kit: StyleOutput) -> None:
    """Edit/re-render/download controls for whichever poster is
    currently active in st.session_state["poster_spec"] — populated
    either by render_poster_variant_picker's "Use this one" or by a
    previous edit here. Renders nothing until a variant has been picked.

    ASSET CACHING (why most edits are free): the resolved background and
    accent images from the active poster are stashed in
    st.session_state["poster_assets"] as a PosterAssets. "Re-render"
    always passes that cache back into render_poster(), so editing
    headline/subtext/layout/colors/font and clicking it only re-runs the
    local drawing code (src/poster_render.py) — no API call. Only the
    two explicit "Regenerate background" / "Regenerate accents" buttons
    below clear their half of the cache and pay for a fresh BFL call.
    """
    if "poster_spec" not in st.session_state:
        return
    spec = st.session_state["poster_spec"]
    assets: PosterAssets = st.session_state["poster_assets"]
    # Written by the checkboxes in render_poster_generation, which
    # always runs earlier in the same script pass.
    use_ai_background = st.session_state.get("poster_use_ai_background", False)
    use_accent_elements = st.session_state.get("poster_use_accent_elements", False)

    st.write("**Edit and re-render**")
    edited_headline = st.text_input("Headline", value=spec.headline, key="poster_headline_edit")
    edited_subtext = st.text_input("Subtext", value=spec.subtext, key="poster_subtext_edit")
    edited_layout = st.selectbox(
        "Layout",
        LAYOUT_OPTIONS,
        index=LAYOUT_OPTIONS.index(spec.layout) if spec.layout in LAYOUT_OPTIONS else 0,
        key="poster_layout_edit",
    )

    color1, color2, color3 = st.columns(3)
    edited_bg_color = color1.color_picker(
        "Background", value=_extract_hex(spec.background_color), key="poster_bg_color_edit"
    )
    edited_text_color = color2.color_picker(
        "Text", value=_extract_hex(spec.text_color), key="poster_text_color_edit"
    )
    edited_accent_color = color3.color_picker(
        "Accent", value=_extract_hex(spec.accent_color), key="poster_accent_color_edit"
    )
    font_choice = st.selectbox(
        "Headline font", list(FONT_CHOICES.keys()), key="poster_font_edit"
    )

    def _do_render(assets_for_render: PosterAssets | None) -> None:
        """Shared by "Re-render" and the two "Regenerate" buttons below —
        only which half of `assets_for_render` is populated differs."""
        edited_spec = replace(
            spec,
            headline=edited_headline,
            subtext=edited_subtext,
            layout=edited_layout,
            background_color=edited_bg_color,
            text_color=edited_text_color,
            accent_color=edited_accent_color,
        )
        result = render_poster(
            edited_spec,
            str(POSTERS_DIR / f"{uuid.uuid4().hex}.png"),
            style_kit=style_kit,
            use_ai_background=use_ai_background,
            reference_image_bytes=st.session_state.get("style_reference_image"),
            use_accent_elements=use_accent_elements,
            assets=assets_for_render,
            headline_font_path=FONT_CHOICES[font_choice],
            # Colors above are now always coming from the pickers (manual
            # controls, pre-filled with the AI's originals) — so contrast
            # auto-correction should flag rather than silently override
            # them from here on, same rule as a manual UI edit anywhere
            # else in this app.
            lock_colors=True,
        )
        st.session_state["poster_spec"] = edited_spec
        st.session_state["poster_png_path"] = str(result.png_path)
        st.session_state["poster_svg_path"] = str(result.svg_path)
        st.session_state["poster_assets"] = result.assets
        st.session_state["poster_contrast_warning"] = result.contrast_warning

    render_col, bg_col, accent_col = st.columns(3)
    if render_col.button("Re-render", help="Free — reuses the existing background/accents."):
        _do_render(assets)
    if bg_col.button(
        "Regenerate background",
        disabled=not use_ai_background,
        help="Costs a new BFL API call. Enable the AI background checkbox above first.",
    ):
        _do_render(PosterAssets(background_image=None, accent_images=assets.accent_images))
    if accent_col.button(
        "Regenerate accents",
        disabled=not use_accent_elements,
        help="Costs a new BFL API call per element. Enable the accents checkbox above first.",
    ):
        _do_render(PosterAssets(background_image=assets.background_image, accent_images=[]))

    contrast_warning = st.session_state.get("poster_contrast_warning")
    if contrast_warning:
        st.warning(contrast_warning)

    st.image(st.session_state["poster_png_path"])
    png_bytes = Path(st.session_state["poster_png_path"]).read_bytes()
    svg_bytes = Path(st.session_state["poster_svg_path"]).read_bytes()
    col1, col2 = st.columns(2)
    col1.download_button("Download PNG", png_bytes, file_name="poster.png", mime="image/png")
    col2.download_button("Download SVG", svg_bytes, file_name="poster.svg", mime="image/svg+xml")


def fetch_suggested_comments(topic_audience: str) -> list[str]:
    """Call the Suggest Agent and return its illustrative example comments.

    Bridges the agent's async call into Streamlit's synchronous script
    execution. Raises on any failure — the caller shows it via st.error.
    """
    agent = SuggestAgent()
    raw = asyncio.run(
        agent.run_parsed({"niche": topic_audience, "audience": topic_audience})
    )
    return SuggestOutput.from_dict(raw).suggestions


def render_inputs() -> tuple[str, dict, bool]:
    """Render the input form and return (research_material, profile, use_cache).

    profile is built from the required topic/audience field plus the
    fine-tune expander's values (or their defaults) — everything
    run_pipeline needs besides the research material itself. use_cache
    is a pipeline-level flag, not a template input, so it's returned
    separately rather than folded into profile.
    """
    topic_audience = st.text_input("What's your content about, and who's it for?")

    # This has to run, and write to st.session_state["research_material"],
    # before the text_area below is instantiated — Streamlit forbids setting
    # a widget's session_state value after that widget already exists in
    # the same script run.
    suggest_clicked = st.button(
        "Suggest example comments", disabled=not topic_audience.strip()
    )
    if suggest_clicked:
        try:
            suggestions = fetch_suggested_comments(topic_audience)
            st.session_state["research_material"] = "\n".join(suggestions)
            st.session_state["suggestions_active"] = True
        except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(f"Couldn't generate example comments: {error}")

    if st.session_state.get("suggestions_active"):
        st.caption(
            "Example comments below are AI-generated illustrations, not real "
            "audience data. Edit them or replace with real comments before "
            "running the pipeline."
        )

    research_material = st.text_area(
        "Paste real audience comments, DMs, or competitor posts",
        height=150,
        key="research_material",
    )

    with st.expander("Fine-tune (optional)", expanded=False):
        platform = st.selectbox("Platform", PLATFORMS, index=0)
        content_format = st.selectbox("Format", FORMATS, index=0)
        brand_voice = st.text_input("Brand voice", value="casual and direct")
        cta = st.text_input("Call to action", value="follow for more")
        person_preference = st.selectbox(
            "Show a person in this video?", PERSON_PREFERENCES, index=0
        )

        st.write("**Voice**")
        voice_mode = st.radio("Narration voice", VOICE_MODES, index=0)
        # Not a widget key, so no instantiation-order constraint — this is
        # just how render_voiceover() (called later, from render_results())
        # finds out what voice setup the user picked.
        if voice_mode == "Clone from a sample":
            sample = st.file_uploader(
                "Upload a short voice sample (10-30s, clear audio)",
                type=["mp3", "wav", "m4a", "ogg"],
            )
            st.session_state["voice_sample"] = sample.getvalue() if sample else None
            st.session_state["voice_sample_name"] = sample.name if sample else None
        else:
            st.session_state["voice_sample"] = None
            st.session_state["voice_sample_name"] = None

        use_cache = st.checkbox(
            "Use cache",
            value=True,
            help="Reuse a cached response for an identical prompt instead of "
            "calling the API again.",
        )

    # The single "topic + audience" field feeds both template inputs — the
    # prompts ask for niche and audience separately, but one phrase covers
    # both well enough without forcing the user to fill two boxes.
    profile = {
        "niche": topic_audience,
        "audience": topic_audience,
        "platform": platform,
        "format": content_format,
        "brand_voice": brand_voice,
        "cta": cta,
        "person_preference": person_preference,
    }
    return research_material, profile, use_cache


def validate_inputs(profile: dict, research_material: str) -> bool:
    """Warn about missing required fields; return True if the pipeline can run."""
    ok = True
    if not profile["niche"].strip():
        st.warning("Tell us what your content is about and who it's for.")
        ok = False
    if not research_material.strip():
        st.warning("Paste some real audience material to work from.")
        ok = False
    return ok


def run_with_progress(profile: dict, research_material: str, use_cache: bool) -> dict | None:
    """Run the pipeline inside a live status panel.

    Returns the outputs dict on success, or None if a step failed (the
    error is already shown via st.error before returning).
    """
    with st.status("Running pipeline...", expanded=True) as status:

        def on_progress(step_num: int, total: int, name: str) -> None:
            status.write(f"[{step_num}/{total}] {name} agent... done")

        try:
            outputs = run_pipeline(
                profile, research_material, on_progress=on_progress, use_cache=use_cache
            )
        except RuntimeError as error:
            status.update(label="Pipeline failed", state="error")
            st.error(str(error))
            return None

        status.update(label="Pipeline complete", state="complete")
        return outputs


def render_verdict(growth) -> None:
    """Show the POST/REWORK/SKIP verdict banner first, above everything else."""
    banner = VERDICT_BANNERS.get(growth.final_call, st.info)
    banner(f"**{growth.final_call}** — {growth.final_call_reason}")


def render_scores(growth) -> None:
    """Show the six quality scores as metrics, two rows of three."""
    scores = growth.scores

    row1 = st.columns(3)
    row1[0].metric("Clarity", f"{scores.clarity}/10")
    row1[1].metric("Retention", f"{scores.retention}/10")
    row1[2].metric("Save potential", f"{scores.save_potential}/10")

    row2 = st.columns(3)
    row2[0].metric("Shareability", f"{scores.shareability}/10")
    row2[1].metric("Audience fit", f"{scores.audience_fit}/10")
    row2[2].metric("CTA strength", f"{scores.cta_strength}/10")


def render_hook(hook) -> None:
    """Show the winning hook, with its reasoning tucked in an expander."""
    st.subheader("Winning hook")
    st.info(hook.winner.text)
    with st.expander("Why this hook won"):
        st.write(hook.winner.reasoning)


def render_script(script) -> None:
    """Show the full script (one-click copy) and its word count."""
    st.subheader(f"Script ({script.word_count} words)")
    st.code(script.script, language=None)


def render_visual(visual) -> None:
    """Show cover options, the frame plan, and an assets checklist."""
    st.subheader("Visual plan")

    st.write("**Cover options**")
    for option in visual.cover_options:
        st.write(f"- {option}")

    st.write("**Frame plan**")
    frame_rows = [
        {
            "Position": frame.position,
            "On-screen text": frame.on_screen_text,
            "Visual direction": frame.visual_direction,
        }
        for frame in visual.frame_plan
    ]
    st.table(frame_rows)

    st.write("**Assets needed**")
    for i, asset in enumerate(visual.assets_needed):
        st.checkbox(asset, value=False, key=f"asset_{i}")


def render_voiceover(script_text: str) -> None:
    """Let the user generate narration audio for the finished script.

    Runs after the pipeline (and its Script step) has already produced
    script_text — this can't happen any earlier. Picks up the voice
    setup (generic vs. cloned-from-sample) that render_inputs() stashed
    in st.session_state during this same rerun.
    """
    st.subheader("Voiceover")
    if st.button("Generate voiceover"):
        try:
            agent = VoiceAgent()
            sample_bytes = st.session_state.get("voice_sample")
            if sample_bytes:
                voice_id = agent.clone_voice(
                    sample_bytes,
                    st.session_state.get("voice_sample_name") or "sample.mp3",
                    name="ContentForge cloned voice",
                )
                audio_path = agent.generate(script_text, voice_id=voice_id)
            else:
                audio_path = agent.generate(script_text)
        except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(f"Couldn't generate voiceover: {error}")
        else:
            st.session_state["voiceover_path"] = str(audio_path)
            run_id = st.session_state.get("outputs", {}).get("run_id")
            if run_id is not None:
                try:
                    history.update_voice_cost(run_id, estimate_cost(script_text))
                except sqlite3.Error as db_error:
                    # Cost tracking is analytics, not the deliverable — the
                    # audio file already exists and is already shown below,
                    # so a DB hiccup here shouldn't read as a failure.
                    logger.error("Failed to record voice cost estimate: %s", db_error)

    if "voiceover_path" in st.session_state:
        audio_bytes = Path(st.session_state["voiceover_path"]).read_bytes()
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "Download voiceover", audio_bytes, file_name="voiceover.mp3", mime="audio/mp3"
        )


def render_captions(growth) -> None:
    """Show both caption options, each in a one-click-copy code block."""
    st.subheader("Captions")
    for i, caption in enumerate(growth.captions, start=1):
        st.write(f"Option {i}")
        st.code(caption, language=None)


def render_results(outputs: dict) -> None:
    """Render the full results package in order: verdict first, then details."""
    render_verdict(outputs["growth"])
    render_scores(outputs["growth"])
    render_hook(outputs["hook"])
    render_script(outputs["script"])
    render_voiceover(outputs["script"].script)
    render_visual(outputs["visual"])
    render_captions(outputs["growth"])
    st.caption(f"Saved to {outputs['saved_path']}")


def render_content_pipeline_tab() -> None:
    """The existing 5-agent Research -> Growth pipeline, plus voiceover —
    unchanged in behavior from before the tab split, just lifted out of
    main() so each tab is a single self-contained call.
    """
    research_material, profile, use_cache = render_inputs()

    if st.button("Run pipeline", type="primary") and validate_inputs(
        profile, research_material
    ):
        outputs = run_with_progress(profile, research_material, use_cache)
        if outputs is not None:
            st.session_state["outputs"] = outputs
        else:
            # A failed run shouldn't leave a stale, mismatched result
            # from a previous successful run sitting on the page.
            st.session_state.pop("outputs", None)

    if "outputs" in st.session_state:
        render_results(st.session_state["outputs"])


def render_poster_studio_tab() -> None:
    """Style Kit extraction, poster generation, the 3-variant picker,
    and all poster editing controls — everything poster-related lives
    in this one tab, separate from the content pipeline.

    st.session_state["style_kit"] / ["style_reference_image"] are
    intentionally NOT scoped to just this tab: they're written here by
    render_style_reference() but are plain, unprefixed session_state
    keys so a future phase (e.g. the Visual step in the content
    pipeline) could read the same extracted style kit without
    duplicating the extraction call. Everything else poster-specific
    (poster_spec, poster_variants, the edit-widget keys, ...) stays
    scoped to this tab in practice simply because nothing outside
    Poster Studio ever reads it.
    """
    render_style_reference()
    st.subheader("Poster")
    style_kit = st.session_state.get("style_kit")
    if style_kit is None:
        st.info(
            "Add a reference image or theme above (in the Reference "
            "section) to generate a poster from its style kit."
        )
        return

    render_poster_generation(style_kit)
    render_poster_variant_picker()
    render_poster_editor(style_kit)


def main() -> None:
    """Wire up the page: header, then the two focused tools as separate
    tabs — Content Pipeline (the 5-agent Research->Growth flow plus
    voiceover) and Poster Studio (Style Kit + poster generation/editing).
    Splitting into st.tabs() keeps each tool's controls from crowding
    the other's on one long page, while both still share the page's one
    st.session_state (e.g. the Style Kit, if a future phase wants it in
    the pipeline tab too — see render_poster_studio_tab's docstring).
    """
    render_header()
    content_tab, poster_tab = st.tabs(["Content Pipeline", "Poster Studio"])

    with content_tab:
        render_content_pipeline_tab()

    with poster_tab:
        render_poster_studio_tab()


if __name__ == "__main__":
    main()
