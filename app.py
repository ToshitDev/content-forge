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
from src.poster_render import render_poster

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
    to reuse.
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


def render_poster_section() -> None:
    """Poster generator: plans a poster from the Style Kit above (if one
    exists) plus real event details, then renders it locally.

    Editing the headline/subtext/layout afterward and clicking
    "Re-render" only re-runs the local drawing code (src/poster_render.py)
    — no API call. See render_poster_section's "Re-render" handling
    below for why that's safe: the colors and overall plan already came
    from PosterAgent, so a text/layout tweak doesn't need Claude's
    involvement again, only Pillow's.
    """
    st.subheader("Poster")
    style_kit = st.session_state.get("style_kit")
    if style_kit is None:
        st.info(
            "Add a reference image or theme above (in the Reference "
            "section) to generate a poster from its style kit."
        )
        return

    name = st.text_input("Event name", key="poster_event_name")
    date = st.text_input("Date", key="poster_event_date")
    event_time = st.text_input("Time", key="poster_event_time")
    location = st.text_input("Location", key="poster_event_location")
    cta = st.text_input("Call to action", key="poster_event_cta")

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
                png_path, svg_path = render_poster(
                    spec, str(POSTERS_DIR / f"{uuid.uuid4().hex}.png")
                )
            except Exception as error:  # noqa: BLE001 - surfaced to the user, not swallowed
                st.error(f"Couldn't generate poster: {error}")
            else:
                st.session_state["poster_spec"] = spec
                st.session_state["poster_png_path"] = str(png_path)
                st.session_state["poster_svg_path"] = str(svg_path)
                # Clear any previous poster's edits — these are widget
                # keys, so this has to happen before they're instantiated
                # below in this same run, same rule as research_material.
                st.session_state.pop("poster_headline_edit", None)
                st.session_state.pop("poster_subtext_edit", None)
                st.session_state.pop("poster_layout_edit", None)

    if "poster_spec" not in st.session_state:
        return
    spec = st.session_state["poster_spec"]

    st.write("**Edit and re-render**")
    edited_headline = st.text_input("Headline", value=spec.headline, key="poster_headline_edit")
    edited_subtext = st.text_input("Subtext", value=spec.subtext, key="poster_subtext_edit")
    edited_layout = st.selectbox(
        "Layout",
        LAYOUT_OPTIONS,
        index=LAYOUT_OPTIONS.index(spec.layout) if spec.layout in LAYOUT_OPTIONS else 0,
        key="poster_layout_edit",
    )

    if st.button("Re-render"):
        edited_spec = replace(
            spec, headline=edited_headline, subtext=edited_subtext, layout=edited_layout
        )
        png_path, svg_path = render_poster(
            edited_spec, str(POSTERS_DIR / f"{uuid.uuid4().hex}.png")
        )
        st.session_state["poster_spec"] = edited_spec
        st.session_state["poster_png_path"] = str(png_path)
        st.session_state["poster_svg_path"] = str(svg_path)

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


def main() -> None:
    """Wire up the page: header, inputs, run button, and persisted results."""
    render_header()
    render_style_reference()
    render_poster_section()
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


if __name__ == "__main__":
    main()
