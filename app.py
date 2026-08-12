"""Streamlit UI for ContentForge.

Purely presentational: collects inputs, calls run_pipeline(), and
renders whatever it returns. No prompt text or API calls live here —
all of that stays in src/agents/ and src/pipeline.py.
"""

import asyncio
import re

import pymupdf
import streamlit as st

from src.agents.style import StyleAgent
from src.agents.suggest import SuggestAgent
from src.logging_config import configure_logging
from src.models import StyleOutput, SuggestOutput
from src.pipeline import run_pipeline

configure_logging()

PLATFORMS = ["Instagram", "YouTube", "LinkedIn", "X"]
FORMATS = ["reel", "carousel", "post"]
HEX_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")

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
    render_visual(outputs["visual"])
    render_captions(outputs["growth"])
    st.caption(f"Saved to {outputs['saved_path']}")


def main() -> None:
    """Wire up the page: header, inputs, run button, and persisted results."""
    render_header()
    render_style_reference()
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
