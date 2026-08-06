"""Streamlit UI for ContentForge.

Purely presentational: collects inputs, calls run_pipeline(), and
renders whatever it returns. No prompt text or API calls live here —
all of that stays in src/agents/ and src/pipeline.py.
"""

import streamlit as st

from src.pipeline import run_pipeline

PLATFORMS = ["Instagram", "YouTube", "LinkedIn", "X"]
FORMATS = ["reel", "carousel", "post"]

VERDICT_BANNERS = {"POST": st.success, "REWORK": st.warning, "SKIP": st.error}


def render_header() -> None:
    """Render the page title and subtitle."""
    st.title("ContentForge")
    st.caption("Real audience input in, reviewed content package out")


def render_inputs() -> tuple[str, dict]:
    """Render the input form and return (research_material, profile).

    profile is built from the required topic/audience field plus the
    fine-tune expander's values (or their defaults) — everything
    run_pipeline needs besides the research material itself.
    """
    topic_audience = st.text_input("What's your content about, and who's it for?")
    research_material = st.text_area(
        "Paste real audience comments, DMs, or competitor posts", height=150
    )

    with st.expander("Fine-tune (optional)", expanded=False):
        platform = st.selectbox("Platform", PLATFORMS, index=0)
        content_format = st.selectbox("Format", FORMATS, index=0)
        brand_voice = st.text_input("Brand voice", value="casual and direct")
        cta = st.text_input("Call to action", value="follow for more")

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
    return research_material, profile


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


def run_with_progress(profile: dict, research_material: str) -> dict | None:
    """Run the pipeline inside a live status panel.

    Returns the outputs dict on success, or None if a step failed (the
    error is already shown via st.error before returning).
    """
    with st.status("Running pipeline...", expanded=True) as status:

        def on_progress(step_num: int, total: int, name: str) -> None:
            status.write(f"[{step_num}/{total}] {name} agent... done")

        try:
            outputs = run_pipeline(profile, research_material, on_progress=on_progress)
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
    research_material, profile = render_inputs()

    if st.button("Run pipeline", type="primary"):
        if validate_inputs(profile, research_material):
            outputs = run_with_progress(profile, research_material)
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
