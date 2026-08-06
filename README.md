# ContentForge

A 5-agent LLM pipeline that turns real audience input into a ready-to-post piece
of social media content — and scores its own output for quality before you publish.

Built on the Anthropic API (`claude-haiku-4-5`) with a Streamlit front end.

<!-- PICTURE 1: hero screenshot goes here -->
![ContentForge in action](docs/contentforge-hero.png)

## The problem

Making good short-form content is not one task, it's five: figuring out what's
actually worth saying, finding an angle that stops the scroll, writing the thing,
deciding what it looks like on screen, and knowing whether it will travel. Ask a
single LLM prompt to do all five and you get a mediocre average of all of them.

ContentForge splits the work across five specialised agents, each with its own
prompt template and its own narrow job, and pipes the structured output of one
into the next.

One rule holds the whole thing together: the system never invents audience data.
You paste real comments, DMs, or competitor posts, and every agent works from
that. Human truth in, machine craft applied, human judgment at the end.

## The pipeline

```
real audience input ─▶ Research ─▶ Hook ─▶ Script ─▶ Visual ─▶ Growth ─▶ scored content
```

| # | Agent | Job | Consumes | Produces |
|---|-----------|--------------------------------------------------------------|--------------------|-------------------|
| 1 | Research  | Extract pain points and angles from real audience material    | audience material  | research brief    |
| 2 | Hook      | Generate and rank opening lines that earn the first 3 seconds | research brief     | ranked hooks      |
| 3 | Script    | Write the full post/script around the winning hook            | brief + hook       | script            |
| 4 | Visual    | Specify shots, captions, on-screen text, and pacing           | script             | visual plan       |
| 5 | Growth    | Score the result and suggest concrete improvements            | everything upstream| quality score     |

Every agent returns **structured JSON**, so each stage is inspectable on its own —
you can see exactly where a weak post went wrong instead of re-rolling the whole
thing.

<!-- PICTURE 2: visual plan screenshot goes here -->
![Frame-by-frame visual plan from the Visual agent](docs/contentforge-plan.png)

## Quality scoring

The Growth agent is the differentiator. Rather than ending at "here's your post,"
the pipeline closes the loop: it grades the output against explicit criteria
(hook strength, clarity, payoff, platform fit) and returns a score plus specific
revision notes. That makes the system measurable — you can A/B a prompt change and
see whether the score moves.

It also has teeth. On its very first full run, the Growth agent scored the
pipeline's own output and refused to approve it: final call REWORK, with specific
notes on the CTA. The quality gate actually gates.

## Project layout

```
app.py               Streamlit UI
src/
  agents/            One module per agent, all inheriting a shared BaseAgent
  prompts/           Prompt templates as .txt files — never inlined in code
  models.py          Dataclasses for each agent's output
  pipeline.py        Chains the five agents, saves every run
tests/               pytest suite (parsing + edge cases)
examples/            Saved pipeline runs with full inputs and outputs
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add your real key
streamlit run app.py
```

Config is read from `.env` via `python-dotenv`. `.env` is gitignored — API keys
are never hardcoded or committed.

## Status

Working end to end: five agents, full pipeline, Streamlit UI. Currently
hardening it — CI, caching, and concurrency are next.

## Roadmap

- [x] Phase 1 — Setup: repo, prompt templates, API working
- [x] Phase 2 — BaseAgent with retry logic + first working agent
- [x] Phase 3 — Clean JSON outputs from every agent + tests
- [x] Phase 4 — Chain all 5 agents into one pipeline
- [x] Phase 5 — Streamlit UI so it's usable without a terminal
- [ ] Phase 6 — CI: tests, lint, and type checks on every push
- [ ] Phase 7 — Response caching so repeat runs cost nothing
- [ ] Phase 8 — Parallelize independent stages, measure speedup
- [ ] Phase 9 — Client-side rate limiting + proper logging
- [ ] Phase 10 — SQLite run history + score analytics
- [ ] Phase 11 — Docker + final docs → v1.0.0

Building this in phases and committing as I go, so the history shows how it actually came together. The plan grew mid-project once the basics worked. The later phases harden it into something closer to production software.