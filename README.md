# ContentForge

A 5-agent LLM pipeline that turns a raw topic into a ready-to-post piece of social
media content — and scores its own output for quality before you publish.

Built on the Anthropic API (`claude-opus-5`) with a Streamlit front end.

## The problem

Making good short-form content is not one task, it's five: figuring out what's
actually worth saying, finding an angle that stops the scroll, writing the thing,
deciding what it looks like on screen, and knowing whether it will travel. Ask a
single LLM prompt to do all five and you get a mediocre average of all of them.

ContentForge splits the work across five specialised agents, each with its own
prompt template and its own narrow job, and pipes the structured output of one
into the next.

## The pipeline

```
topic ─▶ Research ─▶ Hook ─▶ Script ─▶ Visual ─▶ Growth ─▶ scored content
```

| # | Agent | Job | Consumes | Produces |
|---|-----------|--------------------------------------------------------------|--------------------|-------------------|
| 1 | Research  | Gather angles, facts, and audience context for the topic      | topic              | research brief    |
| 2 | Hook      | Generate and rank opening lines that earn the first 3 seconds | research brief     | ranked hooks      |
| 3 | Script    | Write the full post/script around the winning hook            | brief + hook       | script            |
| 4 | Visual    | Specify shots, captions, on-screen text, and pacing           | script             | visual plan       |
| 5 | Growth    | Score the result and suggest concrete improvements            | everything upstream| quality score     |

Every agent returns **structured JSON**, so each stage is inspectable on its own —
you can see exactly where a weak post went wrong instead of re-rolling the whole
thing.

## Quality scoring

The Growth agent is the differentiator. Rather than ending at "here's your post,"
the pipeline closes the loop: it grades the output against explicit criteria
(hook strength, clarity, payoff, platform fit) and returns a score plus specific
revision notes. That makes the system measurable — you can A/B a prompt change and
see whether the score moves.

## Project layout

```
app.py               Streamlit UI
src/
  agents/            One module per agent, all inheriting a shared BaseAgent
  prompts/           Prompt templates as .txt files — never inlined in code
  models.py          Dataclasses for each agent's output
tests/               pytest suite (parsing + pipeline logic)
examples/            Sample runs and reference outputs
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add your real key
streamlit run app.py
```

Config is read from `.env` via `python-dotenv`. `.env` is gitignored — API keys
are never hardcoded or committed.

## Status

Scaffold only. Agent implementations are not written yet.

## Roadmap

- [x] Phase 1 — Setup: repo, prompt templates, API working
- [ ] Phase 2 — BaseAgent with retry logic + first working agent
- [ ] Phase 3 — Clean JSON outputs from every agent + tests
- [ ] Phase 4 — Chain all 5 agents into one pipeline
- [ ] Phase 5 — Streamlit UI so it's usable without a terminal
- [ ] Phase 6 — Clean up, docs, maybe deploy it
- [ ] Phase 7 — Track quality scores over time and analyze them

Building this in phases and committing as I go, so the history shows how it actually came together.