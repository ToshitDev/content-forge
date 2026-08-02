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

- [x] Project scaffold
- [ ] `BaseAgent` + output dataclasses
- [ ] The five agents, one at a time
- [ ] Pipeline orchestration
- [ ] Streamlit UI
- [ ] Test suite

Building this in phases and committing as I go partly to learn properly, partly so the history shows the actual process.