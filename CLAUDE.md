# ContentForge — Project Context

## What this project is
A 5-agent LLM content pipeline: Research → Hook → Script → Visual → Growth.
Each agent has a prompt template in src/prompts/, returns structured JSON,
and its output feeds the next agent. Streamlit UI on top. This is a resume
project for a data science student — code must be readable and explainable
in an interview.

## Tech stack
- Python 3.11+, Anthropic API (anthropic SDK), Streamlit, pytest
- Config via .env + python-dotenv. NEVER hardcode or commit API keys.

## Architecture rules
- All agents inherit from a single BaseAgent class in src/agents/base.py
- Agent outputs are dataclasses defined in src/models.py
- Prompts live as .txt templates in src/prompts/ — never inline in code
- Every agent must parse its LLM response into JSON safely (strip
  markdown fences, handle malformed output, raise a clear error)

## Working style (important)
- Work on ONE phase/task at a time. Do not build ahead of what I ask.
- Keep functions small, add docstrings, explain non-obvious choices in comments.
- After each task, tell me what changed and suggest a commit message,
  but let ME run the git commit.
- Write or update tests in tests/ when you touch parsing or pipeline logic.