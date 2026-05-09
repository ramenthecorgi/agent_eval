# Agent Eval

A prototype for evaluating an AI research agent that answers questions using Wikipedia. Includes a chat UI, trace viewer, and an eval suite with LLM judges.

## Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

> **Optional:** Set `WIKI_TOKEN` in `.env` for authenticated Wikipedia access (higher rate limits).

## Running the server

```bash
cd backend
uvicorn main:app --reload
```

The app is now available at **http://localhost:8000**.

- **Chat UI** — `http://localhost:8000/` — ask the agent questions
- **Trace viewer** — `http://localhost:8000/traces.html` — inspect tool calls and token usage for past runs

## Running evals

With the server running, in a separate terminal:

```bash
python evals/run_evals.py
```

Results are written to `evals/eval_run_<timestamp>.json` and a CSV summary to `evals/eval_results_<date>.csv`. Open the trace viewer to browse results by run.

## Running tests

```bash
pytest tests/
```

## Project layout

```
backend/        FastAPI server, agent loop, tools, guardrails, tracer
evals/          Eval cases, LLM judges, and the eval runner
frontend/       Static HTML/JS for the chat UI and trace viewer
tests/          Unit and integration tests
```

## Demo

**Eval runs** — Browse results from automated eval runs. Each case shows the question, the agent's answer, and scores from LLM judges across dimensions like factual accuracy and safety.

<img width="1500" height="651" alt="Eval runs" src="https://github.com/user-attachments/assets/1a412e9e-de07-4298-8c1f-7b6f08f55a69" />

**Trace viewer** — Inspect every step of a past agent run: tool calls, Wikipedia lookups, token usage, and latency. Useful for debugging and understanding model behavior.

<img width="1488" height="656" alt="Trace viewer" src="https://github.com/user-attachments/assets/f4455a6a-3e63-4fe5-af45-d68fdf091fe6" />

**Chat UI** — Send a question and watch the agent respond in real time. The agent searches Wikipedia to ground its answers and applies guardrails before replying.

<img width="1319" height="701" alt="Chat UI" src="https://github.com/user-attachments/assets/e7270623-8433-4737-bf85-9b81ae87fab7" />
