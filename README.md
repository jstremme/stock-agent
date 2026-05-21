# Stock Agent

A conversational CLI agent for tracking stock prices and financial news, built on
[Claude Sonnet 4.6](https://www.anthropic.com/claude). Designed to be **educational** — the code
is deliberately simple so you can trace exactly how a Claude tool-use agent loop works.

Ask about any publicly traded stock. The agent opens in interactive chat mode. Pass
`--briefing` to get an automatic briefing summary for your tracked tickers (ASML, CRDO,
CRWV by default) before the chat begins.

---

## Project Overview

**What it does**

- Fetches real-time and historical stock prices via [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance, no API key)
- Retrieves financial news per ticker or by keyword via [Finnhub](https://finnhub.io) (free API key)
- Runs a proactive briefing on launch: current price + day change + recent headlines for each tracked ticker
- Accepts queued messages — you can type your next question while the agent is still responding
- Logs every agent action (tool calls, results, responses) to a JSONL trajectory file in `logs/`

**Who it's for**

Someone who wants to do a quick daily check on a handful of stocks before the market
opens, or ask follow-up questions like "how does ASML compare to CRDO over the last month?"
or "any news about AI chip tariffs?". Also useful as a learning example for Anthropic's
tool-use API.

---

## Agent Architecture

### The Tool Use Loop

Claude cannot call tools directly. When you ask a question that requires real data,
Claude returns a response with `stop_reason="tool_use"` describing which tool it wants
to call and with what arguments. Your code executes the tool and sends the result back
as a new message. Claude then reads the result and either calls more tools or writes
a final answer.

```
User input
    │
    ▼
client.messages.create(model, tools, full_history)
    │
    ├─ stop_reason == "tool_use"
    │       │
    │       ├─ Execute each tool_use block → get result string
    │       ├─ Bundle all results into one user message
    │       └─ Append to history, loop back ──────────────────┐
    │                                                          │
    └─ stop_reason == "end_turn"                               │
            │                                                  │
            ▼                                           (repeat)
        Print final response
        Wait for next user input
```

**Why a loop?** Claude may need multiple rounds of tool calls. For a briefing on
three tickers, it typically calls `get_stock_price` for all three in parallel (one
`tool_use` response with three blocks), then `get_news` for all three, then writes the
briefing — three iterations of the loop.

**Why does history grow?** The Anthropic API is stateless. Every call sends the full
conversation history so Claude has context for follow-up questions. This is why
`messages` is a list that accumulates throughout the session.

**Why append before processing?** The API requires that `tool_result` blocks in a user
message match `tool_use` blocks already present in the assistant's prior turn. Appending
the assistant turn before dispatching tools ensures valid ordering.

### Parallel Tool Calls

Claude can issue multiple `tool_use` blocks in a single response. When it does, all
results must be bundled into a **single** user message (one `tool_result` block per call,
matched by `tool_use_id`). This is how the briefing fetches prices for all three tickers
simultaneously instead of one at a time.

### Message History Shape

After a briefing + one follow-up, the `messages` list looks like:

```
[0] user      "Briefing for ASML, CRDO, CRWV..."
[1] assistant [ToolUseBlock(get_stock_price, ASML), ToolUseBlock(get_stock_price, CRDO), ...]
[2] user      [tool_result(ASML price), tool_result(CRDO price), ...]
[3] assistant [ToolUseBlock(get_news, ASML), ToolUseBlock(get_news, CRDO), ...]
[4] user      [tool_result(ASML news), tool_result(CRDO news), ...]
[5] assistant "Here is your briefing..."
[6] user      "How did ASML do last quarter?"
[7] assistant [ToolUseBlock(get_stock_history, ASML, 3mo)]
[8] user      [tool_result(ASML history)]
[9] assistant "Over the last quarter, ASML..."
```

### Tools

| Tool | Source | Purpose |
|---|---|---|
| `get_stock_price` | yfinance | Latest close or specific-date price |
| `get_stock_history` | yfinance | OHLCV data for a period (1d → max) |
| `get_news` | Finnhub | Company news by ticker or keyword search |

Tool schemas (in `tools.py`) define what arguments Claude can pass. The `description`
field in each schema is what Claude reads at inference time to decide which tool fits
the user's question — quality descriptions improve tool selection accuracy significantly.

### Message Queue

A background daemon thread reads stdin continuously. When the agent is processing a
response, anything you type goes into a `queue.Queue`. After the current response
completes, queued messages are processed in order — similar to how Claude Code accepts
input during a running task.

### Trajectory Logging

Every session writes to `logs/YYYY-MM-DD-N.jsonl` (where N increments for multiple
sessions on the same day). Each line is a JSON object recording one event:

```jsonl
{"event": "session_start", "timestamp": "2026-05-20T09:15:00", "model": "claude-sonnet-4-6", "tickers": ["ASML"]}
{"event": "user_message",  "timestamp": "2026-05-20T09:15:01", "content": "What is ASML's price?"}
{"event": "tool_call",     "timestamp": "2026-05-20T09:15:02", "tool": "get_stock_price", "input": {"symbol": "ASML"}}
{"event": "tool_result",   "timestamp": "2026-05-20T09:15:03", "tool": "get_stock_price", "result": {"price": 680.50}}
{"event": "assistant_message", "timestamp": "2026-05-20T09:15:04", "content": "ASML is trading at..."}
{"event": "session_end",   "timestamp": "2026-05-20T09:20:00", "turns": 3}
```

Read a trajectory: `cat logs/2026-05-20-0.jsonl | python -m json.tool`
Filter tool calls: `grep '"event": "tool_call"' logs/2026-05-20-0.jsonl | jq .`

---

## Model & Cost

This agent runs on **Claude Sonnet 4.6** by default — the best balance of speed and
analytical quality for correlating news to price movements and answering nuanced
cross-stock questions.

The Anthropic API is pay-as-you-go. Get an API key at **https://console.anthropic.com**.

**Estimated cost with Sonnet 4.6** at 2–3 sessions/day (briefing + ~5 turns):
- ~20–30k input tokens + ~2–3k output tokens per session ≈ **$0.10–$0.15/session**
- Monthly: roughly **$6–14/month**

You can switch models with the `--model` flag if needed:

| Model | `--model` value | Input / Output (per 1M tokens) | When to use |
|---|---|---|---|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | $3 / $15 | **Default** |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | ~$1 / $5 | Simple lookups, lower cost |
| Claude Opus 4.7 | `claude-opus-4-7` | $5 / $25 | Deep analysis, higher cost |

---

## Possible Enhancements

- **Session persistence** — save `messages` to a JSON file between sessions so Claude
  has context from yesterday's conversation
- **`get_fundamentals` tool** — P/E ratio, market cap, 52-week range via `yf.Ticker.info`
- **Earnings calendar tool** — upcoming earnings dates via Finnhub's
  `/api/v1/calendar/earnings` endpoint
- **ASCII sparklines** — render a simple in-terminal price chart using the history data
- **`--watch` mode** — re-run the briefing on a schedule (every 30 min, etc.)
- **Migrate to `claude-code-sdk`** — Anthropic's higher-level agent SDK (used by Claude
  Code) eliminates the manual tool loop. Once you understand the loop from this project,
  the SDK lets you focus on tools and prompts instead of orchestration plumbing. See
  https://github.com/anthropics/claude-code-sdk for details.

---

## How to Run

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — modern Python package manager

```bash
# Install uv (one-time, system-wide)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url> stock-agent
cd stock-agent

# 2. Install dependencies (uv creates a .venv automatically)
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY  — from https://console.anthropic.com/settings/keys
#   FINNHUB_API_KEY    — free key from https://finnhub.io/register
```

### Launch

```bash
# Default: open straight to interactive chat
uv run python agent.py

# Run briefing on launch (prices + news for ASML, CRDO, CRWV)
uv run python agent.py --briefing

# Briefing with custom tickers
uv run python agent.py --briefing --tickers NVDA,TSM,INTC

# Use a cheaper/faster model
uv run python agent.py --model claude-haiku-4-5-20251001

# Use the most capable model
uv run python agent.py --model claude-opus-4-7

# Custom log directory
uv run python agent.py --logs-dir ~/stock-logs

# See all options
uv run python agent.py --help
```

### Example Questions

```
What was CRWV's closing price on May 15?
Compare ASML and CRDO performance over the last 3 months.
Any news about AI data center demand this week?
What's NVDA doing today?
Show me ASML's 1-year trend.
Any semiconductor tariff news in the last 2 weeks?
```

### Reading Trajectory Logs

```bash
# Pretty-print a session log
cat logs/2026-05-20-0.jsonl | python -m json.tool

# Show only tool calls (requires jq)
grep '"event": "tool_call"' logs/2026-05-20-0.jsonl | jq .

# Show the conversation (user + assistant messages only)
grep -E '"event": "(user|assistant)_message"' logs/2026-05-20-0.jsonl | jq '{event, content}'
```
