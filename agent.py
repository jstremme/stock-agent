"""
agent.py — CLI entry point, agent loop, trajectory logging, and message queue.

─── Key Concepts ─────────────────────────────────────────────────────────────

1. THE TOOL USE LOOP
   The Anthropic API is stateless — it has no memory of prior calls. You maintain
   conversation history yourself as a list of message dicts, and send the full
   history on every API call. Claude cannot execute tools directly; it returns a
   description of the tool it wants to call, and YOUR code runs it, then sends
   the results back. Claude may need several tool-call rounds before writing a
   final answer — hence the loop.

2. MESSAGE HISTORY SHAPE
   Messages alternate user/assistant. The `content` field is either a plain string
   (simple user text) or a list of content blocks (assistant turns with tool calls,
   or user turns returning tool results).

3. PROACTIVE BRIEFING
   On launch, a synthetic user message kicks off the agent loop before the user
   types anything. Claude calls tools autonomously, then the session switches to
   interactive mode.

4. MESSAGE QUEUE
   A background thread reads stdin continuously so the user can type while the
   agent is processing. Messages accumulate in a queue and are handled in order,
   similar to how Claude Code accepts queued messages.

5. TRAJECTORY LOGGING
   Every agent action (tool calls, results, messages) is written to a JSONL file
   in the logs/ directory. This lets you replay exactly what the agent did when
   answering a question — useful for debugging and learning.
"""

import json
import os
import sys
import datetime
import threading
import queue as queue_module
from pathlib import Path
from typing import Optional

import anthropic
import typer
from dotenv import load_dotenv

from tools import TOOLS, dispatch_tool

load_dotenv()

app = typer.Typer(
    help="Stock-tracking conversational agent powered by Claude.",
    add_completion=False,
)

# ─── System Prompt ────────────────────────────────────────────────────────────
#
# Sent with every API call (as the `system` parameter, not in the messages list).
# Sets Claude's persistent behavior for the entire session.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a concise financial data assistant. Your job is to fetch and
present real stock data and news so the user can make informed decisions. You do NOT
give investment advice, price predictions, or buy/sell recommendations.

Default tickers for briefings (when no tickers are specified): ASML, CRDO, CRWV.
You can answer questions about ANY publicly traded stock — not just those three.

Rules:
- ALWAYS use tools to fetch stock prices and news. Your training data is stale;
  never state a price, percentage change, or news headline from memory.
- Be concise. Bullets for briefings and comparisons. Prose for analysis.
- Show prices with currency and date. Show changes as both absolute and percentage.
- For news: headline | source | datetime (one line per article). Always include the
  time (HH:MM) alongside the date — the datetime field in tool results contains both.
- When comparing multiple tickers, call their tools in the same response turn
  (Claude can issue parallel tool_use blocks) to minimize round-trips.
- If a tool returns an error, tell the user clearly what failed and why.

Briefing format:
  TICKER  $price  (±X.XX% today)  as of DATE
  News (last 3 days):
    • Headline — Source (YYYY-MM-DD HH:MM)
"""


# ─── Trajectory Logger ────────────────────────────────────────────────────────
#
# Records every agent action to a JSONL file (one JSON object per line).
# JSONL is easy to read (`cat logs/2026-05-20-0.jsonl | python -m json.tool`)
# and easy to analyze programmatically (`jq '.event' logs/2026-05-20-0.jsonl`).
#
# File naming: YYYY-MM-DD-N.jsonl where N increments per session on the same day.
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryLogger:
    def __init__(self, logs_dir: Path):
        logs_dir.mkdir(exist_ok=True)

        today = datetime.date.today().isoformat()
        n = 0
        while (logs_dir / f"{today}-{n}.jsonl").exists():
            n += 1
        self.path = logs_dir / f"{today}-{n}.jsonl"
        self._file = open(self.path, "w", buffering=1)  # line-buffered for real-time reads

    def log(self, event: str, **kwargs) -> None:
        record = {
            "event": event,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            **kwargs,
        }
        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._file.close()


# ─── Agent Loop ───────────────────────────────────────────────────────────────

def run_tool_loop(
    client: anthropic.Anthropic,
    messages: list[dict],
    model: str,
    logger: TrajectoryLogger,
    max_iterations: int = 10,
) -> str:
    """
    The core agentic loop. Calls the Anthropic API repeatedly until Claude
    produces a final text response (stop_reason == "end_turn").

    Each iteration either:
      a) returns Claude's final text answer, or
      b) executes all tool calls Claude requested, appends results, and loops.

    The messages list is mutated in place — the caller's history grows with
    each turn and is passed to the next call, maintaining full context.
    """
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Send the full conversation history on every call — the API is stateless.
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.AuthenticationError:
            return "[Error: Invalid ANTHROPIC_API_KEY. Check your .env file.]"
        except anthropic.RateLimitError:
            return "[Error: Anthropic rate limit hit. Wait a moment and try again.]"
        except anthropic.APIError as e:
            return f"[Anthropic API error: {e}]"

        # CRITICAL: append the assistant turn BEFORE processing tool calls.
        # The API requires tool_result blocks in the following user message to
        # match tool_use blocks already present in history. Wrong order → 400 error.
        messages.append({"role": "assistant", "content": response.content})

        # stop_reason values:
        #   "end_turn"   → Claude is done; final answer is in the text blocks
        #   "tool_use"   → Claude wants to call one or more tools; keep looping
        #   "max_tokens" → response was cut off (rare with max_tokens=4096)
        if response.stop_reason == "end_turn":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            logger.log("assistant_message", content=final_text)
            return final_text

        if response.stop_reason != "tool_use":
            return f"[Unexpected stop_reason: {response.stop_reason}]"

        # ── Tool Use Handling ──────────────────────────────────────────────
        # Claude may request multiple tools in one response (parallel tool use).
        # We execute ALL of them and bundle ALL results into ONE user message.
        # Sending results as separate messages would violate the alternating
        # user/assistant role requirement.
        # ──────────────────────────────────────────────────────────────────

        tool_result_blocks = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            # block.id    — unique ID linking this call to its result
            # block.name  — matches the "name" field in TOOLS schemas
            # block.input — dict of arguments Claude chose to pass
            typer.echo(f"  → {block.name}({block.input})", err=False)
            logger.log("tool_call", tool=block.name, input=block.input, call_id=block.id)

            result_str = dispatch_tool(block.name, block.input)

            logger.log(
                "tool_result",
                tool=block.name,
                call_id=block.id,
                result=json.loads(result_str),
            )

            # tool_use_id links this result back to the tool call that requested it.
            # When Claude issues parallel tool calls, this is how it matches them up.
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        # All tool results as a single user turn — required by the API.
        messages.append({"role": "user", "content": tool_result_blocks})
        # Loop back: Claude reads results, calls more tools or writes final answer.

    return "[Error: agent loop exceeded maximum iterations. Try a simpler question.]"


# ─── Input Queue Thread ───────────────────────────────────────────────────────
#
# Runs in a background daemon thread. Reads stdin continuously so the user can
# type while the agent is processing a previous message. Typed messages queue up
# and are handled in order after the current response completes — similar to how
# Claude Code accepts input during a running operation.
#
# None is used as a sentinel value to signal the main loop to exit.
# ─────────────────────────────────────────────────────────────────────────────

def _input_reader(message_queue: queue_module.Queue, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            line = input()
        except EOFError:
            message_queue.put(None)
            return
        except Exception:
            return
        message_queue.put(line.strip())


# ─── Agent Orchestrator ───────────────────────────────────────────────────────

def run_agent(
    ticker_list: list[str],
    briefing: bool,
    model: str,
    logs_dir: Path,
) -> None:
    """
    Top-level orchestrator. Manages session state, the message queue, and the
    trajectory logger. Each session starts with a fresh message history.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        typer.echo(
            "[Error: ANTHROPIC_API_KEY not found. Copy .env.example to .env and add your key.]"
        )
        raise typer.Exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    logger = TrajectoryLogger(logs_dir)

    # The conversation history is just a list of dicts.
    # It accumulates throughout the session so Claude has full context for
    # follow-up questions. It is NOT persisted between sessions.
    messages: list[dict] = []

    logger.log(
        "session_start",
        model=model,
        tickers=ticker_list,
        trajectory_file=str(logger.path),
    )

    typer.echo(f"\n{'─'*60}")
    typer.echo(f"  Stock Agent  |  {model}")
    typer.echo(f"  Default tickers: {', '.join(ticker_list)}")
    typer.echo(f"  Trajectory: {logger.path}")
    typer.echo(f"{'─'*60}")

    if briefing:
        # Inject a synthetic user turn to trigger the proactive briefing.
        # The agent loop runs autonomously (calling price and news tools for each
        # ticker) before the user has typed anything.
        briefing_prompt = (
            f"Please give me a briefing for: {', '.join(ticker_list)}. "
            f"For each ticker get today's price with day change percentage, "
            f"and the top news from the last 3 days."
        )
        typer.echo("\nFetching briefing...\n")
        logger.log("user_message", content=briefing_prompt, source="auto_briefing")
        messages.append({"role": "user", "content": briefing_prompt})

        briefing = run_tool_loop(client, messages, model, logger)
        typer.echo(briefing)
        typer.echo(f"\n{'─'*60}")

    typer.echo("\nYou can ask about any stock. Type 'quit' or Ctrl-C to exit.\n")

    # Start the background input-reader thread.
    # daemon=True means it exits automatically when the main thread exits.
    message_queue: queue_module.Queue = queue_module.Queue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=_input_reader,
        args=(message_queue, stop_event),
        daemon=True,
    )
    input_thread.start()

    turn_count = 0

    # Print the prompt once to signal we're ready for input
    typer.echo("You: ", nl=False)

    while True:
        # Block until the user sends a message (or queued messages arrive).
        # If the agent was processing while the user typed, those messages
        # are already in the queue and are picked up immediately here.
        try:
            user_input = message_queue.get(timeout=0.1)
        except queue_module.Empty:
            continue

        if user_input is None:
            # EOF sentinel from the input thread
            typer.echo("\nGoodbye.")
            break

        if not user_input:
            typer.echo("You: ", nl=False)
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            typer.echo("Goodbye.")
            break

        turn_count += 1
        logger.log("user_message", content=user_input, turn=turn_count)

        # Append user turn to history and run the agent loop.
        # After this call, messages contains the user turn, all intermediate
        # tool calls/results, and Claude's final response — preserved for context.
        messages.append({"role": "user", "content": user_input})
        response = run_tool_loop(client, messages, model, logger)

        typer.echo(f"\nAssistant: {response}\n")

        # Show "You: " prompt again after response
        # Check if there are already queued messages waiting
        if not message_queue.empty():
            typer.echo("[Processing queued message...]\n")
        else:
            typer.echo("You: ", nl=False)

    stop_event.set()
    logger.log("session_end", turns=turn_count)
    logger.close()
    typer.echo(f"\nTrajectory saved to: {logger.path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    tickers: Optional[str] = typer.Option(
        None,
        "--tickers",
        "-t",
        help="Comma-separated tickers for briefing. Default: ASML,CRDO,CRWV",
    ),
    briefing: bool = typer.Option(
        False,
        "--briefing",
        help="Run a briefing on launch (prices + news for each tracked ticker).",
    ),
    model: str = typer.Option(
        "claude-sonnet-4-6",
        "--model",
        "-m",
        help=(
            "Anthropic model ID. Options:\n"
            "  claude-sonnet-4-6         (default, best balance)\n"
            "  claude-haiku-4-5-20251001 (faster, cheaper)\n"
            "  claude-opus-4-7           (most capable, most expensive)"
        ),
    ),
    logs_dir: str = typer.Option(
        "logs",
        "--logs-dir",
        help="Directory for trajectory log files. Created if it doesn't exist.",
    ),
) -> None:
    """
    Launch the stock-tracking conversational agent.

    Opens in interactive chat mode. You can ask about any publicly traded stock.
    Use --briefing to fetch an automatic briefing summary on launch.

    Examples:\n
      uv run python agent.py\n
      uv run python agent.py --briefing\n
      uv run python agent.py --briefing --tickers NVDA,TSM,INTC\n
      uv run python agent.py --model claude-haiku-4-5-20251001
    """
    ticker_list = (
        [t.strip().upper() for t in tickers.split(",")]
        if tickers
        else ["ASML", "CRDO", "CRWV"]
    )
    run_agent(
        ticker_list=ticker_list,
        briefing=briefing,
        model=model,
        logs_dir=Path(logs_dir),
    )


if __name__ == "__main__":
    app()
