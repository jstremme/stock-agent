"""
tools.py — Data-fetching layer for the stock agent.

This module is intentionally separate from agent.py so the data logic is
easy to read, test, and modify without touching the agent orchestration.

Two external services are used:
  - yfinance: Python wrapper around Yahoo Finance. No API key needed.
  - Finnhub:  Financial data + news API. Free tier: 60 calls/min.
              Requires FINNHUB_API_KEY in your .env file.

At the bottom of this file, TOOLS is a list of JSON schema dicts that describe
these functions to Claude. The schemas are what allow Claude to "know" how to
call each tool — think of them as docstrings Claude reads at inference time.
"""

import json
import os
import datetime
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"


def get_stock_price(symbol: str, date: str | None = None) -> dict:
    """
    Fetch the closing price for a stock ticker.

    Latest price: uses yfinance fast_info, a lightweight scrape that avoids
    downloading full fundamental data. Much faster than .info for price-only lookups.

    Specific date: downloads a small window around that date and finds the
    closest prior trading day (handles weekends and market holidays automatically).
    """
    ticker = yf.Ticker(symbol)

    if date is None:
        info = ticker.fast_info
        price = info.last_price
        currency = getattr(info, "currency", "USD")

        prev_close = getattr(info, "previous_close", None)
        change_pct = None
        if prev_close and prev_close != 0:
            change_pct = round(((price - prev_close) / prev_close) * 100, 2)

        return {
            "symbol": symbol.upper(),
            "date": "latest",
            "price": round(float(price), 2),
            "currency": currency,
            "change_pct": change_pct,
            "previous_close": round(float(prev_close), 2) if prev_close else None,
        }
    else:
        # Download a 7-day window to catch weekends/holidays around the target date.
        target = datetime.date.fromisoformat(date)
        start = target - datetime.timedelta(days=5)
        end = target + datetime.timedelta(days=2)
        hist = ticker.history(start=str(start), end=str(end))

        if hist.empty:
            return {"error": f"No trading data found for {symbol} around {date}"}

        hist.index = hist.index.date
        available = [d for d in hist.index if d <= target]
        if not available:
            return {"error": f"No trading data found for {symbol} on or before {date}"}

        actual_date = max(available)
        row = hist.loc[actual_date]

        daily_change_pct = None
        if row["Open"] and row["Open"] != 0:
            daily_change_pct = round(((row["Close"] - row["Open"]) / row["Open"]) * 100, 2)

        return {
            "symbol": symbol.upper(),
            "date": str(actual_date),
            "price": round(float(row["Close"]), 2),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "volume": int(row["Volume"]),
            "daily_change_pct": daily_change_pct,
            "note": f"Closest trading day to {date}" if str(actual_date) != date else None,
        }


def get_stock_history(symbol: str, period: str = "1mo") -> dict:
    """
    Fetch historical OHLCV data for a ticker over a given period.

    yfinance automatically picks the right interval (daily, weekly) based on
    the period — you don't specify it manually.

    Returns a pre-computed total_change_pct so Claude can answer trend questions
    without needing to do arithmetic on the full rows list.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)

    if hist.empty:
        return {"error": f"No historical data found for {symbol} with period={period}"}

    rows = []
    for date, row in hist.iterrows():
        rows.append({
            "date": str(date.date()) if hasattr(date, "date") else str(date),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    closes = [r["close"] for r in rows]
    first_close = closes[0]
    last_close = closes[-1]
    total_change_pct = (
        round(((last_close - first_close) / first_close) * 100, 2) if first_close else None
    )

    return {
        "symbol": symbol.upper(),
        "period": period,
        "data_points": len(rows),
        "start_date": rows[0]["date"],
        "end_date": rows[-1]["date"],
        "start_close": first_close,
        "end_close": last_close,
        "total_change_pct": total_change_pct,
        "rows": rows,
    }


def get_news(
    symbol: str | None = None,
    query: str | None = None,
    days_back: int = 7,
) -> dict:
    """
    Fetch financial news from Finnhub.

    Two modes:
      symbol provided → company-specific news via /company-news endpoint
      query only      → general market news via /news?category=general,
                        filtered client-side by checking headline and summary

    Results are capped at 10 articles to keep tool responses concise.
    Large tool results inflate token usage and slow the agent loop.
    """
    if not FINNHUB_API_KEY:
        return {
            "error": (
                "FINNHUB_API_KEY not set. "
                "Get a free key at https://finnhub.io/register and add it to .env"
            )
        }

    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=days_back)

    if symbol:
        url = f"{FINNHUB_BASE}/company-news"
        params = {
            "symbol": symbol.upper(),
            "from": str(from_date),
            "to": str(today),
            "token": FINNHUB_API_KEY,
        }
    else:
        url = f"{FINNHUB_BASE}/news"
        params = {"category": "general", "token": FINNHUB_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        return {"error": f"Network error fetching news: {e}"}

    if resp.status_code == 401:
        return {"error": "Invalid Finnhub API key (401). Check FINNHUB_API_KEY in .env."}
    if resp.status_code == 429:
        return {"error": "Finnhub rate limit hit (429). Free tier: 60 calls/min."}
    if resp.status_code != 200:
        return {"error": f"Finnhub API error: HTTP {resp.status_code}"}

    articles_raw = resp.json()

    if not symbol and query:
        q = query.lower()
        articles_raw = [
            a for a in articles_raw
            if q in a.get("headline", "").lower() or q in a.get("summary", "").lower()
        ]

    articles = []
    for item in articles_raw[:10]:
        ts = item.get("datetime", 0)
        try:
            dt_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            dt_str = "unknown"
        articles.append({
            "datetime": dt_str,
            "headline": item.get("headline", ""),
            "source": item.get("source", ""),
            "summary": item.get("summary", "")[:300],
            "url": item.get("url", ""),
        })

    return {
        "source_type": "company" if symbol else "general",
        "symbol": symbol.upper() if symbol else None,
        "query": query,
        "days_back": days_back,
        "article_count": len(articles),
        "articles": articles,
    }


def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    Routes a tool call from Claude to the correct Python function.

    Returns a JSON string — the Anthropic API requires tool results to be
    strings (or a list of content blocks). JSON strings let Claude parse
    structured data and reason about it (e.g., compare two prices).

    All exceptions are caught here so the agent loop never crashes on a
    failed tool call. Claude receives an error dict and surfaces it to the user.
    """
    try:
        if tool_name == "get_stock_price":
            result = get_stock_price(**tool_input)
        elif tool_name == "get_stock_history":
            result = get_stock_history(**tool_input)
        elif tool_name == "get_news":
            result = get_news(**tool_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        result = {"error": str(e), "tool": tool_name, "input": tool_input}

    return json.dumps(result)


# ─── Tool JSON Schemas ────────────────────────────────────────────────────────
#
# These dicts are what Claude reads to understand what tools are available
# and how to call them. The "description" field is the most important part —
# it tells Claude WHEN to use this tool vs. another. Think of it as a docstring
# Claude reads at inference time to decide which tool fits the user's question.
#
# The "input_schema" follows JSON Schema draft-07. "required" lists fields
# Claude MUST provide; all others are optional with sensible defaults.
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_stock_price",
        "description": (
            "Fetches the closing price for a stock ticker symbol. "
            "If no date is given, returns the most recent market close with the day's % change. "
            "If a date (YYYY-MM-DD) is provided, returns the closing price on that trading day "
            "(or the nearest prior trading day if it was a weekend or holiday). "
            "Use for: current price, a specific historical price, or today's change. "
            "Do NOT use for multi-day trends — use get_stock_history for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'ASML', 'CRDO', 'CRWV', 'NVDA'.",
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Optional. Date in YYYY-MM-DD format. "
                        "Omit to get the latest available closing price."
                    ),
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_stock_history",
        "description": (
            "Fetches historical daily OHLCV (open/high/low/close/volume) data for a stock. "
            "The response includes a pre-computed total_change_pct for the full period. "
            "Use for: trends, percentage gains over a time window, volatility, or comparing "
            "performance between multiple tickers over the same period. "
            "Supported periods: '1d','5d','1mo','3mo','6mo','1y','2y','5y','ytd','max'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol.",
                },
                "period": {
                    "type": "string",
                    "description": "Time period for history. Defaults to '1mo'.",
                    "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"],
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_news",
        "description": (
            "Fetches recent financial and market news from Finnhub. "
            "Two modes: (1) provide 'symbol' to get news specifically about that company "
            "(e.g., symbol='ASML' returns ASML-related articles); "
            "(2) provide 'query' without a symbol to search general market news by keyword "
            "(e.g., query='AI data centers' or query='semiconductor tariffs'). "
            "Use 'days_back' to control the lookback window (default: 7, max: 30). "
            "Returns up to 10 articles with headlines, summaries, sources, and URLs. "
            "ALWAYS use this tool for news — never recall or guess news from training data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": (
                        "Optional. Ticker for company-specific news. "
                        "E.g. 'ASML' to get articles specifically about ASML."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional. Keyword or phrase to filter general market news. "
                        "Only applied when no symbol is given."
                    ),
                },
                "days_back": {
                    "type": "integer",
                    "description": "How many days back to search. Default 7, max 30.",
                    "minimum": 1,
                    "maximum": 30,
                },
            },
            "required": [],
        },
    },
]
