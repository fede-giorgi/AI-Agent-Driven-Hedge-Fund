# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the application:**
```bash
python main.py          # Interactive mode (prompts for capital, portfolio, risk, tickers)
python main.py --debug  # Debug mode: hardcoded $10k capital, risk=5, tickers=[AAPL, MSFT, NVDA]
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Environment setup** — create `.env` with:
```
GOOGLE_API_KEY=...
GEMINI_API_KEY=...
FINDAT_API_KEY=...      # financialdatasets.ai
BRAVE_API_KEY=...       # Brave Search MCP
```

## Architecture

This is an educational multi-agent system that simulates a hedge fund using Warren Buffett's value investing principles. The system is entirely sequential Python with no web server or test suite.

### Agent Pipeline (defined in `main.py`)

1. **Research Agent** (`ai_agents/research_agent.py`) — Async; gathers financial data per ticker using Financial Datasets API and Brave Search MCP. Guided by a "research brief" from the Buffett agent.
2. **Warren Buffett Agent** (`ai_agents/warren_buffet_agent.py`) — Analyzes each ticker's `FinancialSummary` using domain tools (DCF, moat, management, consistency); outputs a `WarrenBuffettSignal` (bullish/bearish/neutral + confidence 0-100).
3. **Trading Loop (10 iterations):**
   - **Portfolio Manager** (`ai_agents/portfolio_and_risk_manager.py`) — Proposes trades respecting risk profile (1-10) and capital constraints.
   - **Monitor** (`ai_agents/monitor.py`) — Validates trades: no shorting, sufficient capital, valid tickers/prices.
   - **What-If Agent** (`ai_agents/what_if_agent.py`) — Devil's advocate; generates a concrete counter-proposal. Skipped on iteration 10.
4. **Final Orchestrator** (`ai_agents/final_orchestrator_agent.py`) — Reviews all 10 iteration histories, selects or synthesizes the best trade list.

### Key Data Models (`classes/financial_summary.py`)

All agents communicate via Pydantic models:
- `FinancialSummary` — 50+ fields (valuation ratios, profitability, leverage, growth)
- `WarrenBuffettSignal` — `signal`, `confidence`, `reasoning`
- `ResearchAgentOutput` — List of `Result` per ticker
- `ToolStatus` — Tracks which data-gathering tools succeeded/failed

### LLM Configuration (`llm.py`)

Single `get_llm()` function returns `ChatGoogleGenerativeAI` with `gemini-2.5-flash`, `temperature=0`.

### Financial Tools (`tools/`)

Each tool wraps the Financial Datasets API (`https://api.financialdatasets.ai/`):
- `get_financials.py` — Income statement, balance sheet, cash flow
- `get_metrics.py` — 50+ ratios (annual/quarterly)
- `get_financial_line_items.py` — Granular line items
- `get_stock_prices.py` — Historical OHLCV data
- `calculate_intrinsic_value.py` — 2-stage DCF model (10yr growth + terminal)
- `analyze_*.py` — Domain analysis functions (moat, management, consistency, etc.)
- `mcp.py` — Async MCP client wrapping Brave Search; adapts MCP tools to LangChain format

### MCP Integration

The Brave Search MCP server is started as a subprocess via `mcp-use`. The Research Agent uses it for news/sentiment. If MCP connection fails, the agent falls back gracefully to financial API tools only. Circuit breaker: max 10 news searches per ticker.

### Risk Profiles

Risk level 1-10 affects portfolio behavior:
- **1-3 (Low):** Capital preservation; level 1 = no buys
- **4-7 (Mid):** Balanced; 5-15% cash buffer
- **8-10 (High):** Aggressive; <5% cash buffer
