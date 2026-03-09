<div align="center">

# AI-Agent-Driven Hedge Fund

<img src="/utils/LOGO-AI-AGENT-HEDGE-FUND.png" width="670" alt="Logo">
<br><br>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.14%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green?style=for-the-badge)
![FinancialDatasets](https://img.shields.io/badge/Data-FinancialDatasets.ai-blue?style=for-the-badge)

</div>

---

> 🎓 **Educational Project**
>
> This project was built to understand how to fully leverage LangChain in a real environment and make agents interact with each other meaningfully. We learnt a lot, broke a lot, and that's what matters.
>
> Two known limitations we are aware of: the fixed-iteration trading loop could be replaced by a convergence threshold (descent-style stopping); and the backtesting is a single-point snapshot rather than a proper walk-forward test. A walk-forward approach would re-run the full pipeline at multiple historical dates, track equity curves over time, and compute risk-adjusted metrics like Sharpe ratio and max drawdown, significant LLM cost and execution time make that out of scope here.

---

An autonomous Multi-Agent System (MAS) that simulates a hedge fund end-to-end. Agents handle data collection, value analysis, trade proposal, compliance validation, and final decision, all orchestrated sequentially via **LangChain** with either **Google Gemini** or **Anthropic Claude**.

---

## 🎬 Demo

![Demo](demo.gif)

---

## System Architecture

The pipeline is strictly sequential. Each stage produces an output that becomes the next stage's input as outlined in this flowchart.

<img src="/utils/Flowchart.png">

---

## Agent Pipeline

### 1 — Research Agent

Runs **asynchronously** per ticker. Calls all 8 FinancialDatasets.ai endpoints and structures the results into a `FinancialSummary` Pydantic object with ~60 typed fields including 8-year historical arrays, news headlines, segment revenues, insider trade activity, and analyst consensus estimates.

### 2 — Warren Buffett Agent

Runs **8 domain analysis tools** sequentially against the `FinancialSummary` input class, then issues a `WarrenBuffettSignal`:

| Tool | What it measures |
|------|-----------------|
| `check_fundamentals` | ROE, ROIC, debt levels, operating margin, liquidity (max score 9) |
| `check_consistency` | Multi-year earnings CAGR + monotonic growth (max score 4) |
| `check_moat` | Historical ROE consistency, margin stability, ROIC (max score 4) |
| `check_management` | Multi-year buyback track record, dividend history (max score 2) |
| `check_book_value_growth` | Book value per share CAGR + period-by-period consistency (max score 5) |
| `check_intrinsic_value` | 3-stage DCF with owner earnings; yields margin of safety vs current price |
| `check_pricing_power` | Gross margin trend + absolute level (max score 5) |
| `check_qualitative_factors` | News headlines, insider buy/sell activity, analyst EPS & revenue consensus |

The final signal is **bullish / neutral / bearish** with a **confidence score 0–100** that directly drives position sizing downstream.

### 3 — Trading Loop (10 iterations, 3 in debug)

Each iteration runs three agents in sequence:

- **Portfolio Manager** proposes trades respecting risk profile (1–10) and available capital. Position sizing formula: `(confidence/100) × (risk_profile/10) × total_capital`, with a 30% per-ticker cap.
- **Monitor Agent** acts as a compliance officer. Validates no-shorting, budget constraints, and schema correctness. If invalid, it returns violations for the Portfolio Manager to address next iteration.
- **What-If Agent** plays devil's advocate. Identifies the single biggest risk or inefficiency and proposes a concrete executable counter-scenario. Skipped on the final iteration.

### 4 — Final Orchestrator

Receives the full iteration history (all PM proposals, monitor checks, what-if critiques) plus the Warren Buffett signals, and selects or synthesises the single best trade plan. BEARISH signal = no buy. BULLISH + high confidence = reward with allocation.

The iteration debate is also compressed into a readable summary panel shown to the user, but the orchestrator always receives the full raw history for highest-quality decisions.

---

## 📐 Backtesting

When a `backtesting_date` is provided, the system runs a **single-point comparison** between the agent portfolio's actual return and three canonical financial benchmarks over the holding period (`backtesting_date` to today).

### Benchmarks

| Benchmark | How it's computed | Why it matters |
|-----------|-------------------|----------------|
| **Risk-Free Rate** | 4.5% annual T-bill proxy, scaled linearly: `0.045 × (days / 365)` | The floor — any strategy should beat cash |
| **1/N Equal-Weight** | `initial_capital / N` allocated equally across all researched tickers, shares bought at `backtesting_date` prices, held passively | DeMiguel et al. (2009) showed naive equal-weighting is a surprisingly hard benchmark to beat out-of-sample |
| **S&P 500 (SPY)** | SPY close at `backtesting_date` vs SPY close today (2 API calls) | Standard benchmark used by institutional managers |

### Calculation

```
agent_start  = Σ(shares × price_at_start) + remaining_cash
agent_end    = Σ(shares × price_today)    + remaining_cash
agent_return = (agent_end - agent_start) / agent_start

alpha = agent_return - benchmark_return   (positive = outperformed)
```

`price_at_start` comes from the Research Agent run at `backtesting_date` — no extra API cost. `price_today` requires one fresh call per held ticker plus 2 calls for SPY.

> **Limitation:** This is a single-point evaluation, not a walk-forward backtest. A proper walk-forward test would require re-running the full pipeline across many historical dates, which means hundreds of additional LLM calls per week of history and significant API cost — beyond the scope of this educational demo.

---

## Data Sources

All 8 tools use the same `FINDAT_API_KEY`. No additional credentials are needed.

| Tool | Endpoint | Data |
|------|----------|------|
| `get_financials` | `/financials` | Income statement, balance sheet, cash flow |
| `get_metrics` | `/financial-metrics` | 50+ ratios (P/E, ROE, margins, ROIC, etc.) |
| `get_financial_line_items` | `/financials/search` | Granular line items, 8-year history |
| `get_stock_prices` | `/prices` | OHLCV price history |
| `get_company_news` | `/news` | Recent news headlines & sources |
| `get_segmented_revenues` | `/financials/segmented-revenues` | Revenue by product / geography |
| `get_insider_trades` | `/insider-trades` | Form 4 executive buy/sell filings |
| `get_analyst_estimates` | `/analyst-estimates` | Consensus revenue & EPS forecasts |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangChain (Python) — tool binding, structured output, async loops |
| LLM | Google Gemini (`gemini-3.1-pro-preview`) via `langchain-google-genai` or Anthropic Claude (`claude-opus-4-6`) via `langchain-anthropic` |
| Data models | Pydantic — `FinancialSummary`, `WarrenBuffettSignal`, `ResearchAgentOutput` |
| Financial data | [FinancialDatasets.ai](https://docs.financialdatasets.ai) — single API key, 8 endpoints |
| Terminal UI | [Rich](https://github.com/Textualize/rich) — tables, panels, progress bars, Markdown |

---

## Installation

**1. Clone**
```bash
git clone https://github.com/fede-giorgi/AI-Agent-Driven-Hedge-Fund.git
cd AI-Agent-Driven-Hedge-Fund
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
mv env.example .env
```

Edit `.env`:
```env
FINDAT_API_KEY=your_financialdatasets_key      # required — covers all 8 data tools

GOOGLE_API_KEY=your_gemini_key                 # required if using Google Gemini
ANTHROPIC_API_KEY=your_anthropic_key           # required only if LLM_PROVIDER=anthropic
```

**5. Run**
```bash
python main.py
```

---

## Usage

```bash
python main.py
```

The session prompts you for:

| Prompt | Options |
|--------|---------|
| **Capital** | Any amount 1 to 1,000,000 |
| **LLM** | Google Gemini or Anthropic Claude; choose model |
| **Risk Profile** | 1 (Ultra Conservative) to 10 (Highly Speculative) |
| **Backtesting date** | Optional `YYYY-MM-DD`; enables benchmark comparison |
| **Ticker selection** | `default` (AAPL, MSFT, NVDA, GOOGL, META) · `auto` (5 diversified from ~600-stock universe) · `custom` (up to 5 manual tickers) |
| **Portfolio** | Enter existing holdings or press `no` to start with cash only |

Agents run sequentially and print their outputs live to the terminal.

---

## Debug Mode

`--debug` skips all prompts and uses hardcoded defaults for fast iteration:

```bash
python main.py --debug
```

| Setting | Debug value |
|---------|------------|
| Capital | $100,000 |
| Risk Profile | 5 (Balanced) |
| Tickers | AAPL, MSFT, NVDA, GOOGL, META |
| Iterations | **3** (vs 10 in interactive) |
| Backtesting | Enabled, today minus 90 days |
| LLM | Reads `$LLM_PROVIDER` / `$LLM_MODEL` env vars, defaults to `google` / `gemini-3.1-pro-preview` |

To test Anthropic in debug mode:
```bash
LLM_PROVIDER=anthropic LLM_MODEL=claude-opus-4-6 python main.py --debug
```

---

## 👥 Contributors

| Name | Role |
|------|------|
| **Luca Barattini** | Main lead — system architecture, agent pipeline, LLM integration |
| **Federico Giorgi** | Co-developer, repository setup, backtesting strategy |
| **Blanca Caballero** | Mentoring, peer reviewer and contributor |
| **Myriam Pardo** | Mentoring, peer reviewer and contributor |

<br>

<a href="https://github.com/fede-giorgi/AI-Agent-Driven-Hedge-Fund/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=fede-giorgi/AI-Agent-Driven-Hedge-Fund"/>
</a>

Made with [contributors-img](https://contrib.rocks)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
