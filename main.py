"""Entry point for the AI hedge fund simulation — orchestrates research, analysis, and portfolio management."""

import asyncio
import importlib.metadata
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from classes.financial_summary import FinancialSummary
from classes.tickers import TICKERS
from config import DEFAULT_TICKERS, RISK_FREE_ANNUAL, TOTAL_ITERATIONS, TOTAL_ITERATIONS_DEBUG

console = Console()


def _text(content) -> str:
    """Return plain text from an LLM response content (handles str or list of blocks)."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content if isinstance(content, str) else str(content)


# ──────────────────────────── display helpers ────────────────────────────────

def _section(title: str, style: str = "cyan") -> None:
    """Print a visually distinct section header."""
    console.print(Panel(Text(title, style=f"bold {style}"),
                        box=box.ROUNDED, border_style=style, expand=False, padding=(0, 2)))


def _signal_color(signal: str) -> str:
    s = signal.upper()
    if s == "BULLISH":
        return "green"
    if s == "BEARISH":
        return "red"
    return "yellow"


def print_signals_table(signals: dict) -> None:
    """Prints Warren Buffett signals as a dashed ASCII table."""
    _section("Analyst Signals")
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                  show_header=True, header_style="bold")
    table.add_column("Ticker", style="cyan")
    table.add_column("Signal", justify="center")
    table.add_column("Confidence", justify="right")

    for ticker, data in signals.items():
        s = data.get("signal", "neutral").upper()
        c = data.get("confidence", 0)
        color = _signal_color(s)
        table.add_row(
            ticker,
            f"[{color}]{s}[/{color}]",
            f"[yellow]{c}%[/yellow]",
        )
    console.print(table)


def print_trades_table(trades: list, price_map: dict, title: str = "PROPOSED TRADES:") -> None:
    """Prints a list of trade dicts as a dashed ASCII table."""
    _section(title)
    if not trades:
        console.print("[dim]  (no trades)[/dim]")
        return

    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                  show_header=True, header_style="bold")
    table.add_column("Ticker", style="cyan")
    table.add_column("Action", justify="center")
    table.add_column("Shares", justify="right")
    table.add_column("Est. Value", justify="right")

    for trade in trades:
        ticker = trade.get("ticker", "?")
        action = trade.get("action", "?").upper()
        shares = trade.get("shares", 0)
        price = price_map.get(ticker, 0)
        est_value = shares * price
        color = "green" if action == "BUY" else "red"
        table.add_row(
            ticker,
            f"[{color}]{action}[/{color}]",
            f"[yellow]{shares:,}[/yellow]",
            f"[yellow]${est_value:,.2f}[/yellow]",
        )
    console.print(table)


def print_portfolio_state(portfolio: dict, capital: float, price_map: dict,
                          title: str = "PORTFOLIO STATE:") -> None:
    """Prints current portfolio holdings and cash as a dashed ASCII table."""
    _section(title)
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                  show_header=True, header_style="bold")
    table.add_column("Item", style="cyan")
    table.add_column("Value", justify="right")

    for ticker, shares in portfolio.items():
        price = price_map.get(ticker, 0)
        mv = shares * price
        table.add_row(ticker, f"[yellow]{shares:,} shares  (${mv:,.2f})[/yellow]")

    table.add_row("Cash", f"[green]${capital:,.2f}[/green]")
    console.print(table)


def print_financial_summary(ticker: str, summary) -> None:
    """Prints a compact two-column table of key FinancialSummary fields for a ticker."""
    _section(f"Financial Summary — {ticker}")

    def _fmt(v, pct=False, price=False):
        if v is None:
            return "[dim]n/a[/dim]"
        if pct:
            return f"[yellow]{v:.1%}[/yellow]"
        if price:
            return f"[yellow]${v:,.2f}[/yellow]"
        return f"[yellow]{v:,.2f}[/yellow]"

    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                  show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    rows = [
        ("Price",           _fmt(summary.price, price=True),
         "Market Cap",      _fmt(summary.market_cap, price=True)),
        ("P/E Ratio",       _fmt(summary.price_to_earnings_ratio),
         "P/B Ratio",       _fmt(summary.price_to_book_ratio)),
        ("ROE",             _fmt(summary.return_on_equity, pct=True),
         "ROIC",            _fmt(summary.return_on_invested_capital, pct=True)),
        ("Gross Margin",    _fmt(summary.gross_margin, pct=True),
         "Op. Margin",      _fmt(summary.operating_margin, pct=True)),
        ("Net Margin",      _fmt(summary.net_margin, pct=True),
         "Debt/Equity",     _fmt(summary.debt_to_equity)),
        ("Current Ratio",   _fmt(summary.current_ratio),
         "Interest Cov.",   _fmt(summary.interest_coverage)),
        ("Revenue Growth",  _fmt(summary.revenue_growth, pct=True),
         "EPS Growth",      _fmt(summary.earnings_growth, pct=True)),
        ("EPS",             _fmt(summary.earnings_per_share, price=True),
         "FCF/Share",       _fmt(summary.free_cash_flow_per_share, price=True)),
        ("Analyst Rev Est", _fmt(summary.analyst_revenue_estimate, price=True),
         "Analyst EPS Est", _fmt(summary.analyst_eps_estimate)),
        ("Net Insider Buy", _fmt(summary.net_insider_buying, price=True),
         "Buys / Sells",    f"[yellow]{summary.insider_buy_count or 0}B / {summary.insider_sell_count or 0}S[/yellow]"),
    ]

    for r in rows:
        table.add_row(*r)

    console.print(table)

    # Segmented revenues
    if summary.segmented_revenue:
        seg = summary.segmented_revenue
        top = sorted(seg.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:5]
        seg_text = "  ".join(f"[cyan]{k}[/cyan]: [yellow]${v/1e9:.1f}B[/yellow]" for k, v in top)
        console.print(f"  [dim]Segments:[/dim] {seg_text}")

    # News headlines
    if summary.recent_news:
        console.print(f"  [dim]News:[/dim]")
        for line in summary.recent_news.strip().split("\n")[:3]:
            if line.strip():
                console.print(f"    [dim]{line.strip()}[/dim]")


def print_monitor_result(monitor_output: dict) -> None:
    """Prints the monitor validation result as a compact dashed ASCII table."""
    _section("Monitor Check", "green")
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                  show_header=True, header_style="bold")
    table.add_column("Field", style="cyan")
    table.add_column("Value", justify="left")

    is_valid = monitor_output.get("is_valid", False)
    color = "green" if is_valid else "red"
    label = "VALID" if is_valid else "INVALID"
    table.add_row("Status", f"[{color}]{label}[/{color}]")

    reasons = monitor_output.get("reasons", monitor_output.get("reasoning", ""))
    if isinstance(reasons, list):
        reasons = "; ".join(reasons)
    if reasons:
        table.add_row("Reason", str(reasons))

    approved = monitor_output.get("approved_trades", [])
    if approved:
        table.add_row("Approved trades", f"[yellow]{len(approved)}[/yellow]")

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────

def generate_portfolio_allocation(capital: float, tickers: list, trading_date: str | None = None):
    """
    Generates an initial equal-weight portfolio allocation for the given tickers.

    Fetches the most-recent closing price for each ticker and buys as many
    whole shares as possible with an equal slice of capital.

    Args:
        capital: Total deployment capital in dollars.
        tickers: List of ticker symbols to allocate across.
        trading_date: Optional date string (YYYY-MM-DD) to use as the price end_date.

    Returns:
        dict mapping ticker → integer share count.
    """
    from tools.get_stock_prices import get_stock_prices
    console.print("Calculating initial allocation based on capital...", style="yellow")
    portfolio = {}

    per_stock_capital = capital / len(tickers) if tickers else capital

    for ticker in tickers:
        try:
            kwargs = {"ticker": ticker}
            if trading_date:
                kwargs["end_date"] = trading_date
                dt = datetime.strptime(trading_date, '%Y-%m-%d')
                kwargs["start_date"] = (dt - timedelta(days=7)).strftime('%Y-%m-%d')

            price_data = get_stock_prices.func(**kwargs) if hasattr(get_stock_prices, 'func') else get_stock_prices(**kwargs)

            if price_data and 'prices' in price_data and price_data['prices']:
                price = price_data['prices'][-1].get('close', 0)
                if price > 0:
                    shares = int(per_stock_capital // price)
                    if shares > 0:
                        portfolio[ticker] = shares
        except Exception as e:
            console.print(f"Error fetching price for {ticker}: {e}", style="red")

    if not portfolio:
        console.print("Insufficient capital to buy shares or API error. Starting with empty portfolio.", style="red")

    return portfolio


def get_portfolio(capital: float, tickers: list):
    """
    Prompts the user to enter an existing portfolio or generates an initial allocation.

    The user may enter holdings for any ticker from the provided list, or type
    'done' to finish.  If they have no existing holdings, an equal-weight
    allocation is generated automatically.

    Args:
        capital: Total deployment capital in dollars.
        tickers: List of valid ticker symbols for this session.

    Returns:
        dict mapping ticker → integer share count.
    """
    has_portfolio = console.input("Do you have an existing portfolio? (yes/no): ").lower()

    if has_portfolio == 'yes':
        portfolio = {}
        console.print("Please enter your portfolio holdings.", style="cyan")
        console.print(f"Supported stocks: [bold]{', '.join(tickers)}[/bold]")

        while True:
            ticker = console.input("\nEnter stock ticker or 'done' to finish: ").upper()
            if ticker == 'DONE':
                break

            if ticker not in tickers:
                console.print(f"[red]'{ticker}' is not in the researched ticker list.[/red]")
                continue

            while True:
                try:
                    qty_str = console.input(f"Enter quantity for {ticker}: ")
                    quantity = int(qty_str)
                    if quantity <= 0:
                        console.print("[red]Quantity must be a positive integer.[/red]")
                        continue
                    portfolio[ticker] = quantity
                    break
                except ValueError:
                    console.print("[red]Invalid input. Please enter a valid integer number.[/red]")

        return portfolio
    else:
        console.print("[dim]Starting with empty portfolio (no existing holdings).[/dim]")
        return {}


def get_capital():
    """
    Prompts the user to enter their available capital for deployment.

    Returns:
        float in the range [1, 1_000_000].
    """
    while True:
        try:
            raw_input = console.input("How much capital do you have for deployment? (1-1,000,000): ")
            clean_input = raw_input.replace("_", "").replace(",", "")
            capital = float(clean_input)
            if 1 <= capital <= 1_000_000:
                return capital
            else:
                console.print("Please enter an amount between 1 and 1,000,000.")
        except ValueError:
            console.print("Invalid input. Please enter a number.")


def get_risk_profile():
    """
    Displays a risk-profile table and prompts the user to select a level.

    Returns:
        int in the range [1, 10].
    """
    _section("Risk Profiles")
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                  show_header=True, header_style="bold")
    table.add_column("Level", style="cyan", justify="center")
    table.add_column("Risk Profile", style="cyan")
    table.add_column("Typical Allocation")

    risk_profiles = [
        ("1", "Ultra conservative", "0–20% stocks, 80–100% high‑grade bonds/cash"),
        ("2", "Very conservative", "20–30% stocks, rest investment‑grade bonds"),
        ("3", "Conservative", "30–40% stocks, broad bond funds dominant"),
        ("4", "Mod. conservative", "40–50% stocks, 50–60% bonds"),
        ("5", "Balanced", "~60% stocks, 40% bonds (classic 60/40)"),
        ("6", "Mod. aggressive", "70–80% stocks, 20–30% bonds"),
        ("7", "Aggressive", "80–90% stocks, small bond/cash buffer"),
        ("8", "Very aggressive", "90–100% stocks, globally diversified"),
        ("9", "Speculative", "100% stocks, tilts to sectors/themes"),
        ("10", "Highly speculative", "100% stocks, heavy single‑stock / options"),
    ]

    for level, profile, style in risk_profiles:
        table.add_row(level, profile, style)

    console.print(table)

    while True:
        try:
            level = int(console.input("Please select your risk profile level (1-10): "))
            if 1 <= level <= 10:
                return level
            else:
                console.print("Please enter a level between 1 and 10.")
        except ValueError:
            console.print("Invalid input. Please enter a number.")


def get_tickers_to_research():
    """
    Prompts the user to choose between custom tickers or auto-selection from buckets.

    Auto mode splits the TICKERS universe into 5 equal buckets and picks one
    ticker at random from each bucket, giving a diversified starting set.
    Custom mode lets the user enter up to 5 space-separated tickers directly.

    Returns:
        List of up to 5 ticker strings.
    """
    console.print("\n[bold]Ticker Selection[/bold]")
    console.print(f"  [cyan]default[/cyan] — use {', '.join(DEFAULT_TICKERS)} (same as --debug)")
    console.print("  [cyan]auto[/cyan]    — pick 5 diversified tickers from the ~600-stock universe")
    console.print("  [cyan]custom[/cyan]  — enter up to 5 tickers manually")

    while True:
        choice = console.input("Select mode (default/auto/custom): ").lower().strip()
        if choice == 'default':
            console.print(f"Using tickers: [bold]{DEFAULT_TICKERS}[/bold]")
            return DEFAULT_TICKERS
        elif choice == 'auto':
            n_buckets = 5
            bucket_size = max(1, len(TICKERS) // n_buckets)
            selected = []
            for i in range(n_buckets):
                bucket = TICKERS[i * bucket_size: (i + 1) * bucket_size]
                if bucket:
                    selected.append(random.choice(bucket))
            console.print(f"Auto-selected tickers: [bold]{selected}[/bold]")
            return selected
        elif choice == 'custom':
            raw = console.input("Enter up to 5 tickers separated by spaces: ").upper()
            tickers = [t.strip() for t in raw.split() if t.strip()][:5]
            if not tickers:
                console.print("[red]Please enter at least one ticker.[/red]")
                continue
            console.print(f"Using tickers: [bold]{tickers}[/bold]")
            return tickers
        else:
            console.print("Invalid input. Please enter 'default', 'auto', or 'custom'.")


def get_backtesting_date():
    """
    Prompts the user to opt in to backtesting and enter a historical date.

    Returns:
        Date string (YYYY-MM-DD) if the user opts in, or None.
    """
    choice = console.input("Do you want to enable backtesting? (yes/no): ").lower()
    if choice == 'yes':
        while True:
            date_str = console.input("Enter the backtesting date (YYYY-MM-DD): ")
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except ValueError:
                console.print("Invalid date format. Please use YYYY-MM-DD.")
    return None


def get_llm_choice():
    """
    Interactively prompts the user to select an LLM provider and model.

    Providers: Google (Gemini) or Anthropic (Claude).
    The chosen values are passed to get_llm() for all agents.

    Returns:
        Tuple of (provider: str, model: str).
    """
    _section("LLM Selection")
    table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                  show_header=True, header_style="bold")
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Provider", style="cyan")
    table.add_column("Model")
    table.add_column("Notes")

    table.add_row("1", "Google", "gemini-3.1-pro-preview", "[dim]Default — latest Gemini[/dim]")
    table.add_row("2", "Google", "gemini-2.5-flash", "[dim]Fast, cost-efficient[/dim]")
    table.add_row("3", "Anthropic", "claude-opus-4-6", "[dim]Most powerful Claude model[/dim]")
    table.add_row("4", "Anthropic", "claude-sonnet-4-6", "[dim]Balanced Claude model[/dim]")
    table.add_row("5", "Anthropic", "claude-haiku-4-5-20251001", "[dim]Fast Claude model[/dim]")

    console.print(table)

    options = {
        "1": ("google", "gemini-3.1-pro-preview"),
        "2": ("google", "gemini-2.5-flash"),
        "3": ("anthropic", "claude-opus-4-6"),
        "4": ("anthropic", "claude-sonnet-4-6"),
        "5": ("anthropic", "claude-haiku-4-5-20251001"),
    }

    while True:
        choice = console.input("Select LLM option (1-5, or press Enter for default): ").strip()
        if choice == "":
            return "google", "gemini-2.5-flash"
        if choice in options:
            return options[choice]
        console.print("Invalid choice. Enter 1–5 or press Enter.")


def check_dependencies():
    """
    Verifies that all packages listed in requirements.txt are installed.

    Returns:
        True if all dependencies are present, False otherwise.
    """
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        console.print(f"[yellow]Warning: {req_path} not found.[/yellow]")
        return True

    with open(req_path, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    missing = []
    for req in requirements:
        pkg_name = re.split(r"[<>=~!]", req)[0].strip()
        try:
            importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(req)

    if missing:
        console.print("[bold red]Missing dependencies from requirements.txt:[/bold red]")
        for lib in missing:
            console.print(f"  - {lib}")
        console.print("[bold red]Please install them using: pip install -r requirements.txt[/bold red]")
        return False
    else:
        console.print("[green]All dependencies from requirements.txt are installed.[/green]")
        return True


def summarize_iteration_history(history: list, llm) -> str:
    """
    Compresses 10 iterations of portfolio-manager debate into a concise summary.

    Reduces token usage for the Final Orchestrator by sending a compact
    narrative instead of the full raw JSON history (~70% token reduction).

    Args:
        history: List of iteration dicts, each containing pm_proposal,
                 monitor_check, and what_if_critique keys.
        llm: LLM instance used for summarisation.

    Returns:
        A compact plain-text summary string.
    """
    from langchain_core.messages import HumanMessage

    prompt = f"""You are a financial analyst assistant.  Below is the full debate history of a multi-agent
portfolio optimisation loop across {len(history)} iterations.

Compress this into a concise summary (max 400 words) that captures:
1. The dominant trade proposals across iterations.
2. Key monitor flags or constraints that were repeatedly triggered.
3. The main counter-arguments raised by the what-if agent.
4. Any clear consensus or persistent disagreement.

History (JSON):
{json.dumps(history, indent=2)}

Provide only the summary text — no headers, no JSON.
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return _text(response.content)
    except Exception as e:
        console.print(f"[red]History summarisation failed: {e}[/red]")
        return json.dumps(history)


def run_backtesting(
    portfolio: dict,
    price_map: dict,
    backtesting_date: str,
    remaining_capital: float,
    initial_capital: float,
    all_tickers: list,
) -> None:
    """
    Compares agent portfolio performance against three standard financial benchmarks.

    Benchmarks (no statistical model required):
    1. Risk-free rate  — 4.5% annual T-bill proxy scaled to the holding period.
    2. 1/N equal-weight — split initial_capital equally across all researched tickers
       and hold. Academically established hard-to-beat baseline (DeMiguel et al. 2009).
    3. S&P 500 (SPY)   — market index return over the same period (2 API calls).

    Args:
        portfolio: Ticker → share count after final trades.
        price_map: Ticker → price at backtesting_date (from Research Agent).
        backtesting_date: Historical start date string (YYYY-MM-DD).
        remaining_capital: Cash left after the agent's trades.
        initial_capital: Total capital before any trades (used for 1/N benchmark).
        all_tickers: All tickers researched this session (for 1/N benchmark).
    """
    from tools.get_stock_prices import get_stock_prices

    bt_dt = datetime.strptime(backtesting_date, '%Y-%m-%d')
    today_dt = datetime.today()
    days_held = max((today_dt - bt_dt).days, 1)

    risk_free_return = RISK_FREE_ANNUAL * (days_held / 365)

    console.rule("[bold blue]Backtesting Evaluation[/bold blue]")
    console.print(
        f"Holding period: [cyan]{backtesting_date}[/cyan] → "
        f"[cyan]{today_dt.strftime('%Y-%m-%d')}[/cyan] "
        f"([yellow]{days_held} days[/yellow])"
    )

    # ── Fetch today's prices for all tickers (portfolio + 1/N universe) ────
    console.print("Fetching current prices...", style="dim")
    prices_today: dict = {}
    for ticker in set(portfolio.keys()) | set(all_tickers):
        try:
            data = get_stock_prices.func(ticker=ticker)
            if "error" not in data:
                plist = data.get('prices', [])
                if plist:
                    prices_today[ticker] = plist[-1].get('close', 0)
        except Exception:
            pass

    # ── SPY market benchmark (2 API calls) ─────────────────────────────────
    spy_return = None
    try:
        bt_start_str = (bt_dt - timedelta(days=7)).strftime('%Y-%m-%d')
        spy_hist = get_stock_prices.func(ticker="SPY", start_date=bt_start_str,
                                         end_date=backtesting_date)
        spy_now = get_stock_prices.func(ticker="SPY")
        if "error" not in spy_hist and "error" not in spy_now:
            ph = spy_hist.get('prices', [])
            pn = spy_now.get('prices', [])
            if ph and pn:
                p0 = ph[-1].get('close', 0)
                p1 = pn[-1].get('close', 0)
                if p0 > 0:
                    spy_return = (p1 - p0) / p0
    except Exception:
        pass

    # ── Per-ticker performance table ────────────────────────────────────────
    _section("Per-Ticker Performance")
    ticker_table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                         show_header=True, header_style="bold")
    ticker_table.add_column("Ticker", style="cyan")
    ticker_table.add_column("Shares", justify="right")
    ticker_table.add_column(f"Price ({backtesting_date})", justify="right")
    ticker_table.add_column("Price (Today)", justify="right")
    ticker_table.add_column("Return", justify="right")

    invested_start = 0.0
    invested_end = 0.0
    for ticker, shares in portfolio.items():
        p_start = price_map.get(ticker, 0)
        p_today = prices_today.get(ticker, 0)
        if p_start == 0 or shares == 0:
            continue
        ret = (p_today - p_start) / p_start if p_start > 0 else 0.0
        c = "green" if ret >= 0 else "red"
        ticker_table.add_row(
            ticker,
            f"{shares:,}",
            f"${p_start:,.2f}",
            f"${p_today:,.2f}" if p_today else "[dim]N/A[/dim]",
            f"[{c}]{ret:+.2%}[/{c}]",
        )
        invested_start += shares * p_start
        invested_end += shares * p_today
    console.print(ticker_table)

    # ── Agent portfolio total return ────────────────────────────────────────
    # Invested value + uninvested cash (cash earns 0% — opportunity cost is real)
    agent_start = invested_start + remaining_capital
    agent_end = invested_end + remaining_capital
    agent_return = (agent_end - agent_start) / agent_start if agent_start > 0 else 0.0

    # ── 1/N Equal-weight benchmark ──────────────────────────────────────────
    # Split initial_capital equally across all researched tickers and hold.
    ew_tickers = [t for t in all_tickers if price_map.get(t, 0) > 0]
    ew_return = None
    if ew_tickers:
        alloc = initial_capital / len(ew_tickers)
        ew_start = 0.0
        ew_end = 0.0
        for t in ew_tickers:
            ps = price_map.get(t, 0)
            pt = prices_today.get(t, 0)
            if ps > 0 and pt > 0:
                sh = int(alloc // ps)
                ew_start += sh * ps
                ew_end += sh * pt
        uninvested_ew = initial_capital - ew_start  # fractional cash leftover
        ew_total_start = ew_start + uninvested_ew
        ew_total_end = ew_end + uninvested_ew
        ew_return = (ew_total_end - ew_total_start) / ew_total_start if ew_total_start > 0 else 0.0

    # ── Benchmark comparison table ──────────────────────────────────────────
    def _r(val: float) -> str:
        c = "green" if val >= 0 else "red"
        return f"[{c}]{val:+.2%}[/{c}]"

    def _alpha(agent: float, bench: float) -> str:
        d = agent - bench
        c = "green" if d >= 0 else "red"
        return f"[{c}]{d:+.2%}[/{c}]"

    _section("Benchmark Comparison")
    bench_table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                        show_header=True, header_style="bold")
    bench_table.add_column("Strategy / Benchmark", style="cyan", min_width=22)
    bench_table.add_column("Return", justify="right")
    bench_table.add_column("Alpha vs. Agent", justify="right")
    bench_table.add_column("Description")

    ac = "green" if agent_return >= 0 else "red"
    bench_table.add_row(
        "[bold]Agent Portfolio[/bold]",
        f"[bold {ac}]{agent_return:+.2%}[/bold {ac}]",
        "—",
        "Warren Buffett multi-agent system",
    )
    if ew_return is not None:
        bench_table.add_row(
            "1/N Equal-Weight",
            _r(ew_return),
            _alpha(agent_return, ew_return),
            f"Naïve equal allocation — {len(ew_tickers)} tickers (DeMiguel 2009)",
        )
    if spy_return is not None:
        bench_table.add_row(
            "S&P 500 (SPY)",
            _r(spy_return),
            _alpha(agent_return, spy_return),
            "Market index benchmark",
        )
    else:
        bench_table.add_row("S&P 500 (SPY)", "[dim]N/A[/dim]", "[dim]N/A[/dim]",
                            "SPY unavailable from API")
    bench_table.add_row(
        "Risk-Free (Cash)",
        _r(risk_free_return),
        _alpha(agent_return, risk_free_return),
        f"4.5% annual T-bill proxy — {days_held}-day equivalent",
    )
    console.print(bench_table)

    # ── Verdict ─────────────────────────────────────────────────────────────
    candidates = [
        (ew_return, "1/N Equal-Weight"),
        (spy_return, "S&P 500 (SPY)"),
        (risk_free_return, "Risk-Free Cash"),
    ]
    available = [(r, lbl) for r, lbl in candidates if r is not None]
    beaten = [(r, lbl) for r, lbl in available if agent_return > r]
    n_beaten, n_total = len(beaten), len(available)
    vc = "green" if n_beaten == n_total else "yellow" if n_beaten > 0 else "red"
    console.print(f"\n[{vc}][bold]Verdict: Agent beat {n_beaten}/{n_total} benchmarks[/bold][/{vc}]")
    if beaten:
        console.print(f"  [green]✓ Beat: {', '.join(l for _, l in beaten)}[/green]")
    missed = [(r, l) for r, l in available if agent_return <= r]
    if missed:
        console.print(f"  [red]✗ Underperformed: {', '.join(l for _, l in missed)}[/red]")


def main():
    """
    Entry-point for the AI Hedge Fund simulation.

    Flow:
    1. Parse CLI flags (--debug).
    2. Check installed dependencies.
    3. Gather user inputs: capital, portfolio, risk, tickers, LLM, backtesting date.
    4. Run Research Agent → Warren Buffett Agent.
    5. Run 10-iteration portfolio debate loop (PM → Monitor → What-If).
    6. Summarise history and run Final Orchestrator.
    7. Execute trades and optionally run backtesting comparison.
    """
    load_dotenv()
    start_time = time.time()
    debug_mode = "--debug" in sys.argv
    console.record = True

    console.print(Panel(
        "[bold]Welcome to the Financial Agent[/bold]\n[dim]Warren Buffett-style multi-agent hedge fund simulation[/dim]",
        box=box.DOUBLE, border_style="green", padding=(1, 4), expand=False
    ))

    if debug_mode:
        console.print("[bold red]DEBUG MODE ENABLED[/bold red]")
        if not check_dependencies():
            sys.exit(1)
        capital = 100000
        risk_profile = 8  # aggressive in debug so trades are exercised
        backtesting_date = (datetime.today() - timedelta(days=90)).strftime('%Y-%m-%d')
        portfolio = {}
        llm_provider = os.getenv("LLM_PROVIDER", "google")
        llm_model = os.getenv("LLM_MODEL", "gemini-3.1-pro-preview")
    else:
        capital = get_capital()
        llm_provider, llm_model = get_llm_choice()
        risk_profile = get_risk_profile()
        backtesting_date = get_backtesting_date()

    # Import agents and tools after dependency check
    from ai_agents.research_agent import run_research_agent
    from ai_agents.warren_buffet_agent import warren_buffett_agent
    from ai_agents.portfolio_and_risk_manager import run_portfolio_manager_agent
    from ai_agents.what_if_agent import run_what_if_agent
    from ai_agents.final_orchestrator_agent import run_final_orchestrator_agent, generate_ascii_chart
    from ai_agents.monitor import run_monitor_agent
    from tools.get_stock_prices import get_stock_prices
    from llm import get_llm

    console.print(Panel("[bold]Starting Financial Analysis[/bold]",
                        box=box.ROUNDED, border_style="green", expand=False, padding=(0, 2)))

    if debug_mode:
        tickers_to_research = DEFAULT_TICKERS
    else:
        tickers_to_research = get_tickers_to_research()

    if not debug_mode:
        portfolio = get_portfolio(capital, tickers_to_research)

    console.print(f"Researching {len(tickers_to_research)} tickers...")

    # 1. Research Agent (async)
    research_output = asyncio.run(run_research_agent(tickers_to_research, backtesting_date))
    financial_data = {res.financial_summary.ticker: res.financial_summary for res in research_output.results}
    console.print("Research complete.")

    # Display FinancialSummary for each ticker
    console.rule("[bold cyan]Research Results[/bold cyan]")
    for ticker, summary in financial_data.items():
        print_financial_summary(ticker, summary)

    # 3. Warren Buffett Agent — all tickers analysed in parallel
    console.rule("[bold yellow]Warren Buffett Analysis[/bold yellow]")

    async def _run_warren_buffett_all(data: dict, dbg: bool) -> dict:
        tasks = [warren_buffett_agent(s, debug_mode=dbg) for s in data.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        signals = {}
        for ticker, result in zip(data.keys(), results):
            if isinstance(result, Exception):
                console.print(f"  [red]✗ {ticker}: {result}[/red]")
            elif result and ticker in result:
                signals.update(result)
            else:
                console.print(f"  [red]✗ {ticker}: no signal returned[/red]")
        return signals

    warren_buffett_signals = asyncio.run(_run_warren_buffett_all(financial_data, debug_mode))
    console.print("Warren Buffett analysis complete.")

    price_map = {
        ticker: data.price if data.price else 0.0
        for ticker, data in financial_data.items()
    }

    # Print signals as styled table
    print_signals_table(warren_buffett_signals)

    # Configuration summary table
    _section("Session Configuration", "yellow")
    cfg_table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                      show_header=False)
    cfg_table.add_column("Setting", style="cyan")
    cfg_table.add_column("Value", justify="right")
    cfg_table.add_row("LLM", f"[yellow]{llm_provider} / {llm_model}[/yellow]")
    cfg_table.add_row("Capital", f"[yellow]${capital:,.2f}[/yellow]")
    cfg_table.add_row("Risk Profile", f"[yellow]{risk_profile}[/yellow]")
    cfg_table.add_row("Backtesting Date", f"[yellow]{backtesting_date if backtesting_date else 'Today'}[/yellow]")
    cfg_table.add_row("Tickers", f"[yellow]{', '.join(tickers_to_research)}[/yellow]")
    for ticker, price in price_map.items():
        cfg_table.add_row(f"  {ticker} price", f"[yellow]${price:,.2f}[/yellow]")
    console.print(cfg_table)

    # --- Simulation Loop ---
    history = []
    initial_portfolio = portfolio.copy()
    initial_capital = capital

    total_iterations = TOTAL_ITERATIONS_DEBUG if debug_mode else TOTAL_ITERATIONS
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Trading iterations", total=total_iterations)
        for i in range(1, total_iterations + 1):
            progress.update(task, description=f"Iteration {i}/{total_iterations}")
            console.rule(f"[bold yellow]Iteration {i}/{total_iterations}[/bold yellow]")

            print_signals_table(warren_buffett_signals)
            print_portfolio_state(initial_portfolio, initial_capital, price_map,
                                  title=f"PORTFOLIO STATE (iteration {i}):")

            # Portfolio Manager
            _section("Portfolio Manager")
            pm_output = run_portfolio_manager_agent(
                initial_portfolio, initial_capital, risk_profile, warren_buffett_signals,
                price_map, i, total_iterations, history, force_trades=debug_mode,
            )
            proposed_trades = pm_output.get("proposed_trades", [])
            print_trades_table(proposed_trades, price_map, title="PROPOSED TRADES:")

            # Monitor
            _section("Monitor Agent", "green")
            monitor_output = run_monitor_agent(proposed_trades, initial_portfolio, initial_capital, price_map, i, total_iterations, history)
            print_monitor_result(monitor_output)

            # What-If (skip on last iteration)
            what_if_output = {}
            if i < total_iterations:
                _section("What-If Agent", "magenta")
                what_if_output = run_what_if_agent(initial_portfolio, initial_capital, proposed_trades, price_map, i, total_iterations, warren_buffett_signals, history)
                alt = what_if_output.get("alternative_scenario", {}) or {}
                counter_trades = alt.get("proposed_trades", []) if isinstance(alt, dict) else []
                print_trades_table(counter_trades, price_map, title="COUNTER-PROPOSAL:")
                critique = what_if_output.get("critique", "")
                reasoning = what_if_output.get("reasoning", "")
                if critique:
                    console.print(Panel(str(critique), title="[dim]Critique[/dim]",
                                        border_style="dim", padding=(0, 1)))
                if reasoning:
                    console.print(Panel(str(reasoning), title="[dim]Reasoning[/dim]",
                                        border_style="dim", padding=(0, 1)))

            iteration_data = {
                "iteration": i,
                "pm_proposal": pm_output,
                "monitor_check": monitor_output,
                "what_if_critique": what_if_output
            }
            history.append(iteration_data)
            progress.advance(task)

    # --- History Compression ---
    console.rule("[bold yellow]Compressing History[/bold yellow]")
    llm_instance = get_llm(llm_provider, llm_model)
    with console.status("[bold yellow]Compressing iteration history...[/bold yellow]", spinner="dots"):
        history_summary = summarize_iteration_history(history, llm_instance)
    _section("Iteration Summary", "yellow")
    console.print(Panel(history_summary, border_style="yellow", padding=(1, 2)))

    # --- Final Orchestrator ---
    console.rule("[bold green]Final Decision[/bold green]")
    print_signals_table(warren_buffett_signals)
    console.print(generate_ascii_chart(history))

    _section("Final Orchestrator", "green")
    with console.status("[bold green]Final Orchestrator deliberating...[/bold green]", spinner="dots"):
        final_output = run_final_orchestrator_agent(
            initial_portfolio, initial_capital, warren_buffett_signals, price_map, history
        )

    reasoning = final_output.get("final_decision_reasoning", "No reasoning provided.")
    console.print(Panel(Markdown(reasoning), title="Final Decision Reasoning", border_style="green", padding=(1, 2)))

    final_trades = final_output.get("final_trades", [])
    print_trades_table(final_trades, price_map, title="FINAL TRADES:")

    # Expected portfolio after trades
    expected = final_output.get("expected_portfolio", {})
    if expected:
        _section("Expected Portfolio")
        exp_table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                          show_header=True, header_style="bold")
        exp_table.add_column("Ticker", style="cyan")
        exp_table.add_column("Shares", justify="right")
        for t, s in (expected.items() if isinstance(expected, dict) else []):
            exp_table.add_row(t, f"[yellow]{s}[/yellow]")
        if exp_table.row_count:
            console.print(exp_table)

    # Execute Final Trades
    if final_trades:
        for trade in final_trades:
            ticker = trade['ticker']
            shares = trade['shares']
            action = trade['action']
            price = price_map.get(ticker, 0)

            if action == 'buy':
                portfolio[ticker] = portfolio.get(ticker, 0) + shares
                capital -= shares * price
            elif action == 'sell':
                portfolio[ticker] = max(0, portfolio.get(ticker, 0) - shares)
                capital += shares * price
                if portfolio[ticker] == 0:
                    del portfolio[ticker]

        print_portfolio_state(portfolio, capital, price_map, title="POST-TRADE PORTFOLIO:")
    else:
        console.print("[yellow]No trades executed based on final decision.[/yellow]")

    # --- Backtesting ---
    if backtesting_date and (portfolio or initial_capital > 0):
        run_backtesting(portfolio, price_map, backtesting_date, capital, initial_capital, tickers_to_research)

    # --- Token Usage Summary ---
    from llm import get_usage_summary
    usage = get_usage_summary(model=llm_model)
    _section("Token Usage", "dim")
    usage_table = Table(box=box.ROUNDED, show_edge=True, pad_edge=True,
                        show_header=False)
    usage_table.add_column("Metric", style="cyan")
    usage_table.add_column("Value", justify="right")
    usage_table.add_row("Model", f"[yellow]{usage['model'] or 'unknown'}[/yellow]")
    usage_table.add_row("LLM calls", f"[yellow]{usage['calls']:,}[/yellow]")
    usage_table.add_row("Input tokens", f"[yellow]{usage['input_tokens']:,}[/yellow]")
    usage_table.add_row("Output tokens", f"[yellow]{usage['output_tokens']:,}[/yellow]")
    usage_table.add_row("Total tokens", f"[yellow]{usage['total_tokens']:,}[/yellow]")
    if usage["estimated_cost_usd"] is not None:
        usage_table.add_row("Est. cost (USD)", f"[yellow]${usage['estimated_cost_usd']:.4f}[/yellow]")
    else:
        usage_table.add_row("Est. cost (USD)", "[dim]n/a (model not in price table)[/dim]")
    if os.getenv("LANGCHAIN_API_KEY"):
        usage_table.add_row("LangSmith", f"[green]enabled — project: {os.getenv('LANGCHAIN_PROJECT', 'ai-hedge-fund')}[/green]")
    console.print(usage_table)

    # End Timer
    end_time = time.time()
    elapsed = end_time - start_time
    console.save_text("financial_agent_session.txt")
    console.print(Panel(
        f"[bold]Completed in {elapsed:.1f}s[/bold]  |  Session log: [dim]financial_agent_session.txt[/dim]",
        box=box.ROUNDED, border_style="green", expand=False, padding=(0, 2)
    ))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"An error occurred: {e}", style="bold red")
