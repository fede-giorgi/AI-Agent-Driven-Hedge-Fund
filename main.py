import json
import os
import re
import importlib.metadata
import sys
import time
import asyncio
import random
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich import box
from datetime import datetime, timedelta
from dotenv import load_dotenv

from classes.tickers import TICKERS
from classes.financial_summary import FinancialSummary

console = Console()


def _text(content) -> str:
    """Return plain text from an LLM response content (handles str or list of blocks)."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content if isinstance(content, str) else str(content)


# ──────────────────────────── display helpers ────────────────────────────────

def _signal_color(signal: str) -> str:
    s = signal.upper()
    if s == "BULLISH":
        return "green"
    if s == "BEARISH":
        return "red"
    return "yellow"


def print_signals_table(signals: dict) -> None:
    """Prints Warren Buffett signals as a dashed ASCII table."""
    console.print("\n[bold]ANALYST SIGNALS:[/bold]")
    table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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
    console.print(f"\n[bold]{title}[/bold]")
    if not trades:
        console.print("[dim]  (no trades)[/dim]")
        return

    table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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
    console.print(f"\n[bold]{title}[/bold]")
    table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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
    console.print(f"\n[bold]FINANCIAL SUMMARY — {ticker}:[/bold]")

    def _fmt(v, pct=False, price=False):
        if v is None:
            return "[dim]n/a[/dim]"
        if pct:
            return f"[yellow]{v:.1%}[/yellow]"
        if price:
            return f"[yellow]${v:,.2f}[/yellow]"
        return f"[yellow]{v:,.2f}[/yellow]"

    table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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
    console.print("\n[bold]MONITOR CHECK:[/bold]")
    table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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

def generate_portfolio_allocation(capital: float, tickers: list, trading_date: str = None):
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
        return generate_portfolio_allocation(capital, tickers)


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
    console.print("\n[bold]RISK PROFILES:[/bold]")
    table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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
    console.print("  [cyan]auto[/cyan]   — pick 5 diversified tickers from the ~600-stock universe")
    console.print("  [cyan]custom[/cyan] — enter up to 5 tickers manually")

    while True:
        choice = console.input("Select mode (auto/custom): ").lower().strip()
        if choice == 'auto':
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
            console.print("Invalid input. Please enter 'auto' or 'custom'.")


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
    console.print("\n[bold]LLM SELECTION:[/bold]")
    table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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
        print(f"History summarisation failed: {e}")
        return json.dumps(history)


def run_backtesting(portfolio: dict, price_map: dict, backtesting_date: str, capital: float):
    """
    Compares actual portfolio performance against a linear regression baseline.

    For each held ticker:
    1. price_start is taken from price_map (price at backtesting_date).
    2. price_today is fetched fresh from the API.
    3. 90 days of monthly prices before backtesting_date are fetched.
    4. A linear regression is fitted to project the expected price today.
    5. alpha = actual_return - regression_baseline_return.

    Args:
        portfolio: dict of ticker → share count.
        price_map: dict of ticker → price at backtesting_date.
        backtesting_date: The historical start date string (YYYY-MM-DD).
        capital: Cash not invested (used in total portfolio value calculation).
    """
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except ImportError:
        console.print("[red]scikit-learn and numpy are required for backtesting. "
                      "Install with: pip install scikit-learn numpy[/red]")
        return

    from tools.get_stock_prices import get_stock_prices

    console.rule("[bold blue]Backtesting Evaluation[/bold blue]")
    console.print(f"Comparing portfolio performance: {backtesting_date} → TODAY", style="cyan")

    console.print("\n[bold]BACKTEST vs LINEAR REGRESSION BASELINE:[/bold]")
    backtest_table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
                           show_header=True, header_style="bold")
    backtest_table.add_column("Ticker", style="cyan")
    backtest_table.add_column("Shares", justify="right")
    backtest_table.add_column(f"Price ({backtesting_date})", justify="right")
    backtest_table.add_column("Price (Today)", justify="right")
    backtest_table.add_column("Actual Return", justify="right")
    backtest_table.add_column("Regression Baseline", justify="right")
    backtest_table.add_column("Alpha", justify="right")

    total_start_value = 0
    total_current_value = 0
    total_baseline_value = 0
    alphas = []

    bt_dt = datetime.strptime(backtesting_date, '%Y-%m-%d')

    for ticker, shares in portfolio.items():
        price_start = price_map.get(ticker, 0)
        if price_start == 0:
            continue

        # Fetch today's price
        price_today = 0
        try:
            today_data = get_stock_prices.func(ticker=ticker)
            if "error" not in today_data:
                prices_list = today_data.get('prices', [])
                if prices_list:
                    price_today = prices_list[-1].get('close', 0)
        except Exception as e:
            console.print(f"[red]Error fetching today price for {ticker}: {e}[/red]")

        # Fetch 90-day monthly history before backtesting_date for regression
        projected_price = price_start  # fallback: no movement
        try:
            hist_start = (bt_dt - timedelta(days=90)).strftime('%Y-%m-%d')
            hist_data = get_stock_prices.func(
                ticker=ticker,
                start_date=hist_start,
                end_date=backtesting_date,
                interval="month"
            )
            hist_prices = hist_data.get('prices', []) if "error" not in hist_data else []

            if len(hist_prices) >= 2:
                closes = [p['close'] for p in hist_prices if p.get('close')]
                X = np.array(range(len(closes))).reshape(-1, 1)
                y = np.array(closes)
                reg = LinearRegression().fit(X, y)

                # Project forward: calculate days from hist_start to today
                today = datetime.today()
                days_total = (today - bt_dt).days
                # In monthly intervals, each point ≈ 30 days
                hist_len = len(closes)
                forward_steps = days_total / 30
                projected_price = reg.predict([[hist_len + forward_steps]])[0]
        except Exception as e:
            console.print(f"[yellow]Regression failed for {ticker}: {e}[/yellow]")

        # Returns
        actual_return = (price_today - price_start) / price_start if price_start > 0 else 0
        baseline_return = (projected_price - price_start) / price_start if price_start > 0 else 0
        alpha = actual_return - baseline_return
        alphas.append(alpha)

        val_start = shares * price_start
        val_today = shares * price_today
        val_baseline = shares * projected_price
        total_start_value += val_start
        total_current_value += val_today
        total_baseline_value += val_baseline

        actual_color = "green" if actual_return >= 0 else "red"
        alpha_color = "green" if alpha >= 0 else "red"

        backtest_table.add_row(
            ticker,
            str(shares),
            f"${price_start:,.2f}",
            f"${price_today:,.2f}",
            f"[{actual_color}]{actual_return:+.2%}[/{actual_color}]",
            f"{baseline_return:+.2%}",
            f"[{alpha_color}]{alpha:+.2%}[/{alpha_color}]",
        )

    console.print(backtest_table)

    initial_total = total_start_value + capital
    current_total = total_current_value + capital
    baseline_total = total_baseline_value + capital

    total_actual_return = (current_total - initial_total) / initial_total if initial_total > 0 else 0
    total_baseline_return = (baseline_total - initial_total) / initial_total if initial_total > 0 else 0
    total_alpha = total_actual_return - total_baseline_return

    verdict = "Beat" if total_alpha >= 0 else "Underperformed"
    verdict_color = "green" if total_alpha >= 0 else "red"

    console.print("\n[bold]BACKTEST SUMMARY:[/bold]")
    summary_table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
                          show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row(f"Initial Value ({backtesting_date})", f"[yellow]${initial_total:,.2f}[/yellow]")
    summary_table.add_row("Current Value (Today)", f"[yellow]${current_total:,.2f}[/yellow]")
    summary_table.add_row("Regression Baseline Value", f"[yellow]${baseline_total:,.2f}[/yellow]")
    summary_table.add_row("Total Actual Return", f"[bold]{total_actual_return:+.2%}[/bold]")
    summary_table.add_row("Regression Baseline Return", f"{total_baseline_return:+.2%}")
    summary_table.add_row("Total Alpha", f"[{verdict_color}][bold]{total_alpha:+.2%}[/bold][/{verdict_color}]")
    summary_table.add_row("Verdict", f"[{verdict_color}]{verdict} the linear regression baseline[/{verdict_color}]")
    console.print(summary_table)


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

    console.print("--- Welcome to the Financial Agent ---", style="bold green")

    if debug_mode:
        console.print("[bold red]DEBUG MODE ENABLED[/bold red]")
        if not check_dependencies():
            sys.exit(1)
        capital = 100000
        risk_profile = 5
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
    from ai_agents.warren_buffet_agent import warren_buffett_agent, get_research_brief
    from ai_agents.portfolio_and_risk_manager import run_portfolio_manager_agent
    from ai_agents.what_if_agent import run_what_if_agent
    from ai_agents.final_orchestrator_agent import run_final_orchestrator_agent, generate_ascii_chart
    from ai_agents.monitor import run_monitor_agent
    from tools.get_stock_prices import get_stock_prices
    from llm import get_llm

    console.print("\n--- Starting Financial Analysis ---", style="bold green")

    if debug_mode:
        tickers_to_research = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
    else:
        tickers_to_research = get_tickers_to_research()

    if not debug_mode:
        portfolio = get_portfolio(capital, tickers_to_research)

    console.print(f"Researching {len(tickers_to_research)} tickers...")

    # 1. Research Brief
    console.print("Generating Research Brief based on Warren Buffett strategy...", style="bold yellow")
    research_brief = get_research_brief()

    # 2. Research Agent (async)
    research_output = asyncio.run(run_research_agent(tickers_to_research, research_brief, backtesting_date))
    financial_data = {res.financial_summary.ticker: res.financial_summary for res in research_output.results}
    console.print("Research complete.")

    # Display FinancialSummary for each ticker
    console.rule("[bold cyan]Research Results[/bold cyan]")
    for ticker, summary in financial_data.items():
        print_financial_summary(ticker, summary)

    # 3. Warren Buffett Agent
    console.rule("[bold yellow]Warren Buffett Analysis[/bold yellow]")
    warren_buffett_signals = {}
    for ticker, summary in financial_data.items():
        signal_data = warren_buffett_agent(summary)
        if signal_data and ticker in signal_data:
            warren_buffett_signals.update(signal_data)
        else:
            console.print(f"  - {ticker}: Could not get analysis.")
    console.print("Warren Buffett analysis complete.")

    price_map = {
        ticker: data.price if data.price else 0.0
        for ticker, data in financial_data.items()
    }

    # Print signals as styled table
    print_signals_table(warren_buffett_signals)

    # Configuration summary table
    console.print("\n[bold]SESSION CONFIGURATION:[/bold]")
    cfg_table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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

    total_iterations = 3 if debug_mode else 10
    for i in range(1, total_iterations + 1):
        console.rule(f"[bold yellow]Iteration {i}/{total_iterations}[/bold yellow]")

        print_signals_table(warren_buffett_signals)
        print_portfolio_state(initial_portfolio, initial_capital, price_map,
                              title=f"PORTFOLIO STATE (iteration {i}):")

        # Portfolio Manager
        console.print("\n[bold]PORTFOLIO MANAGER:[/bold]")
        pm_output = run_portfolio_manager_agent(
            initial_portfolio, initial_capital, risk_profile, warren_buffett_signals, price_map, i, total_iterations, history
        )
        proposed_trades = pm_output.get("proposed_trades", [])
        print_trades_table(proposed_trades, price_map, title="PROPOSED TRADES:")

        # Monitor
        console.print("\n[bold]MONITOR AGENT:[/bold]")
        monitor_output = run_monitor_agent(proposed_trades, initial_portfolio, initial_capital, price_map, i, total_iterations, history)
        print_monitor_result(monitor_output)

        # What-If (skip on last iteration)
        what_if_output = {}
        if i < total_iterations:
            console.print("\n[bold]WHAT-IF AGENT:[/bold]")
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

    # --- History Compression ---
    console.rule("[bold yellow]Compressing History[/bold yellow]")
    llm_instance = get_llm(llm_provider, llm_model)
    history_summary = summarize_iteration_history(history, llm_instance)
    console.print("\n[bold]ITERATION SUMMARY:[/bold]")
    console.print(Panel(history_summary, border_style="yellow", padding=(1, 2)))

    # --- Final Orchestrator ---
    console.rule("[bold green]Final Decision[/bold green]")
    print_signals_table(warren_buffett_signals)
    console.print(generate_ascii_chart(history))

    console.print("\n[bold]FINAL ORCHESTRATOR:[/bold]")
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
        console.print("\n[bold]EXPECTED PORTFOLIO:[/bold]")
        exp_table = Table(box=box.ASCII_DOUBLE_HEAD, show_edge=True, pad_edge=True,
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
    if backtesting_date and portfolio:
        run_backtesting(portfolio, price_map, backtesting_date, capital)

    # End Timer
    end_time = time.time()
    console.print(f"\n[bold]Total Execution Time:[/bold] {end_time - start_time:.2f} seconds")

    console.save_text("financial_agent_session.txt")
    console.print("\n[dim]Session log saved to 'financial_agent_session.txt'[/dim]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"An error occurred: {e}", style="bold red")
