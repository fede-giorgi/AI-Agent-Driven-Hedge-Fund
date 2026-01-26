import json
import os
import re
import importlib.metadata
import sys
import time
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from datetime import datetime, timedelta
from dotenv import load_dotenv

from models.tickers import TICKERS
from models.financial_summary import FinancialSummary

console = Console()

def generate_portfolio_allocation(capital: float, trading_date: str = None):
    """
    Generates an initial portfolio allocation based on capital and stock prices.
    """
    from tools.get_stock_prices import get_stock_prices
    console.print("Calculating initial allocation based on capital...", style="yellow")
    tickers = ["AAPL", "MSFT", "NVDA"]
    portfolio = {}
    
    # Allocate evenly (1/3 of capital per stock)
    per_stock_capital = capital / len(tickers)
    
    for ticker in tickers:
        try:
            # Use .func to call the tool directly if available, otherwise call as callable
            kwargs = {"ticker": ticker}
            if trading_date:
                kwargs["end_date"] = trading_date
                # Set start_date to a week before to ensure we get data
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

def get_portfolio(capital: float):
    """
    Prompts the user to either provide their existing portfolio or use a default one.
    If no portfolio is provided, it calculates an initial allocation based on the capital.
    """
    has_portfolio = console.input("Do you have an existing portfolio? (yes/no): ").lower()
    
    if has_portfolio == 'yes':
        portfolio = {}
        console.print("Please enter your portfolio holdings.", style="cyan")
        console.print("Supported stocks: [bold]AAPL, MSFT, NVDA[/bold]")
        
        while True:
            ticker = console.input("\nEnter stock ticker (e.g. AAPL) or 'done' to finish: ").upper()
            if ticker == 'DONE':
                break
            
            if ticker not in ["AAPL", "MSFT", "NVDA"]:
                console.print(f"[red]'{ticker}' is not supported. Please enter AAPL, MSFT, or NVDA.[/red]")
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
        return generate_portfolio_allocation(capital)

def get_capital():
    """
    Prompts the user to enter their available capital for deployment.
    """
    while True:
        try:
            raw_input = console.input("How much capital do you have for deployment? (1-1,000,000): ")
            # Allow inputs like 10_000 or 10,000
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
    Displays a table of risk profiles and asks the user to select one.
    """
    table = Table(title="Risk Profiles")
    table.add_column("Lev.", style="cyan")
    table.add_column("Risk Profile", style="magenta")
    table.add_column("Typical Stock/Bond Style (Illustrative)")

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
    Prompts the user to choose between researching all tickers or a small subset.
    """
    while True:
        choice = console.input("Do you want to research all tickers (approx. 1000) or a small subset (AAPL, MSFT, NVDA)? (all/subset): ").lower()
        if choice in ['all', 'subset']:
            if choice == 'all':
                return TICKERS
            else:
                return ["AAPL", "MSFT", "NVDA"]
        else:
            console.print("Invalid input. Please enter 'all' or 'subset'.")

def get_backtesting_date():
    """
    Prompts the user to opt-in for backtesting and select a date.
    Returns the selected date string (YYYY-MM-DD) or None if disabled.
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

def check_dependencies():
    """Checks if libraries listed in requirements.txt are installed."""
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        console.print(f"[yellow]Warning: {req_path} not found.[/yellow]")
        return True

    with open(req_path, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    missing = []
    for req in requirements:
        # Extract package name, handling version specifiers
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

def main():
    """
    Main function to run the financial agent.
    """
    # Load environment variables
    load_dotenv()

    # Start Timer
    start_time = time.time()
    
    # Check for debug mode
    debug_mode = "--debug" in sys.argv

    # Enable recording to save log later
    console.record = True

    console.print("--- Welcome to the Financial Agent ---", style="bold green")
    
    if debug_mode:
        console.print("[bold red]DEBUG MODE ENABLED[/bold red]")
        if not check_dependencies():
            sys.exit(1)
        capital = 10000
        risk_profile = 5
        backtesting_date = None
        portfolio = {}
    else:
        capital = get_capital()
        portfolio = get_portfolio(capital)
        risk_profile = get_risk_profile()
        backtesting_date = get_backtesting_date()

    # Import agents and tools here to allow dependency check to run first
    from ai_agents.research_agent import run_research_agent
    from ai_agents.warren_buffet_agent import warren_buffett_agent, get_research_brief
    from ai_agents.portfolio_and_risk_manager import run_portfolio_manager_agent
    from ai_agents.what_if_agent import run_what_if_agent
    from ai_agents.final_orchestrator_agent import run_final_orchestrator_agent, generate_ascii_chart
    from ai_agents.monitor import run_monitor_agent
    from tools.get_stock_prices import get_stock_prices

    console.print("\n--- Starting Financial Analysis ---", style="bold green")
    
    if debug_mode:
        tickers_to_research = ["AAPL", "MSFT", "NVDA"]
    else:
        tickers_to_research = get_tickers_to_research()

    console.print(f"Researching {len(tickers_to_research)} tickers...")
    
    # 1. Generate the Research Brief (Intent-Driven)
    console.print("Generating Research Brief based on Warren Buffett strategy...", style="bold yellow")
    research_brief = get_research_brief()

    # Run the async research agent synchronously
    research_output = asyncio.run(run_research_agent(tickers_to_research, research_brief, backtesting_date))
    
    # Directly access the FinancialSummary objects from the results
    financial_data = {res.financial_summary.ticker: res.financial_summary for res in research_output.results}
    
    console.print("Research complete.")

    # 2. Warren Buffett Agent (Run once)
    console.print("\n--- Running Warren Buffett Analysis ---", style="bold yellow")
    warren_buffett_signals = {}
    for ticker, summary in financial_data.items():
        signal_data = warren_buffett_agent(summary)
        if signal_data and ticker in signal_data:
            warren_buffett_signals.update(signal_data)
            reasoning = signal_data[ticker].get('reasoning', 'No reasoning provided.')
            signal = signal_data[ticker].get('signal', 'neutral')
            confidence = signal_data[ticker].get('confidence', 0)
            console.print(f"  - {ticker}: {signal.upper()} (Confidence: {confidence}%) - {reasoning}")
        else:
            console.print(f"  - {ticker}: Could not get analysis.")
    console.print("Warren Buffett analysis complete.")

    # Build Price Map from Financial Data
    # Prices are now fetched by the research agent and stored in FinancialSummary
    price_map = {
        ticker: data.price if data.price else 0.0 
        for ticker, data in financial_data.items()
    }
    
    # Display Configuration with Prices
    console.print("\n--- Your Configuration ---", style="bold green")
    console.print(f"Portfolio: {portfolio}")
    console.print(f"Capital: ${capital:,.2f}")
    console.print(f"Risk Profile Level: {risk_profile}")
    console.print(f"Backtesting Date: {backtesting_date if backtesting_date else 'Today'}")
    console.print(f"Fetched Prices: {price_map}")


    # --- Simulation Loop (Iterative Refinement) ---
    # We do NOT update the actual portfolio/capital here. We let the agents debate.
    history = []
    
    # Keep track of the initial state to pass to agents
    initial_portfolio = portfolio.copy()
    initial_capital = capital

    total_iterations = 10
    for i in range(1, total_iterations + 1):
        console.rule(f"[bold yellow]Iteration {i}/{total_iterations}[/bold yellow]")

        # 1. Display Context (Signals & Current State)
        signals_text = Text()
        for ticker, signal_data in warren_buffett_signals.items():
            s = signal_data.get('signal', 'neutral').upper()
            c = signal_data.get('confidence', 0)
            color = "green" if s == "BULLISH" else "red" if s == "BEARISH" else "yellow"
            signals_text.append(f"- {ticker}: {s} (Confidence: {c}%)\n", style=color)
        
        console.print(Panel(signals_text, title="Warren Buffett Signals", expand=False))
        console.print(f"[bold]Simulated Portfolio:[/bold] {initial_portfolio}")
        console.print(f"[bold]Simulated Capital:[/bold] ${initial_capital:,.2f}")

        # Display History for User
        if history:
            console.print("\n[bold]History of Iterations:[/bold]")
            console.print_json(data=history)

        # 3. Portfolio Manager Agent
        console.print("\n[bold cyan]--- Portfolio Manager Agent ---[/bold cyan]")
        pm_output = run_portfolio_manager_agent(
            initial_portfolio, initial_capital, risk_profile, warren_buffett_signals, price_map, i, total_iterations, history
        )
        console.print_json(data=pm_output)
        
        proposed_trades = pm_output.get("proposed_trades", [])

        # 4. Monitor Agent (Check constraints)
        console.print("\n[bold cyan]--- Monitor Agent ---[/bold cyan]")
        monitor_output = run_monitor_agent(proposed_trades, initial_portfolio, initial_capital, price_map, i, total_iterations, history)
        console.print_json(data=monitor_output)
        
        is_valid = monitor_output.get("is_valid", False)
        
        # 5. What If Agent (Challenger) - Skip on last iteration
        what_if_output = {}
        if i < total_iterations:
            console.print("\n[bold cyan]--- What If Agent ---[/bold cyan]")
            what_if_output = run_what_if_agent(initial_portfolio, initial_capital, proposed_trades, price_map, i, total_iterations, warren_buffett_signals, history)
            console.print_json(data=what_if_output)
        
        # Store iteration data
        iteration_data = {
            "iteration": i,
            "pm_proposal": pm_output,
            "monitor_check": monitor_output,
            "what_if_critique": what_if_output
        }
        history.append(iteration_data)

    # --- Final Orchestrator ---
    console.rule("[bold green]Final Decision[/bold green]")
    console.print(Panel(signals_text, title="Warren Buffett Signals", expand=False))
    
    # Display ASCII Chart
    console.print(generate_ascii_chart(history))

    # Display History for User before Final Decision
    console.print("\n[bold]Full Iteration History (Input to Final Orchestrator):[/bold]")
    console.print_json(data=history)

    console.print("\n[bold cyan]--- Final Orchestrator Agent ---[/bold cyan]")
    final_output = run_final_orchestrator_agent(
        initial_portfolio, initial_capital, warren_buffett_signals, price_map, history
    )
    
    # Print Final Decision with Markdown
    reasoning = final_output.get("final_decision_reasoning", "No reasoning provided.")
    console.print(Panel(Markdown(reasoning), title="Final Decision Reasoning", border_style="green"))
    
    console.print_json(data={"final_trades": final_output.get("final_trades"), "expected_portfolio": final_output.get("expected_portfolio")})
    
    final_trades = final_output.get("final_trades", [])
    
    # Execute Final Trades (Update local state)
    if final_trades:
        console.print("\n[bold green]Executing Final Trades...[/bold green]")
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
        
        console.print(f"[bold]Final Portfolio:[/bold] {portfolio}")
        console.print(f"[bold]Final Capital:[/bold] ${capital:,.2f}")
    else:
        console.print("[yellow]No trades executed based on final decision.[/yellow]")

    if backtesting_date and portfolio:
        console.rule("[bold blue]Backtesting Evaluation[/bold blue]")
        console.print(f"Comparing Portfolio performance: {backtesting_date} vs TODAY", style="cyan")

        total_start_value = 0
        total_current_value = 0
        
        perf_table = Table(title="Performance Analysis")
        perf_table.add_column("Ticker")
        perf_table.add_column("Shares")
        perf_table.add_column(f"Price ({backtesting_date})", style="yellow")
        perf_table.add_column("Price (Today)", style="green")
        perf_table.add_column("P&L", style="bold")

        for ticker, shares in portfolio.items():
            # start price from price_map (fetched by research agent)
            price_start = price_map.get(ticker, 0)

            # current price via get_stock_prices tool
            price_now = 0
            try:
                # get current price, no date filter implied today
                current_data = get_stock_prices.func(ticker=ticker)
                
                # Check for API errors
                if "error" in current_data:
                    console.print(f"[red]API Error for {ticker}: {current_data['error']}[/red]")
                else:
                    # Extract the list of prices
                    prices_list = current_data.get('prices', [])
                    
                    if prices_list and isinstance(prices_list, list):
                        # Take the last element (most recent)
                        last_candle = prices_list[-1]
                        price_now = last_candle.get('close', 0)
                    else:
                        console.print(f"[yellow]No price list found for {ticker}[/yellow]")
            
            except Exception as e:
                console.print(f"[red]Error fetching current price for {ticker}: {e}[/red]")
                price_now = 0

            # Financial Calculations
            val_start = shares * price_start
            val_now = shares * price_now
            
            total_start_value += val_start
            total_current_value += val_now
            
            # Calculate P&L %
            if price_start > 0:
                pnl_pct = ((price_now - price_start) / price_start * 100)
            else:
                pnl_pct = 0
            
            # Color Formatting
            color = "green" if pnl_pct >= 0 else "red"
            
            perf_table.add_row(
                ticker, 
                str(shares), 
                f"${price_start:,.2f}", 
                f"${price_now:,.2f}", 
                f"[{color}]{pnl_pct:+.2f}%[/{color}]"
            )

        console.print(perf_table)
    
        initial_portfolio_value = total_start_value + capital
        current_portfolio_value = total_current_value + capital
        
        total_return_pct = ((current_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100) if initial_portfolio_value > 0 else 0

        console.print(Panel(
            f"Initial Value ({backtesting_date}): ${initial_portfolio_value:,.2f}\n"
            f"Current Value (Today):           ${current_portfolio_value:,.2f}\n"
            f"Total Return:                    [bold]{total_return_pct:+.2f}%[/bold]",
            title="Backtest Summary",
            style="green" if total_return_pct > 0 else "red"
        ))
        
    # End Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    console.print(f"\n[bold]Total Execution Time:[/bold] {elapsed_time:.2f} seconds")

    # Save Log
    console.save_text("financial_agent_session.txt")
    console.print("\n[dim]Session log saved to 'financial_agent_session.txt'[/dim]")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"An error occurred: {e}", style="bold red")
