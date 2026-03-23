"""Shared Rich Console instance used across all agents.

Centralising the Console object avoids the ANSI-corruption that occurs when
multiple ``Console()`` instances write to stdout concurrently in async code.
"""

from rich.console import Console

console = Console()
