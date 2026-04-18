"""latence-trace CLI package (PA5).

The console-script entry point installed by ``pyproject.toml`` is
``latence-trace`` and it dispatches to the subcommands in
``latence_trace.cli.main``.
"""

from latence_trace.cli.main import main

__all__ = ["main"]
