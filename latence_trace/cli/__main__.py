"""Allow ``python -m latence_trace.cli`` to dispatch the CLI."""

from latence_trace.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
