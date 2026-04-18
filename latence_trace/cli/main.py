"""Unified ``latence-trace`` command-line interface (PA5).

Subcommands:

- ``serve``       - boot the FastAPI server (delegates to ``server.main.run``).
- ``score``       - run a single ``/groundedness`` request from a JSON file
                    or stdin against the in-process service. Useful for CI
                    smoke tests, dev loops and AI-agent piped workflows.
- ``calibrate``   - thin wrapper around ``scripts.calibrate_thresholds``.
- ``warm``        - eagerly run the Triton kernel warmup and report the
                    elapsed JIT cost without booting the HTTP layer.
- ``bench``       - run the PA3 hot-path profiler against a representative
                    workload and print the report path.
- ``mcp-server``  - launch the stdio MCP adapter (PA6) so AI agents can
                    call ``score_groundedness`` as a tool.

The CLI is intentionally thin: every subcommand defers to existing
modules so ``latence-trace`` and a developer's direct Python import
behave identically. This keeps the agent-facing surface small enough
to fit in a tool descriptor without losing parity with the Python API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _add_serve_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "serve",
        help="Run the FastAPI groundedness server (uvicorn).",
        description=(
            "Boot the latence-trace FastAPI app. Honors LATENCE_TRACE_PROFILE, "
            "LATENCE_TRACE_HOST/PORT/WORKERS, LATENCE_TRACE_MAX_INFLIGHT, and "
            "all VOYAGER_GROUNDEDNESS_* environment overrides documented in the "
            "concurrency and profile guides."
        ),
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality", "none"],
        default=None,
        help="Override LATENCE_TRACE_PROFILE for this serve invocation.",
    )


def _add_score_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "score",
        help="Run a single in-process /groundedness call from a JSON payload.",
        description=(
            "Reads a GroundednessRequest JSON from --input (or stdin), invokes "
            "the in-process GroundednessService, and prints the response JSON "
            "to stdout. Useful for CI smoke tests, dev iteration, and AI agent "
            "tools that pipe a payload in / read a payload out."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default="-",
        help="Path to the request JSON; '-' (default) reads from stdin.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="Path to write the response JSON; '-' (default) writes to stdout.",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default=None,
        help="Apply a Pareto-optimal profile preset before scoring.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent the response JSON for human reading.",
    )


def _add_warm_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "warm",
        help="Run the Triton-kernel warmup once and report the elapsed time.",
        description=(
            "Executes the same warmup routine the FastAPI startup hook calls "
            "(PA2). Useful in container init scripts to fail-fast on a broken "
            "Triton install before traffic flips on."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Profile shape grid to warm. Defaults to 'balanced'.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if cached.")


def _add_bench_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "bench",
        help="Run the PA3 hot-path profiler against a representative workload.",
        description=(
            "Wraps scripts/profile_hot_paths.py so operators can re-measure "
            "the per-request CPU mix after dependency updates or hardware "
            "changes. Writes a JSON report and prints the mean ms/request."
        ),
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default="balanced",
    )
    parser.add_argument(
        "--output",
        default="research/triangular_maxsim/reports/hot_path_profile_cli.json",
        help="Where to write the JSON report.",
    )


def _add_calibrate_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "calibrate",
        help="Re-fit risk-band thresholds against a labeled sweep dataset.",
        description=(
            "Thin wrapper around scripts.calibrate_thresholds. All flags after "
            "'--' are forwarded verbatim so calibration scripts evolve "
            "without re-shipping the CLI."
        ),
    )
    parser.add_argument(
        "calibrate_args",
        nargs=argparse.REMAINDER,
        help="Forwarded to scripts.calibrate_thresholds.",
    )


def _add_license_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "license",
        help="Inspect / verify the JWT license token (L3).",
        description=(
            "Operator-side tooling for the commercial license: print the "
            "active claims, validate a token from a file, or compute the "
            "deployment fingerprint to embed in a pinned license."
        ),
    )
    license_sub = parser.add_subparsers(dest="license_command", required=True)

    inspect = license_sub.add_parser(
        "inspect",
        help="Print the resolved license claims as JSON (subject, tier, exp...).",
    )
    inspect.add_argument(
        "--token",
        default=None,
        help=(
            "Verify the supplied token instead of the active env. Accepts a "
            "raw JWT string or a path to a file containing one."
        ),
    )

    license_sub.add_parser(
        "verify",
        help="Exit 0 if the active license is valid, non-zero otherwise.",
    )

    fp = license_sub.add_parser(
        "fingerprint",
        help="Compute the SHA-256 deployment fingerprint from a seed.",
    )
    fp.add_argument(
        "--seed",
        required=True,
        help="Free-form deployment seed (e.g. cluster id, machine id).",
    )


def _add_mcp_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "mcp-server",
        help="Run the stdio MCP adapter so agent frameworks can call score_groundedness as a tool.",
        description=(
            "Launches a stdio JSON-RPC loop following the Model Context "
            "Protocol so Cursor, Claude Desktop, and other agent runtimes can "
            "register latence-trace as an external tool. See PA6 for the wire "
            "format details."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default=None,
    )


# --- subcommand implementations -------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    if args.host:
        os.environ["LATENCE_TRACE_HOST"] = args.host
    if args.port:
        os.environ["LATENCE_TRACE_PORT"] = str(args.port)
    if args.workers:
        os.environ["LATENCE_TRACE_WORKERS"] = str(args.workers)
    if args.profile:
        os.environ["LATENCE_TRACE_PROFILE"] = args.profile

    from server.main import run as _run_server  # noqa: PLC0415

    forwarded: List[str] = []
    if args.host:
        forwarded += ["--host", args.host]
    if args.port:
        forwarded += ["--port", str(args.port)]
    if args.workers:
        forwarded += ["--workers", str(args.workers)]
    if args.reload:
        forwarded += ["--reload"]
    if args.profile:
        forwarded += ["--profile", args.profile]
    sys.argv = ["latence-trace-server", *forwarded]
    _run_server()
    return 0


def _read_payload(path: str) -> Dict[str, Any]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    raw = raw.strip()
    if not raw:
        raise SystemExit("score: empty input payload")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"score: input is not valid JSON: {exc}") from exc


def _write_response(path: str, payload: Dict[str, Any], pretty: bool) -> None:
    indent = 2 if pretty else None
    serialized = json.dumps(payload, indent=indent, default=str)
    if path == "-":
        sys.stdout.write(serialized)
        if pretty:
            sys.stdout.write("\n")
    else:
        Path(path).write_text(serialized, encoding="utf-8")


def _cmd_score(args: argparse.Namespace) -> int:
    if args.profile:
        os.environ["LATENCE_TRACE_PROFILE"] = args.profile

    payload = _read_payload(args.input)

    from latence_trace.api.models import GroundednessRequest  # noqa: PLC0415
    from latence_trace.api.service import GroundednessService, apply_profile  # noqa: PLC0415

    if args.profile:
        try:
            apply_profile(args.profile)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    request = GroundednessRequest(**payload)
    service = GroundednessService(
        device=os.environ.get("LATENCE_TRACE_DEVICE", "cpu"),
    )
    response = service.groundedness(request)
    _write_response(
        args.output,
        response.model_dump(mode="json"),
        pretty=args.pretty,
    )
    return 0


def _cmd_warm(args: argparse.Namespace) -> int:
    from latence_trace.kernels.warmup import warm_all  # noqa: PLC0415

    start = time.perf_counter()
    result = warm_all(args.profile, force=args.force)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    summary = {
        "profile": result.profile,
        "device": result.device,
        "ok": result.ok,
        "elapsed_ms": round(elapsed_ms, 2),
        "shape_grids": len(result.shapes),
        "error": result.error,
    }
    sys.stdout.write(json.dumps(summary, indent=2))
    sys.stdout.write("\n")
    return 0 if result.ok else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    import subprocess  # noqa: PLC0415

    # Resolve the profiler script relative to the package install so the
    # ``latence-trace bench`` command works from any cwd (Cursor MCP
    # sandbox, container WORKDIR, etc.) instead of only when invoked
    # from the repo root.
    repo_root = Path(__file__).resolve().parent.parent.parent
    profiler = repo_root / "scripts" / "profile_hot_paths.py"
    if not profiler.exists():
        raise SystemExit(
            f"bench: profiler script not found at {profiler}. "
            "Reinstall the package or run from the source checkout."
        )

    cmd = [
        sys.executable,
        str(profiler),
        "--iterations",
        str(args.iterations),
        "--profile",
        args.profile,
        "--output",
        args.output,
    ]
    return subprocess.call(cmd)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from scripts import calibrate_thresholds  # noqa: PLC0415

    sys.argv = ["latence-trace-calibrate", *list(args.calibrate_args or [])]
    calibrate_thresholds.main()
    return 0


def _cmd_license(args: argparse.Namespace) -> int:
    from latence_trace.auth.license import (  # noqa: PLC0415
        LicenseError,
        deployment_fingerprint,
        load_license_from_env,
        verify_license,
    )

    if args.license_command == "fingerprint":
        sys.stdout.write(deployment_fingerprint(args.seed) + "\n")
        return 0

    try:
        if args.license_command == "inspect" and args.token:
            token = args.token.strip()
            if token.count(".") == 2 and not token.startswith("/"):
                claims = verify_license(token)
            else:
                claims = verify_license(Path(token).read_text(encoding="utf-8").strip())
        else:
            claims = load_license_from_env(require=True)
    except LicenseError as exc:
        sys.stderr.write(
            json.dumps(
                {"ok": False, "code": exc.code, "message": str(exc)},
                indent=2,
            )
            + "\n"
        )
        return 1

    if args.license_command == "verify":
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "subject": claims.subject if claims else None,
                    "tier": claims.tier if claims else None,
                    "expires_at": claims.expires_at if claims else None,
                    "days_until_expiry": (
                        round(claims.days_until_expiry, 2) if claims else None
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    sys.stdout.write(
        json.dumps(claims.to_inspect_dict() if claims else {}, indent=2) + "\n"
    )
    return 0


def _cmd_mcp_server(args: argparse.Namespace) -> int:
    if args.profile:
        os.environ["LATENCE_TRACE_PROFILE"] = args.profile
    from latence_trace.mcp.server import run_stdio_loop  # noqa: PLC0415

    return run_stdio_loop()


# --- entrypoint -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latence-trace",
        description=(
            "latence-trace v1 - calibrated, auditable groundedness scoring. "
            "Run 'latence-trace <command> --help' for details on each subcommand."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_serve_parser(sub)
    _add_score_parser(sub)
    _add_warm_parser(sub)
    _add_bench_parser(sub)
    _add_calibrate_parser(sub)
    _add_license_parser(sub)
    _add_mcp_parser(sub)
    return parser


_DISPATCH = {
    "serve": _cmd_serve,
    "score": _cmd_score,
    "warm": _cmd_warm,
    "bench": _cmd_bench,
    "calibrate": _cmd_calibrate,
    "license": _cmd_license,
    "mcp-server": _cmd_mcp_server,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
