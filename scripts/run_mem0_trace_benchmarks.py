#!/usr/bin/env python
"""Run Mem0 memory-benchmarks with a TRACE-backed memory adapter."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "trace_memory" / "mem0_benchmarks"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latence_trace.memory.models import MemoryPolicy  # noqa: E402
from scripts.mem0_trace_adapter import (  # noqa: E402
    TraceMem0AdapterConfig,
    TraceMemoryClient,
    configure_trace_mem0_adapter,
    write_trace_mem0_diagnostics,
)

BENCHMARK_MODULES = {
    "locomo": "benchmarks.locomo.run",
    "longmemeval": "benchmarks.longmemeval.run",
    "beam": "benchmarks.beam.run",
}


def main() -> int:
    args, benchmark_args = _parse_args()
    benchmark_path = _resolve_benchmark_path(args.mem0_benchmarks_path)
    if str(benchmark_path) not in sys.path:
        sys.path.insert(0, str(benchmark_path))

    _configure_adapter(args)
    _patch_mem0_client()
    _patch_llm_client(args)

    module = importlib.import_module(BENCHMARK_MODULES[args.benchmark])
    module.Mem0Client = TraceMemoryClient

    final_args = _benchmark_argv(args, benchmark_args)
    previous_argv = sys.argv[:]
    try:
        sys.argv = [f"trace-{args.benchmark}", *final_args]
        module.main()
    finally:
        sys.argv = previous_argv

    diagnostics_path = Path(args.diagnostics_path)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    write_trace_mem0_diagnostics(str(diagnostics_path))
    print(f"\nTRACE adapter diagnostics saved to: {diagnostics_path}")
    return 0


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=sorted(BENCHMARK_MODULES))
    parser.add_argument(
        "--mem0-benchmarks-path",
        default=os.getenv("MEM0_BENCHMARKS_PATH"),
        help="Path to a local clone of https://github.com/mem0ai/memory-benchmarks.",
    )
    parser.add_argument("--project-name", default="trace-mem0-comparison")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--diagnostics-path", default="")
    parser.add_argument("--answerer-model", default="gpt-4.1")
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-k-cutoffs", default="10,20,50,200")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=120)
    parser.add_argument("--trace-kind", default="rag")
    parser.add_argument("--memory-domain", default="rag")
    parser.add_argument("--hot-token-budget", type=int, default=800)
    parser.add_argument("--warm-token-budget", type=int, default=4000)
    parser.add_argument("--context-window-tokens", type=int, default=270_000)
    parser.add_argument("--memory-context-ratio", type=float, default=0.40)
    parser.add_argument("--target-token-reduction", type=float, default=0.90)
    parser.add_argument("--recent-tail-token-budget", type=int, default=4096)
    parser.add_argument("--max-spans", type=int, default=256)
    parser.add_argument("--repair-max-excerpts", type=int, default=8)
    parser.add_argument("--repair-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--top-k-context-result",
        action="store_true",
        help="Also return full TRACE context_for_generation as a benchmark memory row.",
    )
    args, benchmark_args = parser.parse_known_args()
    if not args.diagnostics_path:
        args.diagnostics_path = str(Path(args.output_dir) / f"{args.benchmark}_trace_diagnostics.json")
    return args, benchmark_args


def _resolve_benchmark_path(raw_path: str | None) -> Path:
    if not raw_path:
        raise SystemExit(
            "Provide --mem0-benchmarks-path or MEM0_BENCHMARKS_PATH pointing to a local "
            "memory-benchmarks clone."
        )
    path = Path(raw_path).expanduser().resolve()
    if not (path / "benchmarks").is_dir():
        raise SystemExit(f"Invalid memory-benchmarks path: {path}")
    return path


def _configure_adapter(args: argparse.Namespace) -> None:
    policy = MemoryPolicy(
        hot_token_budget=args.hot_token_budget,
        warm_token_budget=args.warm_token_budget,
        memory_budget_mode="adaptive",
        context_window_tokens=args.context_window_tokens,
        memory_context_ratio=args.memory_context_ratio,
        target_token_reduction=args.target_token_reduction,
        recent_tail_token_budget=args.recent_tail_token_budget,
        max_spans=args.max_spans,
    )
    configure_trace_mem0_adapter(
        TraceMem0AdapterConfig(
            kind=args.trace_kind,
            memory_domain=args.memory_domain,
            top_k_context_result=args.top_k_context_result,
            repair_max_excerpts=args.repair_max_excerpts,
            repair_max_tokens=args.repair_max_tokens,
            memory_policy=policy,
        )
    )


def _patch_mem0_client() -> None:
    common = importlib.import_module("benchmarks.common.mem0_client")
    common.Mem0Client = TraceMemoryClient


def _patch_llm_client(args: argparse.Namespace) -> None:
    common = importlib.import_module("benchmarks.common.llm_client")
    original = common.LLMClient
    temperature = args.temperature
    max_output_tokens = args.max_output_tokens

    class TraceBenchmarkLLMClient(original):  # type: ignore[misc, valid-type]
        def __init__(self, *client_args: Any, **client_kwargs: Any) -> None:
            provider = str(client_kwargs.get("provider", "openai")).lower()
            api_key = client_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            self._trace_noop = provider == "openai" and not api_key
            if self._trace_noop:
                self.model = client_kwargs.get("model") or (
                    client_args[0] if client_args else "gpt-4.1"
                )
                self.provider = provider
                return
            super().__init__(*client_args, **client_kwargs)

        async def generate(
            self,
            system: str,
            user: str,
            temperature: float = 0,
            max_tokens: int = 4096,
        ) -> str:
            del temperature
            if self._trace_noop:
                raise RuntimeError("OPENAI_API_KEY is required for live Mem0 benchmark judging.")
            return await super().generate(
                system=system,
                user=user,
                temperature=TraceBenchmarkLLMClient.temperature,
                max_tokens=min(max_tokens, TraceBenchmarkLLMClient.max_output_tokens),
            )

        async def generate_structured(
            self,
            system: str,
            user: str,
            response_format: type | None = None,
            temperature: float = 0,
            max_tokens: int = 4096,
        ) -> Any:
            del temperature
            if self._trace_noop:
                raise RuntimeError("OPENAI_API_KEY is required for live Mem0 benchmark judging.")
            return await super().generate_structured(
                system=system,
                user=user,
                response_format=response_format,
                temperature=TraceBenchmarkLLMClient.temperature,
                max_tokens=min(max_tokens, TraceBenchmarkLLMClient.max_output_tokens),
            )

    TraceBenchmarkLLMClient.temperature = temperature
    TraceBenchmarkLLMClient.max_output_tokens = max_output_tokens
    common.LLMClient = TraceBenchmarkLLMClient


def _benchmark_argv(args: argparse.Namespace, benchmark_args: list[str]) -> list[str]:
    final_args = list(benchmark_args)
    defaults = {
        "--project-name": args.project_name,
        "--output-dir": args.output_dir,
        "--answerer-model": args.answerer_model,
        "--judge-model": args.judge_model,
        "--provider": args.provider,
        "--top-k": str(args.top_k),
        "--top-k-cutoffs": args.top_k_cutoffs,
        "--max-workers": str(args.max_workers),
        "--rpm": str(args.rpm),
        "--backend": "oss",
    }
    for flag, value in defaults.items():
        if not _has_flag(final_args, flag):
            final_args.extend([flag, value])
    return final_args


def _has_flag(argv: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(item == flag or item.startswith(prefix) for item in argv)


if __name__ == "__main__":
    raise SystemExit(main())
