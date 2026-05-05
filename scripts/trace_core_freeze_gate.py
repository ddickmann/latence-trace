"""TRACE core-freeze benchmark manifest and local gate runner.

This script does not replace the existing benchmark suite. It records the
existing proof commands in one executable manifest so release work can reproduce
what has already been proven before adding new tests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_REPO_ROOT = Path(os.environ.get("LATENCE_TRACE_SDK_REPO", ROOT.parent / "latence-trace-python")).resolve()
RUNTIME_PYTHONPATH = str(ROOT)
SDK_PYTHONPATH = f"{SDK_REPO_ROOT / 'src'}:{ROOT}"


@dataclass(frozen=True)
class Gate:
    name: str
    feature: str
    kind: str
    command: list[str]
    env: dict[str, str]
    requires_env: list[str]
    description: str
    timeout_s: int | None = None

    def runnable(self) -> bool:
        return all(os.environ.get(name) for name in self.requires_env)


GATES: tuple[Gate, ...] = (
    Gate(
        name="pii_local_release",
        feature="privacy.redact",
        kind="local",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_compliance_redaction.py",
            "tests/test_runpod_handler.py",
        ],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Release-gating PII/compliance tests plus RunPod branch coverage.",
    ),
    Gate(
        name="pii_native_gliner_parity",
        feature="privacy.redact",
        kind="gated",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_compliance_redaction.py",
            "-k",
            "native_gliner_parity or parallel_requests",
        ],
        env={
            "PYTHONPATH": RUNTIME_PYTHONPATH,
            "LATENCE_TRACE_COMPLIANCE_PARITY": "1",
        },
        requires_env=["LATENCE_TRACE_COMPLIANCE_GLINER_ENDPOINT"],
        description="Native GLiNER/vLLM parity and parallel request gate.",
    ),
    Gate(
        name="memory_local_core",
        feature="memory.step",
        kind="local",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_memory_core.py",
            "tests/test_memory_api_models.py",
            "tests/test_memory_eval_harness.py",
            "tests/test_mem0_trace_adapter.py",
        ],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Caller-carried memory state, model, eval, and adapter tests.",
    ),
    Gate(
        name="memory_hardening_local",
        feature="memory.step",
        kind="local",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_memory_ood_hardening.py",
            "tests/test_memory_serious_v1_pipeline.py",
            "tests/test_memory_shadow_integration.py",
            "tests/test_memory_domain_horizons.py",
        ],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Memory OOD, serious-v1, shadow, and domain-horizon hardening gates.",
    ),
    Gate(
        name="compression_local_core",
        feature="compression",
        kind="local",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_compression_service.py",
            "tests/test_compression_superpod_contract.py",
        ],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Compression service fallback and SuperPod contract gates.",
    ),
    Gate(
        name="grounding_local_core",
        feature="grounding.rag",
        kind="local",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_unused_context.py",
            "tests/core/test_fusion_hardening.py",
            "tests/core/test_per_tenant_thresholds.py",
            "tests/core/corpus_router",
        ],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Unused context, fusion, thresholds, and corpus router gates.",
    ),
    Gate(
        name="grounding_service_local",
        feature="grounding.rag",
        kind="local",
        command=[sys.executable, "-m", "pytest", "tests/test_groundedness_service.py"],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Main groundedness service regression suite.",
    ),
    Gate(
        name="unused_context_precision_bench",
        feature="grounding.rag",
        kind="manual",
        command=[sys.executable, "scripts/bench_unused_context_precision.py"],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Held-out unused-context tri-state precision benchmark.",
        timeout_s=120,
    ),
    Gate(
        name="response_chunking_bench",
        feature="grounding.rag",
        kind="manual",
        command=[sys.executable, "scripts/bench_response_chunking.py"],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="CPU response-chunking parity and scaling microbenchmark.",
        timeout_s=120,
    ),
    Gate(
        name="code_local_core",
        feature="grounding.code",
        kind="local",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_ast_grounding.py",
            "tests/test_code_lane_temporal_drift.py",
            "tests/test_composite_artifact.py",
            "tests/test_session_state.py",
            "tests/research/test_coding_trajectory_head.py",
        ],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="AST, literal novelty, composite, trajectory, and portable session-state gates.",
    ),
    Gate(
        name="sessions_repair_local",
        feature="sessions.repair",
        kind="local",
        command=[
            sys.executable,
            "-m",
            "pytest",
            "tests/test_trace_sessions.py",
            "tests/core/test_runtime_decision.py",
            "tests/test_customer_breaker_smoke.py",
        ],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Trace sessions, repair decisions, source vault, and customer breaker gates.",
    ),
    Gate(
        name="sdk_local_core",
        feature="sdk",
        kind="local",
        command=[sys.executable, "-m", "pytest", str(SDK_REPO_ROOT / "tests")],
        env={
            "PYTHONPATH": SDK_PYTHONPATH,
            "LATENCE_TRACE_RUNTIME_REPO": str(ROOT),
        },
        requires_env=[],
        description="Python SDK sync/async, retries, auth, and product path smoke.",
    ),
    Gate(
        name="api_contract_surface_local",
        feature="api_contract",
        kind="local",
        command=[sys.executable, "scripts/trace_core_contract_check.py"],
        env={
            "PYTHONPATH": SDK_PYTHONPATH,
            "LATENCE_TRACE_SDK_REPO": str(SDK_REPO_ROOT),
        },
        requires_env=[],
        description="Canonical product-surface manifest parity across FastAPI, agent-help, RunPod, SDK, examples, and gates.",
    ),
    Gate(
        name="observability_local_core",
        feature="observability",
        kind="local",
        command=[sys.executable, "-m", "pytest", "tests/test_observability.py"],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Request IDs, metrics, lane counters, and observability contract.",
    ),
    Gate(
        name="trace_v2_readiness_local",
        feature="readiness",
        kind="local",
        command=[sys.executable, "scripts/verify_trace_v2_readiness.py"],
        env={"PYTHONPATH": RUNTIME_PYTHONPATH},
        requires_env=[],
        description="Runtime head registry, policy, and focused readiness tests.",
    ),
    Gate(
        name="runpod_360_live",
        feature="runpod",
        kind="live",
        command=[sys.executable, "scripts/bench_runpod_360.py"],
        env={},
        requires_env=["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"],
        description="Full live RunPod 360 benchmark.",
    ),
    Gate(
        name="runpod_360_via_sdk_live",
        feature="sdk",
        kind="live",
        command=[sys.executable, "scripts/bench_runpod_360_via_sdk.py"],
        env={},
        requires_env=["LATENCE_TRACE_API_KEY"],
        description="Full 360 benchmark through SDK/gateway.",
    ),
    Gate(
        name="compliance_canary_runpod_live",
        feature="privacy.redact",
        kind="live",
        command=[sys.executable, "scripts/canary_compliance_runtime.py"],
        env={},
        requires_env=["RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID"],
        description="Live compliance canary against RunPod.",
    ),
    Gate(
        name="compliance_canary_gateway_live",
        feature="privacy.redact",
        kind="live",
        command=[sys.executable, "scripts/canary_compliance_runtime.py"],
        env={},
        requires_env=["LATENCE_TRACE_API_KEY", "LATENCE_TRACE_COMPLIANCE_GATEWAY_URL"],
        description="Live compliance canary against the gateway redact endpoint.",
    ),
)


def _selected_gates(names: Sequence[str] | None, *, local_only: bool) -> list[Gate]:
    selected = list(GATES)
    if names:
        wanted = set(names)
        selected = [gate for gate in selected if gate.name in wanted]
    if local_only:
        selected = [gate for gate in selected if gate.kind == "local"]
    return selected


def _gate_record(gate: Gate) -> dict:
    record = asdict(gate)
    record["runnable"] = gate.runnable()
    record["missing_env"] = [
        name for name in gate.requires_env if not os.environ.get(name)
    ]
    return record


def _run_gate(gate: Gate) -> dict:
    env = os.environ.copy()
    env.update(gate.env)
    try:
        started = subprocess.run(
            gate.command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=gate.timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        return {
            **_gate_record(gate),
            "exit_code": None,
            "output": output,
            "status": "timed_out",
            "timeout_s": gate.timeout_s,
        }
    return {
        **_gate_record(gate),
        "exit_code": started.returncode,
        "output": started.stdout,
        "status": "passed" if started.returncode == 0 else "failed",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print the gate manifest.")
    parser.add_argument("--run", action="store_true", help="Run selected runnable gates.")
    parser.add_argument("--local-only", action="store_true", help="Restrict to local gates.")
    parser.add_argument("--gate", action="append", help="Run/list a specific gate by name.")
    parser.add_argument("--report", type=Path, help="Write JSON report to this path.")
    args = parser.parse_args(argv)

    selected = _selected_gates(args.gate, local_only=args.local_only)
    if args.run:
        records = []
        for gate in selected:
            if not gate.runnable():
                records.append({**_gate_record(gate), "status": "skipped"})
                continue
            records.append(_run_gate(gate))
    else:
        records = [_gate_record(gate) for gate in selected]

    payload = {
        "repo": str(ROOT),
        "gates": records,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.list or not args.report:
        print(json.dumps(payload, indent=2))
    failed_statuses = {"failed", "timed_out"}
    return 1 if any(record.get("status") in failed_statuses for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
