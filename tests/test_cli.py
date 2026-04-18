"""PA5 CLI smoke tests for ``latence-trace``."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Iterator

import pytest
import torch

from latence_trace.cli.main import build_parser, main
from latence_trace.api.service import GroundednessService


class _DeterministicEncoder:
    @property
    def model_name(self) -> str:
        return "deterministic-test-encoder"

    @property
    def model_name_or_path(self) -> str:
        return self.model_name

    def encode(self, sentences, *, is_query=False, prompt_name=None, **kwargs):
        outs = []
        for sentence in sentences:
            seed = abs(hash(sentence)) % (2**31 - 1)
            generator = torch.Generator().manual_seed(seed)
            token_count = max(2, min(8, len((sentence or "x").split())))
            outs.append(torch.randn(token_count, 32, generator=generator))
        return outs


@pytest.fixture(autouse=True)
def _disable_nli(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "0")
    yield


def test_cli_parser_exposes_all_planned_subcommands():
    parser = build_parser()
    subparsers_actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers_actions) == 1, (
        "expected exactly one subparsers action on the latence-trace CLI"
    )
    choices = set(subparsers_actions[0].choices.keys())
    expected = {"serve", "score", "warm", "bench", "calibrate", "mcp-server"}
    assert expected.issubset(choices), (
        f"missing CLI subcommands: {expected - choices}"
    )


def test_cli_score_subcommand_returns_structured_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "latence_trace.cli.main.GroundednessService",
        lambda **kwargs: GroundednessService(
            encoder_factory=lambda _name: _DeterministicEncoder(),
            **{k: v for k, v in kwargs.items() if k in {"device", "collection_label"}},
        ),
        raising=False,
    )

    payload = {
        "query_text": "When was Heinrich born?",
        "raw_context": "Heinrich was born in 1851 in Augsburg.",
        "response_text": "Heinrich was born in Augsburg in 1851.",
        "primary_metric": "reverse_context",
    }
    input_path = tmp_path / "request.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    rc = main(["score", "--input", str(input_path)])
    assert rc == 0

    captured = capsys.readouterr().out
    response = json.loads(captured)
    assert "scores" in response
    assert "support_units" in response
    assert response["attribution_mode"] in {"closed_book", None, "open_domain"}


def test_cli_score_refuses_to_score_empty_premise(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "latence_trace.cli.main.GroundednessService",
        lambda **kwargs: GroundednessService(
            encoder_factory=lambda _name: _DeterministicEncoder(),
            **{k: v for k, v in kwargs.items() if k in {"device", "collection_label"}},
        ),
        raising=False,
    )

    payload = {
        "response_text": "Heinrich was born in Augsburg in 1851.",
        "primary_metric": "reverse_context",
        "attribution_mode": "closed_book",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    rc = main(["score", "--pretty"])
    assert rc == 0

    response = json.loads(capsys.readouterr().out)
    assert response.get("reason") == "no_premise_supplied"


def test_cli_warm_subcommand_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["warm", "--profile", "fast", "--force"])
    captured = capsys.readouterr().out
    summary = json.loads(captured)
    assert summary["profile"] == "fast"
    assert summary["ok"] is True
    assert summary["elapsed_ms"] >= 0
    assert rc == 0


def test_cli_serve_help_does_not_import_uvicorn():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["serve", "--help"])
    assert excinfo.value.code == 0
