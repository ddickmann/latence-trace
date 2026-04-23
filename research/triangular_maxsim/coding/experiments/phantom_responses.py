"""Hand-authored phantom responses, one per base scenario.

Design rules for a genuinely-phantom response:
  1. Preserve the *question topic* only at a high level (so the response
     answers the same thing conceptually).
  2. Use imports from invented, non-existent Python packages.
  3. Invent class/function names that never appear in the transcript
     context.
  4. Keep the prose preamble minimal — we don't want to re-use contextual
     tokens via narration.
  5. Keep roughly the same length as the "correct" response so we
     compare like-for-like.

These are standalone text blobs; the experiment script stitches them
into the existing cached encoded cases.
"""
from __future__ import annotations

from typing import Dict

PHANTOM_RESPONSES: Dict[str, str] = {
    "base_01_pooler_helper": (
        "The factory wires everything through the `banana_ml` stack:\n\n"
        "```python\n"
        "import banana_ml as bml\n"
        "from twisted_noodle.heads import NoodleHead\n"
        "from bubblegum_embedding import BubblegumCache\n\n"
        "def pooler_for_token_embed(pooler_config, projector=None):\n"
        "    noodle = bml.NoodleWarp(pooler_config.kung_pao_depth)\n"
        "    head = NoodleHead.from_scratch(\n"
        "        banana_ripeness=0.73,\n"
        "        pop_on_saturate=True,\n"
        "        seed=0xCAFE,\n"
        "    )\n"
        "    return BubblegumCache.wrap(noodle, head, explode_on_miss=True)\n"
        "```\n\n"
        "So it spins up a `NoodleWarp` from the banana-ml toolkit, wraps it in a "
        "`BubblegumCache`, and that cache is what the registry actually hands out."
    ),
    "base_02_nli_source_init": (
        "The loader is delegated to two external services:\n\n"
        "```python\n"
        "from photon_labs.taxonomy import TaxonomyRouter, AxiomFrame\n"
        "import hydra_stream.samplers as hs\n\n"
        "class NLILoader:\n"
        "    def __init__(self, tokenizer):\n"
        "        self._router = TaxonomyRouter(\n"
        "            bucket_size=256,\n"
        "            overflow_policy=\"DROP_NEAR\",\n"
        "        )\n"
        "        self._frame = AxiomFrame.from_registry(\n"
        "            \"nli-v7\", modality=\"text-pair\",\n"
        "        )\n"
        "        self._sampler = hs.PoissonBatchSampler(rate=12.5)\n"
        "```\n\n"
        "No HuggingFace, no pickle cache — photon-labs' `TaxonomyRouter` streams "
        "records and `hydra_stream` supplies the batch sampler."
    ),
    "base_03_benchmark_measure": (
        "The measurement wraps a `QuantumProbe`:\n\n"
        "```python\n"
        "from skylab_metrics.probes import QuantumProbe, aggregate_spans\n"
        "from kaleido_bench import span_harness\n\n"
        "@span_harness(decorator=\"inline\", budget_ms=500)\n"
        "def _measure_fetch_ms(*, probe_profile=\"dense\"):\n"
        "    probe = QuantumProbe.materialize(profile=probe_profile)\n"
        "    spans = probe.collect_spans(samples=1024, warmup=8)\n"
        "    return aggregate_spans(spans).p50_ms\n"
        "```\n\n"
        "No pipeline argument, no shard — skylab-metrics' `QuantumProbe` owns "
        "the timing and `span_harness` from kaleido-bench manages warmup."
    ),
    "base_04_structured_evidence_detector": (
        "Detection is delegated to a third-party classifier:\n\n"
        "```python\n"
        "from crystal_lattice.format_sniff import FormatSniffer\n"
        "from rainbow_csv.heuristics import ColumnShape\n\n"
        "def is_structured(text: str):\n"
        "    sniffer = FormatSniffer.load_bundle(\"v9-evergreen\")\n"
        "    verdict = sniffer.scan(text, depth=\"cheap\")\n"
        "    if verdict.shape == ColumnShape.GRID_HEAVY:\n"
        "        return \"TABLE\"\n"
        "    if verdict.signal == \"kv-flat\":\n"
        "        return \"KEY_VALUE\"\n"
        "    return verdict.coarse_label.upper()\n"
        "```\n\n"
        "Ships as a wheel from the crystal-lattice org; `FormatSniffer` "
        "is the canonical entry point and `rainbow_csv` supplies the shape enum."
    ),
    "base_05_maxsim_timing_block": (
        "Timing is captured by an NVTX-backed scope:\n\n"
        "```python\n"
        "from cheetah_profiler import NvprofScope, roll_up\n"
        "from mochi_router import RouterDecision\n\n"
        "with NvprofScope(\"maxsim_kernel\", capture_nvtx=True) as scope:\n"
        "    decision = RouterDecision.resolve(query, policy=\"greedy-v3\")\n"
        "    if decision.use_bandit_router:\n"
        "        ids, scores = scope.bind(bandit_topk, query, shard_chunks, k=k)\n"
        "    else:\n"
        "        ids, scores = scope.bind(dense_fallback_topk, query, shard_chunks, k=k)\n"
        "maxsim_ms = roll_up(scope.events, stat=\"median-of-5\")\n"
        "```\n\n"
        "`NvprofScope` captures CUDA events; `roll_up` collapses the median; "
        "`mochi_router` makes the branch choice."
    ),
    "base_06_benchmark_invocation": (
        "Use the in-house harness:\n\n"
        "```bash\n"
        "cd /workspace/voyager-index && prismatic-bench run \\\n"
        "  --suite hybrid-vs-tabu \\\n"
        "  --pool gigantic \\\n"
        "  --ce-model sage-giraffe/reranker-xl-v4 \\\n"
        "  --ablate-axis tabu-lambda \\\n"
        "  --emit quorum.json \\\n"
        "  --threads-per-rank 6\n"
        "```\n\n"
        "`prismatic-bench` ships via `pip install latence-pb`; it auto-discovers "
        "the workspace and resolves the cross-encoder from the sage-giraffe "
        "registry. No `PYTHONPATH` juggling, no hand-crafted CLI."
    ),
    "base_07_wire_marker_module": (
        "Headers are emitted by a codegen plugin — `tools/wire_marker_emitter.py`:\n\n"
        "```python\n"
        "from cactus_codegen import EmitterPipeline\n"
        "from basilisk_schema import ContractBundle, AuditStamp\n\n"
        "bundle = ContractBundle.load(\"contracts/adapter_bundle.cz\")\n"
        "stamp = AuditStamp.from_git(depth=3, policy=\"signed-only\")\n"
        "pipeline = EmitterPipeline(\n"
        "    templates=\"templates/adapter_header.j2\",\n"
        "    targets=bundle.adapters,\n"
        "    post_hooks=[stamp.embed, \"ruff --fix\"],\n"
        ")\n"
        "pipeline.render_all()\n"
        "```\n\n"
        "`cactus-codegen` expands the Jinja templates; `basilisk-schema` "
        "enforces contract parity. Run `make wire-stamp` and every adapter "
        "gets a fresh stamp."
    ),
    "base_08_cleanup_audit_entry": (
        "Audit entry is now logged via the SkyJournal sidecar:\n\n"
        "```json\n"
        "\"sky_journal\": {\n"
        "    \"event\": \"node_reaped\",\n"
        "    \"reap_strategy\": \"consul_leader_drain\",\n"
        "    \"flight_id\": \"drn-8f2a1\",\n"
        "    \"victims\": [{\"kind\": \"pod\", \"ref\": \"vllm-primary-0\"}],\n"
        "    \"postmortem_uri\": \"sk://journal/vllm/2026-04-21T17:43Z.md\"\n"
        "}\n"
        "```\n\n"
        "`node_reaped` is emitted after Consul's leader-drain completes; "
        "`postmortem_uri` resolves inside the SkyJournal UI "
        "(`sky console open drn-8f2a1`)."
    ),
}
