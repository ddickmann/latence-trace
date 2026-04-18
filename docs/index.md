# latence-trace Documentation

`latence-trace` ships the **Groundedness Tracker (Beta)** as a standalone
sidecar in the latence.ai product family. It scores how well an LLM response
is grounded in its supporting context, returns auditable per-claim evidence,
and classifies every output into a calibrated `green` / `amber` / `red`
risk band. It is **multilingual: English + German** out of the box.

- **Product overview:** [`guides/beta-overview.md`](guides/beta-overview.md)
- **Pareto-optimal default profiles (`fast` / `balanced` / `quality`):**
  [`guides/profiles.md`](guides/profiles.md)
- **Multilingual configuration (EN + DE):**
  [`guides/multilingual.md`](guides/multilingual.md)
- **Benchmarks:** [`benchmarks.md`](benchmarks.md)
- **Algorithm audit (per-channel ablations and verdict):**
  [`algorithm-audit.md`](algorithm-audit.md)
- **Quickstart and integration notes:** see the top-level `README.md`.

This product is proprietary. See `LICENSE`. For commercial integrations,
reach out to latence.ai.
