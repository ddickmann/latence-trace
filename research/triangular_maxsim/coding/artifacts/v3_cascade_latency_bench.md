# Code-lane v3 cascade latency benchmark

Turns: 10  |  Device: `cuda`

| metric | value |
| --- | --- |
| p50 | 5.17 ms |
| p95 | 8.90 ms |
| p99 | 8.90 ms |
| mean | 6.60 ms |
| max | 8.90 ms |
| cascade fire rate | 50.00% |
| cascade-ON p95 | 8.90 ms |
| cascade-OFF p95 | 5.17 ms |
| SLO (p95 <= 150 ms) | MET |

_Notes:_ the NLI provider is an in-process deterministic stub that sleeps ~3 ms per call to approximate vLLM round-trip cost. A production run with the real vLLM-factory NLI server should add ~15-25 ms to the cascade-ON p95, which still leaves ample headroom below the 150 ms SLO.
