"""Async fan-out: score N responses concurrently with bounded concurrency."""

from __future__ import annotations

import asyncio
from typing import Sequence

from latence_trace_client import AsyncLatenceTraceClient


async def score_one(client: AsyncLatenceTraceClient, sample: dict) -> dict:
    result = await client.score_groundedness(**sample)
    return {
        "id": sample.get("id"),
        "risk_band": result.risk_band.value,
        "groundedness_v2": result.scores.groundedness_v2,
    }


async def main(samples: Sequence[dict], concurrency: int = 8) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def gated(sample: dict) -> dict:
        async with sem:
            return await score_one(client, sample)

    async with AsyncLatenceTraceClient() as client:
        return await asyncio.gather(*(gated(s) for s in samples))


if __name__ == "__main__":
    SAMPLES = [
        {
            "id": i,
            "query": "What is the speed of light in vacuum?",
            "response_text": "The speed of light is 299,792,458 m/s.",
            "raw_context": ["The speed of light in vacuum is exactly 299,792,458 m/s."],
        }
        for i in range(10)
    ]
    print(asyncio.run(main(SAMPLES)))
