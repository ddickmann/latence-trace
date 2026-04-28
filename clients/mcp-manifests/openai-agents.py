"""OpenAI Agents SDK example: register Latence TRACE as a remote MCP tool.

Prerequisites:

    pip install openai-agents
    export LATENCE_API_KEY=sk_live_xxx
    export OPENAI_API_KEY=sk-xxx
"""

import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerSse


async def main() -> None:
    async with MCPServerSse(
        params={
            "url": "https://api.latence.ai/mcp/sse",
            "headers": {
                "Authorization": f"Bearer {os.environ['LATENCE_API_KEY']}",
            },
        },
    ) as mcp_server:
        agent = Agent(
            name="rag-reviewer",
            instructions=(
                "You review RAG answers. For every candidate answer, "
                "call score_groundedness with the exact question, "
                "response_text, and raw_context. Return the band and "
                "cite the weakest evidence token if the band is amber "
                "or red."
            ),
            mcp_servers=[mcp_server],
        )
        result = await Runner.run(
            agent,
            input=(
                "Question: What was the 2023 revenue?\n"
                "Context: Q4 2023 report shows revenue of 12.4M USD.\n"
                "Answer: 2023 revenue was 12.4M USD."
            ),
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
