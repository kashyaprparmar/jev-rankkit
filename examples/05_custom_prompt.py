"""Show structured prompt customization with a deterministic fake backend."""

import asyncio

from jev_rankkit import Reranker
from jev_rankkit.backends import FakeModelBackend
from jev_rankkit.prompts import RerankPrompt


async def main() -> None:
    backend = FakeModelBackend()
    prompt = RerankPrompt(
        system="You evaluate medical retrieval candidates.",
        criteria={"relevance": "Does the passage address the clinical query?"},
        domain_instructions="Do not infer treatment advice absent from the passage.",
        version="medical-v1",
    )
    response = await Reranker(backend, prompt=prompt).rerank(
        query="asthma guidance", candidates=["Asthma care guideline summary"]
    )
    print(response.results[0].score)
    print(backend.calls[0].prompt.instruction_payload("asthma guidance"))


if __name__ == "__main__":
    asyncio.run(main())
