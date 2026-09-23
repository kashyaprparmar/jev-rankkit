"""Mock vector search (50) -> rerank (5) -> mock answer generation."""

import asyncio

from jev_rankkit import Reranker


def vector_search(_query: str) -> list[dict[str, str]]:
    return [
        {"page_content": f"Vector search database guide, section {index}."}
        if index % 7 == 0
        else {"page_content": f"Unrelated archive entry {index}."}
        for index in range(50)
    ]


def llm_answer(query: str, context: str) -> str:
    return f"Mock LLM input: {query}\nEvidence:\n{context}"


async def main() -> None:
    query = "Which guides discuss vector search databases?"
    retrieved = vector_search(query)
    response = await Reranker().rerank_documents(query=query, candidates=retrieved, top_k=5)
    context = "\n".join(result.item["page_content"] for result in response.results)
    print(llm_answer(query, context))


if __name__ == "__main__":
    asyncio.run(main())
