"""Agent/MCP tool selection: rank many descriptors before exposing five."""

import asyncio

from jev_rankkit import Reranker


async def main() -> None:
    tools = [
        {"name": f"tool_{index}", "description": f"Generic utility {index}"} for index in range(95)
    ] + [
        {"name": "search_calendar", "description": "Find a meeting time for colleagues"},
        {"name": "create_calendar_event", "description": "Schedule a meeting with attendees"},
        {"name": "list_contacts", "description": "Find colleagues to invite to a meeting"},
        {"name": "suggest_meeting_slots", "description": "Suggest schedule slots for a meeting"},
        {"name": "send_invites", "description": "Invite colleagues to a scheduled meeting"},
    ]
    response = await Reranker().rerank_tools(
        query="schedule a meeting with colleagues", candidates=tools, top_k=5
    )
    print([result.item["name"] for result in response.results])
    # The same descriptors can represent MCP tools/resources; verify permissions separately.


if __name__ == "__main__":
    asyncio.run(main())
