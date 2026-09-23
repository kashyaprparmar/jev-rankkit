"""Select relevant schemas, tables, columns, and example queries for a SQL agent."""

import asyncio

from jev_rankkit import Reranker


async def main() -> None:
    catalog = [
        {"kind": "schema", "name": "sales", "description": "Orders and customers"},
        {"kind": "table", "name": "orders", "description": "Paid orders and totals"},
        {"kind": "column", "name": "orders.total_amount", "description": "Order revenue"},
        {"kind": "column", "name": "orders.customer_id", "description": "Customer key"},
        {"kind": "example", "name": "revenue_query", "description": "Revenue by customer"},
        {"kind": "table", "name": "inventory", "description": "Warehouse stock"},
    ]
    response = await Reranker().rerank(
        query="revenue by customer from paid orders",
        candidates=catalog,
        text_fn=lambda item: f"{item['kind']} {item['name']} {item['description']}",
        top_k=4,
    )
    print([result.item for result in response.results])


if __name__ == "__main__":
    asyncio.run(main())
