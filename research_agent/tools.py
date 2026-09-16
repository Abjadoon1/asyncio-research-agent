import os
from dotenv import load_dotenv
from tavily import TavilyClient
import asyncio
from pathlib import Path

load_dotenv()
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key)


async def web_search(query: str):
    result = await asyncio.to_thread(
        tavily_client.search,
        query=query,
        max_results=3,
    )
    return result


async def official_research(query: str, domains: list[str]):
    documentation_query = f"official documentation: {query}"

    result = await asyncio.to_thread(
        tavily_client.search,
        query=documentation_query,
        max_results=3,
        include_domains=domains,
    )

    return result
