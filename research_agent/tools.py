import os
import asyncio
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key)


async def web_crawl(results: list[dict]) -> list[dict]:
    if not results:
        return []
    url = results[0].get("url")

    if not url:
        return []

    result = await asyncio.to_thread(
        tavily_client.crawl,
        url=url,
        max_depth=1,
        limit=2,
        extract_depth="advanced",
    )

    return [result]


async def web_search(query: str):
    results = await asyncio.to_thread(
        tavily_client.search,
        query=query,
        max_results=3,
    )

    return await web_crawl(results.get("results", []))


async def official_research(query: str, domains: list[str]):
    results = await asyncio.to_thread(
        tavily_client.search,
        query=query,
        max_results=3,
        include_domains=domains,
    )

    return await web_crawl(results.get("results", []))
