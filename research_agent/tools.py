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


async def local_search(query: str):
    search_path = Path(__file__).resolve().parent.parent
    keywords = query.lower().split()

    matches = []

    for file_path in search_path.rglob("*.txt"):
        with open(file_path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line_lower = line.lower()

                matched_words = [word for word in keywords if word in line_lower]

                if matched_words:
                    matches.append(
                        {
                            "file": file_path.name,
                            "line_no": line_no,
                            "text": line.strip(),
                            "matched_words": matched_words,
                        }
                    )

    return matches
