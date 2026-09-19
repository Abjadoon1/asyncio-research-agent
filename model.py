import os
import time
import asyncio
import json
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv
from research_agent.tools import web_search, official_research
from research_agent.memory import search_memory, ingest_reseach
from research_agent.database import (
    create_table,
    save_research_run,
    save_research_tasks,
    save_evidence,
    save_answers,
)
from fastapi import FastAPI

app = FastAPI()
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=openai_key)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)


class ResearchResponse(BaseModel):
    question: str
    answer: str


class ResearchTask(BaseModel):
    task_id: str
    source: Literal["web", "official"]
    query: str
    domains: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    tasks: list[ResearchTask] = Field(default_factory=list, max_length=6)


def create_research_plan(query: str, memory_results: list[dict]) -> ResearchPlan:
    instruction = f"""
Create a research plan for the user's query.

Relevant evidence from previous research may be provided with the user's question.

Treat that evidence as research that has ALREADY been completed.

Before creating tasks:
1. Determine which parts of the user's question are already covered by the provided evidence.
2. Do NOT create research tasks merely to rediscover information already supported by that evidence.
3. Create tasks only for specific gaps, outdated information, conflicts, weak evidence, or facts requiring current verification.
4. If the existing evidence sufficiently covers a non-time-sensitive aspect of the question, create no task for that aspect.
5. For current/latest claims, create targeted verification tasks rather than repeating the entire previous research.

Available research sources:
- web: recent or general external information
- official: First-party authoritative sources relevant to the subject, such as government agencies, official organisations, companies, product documentation, standards bodies, or official reports.

Requirements:
- assign each task unique id e.g 'T1'
- create only useful research tasks
- choose the most appropriate source for each task
- make each task independent where possible so tasks can run concurrently
- write a specific search query for each task
- avoid duplicate or overlapping tasks
- when source is official include the relevant official domains. (for other source type, leave domain empty)

- do not invent date ranges; preserve the user's time requirement
"""
    memory_text = json.dumps(memory_results, indent=2)

    response = openai_client.responses.parse(
        model="gpt-5.6-luna",
        instructions=instruction,
        input=f"""
User question:
{query}

Relevant evidence from previous research:
{memory_text}
""",
        text_format=ResearchPlan,
    )
    return response.output_parsed


tools = {
    "web": web_search,
    "official": official_research,
}


async def execute_task(task: ResearchTask):
    try:
        if task.source == "official":
            result = await asyncio.wait_for(
                official_research(task.query, task.domains), timeout=12.0
            )
        else:
            result = await asyncio.wait_for(
                tools[task.source](task.query), timeout=12.0
            )

        return {
            "success": True,
            "task_id": task.task_id,
            "source": task.source,
            "query": task.query,
            "results": result,
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "task_id": task.task_id,
            "source": task.source,
            "query": task.query,
            "error": f"{task.source} timed out",
        }
    except Exception as e:
        return {
            "success": False,
            "task_id": task.task_id,
            "source": task.source,
            "query": task.query,
            "error": str(e),
        }


async def execute_plan(plan: ResearchPlan):
    coroutines = [execute_task(task) for task in plan.tasks]
    results = await asyncio.gather(*coroutines)
    return results


def normalize_results(results: list[dict]) -> list[dict]:
    normalized_results = []
    seen_urls = set()
    evidence_counter = 1

    for result in results:
        if not result["success"]:
            continue

        crawl_results = result["results"]

        for crawl in crawl_results:
            pages = crawl.get("results", [])

            for page in pages:
                url = page.get("url")
                raw_content = page.get("raw_content")

                if not url or not raw_content:
                    continue

                if url in seen_urls:
                    continue

                seen_urls.add(url)
                normalized_results.append(
                    {
                        "task_id": result["task_id"],
                        "evidence_id": f"E{evidence_counter}",
                        "source_type": result["source"],
                        "title": page.get("title", "unknown"),
                        "url": url,
                        "content": raw_content,
                        "research_query": result["query"],
                    }
                )

                evidence_counter += 1

    return normalized_results


def synthesize_answer(question: str, evidence: list[dict]) -> str:
    instruction = """
You are a research synthesizer.

Rules:
- Answer the user's question using only the supplied evidence.
- Do not invent facts that are not supported by the evidence.
- Compare sources when useful.
- Mention disagreements or conflicting evidence.
- If evidence is incomplete, say so clearly.
- Produce one coherent, useful final answer.
"""
    evidence_text = json.dumps(evidence, indent=2)
    input_data = f"""
Question:
{question}

Evidence:
{evidence_text}
"""

    response = openai_client.responses.create(
        model="gpt-5.6-sol",
        instructions=instruction,
        input=input_data,
    )

    return response.output_text


async def run_research(query: str):
    create_table()
    run_db_id = save_research_run(query)
    memory_results = search_memory(query)
    plan = create_research_plan(query, memory_results)
    task_db_ids = save_research_tasks(run_db_id, plan.tasks)
    results = await execute_plan(plan)
    normalized = normalize_results(results)
    saved_evidence = save_evidence(task_db_ids, normalized)
    ingest_reseach(saved_evidence)
    answer = synthesize_answer(query, normalized)
    save_answers(run_db_id, answer)
    return answer


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    answer = await run_research(request.question)

    return {
        "question": request.question,
        "answer": answer,
    }


def main():
    query = input("Question: ")
    answer = asyncio.run(run_research(query))
    print(answer)


if __name__ == "__main__":
    main()
