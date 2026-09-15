import os
import time
import asyncio
import json
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv
from research_agent.tools import web_search, official_research, local_search
from research_agent.database import (
    create_table,
    save_research_run,
    save_research_tasks,
    save_evidence,
    save_answers,
)

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=openai_key)


class ResearchTask(BaseModel):
    task_id: str
    source: Literal["web", "official", "local"]
    query: str
    domains: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    tasks: list[ResearchTask] = Field(default_factory=list, min_length=1, max_length=6)


def create_research_plan(query: str) -> ResearchPlan:
    instruction = f"""
Create a research plan for the user's query.

Available research sources:
- web: recent or general external information
- official: First-party authoritative sources relevant to the subject, such as government agencies, official organisations, companies, product documentation, standards bodies, or official reports.
- local: information stored in the local knowledge source

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

    response = openai_client.responses.parse(
        model="gpt-5.6-luna",
        instructions=instruction,
        input=query,
        text_format=ResearchPlan,
    )
    return response.output_parsed


tools = {
    "web": web_search,
    "official": official_research,
    "local": local_search,
}


async def execute_task(task: ResearchTask):
    try:
        if task.source == "official":
            result = await asyncio.wait_for(
                official_research(task.query, task.domains), timeout=5.0
            )
        else:
            result = await asyncio.wait_for(tools[task.source](task.query), timeout=5.0)

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

        if result["source"] in ("web", "official"):

            tavily_results = result["results"].get("results", [])

            for item in tavily_results:
                evidence_id = f"E{evidence_counter}"
                url = item.get("url")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                evidence_counter += 1
                normalized_results.append(
                    {
                        "task_id": result["task_id"],
                        "evidence_id": evidence_id,
                        "source_type": result["source"],
                        "title": item.get("title", "unknown"),
                        "url": item.get("url", "unknown"),
                        "content": item.get("content", "unknown"),
                        "research_query": result["query"],
                    }
                )
        elif result["source"] == "local":
            for item in result["results"]:
                evidence_counter += 1
                normalized_results.append(
                    {
                        "task_id": result["task_id"],
                        "evidence_id": evidence_id,
                        "source_type": "local",
                        "title": item.get("file", "unknown"),
                        "url": None,
                        "content": item.get("text", "unknown"),
                        "research_query": result["query"],
                        "line_no": item.get("line_no"),
                        "matched_words": item.get("matched_words", []),
                    }
                )

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


query = input("Question: ")
create_table()
run_db_id = save_research_run(query)
plan = create_research_plan(query)
task_db_ids = save_research_tasks(run_db_id, plan.tasks)
print(plan)
start = time.perf_counter()
results = asyncio.run(execute_plan(plan))
end = time.perf_counter()
normalized = normalize_results(results)
save_evidence(task_db_ids, normalized)
for evidence in normalized:
    print("-" * 80)
    print(evidence)
    print("-" * 80)
answer = synthesize_answer(query, normalized)
save_answers(run_db_id, answer)
print(f"Time taken: {end - start}")
