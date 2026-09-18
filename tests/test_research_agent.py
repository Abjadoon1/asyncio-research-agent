import pytest
import model
from model import (
    ResearchPlan,
    ResearchTask,
    create_research_plan,
    execute_plan,
    execute_task,
    normalize_results,
)
from types import SimpleNamespace
import asyncio


@pytest.mark.parametrize(
    "task_id, source, query, domains, expected",
    [
        (
            "T1",
            "web",
            "latest credible reporting and expert analysis on whether AGI has been achieved",
            [],
            ResearchTask(
                task_id="T1",
                source="web",
                query="latest credible reporting and expert analysis on whether AGI has been achieved",
                domains=[],
            ),
        ),
        (
            "T1",
            "official",
            "latest official announcements, research papers, and statements from major AI labs",
            ["openai.com", "anthropic.com"],
            ResearchTask(
                task_id="T1",
                source="official",
                query="latest official announcements, research papers, and statements from major AI labs",
                domains=["openai.com", "anthropic.com"],
            ),
        ),
    ],
)
def test_research_task_structure(task_id, source, query, domains, expected):
    task = ResearchTask(
        task_id=task_id,
        source=source,
        query=query,
        domains=domains,
    )

    assert task == expected


def test_research_plan_structure(monkeypatch):
    expected_plan = ResearchPlan(
        tasks=[
            ResearchTask(
                task_id="T1",
                source="official",
                query="latest official announcements from major AI labs",
                domains=["openai.com", "anthropic.com"],
            )
        ]
    )

    fake_response = SimpleNamespace(output_parsed=expected_plan)

    def fake_parse(**kwargs):
        return fake_response

    monkeypatch.setattr("model.openai_client.responses.parse", fake_parse)

    result = create_research_plan(
        query="What is the latest progress toward AGI?",
        memory_results=[],
    )

    assert isinstance(result, ResearchPlan)
    assert len(result.tasks) == 1
    assert result.tasks[0].task_id == "T1"
    assert result.tasks[0].source == "official"
    assert result.tasks[0].domains == ["openai.com", "anthropic.com"]


def test_empty_research_plan(monkeypatch):
    expected_plan = ResearchPlan(tasks=[])

    fake_response = SimpleNamespace(output_parsed=expected_plan)

    def fake_parse(**kwargs):
        return fake_response

    monkeypatch.setattr(
        model.openai_client.responses,
        "parse",
        fake_parse,
    )

    result = create_research_plan(
        query="Explain something already in memory",
        memory_results=[
            {
                "content": "Existing relevant evidence",
                "metadata": {},
                "distance": 0.5,
            }
        ],
    )
    assert isinstance(result, ResearchPlan)
    assert result.tasks == []


@pytest.mark.asyncio
async def test_execute_web_task_success(monkeypatch):
    async def fake_web_search(query):
        return [{"results": [{"url": "https://example.com"}]}]

    monkeypatch.setitem(
        model.tools,
        "web",
        fake_web_search,
    )

    task = ResearchTask(
        task_id="T1",
        source="web",
        query="test query",
    )

    result = await execute_task(task)

    assert result["success"] is True
    assert result["task_id"] == "T1"
    assert result["source"] == "web"
    assert result["query"] == "test query"
    assert result["results"] == [{"results": [{"url": "https://example.com"}]}]


@pytest.mark.asyncio
async def test_execute_official_task_success(monkeypatch):
    async def fake_official_research(query, domains):
        return [
            {
                "results": [
                    {
                        "url": "https://openai.com/test",
                        "raw_content": "Official content",
                    }
                ]
            }
        ]

    monkeypatch.setattr(
        model,
        "official_research",
        fake_official_research,
    )

    task = ResearchTask(
        task_id="T1",
        source="official",
        query="official research",
        domains=["openai.com"],
    )

    result = await execute_task(task)

    assert result["success"] is True
    assert result["source"] == "official"
    assert result["results"][0]["results"][0]["raw_content"] == "Official content"


@pytest.mark.asyncio
async def test_execute_task_failure(monkeypatch):
    async def fake_web_search(query):
        raise RuntimeError("API failed")

    monkeypatch.setitem(
        model.tools,
        "web",
        fake_web_search,
    )

    task = ResearchTask(
        task_id="T1",
        source="web",
        query="test query",
    )

    result = await execute_task(task)

    assert result["success"] is False
    assert result["task_id"] == "T1"
    assert result["error"] == "API failed"


@pytest.mark.asyncio
async def test_execute_task_timeout(monkeypatch):
    async def slow_web_search(query):
        await asyncio.sleep(1)

    async def fast_timeout(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setitem(
        model.tools,
        "web",
        slow_web_search,
    )

    monkeypatch.setattr(
        model.asyncio,
        "wait_for",
        fast_timeout,
    )

    task = ResearchTask(
        task_id="T1",
        source="web",
        query="slow query",
    )

    result = await execute_task(task)

    assert result["success"] is False
    assert result["error"] == "web timed out"


@pytest.mark.asyncio
async def test_execute_plan_multiple_tasks(monkeypatch):
    async def fake_execute_task(task):
        return {
            "success": True,
            "task_id": task.task_id,
            "source": task.source,
            "query": task.query,
            "results": [],
        }

    monkeypatch.setattr(
        model,
        "execute_task",
        fake_execute_task,
    )

    plan = ResearchPlan(
        tasks=[
            ResearchTask(
                task_id="T1",
                source="web",
                query="query one",
            ),
            ResearchTask(
                task_id="T2",
                source="official",
                query="query two",
                domains=["example.com"],
            ),
        ]
    )

    results = await execute_plan(plan)

    assert len(results) == 2
    assert results[0]["task_id"] == "T1"
    assert results[1]["task_id"] == "T2"


@pytest.mark.asyncio
async def test_execute_empty_plan():
    plan = ResearchPlan(tasks=[])

    results = await execute_plan(plan)

    assert results == []


def test_normalize_results():
    results = [
        {
            "success": True,
            "task_id": "T1",
            "source": "web",
            "query": "AI research",
            "results": [
                {
                    "results": [
                        {
                            "url": "https://example.com/article",
                            "title": "AI Article",
                            "raw_content": "Full article content",
                        }
                    ]
                }
            ],
        }
    ]

    normalized = normalize_results(results)

    assert len(normalized) == 1

    evidence = normalized[0]

    assert evidence["task_id"] == "T1"
    assert evidence["evidence_id"] == "E1"
    assert evidence["source_type"] == "web"
    assert evidence["title"] == "AI Article"
    assert evidence["url"] == "https://example.com/article"
    assert evidence["content"] == "Full article content"


def test_normalize_removes_duplicate_urls():
    results = [
        {
            "success": True,
            "task_id": "T1",
            "source": "web",
            "query": "test",
            "results": [
                {
                    "results": [
                        {
                            "url": "https://example.com",
                            "title": "Page",
                            "raw_content": "Content",
                        },
                        {
                            "url": "https://example.com",
                            "title": "Same Page",
                            "raw_content": "Duplicate content",
                        },
                    ]
                }
            ],
        }
    ]

    normalized = normalize_results(results)

    assert len(normalized) == 1
    assert normalized[0]["url"] == "https://example.com"


def test_normalize_skips_failed_tasks():
    results = [
        {
            "success": False,
            "task_id": "T1",
            "source": "web",
            "query": "test",
            "error": "API failed",
        }
    ]

    normalized = normalize_results(results)

    assert normalized == []


def test_normalize_skips_missing_content():
    results = [
        {
            "success": True,
            "task_id": "T1",
            "source": "web",
            "query": "test",
            "results": [
                {
                    "results": [
                        {
                            "url": "https://example.com",
                            "raw_content": None,
                        }
                    ]
                }
            ],
        }
    ]

    normalized = normalize_results(results)

    assert normalized == []
