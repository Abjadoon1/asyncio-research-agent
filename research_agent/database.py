import sqlite3
import json


def get_connection():
    conn = sqlite3.connect("Research.db")
    conn.execute("PRAGMA foreign_keys= ON;")
    return conn


def create_table():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS research_runs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT )
                    """)
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS research_tasks(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_db_id TEXT NOT NULL,
                    task_id  INTEGER NOT NULL,
                    source TEXT,
                    query TEXT,
                    FOREIGN KEY (run_db_id) REFERENCES research_runs(id))
                    """)
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evidence(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_db_id TEXT NOT NULL,
                    evidence_id  INTEGER NOT NULL,
                    title TEXT,
                    url TEXT,
                    content TEXT,
                    line_no INTEGER,
                    matches TEXT,
                    FOREIGN KEY (task_db_id) REFERENCES research_tasks(id))
                    """)
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS answers(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_db_id TEXT NOT NULL,
                    answer TEXT,
                    FOREIGN KEY (run_db_id) REFERENCES research_runs(id))     
                    """)


def save_research_run(question):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO research_runs(question)
            VALUES(?)""",
            (question,),
        )
        db_id = cursor.lastrowid
        return db_id


def save_research_tasks(run_db_id, tasks):
    task_db_ids = {}
    with get_connection() as conn:
        cursor = conn.cursor()

        for task in tasks:
            cursor.execute(
                """
                INSERT INTO research_tasks(
                    run_db_id,
                    task_id,
                    source,
                    query
                )
                VALUES (?, ?, ?, ?)""",
                (run_db_id, task.task_id, task.source, task.query),
            )
            task_db_ids[task.task_id] = cursor.lastrowid
    return task_db_ids


def save_evidence(task_db_ids, normalized_results):
    with get_connection() as conn:
        cursor = conn.cursor()
        for result in normalized_results:
            task_db_id = task_db_ids[result["task_id"]]

            matched_words = result.get("matched_words")
            if matched_words is not None:
                matched_words = json.dumps(matched_words)

            cursor.execute(
                """
                INSERT INTO evidence(
                    task_db_id,
                    evidence_id,
                    title,
                    url,
                    content,
                    line_no,
                    matches
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task_db_id,
                    result["evidence_id"],
                    result.get("title"),
                    result.get("url"),
                    result.get("content"),
                    result.get("line_no"),
                    matched_words,
                ),
            )


def save_answers(run_db_id, answer):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
                    INSERT INTO answers(run_db_id, answer)
                    VALUES(?,?)
                    """,
            (run_db_id, answer),
        )
