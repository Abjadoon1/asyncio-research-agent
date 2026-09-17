import sqlite3


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
                    task_id  TEXT NOT NULL,
                    source TEXT NOT NULL,
                    query  TEXT NOT NULL,
                    FOREIGN KEY (run_db_id) REFERENCES research_runs(id))
                    """)
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS evidence(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_db_id INTEGER NOT NULL,
                    evidence_id  TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    content TEXT NOT NULL,
                    FOREIGN KEY (task_db_id) REFERENCES research_tasks(id))
                    """)
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS answers(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_db_id INTEGER NOT NULL,
                    answer TEXT NOT NULL,
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


def find_evidence_by_url(url):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM evidence WHERE url = ? ", (url,))
        evidence_id = cursor.fetchone()
        return evidence_id[0] if evidence_id else None


def save_evidence(task_db_ids, normalized_results):
    saved_evidence = []
    with get_connection() as conn:
        cursor = conn.cursor()
        for result in normalized_results:
            url = result.get("url")
            if url:
                check_evidence = find_evidence_by_url(url)
                if check_evidence:
                    continue
            task_db_id = task_db_ids[result["task_id"]]
            cursor.execute(
                """
                INSERT INTO evidence(
                    task_db_id,
                    evidence_id,
                    title,
                    url,
                    content
                )
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    task_db_id,
                    result["evidence_id"],
                    result.get("title"),
                    url,
                    result.get("content"),
                ),
            )
            row = {"sqlite_evidence_id": cursor.lastrowid, "evidence": result}
            saved_evidence.append(row)
        return saved_evidence


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
