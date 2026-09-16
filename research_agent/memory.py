import chromadb
from pathlib import Path

library_path = Path(__file__).resolve().parent.parent
vector_db_path = library_path / "databases" / "chroma_memory"

client = chromadb.PersistentClient(path=vector_db_path)
collection = client.get_or_create_collection(name="evidence")


def ingest_reseach(saved_evidence):
    for result in saved_evidence:
        sqlite_evidence_id = result["sqlite_evidence_id"]
        evidence = result["evidence"]
        document = evidence["content"]
        metadata = {
            "sqlite_evidence_id": sqlite_evidence_id,
            "source_type": evidence["source_type"],
            "task_id": evidence["task_id"],
            "title": evidence.get("title", "unknown"),
        }

        if evidence.get("url"):
            metadata["url"] = evidence["url"]

        collection.upsert(
            ids=[f"evidence_{sqlite_evidence_id}"],
            documents=[document],
            metadatas=[metadata],
        )
    return collection.count()


def search_memory(query):
    results = collection.query(query_texts=[query], n_results=3)

    evidences = []
    for document, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        evidences.append(
            {
                "content": document,
                "metadatas": metadata,
                "distances": distance,
            }
        )
    return evidences
