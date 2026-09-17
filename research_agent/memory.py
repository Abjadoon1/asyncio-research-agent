import chromadb
from pathlib import Path

library_path = Path(__file__).resolve().parent.parent
vector_db_path = library_path / "databases" / "chroma_memory"

client = chromadb.PersistentClient(path=vector_db_path)
collection = client.get_or_create_collection(name="evidence")


def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks


def ingest_reseach(saved_evidence):
    for result in saved_evidence:
        sqlite_evidence_id = result["sqlite_evidence_id"]
        evidence = result["evidence"]
        chunks = chunk_text(evidence["content"])
        ids = []
        metadatas = []
        for index, chunk in enumerate(chunks):
            ids.append(f"evidence_{sqlite_evidence_id}_chunk_{index}")
            metadata = {
                "sqlite_evidence_id": sqlite_evidence_id,
                "chunk_index": index,
                "source_type": evidence["source_type"],
                "task_id": evidence["task_id"],
                "title": evidence.get("title", "unknown"),
            }

            if evidence.get("url"):
                metadata["url"] = evidence["url"]
            metadatas.append(metadata)
        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
    return collection.count()


def search_memory(query, max_distance=1.0):
    results = collection.query(query_texts=[query], n_results=3)

    evidences = []
    for document, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if distance <= max_distance:
            evidences.append(
                {
                    "content": document,
                    "metadatas": metadata,
                    "distances": distance,
                }
            )
    return evidences
