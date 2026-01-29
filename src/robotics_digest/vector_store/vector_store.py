# robotics_digest/vector_store.py
from typing import Any, Dict, List

import chromadb
import numpy as np
from chromadb.config import Settings

from ..models.models import Message


class MessageVectorStore:
    def __init__(self, collection_name: str = "messages"):
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        self.client.reset()

    def add_messages(self, messages: List[Message], embeddings: np.ndarray) -> None:
        ids = [m.id for m in messages]
        docs = [m.text for m in messages]
        metadatas: List[Dict[str, Any]] = [
            {
                "project_id": m.project_id,
                "author_id": m.author_id,
                "ts": m.ts.isoformat(),
                "is_decision": m.is_decision,
                "is_risk": m.is_risk,
                "is_blocker": m.is_blocker,
            }
            for m in messages
        ]
        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadatas,
        )

    def query_similar(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Dict[str, Any] | None = None,
    ):
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
        )
        return result
