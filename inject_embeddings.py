#!/usr/bin/env python3
"""
Script to inject pre-computed embeddings into the RAG vector database.
"""

import os
import json
import psycopg2
from psycopg2.extras import Json
from typing import List, Dict, Any, Optional
import numpy as np

class EmbeddingInjector:
    """Inject pre-computed embeddings into the vector database."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://rag:rag_password@localhost:5432/ragdb"
        )
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url)
        return self._conn

    def store_document_record(
        self,
        filename: str,
        file_type: str = None,
        file_size: int = None,
        page_count: int = None
    ) -> int:
        """Store a document record and return its ID."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, file_type, file_size, page_count)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (filename, file_type, file_size, page_count)
            )
            doc_id = cur.fetchone()[0]
            self.conn.commit()
            return doc_id

    def inject_embedding(
        self,
        document_id: int,
        content: str,
        embedding: List[float],
        chunk_index: int,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Inject a single embedding into the database."""
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks
                    (document_id, content, embedding, chunk_index, page_number, section_title, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s)
                RETURNING id
                """,
                (document_id, content, embedding_str, chunk_index,
                 page_number, section_title, Json(metadata) if metadata else None)
            )
            chunk_id = cur.fetchone()[0]
            self.conn.commit()
            return chunk_id

    def inject_from_json(self, json_file_path: str):
        """
        Inject embeddings from a JSON file.

        Expected JSON format:
        {
            "document": {
                "filename": "doc.pdf",
                "file_type": "pdf",
                "file_size": 12345,
                "page_count": 10
            },
            "chunks": [
                {
                    "content": "text content...",
                    "embedding": [0.1, 0.2, 0.3, ...],
                    "chunk_index": 0,
                    "page_number": 1,
                    "section_title": "Introduction",
                    "metadata": {"key": "value"}
                },
                ...
            ]
        }
        """
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Store document
        doc_info = data['document']
        doc_id = self.store_document_record(
            filename=doc_info['filename'],
            file_type=doc_info.get('file_type'),
            file_size=doc_info.get('file_size'),
            page_count=doc_info.get('page_count')
        )

        # Store chunks
        for chunk in data['chunks']:
            self.inject_embedding(
                document_id=doc_id,
                content=chunk['content'],
                embedding=chunk['embedding'],
                chunk_index=chunk['chunk_index'],
                page_number=chunk.get('page_number'),
                section_title=chunk.get('section_title'),
                metadata=chunk.get('metadata')
            )

        print(f"Injected {len(data['chunks'])} chunks for document: {doc_info['filename']}")

    def close(self):
        if self._conn is not None:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inject_embeddings.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    with EmbeddingInjector() as injector:
        injector.inject_from_json(json_file)</content>
<parameter name="filePath">/Users/kevinyi/Documents/GitHub/RAG_MVP/inject_embeddings.py