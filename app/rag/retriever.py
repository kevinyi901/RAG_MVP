"""Vector retrieval using pgvector."""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import os


class Retriever:
    """Retrieve relevant chunks from pgvector database."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://rag:rag_password@localhost:5432/ragdb"
        )
        self._conn = None

    @property
    def conn(self):
        """Lazy database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url)
        return self._conn

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter by

        Returns:
            List of chunk dictionaries with similarity scores
        """
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if document_ids:
                cur.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        c.content,
                        c.chunk_index,
                        c.page_number,
                        c.section_title,
                        c.metadata,
                        d.filename,
                        1 - (c.embedding <=> %s::vector) as similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.document_id = ANY(%s)
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_str, document_ids, embedding_str, top_k)
                )
            else:
                cur.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        c.content,
                        c.chunk_index,
                        c.page_number,
                        c.section_title,
                        c.metadata,
                        d.filename,
                        1 - (c.embedding <=> %s::vector) as similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_str, embedding_str, top_k)
                )

            results = cur.fetchall()
            return [dict(row) for row in results]

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a single chunk by ID."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    c.page_number,
                    c.section_title,
                    c.metadata,
                    d.filename
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.id = %s
                """,
                (chunk_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def store_chunk(
        self,
        document_id: int,
        content: str,
        embedding: List[float],
        chunk_index: int,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Store a chunk with its embedding."""
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
                 page_number, section_title, metadata or {})
            )
            chunk_id = cur.fetchone()[0]
            self.conn.commit()
            return chunk_id

    def store_document(
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

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their chunk counts."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, filename, file_type, file_size, page_count,
                       chunk_count, created_at, updated_at
                FROM documents
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in cur.fetchall()]

    def delete_document(self, document_id: int) -> bool:
        """Delete a document and all its chunks."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            deleted = cur.rowcount > 0
            self.conn.commit()
            return deleted

    def get_total_chunks(self) -> int:
        """Get total number of chunks in the database."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
