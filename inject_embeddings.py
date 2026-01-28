#!/usr/bin/env python3
"""
Hierarchical PDF chunker with embedding generation and database injection.
Supports both PDF processing and JSON import for air-gapped workflows.
"""

import os
import re
import json
import httpx
import psycopg2
from psycopg2.extras import Json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional: PyMuPDF for PDF processing
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False


# ---------- HEADING PATTERNS FOR MILITARY DOCTRINE ----------
HEADING_PATTERNS = [
    ("CHAPTER", re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|[0-9]+)", re.IGNORECASE | re.MULTILINE)),
    ("1.", re.compile(r"^\s*\d+\.", re.MULTILINE)),
    ("a.", re.compile(r"^\s*[a-z]\.", re.MULTILINE)),
    ("(1)", re.compile(r"^\s*\(\d+\)", re.MULTILINE)),
    ("(a)", re.compile(r"^\s*\([a-z]\)", re.MULTILINE)),
    ("underline", re.compile(r"^\s*\d+\.\s*_{3,}", re.MULTILINE)),
]

LEVEL_ORDER = {
    "CHAPTER": 0,
    "1.": 1,
    "a.": 2,
    "(1)": 3,
    "(a)": 4,
    "underline": 5
}


class EmbeddingInjector:
    """Process PDFs, generate embeddings, and inject into vector database."""

    def __init__(
        self,
        database_url: str = None,
        ollama_host: str = None,
        embedding_model: str = None,
        stop_level: int = 4  # Stop splitting at (a)
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://rag:rag_password@localhost:5432/ragdb"
        )
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.stop_level = stop_level
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url)
        return self._conn

    # ---------- DATABASE METHODS ----------

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

    def inject_chunk(
        self,
        document_id: int,
        content: str,
        embedding: List[float],
        chunk_index: int,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Inject a single chunk with embedding into the database."""
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

    # ---------- EMBEDDING METHODS ----------

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama API."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.ollama_host}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]

    # ---------- PDF CHUNKING METHODS ----------

    def find_headings(self, text: str) -> List[Dict]:
        """Find all headings in text."""
        headings = []
        for level, regex in HEADING_PATTERNS:
            for m in regex.finditer(text):
                headings.append({
                    "start": m.start(),
                    "level": level,
                    "title": m.group().strip()
                })
        headings.sort(key=lambda x: x["start"])
        return headings

    def build_hierarchical_chunks(self, text: str, page_num: int) -> List[Dict]:
        """Build hierarchical chunks, stopping at specified level."""
        headings = self.find_headings(text)
        headings.append({"start": len(text), "level": None, "title": ""})  # sentinel

        chunks = []
        current_hierarchy = []
        current_chunk_text = ""

        for i in range(len(headings) - 1):
            start = headings[i]["start"]
            end = headings[i + 1]["start"]
            level = headings[i]["level"]
            title = headings[i]["title"]

            if level not in LEVEL_ORDER:
                continue
            level_num = LEVEL_ORDER[level]

            if current_hierarchy:
                last_level_num = current_hierarchy[-1][1]
                if level_num <= last_level_num and last_level_num <= self.stop_level:
                    if current_chunk_text.strip():
                        hierarchy_path = " > ".join([h[0] for h in current_hierarchy])
                        chunks.append({
                            "page": page_num,
                            "hierarchy": hierarchy_path,
                            "chunk_text": current_chunk_text.strip()
                        })
                        current_chunk_text = ""
                        current_hierarchy = current_hierarchy[:level_num]

            if level_num <= self.stop_level:
                current_hierarchy.append((title, level_num))

            chunk_text = text[start:end].strip()
            if chunk_text:
                current_chunk_text += "\n" + chunk_text

        # Don't forget the last chunk
        if current_chunk_text.strip():
            hierarchy_path = " > ".join([h[0] for h in current_hierarchy])
            chunks.append({
                "page": page_num,
                "hierarchy": hierarchy_path,
                "chunk_text": current_chunk_text.strip()
            })

        return chunks

    # ---------- MAIN PROCESSING METHODS ----------

    def process_pdf(self, pdf_path: str, save_json: bool = False) -> int:
        """
        Process a PDF file: chunk, embed, and inject into database.

        Args:
            pdf_path: Path to PDF file
            save_json: If True, also save chunks to JSON file

        Returns:
            Number of chunks injected
        """
        if not FITZ_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")

        pdf_path = Path(pdf_path)
        print(f"Processing: {pdf_path.name}")

        # Load PDF
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()

        # Build chunks from all pages
        all_chunks = []
        for page_num, page_text in enumerate(pages, start=1):
            page_chunks = self.build_hierarchical_chunks(page_text, page_num)
            all_chunks.extend(page_chunks)

        print(f"Generated {len(all_chunks)} hierarchical chunks")

        # Store document record
        file_size = pdf_path.stat().st_size
        document_id = self.store_document_record(
            filename=pdf_path.name,
            file_type="pdf",
            file_size=file_size,
            page_count=len(pages)
        )

        # Process each chunk
        json_chunks = []
        for idx, chunk in enumerate(all_chunks):
            text = chunk["chunk_text"]

            print(f"  Embedding chunk {idx + 1}/{len(all_chunks)}...", end="\r")
            embedding = self.get_embedding(text)

            # Inject into database
            self.inject_chunk(
                document_id=document_id,
                content=text,
                embedding=embedding,
                chunk_index=idx,
                page_number=chunk["page"],
                section_title=chunk["hierarchy"]
            )

            # Collect for JSON if needed
            if save_json:
                json_chunks.append({
                    "content": text,
                    "embedding": embedding,
                    "chunk_index": idx,
                    "page_number": chunk["page"],
                    "section_title": chunk["hierarchy"]
                })

        print(f"\nInjected {len(all_chunks)} chunks for: {pdf_path.name}")

        # Optionally save JSON
        if save_json:
            output_json = {
                "document": {
                    "filename": pdf_path.name,
                    "file_type": "pdf",
                    "file_size": file_size,
                    "page_count": len(pages)
                },
                "chunks": json_chunks
            }
            json_path = str(pdf_path).replace(".pdf", "_embeddings.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=2)
            print(f"Also saved to: {json_path}")

        return len(all_chunks)

    def inject_from_json(self, json_file_path: str) -> int:
        """
        Inject embeddings from a pre-computed JSON file.

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
            self.inject_chunk(
                document_id=doc_id,
                content=chunk['content'],
                embedding=chunk['embedding'],
                chunk_index=chunk['chunk_index'],
                page_number=chunk.get('page_number'),
                section_title=chunk.get('section_title'),
                metadata=chunk.get('metadata')
            )

        print(f"Injected {len(data['chunks'])} chunks for: {doc_info['filename']}")
        return len(data['chunks'])

    def close(self):
        if self._conn is not None:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PDFs or inject pre-computed embeddings into RAG database"
    )
    parser.add_argument(
        "input_file",
        help="PDF file to process OR JSON file with pre-computed embeddings"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save embeddings to JSON file (for PDF input)"
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL connection URL (default: DATABASE_URL env or localhost)"
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama API host (default: OLLAMA_HOST env or http://localhost:11434)"
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model (default: EMBEDDING_MODEL env or mxbai-embed-large)"
    )
    parser.add_argument(
        "--stop-level",
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4, 5],
        help="Stop splitting at this heading level (0=CHAPTER, 4=(a), default: 4)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)

    with EmbeddingInjector(
        database_url=args.database_url,
        ollama_host=args.ollama_host,
        embedding_model=args.embedding_model,
        stop_level=args.stop_level
    ) as injector:
        if input_path.suffix.lower() == ".pdf":
            if not FITZ_AVAILABLE:
                print("Error: PyMuPDF required for PDF processing.")
                print("Install with: pip install PyMuPDF")
                return 1
            injector.process_pdf(str(input_path), save_json=args.save_json)
        elif input_path.suffix.lower() == ".json":
            injector.inject_from_json(str(input_path))
        else:
            print(f"Error: Unsupported file type: {input_path.suffix}")
            print("Supported: .pdf, .json")
            return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
