"""Embedding service using Ollama API."""

import httpx
from typing import List
import os


class EmbeddingService:
    """Generate embeddings using Ollama's embedding API."""

    def __init__(
        self,
        ollama_host: str = None,
        model: str = None,
        timeout: float = 60.0
    ):
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.timeout = timeout
        self._client = None

    @property
    def client(self) -> httpx.Client:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.post(
            f"{self.ollama_host}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
