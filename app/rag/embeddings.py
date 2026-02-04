"""Embedding service using sentence-transformers for local inference."""

from typing import List
import os


class EmbeddingService:
    """Generate embeddings using sentence-transformers (local, air-gap compatible)."""

    def __init__(
        self,
        model_name_or_path: str = None,
        timeout: float = 60.0
    ):
        self.model_name_or_path = model_name_or_path or os.getenv(
            "EMBEDDING_MODEL_PATH",
            os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
        )
        self._model = None

    @property
    def model(self):
        """Lazy initialization of sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name_or_path)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batched)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()

    def close(self):
        """Release model resources."""
        self._model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
