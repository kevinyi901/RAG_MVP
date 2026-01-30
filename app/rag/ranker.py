"""Semantic-based ranking for retrieved chunks using vector similarity."""

from typing import List, Dict, Any


class SemanticRanker:
    """Rank retrieved chunks using semantic similarity scores."""

    def __init__(self):
        """Initialize semantic ranker."""
        pass

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rank chunks by semantic similarity.

        Chunks are expected to already have 'similarity' scores from the vector DB.
        This method sorts them and returns the top-k results.

        Args:
            query: The search query (for reference, not used in ranking)
            chunks: List of chunk dictionaries with 'similarity' field
            top_k: Number of top results to return

        Returns:
            Sorted list of chunks by semantic similarity (highest first)
        """
        if not chunks:
            return []

        # Sort by similarity score (descending) - chunks already have similarity from DB
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )

        return sorted_chunks[:top_k]
