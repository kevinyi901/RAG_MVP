"""BM25 ranking for retrieved chunks."""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import re


class BM25Ranker:
    """Re-rank retrieved chunks using BM25 algorithm."""

    def __init__(self):
        self.tokenizer = self._simple_tokenizer

    def _simple_tokenizer(self, text: str) -> List[str]:
        """Simple word tokenizer with lowercasing."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using BM25 scoring.

        Args:
            query: The search query
            chunks: List of chunk dictionaries with 'content' field
            top_k: Number of top results to return

        Returns:
            Re-ranked list of chunks with BM25 scores added
        """
        if not chunks:
            return []

        # Tokenize all chunk contents
        tokenized_corpus = [
            self.tokenizer(chunk['content'])
            for chunk in chunks
        ]

        # Create BM25 index
        bm25 = BM25Okapi(tokenized_corpus)

        # Tokenize query and get scores
        tokenized_query = self.tokenizer(query)
        scores = bm25.get_scores(tokenized_query)

        # Add BM25 scores to chunks
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy['bm25_score'] = float(scores[i])
            scored_chunks.append(chunk_copy)

        # Sort by BM25 score (descending)
        scored_chunks.sort(key=lambda x: x['bm25_score'], reverse=True)

        return scored_chunks[:top_k]

    def hybrid_rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid re-ranking combining vector similarity and BM25 scores.

        Args:
            query: The search query
            chunks: List of chunk dictionaries with 'content' and 'similarity' fields
            top_k: Number of top results to return
            vector_weight: Weight for vector similarity score
            bm25_weight: Weight for BM25 score

        Returns:
            Re-ranked list of chunks with combined scores
        """
        if not chunks:
            return []

        # Get BM25 scores
        tokenized_corpus = [
            self.tokenizer(chunk['content'])
            for chunk in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self.tokenizer(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        normalized_bm25 = [s / max_bm25 for s in bm25_scores]

        # Calculate combined scores
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            vector_sim = chunk.get('similarity', 0)
            bm25_score = normalized_bm25[i]

            chunk_copy['bm25_score'] = float(bm25_scores[i])
            chunk_copy['combined_score'] = (
                vector_weight * vector_sim +
                bm25_weight * bm25_score
            )
            scored_chunks.append(chunk_copy)

        # Sort by combined score (descending)
        scored_chunks.sort(key=lambda x: x['combined_score'], reverse=True)

        return scored_chunks[:top_k]
