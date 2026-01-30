"""Text chunking with token-based sizing."""

import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import tiktoken


@dataclass
class Chunk:
    """Represents a text chunk."""
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Split text into chunks with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "100"))
        self.encoding = tiktoken.get_encoding(encoding_name)

    @staticmethod
    def _is_junk(text: str) -> bool:
        """Return True if chunk looks like appendix junk (mostly numbers, dates, codes)."""
        stripped = text.strip()
        if not stripped:
            return True
        # Ratio of digit characters to total
        digit_chars = sum(1 for c in stripped if c.isdigit())
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        total = len(stripped)
        if total < 20:
            return True
        # If more than 60% digits, likely a table/appendix
        if digit_chars / total > 0.6:
            return True
        # If very few alphabetic chars relative to length
        if alpha_chars / total < 0.2:
            return True
        # Mostly short tokens separated by whitespace (e.g. "12 34 56 78 90 ...")
        tokens = stripped.split()
        if tokens:
            avg_token_len = sum(len(t) for t in tokens) / len(tokens)
            short_ratio = sum(1 for t in tokens if len(t) <= 3) / len(tokens)
            if avg_token_len < 3 and short_ratio > 0.8:
                return True
        return False

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def chunk_text(
        self,
        text: str,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into chunks based on token count.

        Args:
            text: Text to chunk
            page_number: Optional page number for all chunks
            section_title: Optional section title for all chunks
            metadata: Optional metadata dict for all chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        # Split into sentences for better chunking
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # First, save current chunk if not empty
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Chunk(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        page_number=page_number,
                        section_title=section_title,
                        token_count=current_tokens,
                        metadata=metadata or {}
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence into smaller parts
                sub_chunks = self._split_long_text(sentence)
                for sub_chunk in sub_chunks:
                    sub_tokens = self.count_tokens(sub_chunk)
                    chunks.append(Chunk(
                        content=sub_chunk,
                        chunk_index=chunk_index,
                        page_number=page_number,
                        section_title=section_title,
                        token_count=sub_tokens,
                        metadata=metadata or {}
                    ))
                    chunk_index += 1
                continue

            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Chunk(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        page_number=page_number,
                        section_title=section_title,
                        token_count=current_tokens,
                        metadata=metadata or {}
                    ))
                    chunk_index += 1

                    # Handle overlap
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk, self.chunk_overlap
                    )
                    current_chunk = overlap_sentences
                    current_tokens = self.count_tokens(" ".join(current_chunk))

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text,
                chunk_index=chunk_index,
                page_number=page_number,
                section_title=section_title,
                token_count=self.count_tokens(chunk_text),
                metadata=metadata or {}
            ))

        # Filter out junk chunks (appendix-like content with mostly numbers)
        chunks = [c for c in chunks if not self._is_junk(c.content)]

        return chunks

    def chunk_document(
        self,
        pages: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Chunk a document with page information.

        Args:
            pages: List of page dicts with 'page_number', 'content', 'section_title'

        Returns:
            List of Chunk objects with page information preserved
        """
        all_chunks = []
        global_index = 0

        for page in pages:
            page_chunks = self.chunk_text(
                text=page.get('content', ''),
                page_number=page.get('page_number'),
                section_title=page.get('section_title'),
                metadata={'original_page': page.get('page_number')}
            )

            # Update global index
            for chunk in page_chunks:
                chunk.chunk_index = global_index
                global_index += 1

            all_chunks.extend(page_chunks)

        return all_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting using regex
        # Handles common sentence endings while preserving abbreviations
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)

        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _split_long_text(self, text: str) -> List[str]:
        """Split a long text that exceeds chunk size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = self.count_tokens(word + " ")

            if current_tokens + word_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Get overlap words
                    overlap_words = self._get_overlap_words(
                        current_chunk, self.chunk_overlap
                    )
                    current_chunk = overlap_words
                    current_tokens = self.count_tokens(" ".join(current_chunk))

            current_chunk.append(word)
            current_tokens += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_tokens: int
    ) -> List[str]:
        """Get sentences from the end for overlap."""
        overlap = []
        current_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if current_tokens + sentence_tokens > target_tokens:
                break
            overlap.insert(0, sentence)
            current_tokens += sentence_tokens

        return overlap

    def _get_overlap_words(
        self,
        words: List[str],
        target_tokens: int
    ) -> List[str]:
        """Get words from the end for overlap."""
        overlap = []
        current_tokens = 0

        for word in reversed(words):
            word_tokens = self.count_tokens(word + " ")
            if current_tokens + word_tokens > target_tokens:
                break
            overlap.insert(0, word)
            current_tokens += word_tokens

        return overlap
