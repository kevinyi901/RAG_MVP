"""Text chunking with token-based sizing and optional AI refinement."""

import os
import re
import json
import httpx
from typing import List, Dict, Any, Optional, Callable
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


class SemanticChunker:
    """Hybrid chunker: token-split first, then LLM refines boundaries."""

    def __init__(
        self,
        ollama_host: str = None,
        model: str = None,
        chunk_size: int = 512,
        encoding_name: str = "cl100k_base"
    ):
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("LLM_MODEL", "mistral:7b")
        self.chunk_size = chunk_size
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_with_ai(
        self,
        text: str,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Hybrid chunking: initial token split, then AI identifies better boundaries.
        """
        if not text.strip():
            return []

        # Step 1: Initial rough split into ~1500 token windows for AI processing
        windows = self._create_windows(text, window_size=1500, overlap=200)

        all_chunks = []
        chunk_index = 0

        for window in windows:
            # Step 2: Ask LLM to identify semantic boundaries
            boundaries = self._get_semantic_boundaries(window)

            # Step 3: Split at boundaries
            segments = self._split_at_boundaries(window, boundaries)

            for segment in segments:
                if segment.strip():
                    token_count = self.count_tokens(segment)
                    all_chunks.append(Chunk(
                        content=segment.strip(),
                        chunk_index=chunk_index,
                        page_number=page_number,
                        section_title=section_title,
                        token_count=token_count,
                        metadata=metadata or {}
                    ))
                    chunk_index += 1

        return all_chunks

    def _create_windows(self, text: str, window_size: int, overlap: int) -> List[str]:
        """Split text into overlapping windows for AI processing."""
        words = text.split()
        windows = []
        start = 0

        while start < len(words):
            window_words = []
            token_count = 0
            i = start

            while i < len(words) and token_count < window_size:
                window_words.append(words[i])
                token_count += self.count_tokens(words[i] + " ")
                i += 1

            windows.append(" ".join(window_words))

            # Move start forward, accounting for overlap
            overlap_tokens = 0
            while start < i and overlap_tokens < overlap:
                overlap_tokens += self.count_tokens(words[start] + " ")
                start += 1

            if start >= len(words):
                break

        return windows

    def _get_semantic_boundaries(self, text: str) -> List[int]:
        """Use LLM to identify natural semantic break points."""
        prompt = f"""Analyze this text and identify the character positions where natural semantic breaks occur.
A semantic break is where one topic/concept ends and another begins.

Return ONLY a JSON array of character positions (integers), nothing else.
Example: [150, 423, 891]

If there are no clear breaks, return an empty array: []

Text:
{text[:3000]}"""  # Limit to prevent token overflow

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "[]")

                # Extract JSON array from response
                match = re.search(r'\[[\d\s,]*\]', result)
                if match:
                    return json.loads(match.group())
        except Exception:
            pass

        return []

    def _split_at_boundaries(self, text: str, boundaries: List[int]) -> List[str]:
        """Split text at the given character positions."""
        if not boundaries:
            # No boundaries found, use simple sentence splitting
            return self._simple_split(text)

        segments = []
        prev = 0
        for pos in sorted(boundaries):
            if 0 < pos < len(text):
                segments.append(text[prev:pos])
                prev = pos
        segments.append(text[prev:])

        # Merge small segments
        merged = []
        current = ""
        for seg in segments:
            if self.count_tokens(current + seg) < self.chunk_size:
                current += seg
            else:
                if current:
                    merged.append(current)
                current = seg
        if current:
            merged.append(current)

        return merged

    def _simple_split(self, text: str) -> List[str]:
        """Fallback: split by paragraphs or sentences."""
        paragraphs = re.split(r'\n\s*\n', text)
        result = []
        current = ""

        for para in paragraphs:
            if self.count_tokens(current + para) < self.chunk_size:
                current += "\n\n" + para if current else para
            else:
                if current:
                    result.append(current)
                current = para

        if current:
            result.append(current)

        return result


class TextChunker:
    """Split text into chunks with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "50"))
        self.encoding = tiktoken.get_encoding(encoding_name)

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
