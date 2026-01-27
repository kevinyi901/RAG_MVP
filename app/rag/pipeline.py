"""RAG Pipeline orchestration."""

import httpx
import os
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from .embeddings import EmbeddingService
from .retriever import Retriever
from .ranker import BM25Ranker


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    chain_of_thought: str
    query: str


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    sources: Optional[List[Dict[str, Any]]] = None


class RAGPipeline:
    """Orchestrate the RAG process: embed, retrieve, rerank, generate."""

    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        retriever: Retriever = None,
        ranker: BM25Ranker = None,
        ollama_host: str = None,
        llm_model: str = None,
        top_k_retrieval: int = None,
        top_k_rerank: int = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.retriever = retriever or Retriever()
        self.ranker = ranker or BM25Ranker()

        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-oss:20b")
        self.top_k_retrieval = top_k_retrieval or int(os.getenv("TOP_K_RETRIEVAL", "20"))
        self.top_k_rerank = top_k_rerank or int(os.getenv("TOP_K_RERANK", "5"))

    def query(
        self,
        question: str,
        document_ids: Optional[List[int]] = None,
        stream: bool = False
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Args:
            question: User's question
            document_ids: Optional filter by specific documents
            stream: Whether to stream the response

        Returns:
            RAGResponse with answer, sources, and chain of thought
        """
        # Step 1: Embed the query
        query_embedding = self.embedding_service.embed_text(question)

        # Step 2: Retrieve similar chunks
        retrieved_chunks = self.retriever.search(
            query_embedding,
            top_k=self.top_k_retrieval,
            document_ids=document_ids
        )

        # Step 3: Rerank with BM25
        reranked_chunks = self.ranker.rerank(
            question,
            retrieved_chunks,
            top_k=self.top_k_rerank
        )

        # Step 4: Build context and generate response
        context = self._build_context(reranked_chunks)
        prompt = self._build_prompt(question, context)

        if stream:
            return self._generate_streaming(question, prompt, reranked_chunks)
        else:
            answer, chain_of_thought = self._generate(prompt)

            return RAGResponse(
                answer=answer,
                sources=self._format_sources(reranked_chunks),
                chain_of_thought=chain_of_thought,
                query=question
            )

    def query_streaming(
        self,
        question: str,
        document_ids: Optional[List[int]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Execute RAG pipeline with streaming response.

        Yields dictionaries with 'type' and 'content' keys:
        - type: 'status' - progress updates
        - type: 'chunk' - response text chunks
        - type: 'sources' - source documents
        - type: 'done' - completion signal
        """
        # Step 1: Embed the query
        yield {"type": "status", "content": "Embedding query..."}
        query_embedding = self.embedding_service.embed_text(question)

        # Step 2: Retrieve similar chunks
        total_chunks = self.retriever.get_total_chunks()
        yield {"type": "status", "content": f"Searching {total_chunks} chunks..."}
        retrieved_chunks = self.retriever.search(
            query_embedding,
            top_k=self.top_k_retrieval,
            document_ids=document_ids
        )
        yield {"type": "status", "content": f"Found {len(retrieved_chunks)} relevant chunks"}

        # Step 3: Rerank with BM25
        yield {"type": "status", "content": "Re-ranking with BM25..."}
        reranked_chunks = self.ranker.rerank(
            question,
            retrieved_chunks,
            top_k=self.top_k_rerank
        )
        yield {"type": "status", "content": f"Selected top {len(reranked_chunks)} chunks"}

        # Send sources
        yield {"type": "sources", "content": self._format_sources(reranked_chunks)}

        # Step 4: Generate response with streaming
        yield {"type": "status", "content": "Generating response..."}
        context = self._build_context(reranked_chunks)
        prompt = self._build_prompt(question, context)

        full_response = ""
        chain_of_thought = ""
        in_thinking = False

        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            token = data["response"]
                            full_response += token

                            # Track chain of thought
                            if "<thinking>" in full_response and not in_thinking:
                                in_thinking = True
                            if in_thinking:
                                chain_of_thought += token
                            if "</thinking>" in full_response:
                                in_thinking = False

                            yield {"type": "chunk", "content": token}

                        if data.get("done", False):
                            break

        # Clean up the response
        answer = self._extract_answer(full_response)
        chain_of_thought = self._extract_chain_of_thought(full_response)

        yield {
            "type": "done",
            "content": {
                "answer": answer,
                "chain_of_thought": chain_of_thought
            }
        }

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"[Source {i}: {chunk['filename']}"
            if chunk.get('section_title'):
                source_info += f", Section: {chunk['section_title']}"
            if chunk.get('page_number'):
                source_info += f", Page: {chunk['page_number']}"
            source_info += "]"

            context_parts.append(f"{source_info}\n{chunk['content']}")

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM."""
        return f"""You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question. If you cannot find the answer in the context, say so clearly.

When answering:
1. First, wrap your reasoning in <thinking> tags to show your chain of thought
2. Then provide a clear, concise answer based on the context
3. Reference specific sources when possible

Context:
{context}

Question: {question}

Remember to:
- Show your thinking process in <thinking> tags
- Provide a direct answer after your thinking
- Cite sources when referencing specific information"""

    def _generate(self, prompt: str) -> tuple[str, str]:
        """Generate response using Ollama."""
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            full_response = response.json()["response"]

        answer = self._extract_answer(full_response)
        chain_of_thought = self._extract_chain_of_thought(full_response)

        return answer, chain_of_thought

    def _extract_answer(self, response: str) -> str:
        """Extract the answer portion from the response."""
        # Remove thinking tags and content
        import re
        answer = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        return answer.strip()

    def _extract_chain_of_thought(self, response: str) -> str:
        """Extract chain of thought from thinking tags."""
        import re
        match = re.search(r'<thinking>(.*?)</thinking>', response, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source chunks for display."""
        sources = []
        for chunk in chunks:
            sources.append({
                "document": chunk['filename'],
                "section": chunk.get('section_title', 'N/A'),
                "page": chunk.get('page_number'),
                "relevance": round(chunk.get('similarity', 0) * 100, 1),
                "bm25_score": round(chunk.get('bm25_score', 0), 2),
                "excerpt": chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'],
                "chunk_id": chunk['id']
            })
        return sources

    def chat_streaming(
        self,
        message: str,
        conversation_history: List[Dict[str, str]] = None,
        document_ids: Optional[List[int]] = None,
        max_history_tokens: int = 2000
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Chat with conversation memory and RAG.

        Args:
            message: Current user message
            conversation_history: List of {"role": "user"|"assistant", "content": "..."}
            document_ids: Optional filter by specific documents
            max_history_tokens: Max tokens to use for conversation history

        Yields:
            Same format as query_streaming
        """
        conversation_history = conversation_history or []

        # Step 1: Embed the current message for retrieval
        yield {"type": "status", "content": "Embedding query..."}
        query_embedding = self.embedding_service.embed_text(message)

        # Step 2: Retrieve similar chunks
        total_chunks = self.retriever.get_total_chunks()
        yield {"type": "status", "content": f"Searching {total_chunks} chunks..."}
        retrieved_chunks = self.retriever.search(
            query_embedding,
            top_k=self.top_k_retrieval,
            document_ids=document_ids
        )
        yield {"type": "status", "content": f"Found {len(retrieved_chunks)} relevant chunks"}

        # Step 3: Rerank with BM25
        yield {"type": "status", "content": "Re-ranking with BM25..."}
        reranked_chunks = self.ranker.rerank(
            message,
            retrieved_chunks,
            top_k=self.top_k_rerank
        )
        yield {"type": "status", "content": f"Selected top {len(reranked_chunks)} chunks"}

        # Send sources
        yield {"type": "sources", "content": self._format_sources(reranked_chunks)}

        # Step 4: Build chat prompt with history and context
        yield {"type": "status", "content": "Generating response..."}
        context = self._build_context(reranked_chunks)
        prompt = self._build_chat_prompt(message, context, conversation_history, max_history_tokens)

        full_response = ""
        chain_of_thought = ""
        in_thinking = False

        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            token = data["response"]
                            full_response += token

                            # Track chain of thought
                            if "<thinking>" in full_response and not in_thinking:
                                in_thinking = True
                            if in_thinking:
                                chain_of_thought += token
                            if "</thinking>" in full_response:
                                in_thinking = False

                            yield {"type": "chunk", "content": token}

                        if data.get("done", False):
                            break

        # Clean up the response
        answer = self._extract_answer(full_response)
        chain_of_thought = self._extract_chain_of_thought(full_response)

        yield {
            "type": "done",
            "content": {
                "answer": answer,
                "chain_of_thought": chain_of_thought
            }
        }

    def _build_chat_prompt(
        self,
        message: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        max_history_tokens: int = 2000
    ) -> str:
        """Build prompt with conversation history and RAG context."""
        # Truncate history if needed (simple approach - keep recent messages)
        truncated_history = self._truncate_history(conversation_history, max_history_tokens)

        # Format conversation history
        history_text = ""
        if truncated_history:
            history_parts = []
            for msg in truncated_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history_text = "\n\n".join(history_parts)

        prompt = f"""You are a helpful assistant that answers questions based on the provided context and conversation history.
Use the following pieces of context to answer the question. If you cannot find the answer in the context, say so clearly.

When answering:
1. First, wrap your reasoning in <thinking> tags to show your chain of thought
2. Then provide a clear, concise answer based on the context
3. Reference specific sources when possible
4. Consider the conversation history for context about what the user is asking

Context from documents:
{context}

"""
        if history_text:
            prompt += f"""Previous conversation:
{history_text}

"""
        prompt += f"""Current question: {message}

Remember to:
- Show your thinking process in <thinking> tags
- Provide a direct answer after your thinking
- Cite sources when referencing specific information
- Use conversation history to understand follow-up questions"""

        return prompt

    def _truncate_history(
        self,
        history: List[Dict[str, str]],
        max_tokens: int
    ) -> List[Dict[str, str]]:
        """Truncate conversation history to fit within token limit."""
        if not history:
            return []

        # Simple estimation: ~4 chars per token
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token

        total_chars = 0
        truncated = []

        # Keep most recent messages first
        for msg in reversed(history):
            msg_chars = len(msg.get("content", ""))
            if total_chars + msg_chars > max_chars:
                break
            truncated.insert(0, msg)
            total_chars += msg_chars

        return truncated

    def close(self):
        """Close all connections."""
        self.embedding_service.close()
        self.retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
