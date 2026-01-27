"""Streamlit chat frontend for RAG system."""

import streamlit as st
import requests
import json
from typing import Optional, List

# Configuration
API_URL = "http://api:8000"  # Use container name in Docker/Podman


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": "user"|"assistant", "content": "...", "sources": [...], "chain_of_thought": "..."}]
    if "documents" not in st.session_state:
        st.session_state.documents = []


def fetch_documents():
    """Fetch document list from API."""
    try:
        response = requests.get(f"{API_URL}/api/documents", timeout=10)
        response.raise_for_status()
        st.session_state.documents = response.json()
    except Exception as e:
        pass  # Silently fail on startup


def upload_document(file):
    """Upload a document to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(
            f"{API_URL}/api/documents/upload",
            files=files,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def delete_document(doc_id: int):
    """Delete a document."""
    try:
        response = requests.delete(f"{API_URL}/api/documents/{doc_id}", timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


def chat_stream(message: str, conversation_history: List[dict], document_ids: Optional[list] = None):
    """Stream chat response from API."""
    try:
        history = [{"role": m["role"], "content": m["content"]} for m in conversation_history]

        response = requests.post(
            f"{API_URL}/api/chat/stream",
            json={
                "message": message,
                "conversation_history": history,
                "document_ids": document_ids
            },
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    yield data
    except Exception as e:
        yield {"type": "error", "content": str(e)}


def submit_feedback(query_text: str, response_text: str, sources: list,
                    chain_of_thought: str, feedback: str):
    """Submit user feedback."""
    try:
        response = requests.post(
            f"{API_URL}/api/feedback",
            json={
                "query_text": query_text,
                "response_text": response_text,
                "sources": sources,
                "chain_of_thought": chain_of_thought,
                "feedback": feedback
            },
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception:
        return False


def render_sidebar():
    """Render the sidebar with document management."""
    with st.sidebar:
        st.header("📁 Documents")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    result = upload_document(file)
                    if result:
                        st.success(f"✓ {file.name} ({result['chunk_count']} chunks)")
                        fetch_documents()

        st.divider()

        # Document list header with refresh
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Loaded")
        with col2:
            if st.button("🔄", help="Refresh"):
                fetch_documents()
                st.rerun()

        if st.session_state.documents:
            for doc in st.session_state.documents:
                col1, col2 = st.columns([4, 1])
                with col1:
                    name = doc['filename']
                    if len(name) > 20:
                        name = name[:17] + "..."
                    st.text(f"📄 {name}")
                    st.caption(f"   {doc['chunk_count']} chunks")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Delete"):
                        if delete_document(doc['id']):
                            fetch_documents()
                            st.rerun()
        else:
            st.info("No documents uploaded")

        st.divider()

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Stats
        st.divider()
        try:
            response = requests.get(f"{API_URL}/api/stats", timeout=5)
            if response.ok:
                stats = response.json()
                st.caption(f"📊 {stats['total_documents']} docs | {stats['total_chunks']} chunks")
        except Exception:
            st.caption("📊 Connecting...")


def render_sources(sources: list, key_prefix: str = ""):
    """Render sources in an expander."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**{i}. {source['document']}**")

            info_parts = []
            if source.get('section') and source['section'] != 'N/A':
                info_parts.append(f"Section: {source['section']}")
            if source.get('page'):
                info_parts.append(f"Page: {source['page']}")
            info_parts.append(f"Relevance: {source['relevance']}%")
            st.caption(" | ".join(info_parts))

            st.markdown(
                f"<div style='background-color: #f0f2f6; padding: 8px; "
                f"border-radius: 4px; font-size: 0.85em; margin-bottom: 10px;'>"
                f"{source['excerpt']}</div>",
                unsafe_allow_html=True
            )


def render_chain_of_thought(chain_of_thought: str, key_prefix: str = ""):
    """Render chain of thought."""
    if not chain_of_thought:
        return

    with st.expander("🧠 Reasoning", expanded=False):
        st.markdown(chain_of_thought)


def main():
    """Main chat application."""
    st.set_page_config(
        page_title="RAG Chat",
        page_icon="💬",
        layout="wide"
    )

    init_session_state()
    fetch_documents()

    # Sidebar
    render_sidebar()

    # Main chat area
    st.title("💬 RAG Chat")
    st.caption("Ask questions about your documents. I remember our conversation.")

    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources and reasoning for assistant messages
            if message["role"] == "assistant":
                if message.get("sources"):
                    render_sources(message["sources"], f"hist_{i}")
                if message.get("chain_of_thought"):
                    render_chain_of_thought(message["chain_of_thought"], f"hist_{i}")

                # Feedback buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
                with col1:
                    if st.button("👍", key=f"fb_{i}_up", help="Helpful"):
                        if i > 0:
                            user_msg = st.session_state.messages[i-1]["content"]
                            submit_feedback(
                                user_msg, message["content"],
                                message.get("sources", []),
                                message.get("chain_of_thought", ""),
                                "helpful"
                            )
                            st.toast("Thanks!")
                with col2:
                    if st.button("😐", key=f"fb_{i}_mid", help="Neutral"):
                        if i > 0:
                            user_msg = st.session_state.messages[i-1]["content"]
                            submit_feedback(
                                user_msg, message["content"],
                                message.get("sources", []),
                                message.get("chain_of_thought", ""),
                                "neutral"
                            )
                            st.toast("Thanks!")
                with col3:
                    if st.button("👎", key=f"fb_{i}_down", help="Not helpful"):
                        if i > 0:
                            user_msg = st.session_state.messages[i-1]["content"]
                            submit_feedback(
                                user_msg, message["content"],
                                message.get("sources", []),
                                message.get("chain_of_thought", ""),
                                "not_helpful"
                            )
                            st.toast("Thanks!")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()

            full_response = ""
            sources = []
            chain_of_thought = ""

            # Get history (exclude current message)
            history = st.session_state.messages[:-1]

            for event in chat_stream(prompt, history):
                event_type = event.get("type")
                content = event.get("content")

                if event_type == "status":
                    status_placeholder.caption(f"⏳ {content}")

                elif event_type == "sources":
                    sources = content

                elif event_type == "chunk":
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")

                elif event_type == "done":
                    if isinstance(content, dict):
                        full_response = content.get("answer", full_response)
                        chain_of_thought = content.get("chain_of_thought", "")

                    status_placeholder.empty()
                    response_placeholder.markdown(full_response)

                elif event_type == "error":
                    status_placeholder.empty()
                    st.error(f"Error: {content}")
                    full_response = f"Sorry, an error occurred: {content}"

            # Show sources and reasoning
            if sources:
                render_sources(sources, "current")
            if chain_of_thought:
                render_chain_of_thought(chain_of_thought, "current")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "chain_of_thought": chain_of_thought
            })

            st.rerun()


if __name__ == "__main__":
    main()
