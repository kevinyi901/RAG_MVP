"""Streamlit chat frontend for RAG system with pinning + citations."""
import streamlit as st
import requests
import json
import random
from typing import Optional, List

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
API_URL = "http://api:8000"
AVAILABLE_MODELS = ["mistral:7b", "gpt-oss:20b"]

# -------------------------------------------------------------------
# Session State
# -------------------------------------------------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0]
    if "pinned_message" not in st.session_state:
        st.session_state.pinned_message = None
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False
    if "uploaded_doc_ids" not in st.session_state:
        st.session_state.uploaded_doc_ids = set()

# -------------------------------------------------------------------
# API helpers
# -------------------------------------------------------------------
def fetch_documents():
    try:
        r = requests.get(f"{API_URL}/api/documents", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def upload_document(file, use_ocr=True, extract_tables=True, use_semantic=False):
    """Upload a document to the API with options."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        params = {
            "use_ocr": use_ocr,
            "extract_tables": extract_tables,
            "use_semantic_chunking": use_semantic
        }
        r = requests.post(
            f"{API_URL}/api/documents/upload",
            files=files,
            params=params,
            timeout=300  # Longer timeout for semantic chunking
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def delete_document(doc_id):
    """Delete a document from the API."""
    try:
        r = requests.delete(f"{API_URL}/api/documents/{doc_id}", timeout=10)
        r.raise_for_status()
        return True
    except Exception:
        return False

def chat_stream(message, history, model):
    try:
        payload = {
            "message": message,
            "conversation_history": [
                {"role": m["role"], "content": m["content"]} for m in history
            ],
            "model": model,
        }
        r = requests.post(
            f"{API_URL}/api/chat/stream",
            json=payload,
            stream=True,
            timeout=300,
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    yield json.loads(line[6:])
    except Exception as e:
        yield {"type": "error", "content": str(e)}

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        st.selectbox("Model", AVAILABLE_MODELS, key="selected_model")

        st.divider()

        # Document upload section
        st.header("📁 Documents")

        with st.expander("➕ Add Document", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload",
                type=["pdf", "txt", "md", "docx"],
                label_visibility="collapsed",
                key="doc_uploader"
            )
            if uploaded_file:
                st.caption("Options:")
                use_ocr = st.checkbox("OCR (scanned docs)", value=True, key="opt_ocr")
                extract_tables = st.checkbox("Extract tables", value=True, key="opt_tables")
                use_semantic = st.checkbox("AI chunking", value=False, key="opt_semantic",
                                          help="Slower but better boundaries")

                if st.button("Upload", key="upload_confirm"):
                    with st.spinner("Processing..." + (" (AI chunking)" if use_semantic else "")):
                        result = upload_document(
                            uploaded_file,
                            use_ocr=use_ocr,
                            extract_tables=extract_tables,
                            use_semantic=use_semantic
                        )
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            if "document_id" in result:
                                st.session_state.uploaded_doc_ids.add(result["document_id"])
                            st.success(f"Added {result['chunk_count']} chunks")
                            st.rerun()

        # Document list - only show docs uploaded via this UI
        docs = fetch_documents()
        ui_docs = [d for d in docs if d['id'] in st.session_state.uploaded_doc_ids]
        if ui_docs:
            for d in ui_docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    name = d['filename'][:18] + "..." if len(d['filename']) > 18 else d['filename']
                    st.caption(f"📄 {name}")
                with col2:
                    if st.button("✕", key=f"del_{d['id']}", help="Remove"):
                        if delete_document(d['id']):
                            st.session_state.uploaded_doc_ids.discard(d['id'])
                            st.rerun()
        else:
            st.caption("No uploaded documents")

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pinned_message = None
            st.rerun()

# -------------------------------------------------------------------
# Thinking animation phrases
# -------------------------------------------------------------------
THINKING_PHRASES = [
    "Thinking",
    "Reasoning",
    "Connecting ideas",
    "Reviewing context",
    "Weighing sources",
    "Synthesizing information",
    "Formulating response",
    "Checking consistency",
    "Generating answer",
]

def thinking_text():
    dots = ["",  "..", "..."]
    return f"*{random.choice(THINKING_PHRASES)}{random.choice(dots)}*"

# -------------------------------------------------------------------
# Citations overlay
# -------------------------------------------------------------------
def render_with_citations(text, sources):
    if not sources:
        st.markdown(text)
        return

    st.markdown(text)

    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for i, source in enumerate(sources):
                st.markdown(f"**[{i+1}] {source['document']}**")
                st.caption(
                    f"Section: {source.get('section','N/A')} | "
                    f"Page: {source.get('page','?')} | "
                    f"Relevance: {source['relevance']}%"
                )
                st.markdown(source["excerpt"])
                if i < len(sources) - 1:
                    st.divider()

# -------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------
def main():
    st.set_page_config("JFN AI Co-Pilot", layout="wide")
    init_session_state()
    render_sidebar()

    st.title("Joint Fires Network AI Co-Pilot")
    st.caption("Intelligence-driven decision support")

    # ---------------------------------------------------------------
    # Pinned message
    # ---------------------------------------------------------------
    if st.session_state.pinned_message is not None:
        msg = st.session_state.messages[st.session_state.pinned_message]
        with st.container(border=True):
            st.markdown("📌 **Pinned Answer**")
            render_with_citations(msg["content"], msg.get("sources", []))

    # ---------------------------------------------------------------
    # Chat history
    # ---------------------------------------------------------------
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_with_citations(msg["content"], msg.get("sources", []))

                col1, col2, _ = st.columns([1, 1, 8])
                with col1:
                    if st.button("📌", key=f"pin_{i}"):
                        st.session_state.pinned_message = i
                        st.toast("Pinned")
                        st.rerun()
                with col2:
                    st.caption("")
            else:
                st.markdown(msg["content"])

    # ---------------------------------------------------------------
    # Chat input (auto-pinned bottom)
    # ---------------------------------------------------------------
    prompt = st.chat_input("Enter your query...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.stop_generation = False

        with st.chat_message("assistant"):
            # Create layout with stop button
            status_col, stop_col = st.columns([6, 1])
            with status_col:
                status = st.empty()
            with stop_col:
                stop_btn = st.empty()

            output = st.empty()

            full = ""
            sources = []
            history = st.session_state.messages[:-1]
            first_token = False
            stopped = False

            # Show stop button during generation
            if stop_btn.button("⏹", help="Stop generation", key="stop_gen"):
                st.session_state.stop_generation = True

            # Show initial thinking state
            status.markdown(f"*{random.choice(THINKING_PHRASES)}...*")

            # ---------------------------------------------------
            # Stream the response
            # ---------------------------------------------------
            for event in chat_stream(prompt, history, st.session_state.selected_model):
                # Check if stop was requested
                if st.session_state.stop_generation:
                    stopped = True
                    break

                event_type = event.get("type")
                content = event.get("content")

                if event_type == "status":
                    status.markdown(f"*{content}*")

                elif event_type == "chunk":
                    if not first_token:
                        first_token = True
                        status.empty()
                    full += content
                    output.markdown(full + "▌")

                elif event_type == "sources":
                    sources = content

                elif event_type == "done":
                    status.empty()
                    if isinstance(content, dict):
                        full = content.get("answer", full)
                    output.markdown(full)

                elif event_type == "error":
                    status.empty()
                    output.error(content)
                    break

            # Clear stop button and status
            stop_btn.empty()
            status.empty()

            if stopped:
                full += "\n\n*[Generation stopped]*"
                output.markdown(full)

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full,
                "sources": sources,
            })

            st.session_state.stop_generation = False
            st.rerun()


if __name__ == "__main__":
    main()
