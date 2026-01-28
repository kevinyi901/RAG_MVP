"""Streamlit chat frontend for RAG system with pinning + citations."""

import streamlit as st
import requests
import json
import time
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
        st.header("📁 Documents")

        docs = fetch_documents()
        if docs:
            for d in docs:
                st.text(f"📄 {d['filename'][:20]}")
        else:
            st.info("No documents")

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pinned_message = None
            st.rerun()


# -------------------------------------------------------------------
# Citations (inline overlay)
# -------------------------------------------------------------------
def render_with_citations(text, sources):
    if not sources:
        st.markdown(text)
        return

    st.markdown(text)

    cols = st.columns(len(sources))
    for i, source in enumerate(sources):
        with cols[i]:
            with st.popover(f"[{i+1}]"):
                st.markdown(f"**{source['document']}**")
                st.caption(
                    f"Section: {source.get('section','N/A')} | "
                    f"Page: {source.get('page','?')} | "
                    f"Relevance: {source['relevance']}%"
                )
                st.markdown(source["excerpt"])


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
            render_with_citations(
                msg["content"], msg.get("sources", [])
            )

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
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("assistant"):
            status = st.empty()
            output = st.empty()

            thinking = ["Thinking.", "Thinking..", "Thinking..."]
            idx = 0

            full = ""
            sources = []

            history = st.session_state.messages[:-1]

            for event in chat_stream(
                prompt, history, st.session_state.selected_model
            ):
                # Thinking animation
                if not full:
                    status.markdown(thinking[idx % 3])
                    idx += 1
                    time.sleep(0.2)

                t = event.get("type")
                c = event.get("content")

                if t == "chunk":
                    status.empty()
                    full += c
                    output.markdown(full + "▌")

                elif t == "sources":
                    sources = c

                elif t == "done":
                    status.empty()
                    if isinstance(c, dict):
                        full = c.get("answer", full)
                    output.markdown(full)

                elif t == "error":
                    status.empty()
                    output.error(c)
                    break

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full,
                    "sources": sources,
                }
            )

            # Auto-scroll anchor
            st.markdown(
                "<div id='scroll'></div>"
                "<script>"
                "document.getElementById('scroll').scrollIntoView();"
                "</script>",
                unsafe_allow_html=True,
            )

            st.rerun()


if __name__ == "__main__":
    main()
