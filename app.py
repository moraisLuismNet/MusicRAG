import json
import os
import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from musicrag import __version__
from musicrag.config import EMBED_MODEL, KNOWLEDGE_DIR, LLM_MODEL, TOP_K, VALID_MODES
from musicrag.db import db_stats, list_documents, reset_db
from musicrag.embed_backend import EMBED_BACKEND
from musicrag.ingest import ingest_dir, knowledge_files
from musicrag.ollama_client import list_chat_models

DOC_TEMPLATE = """---
title: Sample artist
type: artist          # artist | album | song | genre | general
genre: rock
active_years: 1990-2010
---
Document body in Markdown or plain text. It is automatically split into
chunks around sentences, embedded and stored in SQLite.

The frontmatter metadata is kept in the documents table and shown next to
the results. Without frontmatter, the title is the file name and the
default type is 'general'.
"""

st.set_page_config(page_title="MusicRAG", layout="wide")


# ------- Auto-close: exit Streamlit when the last tab is closed -------
    # When the http://localhost:8501/ window is closed, Streamlit keeps
    # running; here a watchdog watches the active sessions and, if all of them
    # disconnect for `_EXIT_GRACE` seconds, terminates the process with
    # os._exit(0). That makes `streamlit run` (launched by start.sh) return and
    # start.sh shuts down the Ollama it started itself.
_EXIT_GRACE = int(os.environ.get("MUSICRAG_EXIT_GRACE", "30"))


def _active_session_count() -> int | None:
    """Number of windows/tabs connected to Streamlit (None if no Runtime)."""
    try:
        from streamlit.runtime.runtime import Runtime

        return int(Runtime.instance()._session_mgr.num_active_sessions())
    except Exception:
        return None


@st.cache_resource
def start_close_watchdog() -> None:
    def _loop() -> None:
        seen, idle = False, 0.0
        while True:
            time.sleep(2)
            n = _active_session_count()
            if n is None:
                continue
            if n > 0:
                seen = True
                idle = 0.0
            elif seen:
                idle += 2.0
                if idle >= _EXIT_GRACE:
                    os._exit(0)  # ends `streamlit run` -> start.sh cleans up
            else:
                idle = 0.0

    threading.Thread(target=_loop, daemon=True).start()


start_close_watchdog()


def has_ollama() -> bool:
    try:
        return len(list_chat_models()) > 0
    except Exception:
        return False


def get_side_model_options() -> list[str]:
    models = list_chat_models()
    if not models:
        return [LLM_MODEL]
    if LLM_MODEL in models:
        models.remove(LLM_MODEL)
    return [LLM_MODEL] + models


def parse_meta(row: dict) -> dict:
    meta = row.get("meta")
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return meta or {}


def render_sources(results) -> None:
    with st.expander("Recovered sources"):
        st.caption("Context injected into the response:")
        for i, (score, row) in enumerate(results, 1):
            meta = parse_meta(row)
            year = f" | {meta.get('year')}" if meta.get("year") else ""
            st.markdown(
                f"**{i}. {row['title']}** · `{row['doc_type']}`{year} · sim {score:.3f}"
            )
            st.caption(row.get("source", ""))
            st.markdown(f"> {row['content'][:280]}...")


def relative_label(source: str) -> str:
    try:
        return str(Path(source).relative_to(KNOWLEDGE_DIR))
    except ValueError:
        return source


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("MusicRAG")
    st.caption(f"v{__version__} · 100% local RAG (SQLite + Ollama)")
    st.caption(f"Backend embeddings: **{EMBED_BACKEND}**")

    if not has_ollama():
        st.warning(
            "Ollama is not available or has no chat models. "
            "Searches will use the local backend; for actual embeddings "
            "executes `ollama pull nomic-embed-text`."
        )

    st.divider()
    st.subheader("Configuration")
    chat_model = st.selectbox("Chat model (Ollama)", options=get_side_model_options())
    embed_model = st.text_input(
        "Embedding model",
        value=EMBED_MODEL,
        help="It must appear in `ollama list` for actual semantic search.",
    )
    top_k = st.slider("Retrieved results (top-k)", 1, 20, TOP_K)
    mode = st.radio(
        "Search mode (Chat)",
        list(VALID_MODES),
        index=0,
        captions=["semantics + text", "semantics only", "text only"],
    )

    st.divider()
    stats = db_stats()
    c = st.columns(3)
    c[0].metric("Documents", stats["documents"])
    c[1].metric("Chunks", stats["chunks"])
    c[2].metric("Knowledge Files/", len(knowledge_files()))

tabs = st.tabs(["Chat"])

# ---------------------------------------------------------------- Chat
with tabs[0]:
    st.subheader("Chat with the music knowledge base")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    def generate_answer(question: str) -> None:
        from musicrag.player import handle_play_request
        from musicrag.rag import stream_answer

        def _speak(text: str) -> None:
            if not tts_enabled:
                return
            from musicrag.tts import clean_for_speech, available, speak

            if available():
                speak(clean_for_speech(text))

        play = handle_play_request(question)
        if play["matched"]:
            st.session_state.messages.append({"role": "assistant", "content": play["reply"]})
            with st.chat_message("assistant"):
                st.markdown(play["reply"])
            _speak(play["reply"])
            return

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating response..."):
                try:
                    results, stream = stream_answer(
                        question, top_k=top_k, mode=mode,
                        model=chat_model, embed_model=embed_model,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"The query could not be prepared.: {exc}")
                    results, stream = [], None
            if stream is not None:
                try:
                    answer = st.write_stream(stream)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"No response was generated.: {exc}")
                    answer = ""
            else:
                answer = ""
            if answer:
                st.session_state.messages.append({"role": "assistant", "content": answer})
                render_sources(results)
                _speak(answer)

    voice_col, tts_col, _ = st.columns([1, 1, 5])
    tts_enabled = tts_col.toggle(
        "🔊 Respond by voice",
        help="Read answers aloud (Windows TTS, offline).",
    )
    if tts_enabled:
        from musicrag.tts import find_voice

        if not find_voice():
            st.caption("No TTS voice available on this system.")

    pending_question = None
    if voice_col.button("🎤 Ask by voice", use_container_width=True):
        from musicrag.speech import available, record_and_recognize

        ok, reason = available()
        if not ok:
            st.warning(reason)
        else:
            with st.spinner("Listening… (say your question)"):
                question = record_and_recognize()
            if question:
                st.caption(f"I heard you: «{question}»")
                pending_question = question
            else:
                from musicrag.speech import last_stats
                rev = last_stats()
                det = (
                    f"Caught {rev['seconds']:.1f} s, signal {rev['rms']:.4f}"
                    if rev else "No data available"
                )
                st.info(
                    f"I didn't hear you. {det}.\n"
                    "Try again while moving closer to the microphone (is any audio "
                    "playing through the speakers?). If the issue persists: `python -m musicrag mic-test`."
                )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if pending_question:
        generate_answer(pending_question)

    prompt = st.chat_input("Ask about music, artists, albums, genres...")
    if prompt:
        generate_answer(prompt)