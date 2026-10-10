
"""Streamlit chat UI for SAARTH with voice input and spoken responses."""

from __future__ import annotations

import json
import os

import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import speech_to_text

API = os.getenv("SAARTH_API_URL", "http://localhost:8000")

st.set_page_config(page_title="SAARTH", page_icon="🧭", layout="wide")
st.title("🧭 SAARTH")
st.caption("Your AI assistant — type or speak to interact")

with st.sidebar:
    st.header("Knowledge base")
    try:
        h = requests.get(f"{API}/health", timeout=5).json()
        st.success(
            f"API online · {h['llm']} · "
            f"{h['vector_backend']} · {h['chunks']} chunks"
        )
    except (requests.RequestException, ValueError, KeyError):
        st.error(f"API unreachable at {API}")

    files = st.file_uploader(
        "Add documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if files and st.button("Ingest"):
        try:
            r = requests.post(
                f"{API}/ingest",
                files=[
                    ("files", (f.name, f.getvalue()))
                    for f in files
                ],
                timeout=300,
            )
            r.raise_for_status()
            st.success(f"Added {r.json()['chunks']} chunks")
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

    top_k = st.slider("Passages to retrieve", 1, 10, 4)
    speak_answers = st.toggle("🔊 Read answers aloud", value=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Display conversation history.
for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

st.subheader("🎤 Voice input")
st.caption("Click the microphone, speak your question, and wait for transcription.")

spoken_question = speech_to_text(
    language="en",
    start_prompt="🎤 Start speaking",
    stop_prompt="⏹️ Stop recording",
    just_once=True,
    key="saarth_voice_input",
)

if spoken_question:
    st.session_state.pending_voice_question = spoken_question

if st.session_state.get("pending_voice_question"):
    st.info(f"Recognized: {st.session_state.pending_voice_question}")
    col1, col2 = st.columns(2)

    with col1:
        send_voice = st.button("Send voice question", type="primary")

    with col2:
        cancel_voice = st.button("Cancel")

    if cancel_voice:
        st.session_state.pop("pending_voice_question", None)
        st.rerun()

    if send_voice:
        question = st.session_state.pop("pending_voice_question")
        st.session_state.active_question = question
        st.rerun()

# Typed questions continue to work as before.
typed_question = st.chat_input("Or type your question here…")

question = st.session_state.pop("active_question", None)
if not question and typed_question:
    question = typed_question

if question:
    st.chat_message("user").markdown(question)
    st.session_state.history.append(("user", question))

    with st.chat_message("assistant"), st.spinner("SAARTH is thinking…"):
        try:
            r = requests.post(
                f"{API}/ask",
                json={"question": question, "top_k": top_k},
                timeout=180,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        answer = data["answer"]
        st.markdown(answer)

        # Browser-native text-to-speech; no separate TTS API key required.
        if speak_answers:
            safe_answer = json.dumps(answer)
            components.html(
                f"""
                <script>
                const answer = {safe_answer};
                if ("speechSynthesis" in window) {{
                    window.speechSynthesis.cancel();
                    const utterance = new SpeechSynthesisUtterance(answer);
                    utterance.lang = "en-IN";
                    utterance.rate = 1;
                    window.speechSynthesis.speak(utterance);
                }}
                </script>
                """,
                height=0,
            )

        c1, c2 = st.columns(2)
        c1.metric("Grounding", f"{data['grounding_score']:.0%}")
        c2.metric("Revisions", data["revisions"])

        if data["tool_results"]:
            with st.expander("Tool calls"):
                for t in data["tool_results"]:
                    st.code(
                        f"{t['tool']}({t['arguments']}) → {t['output']}"
                    )

        if data["citations"]:
            with st.expander("Sources"):
                for c in data["citations"]:
                    st.markdown(
                        f"**[{c['index']}] {c['source']}** · "
                        f"score {c['score']:.2f}\n\n"
                        f"> {c['snippet']}…"
                    )

        with st.expander("Agent trace"):
            for s in data["trace"]:
                st.write(f"**{s['agent']}** — {s['detail']}")

    st.session_state.history.append(("assistant", answer))
