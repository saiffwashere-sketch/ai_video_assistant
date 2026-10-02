import streamlit as st
import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_questions, extract_key_decisions
from core.vector_store import build_vector_store
from core.rag_engine import build_rag_chain, ask_question

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark navy background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Top header */
.vi-header {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.vi-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e6edf3;
    margin: 0;
    letter-spacing: -0.5px;
}
.vi-header p {
    color: #7d8590;
    font-size: 0.9rem;
    margin: 0.4rem 0 0 0;
}

/* Input card */
.vi-input-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Result title */
.vi-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #58a6ff;
    margin: 0 0 1.5rem 0;
    letter-spacing: -0.3px;
}

/* Section cards */
.vi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.vi-card-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #7d8590;
    margin-bottom: 0.6rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #21262d;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7d8590;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
}

/* Progress steps */
.vi-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    color: #7d8590;
    font-size: 0.85rem;
}
.vi-step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #30363d;
    flex-shrink: 0;
}
.vi-step-dot.active { background: #58a6ff; }
.vi-step-dot.done   { background: #3fb950; }

/* Q&A */
.vi-qa-q {
    background: #1c2128;
    border-left: 3px solid #58a6ff;
    padding: 0.75rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
    color: #e6edf3;
}
.vi-qa-a {
    background: #161b22;
    border: 1px solid #21262d;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-size: 0.9rem;
    color: #c9d1d9;
    margin-bottom: 1.25rem;
}

/* Transcript box */
.vi-transcript {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #8b949e;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

/* Buttons */
.stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.5rem 1.25rem;
    transition: background 0.15s;
}
.stButton > button:hover { background: #2ea043; }

/* Text input */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    font-size: 0.9rem !important;
}

/* Status badge */
.vi-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.vi-badge-blue { background: #1f3d5c; color: #58a6ff; }
.vi-badge-green { background: #1a3824; color: #3fb950; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vi-header">
    <h1>🎬 Video Intelligence</h1>
    <p>Transcribe, summarize, and interrogate any video or audio in seconds</p>
</div>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
for key in ["result", "qa_history", "processing"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "qa_history" else []

# ── Input section ──────────────────────────────────────────────────────────────
col_input, col_gap, col_steps = st.columns([3, 0.2, 1.2])

with col_input:
    st.markdown('<div class="vi-input-card">', unsafe_allow_html=True)
    source = st.text_input(
        "YouTube URL or local file path",
        placeholder="https://youtube.com/watch?v=... or C:/path/to/file.mp4",
        label_visibility="visible",
    )
    lang = st.selectbox("Language", ["english", "hinglish"], label_visibility="visible")
    run = st.button("▶  Analyse", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_steps:
    st.markdown("**Pipeline**")
    result = st.session_state.result
    steps = [
        ("Download & chunk audio", bool(result)),
        ("Transcribe",             bool(result)),
        ("Summarise",              bool(result)),
        ("Extract insights",       bool(result)),
        ("Build Q&A index",        bool(result)),
    ]
    for label, done in steps:
        dot_class = "done" if done else "active" if st.session_state.processing else ""
        st.markdown(f"""
        <div class="vi-step">
            <div class="vi-step-dot {dot_class}"></div>
            {label}
        </div>""", unsafe_allow_html=True)

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run and source:
    st.session_state.processing = True
    st.session_state.result = None
    st.session_state.qa_history = []

    with st.status("Running pipeline…", expanded=True) as status:
        try:
            st.write("⬇️  Downloading and chunking audio…")
            chunks = process_input(source)

            st.write(f"🎙️  Transcribing {len(chunks)} chunk(s) with Whisper…")
            transcript = transcribe_all(chunks, language=lang)

            st.write("📝  Summarising…")
            summary = summarize(transcript)
            title   = generate_title(transcript)

            st.write("🔍  Extracting action items, decisions, questions…")
            action_items = extract_action_items(transcript)
            questions    = extract_questions(transcript)
            decisions    = extract_key_decisions(transcript)

            st.write("🗂️  Building Q&A index…")
            rag_chain = build_rag_chain(transcript)

            st.session_state.result = {
                "transcript":    transcript,
                "summary":       summary,
                "title":         title,
                "action_items":  action_items,
                "key_decisions": decisions,
                "open_questions":questions,
                "rag_chain":     rag_chain,
            }
            status.update(label="✅  Done!", state="complete")

        except Exception as e:
            status.update(label="❌  Error", state="error")
            st.error(str(e))

    st.session_state.processing = False

elif run and not source:
    st.warning("Please enter a YouTube URL or file path first.")

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.markdown(f'<div class="vi-title">{r["title"]}</div>', unsafe_allow_html=True)

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_qa, tab_transcript = st.tabs([
        "📋 Summary", "✅ Action Items", "🎯 Decisions", "❓ Questions", "💬 Ask Anything", "📄 Transcript"
    ])

    with tab_summary:
        st.markdown('<div class="vi-card"><div class="vi-card-label">Meeting Summary</div>', unsafe_allow_html=True)
        st.markdown(r["summary"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_actions:
        st.markdown('<div class="vi-card"><div class="vi-card-label">Action Items</div>', unsafe_allow_html=True)
        st.markdown(r["action_items"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_decisions:
        st.markdown('<div class="vi-card"><div class="vi-card-label">Key Decisions</div>', unsafe_allow_html=True)
        st.markdown(r["key_decisions"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_questions:
        st.markdown('<div class="vi-card"><div class="vi-card-label">Open Questions</div>', unsafe_allow_html=True)
        st.markdown(r["open_questions"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_qa:
        st.markdown('<div class="vi-card-label" style="color:#7d8590;font-size:0.7rem;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;">Ask anything about the video</div>', unsafe_allow_html=True)

        # Show history
        for q, a in st.session_state.qa_history:
            st.markdown(f'<div class="vi-qa-q">🙋 {q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vi-qa-a">🤖 {a}</div>', unsafe_allow_html=True)

        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input("Your question", placeholder="What was decided about the budget?", label_visibility="collapsed")
            submitted = st.form_submit_button("Ask →", use_container_width=True)

        if submitted and question:
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], question)
            st.session_state.qa_history.append((question, answer))
            st.rerun()

    with tab_transcript:
        st.markdown('<div class="vi-card-label" style="color:#7d8590;font-size:0.7rem;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;margin-bottom:0.5rem;">Full Transcript</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="vi-transcript">{r["transcript"]}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download transcript",
            data=r["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )

elif not st.session_state.processing:
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;color:#30363d;">
        <div style="font-size:3rem;margin-bottom:1rem;">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#7d8590;">
            Paste a YouTube link or file path above to get started
        </div>
    </div>
    """, unsafe_allow_html=True)