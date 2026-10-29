"""
app.py
======
NotesAI — AI-Powered Notes Summarizer and Question Generator
Main Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Page config must be FIRST Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="NotesAI — Smart Study Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "NotesAI — AI-Powered Notes Summarizer\nPowered by HuggingFace Transformers",
    },
)

# ── Imports after page config ─────────────────────────────────────────────
from document_processor import (
    DocumentError,
    load_document,
    word_count,
)
from export_service import (
    export_questions_pdf,
    export_questions_txt,
    export_summary_pdf,
    export_summary_txt,
)
from question_generator import calculate_score, generate_questions
from summarizer import generate_summary
from validators import (
    AnyQuestion,
    LongAnswerQuestion,
    MCQQuestion,
    ShortAnswerQuestion,
    TrueFalseQuestion,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM CSS
# ============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] .stRadio label {
    color: #c8c8e8 !important;
    font-size: 0.95rem;
    padding: 4px 0;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #c8c8e8 !important;
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.notes-card {
    background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
    border: 1px solid #3a3a6a;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(108,99,255,0.08);
    transition: border-color 0.2s;
}
.notes-card:hover { border-color: #6C63FF; }

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 8px;
}
.hero-sub {
    font-size: 1.1rem;
    color: #9999bb;
    margin-bottom: 24px;
}

/* ── Feature cards ────────────────────────────────────────────────────── */
.feature-card {
    background: linear-gradient(135deg, #1a1a35 0%, #20203a 100%);
    border: 1px solid #333366;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    height: 100%;
}
.feature-icon { font-size: 2.2rem; margin-bottom: 10px; }
.feature-title {
    font-size: 1.0rem;
    font-weight: 600;
    color: #c8c8f0;
    margin-bottom: 6px;
}
.feature-desc { font-size: 0.85rem; color: #8888aa; line-height: 1.4; }

/* ── Stat chips ───────────────────────────────────────────────────────── */
.stat-chip {
    display: inline-block;
    background: linear-gradient(135deg, #6C63FF22, #6C63FF44);
    border: 1px solid #6C63FF66;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.85rem;
    color: #b0a8ff;
    margin: 4px;
}

/* ── Step badge ───────────────────────────────────────────────────────── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #6C63FF, #a78bfa);
    border-radius: 50%;
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 12px;
}

/* ── Section title ────────────────────────────────────────────────────── */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #d0d0f0;
    border-left: 4px solid #6C63FF;
    padding-left: 14px;
    margin: 24px 0 16px 0;
}

/* ── Question card ────────────────────────────────────────────────────── */
.q-card {
    background: #1c1c36;
    border: 1px solid #2e2e56;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
}
.q-type-badge {
    display: inline-block;
    background: #6C63FF33;
    border: 1px solid #6C63FF88;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #a090ff;
    margin-bottom: 8px;
}
.correct-ans { color: #4ade80; font-weight: 600; }
.wrong-ans   { color: #f87171; font-weight: 600; }
.evidence-box {
    background: #111128;
    border-left: 3px solid #6C63FF;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #9999bb;
    margin-top: 8px;
    font-style: italic;
}

/* ── Score card ───────────────────────────────────────────────────────── */
.score-card {
    background: linear-gradient(135deg, #1a2a1a, #1e351e);
    border: 1px solid #2e6b2e;
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
}
.score-pct {
    font-size: 3.5rem;
    font-weight: 700;
    color: #4ade80;
}

/* ── Hardware badge ───────────────────────────────────────────────────── */
.hw-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a1a35;
    border: 1px solid #3a3a6a;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.82rem;
    color: #9090c0;
    margin: 3px;
}

/* ── Warning box ──────────────────────────────────────────────────────── */
.warn-box {
    background: #2a1f0a;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 10px 16px;
    color: #fbbf24;
    font-size: 0.88rem;
    margin: 8px 0;
}

/* ── Status indicators ────────────────────────────────────────────────── */
.status-ok   { color: #4ade80; }
.status-warn { color: #fbbf24; }
.status-err  { color: #f87171; }

/* ── Streamlit overrides ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #5b53ee);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}
.stTextArea textarea { background: #16162e; color: #e0e0f0; }
.stSelectbox select { background: #16162e; }

div[data-testid="stExpander"] {
    background: #1a1a30;
    border: 1px solid #2e2e55;
    border-radius: 10px;
}

.stProgress > div > div { background: linear-gradient(90deg, #6C63FF, #a78bfa); }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALISATION
# ============================================================================

def init_session_state():
    defaults = {
        "page": "Dashboard",
        "doc_text": None,
        "doc_title": "",
        "doc_hash": None,
        "doc_source": None,
        "doc_wc": 0,
        "summary": None,
        "summary_hash": None,
        "questions": None,
        "questions_hash": None,
        "questions_type": None,
        "student_answers": {},
        "practice_submitted": False,
        "practice_score": None,
        "ai_loaded": False,
        "hw_status": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ============================================================================
# AI SERVICE (loaded lazily on first use)
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_ai_service():
    """Singleton AI service — loaded once per Streamlit session."""
    from ai_service import AIService
    return AIService.get_instance()


# ============================================================================
# HELPERS
# ============================================================================

def invalidate_dependent_results():
    """Clear summary and questions when source document changes."""
    st.session_state.summary = None
    st.session_state.summary_hash = None
    st.session_state.questions = None
    st.session_state.questions_hash = None
    st.session_state.student_answers = {}
    st.session_state.practice_submitted = False
    st.session_state.practice_score = None


def set_document(text: str, title: str, source: str, doc_hash: str):
    """Store a newly loaded document and invalidate stale results."""
    if st.session_state.doc_hash != doc_hash:
        invalidate_dependent_results()
    st.session_state.doc_text = text
    st.session_state.doc_title = title or "Untitled Notes"
    st.session_state.doc_hash = doc_hash
    st.session_state.doc_source = source
    st.session_state.doc_wc = word_count(text)


def has_document() -> bool:
    return bool(st.session_state.doc_text)


def has_summary() -> bool:
    return st.session_state.summary is not None


def has_questions() -> bool:
    return bool(st.session_state.questions)


def render_hw_status():
    """Render hardware/model status badges in the sidebar."""
    try:
        svc = get_ai_service()
        hw = svc.hardware_status()
        st.session_state.hw_status = hw
    except Exception:
        return

    hw = st.session_state.hw_status
    if not hw:
        return

    with st.sidebar.expander("⚙️ Hardware & Models", expanded=False):
        gpu_icon = "🟢" if hw["cuda_available"] else "🔴"
        st.markdown(
            f'<div class="hw-badge">{gpu_icon} GPU: {hw["gpu_name"] or "None"}</div>',
            unsafe_allow_html=True,
        )
        if hw["cuda_available"]:
            st.markdown(
                f'<div class="hw-badge">💾 VRAM: {hw["free_vram_gb"]} / {hw["total_vram_gb"]} GB free</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="hw-badge">📝 Summarizer: {hw["summarizer_device"].upper()}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="hw-badge">🤖 Q&A Model: {hw["instruct_device"].upper()}</div>',
            unsafe_allow_html=True,
        )
        for w in hw.get("warnings", []):
            st.markdown(
                f'<div class="warn-box">⚠️ {w}</div>', unsafe_allow_html=True
            )


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center; padding: 16px 0 8px 0;">'
            '<span style="font-size:2.2rem;">🧠</span>'
            '<div style="font-size:1.3rem; font-weight:700; color:#a090ff; margin-top:4px;">NotesAI</div>'
            '<div style="font-size:0.78rem; color:#666690; margin-bottom:16px;">AI Study Assistant</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        pages = [
            ("🏠", "Dashboard"),
            ("📝", "Summarize Notes"),
            ("❓", "Generate Questions"),
            ("🎯", "Practice Mode"),
            ("ℹ️", "About Project"),
        ]

        st.markdown("**Navigation**")
        for icon, name in pages:
            if st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                use_container_width=True,
                type="secondary" if st.session_state.page != name else "primary",
            ):
                st.session_state.page = name
                st.rerun()

        st.markdown("---")

        # Document status
        if has_document():
            st.markdown(
                f'<div class="hw-badge">📄 {st.session_state.doc_title[:22]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="hw-badge">📊 {st.session_state.doc_wc:,} words</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="hw-badge">📄 No document loaded</div>',
                unsafe_allow_html=True,
            )

        render_hw_status()


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

def page_dashboard():
    st.markdown(
        '<div class="hero-title">🧠 NotesAI</div>'
        '<div class="hero-sub">AI-Powered Notes Summarizer &amp; Question Generator</div>',
        unsafe_allow_html=True,
    )

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        doc_status = f"✅ {st.session_state.doc_title[:16]}" if has_document() else "⭕ None"
        st.markdown(
            f'<div class="notes-card"><div style="font-size:0.8rem;color:#888">Document</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:#c8c8f0">{doc_status}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        wc = f"{st.session_state.doc_wc:,}" if has_document() else "—"
        st.markdown(
            f'<div class="notes-card"><div style="font-size:0.8rem;color:#888">Word Count</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:#c8c8f0">{wc}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        sum_status = "✅ Ready" if has_summary() else "⭕ Not generated"
        st.markdown(
            f'<div class="notes-card"><div style="font-size:0.8rem;color:#888">Summary</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:#c8c8f0">{sum_status}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        q_count = len(st.session_state.questions) if has_questions() else 0
        q_status = f"✅ {q_count} questions" if q_count else "⭕ None"
        st.markdown(
            f'<div class="notes-card"><div style="font-size:0.8rem;color:#888">Questions</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:#c8c8f0">{q_status}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Feature cards
    st.markdown('<div class="section-title">What NotesAI Can Do</div>', unsafe_allow_html=True)
    fc = st.columns(4)
    features = [
        ("📄", "Smart Summarization", "Converts your notes into concise, structured summaries with BART."),
        ("❓", "Question Generation", "Generates MCQs, short, long, and True/False questions from your notes."),
        ("🎯", "Practice Mode", "Interactive quiz mode with instant feedback and scoring."),
        ("📥", "PDF & TXT Export", "Download summaries and question sets for offline study."),
    ]
    for col, (icon, title, desc) in zip(fc, features):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-desc">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # How it works
    st.markdown('<div class="section-title">How to Use NotesAI</div>', unsafe_allow_html=True)
    steps = [
        ("📤", "Upload Notes", "Upload a PDF, TXT file, or paste your notes directly."),
        ("🔍", "Summarize", "Choose a length and format, then click Generate Summary."),
        ("❓", "Get Questions", "Select question type, difficulty, and count, then generate."),
        ("🎯", "Practice", "Test yourself in interactive practice mode and track your score."),
    ]
    cols = st.columns(4)
    for i, (col, (icon, title, desc)) in enumerate(zip(cols, steps), start=1):
        with col:
            st.markdown(
                f'<div class="notes-card" style="text-align:center;">'
                f'<div class="step-badge">{i}</div>'
                f'<span style="font-size:1.5rem">{icon}</span>'
                f'<div style="font-weight:600;color:#c8c8f0;margin:6px 0 4px">{title}</div>'
                f'<div style="font-size:0.82rem;color:#888">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    if not has_document():
        st.info(
            "👆 **Get started** — Upload your notes using **Summarize Notes** or "
            "**Generate Questions** in the left sidebar."
        )
    else:
        st.success(
            f"✅ **Document loaded:** {st.session_state.doc_title} "
            f"({st.session_state.doc_wc:,} words). "
            "Use the sidebar to summarize or generate questions."
        )


# ============================================================================
# DOCUMENT INPUT WIDGET (shared between Summarize and Generate pages)
# ============================================================================

def render_document_input(page_key: str):
    """Render the document upload / paste UI and store in session state."""
    st.markdown('<div class="section-title">📤 Load Your Notes</div>', unsafe_allow_html=True)

    input_method = st.radio(
        "Input method",
        ["Upload PDF", "Upload TXT", "Paste Text"],
        horizontal=True,
        key=f"input_method_{page_key}",
    )

    doc_title = st.text_input(
        "Document title (optional)",
        value=st.session_state.doc_title or "",
        placeholder="e.g. Chapter 3 — Data Structures",
        key=f"doc_title_{page_key}",
    )

    uploaded_file = None
    pasted = None

    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Choose a PDF file (max 50 MB)",
            type=["pdf"],
            key=f"uploader_{page_key}_pdf",
        )
    elif input_method == "Upload TXT":
        uploaded_file = st.file_uploader(
            "Choose a TXT file (max 50 MB)",
            type=["txt"],
            key=f"uploader_{page_key}_txt",
        )
    else:
        pasted = st.text_area(
            "Paste your notes here",
            height=250,
            placeholder="Paste your lecture notes, textbook extracts, or any study material…",
            key=f"paste_{page_key}",
        )

    col_load, col_clear = st.columns([2, 1])

    with col_load:
        if st.button("📥 Load Document", key=f"load_btn_{page_key}", use_container_width=True):
            with st.spinner("Extracting and validating document…"):
                try:
                    if input_method == "Upload PDF" and uploaded_file:
                        text, doc_hash = load_document(
                            "pdf",
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                        )
                        title = doc_title or uploaded_file.name.replace(".pdf", "")
                        set_document(text, title, "pdf", doc_hash)
                        st.success(f"✅ PDF loaded: {word_count(text):,} words extracted.")
                    elif input_method == "Upload TXT" and uploaded_file:
                        text, doc_hash = load_document(
                            "txt",
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                        )
                        title = doc_title or uploaded_file.name.replace(".txt", "")
                        set_document(text, title, "txt", doc_hash)
                        st.success(f"✅ TXT loaded: {word_count(text):,} words.")
                    elif input_method == "Paste Text" and pasted:
                        text, doc_hash = load_document("paste", pasted_text=pasted)
                        title = doc_title or "Pasted Notes"
                        set_document(text, title, "paste", doc_hash)
                        st.success(f"✅ Text loaded: {word_count(text):,} words.")
                    else:
                        st.warning("Please select a file or paste some text first.")
                except DocumentError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    logger.exception("Document load error")

    with col_clear:
        if st.button("🗑️ Clear", key=f"clear_btn_{page_key}", use_container_width=True):
            st.session_state.doc_text = None
            st.session_state.doc_title = ""
            st.session_state.doc_hash = None
            invalidate_dependent_results()
            st.rerun()

    # Preview
    if has_document():
        with st.expander("👁️ Preview extracted text", expanded=False):
            preview = st.session_state.doc_text[:2000]
            if len(st.session_state.doc_text) > 2000:
                preview += f"\n\n… [preview truncated — {st.session_state.doc_wc:,} words total]"
            st.text(preview)


# ============================================================================
# PAGE: SUMMARIZE NOTES
# ============================================================================

def page_summarize():
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">📝 Summarize Notes</div>', unsafe_allow_html=True)

    render_document_input("summarize")

    if not has_document():
        return

    st.markdown("---")
    st.markdown('<div class="section-title">⚙️ Summary Options</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        length = st.selectbox(
            "Summary length",
            options=["short", "medium", "detailed"],
            format_func=lambda x: {"short": "Short (brief overview)", "medium": "Medium (balanced)", "detailed": "Detailed (comprehensive)"}[x],
            key="sum_length",
        )
        fmt = st.selectbox(
            "Summary format",
            options=["structured_notes", "bullet_points", "exam_revision"],
            format_func=lambda x: {
                "structured_notes": "📋 Structured Notes (headings & paragraphs)",
                "bullet_points": "• Bullet Point Summary",
                "exam_revision": "📖 Exam Revision Notes",
            }[x],
            key="sum_format",
        )
    with col2:
        include_concepts = st.checkbox("Include key concepts", value=True, key="sum_concepts")
        include_defs = st.checkbox("Include definitions", value=True, key="sum_defs")
        preserve_examples = st.checkbox("Preserve important examples", value=True, key="sum_examples")

    st.markdown("---")

    if st.button("🚀 Generate Summary", key="gen_summary_btn", use_container_width=True, type="primary"):
        _run_summarization(length, fmt, include_concepts, include_defs, preserve_examples)

    if has_summary():
        _render_summary_results()


def _run_summarization(length, fmt, include_concepts, include_defs, preserve_examples):
    """Run summarization and store result in session state."""
    progress_bar = st.progress(0, text="Starting summarization…")
    status_text = st.empty()

    def progress_cb(current, total):
        if total > 0:
            pct = int((current / total) * 80)
            progress_bar.progress(pct, text=f"Processing section {current + 1} of {total}…")

    try:
        svc = get_ai_service()
        status_text.info("🔄 Loading AI models (first run may take several minutes)…")

        with st.spinner("Summarizing your notes… this may take a few minutes on CPU."):
            result = generate_summary(
                text=st.session_state.doc_text,
                ai_service=svc,
                length=length,
                fmt=fmt,
                include_key_concepts=include_concepts,
                include_definitions=include_defs,
                preserve_examples=preserve_examples,
                doc_title=st.session_state.doc_title,
                progress_callback=progress_cb,
            )

        progress_bar.progress(100, text="Done!")
        status_text.empty()
        st.session_state.summary = result
        st.session_state.summary_hash = st.session_state.doc_hash
        st.success("✅ Summary generated successfully!")

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        logger.exception("Summarization failed")
        st.error(f"❌ Summarization failed: {exc}")
        _show_model_help()


def _render_summary_results():
    summary = st.session_state.summary
    if not summary:
        return

    st.markdown("---")
    st.markdown('<div class="section-title">📄 Summary Results</div>', unsafe_allow_html=True)

    # Meta info
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.markdown(
            f'<div class="stat-chip">📏 {summary["word_count"]} words</div>',
            unsafe_allow_html=True,
        )
    with meta_cols[1]:
        st.markdown(
            f'<div class="stat-chip">📂 {summary["chunk_count"]} section(s) processed</div>',
            unsafe_allow_html=True,
        )
    with meta_cols[2]:
        fmt_labels = {
            "structured_notes": "Structured Notes",
            "bullet_points": "Bullet Points",
            "exam_revision": "Exam Revision",
        }
        st.markdown(
            f'<div class="stat-chip">🗂️ {fmt_labels.get(summary["format_used"], summary["format_used"])}</div>',
            unsafe_allow_html=True,
        )

    # Summary text
    st.markdown(
        f'<div class="notes-card">{summary["summary_text"]}</div>',
        unsafe_allow_html=True,
    )

    # Key concepts
    if summary.get("key_concepts"):
        with st.expander("🔑 Key Concepts", expanded=True):
            for c in summary["key_concepts"]:
                st.markdown(f"• {c}")

    # Definitions
    if summary.get("important_definitions"):
        with st.expander("📖 Important Definitions", expanded=False):
            for d in summary["important_definitions"]:
                st.markdown(f"• {d}")

    # Coverage notice
    if summary["chunk_count"] > 1:
        st.info(
            f"ℹ️ Your document was split into **{summary['chunk_count']} sections** "
            "for processing. The summary covers all sections."
        )

    # Download buttons
    st.markdown("---")
    st.markdown("**📥 Download Summary**")
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])
    with dl_col1:
        txt_bytes = export_summary_txt(summary, st.session_state.doc_title)
        st.download_button(
            "⬇️ Download TXT",
            data=txt_bytes,
            file_name=f"notesai_summary_{st.session_state.doc_title[:20]}.txt",
            mime="text/plain",
            key="dl_sum_txt",
        )
    with dl_col2:
        try:
            pdf_bytes = export_summary_pdf(summary, st.session_state.doc_title)
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"notesai_summary_{st.session_state.doc_title[:20]}.pdf",
                mime="application/pdf",
                key="dl_sum_pdf",
            )
        except Exception as e:
            st.warning(f"PDF export error: {e}")


# ============================================================================
# PAGE: GENERATE QUESTIONS
# ============================================================================

def page_generate_questions():
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">❓ Generate Questions</div>', unsafe_allow_html=True)

    render_document_input("questions")

    if not has_document():
        return

    st.markdown("---")
    st.markdown('<div class="section-title">⚙️ Question Options</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        q_type = st.selectbox(
            "Question type",
            options=["mcq", "short", "long", "true_false", "mixed"],
            format_func=lambda x: {
                "mcq": "📋 Multiple Choice (MCQ)",
                "short": "✏️ Short Answer",
                "long": "📝 Long Answer",
                "true_false": "✅ True / False",
                "mixed": "🔀 Mixed (all types)",
            }[x],
            key="q_type",
        )
        difficulty = st.selectbox(
            "Difficulty level",
            options=["easy", "medium", "hard"],
            index=1,
            format_func=lambda x: {
                "easy": "🟢 Easy — Recall & definitions",
                "medium": "🟡 Medium — Understanding & explanation",
                "hard": "🔴 Hard — Analysis & application",
            }[x],
            key="q_difficulty",
        )
    with col2:
        q_count = st.slider(
            "Number of questions",
            min_value=1,
            max_value=20,
            value=5,
            key="q_count",
        )
        st.info(
            "ℹ️ Fewer valid questions may be returned if the notes "
            "don't contain enough distinct facts for the requested count."
        )

    st.markdown("---")

    if st.button("🚀 Generate Questions", key="gen_q_btn", use_container_width=True, type="primary"):
        _run_question_generation(q_type, q_count, difficulty)

    if has_questions():
        _render_question_results()


def _run_question_generation(q_type, q_count, difficulty):
    progress_bar = st.progress(0, text="Starting question generation…")
    status_text = st.empty()

    def progress_cb(current, total):
        if total > 0:
            pct = int((current / total) * 90)
            progress_bar.progress(pct, text=f"Processing chunk {current + 1} of {total}…")

    try:
        svc = get_ai_service()
        status_text.info("🔄 Loading AI models (first run may take several minutes)…")

        with st.spinner("Generating questions… this may take a few minutes on CPU."):
            questions, warnings = generate_questions(
                text=st.session_state.doc_text,
                ai_service=svc,
                q_type=q_type,
                count=q_count,
                difficulty=difficulty,
                progress_callback=progress_cb,
            )

        progress_bar.progress(100, text="Done!")
        status_text.empty()

        st.session_state.questions = questions
        st.session_state.questions_hash = st.session_state.doc_hash
        st.session_state.questions_type = q_type
        st.session_state.student_answers = {}
        st.session_state.practice_submitted = False
        st.session_state.practice_score = None

        for w in warnings:
            st.warning(f"⚠️ {w}")

        if questions:
            st.success(f"✅ {len(questions)} question(s) generated successfully!")
        else:
            st.error(
                "❌ No valid questions could be generated. The notes may be too short "
                "or the model output may not have passed validation. Try a shorter document "
                "or a different question type."
            )

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        logger.exception("Question generation failed")
        st.error(f"❌ Question generation failed: {exc}")
        _show_model_help()


def _render_question_results():
    questions = st.session_state.questions
    if not questions:
        return

    st.markdown("---")
    st.markdown(
        f'<div class="section-title">📋 Generated Questions ({len(questions)})</div>',
        unsafe_allow_html=True,
    )

    for idx, q in enumerate(questions, start=1):
        _render_single_question(q, idx, show_answers=True, interactive=False)

    # Download buttons
    st.markdown("---")
    st.markdown("**📥 Download Questions**")
    dl_cols = st.columns(4)
    title = st.session_state.doc_title

    with dl_cols[0]:
        st.download_button(
            "⬇️ Q&A TXT",
            data=export_questions_txt(questions, title, include_answers=True),
            file_name=f"notesai_qa_{title[:20]}.txt",
            mime="text/plain",
            key="dl_qa_txt",
        )
    with dl_cols[1]:
        try:
            st.download_button(
                "⬇️ Q&A PDF",
                data=export_questions_pdf(questions, title, include_answers=True),
                file_name=f"notesai_qa_{title[:20]}.pdf",
                mime="application/pdf",
                key="dl_qa_pdf",
            )
        except Exception as e:
            st.warning(f"PDF: {e}")
    with dl_cols[2]:
        st.download_button(
            "⬇️ Practice TXT",
            data=export_questions_txt(questions, title, include_answers=False),
            file_name=f"notesai_practice_{title[:20]}.txt",
            mime="text/plain",
            key="dl_prac_txt",
        )
    with dl_cols[3]:
        try:
            st.download_button(
                "⬇️ Practice PDF",
                data=export_questions_pdf(questions, title, include_answers=False),
                file_name=f"notesai_practice_{title[:20]}.pdf",
                mime="application/pdf",
                key="dl_prac_pdf",
            )
        except Exception as e:
            st.warning(f"PDF: {e}")


# ============================================================================
# PAGE: PRACTICE MODE
# ============================================================================

def page_practice():
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">🎯 Practice Mode</div>', unsafe_allow_html=True)

    if not has_questions():
        st.info(
            "⭕ No questions available yet. Go to **Generate Questions** to create a question set first."
        )
        return

    questions = st.session_state.questions
    total = len(questions)

    st.markdown(
        f'<div class="notes-card">'
        f'<b>{st.session_state.doc_title}</b> — {total} question(s) loaded.'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Score summary (if submitted)
    if st.session_state.practice_submitted and st.session_state.practice_score:
        _render_score_summary(st.session_state.practice_score)
        st.markdown("---")

    # Render each question interactively
    for idx, q in enumerate(questions):
        _render_practice_question(q, idx)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Submit & Score", key="submit_practice", use_container_width=True, type="primary"):
            score = calculate_score(questions, st.session_state.student_answers)
            st.session_state.practice_score = score
            st.session_state.practice_submitted = True
            st.rerun()
    with col2:
        if st.button("🔄 Reset Answers", key="reset_practice", use_container_width=True):
            st.session_state.student_answers = {}
            st.session_state.practice_submitted = False
            st.session_state.practice_score = None
            st.rerun()


def _render_practice_question(q: AnyQuestion, idx: int):
    submitted = st.session_state.practice_submitted
    score_data = st.session_state.practice_score
    student_ans = st.session_state.student_answers.get(idx)

    # Determine result color if submitted
    result_style = ""
    if submitted and score_data:
        per_q = {r["index"]: r for r in score_data.get("per_question", [])}
        if idx in per_q:
            r = per_q[idx]["result"]
            if r == "correct":
                result_style = "border-color: #4ade80;"
            elif r == "incorrect":
                result_style = "border-color: #f87171;"

    q_label = getattr(q, "question", None) or getattr(q, "statement", "")
    q_type_label = {
        "mcq": "MCQ",
        "short": "Short Answer",
        "long": "Long Answer",
        "true_false": "True / False",
    }.get(q.question_type.value, q.question_type.value.upper())

    st.markdown(
        f'<div class="q-card" style="{result_style}">'
        f'<div class="q-type-badge">{q_type_label} • Q{idx + 1} • {q.difficulty.value.upper()}</div>',
        unsafe_allow_html=True,
    )

    if isinstance(q, MCQQuestion):
        st.markdown(f"**{q.question}**")
        options_list = [f"{k}: {v}" for k, v in q.options.items()]
        sel = st.radio(
            f"Select answer for Q{idx + 1}",
            options=options_list,
            key=f"practice_mcq_{idx}",
            label_visibility="collapsed",
            index=None,
        )
        if sel:
            chosen_key = sel.split(":")[0].strip()
            st.session_state.student_answers[idx] = chosen_key

        if submitted:
            correct = q.correct_answer
            if student_ans == correct:
                st.markdown(f'<span class="correct-ans">✓ Correct!</span>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span class="wrong-ans">✗ Incorrect. Correct answer: {correct}. {q.options[correct]}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**Explanation:** {q.explanation}")
            st.markdown(
                f'<div class="evidence-box">📎 Source: "{q.source_evidence[:150]}"</div>',
                unsafe_allow_html=True,
            )

    elif isinstance(q, TrueFalseQuestion):
        st.markdown(f"**Statement:** {q.statement}")
        sel = st.radio(
            f"True or False for Q{idx + 1}",
            options=["True", "False"],
            key=f"practice_tf_{idx}",
            horizontal=True,
            label_visibility="collapsed",
            index=None,
        )
        if sel:
            st.session_state.student_answers[idx] = sel == "True"

        if submitted:
            correct = "True" if q.correct_answer else "False"
            student_str = "True" if student_ans else "False"
            if student_ans == q.correct_answer:
                st.markdown('<span class="correct-ans">✓ Correct!</span>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span class="wrong-ans">✗ Incorrect. Answer: {correct}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**Explanation:** {q.explanation}")
            st.markdown(
                f'<div class="evidence-box">📎 Source: "{q.source_evidence[:150]}"</div>',
                unsafe_allow_html=True,
            )

    elif isinstance(q, ShortAnswerQuestion):
        st.markdown(f"**{q.question}**")
        answer_input = st.text_area(
            f"Your answer for Q{idx + 1}",
            key=f"practice_short_{idx}",
            height=80,
            label_visibility="collapsed",
            placeholder="Type your answer here…",
        )
        if answer_input:
            st.session_state.student_answers[idx] = answer_input

        if submitted:
            with st.expander("📖 See Model Answer & Explanation"):
                st.markdown(f"**Expected Answer:** {q.expected_answer}")
                if q.key_points:
                    st.markdown("**Key Points:**")
                    for kp in q.key_points:
                        st.markdown(f"• {kp}")
                st.markdown(f"**Explanation:** {q.explanation}")
                st.markdown(
                    f'<div class="evidence-box">📎 Source: "{q.source_evidence[:150]}"</div>',
                    unsafe_allow_html=True,
                )
            st.info(
                "ℹ️ Short-answer and long-answer responses are not auto-scored. "
                "Compare your answer with the model answer above."
            )

    elif isinstance(q, LongAnswerQuestion):
        st.markdown(f"**{q.question}**")
        answer_input = st.text_area(
            f"Your answer for Q{idx + 1}",
            key=f"practice_long_{idx}",
            height=160,
            label_visibility="collapsed",
            placeholder="Write your detailed answer here…",
        )
        if answer_input:
            st.session_state.student_answers[idx] = answer_input

        if submitted:
            with st.expander("📖 See Model Answer & Marking Guidance"):
                st.markdown(f"**Model Answer:**\n\n{q.model_answer}")
                st.markdown("**Main Points Expected:**")
                for mp in q.main_points:
                    st.markdown(f"• {mp}")
                if q.marking_guidance:
                    st.markdown(
                        f"*Marking guidance (AI-generated, not official):* {q.marking_guidance}"
                    )
                st.markdown(
                    f'<div class="evidence-box">📎 Source: "{q.source_evidence[:150]}"</div>',
                    unsafe_allow_html=True,
                )
            st.info(
                "ℹ️ Long-answer responses are not auto-scored. "
                "Use the marking guidance above as a self-assessment reference."
            )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_score_summary(score: dict):
    total = score["total"]
    correct = score["correct"]
    pct = score["score_pct"]

    color = "#4ade80" if pct >= 70 else "#fbbf24" if pct >= 40 else "#f87171"
    grade = "Excellent!" if pct >= 80 else "Good job!" if pct >= 60 else "Keep practicing!"

    st.markdown(
        f'<div class="score-card">'
        f'<div class="score-pct" style="color:{color};">{pct}%</div>'
        f'<div style="color:#aaaacc; margin: 4px 0;">{grade}</div>'
        f'<div style="font-size:0.9rem; color:#888;">Scored {correct} / {total} objective questions</div>'
        f'<div style="font-size:0.82rem; color:#666; margin-top:8px;">'
        f'Attempted: {score["attempted"]} | Correct: {score["correct"]} | '
        f'Incorrect: {score["incorrect"]} | Skipped: {score["skipped"]}'
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# QUESTION RENDERER (read-only, for Generate Questions page)
# ============================================================================

def _render_single_question(q: AnyQuestion, idx: int, show_answers: bool = True, interactive: bool = False):
    q_label = getattr(q, "question", None) or getattr(q, "statement", "")
    q_type_label = {
        "mcq": "MCQ",
        "short": "Short Answer",
        "long": "Long Answer",
        "true_false": "True / False",
    }.get(q.question_type.value, "")

    with st.expander(f"**Q{idx}** ({q_type_label} | {q.difficulty.value.upper()}): {q_label[:70]}…", expanded=False):
        if isinstance(q, MCQQuestion):
            st.markdown(f"**{q.question}**")
            for key, val in q.options.items():
                marker = "✅ " if (show_answers and key == q.correct_answer) else "   "
                st.markdown(f"{marker}**{key}.** {val}")
            if show_answers:
                st.markdown(f"**✓ Correct Answer:** {q.correct_answer}")
                st.markdown(f"**Explanation:** {q.explanation}")
                st.markdown(
                    f'<div class="evidence-box">📎 Source: "{q.source_evidence[:200]}"</div>',
                    unsafe_allow_html=True,
                )

        elif isinstance(q, ShortAnswerQuestion):
            st.markdown(f"**{q.question}**")
            if show_answers:
                st.markdown(f"**Expected Answer:** {q.expected_answer}")
                if q.key_points:
                    st.markdown("**Key Points:** " + " | ".join(q.key_points))
                st.markdown(f"**Explanation:** {q.explanation}")
                st.markdown(
                    f'<div class="evidence-box">📎 Source: "{q.source_evidence[:200]}"</div>',
                    unsafe_allow_html=True,
                )

        elif isinstance(q, LongAnswerQuestion):
            st.markdown(f"**{q.question}**")
            if show_answers:
                st.markdown(f"**Model Answer:**\n\n{q.model_answer}")
                st.markdown("**Main Points:** " + " | ".join(q.main_points))
                if q.marking_guidance:
                    st.markdown(f"*Marking guidance (AI-generated):* {q.marking_guidance}")
                st.markdown(
                    f'<div class="evidence-box">📎 Source: "{q.source_evidence[:200]}"</div>',
                    unsafe_allow_html=True,
                )

        elif isinstance(q, TrueFalseQuestion):
            st.markdown(f"**Statement:** {q.statement}")
            if show_answers:
                ans_str = "✅ TRUE" if q.correct_answer else "❌ FALSE"
                st.markdown(f"**Answer:** {ans_str}")
                st.markdown(f"**Explanation:** {q.explanation}")
                st.markdown(
                    f'<div class="evidence-box">📎 Source: "{q.source_evidence[:200]}"</div>',
                    unsafe_allow_html=True,
                )


# ============================================================================
# PAGE: ABOUT PROJECT
# ============================================================================

def page_about():
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">ℹ️ About NotesAI</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
### Project Abstract
**NotesAI** is an AI-powered academic study assistant built as a college
Continuous Assessment (CA) project. It helps students study more effectively
by transforming their raw lecture notes and textbook extracts into accurate
summaries and exam-oriented questions.

### Problem Statement
Students often struggle to identify the most important concepts in large sets
of notes, and manually writing practice questions is time-consuming. NotesAI
automates both tasks using locally running AI models, requiring no internet
connection after the initial model download.

### Objectives
1. Extract and summarize key information from academic notes.
2. Generate diverse, accurate exam questions grounded in the notes.
3. Provide an interactive practice environment for self-assessment.
4. Produce downloadable study materials.
5. Run entirely on consumer hardware without cloud API costs.
""")

    with col2:
        st.markdown(
            '<div class="notes-card">',
            unsafe_allow_html=True,
        )
        st.markdown("**🛠️ Technology Stack**")
        tech = [
            ("🐍", "Python 3.10/3.11"),
            ("🖥️", "Streamlit — UI"),
            ("🤗", "HuggingFace Transformers"),
            ("🔥", "PyTorch"),
            ("📄", "PyMuPDF — PDF extraction"),
            ("📊", "ReportLab — PDF reports"),
            ("✅", "Pydantic — Output validation"),
            ("🧪", "pytest — Testing"),
        ]
        for icon, name in tech:
            st.markdown(f"{icon} {name}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
### 🤖 AI Models Used

**Summarization — `facebook/bart-large-cnn`**
- Parameters: ~406 million
- Download size: ~1.6 GB
- Context window: 1 024 tokens
- Why chosen: Specifically fine-tuned for abstractive text summarization;
  runs reliably on both CPU and GPU; well-documented; permissive license.

**Q&A Generation — `Qwen/Qwen2.5-1.5B-Instruct`**
- Parameters: ~1.54 billion
- Download size: ~3 GB (FP16)
- Why chosen: Strong instruction-following and JSON compliance at low latency;
  fits directly on RTX 3050 (6GB) GPU for real-time inference in seconds;
  permissive Qwen Research License.
""")

    with col4:
        st.markdown("""
### 💻 Hardware Requirements

| Component | Recommended |
|-----------|-------------|
| RAM | 16 GB or more |
| CPU | Intel i5 or equivalent |
| GPU | Optional (RTX 3050+) |
| Disk | ~6 GB for models |
| Python | 3.10 or 3.11 |

### ⚠️ Known Limitations
- First-run model download requires internet access.
- CPU inference is slower than GPU inference.
- BART context window limits chunk size to ~1 024 tokens.
- JSON output parsing may occasionally fail on malformed model output;
  such questions are discarded and a warning is shown.
- Qwen2.5-1.5B may not always produce exactly the requested number of
  valid questions for very short notes.
""")

    st.markdown("---")
    st.markdown("""
### 📐 Project Architecture

```
NotesAI/
├── app.py               — Streamlit UI, navigation, session state
├── ai_service.py        — Hardware detection, model loading, inference
├── summarizer.py        — BART summarization + Qwen formatting
├── question_generator.py— Question generation, validation, scoring
├── document_processor.py— PDF/TXT extraction, chunking, normalisation
├── validators.py        — Pydantic models for all question types
├── export_service.py    — TXT and PDF report generation
├── requirements.txt     — Python dependencies
├── README.md            — Full setup and usage guide
└── tests/               — Automated pytest test suite
```
""")

    st.markdown(
        '<div class="warn-box">📌 This project uses locally downloaded pretrained Hugging Face '
        "models. No custom model training was performed. The models are used "
        "as-is with carefully designed prompts to ground all output in the "
        "student's notes.</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# MODEL HELP
# ============================================================================

def _show_model_help():
    with st.expander("💡 Model Download Instructions"):
        st.markdown("""
If models are not yet downloaded, run the following in your terminal
(inside your virtual environment):

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Download summarization model (~1.6 GB)
pipeline("summarization", model="facebook/bart-large-cnn")

# Download instruction model (~3 GB)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
```

Or run the provided download script:
```
python download_models.py
```

Models are cached automatically in `~/.cache/huggingface/hub/`.
""")


# ============================================================================
# MAIN ROUTER
# ============================================================================

def main():
    render_sidebar()

    page = st.session_state.page

    if page == "Dashboard":
        page_dashboard()
    elif page == "Summarize Notes":
        page_summarize()
    elif page == "Generate Questions":
        page_generate_questions()
    elif page == "Practice Mode":
        page_practice()
    elif page == "About Project":
        page_about()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
