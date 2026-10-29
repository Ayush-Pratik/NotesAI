"""
summarizer.py
=============
Academic notes summarization using facebook/bart-large-cnn via ai_service.py.

Supports three output lengths (short / medium / detailed) and three
presentation formats (structured notes / bullet points / exam revision).
Handles long documents by chunking, summarizing each chunk, and then
producing a coherent final summary.
"""

from __future__ import annotations

import re
import textwrap
from typing import Callable, List, Optional

from document_processor import split_into_chunks, word_count

# BART max input tokens is 1024; keep chunk size safely at 2200 chars (~500 tokens)
BART_CHUNK_CHARS = 2_200
BART_CHUNK_OVERLAP = 200

# Output token limits per length setting
LENGTH_PARAMS = {
    "short":    {"min_length": 60,  "max_length": 180},
    "medium":   {"min_length": 120, "max_length": 350},
    "detailed": {"min_length": 200, "max_length": 512},
}

# Qwen system prompt for format post-processing
FORMAT_SYSTEM_PROMPT = """\
You are an academic study assistant. Your job is to reformat a raw summary \
into a specific study format. You must not add any facts, definitions, or \
examples that are not already present in the raw summary. Preserve all \
technical terms, formulas, and numerical values exactly as they appear. \
Do not mention that you are an AI."""

FORMAT_USER_TEMPLATES = {
    "structured_notes": """\
Reformat the following raw summary into well-organised structured notes with \
clear headings (using Markdown ## and ### syntax), short paragraphs, and \
bullet points where appropriate. Do NOT add any new information.

Raw summary:
{summary}
""",
    "bullet_points": """\
Convert the following raw summary into a clean bullet-point list. Use \
- for main points and  • for sub-points. Group related ideas under short \
bold headings. Do NOT add any new information.

Raw summary:
{summary}
""",
    "exam_revision": """\
Reformat the following raw summary into exam revision notes. Include:
1. Key definitions (bold the term).
2. Important facts as numbered points.
3. Formulas or processes if present.
4. A "Quick Recall" section at the end with 3–5 single-sentence reminders.
Do NOT add any new information beyond what is in the raw summary.

Raw summary:
{summary}
""",
}


def _chunk_and_summarize(
    text: str,
    ai_service,
    length: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Split `text` into chunks, summarize each with BART, then combine.

    Parameters
    ----------
    text : str
        Full document text.
    ai_service : AIService
        Service instance for model inference.
    length : str
        "short" | "medium" | "detailed"
    progress_callback : callable, optional
        Called with (current_chunk_index, total_chunks) for progress bars.

    Returns
    -------
    str
        Combined intermediate summary (plain text, not yet formatted).
    """
    params = LENGTH_PARAMS[length]
    chunks = split_into_chunks(text, chunk_size=BART_CHUNK_CHARS, overlap=BART_CHUNK_OVERLAP)

    if not chunks:
        raise ValueError("Document produced no chunks for summarization.")

    chunk_summaries: List[str] = []
    for idx, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(idx, len(chunks))
        if not chunk.strip():
            continue
        try:
            summary = ai_service.summarize(
                chunk,
                min_length=params["min_length"],
                max_length=params["max_length"],
            )
            chunk_summaries.append(summary)
        except Exception as exc:
            # Log and skip failed chunks rather than silently dropping them
            chunk_summaries.append(
                f"[Summarization failed for section {idx + 1}: {exc}]"
            )

    if not chunk_summaries:
        raise RuntimeError("All chunks failed to summarize. Check model availability.")

    combined = " ".join(chunk_summaries)

    # If we had multiple chunks, do one more consolidation pass if the
    # combined text is still longer than a single chunk
    if len(chunks) > 1 and len(combined) > BART_CHUNK_CHARS:
        try:
            combined = ai_service.summarize(
                combined,
                min_length=params["min_length"],
                max_length=params["max_length"],
            )
        except Exception:
            pass  # Use the concatenated version if consolidation fails

    return combined


def _extract_key_concepts(text: str, ai_service) -> List[str]:
    """
    Ask Qwen to extract a bullet list of key concepts from the notes.
    Returns a list of concept strings.
    """
    system = (
        "You are an academic assistant. Extract key concepts, terms, and definitions "
        "from the supplied notes. Return them as a simple bullet list (one per line, "
        "starting with -). Only include concepts explicitly mentioned in the notes. "
        "Do not add external knowledge."
    )
    user = f"Notes:\n{text[:4000]}\n\nList the key concepts:"
    try:
        raw = ai_service.generate(system, user, max_new_tokens=400, temperature=0.1)
        concepts = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if line and len(line) > 3:
                concepts.append(line)
        return concepts[:20]  # cap at 20
    except Exception:
        return []


def _extract_definitions(text: str, ai_service) -> List[str]:
    """
    Ask Qwen to extract definitions from the notes.
    Returns a list of "Term: definition" strings.
    """
    system = (
        "You are an academic assistant. Extract all definitions from the notes. "
        "Format each as 'Term: definition'. Only include definitions explicitly "
        "stated in the notes. Do not add external knowledge."
    )
    user = f"Notes:\n{text[:4000]}\n\nList the definitions:"
    try:
        raw = ai_service.generate(system, user, max_new_tokens=400, temperature=0.1)
        defs = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if ":" in line and len(line) > 5:
                defs.append(line)
        return defs[:15]
    except Exception:
        return []


def generate_summary(
    text: str,
    ai_service,
    length: str = "medium",
    fmt: str = "structured_notes",
    include_key_concepts: bool = True,
    include_definitions: bool = True,
    preserve_examples: bool = True,
    doc_title: str = "Study Notes",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Main entry point for summary generation.

    Parameters
    ----------
    text : str
        Full extracted document text.
    ai_service : AIService
        Loaded AI service instance.
    length : str
        "short" | "medium" | "detailed"
    fmt : str
        "structured_notes" | "bullet_points" | "exam_revision"
    include_key_concepts : bool
        Whether to extract and append key concepts.
    include_definitions : bool
        Whether to extract and append definitions.
    preserve_examples : bool
        Hint for the user — currently handled via prompt framing.
    doc_title : str
        Document title for the output header.
    progress_callback : callable, optional
        Progress reporting function.

    Returns
    -------
    dict with keys:
        title, summary_text, key_concepts, important_definitions,
        word_count, format_used, length_used, chunk_count
    """
    if length not in LENGTH_PARAMS:
        raise ValueError(f"Invalid length: {length!r}. Choose from {list(LENGTH_PARAMS)}")
    if fmt not in FORMAT_USER_TEMPLATES:
        raise ValueError(f"Invalid format: {fmt!r}.")

    # Step 1: Chunk-based BART summarization
    raw_summary = _chunk_and_summarize(text, ai_service, length, progress_callback)

    # Step 2: Format with Qwen (if raw summary is non-trivial)
    formatted_summary = raw_summary
    if len(raw_summary.strip()) > 50:
        try:
            user_prompt = FORMAT_USER_TEMPLATES[fmt].format(summary=raw_summary)
            formatted_summary = ai_service.generate(
                system_prompt=FORMAT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=700,
                temperature=0.1,
            )
            if not formatted_summary or len(formatted_summary) < 30:
                formatted_summary = raw_summary  # fall back to raw
        except Exception:
            formatted_summary = raw_summary

    # Step 3: Extract key concepts (optional)
    key_concepts: List[str] = []
    if include_key_concepts:
        key_concepts = _extract_key_concepts(text, ai_service)

    # Step 4: Extract definitions (optional)
    definitions: List[str] = []
    if include_definitions:
        definitions = _extract_definitions(text, ai_service)

    chunks = split_into_chunks(text, chunk_size=BART_CHUNK_CHARS, overlap=BART_CHUNK_OVERLAP)

    return {
        "title": doc_title,
        "summary_text": formatted_summary,
        "raw_summary": raw_summary,
        "key_concepts": key_concepts,
        "important_definitions": definitions,
        "word_count": word_count(formatted_summary),
        "format_used": fmt,
        "length_used": length,
        "chunk_count": len(chunks),
    }
