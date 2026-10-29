"""
document_processor.py
======================
Handles extraction and validation of text from PDF files, TXT files,
and directly pasted text. Also provides chunking utilities for long documents.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PASTE_CHARS = 150_000      # ~25 000 words — sane limit for pasted text
MAX_FILE_SIZE_MB = 50          # Enforced at Streamlit config level too
MIN_MEANINGFUL_CHARS = 50      # Below this a document is considered empty
CHUNK_CHAR_SIZE = 3_000        # Characters per chunk for long-doc processing
CHUNK_OVERLAP_CHARS = 300      # Overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DocumentError(Exception):
    """Raised when document extraction or validation fails."""


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """
    Normalize Unicode, collapse excessive whitespace, and strip leading/
    trailing whitespace from each line.
    """
    # Normalize Unicode to NFC form
    text = unicodedata.normalize("NFC", text)
    # Replace common ligatures / smart quotes
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u00ad": "",   # soft hyphen
        "\ufeff": "",   # BOM
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def word_count(text: str) -> int:
    """Return approximate word count."""
    return len(text.split())


def compute_hash(text: str) -> str:
    """Return a short SHA-256 hex digest used to detect document changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, List[int]]:
    """
    Extract text from PDF bytes using PyMuPDF.

    Returns
    -------
    text : str
        Concatenated, normalised text from all pages.
    page_char_offsets : list[int]
        Character offset in `text` where each page starts.
        Useful for rough source attribution.

    Raises
    ------
    DocumentError
        If the file is not a valid PDF, is encrypted, or yields no text.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise DocumentError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        ) from exc

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise DocumentError(f"Cannot open PDF: {exc}") from exc

    if doc.is_encrypted:
        raise DocumentError(
            "The PDF is password-protected. Please provide an unlocked file."
        )

    pages_text: List[str] = []
    page_char_offsets: List[int] = []
    offset = 0

    for page_num, page in enumerate(doc, start=1):
        try:
            raw = page.get_text("text")  # plain text extraction
        except Exception:
            raw = ""
        cleaned = normalise_text(raw)
        if cleaned:
            pages_text.append(f"[Page {page_num}]\n{cleaned}")
            page_char_offsets.append(offset)
            offset += len(pages_text[-1]) + 1  # +1 for newline separator

    doc.close()

    if not pages_text:
        raise DocumentError(
            "No readable text was found in this PDF. "
            "The file may contain scanned images without OCR text."
        )

    full_text = "\n\n".join(pages_text)
    return full_text, page_char_offsets


# ---------------------------------------------------------------------------
# TXT extraction
# ---------------------------------------------------------------------------

def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decode a TXT file (UTF-8 with BOM fallback, then latin-1).

    Raises
    ------
    DocumentError
        If the file cannot be decoded or is empty.
    """
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            text = normalise_text(text)
            if len(text) < MIN_MEANINGFUL_CHARS:
                raise DocumentError("The uploaded TXT file appears to be empty.")
            return text
        except (UnicodeDecodeError, ValueError):
            continue
    raise DocumentError(
        "Cannot decode the TXT file. Please save it as UTF-8 and try again."
    )


# ---------------------------------------------------------------------------
# Pasted text
# ---------------------------------------------------------------------------

def process_pasted_text(text: str) -> str:
    """
    Validate and normalise text entered directly in the UI.

    Raises
    ------
    DocumentError
        If the text is too short or too long.
    """
    text = normalise_text(text)
    if len(text) < MIN_MEANINGFUL_CHARS:
        raise DocumentError(
            "The pasted text is too short. Please paste your full notes."
        )
    if len(text) > MAX_PASTE_CHARS:
        raise DocumentError(
            f"The pasted text is too long ({len(text):,} characters). "
            f"Maximum allowed is {MAX_PASTE_CHARS:,} characters (~25 000 words). "
            "Please upload a file instead."
        )
    return text


# ---------------------------------------------------------------------------
# Chunking for long documents
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_CHAR_SIZE,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """
    Split `text` into overlapping character-level chunks.

    Splits prefer paragraph boundaries (double newline) to avoid cutting
    sentences mid-way.  Each chunk contains at most `chunk_size` characters.

    Parameters
    ----------
    text : str
        The full document text.
    chunk_size : int
        Maximum characters per chunk.
    overlap : int
        Number of characters shared between consecutive chunks.

    Returns
    -------
    list[str]
        Ordered list of text chunks (never empty).
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r"\n\n+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 2 > chunk_size and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            # Build overlap: take tail of current chunk
            overlap_text = chunk_text[-overlap:] if overlap > 0 else ""
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
        current.append(para)
        current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Document validation
# ---------------------------------------------------------------------------

def validate_document(text: str) -> None:
    """
    Raise DocumentError if the extracted text does not meet quality thresholds.
    """
    if not text or not text.strip():
        raise DocumentError("The document is empty.")
    if len(text.strip()) < MIN_MEANINGFUL_CHARS:
        raise DocumentError(
            f"The document contains too little text ({len(text.strip())} characters). "
            "Please provide more substantial notes."
        )
    # Check that it contains real words (not just numbers/symbols)
    real_words = re.findall(r"[A-Za-z]{2,}", text)
    if len(real_words) < 5:
        raise DocumentError(
            "The document does not appear to contain readable English text."
        )


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def load_document(
    source: str,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    pasted_text: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Unified entry point for loading a document.

    Parameters
    ----------
    source : str
        One of "pdf", "txt", "paste".
    file_bytes : bytes, optional
        Raw file content when source is "pdf" or "txt".
    filename : str, optional
        Original filename (used for display only).
    pasted_text : str, optional
        Raw text when source is "paste".

    Returns
    -------
    text : str
        Extracted and normalised text.
    doc_hash : str
        Short hash of the text for change detection.

    Raises
    ------
    DocumentError
        On any extraction or validation failure.
    """
    if source == "pdf":
        if not file_bytes:
            raise DocumentError("No PDF bytes provided.")
        text, _ = extract_text_from_pdf(file_bytes)
    elif source == "txt":
        if not file_bytes:
            raise DocumentError("No TXT bytes provided.")
        text = extract_text_from_txt(file_bytes)
    elif source == "paste":
        if not pasted_text:
            raise DocumentError("No text was pasted.")
        text = process_pasted_text(pasted_text)
    else:
        raise DocumentError(f"Unknown source type: {source!r}")

    validate_document(text)
    doc_hash = compute_hash(text)
    return text, doc_hash
