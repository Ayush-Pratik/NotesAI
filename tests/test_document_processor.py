"""
tests/test_document_processor.py
=================================
Tests for document extraction, validation, chunking, and normalisation.
AI inference is NOT required — these tests run fully offline.
"""

import io
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_processor import (
    DocumentError,
    normalise_text,
    word_count,
    compute_hash,
    extract_text_from_txt,
    process_pasted_text,
    split_into_chunks,
    validate_document,
    load_document,
    CHUNK_CHAR_SIZE,
    MIN_MEANINGFUL_CHARS,
)


# ── normalise_text ───────────────────────────────────────────────────────────

class TestNormaliseText:
    def test_collapses_multiple_blank_lines(self):
        text = "Paragraph one.\n\n\n\nParagraph two."
        result = normalise_text(text)
        assert "\n\n\n" not in result
        assert "Paragraph one." in result
        assert "Paragraph two." in result

    def test_strips_trailing_whitespace_per_line(self):
        text = "Line one   \nLine two  "
        result = normalise_text(text)
        for line in result.splitlines():
            assert line == line.rstrip()

    def test_replaces_smart_quotes(self):
        text = "\u2018Hello\u2019 and \u201cWorld\u201d"
        result = normalise_text(text)
        assert "'" in result
        assert '"' in result
        assert "\u2018" not in result
        assert "\u201c" not in result

    def test_removes_bom(self):
        text = "\ufeffStart of file"
        result = normalise_text(text)
        assert "\ufeff" not in result

    def test_empty_string(self):
        assert normalise_text("") == ""


# ── word_count ───────────────────────────────────────────────────────────────

class TestWordCount:
    def test_simple_sentence(self):
        assert word_count("The quick brown fox") == 4

    def test_empty_string(self):
        assert word_count("") == 0

    def test_multiple_spaces(self):
        # split() handles multiple spaces correctly
        assert word_count("one  two   three") == 3


# ── compute_hash ─────────────────────────────────────────────────────────────

class TestComputeHash:
    def test_same_text_same_hash(self):
        assert compute_hash("hello") == compute_hash("hello")

    def test_different_text_different_hash(self):
        assert compute_hash("hello") != compute_hash("world")

    def test_returns_16_chars(self):
        assert len(compute_hash("test")) == 16


# ── extract_text_from_txt ────────────────────────────────────────────────────

class TestExtractTextFromTxt:
    def test_utf8_text(self):
        text = "This is a test document with sufficient content for validation.\nIt has multiple lines of text."
        result = extract_text_from_txt(text.encode("utf-8"))
        assert "test document" in result

    def test_utf8_bom(self):
        text = "BOM encoded file content here for testing purposes. This is long enough."
        result = extract_text_from_txt(text.encode("utf-8-sig"))
        assert "BOM encoded" in result

    def test_latin1_fallback(self):
        text = "Caf\xe9 au lait is a popular French drink with long history in France."
        result = extract_text_from_txt(text.encode("latin-1"))
        assert "popular" in result

    def test_empty_file_raises(self):
        with pytest.raises(DocumentError, match="empty"):
            extract_text_from_txt(b"")

    def test_very_short_raises(self):
        with pytest.raises(DocumentError):
            extract_text_from_txt("hi".encode("utf-8"))


# ── process_pasted_text ──────────────────────────────────────────────────────

class TestProcessPastedText:
    def test_valid_text(self):
        text = "A" * 100 + " some real words here for testing."
        result = process_pasted_text(text)
        assert len(result) >= MIN_MEANINGFUL_CHARS

    def test_too_short_raises(self):
        with pytest.raises(DocumentError, match="too short"):
            process_pasted_text("Short text.")

    def test_too_long_raises(self):
        long_text = "word " * 35_000  # ~175 000 chars
        with pytest.raises(DocumentError, match="too long"):
            process_pasted_text(long_text)

    def test_normalises_whitespace(self):
        text = "Line one.   \n\n\n\nLine two.   " + " extra words" * 5
        result = process_pasted_text(text)
        assert "\n\n\n" not in result


# ── split_into_chunks ────────────────────────────────────────────────────────

class TestSplitIntoChunks:
    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = split_into_chunks(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        assert split_into_chunks("") == []

    def test_chunks_ordered(self):
        """Chunks must preserve document order."""
        paragraphs = [f"Paragraph {i}. " + "A" * 200 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = split_into_chunks(text, chunk_size=800, overlap=0)
        # Reconstruct and check order
        combined = " ".join(chunks)
        last_seen = -1
        for i in range(10):
            pos = combined.find(f"Paragraph {i}")
            assert pos > last_seen, f"Chunk order violated at paragraph {i}"
            last_seen = pos

    def test_chunk_size_respected(self):
        # Use paragraph-separated text so the chunker has split points
        paragraphs = ["Sentence about topic number " + str(i) + ". " * 20 for i in range(30)]
        text = "\n\n".join(paragraphs)
        chunks = split_into_chunks(text, chunk_size=500, overlap=0)
        # Each chunk should be reasonably sized (allow ~2x for boundary tolerance)
        for chunk in chunks:
            assert len(chunk) <= 1500, f"Chunk too large: {len(chunk)}"

    def test_no_empty_chunks(self):
        text = "\n\n".join(["Sentence " + str(i) + " content." * 20 for i in range(20)])
        chunks = split_into_chunks(text, chunk_size=500)
        for chunk in chunks:
            assert chunk.strip(), "Empty chunk found"


# ── validate_document ────────────────────────────────────────────────────────

class TestValidateDocument:
    def test_valid_document(self):
        validate_document("This is a proper document with enough content and words.")

    def test_empty_raises(self):
        with pytest.raises(DocumentError):
            validate_document("")

    def test_whitespace_only_raises(self):
        with pytest.raises(DocumentError):
            validate_document("   \n   \t  ")

    def test_numbers_only_raises(self):
        with pytest.raises(DocumentError):
            validate_document("1 2 3 4 5 6 7 8 9 0" * 5)

    def test_too_short_raises(self):
        with pytest.raises(DocumentError):
            validate_document("Hi.")


# ── load_document dispatcher ─────────────────────────────────────────────────

class TestLoadDocument:
    def test_txt_source(self):
        content = "This is a test document for the loading function. " * 5
        text, doc_hash = load_document("txt", file_bytes=content.encode("utf-8"))
        assert "test document" in text
        assert len(doc_hash) == 16

    def test_paste_source(self):
        content = "A detailed set of notes about machine learning. " * 5
        text, doc_hash = load_document("paste", pasted_text=content)
        assert "machine learning" in text

    def test_unknown_source_raises(self):
        with pytest.raises(DocumentError):
            load_document("unknown")

    def test_missing_bytes_raises(self):
        with pytest.raises(DocumentError):
            load_document("pdf")

    def test_hash_changes_with_content(self):
        content1 = "First document content. " * 5
        content2 = "Second document content. " * 5
        _, hash1 = load_document("txt", file_bytes=content1.encode("utf-8"))
        _, hash2 = load_document("txt", file_bytes=content2.encode("utf-8"))
        assert hash1 != hash2

    def test_pdf_source_mock(self):
        """Test PDF source with a mocked PyMuPDF call."""
        mock_text = "Mocked PDF content with enough words for validation. " * 3

        with patch("document_processor.extract_text_from_pdf") as mock_pdf:
            mock_pdf.return_value = (mock_text, [0])
            text, doc_hash = load_document("pdf", file_bytes=b"fake_pdf")
            assert "Mocked PDF content" in text
            assert doc_hash
