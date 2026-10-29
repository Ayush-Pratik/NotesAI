"""
tests/test_text_processing.py
==============================
Tests for text processing, question generation helpers, scoring,
MCQ validation, and export utilities.
AI inference is mocked — no model downloads required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from unittest.mock import MagicMock, patch

from document_processor import split_into_chunks, normalise_text, word_count
from question_generator import (
    _extract_json_array,
    calculate_score,
    _generate_from_chunk,
)
from validators import (
    MCQQuestion,
    ShortAnswerQuestion,
    TrueFalseQuestion,
    LongAnswerQuestion,
    detect_duplicate_questions,
)
from export_service import export_summary_txt, export_questions_txt, _safe


# ── _extract_json_array ───────────────────────────────────────────────────────

class TestExtractJsonArray:
    def test_plain_json(self):
        raw = '[{"key": "value"}, {"key": "value2"}]'
        result = _extract_json_array(raw)
        assert len(result) == 2
        assert result[0]["key"] == "value"

    def test_with_preamble(self):
        raw = 'Sure, here is the JSON:\n\n[{"a": 1}]'
        result = _extract_json_array(raw)
        assert result[0]["a"] == 1

    def test_with_markdown_fence(self):
        raw = "```json\n[{\"x\": 42}]\n```"
        result = _extract_json_array(raw)
        assert result[0]["x"] == 42

    def test_no_array_raises(self):
        with pytest.raises(ValueError, match="No JSON array"):
            _extract_json_array("This has no array at all.")

    def test_unclosed_array_raises(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _extract_json_array("[{unclosed}")

    def test_empty_array(self):
        result = _extract_json_array("[]")
        assert result == []

    def test_nested_objects(self):
        raw = '[{"options": {"A": "one", "B": "two"}}]'
        result = _extract_json_array(raw)
        assert result[0]["options"]["A"] == "one"


# ── calculate_score ───────────────────────────────────────────────────────────

class TestCalculateScore:
    def _make_mcq(self, correct="B"):
        return MCQQuestion(
            question="What is the capital of France per the notes?",
            options={"A": "Berlin", "B": "Paris", "C": "Madrid", "D": "Rome"},
            correct_answer=correct,
            explanation="Paris is stated as the capital.",
            source_evidence="Paris is the capital of France.",
            difficulty="easy",
        )

    def _make_tf(self, correct=True):
        return TrueFalseQuestion(
            statement="Python is an interpreted language.",
            correct_answer=correct,
            explanation="Stated in notes.",
            source_evidence="Python is interpreted at runtime.",
            difficulty="easy",
        )

    def test_all_correct(self):
        q1 = self._make_mcq("B")
        q2 = self._make_tf(True)
        score = calculate_score([q1, q2], {0: "B", 1: True})
        assert score["correct"] == 2
        assert score["score_pct"] == 100.0

    def test_all_wrong(self):
        q1 = self._make_mcq("B")
        score = calculate_score([q1], {0: "A"})
        assert score["correct"] == 0
        assert score["score_pct"] == 0.0

    def test_skipped_question(self):
        q1 = self._make_mcq("B")
        score = calculate_score([q1], {})
        assert score["skipped"] == 1
        assert score["attempted"] == 0

    def test_mixed_results(self):
        q1 = self._make_mcq("B")
        q2 = self._make_tf(True)
        q3 = self._make_mcq("A")
        score = calculate_score([q1, q2, q3], {0: "B", 1: False, 2: "A"})
        assert score["correct"] == 2
        assert score["incorrect"] == 1
        assert score["score_pct"] == pytest.approx(66.7, abs=0.2)

    def test_long_answer_not_scored(self):
        """Long and short answer questions should not be included in scoring."""
        long_q = LongAnswerQuestion(
            question="Explain photosynthesis in detail.",
            model_answer="Photosynthesis converts CO2 and water into glucose.",
            main_points=["CO2 input", "Water input", "Glucose output"],
            source_evidence="Plants convert CO2 and water into glucose via sunlight.",
            difficulty="medium",
        )
        mcq = self._make_mcq("B")
        score = calculate_score([long_q, mcq], {0: "some answer", 1: "B"})
        # Only the MCQ is scored
        assert score["total"] == 1
        assert score["correct"] == 1

    def test_zero_questions(self):
        score = calculate_score([], {})
        assert score["total"] == 0
        assert score["score_pct"] == 0.0

    def test_case_insensitive_answer(self):
        q1 = self._make_mcq("B")
        score = calculate_score([q1], {0: "b"})
        assert score["correct"] == 1


# ── _generate_from_chunk (mocked AI) ─────────────────────────────────────────

class TestGenerateFromChunk:
    def _make_ai_service(self, response: str) -> MagicMock:
        svc = MagicMock()
        svc.generate.return_value = response
        return svc

    def test_valid_mcq_returned(self):
        chunk = "The CPU (Central Processing Unit) is the brain of the computer."
        response = json.dumps([{
            "question": "What does CPU stand for as stated in the notes?",
            "options": {
                "A": "Central Processing Unit",
                "B": "Computer Power Unit",
                "C": "Core Processing Unit",
                "D": "Central Program Utility",
            },
            "correct_answer": "A",
            "explanation": "CPU stands for Central Processing Unit per the notes.",
            "source_evidence": "The CPU (Central Processing Unit) is the brain of the computer.",
            "difficulty": "easy",
            "topic": "Computer Hardware",
        }])
        svc = self._make_ai_service(response)
        questions = _generate_from_chunk(chunk, "mcq", 1, "easy", svc, chunk)
        assert len(questions) == 1
        assert isinstance(questions[0], MCQQuestion)
        assert questions[0].correct_answer == "A"

    def test_invalid_json_returns_empty(self):
        svc = self._make_ai_service("This is not valid JSON at all!!!")
        questions = _generate_from_chunk("some text", "mcq", 1, "easy", svc, "some text")
        assert questions == []

    def test_invalid_mcq_structure_filtered(self):
        """MCQ with only 3 options should be rejected by Pydantic."""
        response = json.dumps([{
            "question": "Which of these is correct as per the notes?",
            "options": {"A": "One", "B": "Two", "C": "Three"},  # Missing D
            "correct_answer": "A",
            "explanation": "Option A.",
            "source_evidence": "The notes say A.",
            "difficulty": "easy",
        }])
        svc = self._make_ai_service(response)
        questions = _generate_from_chunk("notes text", "mcq", 1, "easy", svc, "notes text")
        assert questions == []

    def test_short_answer_parsed(self):
        chunk = "Machine learning is a subset of artificial intelligence."
        response = json.dumps([{
            "question": "What is machine learning according to the notes?",
            "expected_answer": "A subset of artificial intelligence.",
            "explanation": "The notes define it as a subset of AI.",
            "source_evidence": "Machine learning is a subset of artificial intelligence.",
            "key_points": ["Subset of AI"],
            "difficulty": "easy",
            "topic": "AI",
        }])
        svc = self._make_ai_service(response)
        questions = _generate_from_chunk(chunk, "short", 1, "easy", svc, chunk)
        assert len(questions) == 1
        assert isinstance(questions[0], ShortAnswerQuestion)

    def test_true_false_parsed(self):
        chunk = "Water boils at 100 degrees Celsius at standard atmospheric pressure."
        response = json.dumps([{
            "statement": "Water boils at 100 degrees Celsius at standard pressure.",
            "correct_answer": True,
            "explanation": "Stated explicitly in the notes.",
            "source_evidence": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "difficulty": "easy",
            "topic": "Physics",
        }])
        svc = self._make_ai_service(response)
        questions = _generate_from_chunk(chunk, "true_false", 1, "easy", svc, chunk)
        assert len(questions) == 1
        assert isinstance(questions[0], TrueFalseQuestion)
        assert questions[0].correct_answer is True

    def test_empty_model_response(self):
        svc = self._make_ai_service("")
        questions = _generate_from_chunk("notes", "mcq", 1, "easy", svc, "notes")
        assert questions == []


# ── Text normalisation edge cases ─────────────────────────────────────────────

class TestNormaliseEdgeCases:
    def test_mixed_line_endings(self):
        text = "Line one\r\nLine two\rLine three"
        result = normalise_text(text)
        assert "\r" not in result

    def test_unicode_normalisation(self):
        # Composed vs decomposed form of é
        composed = "\u00e9"       # é (NFC)
        decomposed = "e\u0301"   # e + combining acute accent (NFD)
        r1 = normalise_text(composed)
        r2 = normalise_text(decomposed)
        assert r1 == r2  # both normalised to NFC

    def test_soft_hyphen_removed(self):
        text = "super\u00adconductor"
        result = normalise_text(text)
        assert "\u00ad" not in result


# ── Export service ────────────────────────────────────────────────────────────

class TestExportService:
    def _sample_summary(self):
        return {
            "title": "Test Notes",
            "summary_text": "This is a test summary of the notes.",
            "key_concepts": ["Concept A", "Concept B"],
            "important_definitions": ["Term X: definition of X."],
            "word_count": 10,
            "format_used": "structured_notes",
            "length_used": "medium",
            "chunk_count": 1,
        }

    def _sample_mcq(self):
        return MCQQuestion(
            question="What is the test question per the notes?",
            options={"A": "Answer A", "B": "Answer B", "C": "Answer C", "D": "Answer D"},
            correct_answer="A",
            explanation="Answer A is correct per the notes.",
            source_evidence="The notes state Answer A.",
            difficulty="easy",
            topic="Testing",
        )

    def test_summary_txt_contains_title(self):
        result = export_summary_txt(self._sample_summary(), "Test Notes")
        decoded = result.decode("utf-8")
        assert "Test Notes" in decoded

    def test_summary_txt_contains_summary(self):
        result = export_summary_txt(self._sample_summary(), "Test Notes")
        decoded = result.decode("utf-8")
        assert "test summary" in decoded

    def test_summary_txt_contains_concepts(self):
        result = export_summary_txt(self._sample_summary(), "Test Notes")
        decoded = result.decode("utf-8")
        assert "Concept A" in decoded

    def test_questions_txt_with_answers(self):
        result = export_questions_txt([self._sample_mcq()], "Test", include_answers=True)
        decoded = result.decode("utf-8")
        assert "Answer A is correct" in decoded
        assert "Correct Answer" in decoded

    def test_questions_txt_without_answers(self):
        result = export_questions_txt([self._sample_mcq()], "Test", include_answers=False)
        decoded = result.decode("utf-8")
        # Answers should not appear in practice mode
        assert "Correct Answer" not in decoded

    def test_safe_html_escaping(self):
        assert _safe("A & B") == "A &amp; B"
        assert _safe("<tag>") == "&lt;tag&gt;"
        assert _safe("a > b") == "a &gt; b"

    def test_questions_txt_returns_bytes(self):
        result = export_questions_txt([], "Empty", include_answers=True)
        assert isinstance(result, bytes)

    def test_summary_txt_returns_bytes(self):
        result = export_summary_txt(self._sample_summary(), "Test")
        assert isinstance(result, bytes)


# ── Source document change detection ─────────────────────────────────────────

class TestDocumentChangeDetection:
    """Verify that hash-based change detection works correctly."""

    def test_same_content_same_hash(self):
        from document_processor import compute_hash
        text = "The content of my notes about machine learning."
        assert compute_hash(text) == compute_hash(text)

    def test_different_content_different_hash(self):
        from document_processor import compute_hash
        text1 = "Content about machine learning and neural networks."
        text2 = "Content about deep learning and transformers."
        assert compute_hash(text1) != compute_hash(text2)

    def test_whitespace_change_changes_hash(self):
        from document_processor import compute_hash
        text1 = "Same content here."
        text2 = "Same content  here."  # extra space
        assert compute_hash(text1) != compute_hash(text2)
