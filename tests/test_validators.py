"""
tests/test_validators.py
=========================
Tests for Pydantic models and validation helpers in validators.py.
No AI inference required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pydantic import ValidationError

from validators import (
    Difficulty,
    LongAnswerQuestion,
    MCQQuestion,
    QuestionType,
    ShortAnswerQuestion,
    TrueFalseQuestion,
    SummaryOutput,
    detect_duplicate_questions,
    has_evidence_in_source,
    validate_question_list,
)


# ── Shared test data ──────────────────────────────────────────────────────────

def make_mcq(**overrides) -> dict:
    base = {
        "question": "What is the capital city of France, as stated in the notes?",
        "options": {"A": "Berlin", "B": "Paris", "C": "Madrid", "D": "Rome"},
        "correct_answer": "B",
        "explanation": "The notes state that Paris is the capital of France.",
        "source_evidence": "Paris is the capital city of France.",
        "difficulty": "easy",
        "topic": "Geography",
    }
    base.update(overrides)
    return base


def make_short(**overrides) -> dict:
    base = {
        "question": "Define the term 'algorithm' as described in the notes.",
        "expected_answer": "A step-by-step procedure for solving a problem.",
        "explanation": "The notes define algorithm as a step-by-step procedure.",
        "source_evidence": "An algorithm is a step-by-step procedure for solving a problem.",
        "key_points": ["Step-by-step", "Problem-solving procedure"],
        "difficulty": "easy",
        "topic": "CS Fundamentals",
    }
    base.update(overrides)
    return base


def make_long(**overrides) -> dict:
    base = {
        "question": "Explain the process of photosynthesis as described in the notes.",
        "model_answer": "Photosynthesis is the process by which plants convert sunlight into glucose.",
        "main_points": ["Sunlight absorbed by chlorophyll", "CO2 and water as inputs", "Glucose as output"],
        "source_evidence": "Plants use sunlight to convert CO2 and water into glucose.",
        "marking_guidance": "2 marks for inputs, 2 marks for output, 1 mark for chlorophyll.",
        "difficulty": "medium",
        "topic": "Biology",
    }
    base.update(overrides)
    return base


def make_tf(**overrides) -> dict:
    base = {
        "statement": "Python is a compiled programming language.",
        "correct_answer": False,
        "explanation": "Python is an interpreted language, not compiled.",
        "source_evidence": "Python is interpreted at runtime.",
        "difficulty": "easy",
        "topic": "Python",
    }
    base.update(overrides)
    return base


# ── MCQQuestion ───────────────────────────────────────────────────────────────

class TestMCQQuestion:
    def test_valid_mcq(self):
        q = MCQQuestion(**make_mcq())
        assert q.correct_answer == "B"
        assert len(q.options) == 4

    def test_missing_option_raises(self):
        data = make_mcq()
        del data["options"]["D"]
        with pytest.raises(ValidationError, match="A, B, C, D"):
            MCQQuestion(**data)

    def test_duplicate_options_raises(self):
        data = make_mcq()
        data["options"]["D"] = data["options"]["A"]  # duplicate
        with pytest.raises(ValidationError, match="distinct"):
            MCQQuestion(**data)

    def test_invalid_correct_answer_key(self):
        data = make_mcq(correct_answer="E")
        with pytest.raises(ValidationError):
            MCQQuestion(**data)

    def test_correct_answer_not_in_options(self):
        data = make_mcq(correct_answer="A")
        data["options"] = {"B": "Berlin", "C": "Paris", "D": "Madrid", "A": "Nope"}
        # This is actually valid since A is in options
        q = MCQQuestion(**data)
        assert q.correct_answer == "A"

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError):
            MCQQuestion(**make_mcq(question=""))

    def test_too_short_question_raises(self):
        with pytest.raises(ValidationError):
            MCQQuestion(**make_mcq(question="Hi?"))

    def test_empty_explanation_raises(self):
        with pytest.raises(ValidationError):
            MCQQuestion(**make_mcq(explanation=""))

    def test_empty_evidence_raises(self):
        with pytest.raises(ValidationError):
            MCQQuestion(**make_mcq(source_evidence=""))

    def test_correct_answer_uppercased(self):
        q = MCQQuestion(**make_mcq(correct_answer="b"))
        assert q.correct_answer == "B"

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            MCQQuestion(**make_mcq(difficulty="extreme"))

    def test_all_difficulties(self):
        for d in ["easy", "medium", "hard"]:
            q = MCQQuestion(**make_mcq(difficulty=d))
            assert q.difficulty.value == d


# ── ShortAnswerQuestion ───────────────────────────────────────────────────────

class TestShortAnswerQuestion:
    def test_valid_short(self):
        q = ShortAnswerQuestion(**make_short())
        assert q.question_type == QuestionType.SHORT

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError):
            ShortAnswerQuestion(**make_short(question=""))

    def test_empty_answer_raises(self):
        with pytest.raises(ValidationError):
            ShortAnswerQuestion(**make_short(expected_answer=""))

    def test_key_points_optional(self):
        data = make_short()
        del data["key_points"]
        q = ShortAnswerQuestion(**data)
        assert q.key_points == []


# ── LongAnswerQuestion ────────────────────────────────────────────────────────

class TestLongAnswerQuestion:
    def test_valid_long(self):
        q = LongAnswerQuestion(**make_long())
        assert q.question_type == QuestionType.LONG
        assert len(q.main_points) >= 1

    def test_empty_main_points_raises(self):
        with pytest.raises(ValidationError):
            LongAnswerQuestion(**make_long(main_points=[]))

    def test_whitespace_only_points_filtered(self):
        # Mixed whitespace and real points: whitespace stripped, real points kept
        q = LongAnswerQuestion(**make_long(main_points=["  ", "Real point"]))
        assert "Real point" in q.main_points
        assert len(q.main_points) == 1  # blank point removed

    def test_all_whitespace_points_raises(self):
        # All blank main_points should raise ValidationError
        with pytest.raises(ValidationError):
            LongAnswerQuestion(**make_long(main_points=["   ", "  "]))


    def test_marking_guidance_optional(self):
        data = make_long()
        data["marking_guidance"] = None
        q = LongAnswerQuestion(**data)
        assert q.marking_guidance is None


# ── TrueFalseQuestion ─────────────────────────────────────────────────────────

class TestTrueFalseQuestion:
    def test_valid_true(self):
        q = TrueFalseQuestion(**make_tf(correct_answer=True))
        assert q.correct_answer is True

    def test_valid_false(self):
        q = TrueFalseQuestion(**make_tf(correct_answer=False))
        assert q.correct_answer is False

    def test_empty_statement_raises(self):
        with pytest.raises(ValidationError):
            TrueFalseQuestion(**make_tf(statement=""))

    def test_empty_explanation_raises(self):
        with pytest.raises(ValidationError):
            TrueFalseQuestion(**make_tf(explanation=""))


# ── SummaryOutput ─────────────────────────────────────────────────────────────

class TestSummaryOutput:
    def test_valid_summary(self):
        s = SummaryOutput(
            title="Test Notes",
            summary_text="This is a valid summary of the test notes.",
        )
        assert s.title == "Test Notes"

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError):
            SummaryOutput(title="", summary_text="Some text.")

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            SummaryOutput(title="Title", summary_text="")


# ── has_evidence_in_source ────────────────────────────────────────────────────

class TestHasEvidenceInSource:
    SOURCE = (
        "Data structures are used to organise and store data efficiently. "
        "A stack is a LIFO (Last In, First Out) data structure. "
        "A queue is a FIFO (First In, First Out) data structure."
    )

    def test_exact_match(self):
        assert has_evidence_in_source("A stack is a LIFO", self.SOURCE)

    def test_case_insensitive(self):
        assert has_evidence_in_source("a stack is a lifo", self.SOURCE)

    def test_not_present(self):
        assert not has_evidence_in_source("Binary trees were invented in 1960", self.SOURCE)

    def test_empty_evidence(self):
        assert not has_evidence_in_source("", self.SOURCE)

    def test_empty_source(self):
        assert not has_evidence_in_source("some evidence", "")


# ── detect_duplicate_questions ────────────────────────────────────────────────

class TestDetectDuplicates:
    def test_no_duplicates(self):
        qs = [
            MCQQuestion(**make_mcq(question="What is the first unique question about A?")),
            MCQQuestion(**make_mcq(question="What is the second unique question about B?")),
        ]
        assert detect_duplicate_questions(qs) == []

    def test_exact_duplicate(self):
        q1 = MCQQuestion(**make_mcq())
        q2 = MCQQuestion(**make_mcq())
        result = detect_duplicate_questions([q1, q2])
        assert 1 in result  # second occurrence flagged

    def test_case_insensitive_duplicate(self):
        q1 = MCQQuestion(**make_mcq(question="What is the capital city of France, as stated in the notes?"))
        q2 = MCQQuestion(**make_mcq(question="WHAT IS THE CAPITAL CITY OF FRANCE, AS STATED IN THE NOTES?"))
        result = detect_duplicate_questions([q1, q2])
        assert 1 in result

    def test_tf_duplicates(self):
        q1 = TrueFalseQuestion(**make_tf())
        q2 = TrueFalseQuestion(**make_tf())
        result = detect_duplicate_questions([q1, q2])
        assert 1 in result

    def test_first_occurrence_kept(self):
        q1 = MCQQuestion(**make_mcq())
        q2 = MCQQuestion(**make_mcq())
        dups = detect_duplicate_questions([q1, q2])
        assert 0 not in dups   # first kept
        assert 1 in dups       # second removed


# ── validate_question_list ────────────────────────────────────────────────────

class TestValidateQuestionList:
    def test_removes_duplicates(self):
        q1 = MCQQuestion(**make_mcq())
        q2 = MCQQuestion(**make_mcq())
        q3 = MCQQuestion(**make_mcq(
            question="A completely different question about data structures?",
            correct_answer="A",
            options={"A": "Array", "B": "Stack", "C": "Queue", "D": "Tree"},
        ))
        result = validate_question_list([q1, q2, q3], source_text="")
        assert len(result) == 2

    def test_preserves_order(self):
        q1 = MCQQuestion(**make_mcq(question="First unique question about topic X in our notes?"))
        q2 = MCQQuestion(**make_mcq(
            question="Second unique question about topic Y here?",
            correct_answer="C",
            options={"A": "One", "B": "Two", "C": "Three", "D": "Four"},
        ))
        result = validate_question_list([q1, q2], source_text="")
        assert result[0].question == q1.question
        assert result[1].question == q2.question
