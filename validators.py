"""
validators.py
=============
Pydantic models for structured output validation of AI-generated content.
Validates questions, answers, MCQ options, and summaries to ensure that
nothing malformed reaches the student UI.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QuestionType(str, Enum):
    MCQ = "mcq"
    SHORT = "short"
    LONG = "long"
    TRUE_FALSE = "true_false"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# MCQ model
# ---------------------------------------------------------------------------

class MCQQuestion(BaseModel):
    question_type: QuestionType = QuestionType.MCQ
    question: str
    options: Dict[str, str]          # e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str              # One of "A", "B", "C", "D"
    explanation: str
    source_evidence: str
    difficulty: Difficulty
    topic: Optional[str] = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question text must not be empty.")
        if len(v) < 10:
            raise ValueError("Question text is too short.")
        return v

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: Dict[str, str]) -> Dict[str, str]:
        required_keys = {"A", "B", "C", "D"}
        if set(v.keys()) != required_keys:
            raise ValueError(f"MCQ must have exactly options A, B, C, D. Got: {set(v.keys())}")
        # Check all options are non-empty and distinct
        values = [opt.strip() for opt in v.values()]
        if any(not val for val in values):
            raise ValueError("All MCQ option values must be non-empty.")
        if len(set(values)) != 4:
            raise ValueError("All four MCQ options must be distinct.")
        return {k: v_item.strip() for k, v_item in v.items()}

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in {"A", "B", "C", "D"}:
            raise ValueError(f"correct_answer must be A, B, C, or D. Got: {v!r}")
        return v

    @field_validator("explanation")
    @classmethod
    def explanation_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Explanation must not be empty.")
        return v

    @field_validator("source_evidence")
    @classmethod
    def evidence_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Source evidence must not be empty.")
        return v

    @model_validator(mode="after")
    def correct_answer_in_options(self) -> "MCQQuestion":
        if self.correct_answer not in self.options:
            raise ValueError(
                f"correct_answer '{self.correct_answer}' is not a key in options."
            )
        return self


# ---------------------------------------------------------------------------
# Short-answer model
# ---------------------------------------------------------------------------

class ShortAnswerQuestion(BaseModel):
    question_type: QuestionType = QuestionType.SHORT
    question: str
    expected_answer: str
    explanation: str
    source_evidence: str
    key_points: List[str] = []
    difficulty: Difficulty
    topic: Optional[str] = None

    @field_validator("question", "expected_answer", "explanation", "source_evidence")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty.")
        return v


# ---------------------------------------------------------------------------
# Long-answer model
# ---------------------------------------------------------------------------

class LongAnswerQuestion(BaseModel):
    question_type: QuestionType = QuestionType.LONG
    question: str
    model_answer: str
    main_points: List[str]
    source_evidence: str
    marking_guidance: Optional[str] = None
    difficulty: Difficulty
    topic: Optional[str] = None

    @field_validator("question", "model_answer", "source_evidence")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty.")
        return v

    @field_validator("main_points")
    @classmethod
    def at_least_one_point(cls, v: List[str]) -> List[str]:
        cleaned = [p.strip() for p in v if p.strip()]
        if not cleaned:
            raise ValueError("Long-answer questions must include at least one main point.")
        return cleaned


# ---------------------------------------------------------------------------
# True/False model
# ---------------------------------------------------------------------------

class TrueFalseQuestion(BaseModel):
    question_type: QuestionType = QuestionType.TRUE_FALSE
    statement: str
    correct_answer: bool
    explanation: str
    source_evidence: str
    difficulty: Difficulty
    topic: Optional[str] = None

    @field_validator("statement", "explanation", "source_evidence")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty.")
        return v


# ---------------------------------------------------------------------------
# Union type alias
# ---------------------------------------------------------------------------

AnyQuestion = MCQQuestion | ShortAnswerQuestion | LongAnswerQuestion | TrueFalseQuestion


# ---------------------------------------------------------------------------
# Summary validation
# ---------------------------------------------------------------------------

class SummaryOutput(BaseModel):
    title: str
    summary_text: str
    key_concepts: List[str] = []
    important_definitions: List[str] = []
    word_count: int = 0

    @field_validator("title", "summary_text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field must not be empty.")
        return v


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def has_evidence_in_source(evidence: str, source_text: str) -> bool:
    """
    Check whether `evidence` (or a meaningful substring of it) appears in
    `source_text` after basic whitespace normalisation.

    Returns True if at least a 40-character substring of evidence is found.
    """
    if not evidence or not source_text:
        return False
    # Normalise whitespace for comparison
    norm_src = re.sub(r"\s+", " ", source_text).lower()
    norm_ev = re.sub(r"\s+", " ", evidence).lower().strip()
    if not norm_ev:
        return False
    # Try a sliding window of the first 60 chars of the evidence
    window = norm_ev[:60]
    return window in norm_src


def detect_duplicate_questions(questions: List[AnyQuestion]) -> List[int]:
    """
    Return indices of duplicate questions (case-insensitive text comparison).
    The first occurrence is kept; later duplicates are flagged.
    """
    seen: set[str] = set()
    duplicates: List[int] = []
    for idx, q in enumerate(questions):
        # Get question text regardless of type
        text = getattr(q, "question", None) or getattr(q, "statement", "")
        key = re.sub(r"\s+", " ", text).lower().strip()
        if key in seen:
            duplicates.append(idx)
        else:
            seen.add(key)
    return duplicates


def validate_question_list(
    questions: List[AnyQuestion],
    source_text: str,
    flag_missing_evidence: bool = True,
) -> List[AnyQuestion]:
    """
    Post-process a list of generated questions:
    1. Remove duplicates (keeping first occurrence).
    2. Optionally warn about questions whose evidence is not found in source.

    Returns the cleaned list.
    """
    dup_indices = set(detect_duplicate_questions(questions))
    clean: List[AnyQuestion] = []
    for idx, q in enumerate(questions):
        if idx in dup_indices:
            continue
        clean.append(q)
    return clean
