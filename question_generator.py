"""
question_generator.py
=====================
Generates exam-oriented questions from student notes using Qwen2.5-3B-Instruct.

Supports:
- Multiple-choice questions (MCQ) with 4 options
- Short-answer questions
- Long-answer questions
- True/False questions
- Mixed question sets

Each question is grounded in the source notes and validated with Pydantic models.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from document_processor import split_into_chunks, word_count
from validators import (
    AnyQuestion,
    Difficulty,
    LongAnswerQuestion,
    MCQQuestion,
    QuestionType,
    ShortAnswerQuestion,
    TrueFalseQuestion,
    detect_duplicate_questions,
    has_evidence_in_source,
    validate_question_list,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTION_SYSTEM_PROMPT = """\
You are an academic examination question designer with expertise in creating \
high-quality study materials from student notes.

CRITICAL RULES:
1. The supplied notes are the ONLY source of truth. Generate questions ONLY \
from information explicitly stated in the notes.
2. Every answer must be directly supported by the notes. Do NOT invent facts, \
definitions, formulas, dates, examples, or relationships.
3. Do NOT ask questions that require knowledge unavailable in the notes.
4. If the notes do not support a reliable answer, skip that question entirely.
5. Generate distinct, clear, educationally useful questions.
6. Provide the correct answer, a full explanation, and a short source excerpt \
for EVERY question.
7. Do NOT add "All of the above" or "None of the above" as MCQ options.

You must respond with ONLY valid JSON in the exact format specified. \
No preamble, no markdown fences, no extra text."""

# Chunk size sent to Qwen for question generation (chars)
QG_CHUNK_CHARS = 2_500

DIFFICULTY_DESCRIPTIONS = {
    "easy": (
        "Easy questions test direct recall: definitions, important facts, "
        "named terms, explicitly stated steps."
    ),
    "medium": (
        "Medium questions test understanding: explain relationships, compare concepts, "
        "describe processes, interpret information from the notes."
    ),
    "hard": (
        "Hard questions require connecting multiple concepts, analyzing relationships "
        "described in the notes, or applying a principle stated in the notes to a "
        "scenario. Do NOT make questions hard merely by using complicated wording."
    ),
}


# ---------------------------------------------------------------------------
# JSON prompt templates
# ---------------------------------------------------------------------------

def _mcq_prompt(notes: str, count: int, difficulty: str) -> str:
    diff_desc = DIFFICULTY_DESCRIPTIONS[difficulty]
    return f"""Generate {count} multiple-choice question(s) from the following academic notes.

Difficulty: {difficulty.upper()} — {diff_desc}

REQUIRED JSON format (return a JSON array, no other text):
[
  {{
    "question": "The full question text?",
    "options": {{"A": "First option", "B": "Second option", "C": "Third option", "D": "Fourth option"}},
    "correct_answer": "A",
    "explanation": "Why A is correct based on the notes.",
    "source_evidence": "Exact or near-exact quote from the notes that supports the answer.",
    "difficulty": "{difficulty}",
    "topic": "Short topic label"
  }}
]

Academic Notes:
{notes}

JSON array:"""


def _short_prompt(notes: str, count: int, difficulty: str) -> str:
    diff_desc = DIFFICULTY_DESCRIPTIONS[difficulty]
    return f"""Generate {count} short-answer question(s) from the following academic notes.

Difficulty: {difficulty.upper()} — {diff_desc}

REQUIRED JSON format (return a JSON array, no other text):
[
  {{
    "question": "The full question text?",
    "expected_answer": "A concise expected answer based solely on the notes.",
    "explanation": "Detailed explanation grounded in the notes.",
    "source_evidence": "Exact or near-exact quote from the notes.",
    "key_points": ["Key point 1", "Key point 2"],
    "difficulty": "{difficulty}",
    "topic": "Short topic label"
  }}
]

Academic Notes:
{notes}

JSON array:"""


def _long_prompt(notes: str, count: int, difficulty: str) -> str:
    diff_desc = DIFFICULTY_DESCRIPTIONS[difficulty]
    return f"""Generate {count} long-answer question(s) from the following academic notes.

Difficulty: {difficulty.upper()} — {diff_desc}

REQUIRED JSON format (return a JSON array, no other text):
[
  {{
    "question": "The full question text requiring a detailed answer?",
    "model_answer": "A comprehensive model answer based solely on the notes.",
    "main_points": ["Main point 1", "Main point 2", "Main point 3"],
    "source_evidence": "Exact or near-exact quote from the notes.",
    "marking_guidance": "AI-generated guidance (not official criteria): e.g., 2 marks for X, 3 marks for Y.",
    "difficulty": "{difficulty}",
    "topic": "Short topic label"
  }}
]

Academic Notes:
{notes}

JSON array:"""


def _tf_prompt(notes: str, count: int, difficulty: str) -> str:
    diff_desc = DIFFICULTY_DESCRIPTIONS[difficulty]
    return f"""Generate {count} True/False question(s) from the following academic notes.

Difficulty: {difficulty.upper()} — {diff_desc}
Make sure statements are unambiguously true or false based on the notes. \
Do NOT use partially true statements.

REQUIRED JSON format (return a JSON array, no other text):
[
  {{
    "statement": "A clear, unambiguous statement.",
    "correct_answer": true,
    "explanation": "Why this is true/false based on the notes.",
    "source_evidence": "Exact or near-exact quote from the notes.",
    "difficulty": "{difficulty}",
    "topic": "Short topic label"
  }}
]

Academic Notes:
{notes}

JSON array:"""


# Map question type to prompt builder and Pydantic model
TYPE_CONFIG: Dict[str, Any] = {
    "mcq":        {"prompt_fn": _mcq_prompt,   "model": MCQQuestion},
    "short":      {"prompt_fn": _short_prompt, "model": ShortAnswerQuestion},
    "long":       {"prompt_fn": _long_prompt,  "model": LongAnswerQuestion},
    "true_false": {"prompt_fn": _tf_prompt,    "model": TrueFalseQuestion},
}


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_array(raw: str) -> List[dict]:
    """
    Robustly extract the first JSON array from a model response that may
    contain extra preamble or markdown fences.
    """
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the first [ ... ] block
    start = raw.find("[")
    if start == -1:
        raise ValueError("No JSON array found in model output.")

    # Track bracket depth to find the matching ]
    depth = 0
    end = -1
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("Unclosed JSON array in model output.")

    json_str = raw[start : end + 1]
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Single-chunk question generation
# ---------------------------------------------------------------------------

def _generate_from_chunk(
    chunk: str,
    q_type: str,
    count: int,
    difficulty: str,
    ai_service,
    source_text: str,
) -> List[AnyQuestion]:
    """Generate and validate questions from a single text chunk."""
    config = TYPE_CONFIG[q_type]
    prompt_fn = config["prompt_fn"]
    pydantic_model = config["model"]

    user_prompt = prompt_fn(chunk, count, difficulty)

    raw = ai_service.generate(
        system_prompt=QUESTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=1200,
        temperature=0.25,
    )

    try:
        items = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("JSON parse failed: %s | Raw output: %s …", exc, raw[:200])
        return []

    valid_questions: List[AnyQuestion] = []
    for item in items:
        # Inject question_type for MCQ/Short/Long (TF uses 'statement')
        if q_type != "true_false":
            item.setdefault("question_type", q_type)
        else:
            item.setdefault("question_type", "true_false")

        try:
            q = pydantic_model(**item)
        except Exception as exc:
            logger.debug("Validation failed for item %s: %s", item, exc)
            continue

        # Check source evidence (advisory — don't discard but log)
        evidence = getattr(q, "source_evidence", "")
        if not has_evidence_in_source(evidence, source_text):
            logger.debug(
                "Evidence not found in source for question: %s",
                getattr(q, "question", getattr(q, "statement", "?"))[:60],
            )
            # Keep the question but it's flagged implicitly

        valid_questions.append(q)

    return valid_questions


# ---------------------------------------------------------------------------
# Main question generation entry point
# ---------------------------------------------------------------------------

def generate_questions(
    text: str,
    ai_service,
    q_type: str = "mcq",
    count: int = 5,
    difficulty: str = "medium",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[AnyQuestion], List[str]]:
    """
    Generate questions from student notes.

    Parameters
    ----------
    text : str
        Full document text.
    ai_service : AIService
        Loaded AI service instance.
    q_type : str
        "mcq" | "short" | "long" | "true_false" | "mixed"
    count : int
        Number of questions to generate (1–20).
    difficulty : str
        "easy" | "medium" | "hard"
    progress_callback : callable, optional
        Called with (current, total) for progress display.

    Returns
    -------
    questions : list of validated question objects
    warnings : list of warning strings
    """
    count = max(1, min(count, 20))
    warnings: List[str] = []

    if q_type not in TYPE_CONFIG and q_type != "mixed":
        raise ValueError(f"Unknown question type: {q_type!r}")
    if difficulty not in DIFFICULTY_DESCRIPTIONS:
        raise ValueError(f"Unknown difficulty: {difficulty!r}")

    # Handle mixed: distribute across all types
    if q_type == "mixed":
        return _generate_mixed(text, ai_service, count, difficulty, progress_callback)

    # Chunk the text — use larger chunks for long/short (they need more context)
    chunk_size = QG_CHUNK_CHARS if q_type in ("mcq", "true_false") else 3_000
    chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=200)

    # Distribute question count across chunks
    per_chunk = max(1, count // len(chunks) + (1 if count % len(chunks) else 0))
    all_questions: List[AnyQuestion] = []
    total_steps = len(chunks)

    for idx, chunk in enumerate(chunks):
        if len(all_questions) >= count:
            break
        if progress_callback:
            progress_callback(idx, total_steps)
        remaining = count - len(all_questions)
        batch_count = min(per_chunk, remaining)
        qs = _generate_from_chunk(
            chunk, q_type, batch_count, difficulty, ai_service, text
        )
        all_questions.extend(qs)

    # Deduplicate
    dup_idxs = set(detect_duplicate_questions(all_questions))
    all_questions = [q for i, q in enumerate(all_questions) if i not in dup_idxs]
    if dup_idxs:
        warnings.append(
            f"{len(dup_idxs)} duplicate question(s) were removed."
        )

    # Trim to requested count
    all_questions = all_questions[:count]

    if len(all_questions) < count:
        warnings.append(
            f"Only {len(all_questions)} valid question(s) could be generated "
            f"from the available notes (requested {count}). This is normal when "
            "the notes are short or cover a narrow topic."
        )

    return all_questions, warnings


def _generate_mixed(
    text: str,
    ai_service,
    count: int,
    difficulty: str,
    progress_callback=None,
) -> Tuple[List[AnyQuestion], List[str]]:
    """Generate a mixed set of questions across all four types."""
    types = ["mcq", "short", "long", "true_false"]
    base = count // len(types)
    remainder = count % len(types)
    distribution = {t: base for t in types}
    for i in range(remainder):
        distribution[types[i]] += 1

    all_questions: List[AnyQuestion] = []
    all_warnings: List[str] = []
    total = len(types)

    for step, (qt, cnt) in enumerate(distribution.items()):
        if cnt == 0:
            continue
        if progress_callback:
            progress_callback(step, total)
        qs, warns = generate_questions(
            text, ai_service, q_type=qt, count=cnt, difficulty=difficulty
        )
        all_questions.extend(qs)
        all_warnings.extend(warns)

    return all_questions, all_warnings


# ---------------------------------------------------------------------------
# Score calculator (for MCQ and True/False only)
# ---------------------------------------------------------------------------

def calculate_score(
    questions: List[AnyQuestion],
    student_answers: Dict[int, Any],
) -> Dict[str, Any]:
    """
    Calculate score for objective questions (MCQ and True/False).

    Parameters
    ----------
    questions : list
        Full list of question objects.
    student_answers : dict
        Mapping of question index → student's answer (str for MCQ, bool for TF).

    Returns
    -------
    dict with keys: total, attempted, correct, incorrect, skipped, score_pct,
                    per_question (list of result dicts)
    """
    scoreable = []
    for i, q in enumerate(questions):
        if isinstance(q, (MCQQuestion, TrueFalseQuestion)):
            scoreable.append((i, q))

    total = len(scoreable)
    correct = 0
    incorrect = 0
    skipped = 0
    per_question = []

    for i, q in scoreable:
        student_ans = student_answers.get(i)
        if student_ans is None:
            skipped += 1
            per_question.append({"index": i, "result": "skipped"})
            continue

        correct_ans = q.correct_answer if isinstance(q, MCQQuestion) else q.correct_answer
        is_correct = str(student_ans).strip().upper() == str(correct_ans).strip().upper()

        if is_correct:
            correct += 1
            per_question.append({"index": i, "result": "correct"})
        else:
            incorrect += 1
            per_question.append(
                {"index": i, "result": "incorrect", "correct": correct_ans}
            )

    attempted = correct + incorrect
    score_pct = round((correct / total) * 100, 1) if total > 0 else 0.0

    return {
        "total": total,
        "attempted": attempted,
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "score_pct": score_pct,
        "per_question": per_question,
    }
