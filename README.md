# NotesAI — AI-Powered Notes Summarizer and Question Generator

## Abstract

NotesAI is a locally-running AI application that transforms academic notes into
structured summaries and exam-ready questions. Built for college students, it
uses state-of-the-art natural language processing models to extract key concepts,
generate multiple question types, and provide an interactive practice environment —
all without cloud APIs or subscription fees.

---

## Problem Statement

Students accumulate large amounts of lecture notes and textbook extracts but
struggle to:
1. Identify the most important concepts efficiently.
2. Create effective practice questions for self-assessment.
3. Organize information for exam revision.

NotesAI solves all three problems using locally downloaded pretrained AI models.

---

## Objectives

1. Summarize academic notes into structured, exam-oriented formats.
2. Generate diverse questions (MCQ, short-answer, long-answer, True/False) grounded in the notes.
3. Provide interactive self-assessment with scoring.
4. Export study materials as TXT and PDF.
5. Run fully offline after the initial model download, on consumer hardware.

---

## Main Features

| Feature | Description |
|---------|-------------|
| 📄 PDF & TXT Upload | Extract text from PDF and TXT files |
| ✏️ Paste Notes | Directly paste lecture notes |
| 📋 Smart Summarization | Short / Medium / Detailed summaries in multiple formats |
| ❓ Question Generation | MCQ, Short-answer, Long-answer, True/False, Mixed |
| 🎯 Practice Mode | Interactive quiz with scoring and feedback |
| 📥 PDF & TXT Export | Download summaries and question sets |
| ⚙️ Local AI | All inference runs on your computer — no API keys needed |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10 / 3.11 |
| UI Framework | Streamlit |
| Summarization Model | facebook/bart-large-cnn (~406M params) |
| Q&A Model | Qwen/Qwen2.5-1.5B-Instruct (~1.54B params) |
| ML Framework | HuggingFace Transformers + PyTorch (CUDA 12.4) |
| PDF Extraction | PyMuPDF (fitz) |
| PDF Generation | ReportLab |
| Output Validation | Pydantic v2 |
| Testing | pytest + pytest-mock |

---

## Hardware Requirements

| Specification | Minimum | Recommended |
|---------------|---------|-------------|
| RAM | 12 GB | 16 GB or more |
| CPU | Quad-core | Intel i5-13th gen or better |
| GPU | Optional | NVIDIA RTX 3050 (4–6 GB VRAM) |
| Storage | 10 GB free | 15 GB free |
| OS | Windows 10/11 | Windows 11 |
| Python | 3.10 | 3.11 |

> **Note:** GPU is optional. On CPU, summarization takes ~30–60 seconds per chunk;
> question generation takes ~2–5 minutes per batch. On GPU, this is significantly faster.

---

## AI Model Selection Rationale

### Summarization: `facebook/bart-large-cnn`

- **Why chosen:** Fine-tuned on CNN/DailyMail for abstractive summarization;
  well-documented; generalises reliably to academic text; fast inference; MIT license.
- **Parameters:** ~406 million
- **Download size:** ~1.6 GB
- **VRAM (inference):** ~3–4 GB on GPU, or CPU with ~2 GB RAM
- **Context window:** 1,024 tokens (long documents are chunked automatically)
- **License:** MIT

### Instruction-Following: `Qwen/Qwen2.5-1.5B-Instruct`

- **Why chosen:** Outstanding instruction-following and JSON compliance at the 1.5B parameter scale;
  fits directly on NVIDIA GeForce RTX 3050 (6GB VRAM) in `torch.float16` for real-time inference (2–4 seconds);
  Grouped-Query Attention (GQA) & Rotary Position Embeddings (RoPE) up to 32k tokens;
  permissive Qwen Research License.
- **Parameters:** ~1.54 billion
- **Download size:** ~3.0 GB (FP16)
- **VRAM (inference):** ~3.0 GB on GPU with Tensor Core acceleration.
- **License:** Qwen Research License

### How to change models

Edit the two constants at the top of `ai_service.py`:

```python
MODEL_SUMMARIZER = "facebook/bart-large-cnn"
MODEL_INSTRUCT   = "Qwen/Qwen2.5-1.5B-Instruct"
```

---

## Project Structure

```
NotesAI/
│
├── app.py                   Main Streamlit application, all UI and navigation
├── ai_service.py            Hardware detection, model loading, inference facade
├── summarizer.py            BART summarization + Qwen formatting pipeline
├── question_generator.py    Question generation, JSON parsing, scoring
├── document_processor.py    PDF/TXT extraction, chunking, normalisation
├── validators.py            Pydantic models for all question types
├── export_service.py        TXT and PDF report generation via ReportLab
├── download_models.py       One-time model download script
├── requirements.txt         Python dependencies
├── README.md                This file
│
├── .streamlit/
│   └── config.toml          Streamlit theme configuration
│
└── tests/
    ├── __init__.py
    ├── test_document_processor.py   Tests for text extraction and chunking
    ├── test_validators.py           Tests for Pydantic models and helpers
    └── test_text_processing.py     Tests for processing, scoring, exports
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `app.py` | Streamlit pages, session state, navigation, download buttons |
| `ai_service.py` | Detects GPU/CPU, loads BART and Qwen once, handles OOM fallback |
| `summarizer.py` | Chunks text, calls BART per chunk, consolidates, calls Qwen for formatting |
| `question_generator.py` | Builds JSON prompts, calls Qwen, parses JSON, validates with Pydantic |
| `document_processor.py` | Extracts text from PDF/TXT/paste, normalises whitespace, splits into chunks |
| `validators.py` | Pydantic v2 data models; MCQ, Short, Long, True/False; dedup helpers |
| `export_service.py` | Generates readable TXT and PDF reports using ReportLab |

---

## Installation Instructions (Windows)

### Step 1 — Install Python

Download Python 3.11 from https://www.python.org/downloads/
During installation, check **"Add Python to PATH"**.

Verify:
```cmd
python --version
```

### Step 2 — Create a virtual environment

Open Command Prompt in the `NotesAI` folder:

```cmd
cd C:\Users\YourName\Desktop\Notes_Summarizer
python -m venv venv
venv\Scripts\activate
```

Your prompt will show `(venv)` when the virtual environment is active.

### Step 3 — Install PyTorch (CUDA for RTX 3050)

Visit https://pytorch.org/get-started/locally/ to get the exact command.
For CUDA 12.1 (recommended for RTX 3050 with Windows):

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you prefer CPU-only (simpler, but slower):
```cmd
pip install torch torchvision torchaudio
```

### Step 4 — Install all other dependencies

```cmd
pip install -r requirements.txt
```

### Step 5 — Download AI models (requires internet, ~7–8 GB)

```cmd
python download_models.py
```

This downloads both models to `~/.cache/huggingface/hub/`.
After this step, the application works offline.

### Step 6 — Launch the application

```cmd
streamlit run app.py
```

The app opens automatically at: http://localhost:8501

---

## How to Run Tests

Make sure the virtual environment is active:

```cmd
venv\Scripts\activate
pip install pytest pytest-mock
pytest tests\ -v
```

Expected output: `110 passed, 0 failed in 0.22s` (all automated tests run fully offline without requiring GPU hardware or active AI model weights).

---

## Technical Documentation Report

A complete Microsoft Word technical project report has been generated:
- File: `NotesAI_Complete_Project_Report.docx`
- Includes: Executive summary, system architecture, model mechanics (BART & Qwen), mathematical equations (RoPE, SwiGLU, cross-attention), engineering challenges solved (CUDA assertions, memory thrashing), and free deployment methods (Cloudflare Tunnel & Hugging Face Spaces).

---

## Example Workflow

1. Launch the app: `streamlit run app.py`
2. Click **Summarize Notes** in the sidebar.
3. Upload a PDF or paste your lecture notes.
4. Choose **Medium** length and **Exam Revision Notes** format.
5. Click **Generate Summary** and wait for the AI to process your notes.
6. Review the summary, key concepts, and definitions.
7. Download the summary as PDF.
8. Click **Generate Questions** in the sidebar.
9. Select **MCQ**, **5 questions**, **Medium** difficulty.
10. Click **Generate Questions**.
11. Click **Practice Mode** to attempt the questions interactively.
12. Submit your answers to see your score and explanations.

---

## Known Limitations

1. **First run requires internet** to download models (~7–8 GB).
2. **CPU inference is slow** — summarizing a 10-page document may take 3–5 minutes; questions 2–5 minutes per batch.
3. **BART context window** is 1,024 tokens. Documents are chunked automatically, but very dense technical documents may lose some nuance across chunk boundaries.
4. **JSON output parsing** — Qwen may occasionally produce malformed JSON; such questions are automatically discarded and a warning is shown.
5. **Question count** — fewer valid questions may be returned than requested for very short notes.
6. **Long-answer scoring** — long and short answer responses are not auto-scored (model comparison is provided instead).
7. **GPU VRAM** — the RTX 3050's 4–6 GB VRAM may only be sufficient for BART on GPU; Qwen may fall back to CPU automatically.
8. **Language** — optimized for English-language notes.

---

## Future Improvements

- OCR support for scanned PDFs (Tesseract integration).
- Topic-based question filtering.
- Export to DOCX format.
- Support for multiple documents in a session.
- Flashcard generation mode.
- Spaced-repetition tracking.
- Fine-tuning on domain-specific academic datasets.
- Support for larger Qwen models (7B) when hardware permits.

---

## Disclaimer

This project uses locally downloaded pretrained Hugging Face models
(`facebook/bart-large-cnn` and `Qwen/Qwen2.5-3B-Instruct`).
No custom model training was performed.
The models are used with carefully designed prompts that instruct them to
ground all generated content in the student's notes.
AI-generated content should always be verified against the original source material.
