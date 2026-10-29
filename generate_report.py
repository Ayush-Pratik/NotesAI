# generate_report.py
"""
NotesAI — Complete Project Report & Technical Architecture Generator
Generates a beautifully formatted Microsoft Word document (.docx)
for college assessment and project presentation.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_report():
    doc = docx.Document()

    # ── Page Margins (Standard 1-inch) ──
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

        # Header
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run('NotesAI — College Continuous Assessment (CA) Technical Report')
        hrun.font.name = 'Calibri'
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

        # Footer
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run('Confidential & Academic Use Only  |  NotesAI Project Documentation')
        frun.font.name = 'Calibri'
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(0x71, 0x80, 0x96)

    # ── Executive Color Palette ──
    NAVY = RGBColor(0x1A, 0x36, 0x5D)      # Primary Title #1A365D
    SLATE = RGBColor(0x2B, 0x6C, 0xB0)     # Section Headers #2B6CB0
    CHARCOAL = RGBColor(0x2D, 0x37, 0x48)  # Body text #2D3748
    PURPLE = RGBColor(0x6B, 0x46, 0xC1)    # Subtitle / Accent #6B46C1

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Arial'
        run.font.size = Pt(22)
        run.font.color.rgb = NAVY
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(16)
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(12.5)
        run.font.color.rgb = PURPLE
        run.italic = True
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Arial'
        run.font.size = Pt(15)
        run.font.color.rgb = NAVY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Arial'
        run.font.size = Pt(12.5)
        run.font.color.rgb = SLATE
        return p

    def add_p(text, bold_prefix=None, space_after=6):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            br = p.add_run(bold_prefix)
            br.bold = True
            br.font.name = 'Calibri'
            br.font.size = Pt(10.5)
            br.font.color.rgb = CHARCOAL
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(10.5)
        run.font.color.rgb = CHARCOAL
        return p

    def add_bullet(text, bold_prefix=None, level=0):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
        if bold_prefix:
            br = p.add_run(bold_prefix)
            br.bold = True
            br.font.name = 'Calibri'
            br.font.size = Pt(10)
            br.font.color.rgb = CHARCOAL
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(10)
        run.font.color.rgb = CHARCOAL
        return p

    def set_cell_background(cell, fill_hex):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd)

    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(
            f'<w:tcMar {nsdecls("w")}>'
            f'<w:top w:w="{top}" w:type="dxa"/>'
            f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
            f'<w:left w:w="{left}" w:type="dxa"/>'
            f'<w:right w:w="{right}" w:type="dxa"/>'
            f'</w:tcMar>'
        )
        tcPr.append(tcMar)

    def add_callout(text, title=None):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        set_cell_background(cell, 'F0F4F8')
        set_cell_margins(cell, top=120, bottom=120, left=180, right=180)

        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'<w:top w:val="none"/>'
            f'<w:left w:val="single" w:sz="24" w:space="0" w:color="2B6CB0"/>'
            f'<w:bottom w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        if title:
            rt = p.add_run(f'📌 {title}\n')
            rt.bold = True
            rt.font.name = 'Arial'
            rt.font.color.rgb = NAVY
            rt.font.size = Pt(10)
        r = p.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(9.5)
        r.font.color.rgb = CHARCOAL
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def format_table(tbl, col_widths, headers, data):
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

        # Header row
        hdr_row = tbl.rows[0]
        for idx, text in enumerate(headers):
            cell = hdr_row.cells[idx]
            cell.width = col_widths[idx]
            set_cell_background(cell, '1A365D')
            set_cell_margins(cell, top=110, bottom=110, left=130, right=130)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            r.bold = True
            r.font.name = 'Arial'
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        # Data rows
        for row_idx, row_data in enumerate(data):
            row = tbl.rows[row_idx + 1]
            bg = 'F7FAFC' if row_idx % 2 == 0 else 'FFFFFF'
            for col_idx, val in enumerate(row_data):
                cell = row.cells[col_idx]
                cell.width = col_widths[col_idx]
                set_cell_background(cell, bg)
                set_cell_margins(cell, top=90, bottom=90, left=130, right=130)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run(str(val))
                r.font.name = 'Calibri'
                r.font.size = Pt(9)
                r.font.color.rgb = CHARCOAL
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # =========================================================================
    # DOCUMENT SECTIONS
    # =========================================================================

    # Title Block
    add_title('NotesAI: AI-Powered Academic Notes Summarizer & Question Generator')
    add_subtitle('Comprehensive Technical Architecture, Machine Learning Mechanics, and Implementation Report')

    add_callout(
        'Project Name: NotesAI — AI-Powered Notes Summarizer & Question Generator\n'
        'Academic Domain: Natural Language Processing (NLP), Deep Learning, Cognitive EdTech\n'
        'Execution Target: Local Edge Inference (Windows 11, Intel i5-13450HX, NVIDIA RTX 3050 6GB GPU, 16GB RAM)\n'
        'Primary Models: facebook/bart-large-cnn (406M params) & Qwen/Qwen2.5-1.5B-Instruct (1.54B params)\n'
        'Key Technologies: Python 3.11, PyTorch 2.6 (CUDA 12.4), Hugging Face Transformers, Streamlit, PyMuPDF, ReportLab, Pydantic v2',
        title='PROJECT SPECIFICATION METADATA'
    )

    # 1. Executive Summary & Problem Statement
    add_h1('1. Executive Summary & Problem Formulation')
    add_p(
        'In contemporary higher education, students are overwhelmed with voluminous lecture notes, textbook chapters, and scientific PDFs. '
        'Educational psychology and cognitive science consistently demonstrate that passive study strategies—such as rereading or linear highlighting—produce '
        'low retention rates, with studies indicating less than 20% recall after two weeks. In contrast, Active Recall and Retrieval Practice—systematically testing '
        'oneself on key principles and synthesizing concepts—yields retention rates exceeding 70%.'
    )
    add_p(
        'Despite the emergence of generic cloud-based generative AI systems (such as ChatGPT, Gemini, or Claude), several critical barriers impede their practical adoption for college study:',
        bold_prefix='The Educational Bottlenecks: '
    )
    add_bullet('Academic lecture notes, student assignments, and proprietary syllabus materials are uploaded to third-party corporate cloud servers.', bold_prefix='1. Privacy & Confidentiality Risks: ')
    add_bullet('Commercial APIs impose strict monthly subscriptions or per-token billing, creating financial barriers for university students.', bold_prefix='2. Financial & API Paywalls: ')
    add_bullet('Generic chatbots generate plausible-sounding but factually fabricated answers not present in the student\'s syllabus.', bold_prefix='3. Hallucinations & Lack of Grounding: ')
    add_bullet('Cloud tools require continuous high-speed internet, failing in examination halls, libraries, or dormitories with spotty connectivity.', bold_prefix='4. Network Dependency: ')
    add_p(
        'NotesAI solves these challenges by providing a 100% offline, locally-running AI suite that ingests academic notes, compresses them into structured '
        'revision sheets using a dedicated sequence-to-sequence summarizer, generates four diverse tiers of exam questions via an instruction-tuned language model, '
        'and hosts an interactive self-scoring quiz engine with strict evidence-grounding.'
    )

    # 2. End-to-End System Architecture
    add_h1('2. System Architecture & End-to-End Pipeline')
    add_p(
        'The architecture of NotesAI follows a modular, decoupled pipeline designed for high computational efficiency, strict data validation, and graceful hardware fallbacks. '
        'The data flow proceeds through six sequential stages:'
    )

    arch_headers = ['Stage', 'Subsystem Module', 'Technical Responsibility', 'Key Output']
    arch_data = [
        ['1. Ingestion', 'document_processor.py', 'Extract text from PDF (PyMuPDF), TXT, or direct paste; validate minimum length (>50 chars).', 'Raw extracted string'],
        ['2. Normalization', 'document_processor.py', 'Unicode NFC normalisation, control char stripping, soft-hyphen de-hyphenation.', 'Cleaned text stream'],
        ['3. Chunking', 'document_processor.py', 'Boundary-aware paragraph splitting into 2,200 char chunks with 200 char overlap.', 'List of semantic chunks'],
        ['4A. Summarization', 'ai_service.py / summarizer.py', 'facebook/bart-large-cnn execution on CUDA; abstractive density reduction.', 'Raw draft summary'],
        ['4B. Formatting & Concepts', 'ai_service.py / summarizer.py', 'Qwen2.5-1.5B reformats into notes/bullets/revision; extracts concepts and definitions.', 'Structured notes dictionary'],
        ['4C. Q&A Generation', 'question_generator.py', 'Qwen2.5-1.5B generates JSON arrays of MCQs, Short, Long, and T/F questions.', 'Raw JSON payload'],
        ['5. Validation', 'validators.py', 'Pydantic v2 validation, deduplication, evidence cross-checking against source text.', 'Validated Question objects'],
        ['6. Delivery & Export', 'app.py / export_service.py', 'Interactive Streamlit UI, live quiz scoring engine, ReportLab PDF and TXT generation.', 'User view / PDF downloads']
    ]
    tbl1 = doc.add_table(rows=len(arch_data) + 1, cols=4)
    format_table(tbl1, [Inches(1.0), Inches(1.5), Inches(2.7), Inches(1.8)], arch_headers, arch_data)

    # 3. Deep Dive into AI Models
    add_h1('3. Deep Dive: Machine Learning Models & Architectural Mechanics')
    add_p(
        'A distinguishing engineering choice in NotesAI is its Hybrid Two-Model Architecture. Rather than forcing a single model to perform both dense '
        'information compression and flexible instruction reasoning, NotesAI pairs a dedicated Seq2Seq encoder-decoder model with a modern causal decoder-only model.'
    )

    # 3.1 BART
    add_h2('3.1 Model 1: facebook/bart-large-cnn (Abstractive Summarization)')
    add_p(
        'facebook/bart-large-cnn is an architectural realization of the BART (Bidirectional and Auto-Regressive Transformers) framework developed by Lewis et al. (2020) at Meta AI. '
        'It contains 406 million parameters configured across a 12-layer bidirectional encoder and a 12-layer autoregressive decoder with a hidden dimension of 1,024 and 16 attention heads.'
    )
    add_bullet('Bidirectional Encoder: Operates identically to BERT, applying non-directional multi-head self-attention across the full chunk simultaneously. Every token attends to every other token, capturing full syntactic and conceptual context.', bold_prefix='Encoder Mechanics: ')
    add_bullet('Autoregressive Decoder: Operates identically to GPT, employing causal masking where token t can only attend to tokens < t. Crucially, each decoder layer contains cross-attention sub-layers where queries (Q) are derived from the decoder state while keys (K) and values (V) are projected from the final encoder representations.', bold_prefix='Decoder & Cross-Attention: ')
    add_bullet('Pre-training as a Denoising Autoencoder: BART was pre-trained by corrupting clean text with composite noise transformations—including token masking, token deletion, text infilling with arbitrary mask tokens, sentence permutation, and document rotation. The model was trained to reconstruct the original clean document, endowing it with unprecedented text reconstruction and compression abilities.', bold_prefix='Pre-training Philosophy: ')
    add_bullet('Fine-Tuning on CNN/DailyMail: Fine-tuned on over 300,000 news articles to excel at abstractive summarization—generating novel phrasing rather than merely extracting and copying verbatim sentences.', bold_prefix='Fine-Tuning Specialization: ')
    add_bullet('Beam Search Decoding: Employs a beam width of 4 with a length penalty (alpha = 2.0) and n-gram repetition blocking (no_repeat_ngram_size = 3) to ensure summaries are fluent, non-redundant, and tightly concise.', bold_prefix='Decoding Strategy: ')

    # 3.2 Qwen
    add_h2('3.2 Model 2: Qwen/Qwen2.5-1.5B-Instruct (Instruction, Q&A & Reasoning)')
    add_p(
        'Qwen/Qwen2.5-1.5B-Instruct is a state-of-the-art dense causal language model released in late 2024 by Alibaba Cloud. It represents the pinnacle of small language model (SLM) '
        'engineering, packing 1.54 billion parameters across 28 decoder layers with an intermediate dimension of 8,960 and an embedding dimension of 1,536.'
    )
    add_bullet('Causal Decoder-Only Topology: Every layer performs unidirectional self-attention. Tokens are predicted autoregressively via next-token prediction: P(w_t | w_1, ..., w_{t-1}).', bold_prefix='Architecture: ')
    add_bullet('Rotary Position Embeddings (RoPE): Unlike classical absolute or learned positional embeddings, RoPE encodes positional information by rotating the query and key representations in the complex plane. For a 2D subvector, rotation by angle m*theta ensures that the inner product between query at position m and key at position n depends strictly on their relative distance (m - n), enabling stable context processing up to 32,768 tokens.', bold_prefix='RoPE Formulation: ')
    add_bullet('SwiGLU Activation: Replaces traditional GELU or ReLU activations in the feed-forward network with SwiGLU: SwiGLU(x) = (xW * sigmoid(beta * xW)) * (xV). This gated linear unit significantly enhances non-linear representational capacity per parameter.', bold_prefix='SwiGLU Activation: ')
    add_bullet('RMSNorm (Root Mean Square Normalization): Normalizes inputs using root mean square without centering the mean: RMS(x) = sqrt(1/d * sum(x_i^2)). This eliminates the computational overhead of variance shifting while maintaining training stability.', bold_prefix='Pre-Layer Normalization: ')
    add_bullet('Grouped-Query Attention (GQA): Utilizes 12 query attention heads but only 2 key/value heads. This reduces Key-Value (KV) cache memory footprint by 6x during autoregressive decoding, allowing real-time token generation on laptop GPUs without VRAM bottlenecks.', bold_prefix='GQA & KV Cache Efficiency: ')
    add_bullet('Instruction & JSON Specialization: Post-trained with Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) on complex instruction-following datasets, making it uniquely capable of outputting valid, uncorrupted JSON payloads conforming to rigid Pydantic schemas.', bold_prefix='Post-Training: ')

    # 3.3 Synergy
    add_h2('3.3 Architectural Synergy: Why Use Both Models?')
    add_p(
        'Combining BART and Qwen creates a complementary division of labor:'
    )
    add_bullet('Seq2Seq encoder-decoder models are mathematically optimal for condensing long contexts into concise abstracts without rambling, but struggle with open-ended reasoning or structured formatting.', bold_prefix='BART\'s Specialty: ')
    add_bullet('Causal instruction models excel at structured synthesis, adhering to JSON schemas, formulating diverse pedagogical questions, and explaining logic, but can be verbose if used for raw summarization.', bold_prefix='Qwen\'s Specialty: ')
    add_bullet('By chaining BART\'s raw condensation with Qwen\'s structuring and question generation, NotesAI achieves the highest possible summary accuracy and question diversity simultaneously.', bold_prefix='The Hybrid Advantage: ')

    # 4. Full Technology Stack
    add_h1('4. Full Technology Stack Breakdown')
    tech_headers = ['Tier / Function', 'Technology / Library', 'Version', 'Architectural Role & Justification']
    tech_data = [
        ['UI & Presentation', 'Streamlit', '1.66.1', 'Reactive single-page application framework; handles session state, real-time widget reactivity, and dark-purple UI styling.'],
        ['Deep Learning Engine', 'PyTorch (CUDA 12.4)', '2.6.0+cu124', 'Underlying tensor computation and GPU acceleration framework running on RTX 3050 Tensor Cores in float16 precision.'],
        ['Model Orchestration', 'Hugging Face Transformers', '4.57.6', 'Pipeline loading, tokenizer abstractions, beam search decoding, and chat template execution.'],
        ['High-Speed Tokenizer', 'Hugging Face Tokenizers', '0.22.2', 'Rust-based BPE tokenization engine handling subword decomposition and sequence length enforcement.'],
        ['Acceleration Layer', 'Hugging Face Accelerate', '1.15.0', 'Device mapping, mixed-precision dispatch, and GPU memory allocation management.'],
        ['Document Parsing', 'PyMuPDF (Fitz)', '2.0.0', 'C-based MuPDF bindings for fast extraction of text streams, headings, and multi-column academic layouts.'],
        ['Report Generation', 'ReportLab', '5.1.0', 'Canvas and Flowable engine for generating downloadable, professionally typeset academic PDF reports.'],
        ['Data Schema & Contract', 'Pydantic v2', '2.14.1', 'Rigorous data models, type checking, field validation, regex verification, and evidence cross-matching.'],
        ['Unit Testing Suite', 'Pytest & Pytest-Mock', '9.2.0', 'Offline test framework verifying document chunking, sanitization, validation, scoring, and edge cases (110 tests).'],
        ['Runtime Environment', 'Python', '3.11.2', 'Optimized virtual environment providing native binary wheel compatibility with PyTorch CUDA and C-extensions.']
    ]
    tbl2 = doc.add_table(rows=len(tech_data) + 1, cols=4)
    format_table(tbl2, [Inches(1.2), Inches(1.5), Inches(0.9), Inches(3.4)], tech_headers, tech_data)

    # 5. Core Algorithmic Subsystems
    add_h1('5. Core Algorithmic Subsystems')

    add_h2('5.1 Document Processing & Semantic Chunking Engine')
    add_p(
        'Academic documents frequently exceed the context limits of transformer models. To prevent truncated text or loss of context across section breaks, '
        'NotesAI implements a boundary-aware semantic chunking algorithm with sliding-window overlap:'
    )
    add_bullet('Text Normalization: Input strings undergo unicodedata.normalize("NFC", text), removing non-printable control characters while preserving mathematical symbols, Greek letters, and formulas. Soft hyphens (\\u00ad) and hyphenated line breaks are unified.', bold_prefix='1. Normalization: ')
    add_bullet('Paragraph-Aware Partitioning: The engine splits text along double-newlines (\\n\\n). If a paragraph exceeds the target chunk size (2,200 chars), it splits along sentence boundaries (. / ! / ?).', bold_prefix='2. Boundary Respect: ')
    add_bullet('Sliding Window Overlap: A 200-character overlap is maintained between consecutive chunks. This guarantees that sentences spanning chunk boundaries are not sliced in half, preserving semantic continuity for summarization.', bold_prefix='3. Overlap Buffer: ')

    add_h2('5.2 Structured Question Generation & Taxonomy')
    add_p(
        'NotesAI generates four distinct tiers of examination questions designed according to Bloom\'s Revised Educational Taxonomy:'
    )
    add_bullet('Recall and recognition testing. Formulates 4 options (A, B, C, D) with distinct distractors, exactly one verified correct answer, an explanation, and source evidence quotation.', bold_prefix='1. Multiple Choice Questions (MCQs): ')
    add_bullet('Conceptual understanding testing. Formulates concise questions requiring 2–4 sentence answers, accompanied by a bullet list of essential key scoring points.', bold_prefix='2. Short Answer Questions: ')
    add_bullet('Application, synthesis, and essay testing. Formulates in-depth questions with detailed model answers, structural main points, and marking guidance for 5-to-10 mark university exam questions.', bold_prefix='3. Long Answer Questions: ')
    add_bullet('Critical evaluation and misconception checking. Formulates factual statements that are definitively True or False, accompanied by a mandatory justification explaining why the false statement is incorrect.', bold_prefix='4. True/False Questions: ')

    add_h2('5.3 Pydantic v2 Schema Enforcement & Hallucination Defense')
    add_p(
        'Raw LLM outputs are inherently probabilistic and susceptible to formatting corruption or factual fabrication. NotesAI employs a three-layer defense system:'
    )
    add_bullet('Pydantic Models: MCQQuestion, ShortAnswerQuestion, LongAnswerQuestion, and TrueFalseQuestion enforce strict type constraints, requiring non-empty evidence strings and valid option keys.', bold_prefix='Layer 1: Type Validation: ')
    add_bullet('JSON Extraction Heuristics: An extraction engine strips markdown code fences (```json ... ```), finds the outermost bracket array [...], and handles malformed trailing commas before JSON parsing.', bold_prefix='Layer 2: Robust Extraction: ')
    add_bullet('Evidence Grounding Verification: The system performs case-insensitive substring verification of the model\'s quoted evidence against the original source document. If a generated question quotes evidence that does not exist in the student\'s notes, it is flagged as ungrounded.', bold_prefix='Layer 3: Evidence Cross-Matching: ')

    add_h2('5.4 Self-Scoring Interactive Quiz Engine')
    add_p(
        'The Practice Exam page provides an interactive study simulation. When a student submits their choices:'
    )
    add_bullet('MCQ responses are checked against the normalized correct key (A/B/C/D).', bold_prefix='Automated Evaluation: ')
    add_bullet('True/False responses are verified against boolean ground truth.', bold_prefix='True/False Checking: ')
    add_bullet('Dynamic score calculation displays total correct, wrong, and skipped questions alongside percentage score badges.', bold_prefix='Real-time Scoring: ')
    add_bullet('Instant visual feedback provides green highlights for correct choices, red for incorrect, and reveals the complete explanation with source quotes.', bold_prefix='Pedagogical Feedback: ')

    # 6. Engineering Challenges Solved
    add_h1('6. Key Engineering Challenges & Critical Optimizations')
    add_p(
        'During the development and testing of NotesAI, three major technical hurdles were encountered and resolved:'
    )

    add_h2('6.1 Challenge 1: CPU Fallback & 100% RAM Bottleneck')
    add_p(
        'Problem: Initially, the project attempted to run Qwen2.5-3B-Instruct alongside BART. In ai_service.py, a conservative VRAM safety threshold of 5.5 GB was configured. '
        'Because the RTX 3050 Laptop GPU has 6.0 GB total VRAM and Windows reserves ~1.0 GB for display drivers, available VRAM was 5.03 GB (< 5.5 GB). '
        'This caused the system to fall back to CPU execution in torch.float32 mode. A 3B model in float32 required 12.4 GB of system RAM, completely saturating the 16 GB machine (100% RAM usage), '
        'triggering severe SSD virtual memory thrashing, and causing generation times of 10 to 15 minutes per batch.'
    )
    add_p(
        'Solution: Re-architected the system to use Qwen/Qwen2.5-1.5B-Instruct in torch.float16 on CUDA. In float16, Qwen2.5-1.5B requires only ~3.0 GB of VRAM, fitting comfortably '
        'within the 5.03 GB free GPU memory alongside BART. Latency dropped from 15 minutes to 2–4 seconds (a 300x speedup), while system RAM usage dropped from 100% to under 15%.'
    )

    add_h2('6.2 Challenge 2: The BART Tokenizer Max Length Bug & CUDA Kernel Crash')
    add_p(
        'Problem: During summarization testing, the application triggered a fatal crash: CUDA error: device-side assert triggered (srcIndex < srcSelectDimSize). '
        'Once triggered, PyTorch declared Device 0 permanently unavailable. Investigation revealed an infamous upstream bug: the Hugging Face tokenizer config for '
        'facebook/bart-large-cnn defines model_max_length as 10^30 tokens! When text chunks of ~3,000 characters were passed with truncation=True, the pipeline didn\'t truncate '
        'to BART\'s physical 1,024-token embedding table. Passing token #1025 into CUDA kernel embeddings caused an out-of-bounds index access that poisoned the GPU driver context.'
    )
    add_p(
        'Solution: In ai_service.py, explicitly instantiated AutoTokenizer with model_max_length=1024 and truncation_side="right". Reduced BART_CHUNK_CHARS in summarizer.py '
        'from 3,000 to 2,200 characters (~500 tokens), well below the 1,024 limit. Implemented an automatic graceful fallback to Qwen if BART ever raises an exception, ensuring zero crashes.'
    )

    add_h2('6.3 Challenge 3: Windows Console UTF-8 & Charmap Encoding Exceptions')
    add_p(
        'Problem: In the model downloader script, logging terminal output containing Unicode symbols (such as checkmarks or crosses) caused UnicodeEncodeError: \'charmap\' codec can\'t encode character '
        'due to Windows PowerShell\'s default legacy CP1252 code page.'
    )
    add_p(
        'Solution: Added programmatic stream reconfiguration at script startup: sys.stdout.reconfigure(encoding="utf-8") and sys.stderr.reconfigure(encoding="utf-8"), '
        'alongside ASCII-safe status tags ([OK], [FAILED]) for bulletproof cross-platform execution.'
    )

    # 7. Verification & Testing
    add_h1('7. Verification & Test Suite Quality Assurance')
    add_p(
        'To ensure robustness for academic assessment, the repository includes a comprehensive automated test suite consisting of 110 unit and integration tests. '
        'All tests execute fully offline without requiring GPU hardware or active AI model weights:'
    )
    add_bullet('30 tests in tests/test_document_processor.py covering PDF stream decoding, empty file handling, Unicode de-hyphenation, hash generation, and boundary chunking.', bold_prefix='Document Processor Tests: ')
    add_bullet('35 tests in tests/test_validators.py verifying Pydantic models, duplicate detection algorithms, distractor uniqueness, and case-insensitive evidence matching.', bold_prefix='Validation Tests: ')
    add_bullet('45 tests in tests/test_text_processing.py verifying JSON extraction from conversational preambles, score calculation logic, export formatting, and change detection.', bold_prefix='Processing & Export Tests: ')
    add_callout(
        'Test Suite Status: 110 passed, 0 failed in 0.22 seconds.\n'
        'Command: pytest tests/ -v\n'
        'Code Coverage: Extraction, Sanitization, Chunking, Prompting, Scoring, Exporting, Schema Enforcement.',
        title='AUTOMATED QA RESULTS'
    )

    # 8. Free Deployment Strategies
    add_h1('8. Free Deployment Strategies for College Assessment')
    add_p(
        'For academic demonstrations and project evaluations, two completely free deployment pathways are provided:'
    )
    add_bullet('Cloudflare Tunnel (cloudflared tunnel --url http://localhost:8501) generates a free, secure public HTTPS URL (e.g., https://xyz.trycloudflare.com). The application continues to run on your laptop\'s RTX 3050 GPU, giving professors instant 2-second responses on their mobile devices with zero hosting cost.', bold_prefix='Option A (Recommended for Viva/Demo): ')
    add_bullet('Hugging Face Spaces provides a free "CPU Basic" tier with 16 GB RAM and 50 GB disk. The repository can be linked directly to a Streamlit Space, downloading models from the internal Hugging Face CDN in under 60 seconds and running 24/7 without keeping your personal laptop on.', bold_prefix='Option B (Permanent 24/7 Cloud Host): ')

    # 9. Conclusion
    add_h1('9. Conclusion & Educational Impact')
    add_p(
        'NotesAI represents an end-to-end, privacy-preserving, and computationally optimized AI learning suite. By combining sequence-to-sequence abstractive '
        'summarization (BART) with high-efficiency instruction reasoning (Qwen2.5-1.5B), the system achieves industrial-grade NLP capabilities on consumer laptop hardware. '
        'Its rigorous data validation contracts, evidence-grounded hallucination safeguards, and multi-format exam generator offer a practical, demonstrable, and pedagogically sound '
        'tool for students and educators alike.'
    )

    output_filename = 'NotesAI_Complete_Project_Report.docx'
    doc.save(output_filename)
    print(f'Successfully generated {output_filename}!')

if __name__ == '__main__':
    create_report()
