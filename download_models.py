"""
download_models.py
==================
One-time script to pre-download both AI models to the HuggingFace cache.
Run this before launching the app for the first time, or when you need
to ensure offline availability.

Usage:
    python download_models.py
"""

import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

print("=" * 60)
print("  NotesAI - Model Downloader")
print("=" * 60)
print()
print("This script will download two models:")
print("  1. facebook/bart-large-cnn  (~1.6 GB)")
print("  2. Qwen/Qwen2.5-3B-Instruct (~6 GB)")
print()
print("Total download: approximately 7-8 GB.")
print("Models are cached in: ~/.cache/huggingface/hub/")
print("After download, the app can run offline.")
print()
print("Starting download in 3 seconds... (Ctrl+C to cancel)")
time.sleep(3)

# ── 1. BART summarizer ───────────────────────────────────────────────────────
print("\n[1/2] Downloading facebook/bart-large-cnn ...")
try:
    from transformers import pipeline
    pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    print("      [OK] facebook/bart-large-cnn downloaded successfully.")
    del pipe
except Exception as e:
    print(f"      [FAILED] Error: {e}")
    print("         Check your internet connection and try again.")
    sys.exit(1)

# ── 2. Qwen2.5-1.5B-Instruct ─────────────────────────────────────────────────
print("\n[2/2] Downloading Qwen/Qwen2.5-1.5B-Instruct (~3 GB) ...")
print("      Optimized for RTX 3050 (6GB) GPU acceleration.")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
    )
    print("      [OK] Qwen/Qwen2.5-1.5B-Instruct downloaded successfully.")
    del tokenizer, model
except Exception as e:
    print(f"      [FAILED] Error: {e}")
    print("         Check your internet connection and try again.")
    sys.exit(1)

print()
print("=" * 60)
print("  [OK] All models downloaded. You can now run the app offline.")
print("  Launch with:  streamlit run app.py")
print("=" * 60)
