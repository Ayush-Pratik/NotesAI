"""
ai_service.py
=============
Central model management service.

Responsibilities
----------------
- Inspect available hardware (CUDA / CPU, VRAM).
- Load and cache models lazily (first use) to conserve memory.
- Provide a single inference point for summarization and text generation.
- Handle GPU → CPU fallback gracefully.
- Report hardware status to the UI.

Model choices
-------------
Summarization : facebook/bart-large-cnn
  • ~406 M parameters, ~1.6 GB on disk (FP32), ~1.6 GB VRAM in FP32.
  • Well-established news/article summarizer that generalises well to
    academic text with careful prompting.
  • Context window: 1024 tokens; max output: 1024 tokens.
  • License: MIT.

Instruction-following : Qwen/Qwen2.5-3B-Instruct
  • ~3.09 B parameters, ~6 GB on disk in BF16.
  • Excellent instruction follower at the 3 B scale.
  • Runs comfortably in CPU FP32 on 16 GB RAM.
  • On GPU: needs ~7–8 GB VRAM at BF16; fits in 6 GB VRAM at INT8
    using load_in_8bit (bitsandbytes required) or on CPU otherwise.
  • License: Qwen Research License (permissive for non-commercial use).

To change models, edit MODEL_SUMMARIZER and MODEL_INSTRUCT below.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable model names — change these to switch models
# ---------------------------------------------------------------------------

MODEL_SUMMARIZER = "facebook/bart-large-cnn"
MODEL_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"

# Safety threshold: if free VRAM is below this many GB, use CPU for Instruct model
# Qwen2.5-1.5B in FP16 takes ~3.0 GB VRAM, fitting easily in 5+ GB free VRAM
INSTRUCT_MIN_VRAM_GB = 2.5
SUMMARIZER_MIN_VRAM_GB = 1.5

# Maximum new tokens for generation calls
MAX_NEW_TOKENS_SUMMARIZER = 512
MAX_NEW_TOKENS_INSTRUCT = 1024

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    has_cuda: bool
    gpu_name: str
    total_vram_gb: float
    free_vram_gb: float
    device: str                          # "cuda" or "cpu"
    summarizer_device: str               # device for BART
    instruct_device: str                 # device for Qwen
    warnings: list[str] = field(default_factory=list)


def inspect_hardware() -> HardwareInfo:
    """Detect GPU availability and VRAM, then decide device assignments."""
    has_cuda = torch.cuda.is_available()
    gpu_name = ""
    total_vram_gb = 0.0
    free_vram_gb = 0.0
    warnings: list[str] = []

    if has_cuda:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / (1024 ** 3)
            # Reserve a bit for display driver usage
            torch.cuda.empty_cache()
            free_mem, _ = torch.cuda.mem_get_info(0)
            free_vram_gb = free_mem / (1024 ** 3)
        except Exception as e:
            warnings.append(f"GPU detection error: {e}. Falling back to CPU.")
            has_cuda = False

    # Assign devices
    summarizer_device = "cpu"
    instruct_device = "cpu"

    if has_cuda:
        if free_vram_gb >= SUMMARIZER_MIN_VRAM_GB:
            summarizer_device = "cuda"
        else:
            warnings.append(
                f"Insufficient free VRAM ({free_vram_gb:.1f} GB) for summarizer. "
                "Using CPU."
            )
        # Qwen2.5-1.5B fits comfortably in 3+ GB VRAM
        if free_vram_gb >= INSTRUCT_MIN_VRAM_GB:
            instruct_device = "cuda"
        else:
            warnings.append(
                f"Insufficient free VRAM ({free_vram_gb:.1f} GB) for instruction model. "
                "Running on CPU."
            )

    return HardwareInfo(
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        free_vram_gb=free_vram_gb,
        device="cuda" if has_cuda else "cpu",
        summarizer_device=summarizer_device,
        instruct_device=instruct_device,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Cached model loaders (Streamlit resource cache)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_summarizer_pipeline(device: str):
    """Load and cache the BART summarization pipeline."""
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    logger.info("Loading summarizer model %s on %s …", MODEL_SUMMARIZER, device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SUMMARIZER,
            model_max_length=1024,
            truncation_side="right",
        )
        tokenizer.model_max_length = 1024
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_SUMMARIZER,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        if device == "cuda":
            model = model.to("cuda")
        pipe = hf_pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
        )
        logger.info("Summarizer loaded successfully.")
        return pipe
    except Exception as exc:
        logger.error("Failed to load summarizer: %s", exc)
        raise


@st.cache_resource(show_spinner=False)
def _load_instruct_model(device: str):
    """Load and cache the Qwen2.5-3B-Instruct model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading instruction model %s on %s …", MODEL_INSTRUCT, device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_INSTRUCT,
            trust_remote_code=True,
        )
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_INSTRUCT,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_INSTRUCT,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
        model.eval()
        logger.info("Instruction model loaded successfully on %s.", device)
        return tokenizer, model
    except Exception as exc:
        logger.error("Failed to load instruction model: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Public service class
# ---------------------------------------------------------------------------

class AIService:
    """
    Facade used by summarizer.py and question_generator.py.

    Call `get_instance()` to obtain the singleton; it inspects hardware
    and caches model handles.
    """

    _instance: Optional["AIService"] = None

    def __init__(self) -> None:
        self.hw = inspect_hardware()
        self._summarizer_pipe = None
        self._instruct_tokenizer = None
        self._instruct_model = None

    @classmethod
    def get_instance(cls) -> "AIService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_summarizer(self) -> None:
        """Lazily load the summarizer pipeline (BART)."""
        if self._summarizer_pipe is None:
            self._summarizer_pipe = _load_summarizer_pipeline(
                self.hw.summarizer_device
            )

    def load_instruct(self) -> None:
        """Lazily load the instruction-following model (Qwen)."""
        if self._instruct_model is None:
            self._instruct_tokenizer, self._instruct_model = _load_instruct_model(
                self.hw.instruct_device
            )

    # ------------------------------------------------------------------
    # Summarization inference
    # ------------------------------------------------------------------

    def summarize(
        self,
        text: str,
        min_length: int = 80,
        max_length: int = 300,
    ) -> str:
        """
        Run BART summarization on a single chunk.

        Parameters
        ----------
        text : str
            Input text (≤1024 BART tokens; caller should chunk longer text).
        min_length : int
            Minimum output tokens.
        max_length : int
            Maximum output tokens.

        Returns
        -------
        str
            Generated summary text.
        """
        self.load_summarizer()
        try:
            # Word-count safety guard to prevent BART index assertions
            words = text.split()
            input_words = len(words)
            if input_words < 30:
                return text.strip()

            effective_max = min(max_length, max(40, int(input_words * 0.9)))
            effective_min = min(min_length, max(15, effective_max // 3))
            if effective_min >= effective_max:
                effective_min = max(10, effective_max - 10)

            # Safeguard tokenizer max length
            if hasattr(self._summarizer_pipe, "tokenizer"):
                self._summarizer_pipe.tokenizer.model_max_length = 1024

            result = self._summarizer_pipe(
                text,
                min_length=effective_min,
                max_length=effective_max,
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"].strip()
        except Exception as exc:
            logger.warning("BART summarization warning: %s. Using Qwen fallback.", exc)
            try:
                system = (
                    "You are an academic summarizer. Provide a faithful, concise summary "
                    "of the text without hallucinating or adding external facts."
                )
                user = f"Summarize the following passage in {effective_max} words:\n\n{text[:3000]}"
                return self.generate(system, user, max_new_tokens=effective_max + 50, temperature=0.1)
            except Exception as e2:
                logger.error("Fallback summarization failed: %s", e2)
                raise RuntimeError(f"Summarization failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Instruction-following inference
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS_INSTRUCT,
        temperature: float = 0.3,
    ) -> str:
        """
        Run Qwen2.5-3B-Instruct with a system + user message pair.

        Parameters
        ----------
        system_prompt : str
            Instruction/role context for the model.
        user_prompt : str
            The actual task content.
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature (lower = more deterministic).

        Returns
        -------
        str
            Generated text (model response only, no prompt included).
        """
        self.load_instruct()
        tokenizer = self._instruct_tokenizer
        model = self._instruct_model

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback if chat template not available
            text = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        inputs = tokenizer(text, return_tensors="pt", max_length=3500, truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM during generation. Attempting CPU fallback …")
            torch.cuda.empty_cache()
            gc.collect()
            # Reload model on CPU and retry once
            self._instruct_model = None
            self._instruct_tokenizer = None
            self.hw.instruct_device = "cpu"
            self.load_instruct()
            tokenizer = self._instruct_tokenizer
            model = self._instruct_model
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Hardware status report for UI
    # ------------------------------------------------------------------

    def hardware_status(self) -> dict:
        hw = self.hw
        return {
            "cuda_available": hw.has_cuda,
            "gpu_name": hw.gpu_name or "N/A",
            "total_vram_gb": round(hw.total_vram_gb, 1),
            "free_vram_gb": round(hw.free_vram_gb, 1),
            "summarizer_device": hw.summarizer_device,
            "instruct_device": hw.instruct_device,
            "summarizer_model": MODEL_SUMMARIZER,
            "instruct_model": MODEL_INSTRUCT,
            "warnings": hw.warnings,
        }
