"""Qwen3-0.6B translation module (Persian ↔ English).

Loads the model once and exposes ``translate_text`` for application use.
Designed so ``TRANSLATION_MODEL_ID`` can point at a fine-tuned checkpoint later.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    LANGUAGE_CODES,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    TRANSLATION_MODEL_ID,
    TRANSLATION_REPETITION_PENALTY,
    TRANSLATION_TEMPERATURE,
    TRANSLATION_TOP_K,
    TRANSLATION_TOP_P,
    resolve_device,
)

logger = logging.getLogger(__name__)

# Token ID for </think> in Qwen3 vocabulary (official model card)
_THINK_END_TOKEN_ID = 151668

_PROMPT_MARKERS = (
    "you are a professional",
    "translate the following",
    "return only the translated",
    "do not explain",
)

_EXPLANATION_PREFIXES = (
    "sure,",
    "here is",
    "here's",
    "the translation",
    "translated text",
    "of course",
    "certainly",
)

_model: Any = None
_tokenizer: Any = None
_loaded_model_id: str | None = None
_device: str | None = None


class TranslationError(Exception):
    """Raised when translation cannot be completed successfully."""


def _validate_text(text: str) -> str:
    if text is None:
        raise TranslationError("Input text is required.")
    cleaned = text.strip()
    if not cleaned:
        raise TranslationError("Input text cannot be empty.")
    return cleaned


def _validate_languages(source_language: str, target_language: str) -> tuple[str, str]:
    src = (source_language or "").strip().lower()
    tgt = (target_language or "").strip().lower()
    if src not in LANGUAGE_CODES:
        raise TranslationError(
            f"Unsupported source language '{source_language}'. Use 'fa' or 'en'."
        )
    if tgt not in LANGUAGE_CODES:
        raise TranslationError(
            f"Unsupported target language '{target_language}'. Use 'fa' or 'en'."
        )
    if src == tgt:
        raise TranslationError("Source and target languages must be different.")
    return src, tgt


def _build_instruction(source_language: str, target_language: str) -> str:
    if source_language == "fa" and target_language == "en":
        return (
            "You are a professional Persian-to-English translator. "
            "Translate the following Persian text into natural, accurate English. "
            "Preserve the original meaning, tone, names, numbers, and formatting. "
            "Do not explain the translation or add information. "
            "Return only the translated text."
        )
    if source_language == "en" and target_language == "fa":
        return (
            "You are a professional English-to-Persian translator. "
            "Translate the following English text into natural, accurate Persian. "
            "Preserve the original meaning, tone, names, numbers, and formatting. "
            "Do not explain the translation or add information. "
            "Return only the translated text."
        )
    raise TranslationError(
        f"Unsupported language pair: {source_language} → {target_language}"
    )


def _build_user_message(instruction: str, text: str) -> str:
    return f"{instruction}\n\nText:\n{text}"


def load_translation_model(model_id: str | None = None) -> tuple[Any, Any, str]:
    """Load (or reuse) the translation model and tokenizer.

    Parameters
    ----------
    model_id:
        Hugging Face model ID or local checkpoint directory. Defaults to
        ``TRANSLATION_MODEL_ID`` from config / environment.
    """
    global _model, _tokenizer, _loaded_model_id, _device

    target_id = model_id or TRANSLATION_MODEL_ID
    if _model is not None and _tokenizer is not None and _loaded_model_id == target_id:
        return _model, _tokenizer, _device  # type: ignore[return-value]

    device = resolve_device()
    logger.info("Loading translation model %s on %s", target_id, device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            target_id,
            dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
    except OSError as exc:
        raise TranslationError(
            f"Failed to load translation model '{target_id}'. "
            f"Check the model ID / local path and network access. Details: {exc}"
        ) from exc
    except torch.cuda.OutOfMemoryError as exc:
        raise TranslationError(
            "CUDA out of memory while loading the translation model. "
            "Try DEVICE=cpu or free GPU memory."
        ) from exc

    _model = model
    _tokenizer = tokenizer
    _loaded_model_id = target_id
    _device = device
    return model, tokenizer, device


def unload_translation_model() -> None:
    """Release the cached model (useful in tests / evaluation scripts)."""
    global _model, _tokenizer, _loaded_model_id, _device
    _model = None
    _tokenizer = None
    _loaded_model_id = None
    _device = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_assistant_content(output_ids: list[int], tokenizer: Any) -> str:
    """Decode generated token IDs, stripping Qwen3 thinking content if present."""
    try:
        index = len(output_ids) - output_ids[::-1].index(_THINK_END_TOKEN_ID)
    except ValueError:
        index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
    return content.strip()


def _strip_code_fences(text: str) -> str:
    fence = re.match(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", text.strip())
    if fence:
        return fence.group(1).strip()
    return text


def _postprocess_translation(raw: str, source_text: str) -> str:
    text = _strip_code_fences(raw).strip()
    if not text:
        raise TranslationError("Model returned an empty translation.")

    # Drop accidental instruction echo at the start
    lowered = text.lower()
    for marker in _PROMPT_MARKERS:
        if marker in lowered[:200]:
            # Keep only content after the last colon/newline heuristic
            parts = re.split(r"(?:translated text|translation)\s*:\s*", text, flags=re.I)
            if len(parts) > 1:
                text = parts[-1].strip()
            break

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        first_lower = lines[0].lower()
        if any(first_lower.startswith(prefix) for prefix in _EXPLANATION_PREFIXES):
            if len(lines) > 1:
                lines = lines[1:]
            else:
                # Single explanatory line — likely a failure
                raise TranslationError(
                    "Model returned commentary instead of a translation. "
                    "Please try again or shorten the input."
                )
        text = "\n".join(lines)

    text = text.strip().strip('"').strip("'").strip()
    if not text:
        raise TranslationError("Translation was empty after cleaning.")

    # Detect obvious echo of the source (same string)
    if text.strip() == source_text.strip() and len(source_text) > 20:
        raise TranslationError(
            "Model returned the source text unchanged. Translation may have failed."
        )

    # Collapse pathological repetition of the same short phrase
    if len(text) > 80:
        words = text.split()
        if len(words) >= 12:
            chunk = " ".join(words[:4])
            if text.count(chunk) >= 4:
                raise TranslationError(
                    "Model output appears highly repetitive. Please try again."
                )

    return text


@torch.inference_mode()
def translate_text(
    text: str,
    source_language: str,
    target_language: str,
    model_id: str | None = None,
) -> str:
    """Translate ``text`` from ``source_language`` to ``target_language``.

    Returns only the translated text. Raises ``TranslationError`` on failure.
    """
    source = _validate_text(text)
    src_lang, tgt_lang = _validate_languages(source_language, target_language)
    instruction = _build_instruction(src_lang, tgt_lang)
    user_content = _build_user_message(instruction, source)

    try:
        model, tokenizer, device = load_translation_model(model_id)
    except TranslationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranslationError(f"Unexpected error loading translation model: {exc}") from exc

    messages = [{"role": "user", "content": user_content}]

    try:
        # Official Qwen3 chat template — non-thinking mode for clean translation output
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if encoded["input_ids"].shape[-1] >= MAX_INPUT_LENGTH:
            logger.warning(
                "Input truncated to MAX_INPUT_LENGTH=%s tokens.", MAX_INPUT_LENGTH
            )

        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": True,
            "temperature": TRANSLATION_TEMPERATURE,
            "top_p": TRANSLATION_TOP_P,
            "top_k": TRANSLATION_TOP_K,
            "repetition_penalty": TRANSLATION_REPETITION_PENALTY,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        generated = model.generate(**generate_kwargs)
        new_tokens = generated[0][input_ids.shape[-1] :].tolist()
        raw = _decode_assistant_content(new_tokens, tokenizer)
        return _postprocess_translation(raw, source)

    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise TranslationError(
            "CUDA out of memory during translation. "
            "Try a shorter input or set DEVICE=cpu."
        ) from exc
    except TranslationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranslationError(f"Translation generation failed: {exc}") from exc
