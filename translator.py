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
    TRANSLATION_NO_REPEAT_NGRAM,
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
        from pathlib import Path

        target_path = Path(target_id)
        adapter_config = target_path / "adapter_config.json"
        is_lora_adapter = target_path.exists() and adapter_config.exists()

        dtype = torch.float16 if device == "cuda" else torch.float32

        if is_lora_adapter:
            # Fine-tuned LoRA checkpoint directory
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise TranslationError(
                    "This checkpoint is a LoRA adapter but `peft` is not installed. "
                    "Run: pip install peft"
                ) from exc

            adapter_meta = {}
            meta_file = target_path / "finetune_meta.json"
            if meta_file.exists():
                import json

                adapter_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_id = adapter_meta.get("base_model") or TRANSLATION_MODEL_ID
            # Prefer base recorded in adapter_config if present
            try:
                import json

                cfg = json.loads(adapter_config.read_text(encoding="utf-8"))
                base_id = cfg.get("base_model_name_or_path") or base_id
            except Exception:  # noqa: BLE001
                pass

            logger.info("Loading base %s + LoRA adapter %s", base_id, target_id)
            tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_id,
                dtype=dtype,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, str(target_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                target_id,
                dtype=dtype,
                trust_remote_code=True,
            )

        model.to(device)
        model.eval()
    except TranslationError:
        raise
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


def _tokens(text: str) -> list[str]:
    """Split on whitespace; works for Persian and English."""
    return re.findall(r"\S+", text)


def _is_degenerate_repetition(text: str) -> bool:
    """Detect runaway loops like 'چیزی چیزی چیزی…' that small LLMs sometimes emit."""
    words = _tokens(text)
    if len(words) < 6:
        return False

    # Consecutive identical tokens
    streak = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            streak += 1
            if streak >= 4:
                return True
        else:
            streak = 1

    # One token dominates the output
    from collections import Counter

    counts = Counter(words)
    _word, count = counts.most_common(1)[0]
    if count >= 6 and count / len(words) >= 0.35:
        return True

    # Short phrase (2–4 tokens) repeated many times
    for n in (2, 3, 4):
        if len(words) < n * 4:
            continue
        phrase = " ".join(words[:n])
        if phrase and text.count(phrase) >= 4:
            return True

    # Character-level loop (e.g. same 2–6 chars repeated)
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 24:
        for n in range(2, 7):
            unit = compact[:n]
            if unit and compact.count(unit) >= 8 and len(unit) * compact.count(unit) >= 0.6 * len(compact):
                return True

    return False


def _adaptive_max_new_tokens(source_text: str) -> int:
    """Keep generations short for short inputs to reduce loop risk."""
    # Rough char-based budget: translations rarely need > ~2x source length in tokens
    budget = max(48, min(MAX_NEW_TOKENS, len(source_text) * 2 + 32))
    return int(budget)


def _postprocess_translation(raw: str, source_text: str) -> str:
    text = _strip_code_fences(raw).strip()
    if not text:
        raise TranslationError("Model returned an empty translation.")

    # Drop accidental instruction echo at the start
    lowered = text.lower()
    for marker in _PROMPT_MARKERS:
        if marker in lowered[:200]:
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
                raise TranslationError(
                    "Model returned commentary instead of a translation. "
                    "Please try again or shorten the input."
                )
        text = "\n".join(lines)

    text = text.strip().strip('"').strip("'").strip()
    if not text:
        raise TranslationError("Translation was empty after cleaning.")

    if text.strip() == source_text.strip() and len(source_text) > 20:
        raise TranslationError(
            "Model returned the source text unchanged. Translation may have failed."
        )

    if _is_degenerate_repetition(text):
        raise TranslationError(
            "Model output collapsed into a repetitive loop (common on the base "
            "Qwen3-0.6B for some Persian/English prompts). Please try again, "
            "rephrase the input, or use a shorter sentence. "
            "This is a known base-model limitation — fine-tuning later can help."
        )

    # Reject absurdly long outputs relative to the source
    if len(text) > max(200, len(source_text) * 8):
        raise TranslationError(
            "Translation is unusually long compared to the input and was rejected. "
            "Please try again."
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

        max_new = _adaptive_max_new_tokens(source)
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new,
            "do_sample": True,
            "temperature": max(TRANSLATION_TEMPERATURE, 1e-5),
            "top_p": TRANSLATION_TOP_P,
            "top_k": TRANSLATION_TOP_K,
            "repetition_penalty": TRANSLATION_REPETITION_PENALTY,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if TRANSLATION_NO_REPEAT_NGRAM > 0:
            generate_kwargs["no_repeat_ngram_size"] = TRANSLATION_NO_REPEAT_NGRAM

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
