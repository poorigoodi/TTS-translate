"""Unit tests for translator helpers (no model downloads)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translator import (  # noqa: E402
    TranslationError,
    _build_instruction,
    _postprocess_translation,
    _validate_languages,
    _validate_text,
    translate_text,
)


def test_validate_text_strips():
    assert _validate_text("  hello  ") == "hello"


def test_validate_languages_ok():
    assert _validate_languages("FA", "en") == ("fa", "en")


def test_build_instruction_fa_en():
    text = _build_instruction("fa", "en")
    assert "Persian-to-English" in text
    assert "Return only the translated text" in text


def test_build_instruction_en_fa():
    text = _build_instruction("en", "fa")
    assert "English-to-Persian" in text


def test_postprocess_rejects_empty():
    with pytest.raises(TranslationError, match="empty"):
        _postprocess_translation("   ", "منبع")


def test_postprocess_strips_code_fence():
    result = _postprocess_translation("```\nHello world\n```", "سلام دنیا")
    assert result == "Hello world"


def test_postprocess_rejects_echo_of_long_source():
    source = "This is a reasonably long English sentence used as source."
    with pytest.raises(TranslationError, match="unchanged"):
        _postprocess_translation(source, source)


def test_translate_text_uses_non_thinking_template():
    """Ensure chat template is invoked with enable_thinking=False."""
    prompt_ids = torch.tensor([[10, 11, 12, 13]])
    attention = torch.ones_like(prompt_ids)
    generated = torch.tensor([[10, 11, 12, 13, 20, 21, 22]])

    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "PROMPT"
    fake_tokenizer.pad_token_id = 0
    fake_tokenizer.eos_token_id = 1
    fake_tokenizer.return_value = {
        "input_ids": prompt_ids,
        "attention_mask": attention,
    }
    fake_tokenizer.decode.return_value = "The weather is nice."

    fake_model = MagicMock()
    fake_model.generate.return_value = generated

    with patch(
        "translator.load_translation_model",
        return_value=(fake_model, fake_tokenizer, "cpu"),
    ):
        result = translate_text("امروز هوا خوب است.", "fa", "en")

    assert result == "The weather is nice."
    kwargs = fake_tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs.get("enable_thinking") is False
    assert kwargs.get("add_generation_prompt") is True
