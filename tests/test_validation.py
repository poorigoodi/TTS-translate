"""Unit tests for language / input validation (no model downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translator import TranslationError, translate_text  # noqa: E402
from tts_engine import TTSError, generate_speech, is_tts_language_supported  # noqa: E402


def test_translate_rejects_empty_text():
    with pytest.raises(TranslationError, match="empty"):
        translate_text("   ", "fa", "en")


def test_translate_rejects_same_language():
    with pytest.raises(TranslationError, match="different"):
        translate_text("hello", "en", "en")


def test_translate_rejects_unsupported_source():
    with pytest.raises(TranslationError, match="Unsupported source"):
        translate_text("hello", "de", "en")


def test_translate_rejects_unsupported_target():
    with pytest.raises(TranslationError, match="Unsupported target"):
        translate_text("hello", "en", "de")


def test_tts_rejects_empty_text(tmp_path):
    with pytest.raises(TTSError, match="empty"):
        generate_speech("  ", "en", tmp_path / "out.wav")


def test_tts_rejects_unsupported_app_language(tmp_path):
    with pytest.raises(TTSError, match="Unsupported language"):
        generate_speech("hello", "de", tmp_path / "out.wav")


def test_tts_requires_wav_extension(tmp_path):
    with pytest.raises(TTSError, match="\\.wav"):
        generate_speech("hello", "en", tmp_path / "out.mp3")


def test_tts_language_support_flags():
    assert is_tts_language_supported("fa") is True
    assert is_tts_language_supported("en") is True
    assert is_tts_language_supported("de") is False
