"""Unit tests for TTS validation and output path handling (no model downloads)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_engine import TTSError, generate_speech, is_tts_language_supported  # noqa: E402


def test_is_tts_language_supported():
    assert is_tts_language_supported("fa")
    assert is_tts_language_supported("EN")
    assert not is_tts_language_supported("fr")


def test_generate_speech_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "audio" / "sample.wav"
    fake_audio = [np.zeros(2400, dtype=np.float32)]
    fake_model = MagicMock()
    fake_model.generate.return_value = fake_audio

    with patch("tts_engine.load_tts_model", return_value=fake_model):
        path = generate_speech("Hello world.", "en", out)

    assert Path(path).exists()
    assert Path(path).suffix == ".wav"
    assert out.parent.exists()


def test_generate_speech_rejects_empty(tmp_path):
    with pytest.raises(TTSError, match="empty"):
        generate_speech("", "en", tmp_path / "a.wav")


def test_generate_speech_rejects_bad_extension(tmp_path):
    with pytest.raises(TTSError, match="wav"):
        generate_speech("Hello", "en", tmp_path / "a.mp3")
