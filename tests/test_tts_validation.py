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
    # Speech-band tone so the noise heuristic accepts the mock output.
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    fake_audio = [0.2 * np.sin(2 * np.pi * 800 * t).astype(np.float32)]
    fake_model = MagicMock()
    fake_model.generate.return_value = fake_audio
    fake_model.sampling_rate = sr

    with patch("tts_engine.load_tts_model", return_value=fake_model):
        path = generate_speech("Hello world.", "en", out)

    assert Path(path).exists()
    assert Path(path).suffix == ".wav"
    assert out.parent.exists()
    assert fake_model.generate.call_args.kwargs.get("language") == "en"
    assert "language_id" not in fake_model.generate.call_args.kwargs


def test_generate_speech_persian_skips_voice_design(tmp_path):
    out = tmp_path / "fa.wav"
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    fake_audio = [0.2 * np.sin(2 * np.pi * 800 * t).astype(np.float32)]
    fake_model = MagicMock()
    fake_model.generate.return_value = fake_audio
    fake_model.sampling_rate = sr

    with patch("tts_engine.load_tts_model", return_value=fake_model):
        generate_speech("سلام دنیا، امروز هوا خوب است.", "fa", out)

    kwargs = fake_model.generate.call_args.kwargs
    assert kwargs.get("language") == "fa"
    assert "instruct" not in kwargs


def test_generate_speech_rejects_empty(tmp_path):
    with pytest.raises(TTSError, match="empty"):
        generate_speech("", "en", tmp_path / "a.wav")


def test_generate_speech_rejects_bad_extension(tmp_path):
    with pytest.raises(TTSError, match="wav"):
        generate_speech("Hello", "en", tmp_path / "a.mp3")


def test_generate_speech_rejects_repetitive_text(tmp_path):
    garbage = "سلام " + ("چیزی " * 30)
    with pytest.raises(TTSError, match="repetitive"):
        generate_speech(garbage, "fa", tmp_path / "a.wav")
