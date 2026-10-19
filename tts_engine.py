"""OmniVoice TTS module.

Uses the official ``k2-fsa/OmniVoice`` checkpoint and API.
Speech is generated from translated text for supported target languages.

Persian (``fa``) is listed in the official OmniVoice language map with ~366 hours
of training data. Voice design is primarily trained on Chinese/English; this
module uses Auto Voice by default for broader language coverage, with an optional
voice-design instruct string from config.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from config import (
    LANGUAGE_CODES,
    MAX_CACHED_AUDIO_FILES,
    OMNIVOICE_MODEL_ID,
    OMNIVOICE_NUM_STEP,
    OMNIVOICE_SAMPLE_RATE,
    OMNIVOICE_SPEED,
    OMNIVOICE_VOICE_INSTRUCT,
    TTS_MAX_CHARS,
    TTS_SUPPORTED_LANGUAGES,
    ensure_output_dir,
    resolve_device,
)

logger = logging.getLogger(__name__)

_tts_model: Any = None
_loaded_checkpoint: str | None = None


class TTSError(Exception):
    """Raised when speech generation cannot be completed."""


def is_tts_language_supported(language: str) -> bool:
    """Return True if OmniVoice officially lists this language code."""
    return (language or "").strip().lower() in TTS_SUPPORTED_LANGUAGES


def _looks_repetitive(text: str) -> bool:
    words = text.split()
    if len(words) >= 6:
        streak = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                streak += 1
                if streak >= 4:
                    return True
            else:
                streak = 1
        from collections import Counter

        _w, count = Counter(words).most_common(1)[0]
        if count >= 6 and count / len(words) >= 0.35:
            return True
    return False


def _validate_text(text: str) -> str:
    if text is None:
        raise TTSError("Text for speech generation is required.")
    cleaned = text.strip()
    if not cleaned:
        raise TTSError("Text for speech generation cannot be empty.")
    if len(cleaned) > TTS_MAX_CHARS:
        raise TTSError(
            f"Text is too long for TTS ({len(cleaned)} chars; max {TTS_MAX_CHARS}). "
            "Refuse to speak failed/looping translations."
        )
    if _looks_repetitive(cleaned):
        raise TTSError(
            "Text looks like a repetitive loop. "
            "Translate again before using Speak — TTS cannot read this cleanly."
        )
    return cleaned


def _validate_language(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang not in LANGUAGE_CODES:
        raise TTSError(f"Unsupported language '{language}'. Use 'fa' or 'en'.")
    if lang not in TTS_SUPPORTED_LANGUAGES:
        raise TTSError(
            f"OmniVoice speech generation is not enabled for '{lang}' in this app. "
            "Persian and English are the supported TTS languages."
        )
    return lang


def _validate_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".wav":
        raise TTSError("Output path must end with .wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_tts_model(checkpoint: str | None = None) -> Any:
    """Load (or reuse) the OmniVoice model."""
    global _tts_model, _loaded_checkpoint

    target = checkpoint or OMNIVOICE_MODEL_ID
    if _tts_model is not None and _loaded_checkpoint == target:
        return _tts_model

    try:
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise TTSError(
            "The 'omnivoice' package is not installed. "
            "Install it with: pip install omnivoice"
        ) from exc

    device = resolve_device()
    device_map = "cuda:0" if device == "cuda" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading OmniVoice checkpoint %s on %s", target, device_map)
    try:
        model = OmniVoice.from_pretrained(
            target,
            device_map=device_map,
            dtype=dtype,
        )
    except OSError as exc:
        raise TTSError(
            f"Failed to load OmniVoice checkpoint '{target}'. Details: {exc}"
        ) from exc
    except torch.cuda.OutOfMemoryError as exc:
        raise TTSError(
            "CUDA out of memory while loading OmniVoice. "
            "Try DEVICE=cpu or free GPU memory."
        ) from exc

    _tts_model = model
    _loaded_checkpoint = target
    return model


def unload_tts_model() -> None:
    """Release the cached TTS model."""
    global _tts_model, _loaded_checkpoint
    _tts_model = None
    _loaded_checkpoint = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cleanup_old_audio(directory: Path) -> None:
    """Keep only the newest WAV files to avoid unbounded disk use."""
    wavs = sorted(directory.glob("tts_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in wavs[MAX_CACHED_AUDIO_FILES:]:
        try:
            old.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete old audio file %s: %s", old, exc)


def _estimate_min_duration(text: str, language: str) -> float | None:
    """Return a minimum duration for short clips, else None (use speed).

    OmniVoice often emits static/noise for very short utterances without a
    reference voice (see upstream issue #118). A small floor stabilizes that.
    """
    chars_per_sec = 12.0 if language == "en" else 8.0
    estimated = max(len(text) / chars_per_sec, 1.5)
    if estimated < 3.5:
        return max(3.5, estimated * 1.25)
    return None


def _to_numpy_waveform(audio: Any) -> np.ndarray:
    """Normalize OmniVoice output to a 1-D float32 waveform in [-1, 1]."""
    waveform = audio[0] if isinstance(audio, (list, tuple)) else audio
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise TTSError("OmniVoice returned an empty waveform.")
    peak = float(np.max(np.abs(arr)))
    if peak > 1.0:
        arr = arr / peak
    return arr


def _looks_like_noise(waveform: np.ndarray, sample_rate: int) -> bool:
    """Heuristic: real speech concentrates energy in ~300–3400 Hz."""
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if x.size < int(sample_rate * 0.25):
        return True
    n = min(x.size, sample_rate * 5)
    chunk = x[:n] - x[:n].mean()
    spectrum = np.abs(np.fft.rfft(chunk)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    def band_energy(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(spectrum[mask].sum())

    total = band_energy(20.0, 12000.0) + 1e-12
    speech = band_energy(300.0, 3400.0) / total
    high = band_energy(3400.0, 12000.0) / total
    return speech < 0.25 or high > 0.65


def _run_generate(model: Any, **kwargs: Any) -> np.ndarray:
    with torch.inference_mode():
        audio = model.generate(**kwargs)
    if audio is None:
        raise TTSError("OmniVoice returned no audio.")
    return _to_numpy_waveform(audio)


def generate_speech(
    text: str,
    language: str,
    output_path: str | Path | None = None,
    *,
    use_voice_design: bool = True,
    checkpoint: str | None = None,
) -> str:
    """Generate speech for ``text`` in ``language`` and save a WAV file.

    Returns the absolute path of the written WAV file.

    Uses OmniVoice Auto Voice when ``use_voice_design`` is False, otherwise
    passes the configured voice-design instruct string for English only.
    Voice design is trained mainly on Chinese/English; applying it to Persian
    frequently produces static instead of speech.
    """
    source = _validate_text(text)
    lang = _validate_language(language)

    if output_path is None:
        out_dir = ensure_output_dir()
        stamp = int(time.time())
        output_path = out_dir / f"tts_{lang}_{stamp}_{uuid.uuid4().hex[:8]}.wav"
    path = _validate_output_path(output_path)

    try:
        model = load_tts_model(checkpoint)
    except TTSError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TTSError(f"Unexpected error loading OmniVoice: {exc}") from exc

    sample_rate = int(getattr(model, "sampling_rate", None) or OMNIVOICE_SAMPLE_RATE)
    min_duration = _estimate_min_duration(source, lang)

    # Official API param is ``language`` (code or name), not ``language_id``.
    generate_kwargs: dict[str, Any] = {
        "text": source,
        "language": lang,
        "num_step": OMNIVOICE_NUM_STEP,
        "speed": OMNIVOICE_SPEED,
        "denoise": True,
    }
    if min_duration is not None:
        generate_kwargs["duration"] = min_duration
        generate_kwargs.pop("speed", None)

    # Voice design is stable for English; skip for Persian (Auto Voice).
    apply_instruct = (
        use_voice_design
        and lang == "en"
        and bool(OMNIVOICE_VOICE_INSTRUCT.strip())
    )
    if apply_instruct:
        generate_kwargs["instruct"] = OMNIVOICE_VOICE_INSTRUCT.strip()

    try:
        waveform = _run_generate(model, **generate_kwargs)
        if _looks_like_noise(waveform, sample_rate):
            logger.warning(
                "OmniVoice output looked like noise; retrying with Auto Voice."
            )
            retry_kwargs = {
                k: v
                for k, v in generate_kwargs.items()
                if k != "instruct"
            }
            # Nudge duration upward on retry for short/unstable clips.
            retry_kwargs["duration"] = max(
                float(retry_kwargs.get("duration") or 0.0),
                min_duration or 4.0,
                4.0,
            )
            retry_kwargs.pop("speed", None)
            waveform = _run_generate(model, **retry_kwargs)
            if _looks_like_noise(waveform, sample_rate):
                raise TTSError(
                    "OmniVoice produced noise instead of speech. "
                    "Try a longer sentence, or translate again and Speak."
                )
    except TTSError:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise TTSError(
            "CUDA out of memory during speech generation. "
            "Try a shorter text or set DEVICE=cpu."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise TTSError(f"OmniVoice generation failed: {exc}") from exc

    try:
        sf.write(str(path), waveform, sample_rate, subtype="PCM_16")
    except OSError as exc:
        raise TTSError(f"Failed to write WAV file to {path}: {exc}") from exc

    _cleanup_old_audio(path.parent)
    return str(path.resolve())
