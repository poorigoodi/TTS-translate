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


def _validate_text(text: str) -> str:
    if text is None:
        raise TTSError("Text for speech generation is required.")
    cleaned = text.strip()
    if not cleaned:
        raise TTSError("Text for speech generation cannot be empty.")
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
    passes the configured voice-design instruct string (no reference audio).
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

    generate_kwargs: dict[str, Any] = {
        "text": source,
        "num_step": OMNIVOICE_NUM_STEP,
        "speed": OMNIVOICE_SPEED,
    }
    # Optional language hint when the installed API accepts it
    generate_kwargs["language_id"] = lang
    if use_voice_design and OMNIVOICE_VOICE_INSTRUCT.strip():
        generate_kwargs["instruct"] = OMNIVOICE_VOICE_INSTRUCT.strip()

    try:
        with torch.inference_mode():
            try:
                audio = model.generate(**generate_kwargs)
            except TypeError:
                # Older / stricter signatures may not accept language_id
                generate_kwargs.pop("language_id", None)
                audio = model.generate(**generate_kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise TTSError(
            "CUDA out of memory during speech generation. "
            "Try a shorter text or set DEVICE=cpu."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise TTSError(f"OmniVoice generation failed: {exc}") from exc

    if audio is None:
        raise TTSError("OmniVoice returned no audio.")

    waveform = audio[0] if isinstance(audio, (list, tuple)) else audio
    try:
        sf.write(str(path), waveform, OMNIVOICE_SAMPLE_RATE)
    except OSError as exc:
        raise TTSError(f"Failed to write WAV file to {path}: {exc}") from exc

    _cleanup_old_audio(path.parent)
    return str(path.resolve())
