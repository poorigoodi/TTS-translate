"""Application configuration for Qwen Translator."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Translation model — swap to a local fine-tuned checkpoint path when ready
TRANSLATION_MODEL_ID = os.getenv("TRANSLATION_MODEL_ID", "Qwen/Qwen3-0.6B")

# Device: "auto" | "cuda" | "cpu"
DEVICE = os.getenv("DEVICE", "auto")

# Generation limits (conservative defaults for translation)
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "1024"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# Non-thinking mode sampling — tuned to reduce runaway loops on small Qwen3
TRANSLATION_TEMPERATURE = float(os.getenv("TRANSLATION_TEMPERATURE", "0.2"))
TRANSLATION_TOP_P = float(os.getenv("TRANSLATION_TOP_P", "0.8"))
TRANSLATION_TOP_K = int(os.getenv("TRANSLATION_TOP_K", "20"))
TRANSLATION_REPETITION_PENALTY = float(os.getenv("TRANSLATION_REPETITION_PENALTY", "1.25"))
TRANSLATION_NO_REPEAT_NGRAM = int(os.getenv("TRANSLATION_NO_REPEAT_NGRAM", "4"))
# Hard cap for TTS input length (characters) to avoid noise on garbage text
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "400"))

# OmniVoice TTS (official checkpoint: https://huggingface.co/k2-fsa/OmniVoice)
OMNIVOICE_MODEL_ID = os.getenv("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
OMNIVOICE_SAMPLE_RATE = 24000
OMNIVOICE_NUM_STEP = int(os.getenv("OMNIVOICE_NUM_STEP", "32"))
OMNIVOICE_SPEED = float(os.getenv("OMNIVOICE_SPEED", "1.0"))
# Voice design instruct (stable for English; may generalize to Persian)
OMNIVOICE_VOICE_INSTRUCT = os.getenv(
    "OMNIVOICE_VOICE_INSTRUCT",
    "female, young adult, moderate pitch",
)

# Languages
SUPPORTED_LANGUAGES = {
    "fa": "فارسی",
    "en": "English",
}
LANGUAGE_CODES = frozenset(SUPPORTED_LANGUAGES.keys())

# Official OmniVoice language map lists Persian (fa / fas) with ~366h training data:
# https://github.com/k2-fsa/OmniVoice/blob/master/docs/lang_id_name_map.tsv
TTS_SUPPORTED_LANGUAGES = frozenset({"fa", "en"})

# Keep a bounded number of generated WAV files in outputs/
MAX_CACHED_AUDIO_FILES = int(os.getenv("MAX_CACHED_AUDIO_FILES", "20"))


def resolve_device() -> str:
    """Resolve runtime device string for PyTorch / Transformers."""
    preferred = DEVICE.lower().strip()
    if preferred == "cpu":
        return "cpu"
    try:
        import torch

        if preferred == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("DEVICE=cuda requested but CUDA is not available.")
        # auto
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ensure_output_dir() -> Path:
    """Create the output directory if missing and return it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
