"""Qwen Translator — Streamlit UI.

Workflow: Enter text → Select languages → Translate → Generate speech → Play / download.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import (
    ASSETS_DIR,
    OMNIVOICE_MODEL_ID,
    SUPPORTED_LANGUAGES,
    TRANSLATION_MODEL_ID,
    ensure_output_dir,
)
from translator import TranslationError, load_translation_model, translate_text
from tts_engine import TTSError, generate_speech, is_tts_language_supported, load_tts_model

LANG_OPTIONS = list(SUPPORTED_LANGUAGES.keys())
LANG_LABELS = {code: label for code, label in SUPPORTED_LANGUAGES.items()}


def _load_css() -> None:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading translation model…")
def _cached_translator(model_id: str):
    return load_translation_model(model_id)


@st.cache_resource(show_spinner="Loading OmniVoice TTS…")
def _cached_tts(checkpoint: str):
    return load_tts_model(checkpoint)


def _init_state() -> None:
    defaults = {
        "source_lang": "fa",
        "target_lang": "en",
        "input_text": "",
        "translation": None,
        "translation_for_input": None,
        "audio_path": None,
        "audio_for_translation": None,
        "error": None,
        "tts_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _swap_languages() -> None:
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )
    # Preserve user input; clear stale results tied to the previous direction
    st.session_state.translation = None
    st.session_state.translation_for_input = None
    st.session_state.audio_path = None
    st.session_state.audio_for_translation = None
    st.session_state.error = None
    st.session_state.tts_error = None


def _placeholder(source_lang: str) -> str:
    if source_lang == "fa":
        return "متن فارسی خود را وارد کنید..."
    return "Enter your English text..."


def main() -> None:
    st.set_page_config(
        page_title="Qwen Translator",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _load_css()
    _init_state()
    ensure_output_dir()

    st.markdown(
        """
        <div class="qt-hero">
          <h1>Qwen Translator</h1>
          <p>Translate Text. Generate Speech.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Model & device", expanded=False):
        st.caption(f"Translation model: `{TRANSLATION_MODEL_ID}`")
        st.caption(f"TTS checkpoint: `{OMNIVOICE_MODEL_ID}`")
        st.caption(
            "Set `TRANSLATION_MODEL_ID` to a local fine-tuned checkpoint directory "
            "to use a trained model without changing the UI."
        )

    col_src, col_swap, col_tgt = st.columns([5, 1, 5])
    with col_src:
        st.selectbox(
            "Source language",
            options=LANG_OPTIONS,
            format_func=lambda c: LANG_LABELS[c],
            key="source_lang",
        )
    with col_swap:
        st.write("")
        st.write("")
        if st.button("⇄", help="Swap languages", use_container_width=True):
            _swap_languages()
            st.rerun()
    with col_tgt:
        st.selectbox(
            "Target language",
            options=LANG_OPTIONS,
            format_func=lambda c: LANG_LABELS[c],
            key="target_lang",
        )

    if st.session_state.source_lang == st.session_state.target_lang:
        st.warning("Source and target languages must be different. Use ⇄ to swap.")

    input_class = "qt-rtl-input" if st.session_state.source_lang == "fa" else "qt-ltr-input"
    st.markdown(f'<div class="{input_class}">', unsafe_allow_html=True)
    st.text_area(
        "Input text",
        key="input_text",
        height=180,
        placeholder=_placeholder(st.session_state.source_lang),
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Invalidate stale translation when input changes
    if (
        st.session_state.translation is not None
        and st.session_state.translation_for_input != st.session_state.input_text
    ):
        st.session_state.translation = None
        st.session_state.translation_for_input = None
        st.session_state.audio_path = None
        st.session_state.audio_for_translation = None

    translate_clicked = st.button(
        "Translate",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.source_lang == st.session_state.target_lang,
    )

    if translate_clicked:
        st.session_state.error = None
        st.session_state.tts_error = None
        st.session_state.audio_path = None
        st.session_state.audio_for_translation = None
        try:
            with st.spinner("Translating with Qwen3-0.6B…"):
                _cached_translator(TRANSLATION_MODEL_ID)
                result = translate_text(
                    st.session_state.input_text,
                    st.session_state.source_lang,
                    st.session_state.target_lang,
                )
            st.session_state.translation = result
            st.session_state.translation_for_input = st.session_state.input_text
        except TranslationError as exc:
            st.session_state.translation = None
            st.session_state.translation_for_input = None
            st.session_state.error = str(exc)
        except Exception as exc:  # noqa: BLE001
            st.session_state.translation = None
            st.session_state.error = f"Unexpected translation error: {exc}"

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.translation:
        direction = "rtl" if st.session_state.target_lang == "fa" else "ltr"
        st.markdown('<div class="qt-label">Translation</div>', unsafe_allow_html=True)
        # Escape HTML special characters for safe display
        safe = (
            st.session_state.translation.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        st.markdown(
            f'<div class="qt-result {direction}">{safe}</div>',
            unsafe_allow_html=True,
        )
        st.code(st.session_state.translation, language=None)

        st.divider()
        target = st.session_state.target_lang
        if not is_tts_language_supported(target):
            st.info(
                f"Speech generation is unavailable for target language `{target}`. "
                "OmniVoice support in this app is limited to Persian and English."
            )
        else:
            gen_speech = st.button("Generate Speech", use_container_width=True)
            if gen_speech:
                st.session_state.tts_error = None
                try:
                    with st.spinner("Generating speech with OmniVoice…"):
                        _cached_tts(OMNIVOICE_MODEL_ID)
                        audio_path = generate_speech(
                            st.session_state.translation,
                            target,
                        )
                    st.session_state.audio_path = audio_path
                    st.session_state.audio_for_translation = st.session_state.translation
                except TTSError as exc:
                    st.session_state.audio_path = None
                    st.session_state.tts_error = str(exc)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.audio_path = None
                    st.session_state.tts_error = f"Unexpected TTS error: {exc}"

            if st.session_state.tts_error:
                st.error(st.session_state.tts_error)

            if (
                st.session_state.audio_path
                and st.session_state.audio_for_translation == st.session_state.translation
            ):
                audio_file = Path(st.session_state.audio_path)
                if audio_file.exists():
                    audio_bytes = audio_file.read_bytes()
                    st.audio(audio_bytes, format="audio/wav")
                    st.download_button(
                        label="Download WAV",
                        data=audio_bytes,
                        file_name=audio_file.name,
                        mime="audio/wav",
                        use_container_width=True,
                    )
                else:
                    st.warning("Audio file is missing. Please generate speech again.")


if __name__ == "__main__":
    main()
