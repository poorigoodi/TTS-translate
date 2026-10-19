"""Qwen Translator — single desktop app (tkinter).

Usage:
    python app.py
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from config import (
    OMNIVOICE_MODEL_ID,
    SUPPORTED_LANGUAGES,
    TRANSLATION_MODEL_ID,
    ensure_output_dir,
    resolve_device,
)
from translation_log import TranslationLogError, append_translation
from translator import load_translation_model, translate_text
from tts_engine import generate_speech, is_tts_language_supported, load_tts_model

LABEL_TO_CODE = {label: code for code, label in SUPPORTED_LANGUAGES.items()}
CODE_TO_LABEL = dict(SUPPORTED_LANGUAGES)
LANG_LABELS = list(CODE_TO_LABEL.values())


class TranslatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Qwen Translator")
        self.geometry("920x600")
        self.minsize(800, 520)
        self.configure(bg="#ffffff")

        self._busy = False
        self._swap_guard = False

        self._build_ui()
        self._apply_fonts()
        self._set_status(f"Ready · {resolve_device()} · {TRANSLATION_MODEL_ID}")

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background="#ffffff")
        style.configure(
            "Sub.TLabel", font=("Segoe UI", 10), foreground="#666666", background="#ffffff"
        )
        style.configure("Tool.TButton", font=("Segoe UI", 9), padding=(8, 4))
        style.configure("Root.TFrame", background="#ffffff")

        root = ttk.Frame(self, padding=16, style="Root.TFrame")
        root.pack(fill="both", expand=True)

        # Use grid so bottom controls never disappear
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)  # text panes grow

        # Header
        header = ttk.Frame(root, style="Root.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Qwen Translator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header, text="Translate Text. Generate Speech.", style="Sub.TLabel"
        ).pack(anchor="w", pady=(0, 10))

        # Languages
        lang_row = ttk.Frame(root, style="Root.TFrame")
        lang_row.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        lang_row.columnconfigure(0, weight=1)
        lang_row.columnconfigure(2, weight=1)

        self.src_lang = tk.StringVar(value=CODE_TO_LABEL["en"])
        self.tgt_lang = tk.StringVar(value=CODE_TO_LABEL["fa"])

        self.src_combo = ttk.Combobox(
            lang_row,
            textvariable=self.src_lang,
            values=LANG_LABELS,
            state="readonly",
            font=("Segoe UI", 11),
        )
        self.src_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.src_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        ttk.Button(lang_row, text="Swap", width=6, command=self._swap_languages).grid(
            row=0, column=1
        )

        self.tgt_combo = ttk.Combobox(
            lang_row,
            textvariable=self.tgt_lang,
            values=LANG_LABELS,
            state="readonly",
            font=("Segoe UI", 11),
        )
        self.tgt_combo.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        self.tgt_combo.bind("<<ComboboxSelected>>", self._on_lang_change)

        # Tool rows above text panes
        tools = ttk.Frame(root, style="Root.TFrame")
        tools.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        tools.columnconfigure(0, weight=1)
        tools.columnconfigure(1, weight=1)

        left_tools = ttk.Frame(tools, style="Root.TFrame")
        left_tools.grid(row=0, column=0, sticky="w")
        ttk.Button(
            left_tools, text="Paste", style="Tool.TButton", command=self._paste_source
        ).pack(side="left")
        ttk.Button(
            left_tools, text="Copy", style="Tool.TButton", command=lambda: self._copy("source")
        ).pack(side="left", padx=(8, 0))

        right_tools = ttk.Frame(tools, style="Root.TFrame")
        right_tools.grid(row=0, column=1, sticky="e")
        ttk.Button(
            right_tools, text="Copy", style="Tool.TButton", command=lambda: self._copy("target")
        ).pack(side="left")
        self.speak_btn = ttk.Button(
            right_tools,
            text="Speak",
            style="Tool.TButton",
            command=self._speak_translation,
        )
        self.speak_btn.pack(side="left", padx=(8, 0))

        # Text panes
        panes = ttk.Frame(root, style="Root.TFrame")
        panes.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)
        panes.rowconfigure(0, weight=1)

        self.source_box = tk.Text(
            panes,
            wrap="word",
            font=("Segoe UI", 12),
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            padx=10,
            pady=10,
            undo=True,
        )
        self.source_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.source_box.insert("1.0", "This is a test sentence.")
        # Explicit paste bindings (Ctrl+V / Shift+Insert)
        self.source_box.bind("<<Paste>>", self._on_paste_event)
        self.source_box.bind("<Control-v>", self._on_paste_event)
        self.source_box.bind("<Control-V>", self._on_paste_event)
        self.source_box.bind("<Shift-Insert>", self._on_paste_event)

        self.target_box = tk.Text(
            panes,
            wrap="word",
            font=("Segoe UI", 12),
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            padx=10,
            pady=10,
        )
        self.target_box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.target_box.configure(state="disabled")

        # Blue Translate button centered under the boxes
        bottom = ttk.Frame(root, style="Root.TFrame")
        bottom.grid(row=4, column=0, sticky="ew")

        self.translate_btn = tk.Button(
            bottom,
            text="Translate",
            command=self._start_translate,
            font=("Segoe UI", 11, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            padx=36,
            pady=8,
            cursor="hand2",
            bd=0,
        )
        self.translate_btn.pack(pady=(4, 10))

        self.progress = ttk.Progressbar(bottom, mode="indeterminate")
        self.progress.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var, style="Sub.TLabel").pack(
            anchor="w", pady=(8, 0)
        )

    # ---- helpers ----
    def _code(self, which: str) -> str:
        label = self.src_lang.get() if which == "source" else self.tgt_lang.get()
        return LABEL_TO_CODE.get(label, "en")

    def _read(self, which: str) -> str:
        box = self.source_box if which == "source" else self.target_box
        return box.get("1.0", "end").strip()

    def _set_target(self, text: str) -> None:
        self.target_box.configure(state="normal")
        self.target_box.delete("1.0", "end")
        self.target_box.insert("1.0", text)
        self.target_box.configure(state="disabled")
        self._apply_fonts()

    def _apply_fonts(self) -> None:
        src_font = ("Tahoma", 12) if self._code("source") == "fa" else ("Segoe UI", 12)
        tgt_font = ("Tahoma", 12) if self._code("target") == "fa" else ("Segoe UI", 12)
        self.source_box.configure(font=src_font)
        disabled = str(self.target_box.cget("state")) == "disabled"
        if disabled:
            self.target_box.configure(state="normal")
        self.target_box.configure(font=tgt_font)
        if disabled:
            self.target_box.configure(state="disabled")

    def _on_lang_change(self, _event=None) -> None:
        if self._swap_guard:
            return
        if self.src_lang.get() == self.tgt_lang.get():
            other = "en" if self._code("source") == "fa" else "fa"
            self.tgt_lang.set(CODE_TO_LABEL[other])
        self._apply_fonts()

    def _swap_languages(self) -> None:
        src_text = self._read("source")
        tgt_text = self._read("target")
        self._swap_guard = True
        a, b = self.src_lang.get(), self.tgt_lang.get()
        self.src_lang.set(b)
        self.tgt_lang.set(a)
        self._swap_guard = False
        if tgt_text:
            self.source_box.delete("1.0", "end")
            self.source_box.insert("1.0", tgt_text)
            self._set_target(src_text)
        else:
            self._set_target("")
        self._apply_fonts()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.translate_btn.configure(state=state)
        self.speak_btn.configure(state=state)
        if busy:
            self.progress.start(12)
            self.translate_btn.configure(bg="#93c5fd")
        else:
            self.progress.stop()
            self.translate_btn.configure(bg="#2563eb")
        if message:
            self._set_status(message)

    def _clipboard_text(self) -> str:
        try:
            return self.clipboard_get()
        except tk.TclError:
            return ""

    def _paste_source(self) -> None:
        """Paste clipboard into the source box (replaces current content)."""
        clip = self._clipboard_text()
        if not clip.strip():
            messagebox.showinfo("Paste", "Clipboard is empty.")
            return
        self.source_box.delete("1.0", "end")
        self.source_box.insert("1.0", clip)
        self.source_box.focus_set()
        self._set_status("Pasted from clipboard.")

    def _on_paste_event(self, event=None):
        """Handle Ctrl+V / Shift+Insert into the source box."""
        clip = self._clipboard_text()
        if not clip:
            return "break"
        try:
            # Replace selection if any, otherwise insert at cursor
            self.source_box.delete("sel.first", "sel.last")
        except tk.TclError:
            pass
        self.source_box.insert("insert", clip)
        self._set_status("Pasted.")
        return "break"

    def _copy(self, which: str) -> None:
        text = self._read(which)
        if not text:
            messagebox.showinfo("Copy", "Nothing to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()
        self._set_status("Copied.")

    # ---- translate ----
    def _start_translate(self) -> None:
        if self._busy:
            return
        text = self._read("source")
        if not text:
            messagebox.showwarning("Input", "Enter text to translate.")
            return
        src, tgt = self._code("source"), self._code("target")
        if src == tgt:
            messagebox.showwarning("Languages", "Source and target must differ.")
            return

        self._set_busy(True, "Translating with Qwen3…")

        def worker() -> None:
            try:
                load_translation_model(TRANSLATION_MODEL_ID)
                out = translate_text(text, src, tgt)
                log_path = None
                log_error = None
                try:
                    log_path = append_translation(text, out, src, tgt)
                except TranslationLogError as exc:
                    log_error = str(exc)
                self.after(0, self._on_translate_ok, out, log_path, log_error)
            except Exception as exc:  # noqa: BLE001
                self.after(0, self._on_error, "Translation error", exc)

        threading.Thread(target=worker, daemon=True).start()

    def _on_translate_ok(
        self,
        result: str,
        log_path: str | None = None,
        log_error: str | None = None,
    ) -> None:
        self._set_target(result)
        if log_error:
            self._set_busy(False, f"Done · Excel log failed: {log_error}")
        elif log_path:
            self._set_busy(False, f"Done · logged to {log_path}")
        else:
            self._set_busy(False, "Done. Press Speak to hear the translation.")

    def _on_error(self, title: str, exc: Exception) -> None:
        self._set_busy(False, f"{title} failed.")
        messagebox.showerror(title, str(exc))

    # ---- TTS: only for translated output ----
    def _speak_translation(self) -> None:
        if self._busy:
            return
        text = self._read("target")
        lang = self._code("target")
        if not text:
            messagebox.showinfo("Speak", "Translate first, then press Speak.")
            return
        if not is_tts_language_supported(lang):
            messagebox.showwarning("Speak", f"TTS not available for {lang}.")
            return

        self._set_busy(True, "Generating speech (OmniVoice)…")

        def worker() -> None:
            try:
                load_tts_model(OMNIVOICE_MODEL_ID)
                ensure_output_dir()
                path = generate_speech(text, lang)
                self.after(0, self._on_speak_ok, path)
            except Exception as exc:  # noqa: BLE001
                self.after(0, self._on_error, "TTS error", exc)

        threading.Thread(target=worker, daemon=True).start()

    def _on_speak_ok(self, path: str) -> None:
        self._set_busy(False, f"Playing · saved {path}")
        try:
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            try:
                import os

                os.startfile(path)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                messagebox.showinfo("Audio", f"Saved:\n{path}\n\nPlay failed: {exc}")


def main() -> None:
    app = TranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
