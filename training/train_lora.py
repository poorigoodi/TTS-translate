"""LoRA supervised fine-tuning of Qwen3-0.6B for Persian↔English translation.

Example (Windows, GPU):
    python training/prepare_sft_data.py
    python training/train_lora.py --epochs 2 --batch-size 2 --grad-accum 8

After training, point the app at the adapter directory:
    set TRANSLATION_MODEL_ID=checkpoints\\qwen3-fa-en-lora
    python app.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TRANSLATION_MODEL_ID  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "sft"
DEFAULT_OUT = ROOT / "checkpoints" / "qwen3-fa-en-lora"


class ChatJsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 512):
        self.rows: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        messages = self.rows[index]["messages"]
        # Full conversation (user + assistant) with Qwen3 non-thinking template
        full = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        # Prompt-only (user) for label masking
        prompt = self.tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        full_ids = self.tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        labels = full_ids[:]
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen3 FA↔EN translation")
    parser.add_argument("--base-model", default=TRANSLATION_MODEL_ID)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for a quick smoke train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            f"Missing {train_path} or {val_path}. Run:\n"
            f"  python training/prepare_sft_data.py"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Base model: {args.base_model}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Qwen3 attention modules commonly targeted by LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = ChatJsonlDataset(train_path, tokenizer, max_length=args.max_length)
    val_ds = ChatJsonlDataset(val_path, tokenizer, max_length=args.max_length)
    if args.max_train_samples:
        train_ds.rows = train_ds.rows[: args.max_train_samples]
        print(f"Using first {len(train_ds)} train examples (smoke mode)")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device == "cuda",
        bf16=False,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=device == "cuda",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    # Save final adapter + tokenizer next to a small README for the app
    final_dir = args.output_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    meta = {
        "base_model": args.base_model,
        "adapter_dir": str(final_dir),
        "how_to_use": "Set TRANSLATION_MODEL_ID to this directory, then run python app.py",
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    (final_dir / "finetune_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved LoRA adapter to", final_dir)
    print("Use with: set TRANSLATION_MODEL_ID=" + str(final_dir))


if __name__ == "__main__":
    main()
