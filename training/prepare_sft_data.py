"""Convert Tatoeba-style EN–FA TSV into chat JSONL splits for Qwen SFT.

Input columns (no header):
    en_id <TAB> english <TAB> fa_id <TAB> persian

Writes bidirectional examples (en→fa and fa→en) using the same instructions
as ``translator.py``. Held-out app evaluation samples are excluded from train/val.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = ROOT / "datasets" / "Sentence pairs in English-Persian - 2026-10-18.tsv"
DEFAULT_OUT = ROOT / "data" / "sft"
EVAL_SAMPLES = ROOT / "data" / "evaluation_samples.jsonl"

INSTRUCTION_EN_FA = (
    "You are a professional English-to-Persian translator. "
    "Translate the following English text into natural, accurate Persian. "
    "Preserve the original meaning, tone, names, numbers, and formatting. "
    "Do not explain the translation or add information. "
    "Return only the translated text."
)
INSTRUCTION_FA_EN = (
    "You are a professional Persian-to-English translator. "
    "Translate the following Persian text into natural, accurate English. "
    "Preserve the original meaning, tone, names, numbers, and formatting. "
    "Do not explain the translation or add information. "
    "Return only the translated text."
)


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_eval_blocklist(path: Path) -> set[str]:
    blocked: set[str] = set()
    if not path.exists():
        return blocked
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("source"):
                blocked.add(normalize(item["source"]))
            if item.get("reference"):
                blocked.add(normalize(item["reference"]))
    return blocked


def read_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    text = path.read_text(encoding="utf-8-sig")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            raise ValueError(f"Bad TSV row {line_no}: expected 4 columns, got {len(parts)}")
        en, fa = parts[1].strip(), parts[3].strip()
        if not en or not fa:
            continue
        pairs.append((en, fa))
    return pairs


def make_example(direction: str, source: str, target: str, pair_id: int) -> dict:
    if direction == "en-fa":
        instruction = INSTRUCTION_EN_FA
    elif direction == "fa-en":
        instruction = INSTRUCTION_FA_EN
    else:
        raise ValueError(direction)
    user = f"{instruction}\n\nText:\n{source}"
    return {
        "id": f"{direction}_{pair_id}",
        "direction": direction,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_pairs(
    pairs: list[tuple[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def expand_bidirectional(pairs: list[tuple[str, str]], start_id: int = 0) -> list[dict]:
    rows: list[dict] = []
    for i, (en, fa) in enumerate(pairs):
        pid = start_id + i
        rows.append(make_example("en-fa", en, fa, pid))
        rows.append(make_example("fa-en", fa, en, pid))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EN–FA SFT JSONL splits.")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    # remaining → test (held-out; never used for training)
    args = parser.parse_args()

    if not args.tsv.exists():
        raise SystemExit(f"TSV not found: {args.tsv}")

    blocked = load_eval_blocklist(EVAL_SAMPLES)
    raw_pairs = read_tsv_pairs(args.tsv)
    pairs = [
        (en, fa)
        for en, fa in raw_pairs
        if normalize(en) not in blocked and normalize(fa) not in blocked
    ]
    removed = len(raw_pairs) - len(pairs)
    print(f"Loaded pairs: {len(raw_pairs)}  (removed {removed} overlapping eval texts)")

    train_p, val_p, test_p = split_pairs(pairs, args.seed, args.train_ratio, args.val_ratio)
    train_rows = expand_bidirectional(train_p)
    val_rows = expand_bidirectional(val_p)
    test_rows = expand_bidirectional(test_p)

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "val.jsonl", val_rows)
    write_jsonl(args.out_dir / "test.jsonl", test_rows)

    meta = {
        "source_tsv": str(args.tsv),
        "pairs_total": len(raw_pairs),
        "pairs_used": len(pairs),
        "removed_eval_overlap": removed,
        "pair_splits": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
        "example_counts_bidirectional": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "seed": args.seed,
        "note": "test.jsonl is held-out — do not train on it.",
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote splits under {args.out_dir}")


if __name__ == "__main__":
    main()
