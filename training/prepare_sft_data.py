"""Convert EN–FA parallel data into chat JSONL splits for Qwen SFT.

Preferred source:
    datasets/translation_pairs_merged.parquet
    columns: pair, corpus, source_lang, target_lang, source_text, target_text

Legacy fallback:
    Tatoeba-style TSV (en_id, english, fa_id, persian)

Writes bidirectional examples (en→fa and fa→en) using the same instructions
as ``translator.py``. Held-out app evaluation samples are excluded from train/val.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT / "datasets" / "translation_pairs_merged.parquet"
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


def read_parquet_pairs(
    path: Path,
    *,
    max_pairs: int | None,
    include_corpora: list[str] | None,
    seed: int,
    max_chars: int,
    blocked: set[str],
) -> tuple[list[tuple[str, str]], dict]:
    cols = ["corpus", "source_lang", "target_lang", "source_text", "target_text"]
    table = pq.read_table(path, columns=cols)
    stats: dict = {"rows_raw": table.num_rows, "source": str(path)}

    src_ok = pc.equal(table["source_lang"], "en")
    tgt_ok = pc.equal(table["target_lang"], "fa")
    table = table.filter(pc.and_(src_ok, tgt_ok))
    stats["rows_en_fa"] = table.num_rows

    if include_corpora:
        wanted = sorted({c.strip() for c in include_corpora if c.strip()})
        mask = pc.is_in(table["corpus"], value_set=pa.array(wanted))
        table = table.filter(mask)
        stats["rows_after_corpus_filter"] = table.num_rows
        stats["include_corpora"] = wanted
    else:
        stats["include_corpora"] = "all"

    n = table.num_rows
    if n == 0:
        raise SystemExit("No rows left after filtering parquet.")

    rng = np.random.default_rng(seed)
    if max_pairs is not None and max_pairs < n:
        idx = np.sort(rng.choice(n, size=int(max_pairs), replace=False))
        table = table.take(idx)
        stats["sampled"] = int(max_pairs)
    else:
        stats["sampled"] = n

    en_list = table["source_text"].to_pylist()
    fa_list = table["target_text"].to_pylist()
    corpus_list = table["corpus"].to_pylist()

    pairs: list[tuple[str, str]] = []
    corpus_counts: dict[str, int] = {}
    skipped_empty = skipped_long = skipped_blocked = 0
    for en, fa, corpus in zip(en_list, fa_list, corpus_list):
        en = (en or "").strip()
        fa = (fa or "").strip()
        if not en or not fa:
            skipped_empty += 1
            continue
        if len(en) > max_chars or len(fa) > max_chars:
            skipped_long += 1
            continue
        if normalize(en) in blocked or normalize(fa) in blocked:
            skipped_blocked += 1
            continue
        pairs.append((en, fa))
        key = str(corpus or "unknown")
        corpus_counts[key] = corpus_counts.get(key, 0) + 1

    stats.update(
        {
            "pairs_used": len(pairs),
            "skipped_empty": skipped_empty,
            "skipped_long": skipped_long,
            "skipped_blocked": skipped_blocked,
            "corpus_counts": dict(sorted(corpus_counts.items(), key=lambda x: -x[1])),
        }
    )
    return pairs, stats


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
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Preferred parallel parquet path.",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=None,
        help="Optional legacy Tatoeba TSV (used only if parquet is missing).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=300_000,
        help="Cap pairs sampled from parquet (None/0 = all). Default 300000.",
    )
    parser.add_argument(
        "--include-corpora",
        type=str,
        default="",
        help="Comma-separated corpus names to keep (e.g. MIZAN,TEP). Empty = all.",
    )
    parser.add_argument("--max-chars", type=int, default=800)
    # remaining → test (held-out; never used for training)
    args = parser.parse_args()

    blocked = load_eval_blocklist(EVAL_SAMPLES)
    source_meta: dict

    if args.parquet.exists():
        max_pairs = None if not args.max_pairs else int(args.max_pairs)
        include = [c.strip() for c in args.include_corpora.split(",") if c.strip()] or None
        pairs, source_meta = read_parquet_pairs(
            args.parquet,
            max_pairs=max_pairs,
            include_corpora=include,
            seed=args.seed,
            max_chars=args.max_chars,
            blocked=blocked,
        )
        print(f"Loaded parquet pairs: {len(pairs)}")
    else:
        tsv_path = args.tsv or DEFAULT_TSV
        if not tsv_path.exists():
            raise SystemExit(
                f"Parquet not found ({args.parquet}) and TSV not found ({tsv_path})."
            )
        raw_pairs = read_tsv_pairs(tsv_path)
        pairs = [
            (en, fa)
            for en, fa in raw_pairs
            if normalize(en) not in blocked and normalize(fa) not in blocked
        ]
        removed = len(raw_pairs) - len(pairs)
        source_meta = {
            "source": str(tsv_path),
            "pairs_total": len(raw_pairs),
            "pairs_used": len(pairs),
            "removed_eval_overlap": removed,
        }
        print(f"Loaded TSV pairs: {len(raw_pairs)}  (removed {removed} overlapping eval texts)")

    train_p, val_p, test_p = split_pairs(pairs, args.seed, args.train_ratio, args.val_ratio)
    train_rows = expand_bidirectional(train_p)
    val_rows = expand_bidirectional(val_p)
    test_rows = expand_bidirectional(test_p)

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "val.jsonl", val_rows)
    write_jsonl(args.out_dir / "test.jsonl", test_rows)

    meta = {
        "source": source_meta,
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
