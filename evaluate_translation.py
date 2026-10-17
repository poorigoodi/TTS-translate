"""Lightweight evaluation of Persian ↔ English translation.

Runs held-out examples from ``data/evaluation_samples.jsonl`` against the
configured base model or a fine-tuned checkpoint.

Automatic metrics (e.g. SacreBLEU) do **not** fully measure semantic accuracy
or translation quality — treat them as complementary signals only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from config import DATA_DIR, TRANSLATION_MODEL_ID
from translator import TranslationError, translate_text, unload_translation_model

DEFAULT_SAMPLES = DATA_DIR / "evaluation_samples.jsonl"


def load_samples(path: Path, direction: str | None = None) -> list[dict]:
    samples: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if direction and item.get("direction") != direction:
                continue
            samples.append(item)
    return samples


def parse_direction(direction: str) -> tuple[str, str]:
    src, tgt = direction.split("-", maxsplit=1)
    return src, tgt


def try_sacrebleu(hypotheses: list[str], references: list[str]) -> float | None:
    try:
        from sacrebleu import corpus_bleu
    except ImportError:
        return None
    # SacreBLEU expects a list of reference lists
    result = corpus_bleu(hypotheses, [references])
    return float(result.score)


def run_evaluation(
    samples_path: Path,
    model_id: str,
    direction: str | None,
    limit: int | None,
) -> int:
    samples = load_samples(samples_path, direction=direction)
    if limit is not None:
        samples = samples[:limit]

    if not samples:
        print("No evaluation samples found.")
        return 1

    print(f"Model: {model_id}")
    print(f"Samples file: {samples_path}")
    print(f"Examples: {len(samples)}")
    print("-" * 60)

    successes = 0
    failures = 0
    times: list[float] = []
    hypotheses: list[str] = []
    references: list[str] = []

    for item in samples:
        sample_id = item.get("id", "?")
        direction_key = item["direction"]
        source = item["source"]
        reference = item.get("reference", "")
        src, tgt = parse_direction(direction_key)

        started = time.perf_counter()
        try:
            hypothesis = translate_text(source, src, tgt, model_id=model_id)
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            successes += 1
            hypotheses.append(hypothesis)
            references.append(reference)
            print(f"[{sample_id}] OK ({elapsed:.2f}s)  {src}->{tgt}")
            print(f"  source : {source}")
            print(f"  output : {hypothesis}")
            if reference:
                print(f"  ref    : {reference}")
        except TranslationError as exc:
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            failures += 1
            print(f"[{sample_id}] FAIL ({elapsed:.2f}s)  {src}->{tgt}")
            print(f"  source : {source}")
            print(f"  error  : {exc}")
        print()

    avg_time = statistics.mean(times) if times else 0.0
    print("=" * 60)
    print(f"Successful generations : {successes}")
    print(f"Failed generations     : {failures}")
    print(f"Average inference time : {avg_time:.2f}s")

    if hypotheses and references and all(references):
        bleu = try_sacrebleu(hypotheses, references)
        if bleu is None:
            print(
                "SacreBLEU: not available (install with `pip install sacrebleu`). "
                "Note: automatic metrics do not fully measure semantic accuracy."
            )
        else:
            print(f"SacreBLEU corpus score : {bleu:.2f}")
            print(
                "Note: automatic metrics do not fully measure semantic accuracy "
                "or translation quality."
            )
    else:
        print("SacreBLEU: skipped (missing references or no successful outputs).")

    unload_translation_model()
    return 0 if failures == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen Persian↔English translation on held-out samples."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES,
        help="Path to JSONL evaluation samples.",
    )
    parser.add_argument(
        "--model-id",
        default=TRANSLATION_MODEL_ID,
        help="Hugging Face model ID or local fine-tuned checkpoint directory.",
    )
    parser.add_argument(
        "--direction",
        choices=["fa-en", "en-fa"],
        default=None,
        help="Optional filter for a single translation direction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N matching samples.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Avoid Windows console UnicodeEncodeError on non-ASCII translations
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    args = build_parser().parse_args(argv)
    if not args.samples.exists():
        print(f"Samples file not found: {args.samples}", file=sys.stderr)
        return 1
    return run_evaluation(args.samples, args.model_id, args.direction, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
