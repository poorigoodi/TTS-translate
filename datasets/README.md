# Parallel datasets

Place Persian–English parallel files here.

## Preferred: merged parquet

- `translation_pairs_merged.parquet`  
  Columns: `pair`, `corpus`, `source_lang`, `target_lang`, `source_text`, `target_text`  
  Language direction in file: `en → fa` (~10.4M pairs; corpora include OpenSubtitles, MIZAN, TEP)

Prepare training splits (samples by default — full set is huge):

```bash
python training/prepare_sft_data.py
# optional:
# python training/prepare_sft_data.py --max-pairs 500000 --include-corpora MIZAN,TEP
```

## Legacy TSV (fallback)

- `Sentence pairs in English-Persian - 2026-10-18.tsv`  
  Tatoeba-style TSV: `en_id <TAB> english <TAB> fa_id <TAB> persian` (no header)

Used automatically only if the parquet file is missing.

## Kaggle

Upload `translation_pairs_merged.parquet` as a Kaggle Dataset and **Add Input** to
`notebooks/kaggle_qwen_lora_finetune.ipynb`. The notebook looks for that filename
under `/kaggle/input`.
