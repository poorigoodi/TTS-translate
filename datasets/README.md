# Parallel datasets

Place Persian–English parallel files here.

Current file:

- `Sentence pairs in English-Persian - 2026-10-18.tsv`  
  Tatoeba-style TSV: `en_id <TAB> english <TAB> fa_id <TAB> persian` (no header)

Prepare training splits:

```bash
python training/prepare_sft_data.py
```
