# Manual Review Collection — Workflow

This is the **only** sanctioned way to add reviews from sites we cannot pull
through public datasets. **No scraping. No automated collection.** Copy-paste
only, into a CSV, in your own browser session.

## Allowed sources

- Google Maps reviews (public business pages).
- App Store / Google Play store reviews (public).
- Public blog reviews / forum posts (Tinhte, Voz, etc.) where the page is open.

## Forbidden

- Shopee, Lazada, Tiki, TikTok Shop — all ToS-prohibit automated collection
  and aggressively gate even manual scraping. Skip.
- Anything behind a login.
- Anything you'd be uncomfortable being attributed to you.

## Target volume

~100–300 reviews total. Bias the selection toward **noisy** Vietnamese:

- mixed sentiment (one aspect good, another bad in the same review)
- slang and abbreviations ("ship lâu vcl", "shop oce", "tem chính hãng nha mn")
- typos and missing diacritics ("dong goi dep" instead of "đóng gói đẹp")
- short informal reviews (≤30 chars are fine)
- sarcasm ("nhanh thật, có 2 tuần thôi")

These are exactly the cases public datasets under-represent and where the
two-head tagger is most likely to fail without targeted training data.

## CSV format

Open `data/raw/manual/manual.csv` in any editor (UTF-8). One row per review:

```csv
review,source_hint
"Ship lâu nhưng đóng gói đẹp",google_maps
"Áo mặc oce shop nha tem chuẩn",playstore
```

`source_hint` is informational — it ends up in the JSONL `source` field but is
not used downstream. The first column header **must** be `review`.

## Normalize to JSONL

```bash
python -m src.data.public_sources \
  --source generic_csv \
  --in data/raw/manual \
  --out data/raw/manual.jsonl \
  --text-col review \
  --tag manual
```

This produces `data/raw/manual.jsonl` with rows
`{"id": "manual-NNNNN", "review": "...", "source": "manual"}`,
length-filtered (15–600 chars) and deduplicated by exact text.

## Combine with public sources

Concatenate JSONL files before weak-labeling:

```bash
cat data/raw/uit_vsfc.jsonl data/raw/visfd.jsonl data/raw/manual.jsonl \
  > data/raw/raw_pool.jsonl
```

Then run `python -m src.data.weak_label --in data/raw/raw_pool.jsonl --out data/processed/pseudo.json`.

## Hard-example pool

If the review is **especially** noisy (sarcasm, heavy typos, mixed sentiment),
prefix the `source_hint` with `hard:` (e.g. `hard:google_maps`). The
`scripts/build_hard_split.py` tool uses that marker to carve out a hard-only
evaluation slice.
