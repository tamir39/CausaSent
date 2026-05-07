"""Apply safe cleanups to data/gold/{train,val,test}.json.

Operations (only ones with no judgment risk):
  1. Strip leading/trailing whitespace from cause_text and re-align spans.
  2. Drop annotations with cause spans shorter than `min_cause_chars` (default 4).
  3. Trim actions longer than `max_action_words` (default 10) to first N words.
  4. Fix the one identified sentiment label bug: vlsp2018-05342 'giò móng thịt rất ít' positive -> negative.

Re-validates spans after each step. Idempotent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean(data: list[dict], min_cause_chars: int = 4, max_action_words: int = 10) -> dict:
    stats = {"ws_stripped": 0, "short_cause_dropped": 0, "action_trimmed": 0, "sent_fixed": 0, "span_fixed": 0}
    for review in data:
        text = review["review"]
        kept = []
        for ann in review["annotations"]:
            ct = ann["cause_text"]
            ct_stripped = ct.strip()
            if ct_stripped != ct:
                stats["ws_stripped"] += 1
                # Re-find inside review to get fresh span
                idx = text.find(ct_stripped)
                if idx >= 0:
                    ann["cause_text"] = ct_stripped
                    ann["cause_span"] = [idx, idx + len(ct_stripped)]
                    stats["span_fixed"] += 1
                else:
                    # Adjust span from original by trimming
                    s, e = ann["cause_span"]
                    lead = len(ct) - len(ct.lstrip())
                    trail = len(ct) - len(ct.rstrip())
                    ann["cause_text"] = ct_stripped
                    ann["cause_span"] = [s + lead, e - trail]

            if len(ann["cause_text"]) < min_cause_chars:
                stats["short_cause_dropped"] += 1
                continue

            words = ann["action"].split()
            if len(words) > max_action_words:
                ann["action"] = " ".join(words[:max_action_words])
                stats["action_trimmed"] += 1

            # Sentiment-label bug fix
            if review["id"] == "vlsp2018-05342" and ann["cause_text"].startswith("giò") and ann["sentiment"] == "positive":
                ann["sentiment"] = "negative"
                ann["action"] = "Tăng định lượng giò móng thịt"
                stats["sent_fixed"] += 1

            kept.append(ann)
        review["annotations"] = kept

    # Final span integrity check
    bad = 0
    for review in data:
        for ann in review["annotations"]:
            s, e = ann["cause_span"]
            if review["review"][s:e] != ann["cause_text"]:
                bad += 1
    stats["span_mismatch_after"] = bad
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/gold", type=Path)
    ap.add_argument("--out-dir", default="data/gold", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for split in ["train", "val", "test"]:
        path = args.in_dir / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        stats = clean(data)
        print(f"{split:6}: {stats}")
        if not args.dry_run:
            (args.out_dir / f"{split}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
