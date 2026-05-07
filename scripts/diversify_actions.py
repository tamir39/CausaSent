"""Apply action rewrites to data/gold/train.json (and optionally val/test).

Reads one or more rewrite files of the form `REWRITES[(aspect, sentiment, cause_text)] = new_action`,
plus an OVERRIDES dict that maps `(aspect, sentiment, old_action) -> [new_action_alternatives]`
for bulk replacement. The script then walks every annotation and applies the rewrite,
distributing alternatives round-robin so the same `old_action` becomes different new_action
strings across its occurrences (combats mode collapse).

Usage:
    python scripts/diversify_actions.py \\
        --rewrites scripts/_actions_batch_001.py scripts/_actions_batch_002.py \\
        --in data/gold/train.json --out data/gold/train.json --inplace

Each batch file should expose:

    OVERRIDES: dict[tuple[str, str, str], list[str]]
        # key = (aspect, sentiment, old_action_exact),
        # value = list of new actions; rotated round-robin across occurrences.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_overrides(paths: list[Path]) -> dict[tuple[str, str, str], list[str]]:
    merged: dict[tuple[str, str, str], list[str]] = {}
    for p in paths:
        mod = load_module(p)
        ov = getattr(mod, "OVERRIDES", {})
        for k, v in ov.items():
            if k in merged:
                print(f"warning: duplicate override key {k!r} in {p.name}", file=sys.stderr)
            merged[k] = list(v)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", nargs="+", required=True, type=Path)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    overrides = load_overrides(args.rewrites)
    print(f"loaded {len(overrides)} override keys from {len(args.rewrites)} batch file(s)")

    data = json.loads(args.inp.read_text(encoding="utf-8"))

    # Round-robin counters so that, e.g., 133x "Duy trì thái độ nhân viên" gets spread
    # across [alt1, alt2, alt3, alt1, alt2, alt3, ...].
    rotor: dict[tuple[str, str, str], int] = defaultdict(int)
    rewrites_applied = Counter()
    untouched = 0

    for review in data:
        for ann in review["annotations"]:
            key = (ann["aspect"], ann["sentiment"], ann["action"])
            if key in overrides:
                alts = overrides[key]
                idx = rotor[key] % len(alts)
                rotor[key] += 1
                ann["action"] = alts[idx]
                rewrites_applied[key] += 1
            else:
                untouched += 1

    print(f"\nrewrites applied: {sum(rewrites_applied.values())}")
    print(f"annotations untouched: {untouched}")
    print(f"\ntop 10 keys rewritten:")
    for k, n in rewrites_applied.most_common(10):
        print(f"  {n:4}x  {k}")

    if args.dry_run:
        print("\n--dry-run: not writing")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
