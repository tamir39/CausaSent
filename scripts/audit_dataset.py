"""Comprehensive dataset audit — every quality category I can think of."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict


def main() -> None:
    splits = {s: json.load(open(f'data/gold/{s}.json', encoding='utf-8')) for s in ['train', 'val', 'test']}
    all_anns = [(s, r, a) for s, data in splits.items() for r in data for a in r['annotations']]
    print(f"=== AUDIT — {len(all_anns)} annotations across train/val/test ===\n")

    # 1. Span correctness
    print("[1] SPAN INTEGRITY")
    mismatch = 0
    for s, r, a in all_anns:
        cs, ce = a['cause_span']
        if r['review'][cs:ce] != a['cause_text']:
            mismatch += 1
    print(f"   span/cause_text mismatches: {mismatch} (should be 0)\n")

    # 2. Action length outliers
    print("[2] ACTION LENGTH OUTLIERS")
    empty_act = sum(1 for _, _, a in all_anns if not a.get('action', '').strip())
    too_short = sum(1 for _, _, a in all_anns if a['action'].strip() and len(a['action'].split()) < 2)
    too_long = sum(1 for _, _, a in all_anns if len(a['action'].split()) > 10)
    not_titled = sum(1 for _, _, a in all_anns if a['action'] and not a['action'][0].isupper())
    print(f"   empty action: {empty_act}")
    print(f"   <2 words: {too_short}")
    print(f"   >10 words (PRD limit): {too_long}")
    print(f"   not starting with capital: {not_titled}\n")

    # 3. Cause length outliers
    print("[3] CAUSE LENGTH OUTLIERS")
    clens = [len(a['cause_text']) for _, _, a in all_anns]
    print(f"   chars: min={min(clens)} median={statistics.median(clens):.0f} max={max(clens)}")
    print(f"   <4 chars: {sum(1 for c in clens if c < 4)}  |  >80 chars: {sum(1 for c in clens if c > 80)}\n")

    # 4. Sentiment-action sign consistency
    print("[4] SENTIMENT-ACTION SIGN MISMATCH")
    maintain_verbs = ('duy trì', 'giữ', 'phát huy', 'tiếp tục', 'khen thưởng', 'quảng bá', 'nhấn mạnh')
    fix_verbs = ('cải', 'đào tạo', 'mở rộng', 'tăng', 'giảm', 'nâng', 'thay', 'sửa', 'rà soát',
                 'bổ sung', 'cân nhắc', 'cập nhật', 'tối ưu', 'thêm', 'đổi', 'huấn luyện', 'đầu tư', 'đánh giá')
    neg_with_maintain = []
    pos_with_fix = []
    for s, r, a in all_anns:
        act = a['action'].lower().strip()
        if a['sentiment'] == 'negative' and any(act.startswith(v) for v in maintain_verbs):
            neg_with_maintain.append((s, r['id'], a['cause_text'][:30], a['action']))
        if a['sentiment'] == 'positive' and any(act.startswith(v) for v in fix_verbs):
            pos_with_fix.append((s, r['id'], a['cause_text'][:30], a['action']))
    print(f"   negative + maintain verb (likely WRONG): {len(neg_with_maintain)}")
    for s, rid, c, act in neg_with_maintain[:8]:
        print(f"     {s:5} {rid:25}  cause={c!r:35} action={act!r}")
    print(f"   positive + fix verb (likely WRONG): {len(pos_with_fix)}")
    for s, rid, c, act in pos_with_fix[:8]:
        print(f"     {s:5} {rid:25}  cause={c!r:35} action={act!r}")
    print()

    # 5. Duplicate review texts
    print("[5] DUPLICATE / NEAR-DUPLICATE REVIEWS")
    text_to_ids = defaultdict(list)
    for s, data in splits.items():
        for r in data:
            text_to_ids[r['review']].append((s, r['id']))
    dups = {t: ids for t, ids in text_to_ids.items() if len(ids) > 1}
    print(f"   exact duplicate review texts: {len(dups)}")
    for t, ids in list(dups.items())[:5]:
        print(f"     {len(ids)}x: {t[:70]!r}  (ids: {[i for s, i in ids[:5]]})")
    print()

    # 6. Split leakage by id and by text
    print("[6] SPLIT LEAKAGE")
    ids = {s: set(r['id'] for r in data) for s, data in splits.items()}
    print(f"   train ∩ val ids:  {len(ids['train'] & ids['val'])}")
    print(f"   train ∩ test ids: {len(ids['train'] & ids['test'])}")
    print(f"   val ∩ test ids:   {len(ids['val'] & ids['test'])}")
    texts = {s: set(r['review'] for r in data) for s, data in splits.items()}
    print(f"   text leak train↔val:  {len(texts['train'] & texts['val'])}")
    print(f"   text leak train↔test: {len(texts['train'] & texts['test'])}")
    print(f"   text leak val↔test:   {len(texts['val'] & texts['test'])}\n")

    # 7. Annotation count distribution
    print("[7] ANNOTATIONS PER REVIEW")
    for s, data in splits.items():
        counts = [len(r['annotations']) for r in data]
        zero = sum(1 for c in counts if c == 0)
        print(f"   {s:6}: reviews={len(data):4} zero-ann={zero:3} "
              f"min={min(counts)} median={statistics.median(counts):.0f} max={max(counts)} "
              f"avg={statistics.mean(counts):.2f}")
    print()

    # 8. Cause words leaked into action
    print("[8] CAUSE-WORDS COPIED INTO ACTION (laziness signal)")
    copy_full = 0
    copy_3plus = 0
    for _, _, a in all_anns:
        cause_words = {w for w in re.findall(r'\w+', a['cause_text'].lower()) if len(w) >= 3}
        act_words = set(re.findall(r'\w+', a['action'].lower()))
        if cause_words and cause_words <= act_words:
            copy_full += 1
        if len(cause_words & act_words) >= 3:
            copy_3plus += 1
    n = len(all_anns)
    print(f"   action contains ALL content-words from cause: {copy_full} / {n} ({copy_full/n*100:.1f}%)")
    print(f"   >=3 cause content-words appear in action:     {copy_3plus} / {n} ({copy_3plus/n*100:.1f}%)\n")

    # 9. Lazy templates
    print("[9] LAZY TEMPLATE PATTERNS (post-diversify)")
    patterns = [
        (r'^Duy trì \w+ \w+$', 'Duy trì <X> <Y> (2 trailing words)'),
        (r'^Giữ giá \w+$', 'Giữ giá <X>'),
        (r'^Quảng bá vị trí', 'Quảng bá vị trí ...'),
        (r'^Cải thiện \w+$', 'Cải thiện <X>'),
        (r'^Đào tạo lại \w+$', 'Đào tạo lại <X>'),
    ]
    for pat, name in patterns:
        cnt = sum(1 for _, _, a in all_anns if re.match(pat, a['action']))
        print(f"   {name:42} matches: {cnt}")
    print()

    # 10. Aspect/sentiment cell counts
    print("[10] ASPECT/SENTIMENT CELL COUNTS")
    cells = Counter((a['aspect'], a['sentiment']) for _, _, a in all_anns)
    for asp in ['delivery', 'packaging', 'product_quality', 'price', 'customer_service', 'usability', 'appearance']:
        line = f"   {asp:18}"
        for sent in ['positive', 'negative', 'neutral']:
            line += f"  {sent[:3]}={cells.get((asp, sent), 0):4}"
        total = sum(cells.get((asp, s), 0) for s in ['positive', 'negative', 'neutral'])
        line += f"  total={total:4}"
        print(line)
    print()

    # 11. Encoding / weird chars in reviews
    print("[11] ENCODING / WEIRD CHARS IN REVIEWS")
    seen = set()
    flagged = 0
    for s, r, _ in all_anns:
        if r['id'] in seen:
            continue
        t = r['review']
        if '\x00' in t or '�' in t or t.count('?') > 5 or '\\u' in t:
            print(f"   {s} {r['id']}: {t[:80]!r}")
            seen.add(r['id'])
            flagged += 1
            if flagged >= 8:
                break
    if flagged == 0:
        print("   none found")
    print()

    # 12. Whitespace
    print("[12] WHITESPACE NORMALIZATION")
    lt = sum(1 for _, _, a in all_anns if a['cause_text'] != a['cause_text'].strip())
    dsc = sum(1 for _, _, a in all_anns if '  ' in a['cause_text'])
    dsa = sum(1 for _, _, a in all_anns if '  ' in a.get('action', ''))
    print(f"   cause with leading/trailing whitespace: {lt}")
    print(f"   cause with double-space inside:         {dsc}")
    print(f"   action with double-space inside:        {dsa}")


if __name__ == "__main__":
    main()
