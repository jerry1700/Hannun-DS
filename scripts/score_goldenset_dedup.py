"""골드셋 채점(중복) — 사람이 판정한 기사 쌍으로 dedup 의 precision·recall 을 잰다.

라벨 CSV: article_id_a, article_id_b, 판정(O=같은기사,X=다른기사) 컬럼(접두 매칭).
파이프라인의 "같은 그룹" = 접힘의 대표가 같음 (대표 자신도 자기 그룹).
recall 은 사람이 O 라고 한 쌍 중 파이프라인도 같은 그룹으로 접은 비율(놓침 측정),
precision 은 파이프라인이 같은 그룹으로 접은 라벨 쌍 중 사람도 O 인 비율이다.

    python scripts/score_goldenset_dedup.py --labels <pairs.csv> --gold-root <루트> \
        --start-date 2023-01-02 --end-date 2023-01-02
"""

import argparse
import json

from hannun.dedup import DedupStore
from hannun.quality.goldenset import load_pairs, representative_of


def main():
    p = argparse.ArgumentParser(description="골드셋 중복 채점 (티켓 104)")
    p.add_argument("--labels", required=True, help="쌍 라벨 CSV (article_id_a, article_id_b, 판정)")
    p.add_argument("--gold-root", default="local/gold2023")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    args = p.parse_args()

    dedup_frame = DedupStore(args.gold_root).read(args.start_date, args.end_date,
                                                  columns=["article_id", "duplicate_of"])
    root_of = representative_of(dedup_frame)

    pairs = load_pairs(args.labels)
    known = [(a, b, is_dup) for a, b, is_dup in pairs if a in root_of and b in root_of]
    tp = fn = fp = tn = 0
    for a, b, is_dup in known:
        same = root_of[a] == root_of[b]
        if is_dup and same:
            tp += 1
        elif is_dup:
            fn += 1
        elif same:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    scores = {
        "labeled_pairs": len(pairs),
        "scored_pairs": len(known),
        "skipped_unknown_ids": len(pairs) - len(known),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
    }
    if precision and recall:
        scores["f1"] = round(2 * precision * recall / (precision + recall), 4)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
