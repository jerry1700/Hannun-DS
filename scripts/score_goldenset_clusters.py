"""골드셋 채점(군집) — 사람 라벨 합의본과 파이프라인 이슈 배정을 ARI 로 비교한다.

라벨 CSV 형식은 파일럿(labels.csv)과 동일: article_id, labeler, cluster. 합의 정답을 만드는
방식은 hannun.quality.goldenset.consensus_components 에 있다.

    python scripts/score_goldenset_clusters.py --labels <labels.csv> --gold-root <루트> \
        --start-date 2023-01-02 --end-date 2023-01-02
    # 파이프라인 출력 대신 CSV 로 채점하려면: --pred-csv <article_id,issue 두 컬럼>
"""

import argparse
import csv
import json

from sklearn.metrics import adjusted_rand_score

from hannun.clustering import IssueStore
from hannun.quality import QualityStore
from hannun.quality.goldenset import consensus_components, labeler_agreement, load_labels


def load_predictions(args):
    if args.pred_csv:
        with open(args.pred_csv, encoding="utf-8-sig", newline="") as f:
            return {row["article_id"]: row["issue"] for row in csv.DictReader(f)}
    frame = QualityStore(args.gold_root).read(args.start_date, args.end_date)
    if frame.empty:
        frame = IssueStore(args.gold_root).read(args.start_date, args.end_date)
    return dict(zip(frame.article_id, frame.issue_local))


def main():
    p = argparse.ArgumentParser(description="골드셋 군집 채점 (티켓 104)")
    p.add_argument("--labels", required=True, help="라벨 CSV (article_id, labeler, cluster)")
    p.add_argument("--min-agree", type=int, default=0, help="합의 최소 라벨러 수. 0 이면 과반")
    p.add_argument("--pred-csv", default="", help="예측 CSV (article_id, issue). 없으면 테이블에서")
    p.add_argument("--gold-root", default="local/gold2023")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    args = p.parse_args()

    by_labeler = load_labels(args.labels)
    min_agree = args.min_agree or (len(by_labeler) // 2 + 1)
    gold = consensus_components(by_labeler, min_agree)
    predicted = load_predictions(args)

    ids = sorted(set(gold) & set(predicted))
    missing = sorted(set(gold) - set(predicted))
    gold_labels = [gold[a] for a in ids]
    # 노이즈(-1)는 "각자 따로" 로 취급 — 뭉뚱그리면 노이즈끼리 같은 이슈로 계산된다
    pred_labels = []
    solo = -1
    for a in ids:
        label = int(predicted[a]) if str(predicted[a]).lstrip("-").isdigit() else predicted[a]
        if isinstance(label, int) and label < 0:
            pred_labels.append(solo)
            solo -= 1
        else:
            pred_labels.append(label)

    scores = {
        "labelers": len(by_labeler),
        "min_agree": min_agree,
        "gold_articles": len(gold),
        "gold_issues": len(set(gold.values())),
        "scored_articles": len(ids),
        "missing_predictions": len(missing),
        "labeler_agreement_ari": labeler_agreement(by_labeler),
        "ari": round(adjusted_rand_score(gold_labels, pred_labels), 4),
    }
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    if missing:
        print(f"예측 없는 기사 {len(missing)}건 예시: {missing[:5]}")


if __name__ == "__main__":
    main()
