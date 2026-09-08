"""골드셋 채점(군집) — 사람 라벨 합의본과 파이프라인 이슈 배정을 ARI 로 비교한다.

라벨 CSV 형식은 파일럿(labels.csv)과 동일: article_id, labeler, cluster.
라벨러마다 이슈 번호가 제각각이라 "두 기사를 같은 이슈로 묶은 라벨러 수"가
min-agree 이상인 쌍을 잇고, 연결 요소를 합의 정답으로 삼는다(파일럿 방식 그대로).

    python scripts/score_goldenset_clusters.py --labels <labels.csv> --gold-root <루트> \
        --start-date 2023-01-02 --end-date 2023-01-02
    # 파이프라인 출력 대신 CSV 로 채점하려면: --pred-csv <article_id,issue 두 컬럼>
"""

import argparse
import csv
import json


def load_labels(path):
    by_labeler = {}
    # utf-8-sig: 엑셀을 거친 CSV 는 BOM 이 붙는다
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            by_labeler.setdefault(row["labeler"], {})[row["article_id"]] = row["cluster"]
    return by_labeler


def consensus_components(by_labeler, min_agree):
    """쌍 득표 min_agree 이상을 잇고 연결 요소를 합의 이슈로 만든다."""
    ids = sorted({a for clusters in by_labeler.values() for a in clusters})
    together = {}
    for clusters in by_labeler.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if a in clusters and b in clusters and clusters[a] == clusters[b]:
                    together[(a, b)] = together.get((a, b), 0) + 1

    neighbors = {a: set() for a in ids}
    for (a, b), votes in together.items():
        if votes >= min_agree:
            neighbors[a].add(b)
            neighbors[b].add(a)

    gold, visited = {}, set()
    label = 0
    for a in ids:
        if a in visited:
            continue
        stack = [a]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            gold[node] = label
            stack.extend(neighbors[node] - visited)
        label += 1
    return gold


def labeler_agreement(by_labeler):
    """라벨러 간 평균 쌍별 ARI — 골드셋 자체의 신뢰도 지표."""
    from sklearn.metrics import adjusted_rand_score

    names = sorted(by_labeler)
    scores = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            common = sorted(set(by_labeler[x]) & set(by_labeler[y]))
            if len(common) >= 2:
                scores.append(adjusted_rand_score([by_labeler[x][a] for a in common],
                                                  [by_labeler[y][a] for a in common]))
    return round(sum(scores) / len(scores), 4) if scores else None


def load_predictions(args, ids):
    if args.pred_csv:
        with open(args.pred_csv, encoding="utf-8-sig", newline="") as f:
            return {row["article_id"]: row["issue"] for row in csv.DictReader(f)}
    from hannun.quality import QualityStore

    df = QualityStore(args.gold_root).read(args.start_date, args.end_date)
    if df.empty:
        from hannun.clustering import IssueStore
        df = IssueStore(args.gold_root).read(args.start_date, args.end_date)
    return dict(zip(df.article_id, df.issue_local))


def main():
    p = argparse.ArgumentParser(description="골드셋 군집 채점 (티켓 104)")
    p.add_argument("--labels", required=True, help="라벨 CSV (article_id, labeler, cluster)")
    p.add_argument("--min-agree", type=int, default=0, help="합의 최소 라벨러 수. 0 이면 과반")
    p.add_argument("--pred-csv", default="", help="예측 CSV (article_id, issue). 없으면 테이블에서")
    p.add_argument("--gold-root", default="local/gold2023")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    args = p.parse_args()

    from sklearn.metrics import adjusted_rand_score

    by_labeler = load_labels(args.labels)
    min_agree = args.min_agree or (len(by_labeler) // 2 + 1)
    gold = consensus_components(by_labeler, min_agree)
    predicted = load_predictions(args, set(gold))

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

    result = {
        "labelers": len(by_labeler),
        "min_agree": min_agree,
        "gold_articles": len(gold),
        "gold_issues": len(set(gold.values())),
        "scored_articles": len(ids),
        "missing_predictions": len(missing),
        "labeler_agreement_ari": labeler_agreement(by_labeler),
        "ari": round(adjusted_rand_score(gold_labels, pred_labels), 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        print(f"예측 없는 기사 {len(missing)}건 예시: {missing[:5]}")


if __name__ == "__main__":
    main()
