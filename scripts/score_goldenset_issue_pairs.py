"""골드셋 채점(군집 쌍) — "같은 이슈인가 O/X" 쌍 라벨로 군집 품질을 잰다.

라벨 CSV: article_id_a, article_id_b, 판정(O=같은이슈,X=다른이슈) 컬럼(접두 매칭).
층(같은이슈 예측·경계·무작위)은 시트에 없고 여기서 파이프라인 상태로 재유도한다
— 라벨러가 층을 모르는 채 판정해야 편향이 없다.

  묶음 정밀도 = 파이프라인이 같은 이슈로 묶은 쌍 중 사람도 O 인 비율
  과분리율    = 다른 이슈로 갈렸지만 유사도가 높은 쌍(경계) 중 사람이 O 인 비율

    python scripts/score_goldenset_issue_pairs.py --labels <issue_pairs 라벨본> \
        --gold-root <루트> --start-date <일> --end-date <일>
"""

import argparse
import csv
import json


def load_pairs(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    verdict_col = next(c for c in rows[0] if c.startswith("판정"))
    pairs = []
    for row in rows:
        verdict = row[verdict_col].strip().upper()
        if verdict in ("O", "X"):
            pairs.append((row["article_id_a"], row["article_id_b"], verdict == "O"))
    return pairs


def score_pairs(pairs, label_of, vector_of, boundary_sim=0.75):
    """쌍 라벨을 파이프라인 상태와 대조해 층별 지표를 계산한다. 스윕에서도 재사용."""
    known = [(a, b, o) for a, b, o in pairs
             if a in label_of and b in label_of and a in vector_of and b in vector_of]

    strata = {"machine_same": [], "boundary": [], "random": []}
    for a, b, human_same in known:
        machine_same = label_of[a] == label_of[b] and label_of[a] >= 0
        sim = float(vector_of[a] @ vector_of[b])
        if machine_same:
            strata["machine_same"].append(human_same)
        elif sim >= boundary_sim:
            strata["boundary"].append(human_same)
        else:
            strata["random"].append(human_same)

    def rate(items):
        return round(sum(items) / len(items), 4) if items else None

    human_o = [(a, b) for a, b, o in known if o]
    caught = sum(1 for a, b in human_o if label_of[a] == label_of[b] and label_of[a] >= 0)
    return {
        "labeled_pairs": len(pairs),
        "scored_pairs": len(known),
        "같은이슈_예측쌍": {"n": len(strata["machine_same"]),
                       "묶음_정밀도(사람도_O)": rate(strata["machine_same"])},
        "경계쌍(다른이슈_고유사도)": {"n": len(strata["boundary"]),
                             "과분리율(사람은_O)": rate(strata["boundary"])},
        "무작위쌍": {"n": len(strata["random"]), "사람_O_비율": rate(strata["random"])},
        "표본내_동거_재현율": round(caught / len(human_o), 4) if human_o else None,
        "주의": "층화 표본이라 재현율은 표본 안 기준 — 절대값이 아니라 설정 비교용",
    }


def main():
    p = argparse.ArgumentParser(description="골드셋 군집 쌍 채점 (티켓 104)")
    p.add_argument("--labels", required=True, help="쌍 라벨 CSV (article_id_a/b, 판정)")
    p.add_argument("--gold-root", default="local/gold2023")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--boundary-sim", type=float, default=0.75, help="경계 층으로 보는 최소 유사도")
    args = p.parse_args()

    import numpy as np

    from hannun.embedding import EmbeddingStore
    from hannun.quality import QualityStore

    q = QualityStore(args.gold_root).read(args.start_date, args.end_date)
    label_of = dict(zip(q.article_id, q.issue_local))
    emb = EmbeddingStore(args.gold_root).read(args.start_date, args.end_date)
    vector_of = {a: np.array(v, dtype="float32") for a, v in zip(emb.article_id, emb.vector)}

    result = score_pairs(load_pairs(args.labels), label_of, vector_of, args.boundary_sim)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
