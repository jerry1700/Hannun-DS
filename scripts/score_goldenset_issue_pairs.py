"""골드셋 채점(군집 쌍) — "같은 이슈인가 O/X" 쌍 라벨로 군집 품질을 잰다.

라벨 CSV: article_id_a, article_id_b, 판정(O=같은이슈,X=다른이슈) 컬럼(접두 매칭).
층별 지표의 뜻은 hannun.quality.goldenset.score_pairs 에 있다.

    python scripts/score_goldenset_issue_pairs.py --labels <issue_pairs 라벨본> \
        --gold-root <루트> --start-date <일> --end-date <일>
"""

import argparse
import json

import numpy as np

from hannun.embedding import EmbeddingStore
from hannun.quality import QualityStore
from hannun.quality.goldenset import load_pairs, score_pairs


def main():
    p = argparse.ArgumentParser(description="골드셋 군집 쌍 채점 (티켓 104)")
    p.add_argument("--labels", required=True, help="쌍 라벨 CSV (article_id_a/b, 판정)")
    p.add_argument("--gold-root", default="local/gold2023")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--boundary-sim", type=float, default=0.75, help="경계 층으로 보는 최소 유사도")
    args = p.parse_args()

    quality = QualityStore(args.gold_root).read(args.start_date, args.end_date)
    label_of = dict(zip(quality.article_id, quality.issue_local))
    emb = EmbeddingStore(args.gold_root).read(args.start_date, args.end_date)
    vector_of = {a: np.array(v, dtype="float32") for a, v in zip(emb.article_id, emb.vector)}

    scores = score_pairs(load_pairs(args.labels), label_of, vector_of, args.boundary_sim)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
