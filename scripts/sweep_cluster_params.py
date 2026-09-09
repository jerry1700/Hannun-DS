"""군집 파라미터 스윕 — 조합별 재군집을 골드셋 400쌍으로 자동 채점한다 (티켓 104).

과분리(기계가 사람 기준보다 잘게 쪼갬)를 줄이는 균형점을 찾는다. UMAP 은 이웃 수별로
한 번만 계산해 캐시하고(가장 비싼 단계), min_cluster_size 는 그 위에서 바꿔 돈다.
운영 테이블은 건드리지 않는다 — 조합별 산출은 스크래치 루트에 쓴다.

    python scripts/sweep_cluster_params.py --gold-root ~/gold \
        --labels ~/issue_pairs_labeled_merged.csv \
        --start-date 2026-09-07 --end-date 2026-09-08 --out-root ~/exp_sweep
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score_goldenset_issue_pairs import load_pairs, score_pairs  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="군집 파라미터 스윕 (티켓 104)")
    p.add_argument("--gold-root", required=True, help="embedding 을 읽을 운영 루트 (읽기 전용)")
    p.add_argument("--labels", required=True, help="쌍 라벨 CSV")
    p.add_argument("--start-date", required=True, help="창 시작 (UTC)")
    p.add_argument("--end-date", required=True, help="창 끝 (UTC)")
    p.add_argument("--out-root", default="local/exp_sweep", help="조합별 스크래치 루트")
    p.add_argument("--min-cluster-sizes", type=int, nargs="+", default=[3, 5, 8])
    p.add_argument("--umap-neighbors", type=int, nargs="+", default=[15, 30, 50])
    args = p.parse_args()

    import numpy as np

    from hannun.clustering import ClusterConfig, IssueStore, cluster, reduce_vectors
    from hannun.embedding import EmbeddingStore
    from hannun.quality import QualityStore, qualify

    embeddings = EmbeddingStore(args.gold_root)
    pairs = load_pairs(args.labels)
    emb = embeddings.read(args.start_date, args.end_date)
    vector_of = {a: np.array(v, dtype="float32") for a, v in zip(emb.article_id, emb.vector)}
    matrix = np.array(emb.vector.tolist(), dtype="float32")
    print(f"창 {args.start_date}~{args.end_date}: 벡터 {len(matrix)}개, 라벨 {len(pairs)}쌍", flush=True)

    rows = []
    for nn in args.umap_neighbors:
        points = reduce_vectors(matrix, ClusterConfig(umap_neighbors=nn))
        print(f"[umap nn={nn}] 축소 완료", flush=True)
        for mcs in args.min_cluster_sizes:
            root = Path(args.out_root) / f"mcs{mcs}_nn{nn}"
            config = ClusterConfig(min_cluster_size=mcs, umap_neighbors=nn)
            stats = cluster(embeddings, IssueStore(root), config,
                            start_date=args.start_date, end_date=args.end_date,
                            reduce_fn=lambda m: points)
            qualify(IssueStore(root), embeddings, QualityStore(root),
                    start_date=args.start_date, end_date=args.end_date)
            q = QualityStore(root).read(args.start_date, args.end_date)
            label_of = dict(zip(q.article_id, q.issue_local))
            score = score_pairs(pairs, label_of, vector_of)
            rows.append({
                "mcs": mcs, "nn": nn,
                "issues": stats.issues, "noise": stats.noise, "largest": stats.largest_issue,
                "precision": score["같은이슈_예측쌍"]["묶음_정밀도(사람도_O)"],
                "n_same": score["같은이슈_예측쌍"]["n"],
                "oversplit": score["경계쌍(다른이슈_고유사도)"]["과분리율(사람은_O)"],
                "recall": score["표본내_동거_재현율"],
            })
            r = rows[-1]
            print(f"  mcs={mcs} nn={nn}: 이슈 {r['issues']} 노이즈 {r['noise']} | "
                  f"정밀도 {r['precision']} (같은이슈쌍 {r['n_same']}) | "
                  f"과분리 {r['oversplit']} | 재현율 {r['recall']}", flush=True)

    print("\n== 요약 (재현율 내림차순) ==")
    print(f"{'mcs':>4} {'nn':>4} {'이슈':>6} {'노이즈':>6} {'정밀도':>8} {'과분리':>8} {'재현율':>8}")
    for r in sorted(rows, key=lambda x: -(x["recall"] or 0)):
        print(f"{r['mcs']:>4} {r['nn']:>4} {r['issues']:>6} {r['noise']:>6} "
              f"{r['precision']!s:>8} {r['oversplit']!s:>8} {r['recall']!s:>8}")


if __name__ == "__main__":
    main()
