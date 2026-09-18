"""hannun-cluster — 대표 기사 벡터를 이슈로 묶어 issue 테이블에 쓰는 명령."""

import argparse
import json
import logging
import sys

from hannun.embedding.store import EmbeddingStore

from .clusterer import ClusterConfig
from .pipeline import cluster
from .store import IssueStore
from .umap_map import MapStore


def build_parser():
    p = argparse.ArgumentParser(description="embedding 테이블의 벡터를 이슈로 묶어 issue 테이블에 씁니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="저장소 루트. issue·umap_map 도 이 아래에 만든다")
    p.add_argument("--start-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--end-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--min-cluster-size", type=int, default=ClusterConfig.min_cluster_size,
                   help=f"이슈로 인정할 최소 기사 수 (기본 {ClusterConfig.min_cluster_size})")
    p.add_argument("--umap-dims", type=int, default=ClusterConfig.umap_dims, help="0 이면 축소 생략")
    p.add_argument("--new-issue-sim", type=float, default=ClusterConfig.new_issue_sim,
                   help=f"고정 지도의 노이즈를 새 이슈로 묶을 코사인 문턱 (기본 {ClusterConfig.new_issue_sim})")
    p.add_argument("--no-map", action="store_true", help="고정 지도를 쓰지 않고 매번 UMAP 을 새로 학습")
    p.add_argument("--refit-map", action="store_true", help="이 창의 고정 지도를 지금 벡터로 다시 학습")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = ClusterConfig(min_cluster_size=args.min_cluster_size, umap_dims=args.umap_dims,
                           new_issue_sim=args.new_issue_sim)
    map_store = None if args.no_map else MapStore(args.gold_root)
    stats = cluster(EmbeddingStore(args.gold_root), IssueStore(args.gold_root), config,
                    start_date=args.start_date, end_date=args.end_date,
                    map_store=map_store, refit_map=args.refit_map)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
