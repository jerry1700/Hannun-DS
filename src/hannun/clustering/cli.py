"""hannun-cluster — 대표 기사 벡터를 이슈로 묶어 issue 테이블에 쓰는 명령."""

import argparse
import json
import logging
import sys

from hannun.embedding.store import EmbeddingStore

from .clusterer import ClusterConfig
from .pipeline import cluster
from .store import IssueStore


def build_parser():
    p = argparse.ArgumentParser(description="embedding 테이블의 벡터를 이슈로 묶어 issue 테이블에 씁니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="저장소 루트. issue 도 이 아래에 만든다")
    p.add_argument("--start-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--end-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--min-cluster-size", type=int, default=ClusterConfig.min_cluster_size,
                   help=f"이슈로 인정할 최소 기사 수 (기본 {ClusterConfig.min_cluster_size})")
    p.add_argument("--umap-dims", type=int, default=ClusterConfig.umap_dims, help="0 이면 축소 생략")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = ClusterConfig(min_cluster_size=args.min_cluster_size, umap_dims=args.umap_dims)
    stats = cluster(EmbeddingStore(args.gold_root), IssueStore(args.gold_root), config,
                    start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
