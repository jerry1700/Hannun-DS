"""hannun-feed — 창의 이슈를 요약해 issue_summary 테이블에 쓰는 명령."""

import argparse
import json
import logging
import sys

from hannun.clustering.registry import RegistryStore
from hannun.embedding.store import EmbeddingStore
from hannun.ingest.gold import GoldStore
from hannun.quality.store import QualityStore

from .pipeline import summarize
from .scoring import FeedConfig
from .store import SummaryStore


def build_parser():
    p = argparse.ArgumentParser(description="issue_quality 를 이슈 단위로 요약해 issue_summary 에 씁니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="저장소 루트. issue_summary 도 이 아래에 만든다")
    p.add_argument("--start-date", default=None, help="현재 창 시작 (UTC YYYY-MM-DD, 포함)")
    p.add_argument("--end-date", default=None, help="현재 창 끝 (UTC YYYY-MM-DD, 포함)")
    p.add_argument("--publisher-weight", type=float, default=FeedConfig.publisher_weight,
                   help="화제성에서 언론사 다양성 가중 (기본 2.0)")
    p.add_argument("--recency-tau-hours", type=float, default=FeedConfig.recency_tau_hours,
                   help="화제성 신선도 반감 상수(시간, 기본 24)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = args.gold_root
    config = FeedConfig(publisher_weight=args.publisher_weight,
                        recency_tau_hours=args.recency_tau_hours)
    stats = summarize(QualityStore(root), EmbeddingStore(root), RegistryStore(root),
                      GoldStore(root), SummaryStore(root), config,
                      start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.issues > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
