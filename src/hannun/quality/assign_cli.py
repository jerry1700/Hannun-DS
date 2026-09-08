"""hannun-assign — 새 대표 기사를 기존 이슈에 임시 배정하는 명령 (15분 루프용)."""

import argparse
import json
import logging
import sys

from hannun.clustering.store import IssueStore
from hannun.embedding.store import EmbeddingStore

from .assign import assign
from .rules import QualityConfig


def build_parser():
    p = argparse.ArgumentParser(description="embedding 에는 있고 issue 에는 없는 기사를 기존 이슈에 배정합니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="저장소 루트")
    p.add_argument("--start-date", default=None, help="현재 창 시작 (UTC YYYY-MM-DD, 포함)")
    p.add_argument("--end-date", default=None, help="현재 창 끝 (UTC YYYY-MM-DD, 포함)")
    p.add_argument("--assign-min-sim", type=float, default=QualityConfig.assign_min_sim,
                   help="배정 최소 코사인 유사도 (기본 0.85, 재군집 대비 일치 0.789 실측)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = QualityConfig(assign_min_sim=args.assign_min_sim)
    stats = assign(IssueStore(args.gold_root), EmbeddingStore(args.gold_root), config,
                   start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.existing > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
