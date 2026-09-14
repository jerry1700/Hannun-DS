"""hannun-quality — 군집 결과에 정형 판별·노이즈 구제를 적용하는 명령."""

import argparse
import json
import logging
import sys

from hannun.clustering.store import IssueStore
from hannun.embedding.store import EmbeddingStore

from .pipeline import qualify
from .rules import QualityConfig
from .store import QualityStore


def build_parser():
    p = argparse.ArgumentParser(description="issue 테이블에 STEP 4 판정을 적용해 issue_quality 테이블에 씁니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="저장소 루트. issue_quality 도 이 아래에 만든다")
    p.add_argument("--start-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--end-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--structured-max-publishers", type=int, default=QualityConfig.structured_max_publishers,
                   help=f"이 수 이하의 언론사만 쓰는 이슈를 정형 후보로 (기본 {QualityConfig.structured_max_publishers})")
    p.add_argument("--structured-min-size", type=int, default=QualityConfig.structured_min_size,
                   help=f"정형 판별 최소 이슈 규모 (기본 {QualityConfig.structured_min_size})")
    p.add_argument("--rescue-min-sim", type=float, default=QualityConfig.rescue_min_sim,
                   help=f"노이즈 구제 최소 코사인 유사도 (기본 {QualityConfig.rescue_min_sim})")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = QualityConfig(
        structured_max_publishers=args.structured_max_publishers,
        structured_min_size=args.structured_min_size,
        rescue_min_sim=args.rescue_min_sim,
    )
    stats = qualify(IssueStore(args.gold_root), EmbeddingStore(args.gold_root),
                    QualityStore(args.gold_root), config,
                    start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
