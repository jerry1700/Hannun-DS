"""hannun-succeed — 재군집 직후 서비스 이슈 ID 를 승계·발급하는 명령."""

import argparse
import json
import logging
import sys

from .registry import RegistryStore
from .store import IssueStore
from .succession import SuccessionConfig, succeed


def build_parser():
    p = argparse.ArgumentParser(description="현재 창의 군집에 서비스 이슈 ID 를 배정해 issue_registry 에 씁니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="저장소 루트. issue_registry 도 이 아래에 만든다")
    p.add_argument("--start-date", default=None, help="현재 창 시작 (UTC YYYY-MM-DD, 포함)")
    p.add_argument("--end-date", default=None, help="현재 창 끝 (UTC YYYY-MM-DD, 포함)")
    p.add_argument("--min-shared", type=int, default=SuccessionConfig.min_shared,
                   help="승계 인정 최소 겹침 기사 수 (기본 2)")
    p.add_argument("--min-overlap-frac", type=float, default=SuccessionConfig.min_overlap_frac,
                   help="승계 인정 최소 겹침 비율 (기본 0.5, 과반)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = SuccessionConfig(min_shared=args.min_shared, min_overlap_frac=args.min_overlap_frac)
    stats = succeed(IssueStore(args.gold_root), RegistryStore(args.gold_root), config,
                    start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.issues > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
