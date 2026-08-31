"""hannun-dedup — Gold 의 기사에서 중복을 판정해 dedup 테이블에 쓰는 명령."""

import argparse
import json
import logging
import sys

from hannun.ingest.gold import GoldStore
from hannun.preprocess.store import CleanStore

from .pipeline import DedupConfig, dedup
from .store import DedupStore


def build_parser():
    p = argparse.ArgumentParser(description="날짜 범위의 기사 전체를 한 창으로 놓고 중복을 판정합니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="Gold 저장소 루트. dedup 도 이 아래에 만든다")
    p.add_argument("--start-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--end-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = args.gold_root
    stats = dedup(GoldStore(root), CleanStore(root), DedupStore(root), DedupConfig(),
                  start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
