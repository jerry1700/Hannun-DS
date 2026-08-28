"""hannun-preprocess — Gold 원문을 정제해 clean 테이블에 쓰는 명령."""

import argparse
import json
import logging
import sys

from hannun.ingest.gold import GoldStore

from .clean import PreprocessConfig
from .pipeline import preprocess
from .store import CleanStore


def build_parser():
    p = argparse.ArgumentParser(description="Gold 의 기사 본문을 정제해 clean 테이블에 씁니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="Gold 저장소 루트. clean 도 이 아래에 만든다")
    p.add_argument("--start-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--end-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--min-clean-len", type=int, default=PreprocessConfig.min_clean_len,
                   help="이보다 짧으면 short 로 표시 (기본 100)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = PreprocessConfig(min_clean_len=args.min_clean_len)
    stats = preprocess(GoldStore(args.gold_root), CleanStore(args.gold_root), config,
                       start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
