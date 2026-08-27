"""hannun-ingest — 공통 기사 JSON 을 검증해 Gold 에 적재하는 명령."""

import argparse
import json
import logging
import sys

from .gold import GoldStore
from .pipeline import ingest


def build_parser():
    p = argparse.ArgumentParser(description="공통 기사 JSON 을 검증하고 Gold(parquet) 에 적재합니다.")
    p.add_argument("--input", "-i", nargs="+", required=True,
                   help="JSON/JSONL 파일 또는 디렉토리 (여러 개 가능)")
    p.add_argument("--gold-root", "-g", default="gold",
                   help="Gold 저장소 루트 (기본 gold)")
    p.add_argument("--on-conflict", choices=["keep", "replace"], default="keep",
                   help="이미 Gold 에 있는 article_id: keep=기존 유지(기본), replace=덮어씀")
    p.add_argument("--run-id", default=None,
                   help="리젝트 파일명에 쓸 실행 ID (기본: UTC 타임스탬프)")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    store = GoldStore(args.gold_root)
    stats = ingest(args.input, store, on_conflict=args.on_conflict, run_id=args.run_id)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.files > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
