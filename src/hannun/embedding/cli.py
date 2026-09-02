"""hannun-embed — 대표 기사를 임베딩해 embedding 테이블에 쓰는 명령."""

import argparse
import json
import logging
import sys

from hannun.dedup.store import DedupStore
from hannun.ingest.gold import GoldStore
from hannun.preprocess.store import CleanStore

from .encoder import EncoderConfig
from .pipeline import embed
from .store import EmbeddingStore


def build_parser():
    p = argparse.ArgumentParser(description="대표 기사를 임베딩해 embedding 테이블에 씁니다.")
    p.add_argument("--gold-root", "-g", default="gold", help="Gold 저장소 루트. embedding 도 이 아래에 만든다")
    p.add_argument("--start-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--end-date", default=None, help="UTC YYYY-MM-DD, 포함")
    p.add_argument("--model", default=EncoderConfig.model_name, help="sentence-transformers 모델 이름")
    p.add_argument("--batch-size", type=int, default=EncoderConfig.batch_size)
    p.add_argument("--threads", type=int, default=EncoderConfig.threads, help="0 이면 torch 기본값")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = EncoderConfig(model_name=args.model, batch_size=args.batch_size, threads=args.threads)
    stats = embed(GoldStore(args.gold_root), CleanStore(args.gold_root), DedupStore(args.gold_root),
                  EmbeddingStore(args.gold_root), config,
                  start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0 if stats.rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
