"""Gold 저장소 — 검증이 끝난 기사를 parquet 으로 쌓고 읽는다."""

import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import CommonArticle, article_id_matches

log = logging.getLogger(__name__)

# pandas 3.0 부터 문자열 기본 dtype 이 str(pyarrow) 로 바뀌었다. DataFrame 을
# 거쳐 쓰면 파티션마다 컬럼 타입이 달라질 수 있어, 행을 pyarrow Table 로 직접
# 만들고 이 스키마로만 쓴다.
GOLD_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("schema_version", pa.string()),
        ("publisher_id", pa.string()),
        ("publisher_name", pa.string()),
        ("source_type", pa.string()),
        ("url", pa.string()),
        ("title", pa.string()),
        ("content", pa.string()),
        ("author", pa.string()),
        ("category", pa.string()),
        ("category_str", pa.string()),
        ("thumbnail_url", pa.string()),
        ("language", pa.string()),
        ("published_at", pa.timestamp("us", tz="UTC")),
        # 파티션 디렉토리 이름과 그대로 맞추기 위해 date32 가 아니라 문자열로 둔다.
        ("published_date", pa.string()),
        ("content_len", pa.int32()),
        ("article_id_verified", pa.bool_()),
        ("ingested_at", pa.timestamp("us", tz="UTC")),
        ("source_ref", pa.string()),
    ]
)


@dataclass
class Reject:
    source_ref: str
    reason: str
    record: object


@dataclass
class UpsertResult:
    written: int = 0
    skipped_existing: int = 0
    replaced: int = 0
    partitions: list[str] = field(default_factory=list)


class GoldStore:
    """<root>/articles/published_date=YYYY-MM-DD/articles.parquet 에 발행일(UTC) 단위로 쌓는다.

    파티션당 파일 하나. 같은 article_id 는 한 번만 들어간다.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.articles_dir = self.root / "articles"
        self.rejects_dir = self.root / "rejects"

    def upsert(self, rows: Iterable[dict], on_conflict: str = "keep"):
        """rows 를 파티션별로 기존 파일과 합쳐 다시 쓴다.

        keep 이 기본인 이유: 같은 파일을 두 번 넣어도 결과가 같아야 하고, STEP 1
        이후 결과가 content 를 참조하는데 그게 아래에서 조용히 바뀌면 재현이
        안 된다. 재크롤링을 반영할 때만 replace 를 명시한다.
        """
        if on_conflict not in ("keep", "replace"):
            raise ValueError(f"on_conflict 는 keep 또는 replace: {on_conflict}")

        result = UpsertResult()
        by_date = defaultdict(list)
        for r in rows:
            by_date[r["published_date"]].append(r)

        for date_str in sorted(by_date):
            new_table = pa.Table.from_pylist(by_date[date_str], schema=GOLD_SCHEMA)
            path = self.partition_path(date_str)

            if path.exists():
                existing = pq.read_table(path).cast(GOLD_SCHEMA)
                existing_ids = existing.column("article_id").to_pylist()
                new_ids = new_table.column("article_id").to_pylist()

                if on_conflict == "keep":
                    existing_set = set(existing_ids)
                    mask = [i not in existing_set for i in new_ids]
                    result.skipped_existing += mask.count(False)
                    result.written += mask.count(True)
                    new_table = new_table.filter(pa.array(mask, pa.bool_()))
                    combined = pa.concat_tables([existing, new_table])
                else:
                    new_set = set(new_ids)
                    keep_mask = [i not in new_set for i in existing_ids]
                    replaced = keep_mask.count(False)
                    result.replaced += replaced
                    result.written += len(new_ids) - replaced
                    existing = existing.filter(pa.array(keep_mask, pa.bool_()))
                    combined = pa.concat_tables([existing, new_table])
            else:
                combined = new_table
                result.written += new_table.num_rows

            # 순서를 고정해 두면 같은 입력에서 같은 파일이 나온다. 재현·diff 용.
            combined = combined.sort_by([("published_at", "ascending"), ("article_id", "ascending")])
            self._atomic_write(combined, path)
            result.partitions.append(date_str)
            log.info(f"partition {date_str} → {combined.num_rows} rows")

        return result

    def write_rejects(self, rejects: list[Reject], run_id: str):
        if not rejects:
            return None
        self.rejects_dir.mkdir(parents=True, exist_ok=True)
        path = self.rejects_dir / f"{run_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rejects:
                row = {"source_ref": r.source_ref, "reason": r.reason, "record": r.record}
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return path

    def read_table(self, start_date: str | None = None, end_date: str | None = None,
                   columns: list[str] | None = None):
        """published_date 범위(UTC, YYYY-MM-DD, 양끝 포함)의 기사를 pyarrow Table 로."""
        dates = [
            d for d in self.partition_dates()
            if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
        ]
        if not dates:
            if columns is None:
                return GOLD_SCHEMA.empty_table()
            return pa.schema([GOLD_SCHEMA.field(c) for c in columns]).empty_table()
        return pa.concat_tables([pq.read_table(self.partition_path(d), columns=columns) for d in dates])

    def read(self, start_date: str | None = None, end_date: str | None = None,
             columns: list[str] | None = None):
        return self.read_table(start_date, end_date, columns).to_pandas()

    def existing_ids(self):
        return set(self.read_table(columns=["article_id"]).column("article_id").to_pylist())

    def partition_path(self, published_date: str):
        return self.articles_dir / f"published_date={published_date}" / "articles.parquet"

    def partition_dates(self):
        if not self.articles_dir.exists():
            return []
        dates = []
        for d in self.articles_dir.iterdir():
            if d.is_dir() and d.name.startswith("published_date=") and (d / "articles.parquet").exists():
                dates.append(d.name.split("=", 1)[1])
        return sorted(dates)

    def _atomic_write(self, table, path):
        write_parquet_atomic(table, path)


def write_parquet_atomic(table, path):
    """임시 파일에 쓰고 교체한다. 쓰다가 죽어도 기존 파티션은 남는다. os.replace 는 Windows 에서도 원자적이다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def to_gold_row(article: CommonArticle, source_ref: str, ingested_at: datetime):
    return {
        "article_id": article.article_id,
        "schema_version": article.schema_version,
        "publisher_id": article.publisher_id,
        "publisher_name": article.publisher_name,
        "source_type": article.source_type.value,
        "url": article.url,
        "title": article.title,
        "content": article.content,
        "author": article.author,
        "category": article.category,
        "category_str": article.category_str,
        "thumbnail_url": article.thumbnail_url,
        "language": article.language,
        "published_at": article.published_at,
        "published_date": article.published_at.date().isoformat(),
        "content_len": len(article.content),
        "article_id_verified": article_id_matches(article),
        "ingested_at": ingested_at,
        "source_ref": source_ref,
    }
