"""dedup 저장소 — 중복 판정 결과를 Gold 와 같은 발행일 파티션으로 쌓는다. article_id 로 조인한다."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

DEDUP_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("publisher_id", pa.string()),
        ("published_date", pa.string()),
        # null 이면 접히지 않은 기사(대표 또는 단독). 값이 있으면 그 대표 밑으로 접혔다
        ("duplicate_of", pa.string()),
        # 대표 = 접힌 것 포함 묶음 크기, 단독 = 1, 접힌 기사 = 0
        ("duplicate_count", pa.int32()),
        # 접힌 근거: sha256 / cosine / containment / chained. 접히지 않았으면 null
        ("method", pa.string()),
        ("window_start", pa.string()),
        ("window_end", pa.string()),
        ("deduped_at", pa.timestamp("us", tz="UTC")),
    ]
)


class DedupStore:
    """<root>/dedup/published_date=YYYY-MM-DD/dedup.parquet.

    중복은 창(window) 전체를 놓고 계산되므로 한 파티션의 행이 다른 파티션의 기사를
    대표로 가리킬 수 있다. 같은 창을 다시 돌리면 같은 파일이 나온다.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.dedup_dir = self.root / "dedup"

    def write_partition(self, published_date: str, rows: list[dict]):
        table = pa.Table.from_pylist(rows, schema=DEDUP_SCHEMA).sort_by([("article_id", "ascending")])
        write_parquet_atomic(table, self.partition_path(published_date))
        return table.num_rows

    def read_table(self, start_date: str | None = None, end_date: str | None = None,
                   columns: list[str] | None = None):
        """published_date 범위(UTC, YYYY-MM-DD, 양끝 포함)의 판정 결과를 pyarrow Table 로."""
        dates = [
            d for d in self.partition_dates()
            if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
        ]
        if not dates:
            if columns is None:
                return DEDUP_SCHEMA.empty_table()
            return pa.schema([DEDUP_SCHEMA.field(c) for c in columns]).empty_table()
        return pa.concat_tables([pq.read_table(self.partition_path(d), columns=columns) for d in dates])

    def read(self, start_date: str | None = None, end_date: str | None = None,
             columns: list[str] | None = None):
        return self.read_table(start_date, end_date, columns).to_pandas()

    def partition_path(self, published_date: str):
        return self.dedup_dir / f"published_date={published_date}" / "dedup.parquet"

    def partition_dates(self):
        if not self.dedup_dir.exists():
            return []
        dates = [
            d.name.split("=", 1)[1] for d in self.dedup_dir.iterdir()
            if d.is_dir() and d.name.startswith("published_date=") and (d / "dedup.parquet").exists()
        ]
        return sorted(dates)
