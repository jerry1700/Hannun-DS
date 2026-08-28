"""clean 저장소 — 정제된 본문을 Gold 와 같은 발행일 파티션으로 쌓는다. article_id 로 Gold 와 조인한다."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

CLEAN_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("publisher_id", pa.string()),
        ("published_date", pa.string()),
        ("content_clean", pa.string()),
        ("content_clean_len", pa.int32()),
        ("clean_status", pa.string()),
        ("rules_applied", pa.list_(pa.string())),
        ("removed_chars", pa.int32()),
        ("rules_version", pa.string()),
        ("cleaned_at", pa.timestamp("us", tz="UTC")),
    ]
)


class CleanStore:
    """<root>/clean/published_date=YYYY-MM-DD/clean.parquet.

    Gold 에 컬럼을 덧붙이지 않고 테이블을 따로 두는 이유: 규칙이 바뀔 때 clean 만 다시
    만들면 되고, Gold 적재(ingest)와 정제(preprocess)를 서로 모르는 채 돌릴 수 있다.
    파티션은 통째로 다시 쓴다. 같은 Gold 와 같은 규칙이면 같은 파일이 나온다.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.clean_dir = self.root / "clean"

    def write_partition(self, published_date: str, rows: list[dict]):
        table = pa.Table.from_pylist(rows, schema=CLEAN_SCHEMA).sort_by([("article_id", "ascending")])
        write_parquet_atomic(table, self.partition_path(published_date))
        return table.num_rows

    def read_table(self, start_date: str | None = None, end_date: str | None = None,
                   columns: list[str] | None = None):
        """published_date 범위(UTC, YYYY-MM-DD, 양끝 포함)의 정제 결과를 pyarrow Table 로."""
        dates = [
            d for d in self.partition_dates()
            if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
        ]
        if not dates:
            if columns is None:
                return CLEAN_SCHEMA.empty_table()
            return pa.schema([CLEAN_SCHEMA.field(c) for c in columns]).empty_table()
        return pa.concat_tables([pq.read_table(self.partition_path(d), columns=columns) for d in dates])

    def read(self, start_date: str | None = None, end_date: str | None = None,
             columns: list[str] | None = None):
        return self.read_table(start_date, end_date, columns).to_pandas()

    def partition_path(self, published_date: str):
        return self.clean_dir / f"published_date={published_date}" / "clean.parquet"

    def partition_dates(self):
        if not self.clean_dir.exists():
            return []
        dates = [
            d.name.split("=", 1)[1] for d in self.clean_dir.iterdir()
            if d.is_dir() and d.name.startswith("published_date=") and (d / "clean.parquet").exists()
        ]
        return sorted(dates)
