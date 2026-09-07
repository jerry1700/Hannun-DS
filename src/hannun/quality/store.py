"""issue_quality 저장소 — STEP 4 판정이 반영된 창 단위 최종 이슈 배정."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

QUALITY_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("publisher_id", pa.string()),
        ("published_date", pa.string()),
        # 구제가 반영된 최종 번호(-1 은 여전히 노이즈). 원본 배정은 issue 테이블에 남는다
        ("issue_local", pa.int32()),
        ("issue_size", pa.int32()),
        # 정형(템플릿) 이슈 소속 — 지우지 않고 표시만 한다. 노출 정책은 서비스 몫
        ("structured", pa.bool_()),
        ("rescued", pa.bool_()),
        ("rescue_sim", pa.float32()),
        ("window_start", pa.string()),
        ("window_end", pa.string()),
        ("qualified_at", pa.timestamp("us", tz="UTC")),
    ]
)


class QualityStore:
    """<root>/issue_quality/published_date=YYYY-MM-DD/issue_quality.parquet."""

    def __init__(self, root):
        self.root = Path(root)
        self.quality_dir = self.root / "issue_quality"

    def write_partition(self, published_date: str, rows: list[dict]):
        table = pa.Table.from_pylist(rows, schema=QUALITY_SCHEMA).sort_by([("article_id", "ascending")])
        write_parquet_atomic(table, self.partition_path(published_date))
        return table.num_rows

    def read_table(self, start_date: str | None = None, end_date: str | None = None,
                   columns: list[str] | None = None):
        dates = [
            d for d in self.partition_dates()
            if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
        ]
        if not dates:
            if columns is None:
                return QUALITY_SCHEMA.empty_table()
            return pa.schema([QUALITY_SCHEMA.field(c) for c in columns]).empty_table()
        return pa.concat_tables([pq.read_table(self.partition_path(d), columns=columns) for d in dates])

    def read(self, start_date: str | None = None, end_date: str | None = None,
             columns: list[str] | None = None):
        return self.read_table(start_date, end_date, columns).to_pandas()

    def partition_path(self, published_date: str):
        return self.quality_dir / f"published_date={published_date}" / "issue_quality.parquet"

    def partition_dates(self):
        if not self.quality_dir.exists():
            return []
        dates = [
            d.name.split("=", 1)[1] for d in self.quality_dir.iterdir()
            if d.is_dir() and d.name.startswith("published_date=") and (d / "issue_quality.parquet").exists()
        ]
        return sorted(dates)
