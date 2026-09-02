"""issue 저장소 — 창 단위 군집 결과를 발행일 파티션으로 쌓는다. article_id 로 조인한다."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

ISSUE_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("publisher_id", pa.string()),
        ("published_date", pa.string()),
        # 창 안에서만 유효한 임시 번호. -1 은 노이즈. 서비스 issueId 승계는 티켓 97
        ("issue_local", pa.int32()),
        ("issue_size", pa.int32()),
        ("window_start", pa.string()),
        ("window_end", pa.string()),
        ("clustered_at", pa.timestamp("us", tz="UTC")),
    ]
)


class IssueStore:
    """<root>/issue/published_date=YYYY-MM-DD/issue.parquet.

    dedup 과 같은 창 의미론이다 — 마지막으로 그 파티션을 덮은 창의 판정이 남는다.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.issue_dir = self.root / "issue"

    def write_partition(self, published_date: str, rows: list[dict]):
        table = pa.Table.from_pylist(rows, schema=ISSUE_SCHEMA).sort_by([("article_id", "ascending")])
        write_parquet_atomic(table, self.partition_path(published_date))
        return table.num_rows

    def read_table(self, start_date: str | None = None, end_date: str | None = None,
                   columns: list[str] | None = None):
        """published_date 범위(UTC, YYYY-MM-DD, 양끝 포함)의 군집 결과를 pyarrow Table 로."""
        dates = [
            d for d in self.partition_dates()
            if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
        ]
        if not dates:
            if columns is None:
                return ISSUE_SCHEMA.empty_table()
            return pa.schema([ISSUE_SCHEMA.field(c) for c in columns]).empty_table()
        return pa.concat_tables([pq.read_table(self.partition_path(d), columns=columns) for d in dates])

    def read(self, start_date: str | None = None, end_date: str | None = None,
             columns: list[str] | None = None):
        return self.read_table(start_date, end_date, columns).to_pandas()

    def partition_path(self, published_date: str):
        return self.issue_dir / f"published_date={published_date}" / "issue.parquet"

    def partition_dates(self):
        if not self.issue_dir.exists():
            return []
        dates = [
            d.name.split("=", 1)[1] for d in self.issue_dir.iterdir()
            if d.is_dir() and d.name.startswith("published_date=") and (d / "issue.parquet").exists()
        ]
        return sorted(dates)
