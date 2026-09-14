"""embedding 저장소 — 한 번 계산한 벡터를 발행일 파티션으로 쌓는다. article_id 로 조인한다."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

EMBEDDING_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("publisher_id", pa.string()),
        ("published_date", pa.string()),
        ("vector", pa.list_(pa.float32())),
        ("dim", pa.int32()),
        ("model", pa.string()),
        ("encoded_at", pa.timestamp("us", tz="UTC")),
    ]
)


class EmbeddingStore:
    """<root>/embedding/published_date=YYYY-MM-DD/embedding.parquet.

    벡터를 저장해 두는 이유: 배정은 15분마다, 재군집은 매시 도는데 그때마다 같은
    기사를 다시 인코딩하면 인코딩이 전체 비용을 지배한다. 기사가 들어올 때 한 번
    계산하고, 군집화는 여기서 읽기만 한다. 파티션 안에서 article_id 멱등 — 이미
    인코딩된 기사는 pipeline 이 건너뛴다.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.embedding_dir = self.root / "embedding"

    def write_partition(self, published_date: str, rows: list[dict]):
        table = pa.Table.from_pylist(rows, schema=EMBEDDING_SCHEMA).sort_by([("article_id", "ascending")])
        write_parquet_atomic(table, self.partition_path(published_date))
        return table.num_rows

    def read_table(self, start_date: str | None = None, end_date: str | None = None,
                   columns: list[str] | None = None):
        """published_date 범위(UTC, YYYY-MM-DD, 양끝 포함)의 벡터를 pyarrow Table 로."""
        dates = [
            d for d in self.partition_dates()
            if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)
        ]
        if not dates:
            if columns is None:
                return EMBEDDING_SCHEMA.empty_table()
            return pa.schema([EMBEDDING_SCHEMA.field(c) for c in columns]).empty_table()
        return pa.concat_tables([pq.read_table(self.partition_path(d), columns=columns) for d in dates])

    def read(self, start_date: str | None = None, end_date: str | None = None,
             columns: list[str] | None = None):
        return self.read_table(start_date, end_date, columns).to_pandas()

    def partition_path(self, published_date: str):
        return self.embedding_dir / f"published_date={published_date}" / "embedding.parquet"

    def partition_dates(self):
        if not self.embedding_dir.exists():
            return []
        dates = [
            d.name.split("=", 1)[1] for d in self.embedding_dir.iterdir()
            if d.is_dir() and d.name.startswith("published_date=") and (d / "embedding.parquet").exists()
        ]
        return sorted(dates)
