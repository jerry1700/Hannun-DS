"""issue_summary 저장소 — 이슈 피드용 창 단위 이슈 요약 (대표 기사·규모·시간 범위)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

SUMMARY_SCHEMA = pa.schema(
    [
        ("window_start", pa.string()),
        ("window_end", pa.string()),
        ("issue_local", pa.int32()),
        # 서비스 이슈 ID (issue_registry 승계 결과). 레지스트리가 없으면 -1
        ("issue_id", pa.int64()),
        ("issue_size", pa.int32()),
        ("publishers", pa.int32()),
        ("structured", pa.bool_()),
        # 대표 기사 — 이슈 중심(centroid)에 가장 가까운 기사 (동률이면 최초 발행)
        ("representative", pa.string()),
        ("first_published_at", pa.timestamp("us", tz="UTC")),
        ("last_published_at", pa.timestamp("us", tz="UTC")),
        ("summarized_at", pa.timestamp("us", tz="UTC")),
    ]
)


class SummaryStore:
    """<root>/issue_summary/window_start=YYYY-MM-DD/issue_summary.parquet."""

    def __init__(self, root):
        self.root = Path(root)
        self.summary_dir = self.root / "issue_summary"

    def write_window(self, window_start: str, rows: list[dict]):
        table = pa.Table.from_pylist(rows, schema=SUMMARY_SCHEMA).sort_by([("issue_local", "ascending")])
        write_parquet_atomic(table, self.window_path(window_start))
        return table.num_rows

    def read_window(self, window_start: str):
        path = self.window_path(window_start)
        if not path.exists():
            return SUMMARY_SCHEMA.empty_table()
        return pq.read_table(path)

    def read(self, window_start: str):
        return self.read_window(window_start).to_pandas()

    def window_path(self, window_start: str):
        return self.summary_dir / f"window_start={window_start}" / "issue_summary.parquet"

    def window_starts(self):
        if not self.summary_dir.exists():
            return []
        starts = [
            d.name.split("=", 1)[1] for d in self.summary_dir.iterdir()
            if d.is_dir() and d.name.startswith("window_start=") and (d / "issue_summary.parquet").exists()
        ]
        return sorted(starts)
