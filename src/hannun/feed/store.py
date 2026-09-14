"""issue_summary 저장소 — 이슈 피드용 창 단위 이슈 요약 (대표 기사·카테고리·규모·시간 범위·화제성)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

SUMMARY_SCHEMA = pa.schema(
    [
        ("window_start", pa.string()),
        ("window_end", pa.string()),
        ("issue_local", pa.int32()),
        ("issue_id", pa.int64()),
        ("issue_size", pa.int32()),
        ("publishers", pa.int32()),
        ("structured", pa.bool_()),
        ("category", pa.string()),
        ("representative", pa.string()),
        ("first_published_at", pa.timestamp("us", tz="UTC")),
        ("last_published_at", pa.timestamp("us", tz="UTC")),
        ("hot_score", pa.float32()),
        ("summarized_at", pa.timestamp("us", tz="UTC")),
    ]
)


class SummaryStore:
    """<root>/issue_summary/window_start=YYYY-MM-DD/issue_summary.parquet.

    issue_id 는 issue_registry 승계 결과(레지스트리가 없으면 -1), representative 는 이슈
    중심에 가장 가까운 기사, category 는 구성 기사 카테고리의 다수결(티켓 34), hot_score 는
    피드 정렬 키(내림차순, 티켓 91)다. 컬럼 정의는 티켓 91 에 있다.
    """

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
