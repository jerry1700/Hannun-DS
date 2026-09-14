"""issue_registry 저장소 — 창(재군집)마다 기사 → 서비스 이슈 ID 매핑을 남긴다.

창 단위 파티션(window_start)이라 슬라이딩 창이 issue 파티션을 덮어써도
직전 창의 배정 이력이 보존된다 — 승계는 이 이력만으로 동작한다.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

REGISTRY_SCHEMA = pa.schema(
    [
        ("window_start", pa.string()),
        ("window_end", pa.string()),
        ("article_id", pa.string()),
        ("published_date", pa.string()),
        ("issue_local", pa.int32()),
        ("issue_id", pa.int64()),
        ("status", pa.string()),
        ("succeeded_at", pa.timestamp("us", tz="UTC")),
    ]
)


class RegistryStore:
    """<root>/issue_registry/window_start=YYYY-MM-DD/registry.parquet.

    issue_local 은 창 안의 임시 번호(issue 테이블과의 조인 키), issue_id 는 재군집을 넘어
    유지되는 서비스 이슈 ID 다. status 는 직전 창에서 이어받았으면 inherited, 이 창에서
    태어났으면 new. 컬럼 정의는 티켓 97 에 있다.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.registry_dir = self.root / "issue_registry"

    def write_window(self, window_start: str, rows: list[dict]):
        table = pa.Table.from_pylist(rows, schema=REGISTRY_SCHEMA).sort_by([("article_id", "ascending")])
        write_parquet_atomic(table, self.window_path(window_start))
        return table.num_rows

    def read_window(self, window_start: str):
        path = self.window_path(window_start)
        if not path.exists():
            return REGISTRY_SCHEMA.empty_table()
        return pq.read_table(path)

    def window_path(self, window_start: str):
        return self.registry_dir / f"window_start={window_start}" / "registry.parquet"

    def window_starts(self):
        if not self.registry_dir.exists():
            return []
        starts = [
            d.name.split("=", 1)[1] for d in self.registry_dir.iterdir()
            if d.is_dir() and d.name.startswith("window_start=") and (d / "registry.parquet").exists()
        ]
        return sorted(starts)

    def max_issue_id(self):
        """지금까지 모든 창에 발급된 최대 서비스 ID. 아직 없으면 -1."""
        highest = -1
        for start in self.window_starts():
            column = pq.read_table(self.window_path(start), columns=["issue_id"]).column("issue_id")
            if len(column):
                highest = max(highest, pc.max(column).as_py())
        return highest
