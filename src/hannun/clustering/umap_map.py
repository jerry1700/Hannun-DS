"""고정 UMAP 지도 — 창마다 한 번 학습한 지도와 기사별 좌표를 저장해 매시 재군집이 같은 좌표를 쓴다 (티켓 128).

매시 UMAP 을 처음부터 다시 학습하면 5차원 배치가 매번 미세하게 재배열돼 경계의 이슈가
쪼개졌다 붙었다 한다. 지도를 창(window_start)마다 한 번만 학습하고 그 뒤 들어온 기사는
transform 으로 얹되, 한 번 받은 좌표는 표에 남겨 다시 바꾸지 않는다 — 학습에 쓴 점을 다시
transform 하면 학습 좌표와 다른 자리가 나오기 때문이다. 설정(차원·이웃·seed·임베딩 모델)이
다른 지도는 다시 학습한다.
"""

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hannun.ingest.gold import write_parquet_atomic

POINTS_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("point", pa.list_(pa.float32())),
        ("transformed", pa.bool_()),
    ]
)


class MapStore:
    """<root>/umap_map/window_start=YYYY-MM-DD/{reducer.pkl, points.parquet, meta.json}.

    point 는 지도 위 좌표(umap_dims 개), transformed 는 학습 점(false)인지 뒤에 transform 으로
    얹힌 점(true)인지다. 컬럼 정의는 티켓 128 에 있다.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.map_dir = self.root / "umap_map"

    def coordinates(self, window_start: str, article_ids: list, matrix, config, model: str, fit,
                    refit: bool = False):
        """article_ids 순서의 좌표 행렬과 (새로 학습했는지, transform 한 기사 수) 를 돌려준다.

        지도가 없거나 설정이 다르거나 refit 이면 fit(matrix) 로 학습해 저장한다. fit 은 embedding_
        과 transform 을 가진 학습된 reducer 를 돌려주는 함수다. 있으면 좌표 없는 기사만 transform
        해 표에 덧붙인다.
        """
        setting = {"umap_dims": config.umap_dims, "umap_neighbors": config.umap_neighbors,
                   "seed": config.seed, "model": model}
        window_dir = self.window_dir(window_start)
        if refit or not self._matches(window_dir, setting):
            reducer = fit(matrix)
            points = np.asarray(reducer.embedding_, dtype="float32")
            self._write(window_dir, reducer, article_ids, points, transformed=False, setting=setting)
            return points, True, 0

        stored = self._read_points(window_dir)
        fresh = [i for i, article_id in enumerate(article_ids) if article_id not in stored]
        if fresh:
            with (window_dir / "reducer.pkl").open("rb") as f:
                reducer = pickle.load(f)
            new_points = np.asarray(reducer.transform(matrix[fresh]), dtype="float32")
            for i, point in zip(fresh, new_points):
                stored[article_ids[i]] = point
            self._append(window_dir, [article_ids[i] for i in fresh], new_points)
        points = np.array([stored[article_id] for article_id in article_ids], dtype="float32")
        return points, False, len(fresh)

    def window_dir(self, window_start: str):
        return self.map_dir / f"window_start={window_start}"

    def window_starts(self):
        if not self.map_dir.exists():
            return []
        return sorted(
            d.name.split("=", 1)[1] for d in self.map_dir.iterdir()
            if d.is_dir() and d.name.startswith("window_start=") and (d / "meta.json").exists()
        )

    def _matches(self, window_dir, setting):
        meta_path = window_dir / "meta.json"
        if not meta_path.exists() or not (window_dir / "reducer.pkl").exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return all(meta.get(key) == value for key, value in setting.items())

    def _read_points(self, window_dir):
        table = pq.read_table(window_dir / "points.parquet")
        return {article_id: np.asarray(point, dtype="float32")
                for article_id, point in zip(table.column("article_id").to_pylist(),
                                             table.column("point").to_pylist())}

    def _write(self, window_dir, reducer, article_ids, points, transformed, setting):
        window_dir.mkdir(parents=True, exist_ok=True)
        # 지도 파일도 임시 파일에 쓰고 교체한다 — 쓰다가 죽으면 옛 지도가 남고, 없으면 다음 실행이 다시 학습한다
        tmp = window_dir / "reducer.pkl.tmp"
        with tmp.open("wb") as f:
            pickle.dump(reducer, f)
        os.replace(tmp, window_dir / "reducer.pkl")
        self._write_points(window_dir, article_ids, points, [transformed] * len(article_ids))
        meta = dict(setting, fitted_at=datetime.now(timezone.utc).isoformat(), n_fitted=len(article_ids))
        (window_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                              encoding="utf-8")

    def _append(self, window_dir, article_ids, points):
        table = pq.read_table(window_dir / "points.parquet")
        ids = table.column("article_id").to_pylist() + list(article_ids)
        all_points = table.column("point").to_pylist() + [point.tolist() for point in points]
        flags = table.column("transformed").to_pylist() + [True] * len(article_ids)
        self._write_points(window_dir, ids, all_points, flags)

    def _write_points(self, window_dir, article_ids, points, flags):
        rows = [{"article_id": article_id, "point": [float(x) for x in point], "transformed": flag}
                for article_id, point, flag in zip(article_ids, points, flags)]
        table = pa.Table.from_pylist(rows, schema=POINTS_SCHEMA).sort_by([("article_id", "ascending")])
        write_parquet_atomic(table, window_dir / "points.parquet")
