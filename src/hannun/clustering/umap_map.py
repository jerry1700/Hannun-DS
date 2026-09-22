"""고정 UMAP 지도 — 창마다 한 번 학습한 지도와 기사별 좌표를 저장해 매시 재군집이 같은 좌표를 쓴다 (티켓 128).

매시 UMAP 을 처음부터 다시 학습하면 5차원 배치가 매번 미세하게 재배열돼 경계의 이슈가
쪼개졌다 붙었다 한다. 지도를 창(window_start)마다 한 번만 학습하고 그 뒤 들어온 기사는
transform 으로 얹되, 한 번 받은 좌표는 표에 남겨 다시 바꾸지 않는다 — 학습에 쓴 점을 다시
transform 하면 학습 좌표와 다른 자리가 나오기 때문이다. 설정(차원·이웃·seed·임베딩 모델)이
다른 지도는 다시 학습한다.

지도 하나는 파일 셋(reducer.pkl·points.parquet·meta.json)이라 한 세대 디렉터리에 함께 쓰고,
`current` 포인터 파일만 교체해 활성화한다 — 셋 중 하나만 바뀐 채 죽으면 서로 다른 공간의
좌표가 섞인다(티켓 132).
"""

import json
import os
import pickle
import shutil
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
MAP_FILES = ("reducer.pkl", "points.parquet", "meta.json")


class MapStore:
    """<root>/umap_map/window_start=YYYY-MM-DD/{current, gen=<id>/{reducer.pkl, points.parquet, meta.json}}.

    point 는 지도 위 좌표(umap_dims 개), transformed 는 학습 점(false)인지 뒤에 transform 으로
    얹힌 점(true)인지다. current 는 활성 세대 디렉터리 이름 한 줄. 컬럼 정의는 티켓 128 에 있다.
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
        active = self.active_dir(window_start)
        if refit or active is None or not self._matches(active, setting):
            reducer = fit(matrix)
            points = np.asarray(reducer.embedding_, dtype="float32")
            self._write_generation(window_start, reducer, article_ids, points, setting)
            return points, True, 0

        stored = self._read_points(active)
        fresh = [i for i, article_id in enumerate(article_ids) if article_id not in stored]
        if fresh:
            with (active / "reducer.pkl").open("rb") as f:
                reducer = pickle.load(f)
            new_points = np.asarray(reducer.transform(matrix[fresh]), dtype="float32")
            for i, point in zip(fresh, new_points):
                stored[article_ids[i]] = point
            self._append(active, [article_ids[i] for i in fresh], new_points)
        points = np.array([stored[article_id] for article_id in article_ids], dtype="float32")
        return points, False, len(fresh)

    def meta(self, window_start: str):
        """활성 지도의 설정·학습 시각. 지도가 없으면 None."""
        active = self.active_dir(window_start)
        if active is None:
            return None
        return json.loads((active / "meta.json").read_text(encoding="utf-8"))

    def active_dir(self, window_start: str):
        """활성 세대 디렉터리. 세 파일이 다 있어야 한다 — 포인터만 남고 파일이 빠졌으면 없는 것으로.

        포인터 없이 파일이 창 디렉터리에 바로 있는 것은 세대 도입(티켓 132) 전 지도다. 다음
        재학습까지 그대로 쓴다.
        """
        window_dir = self.window_dir(window_start)
        pointer = window_dir / "current"
        candidate = window_dir
        if pointer.exists():
            candidate = window_dir / pointer.read_text(encoding="utf-8").strip()
        if all((candidate / name).exists() for name in MAP_FILES):
            return candidate
        return None

    def window_dir(self, window_start: str):
        return self.map_dir / f"window_start={window_start}"

    def window_starts(self):
        if not self.map_dir.exists():
            return []
        return sorted(
            d.name.split("=", 1)[1] for d in self.map_dir.iterdir()
            if d.is_dir() and d.name.startswith("window_start=") and self.active_dir(d.name.split("=", 1)[1])
        )

    def _matches(self, active, setting):
        meta = json.loads((active / "meta.json").read_text(encoding="utf-8"))
        return all(meta.get(key) == value for key, value in setting.items())

    def _read_points(self, active):
        table = pq.read_table(active / "points.parquet")
        return {article_id: np.asarray(point, dtype="float32")
                for article_id, point in zip(table.column("article_id").to_pylist(),
                                             table.column("point").to_pylist())}

    def _write_generation(self, window_start, reducer, article_ids, points, setting):
        # 새 세대 디렉터리에 셋을 다 쓴 뒤 포인터를 바꾼다 — 쓰다가 죽으면 포인터는 옛 세대를 가리키고,
        # 미완성 디렉터리는 다음 재학습 때 지워진다
        window_dir = self.window_dir(window_start)
        window_dir.mkdir(parents=True, exist_ok=True)
        generation = "gen=" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        gen_dir = window_dir / generation
        gen_dir.mkdir()
        tmp = gen_dir / "reducer.pkl.tmp"
        with tmp.open("wb") as f:
            pickle.dump(reducer, f)
        os.replace(tmp, gen_dir / "reducer.pkl")
        self._write_points(gen_dir, article_ids, points, [False] * len(article_ids))
        meta = dict(setting, generation=generation, fitted_at=datetime.now(timezone.utc).isoformat(),
                    n_fitted=len(article_ids))
        tmp = gen_dir / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, gen_dir / "meta.json")

        pointer_tmp = window_dir / "current.tmp"
        pointer_tmp.write_text(generation, encoding="utf-8")
        os.replace(pointer_tmp, window_dir / "current")
        self._remove_other_generations(window_dir, generation)

    def _remove_other_generations(self, window_dir, generation):
        for entry in window_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("gen=") and entry.name != generation:
                shutil.rmtree(entry)
        # 세대 도입 전 창 디렉터리에 바로 있던 파일도 새 세대로 대체됐으니 지운다
        for name in MAP_FILES:
            if (window_dir / name).exists():
                (window_dir / name).unlink()

    def _append(self, active, article_ids, points):
        table = pq.read_table(active / "points.parquet")
        ids = table.column("article_id").to_pylist() + list(article_ids)
        all_points = table.column("point").to_pylist() + [point.tolist() for point in points]
        flags = table.column("transformed").to_pylist() + [True] * len(article_ids)
        self._write_points(active, ids, all_points, flags)

    def _write_points(self, target_dir, article_ids, points, flags):
        rows = [{"article_id": article_id, "point": [float(x) for x in point], "transformed": flag}
                for article_id, point, flag in zip(article_ids, points, flags)]
        table = pa.Table.from_pylist(rows, schema=POINTS_SCHEMA).sort_by([("article_id", "ascending")])
        write_parquet_atomic(table, target_dir / "points.parquet")
