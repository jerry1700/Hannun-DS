import json

import numpy as np
from conftest import FakeReducer

from hannun.clustering import ClusterConfig, MapStore

D1 = "2026-08-20"
CONFIG = ClusterConfig(umap_dims=2)


def matrix_of(n, offset=0.0):
    return np.arange(n * 3, dtype="float32").reshape(n, 3) + offset


def test_first_call_fits_and_stores_points(tmp_path):
    store = MapStore(tmp_path / "gold")
    ids = ["a", "b", "c"]
    points, fitted, transformed = store.coordinates(D1, ids, matrix_of(3), CONFIG, "m", FakeReducer)

    assert fitted and transformed == 0
    assert points.tolist() == matrix_of(3)[:, :2].tolist()
    assert store.window_starts() == [D1]
    meta = json.loads((store.window_dir(D1) / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_fitted"] == 3 and meta["model"] == "m" and meta["umap_dims"] == 2


def test_second_call_keeps_fitted_coordinates_and_transforms_only_new(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "m", FakeReducer)

    matrix = matrix_of(4, offset=0.5)   # 옛 기사의 벡터가 달라져도 좌표는 표에 남은 값을 쓴다
    points, fitted, transformed = store.coordinates(D1, ["a", "b", "c", "d"], matrix, CONFIG, "m",
                                                    FakeReducer)

    assert not fitted and transformed == 1
    assert points[:3].tolist() == matrix_of(3)[:, :2].tolist()
    assert points[3].tolist() == (matrix[3, :2] + 100.0).tolist()


def test_coordinates_follow_requested_order(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "m", FakeReducer)

    points, _, _ = store.coordinates(D1, ["c", "a"], matrix_of(3)[[2, 0]], CONFIG, "m", FakeReducer)
    assert points.tolist() == matrix_of(3)[[2, 0], :2].tolist()


def test_changed_setting_refits(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "m", FakeReducer)

    _, fitted_same, _ = store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "m", FakeReducer)
    _, fitted_other, _ = store.coordinates(D1, ["a", "b", "c"], matrix_of(3),
                                           ClusterConfig(umap_dims=2, umap_neighbors=7), "m", FakeReducer)
    _, fitted_model, _ = store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "other-model",
                                           FakeReducer)
    assert not fitted_same and fitted_other and fitted_model


def test_refit_flag_replaces_map_and_forgets_transformed_points(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer)
    store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "m", FakeReducer)

    points, fitted, _ = store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "m", FakeReducer,
                                          refit=True)
    assert fitted and points[2].tolist() == matrix_of(3)[2, :2].tolist()   # c 가 학습 점이 됐다


def test_windows_keep_separate_maps(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a"], matrix_of(1), CONFIG, "m", FakeReducer)
    _, fitted, _ = store.coordinates("2026-08-21", ["a"], matrix_of(1), CONFIG, "m", FakeReducer)
    assert fitted and store.window_starts() == [D1, "2026-08-21"]
