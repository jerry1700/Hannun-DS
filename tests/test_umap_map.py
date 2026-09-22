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
    meta = store.meta(D1)
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


def generations(store, window_start):
    return sorted(d.name for d in store.window_dir(window_start).iterdir() if d.name.startswith("gen="))


def test_refit_switches_pointer_to_a_new_generation_and_drops_the_old(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer)
    first = generations(store, D1)

    store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer, refit=True)

    current = (store.window_dir(D1) / "current").read_text(encoding="utf-8")
    assert generations(store, D1) == [current] and current not in first
    assert store.meta(D1)["generation"] == current


def test_half_written_generation_is_ignored_until_the_pointer_moves(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer)
    # 새 세대를 쓰다가 죽은 상황 — reducer 만 있고 포인터는 옛 세대를 가리킨다
    stray = store.window_dir(D1) / "gen=crashed"
    stray.mkdir()
    (stray / "reducer.pkl").write_bytes(b"")

    points, fitted, _ = store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer)
    assert not fitted and points.tolist() == matrix_of(2)[:, :2].tolist()

    store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer, refit=True)
    assert not stray.exists()   # 다음 재학습이 미완성 세대를 치운다


def test_pointer_to_missing_generation_refits(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer)
    (store.window_dir(D1) / "current").write_text("gen=gone", encoding="utf-8")

    _, fitted, _ = store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer)
    assert fitted and store.window_starts() == [D1]


def test_pre_generation_layout_is_read_and_replaced_on_refit(tmp_path):
    store = MapStore(tmp_path / "gold")
    store.coordinates(D1, ["a", "b"], matrix_of(2), CONFIG, "m", FakeReducer)
    # 티켓 132 전 배치 — 세 파일이 창 디렉터리에 바로 있고 포인터가 없다
    window_dir = store.window_dir(D1)
    gen_dir = window_dir / generations(store, D1)[0]
    for name in ("reducer.pkl", "points.parquet", "meta.json"):
        (gen_dir / name).replace(window_dir / name)
    gen_dir.rmdir()
    (window_dir / "current").unlink()

    _, fitted, transformed = store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "m", FakeReducer)
    assert not fitted and transformed == 1

    store.coordinates(D1, ["a", "b", "c"], matrix_of(3), CONFIG, "m", FakeReducer, refit=True)
    assert not (window_dir / "points.parquet").exists() and len(generations(store, D1)) == 1
