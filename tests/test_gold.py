import pytest

from hannun.ingest import GoldStore, ingest


@pytest.fixture
def store(tmp_path, sample_path):
    gold = GoldStore(tmp_path / "gold")
    ingest([sample_path], gold)
    return gold


def test_read_range_selects_partitions_and_columns(store):
    old = store.read(end_date="2024-12-31", columns=["article_id", "publisher_id"])

    assert list(old.columns) == ["article_id", "publisher_id"]
    assert old.publisher_id.tolist() == ["hani"]


def test_read_empty_range_keeps_schema(store):
    empty = store.read(start_date="2030-01-01")

    assert len(empty) == 0
    assert "content" in empty.columns


def test_empty_author_is_stored_as_null(store):
    author = store.read().set_index("publisher_id").loc["kbs", "author"]
    assert author is None or author != ""


def test_article_id_verified_flags_hash_mismatch(store):
    df = store.read().set_index("publisher_id")

    assert bool(df.loc["chosun", "article_id_verified"]) is False
    assert bool(df.loc["yonhap", "article_id_verified"]) is True


def test_content_len_is_derived_from_content(store):
    row = store.read().set_index("publisher_id").loc["yonhap"]
    assert row.content_len == len(row.content)
