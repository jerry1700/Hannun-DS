import json

from hannun.ingest import GoldStore, ingest
from hannun.ingest.schema import expected_article_id

YONHAP_URL = "https://www.yna.co.kr/view/AKR20260820000100001"


def write_correction(tmp_path, published_at="2026-08-20T05:30:00Z"):
    """샘플 1번(연합) 기사와 같은 article_id 로 제목·본문(원하면 발행 시각까지) 바뀐 재크롤링분."""
    corrected = {
        "schema_version": "1.0",
        "article_id": expected_article_id("yonhap", YONHAP_URL),
        "publisher_id": "yonhap",
        "publisher_name": "연합뉴스",
        "source_type": "RSS",
        "url": YONHAP_URL,
        "title": "수정된 제목",
        "content": "재크롤링으로 수정된 본문",
        "author": None,
        "category": "POLITICS",
        "category_str": None,
        "thumbnail_url": None,
        "language": "ko",
        "published_at": published_at,
    }
    fix = tmp_path / "fix.jsonl"
    fix.write_text(json.dumps(corrected, ensure_ascii=False) + "\n", encoding="utf-8")
    return fix


def test_ingest_sample_stats(tmp_path, sample_path):
    store = GoldStore(tmp_path / "gold")
    stats = ingest([sample_path], store)

    assert stats.files == 1
    assert stats.records == 10
    assert stats.valid == 6                # 1,2,3,4,7,9
    assert stats.rejected == 4             # 5(빈 본문), 6(타임존 없음), 8(깨진 JSON), 10(source_type)
    assert stats.duplicates_in_batch == 1  # 3 == 1
    assert stats.written == 5
    assert stats.skipped_existing == 0
    assert stats.id_mismatch == 1          # 4
    assert stats.partitions == ["2024-03-15", "2026-08-20"]


def test_rejects_written_with_reasons(tmp_path, sample_path):
    store = GoldStore(tmp_path / "gold")
    stats = ingest([sample_path], store, run_id="test-run")

    assert stats.rejects_path is not None
    lines = [json.loads(l) for l in open(stats.rejects_path, encoding="utf-8")]
    assert len(lines) == 4
    reasons = " | ".join(l["reason"] for l in lines)
    assert "content" in reasons
    assert "published_at" in reasons
    assert "invalid JSON" in reasons
    assert "source_type" in reasons
    assert all(":" in l["source_ref"] for l in lines)  # 파일:줄번호


def test_reingest_is_idempotent(tmp_path, sample_path):
    store = GoldStore(tmp_path / "gold")
    ingest([sample_path], store)
    stats2 = ingest([sample_path], store)

    assert stats2.written == 0
    assert stats2.skipped_existing == 5
    assert len(store.read()) == 5


def test_offset_datetime_lands_in_utc_partition(tmp_path, sample_path):
    store = GoldStore(tmp_path / "gold")
    ingest([sample_path], store)
    df = store.read(start_date="2026-08-20", end_date="2026-08-20")

    assert len(df) == 4  # 1,4,7,9 — 9번은 +09:00 02:00 = UTC 08-20 17:00
    mbc = df[df.publisher_id == "mbc"].iloc[0]
    assert str(mbc.published_date) == "2026-08-20"
    assert mbc.published_at.hour == 17


def test_on_conflict_keep_skips_existing_article(tmp_path, sample_path):
    store = GoldStore(tmp_path / "gold")
    ingest([sample_path], store)

    stats = ingest([write_correction(tmp_path)], store, on_conflict="keep")
    assert stats.skipped_existing == 1
    assert store.read().set_index("publisher_id").loc["yonhap", "title"] != "수정된 제목"


def test_on_conflict_replace_updates_content(tmp_path, sample_path):
    store = GoldStore(tmp_path / "gold")
    ingest([sample_path], store)

    stats = ingest([write_correction(tmp_path)], store, on_conflict="replace")
    assert stats.replaced == 1 and stats.written == 0
    df = store.read()
    assert len(df) == 5
    assert df.set_index("publisher_id").loc["yonhap", "title"] == "수정된 제목"


def test_on_conflict_replace_moves_article_when_published_date_changes(tmp_path, sample_path):
    store = GoldStore(tmp_path / "gold")
    ingest([sample_path], store)
    before = store.partition_dates()

    # 발행 시각이 다음 날로 고쳐진 재크롤링 — 옛 날짜 파티션에서 빠지고 새 날짜에만 있어야 한다
    stats = ingest([write_correction(tmp_path, "2026-08-21T01:00:00Z")], store, on_conflict="replace")
    df = store.read()

    assert stats.replaced == 1 and stats.written == 0
    assert len(df) == 5 and df.article_id.is_unique
    assert df.set_index("publisher_id").loc["yonhap", "published_date"] == "2026-08-21"
    assert store.partition_dates() == sorted(set(before) | {"2026-08-21"})


def test_on_conflict_keep_skips_article_found_in_another_partition(tmp_path, sample_path):
    store = GoldStore(tmp_path / "gold")
    ingest([sample_path], store)

    stats = ingest([write_correction(tmp_path, "2026-08-21T01:00:00Z")], store, on_conflict="keep")

    assert stats.skipped_existing == 1 and stats.written == 0
    assert "2026-08-21" not in store.partition_dates() and len(store.read()) == 5
