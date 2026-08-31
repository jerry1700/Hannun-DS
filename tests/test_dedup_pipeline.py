import json

import pandas as pd

from hannun.dedup import DedupStore, dedup
from hannun.ingest import GoldStore, ingest
from hannun.ingest.schema import expected_article_id
from hannun.preprocess import CleanStore, PreprocessConfig, preprocess

BASE = (
    "정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다. "
    "국토교통부는 수도권에 청년 주택 3만 호를 공급하고 전세 보증금 대출 한도를 늘리기로 했다. "
    "고용노동부는 중소기업에 취업하는 청년에게 2년간 장려금을 지급하는 방안을 내놓았다. "
    "야당은 재원 대책이 빠졌다며 국회 심의 과정에서 따져 보겠다고 밝혔다. "
    "전문가들은 공급 물량이 수요에 못 미친다고 지적했다. 시민단체는 전세 사기 대책이 빠졌다고 비판했다. "
    "청년단체는 장려금보다 정규직 전환 지원이 우선이라고 주장했다. 정부는 추가 대책을 검토하겠다고 답했다. "
    "서울시는 자체 청년 주택 사업과의 중복 여부를 확인하겠다고 밝혔다. 국회는 다음 주 상임위에서 논의를 시작한다."
)
OTHER = (
    "프로야구 한화가 기아를 상대로 9회말 역전승을 거두며 3연승을 달렸다. "
    "선발 투수가 6이닝을 2실점으로 막았고 불펜이 무실점으로 뒤를 받쳤다. "
    "타선에서는 4번 타자가 결승 2타점 2루타를 때려 승리를 이끌었다. 관중석은 만원이었다."
)

ARTICLES = [
    # (id 별칭, publisher, 발행시각, 본문) — 매체마다 다른 저작권 꼬리는 preprocess 가 지운다
    ("original", "yonhap", "2026-08-20T05:00:00Z", BASE + "\n<저작권자(c) 연합뉴스, 무단 전재-재배포 금지>"),
    ("reprint_exact", "kbs", "2026-08-20T06:00:00Z", BASE + "\n<저작권자(c) 연합뉴스, 무단 전재-재배포 금지>"),
    ("reprint_edited", "segye", "2026-08-20T07:00:00Z",
     BASE.replace("발표했다", "발표하였다") + "\n[ⓒ 세계일보 & Segye.com, 무단전재 및 재배포 금지]"),
    ("bulletin", "sbs", "2026-08-20T07:30:00Z", BASE[:320]),
    ("unrelated", "hani", "2026-08-20T08:00:00Z", OTHER),
    ("next_day", "dt", "2026-08-21T01:00:00Z", OTHER + " 다음 경기는 주말에 열린다."),
]


def build_stores(tmp_path):
    lines = []
    for alias, publisher, published, content in ARTICLES:
        url = f"https://news.example.com/{alias}"
        lines.append(json.dumps({
            "schema_version": "1.0",
            "article_id": expected_article_id(publisher, url),
            "publisher_id": publisher,
            "publisher_name": publisher,
            "source_type": "RSS",
            "url": url,
            "title": f"{alias} 제목",
            "content": content,
            "author": None,
            "category": "POLITICS",
            "category_str": None,
            "thumbnail_url": None,
            "language": "ko",
            "published_at": published,
        }, ensure_ascii=False))
    source = tmp_path / "articles.jsonl"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    root = tmp_path / "gold"
    gold, clean, store = GoldStore(root), CleanStore(root), DedupStore(root)
    ingest([source], gold)
    preprocess(gold, clean, PreprocessConfig(min_clean_len=10))
    return gold, clean, store


def alias_id(alias):
    publisher = next(p for a, p, _, _ in ARTICLES if a == alias)
    return expected_article_id(publisher, f"https://news.example.com/{alias}")


def test_full_pipeline_folds_exact_near_and_containment(tmp_path):
    gold, clean, store = build_stores(tmp_path)
    stats = dedup(gold, clean, store)
    df = store.read().set_index("article_id")

    original = alias_id("original")
    assert stats.rows == 6
    for alias, expected_method in [("reprint_exact", "sha256"), ("reprint_edited", "cosine"),
                                   ("bulletin", "containment")]:
        row = df.loc[alias_id(alias)]
        assert row.duplicate_of == original, alias
        assert row.method == expected_method, alias
        assert row.duplicate_count == 0, alias
    # 대표의 count 는 exact 로 접힌 것까지 합산해 자기 포함 4
    assert df.loc[original].duplicate_count == 4
    # parquet 의 null 은 판다스에서 NaN 으로 온다 — None 비교가 아니라 isna 로 본다
    assert pd.isna(df.loc[original].duplicate_of)


def test_singles_keep_count_one_and_no_method(tmp_path):
    gold, clean, store = build_stores(tmp_path)
    dedup(gold, clean, store)
    df = store.read().set_index("article_id")

    for alias in ("unrelated", "next_day"):
        row = df.loc[alias_id(alias)]
        assert pd.isna(row.duplicate_of) and pd.isna(row.method)
        assert row.duplicate_count == 1


def test_partitions_cover_window_and_record_it(tmp_path):
    gold, clean, store = build_stores(tmp_path)
    stats = dedup(gold, clean, store)

    assert stats.partitions == ["2026-08-20", "2026-08-21"] == store.partition_dates()
    df = store.read()
    assert set(df.window_start) == {"2026-08-20"} and set(df.window_end) == {"2026-08-21"}


def test_date_range_limits_the_window(tmp_path):
    gold, clean, store = build_stores(tmp_path)
    stats = dedup(gold, clean, store, start_date="2026-08-21")

    assert stats.rows == 1 and stats.partitions == ["2026-08-21"]
    assert store.partition_dates() == ["2026-08-21"]


def test_rerun_is_deterministic(tmp_path):
    gold, clean, store = build_stores(tmp_path)
    dedup(gold, clean, store)
    first = store.read(columns=["article_id", "duplicate_of", "duplicate_count", "method"])
    dedup(gold, clean, store)
    second = store.read(columns=["article_id", "duplicate_of", "duplicate_count", "method"])

    assert first.values.tolist() == second.values.tolist()


def test_stats_add_up(tmp_path):
    gold, clean, store = build_stores(tmp_path)
    stats = dedup(gold, clean, store)

    assert stats.exact_duplicates == 1
    assert stats.confirmed_pairs >= 2
    assert stats.duplicates == 3
    assert stats.singles == 2  # unrelated, next_day (대표 original 은 count 4 라 단독이 아니다)
    assert stats.to_dict()["rows"] == 6
