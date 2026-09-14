import pytest

from hannun.dedup import DedupStore, dedup
from hannun.embedding import EmbeddingStore, embed
from hannun.ingest import GoldStore, ingest
from hannun.ingest.schema import expected_article_id
from hannun.preprocess import CleanStore, PreprocessConfig, preprocess

BASE = (
    "정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다. "
    "국토교통부는 수도권에 청년 주택 3만 호를 공급하고 전세 보증금 대출 한도를 늘리기로 했다. "
    "고용노동부는 중소기업에 취업하는 청년에게 2년간 장려금을 지급하는 방안을 내놓았다. "
    "야당은 재원 대책이 빠졌다며 국회 심의 과정에서 따져 보겠다고 밝혔다."
)
OTHER = (
    "프로야구 한화가 기아를 상대로 9회말 역전승을 거두며 3연승을 달렸다. "
    "선발 투수가 6이닝을 2실점으로 막았고 불펜이 무실점으로 뒤를 받쳤다. "
    "타선에서는 4번 타자가 결승 2타점 2루타를 때려 승리를 이끌었다."
)

ARTICLES = [
    ("original", "yonhap", "2026-08-20T05:00:00Z", BASE),
    ("reprint", "kbs", "2026-08-20T06:00:00Z", BASE),  # 완전 중복 — 접혀서 임베딩에서 빠져야 한다
    ("unrelated", "hani", "2026-08-21T08:00:00Z", OTHER),
]
LATE = ("late", "segye", "2026-08-21T09:00:00Z", OTHER + " 감독은 다음 경기 선발 로테이션을 예고했다.")


def fake_encode(texts):
    # 결정적 스텁 — 모델 다운로드 없이 파이프라인만 검증한다
    return [[float(len(t)), 1.0] for t in texts], 100.0


@pytest.fixture
def stores(tmp_path, article_jsonl):
    def build(articles=ARTICLES, run_dedup=True):
        root = tmp_path / "gold"
        gold, clean, ded, store = GoldStore(root), CleanStore(root), DedupStore(root), EmbeddingStore(root)
        ingest([article_jsonl(articles)], gold)
        preprocess(gold, clean, PreprocessConfig(min_clean_len=10))
        if run_dedup:
            dedup(gold, clean, ded)
        return gold, clean, ded, store
    return build


def alias_id(alias):
    publisher = next(p for a, p, _, _ in ARTICLES + [LATE] if a == alias)
    return expected_article_id(publisher, f"https://news.example.com/{alias}")


def test_folded_articles_are_skipped(stores):
    gold, clean, ded, store = stores()
    stats = embed(gold, clean, ded, store, encode_fn=fake_encode)
    df = store.read().set_index("article_id")

    assert stats.rows == 3 and stats.folded == 1 and stats.encoded == 2
    assert set(df.index) == {alias_id("original"), alias_id("unrelated")}


def test_encoder_input_starts_with_title(stores):
    gold, clean, ded, store = stores()
    embed(gold, clean, ded, store, encode_fn=fake_encode)
    row = store.read().set_index("article_id").loc[alias_id("original")]

    # 스텁 벡터의 첫 성분이 입력 길이 — 제목만 넣었을 때보다 길어야 본문이 붙은 것이다
    assert row.vector[0] > len("original 제목")
    assert row.dim == 2


def test_rerun_encodes_nothing_and_keeps_vectors(stores):
    gold, clean, ded, store = stores()
    embed(gold, clean, ded, store, encode_fn=fake_encode)
    first = store.read(columns=["article_id", "vector"]).set_index("article_id")

    stats = embed(gold, clean, ded, store, encode_fn=fake_encode)
    second = store.read(columns=["article_id", "vector"]).set_index("article_id")

    assert stats.encoded == 0 and stats.already == 2
    assert first.vector.map(list).to_dict() == second.vector.map(list).to_dict()


def test_new_article_is_encoded_incrementally(stores):
    gold, clean, ded, store = stores()
    embed(gold, clean, ded, store, encode_fn=fake_encode)

    stores(ARTICLES + [LATE])  # 같은 루트에 재적재 — 새 기사만 추가된다
    stats = embed(gold, clean, ded, store, encode_fn=fake_encode)

    assert stats.encoded == 1 and stats.already == 2
    assert alias_id("late") in set(store.read().article_id)


def test_missing_dedup_embeds_everything(stores):
    gold, clean, ded, store = stores(run_dedup=False)
    stats = embed(gold, clean, ded, store, encode_fn=fake_encode)

    assert stats.folded == 0 and stats.encoded == 3


def test_partitions_follow_published_date(stores):
    gold, clean, ded, store = stores()
    embed(gold, clean, ded, store, encode_fn=fake_encode)

    assert store.partition_dates() == ["2026-08-20", "2026-08-21"]
    df = store.read("2026-08-21", "2026-08-21")
    assert set(df.article_id) == {alias_id("unrelated")}
